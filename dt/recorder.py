"""
Flight data recorder and mission replay.

Three tiers, mirroring how airborne recorders actually work:

  continuous  a decimated stream (default 2 Hz) carrying only the measured
              channels, health indices and headline derived values - enough for
              trend analysis, health reporting and replay.
  summary     the complete twin snapshot once a minute, carrying the expensive
              structured sections (RUL, ranked hypotheses, advisories) that
              change on a timescale of minutes and have no business being
              written twice a second.
  event       a full-rate ring buffer held in memory; when an exceedance or an
              anomaly fires, the seconds either side of the trigger are flushed
              to disk at full rate, which is what an investigator needs.

Payloads are zlib-compressed JSON. A 24 h sortie costs roughly 60 MB rather than
the 0.9 GB that writing whole snapshots at 2 Hz would produce - the difference
between a recorder an aircraft can carry and one it cannot.

Replay reads the recorded stream back through the *same* twin pipeline, so a
post-flight analysis run produces bit-identical health indices to the live one.
"""

from __future__ import annotations

import collections
import json
import os
import tempfile
import sqlite3
import threading
import time
import zlib
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

# Sections carried by the 2 Hz continuous stream. Everything omitted here is
# preserved at 1/60 Hz on the summary tier and at full rate around any event, so
# nothing is lost - it is simply not written 172 800 times per sortie.
CONTINUOUS_SECTIONS = ("t_s", "phase", "running", "regime", "engine_hours",
                       "analytics_valid", "steady_state", "env", "cmd", "meas",
                       "peak", "health")
CONTINUOUS_DERIVED = ("power_kw", "bsfc_g_kwh", "ve_index", "power_index",
                      "ff_index", "cooling_effectiveness", "oil_press_index",
                      "comp_eff_index", "vib_halforder_g", "vib_1x_g",
                      "knock_index", "boost_error_kpa")
CONTINUOUS_ANOMALY = ("score", "status", "t2_norm", "spe_norm", "recon_norm",
                      "drift_score")


def _compact(snap: dict) -> dict:
    """Project a twin snapshot onto the fields the continuous tier carries."""
    out = {k: snap[k] for k in CONTINUOUS_SECTIONS if k in snap}
    d = snap.get("derived", {})
    out["derived"] = {k: d[k] for k in CONTINUOUS_DERIVED if k in d}
    a = snap.get("anomaly", {})
    out["anomaly"] = {k: a[k] for k in CONTINUOUS_ANOMALY if k in a}
    out["mission"] = (snap.get("mission", {}) or {}).get("decision")
    out["n_exceedances"] = len(snap.get("exceedances", []))
    top = (snap.get("diagnostics", {}) or {}).get("hypotheses") or []
    if top:
        out["top_fault"] = {"fault": top[0]["fault"], "confidence": top[0]["confidence"],
                            "cylinder": top[0].get("cylinder"),
                            "channel": top[0].get("channel")}
    return out


def _pack(obj: dict) -> bytes:
    return zlib.compress(json.dumps(obj, separators=(",", ":")).encode(), 6)


def _unpack(blob) -> dict:
    if isinstance(blob, str):            # rows written before compression
        return json.loads(blob)
    try:
        return json.loads(zlib.decompress(blob))
    except zlib.error:
        return json.loads(blob.decode())


SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL,
    started_utc  REAL NOT NULL,
    ended_utc    REAL,
    profile      TEXT,
    scenario     TEXT,
    tail_number  TEXT,
    config_json  TEXT
);
CREATE TABLE IF NOT EXISTS telemetry (
    session_id   INTEGER NOT NULL,
    t_s          REAL NOT NULL,
    rate         TEXT NOT NULL DEFAULT 'continuous',
    payload      BLOB NOT NULL,
    PRIMARY KEY (session_id, t_s, rate)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   INTEGER NOT NULL,
    t_s          REAL NOT NULL,
    kind         TEXT NOT NULL,
    severity     TEXT NOT NULL,
    summary      TEXT,
    payload      TEXT
);
CREATE INDEX IF NOT EXISTS ix_tel_session ON telemetry(session_id, t_s);
CREATE INDEX IF NOT EXISTS ix_evt_session ON events(session_id, t_s);
"""


class FlightRecorder:
    def __init__(self, path: str = "data/fdr.sqlite", continuous_hz: float = 2.0,
                 ring_seconds: float = 60.0, ring_hz: float = 20.0,
                 summary_period_s: float = 60.0):
        d = os.path.dirname(path)
        if d:
            try:
                os.makedirs(d, exist_ok=True)
                # sqlite needs to create journal files next to the database
                if not os.access(d, os.W_OK):
                    raise OSError("data directory not writable")
            except OSError:
                # read-only / ephemeral hosts: fall back to a temp directory
                d = os.path.join(tempfile.gettempdir(), "aerodt")
                os.makedirs(d, exist_ok=True)
                path = os.path.join(d, os.path.basename(path))
        self.path = path
        # The recorder is created on the thread that builds the twin but written
        # from the thread that runs it, so the connection must be shareable and
        # every statement serialised behind a lock.
        self.conn = sqlite3.connect(path, check_same_thread=False)
        try:
            self.conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.DatabaseError:
            # WAL needs shared memory / locking that some network or container
            # filesystems lack; the default rollback journal always works.
            self.conn.execute("PRAGMA journal_mode=DELETE")
        self.lock = threading.Lock()
        with self.lock:
            self.conn.executescript(SCHEMA)
            self.conn.commit()
        self.continuous_period = 1.0 / max(continuous_hz, 1e-6)
        self.summary_period = summary_period_s
        self._next_write = 0.0
        self._next_summary = 0.0
        self.ring: collections.deque = collections.deque(maxlen=int(ring_seconds * ring_hz))
        self.session_id: Optional[int] = None
        self._pending: List[tuple] = []

    # -- session lifecycle ---------------------------------------------------
    def start_session(self, name: str, profile: str = "", scenario: str = "",
                      tail_number: str = "UAV-001", config: Optional[dict] = None) -> int:
        with self.lock:
            cur = self.conn.execute(
                "INSERT INTO sessions(name, started_utc, profile, scenario, tail_number,"
                " config_json) VALUES (?,?,?,?,?,?)",
                (name, time.time(), profile, scenario, tail_number,
                 json.dumps(config or {})))
            self.conn.commit()
        self.session_id = cur.lastrowid
        self._next_write = 0.0
        self._next_summary = 0.0
        return self.session_id

    def end_session(self) -> None:
        self.flush()
        if self.session_id is not None:
            with self.lock:
                self.conn.execute("UPDATE sessions SET ended_utc=? WHERE id=?",
                                  (time.time(), self.session_id))
                self.conn.commit()

    # -- writing --------------------------------------------------------------
    def record(self, t_s: float, snapshot: Dict[str, Any]) -> None:
        self.ring.append((t_s, snapshot))
        if self.session_id is None:
            return
        if t_s + 1e-9 >= self._next_summary:
            self._next_summary = t_s + self.summary_period
            self._pending.append((self.session_id, round(t_s, 3), "summary",
                                  _pack(snapshot)))
        if t_s + 1e-9 < self._next_write:
            return
        self._next_write = t_s + self.continuous_period
        self._pending.append((self.session_id, round(t_s, 3), "continuous",
                              _pack(_compact(snapshot))))
        if len(self._pending) >= 200:
            self.flush()

    def flush(self) -> None:
        if self._pending:
            with self.lock:
                self.conn.executemany(
                    "INSERT OR REPLACE INTO telemetry(session_id,t_s,rate,payload)"
                    " VALUES (?,?,?,?)", self._pending)
                self._pending.clear()
                self.conn.commit()

    def log_event(self, t_s: float, kind: str, severity: str, summary: str,
                  payload: Optional[dict] = None, dump_ring: bool = False) -> None:
        if self.session_id is None:
            return
        with self.lock:
            self.conn.execute(
                "INSERT INTO events(session_id,t_s,kind,severity,summary,payload)"
                " VALUES (?,?,?,?,?,?)",
                (self.session_id, round(t_s, 3), kind, severity, summary,
                 json.dumps(payload or {}, separators=(",", ":"))))
            if dump_ring and self.ring:
                rows = [(self.session_id, round(t, 3), "event", _pack(s))
                        for t, s in self.ring]
                self.conn.executemany(
                    "INSERT OR REPLACE INTO telemetry(session_id,t_s,rate,payload)"
                    " VALUES (?,?,?,?)", rows)
            self.conn.commit()

    # -- reading / replay -------------------------------------------------------
    def sessions(self) -> List[dict]:
        cur = self.conn.execute(
            "SELECT s.id,s.name,s.started_utc,s.ended_utc,s.profile,s.scenario,s.tail_number,"
            " (SELECT COUNT(*) FROM telemetry t WHERE t.session_id=s.id),"
            " (SELECT COUNT(*) FROM events e WHERE e.session_id=s.id),"
            " (SELECT MAX(t_s) FROM telemetry t WHERE t.session_id=s.id)"
            " FROM sessions s ORDER BY s.id DESC")
        keys = ("id", "name", "started_utc", "ended_utc", "profile", "scenario",
                "tail_number", "samples", "events", "duration_s")
        return [dict(zip(keys, r)) for r in cur.fetchall()]

    def read(self, session_id: int, t_from: float = 0.0, t_to: float = 1e12,
             rate: str = "continuous", limit: int = 200000) -> List[dict]:
        cur = self.conn.execute(
            "SELECT t_s,payload FROM telemetry WHERE session_id=? AND rate=?"
            " AND t_s BETWEEN ? AND ? ORDER BY t_s LIMIT ?",
            (session_id, rate, t_from, t_to, limit))
        out = []
        for t, p in cur.fetchall():
            d = _unpack(p)
            d["t_s"] = t
            out.append(d)
        return out

    def events(self, session_id: int) -> List[dict]:
        cur = self.conn.execute(
            "SELECT t_s,kind,severity,summary,payload FROM events"
            " WHERE session_id=? ORDER BY t_s", (session_id,))
        return [{"t_s": t, "kind": k, "severity": s, "summary": m,
                 "payload": json.loads(p or "{}")} for t, k, s, m, p in cur.fetchall()]

    def replay(self, session_id: int, speed: float = 0.0,
               rate: str = "continuous") -> Iterator[dict]:
        """Yield recorded samples. `speed` 0 = as fast as possible, 1 = real time."""
        rows = self.read(session_id, rate=rate)
        prev = None
        for r in rows:
            if speed > 0 and prev is not None:
                dt = (r["t_s"] - prev) / speed
                if dt > 0:
                    time.sleep(min(dt, 1.0))
            prev = r["t_s"]
            yield r

    def storage(self, session_id: int) -> dict:
        """Bytes on disk per tier, and the implied cost of a 24 h sortie."""
        cur = self.conn.execute(
            "SELECT rate, COUNT(*), SUM(LENGTH(payload)) FROM telemetry"
            " WHERE session_id=? GROUP BY rate", (session_id,))
        tiers = {r: {"rows": n, "bytes": b or 0} for r, n, b in cur.fetchall()}
        dur = self.conn.execute("SELECT MAX(t_s) FROM telemetry WHERE session_id=?",
                                (session_id,)).fetchone()[0] or 1.0
        total = sum(t["bytes"] for t in tiers.values())
        # Only the continuous and summary tiers scale with sortie length. The
        # event tier scales with the number of triggers, so folding it into a
        # per-hour rate would make a quiet 24 h flight look like a noisy one.
        timed = sum(tiers.get(k, {}).get("bytes", 0) for k in ("continuous", "summary"))
        ev = tiers.get("event", {})
        n_ev = self.conn.execute("SELECT COUNT(*) FROM events WHERE session_id=?",
                                 (session_id,)).fetchone()[0] or 0
        return {"tiers": tiers, "bytes": total, "duration_s": dur, "events": n_ev,
                "mb_per_hour": timed * 3600.0 / max(dur, 1e-6) / 1e6,
                "mb_per_24h": timed * 86400.0 / max(dur, 1e-6) / 1e6,
                "mb_per_event": (ev.get("bytes", 0) / n_ev / 1e6) if n_ev else 0.0}

    def summary(self, session_id: int) -> dict:
        rows = self.read(session_id)
        if not rows:
            return {"session_id": session_id, "samples": 0}
        def series(path: str) -> List[float]:
            out = []
            for r in rows:
                v: Any = r
                for k in path.split("."):
                    v = v.get(k) if isinstance(v, dict) else None
                    if v is None:
                        break
                if isinstance(v, (int, float)):
                    out.append(float(v))
            return out
        agg = {}
        for label, path in (("cht_max", "peak.cht_max_c"), ("egt_max", "peak.egt_max_c"),
                            ("oil_temp", "meas.oil_temp_c"), ("oil_press", "meas.oil_press_kpa"),
                            ("rpm", "meas.rpm"), ("fuel_flow_lph", "meas.fuel_flow_lph"),
                            ("health", "health.overall"), ("anomaly", "anomaly.score")):
            s = series(path)
            if s:
                agg[label] = {"min": round(min(s), 2), "max": round(max(s), 2),
                              "mean": round(sum(s) / len(s), 2)}
        ev = self.events(session_id)
        return {"session_id": session_id, "samples": len(rows),
                "duration_s": round(rows[-1]["t_s"], 1), "aggregates": agg,
                "event_counts": _counts(e["severity"] for e in ev),
                "events": ev[:200]}

    def close(self) -> None:
        self.flush()
        self.conn.close()


def _counts(it) -> Dict[str, int]:
    d: Dict[str, int] = {}
    for x in it:
        d[x] = d.get(x, 0) + 1
    return d
