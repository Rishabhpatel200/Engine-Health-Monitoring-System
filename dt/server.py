"""
Ground Control Station service.

A dependency-free HTTP/SSE server exposing the twin to the operator dashboard,
to maintenance tooling and to fleet-level infrastructure:

    GET  /                        dashboard
    GET  /api/state               latest twin snapshot
    GET  /api/stream              server-sent event stream of snapshots
    GET  /api/config              engine spec, limits and CAN ICD
    GET  /api/profiles            available mission profiles
    GET  /api/scenarios           available fault scenarios
    POST /api/control             run / pause / reset / speed / inject fault
    GET  /api/sessions            recorded sorties
    GET  /api/session/<id>        replay data and mission report
    GET  /api/report/<id>         mission-wise health report

The simulator runs on its own thread so the HTTP layer never blocks it. In a
real installation the same service is fed by the CAN receive thread instead;
`TwinService.feed()` is the seam.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from .anomaly import AnomalyEngine
from .canbus import PGN_MAP, bus_load_percent
from .config import TwinConfig
from .faults import FAULT_CATALOGUE, FaultEvent, FaultInjector
from .missions import PROFILES
from .paths import resolve
from .recorder import FlightRecorder
from .simulator import MissionSimulator, SimConfig

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


class TwinService:
    """Owns the simulator thread and fans snapshots out to SSE subscribers."""

    def __init__(self, sim_cfg: SimConfig, model_path: Optional[str] = None):
        self.sim_cfg = sim_cfg
        self.model_path = model_path
        self.lock = threading.Lock()
        self.subscribers: List[queue.Queue] = []
        self.latest: Dict[str, Any] = {}
        self.running = False
        self.speed = 20.0            # simulated seconds per wall-clock second
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.sim: Optional[MissionSimulator] = None
        self.error: Optional[str] = None
        self._build()

    def _load_anomaly(self) -> Optional[AnomalyEngine]:
        if self.model_path and os.path.exists(self.model_path):
            try:
                return AnomalyEngine.load(self.model_path)
            except Exception as e:
                print(f"  WARNING: could not load {self.model_path}: {e} - running untrained")
                return None
        print(f"  WARNING: model file not found ({self.model_path}) - running untrained")
        return None

    def _build(self) -> None:
        self.sim = MissionSimulator(self.sim_cfg, anomaly=self._load_anomaly())
        self.latest = {}
        self.error = None
        # The service drives `sim.step()` directly rather than through
        # `sim.run()`, so the recording session has to be opened here - without
        # it the recorder silently discards every sample and the replay tab
        # stays empty.
        if self.sim.recorder is not None:
            self.sim.recorder.start_session(
                f"{self.sim_cfg.profile}/{self.sim_cfg.scenario}",
                self.sim_cfg.profile, self.sim_cfg.scenario,
                config=self.sim.cfg.to_dict())

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def pause(self) -> None:
        self.running = False
        self._stop.set()

    def reset(self, **overrides) -> None:
        self.pause()
        if self._thread:
            self._thread.join(timeout=2.0)
        for k, v in overrides.items():
            if hasattr(self.sim_cfg, k):
                setattr(self.sim_cfg, k, v)
        if self.sim:
            if self.sim.recorder is not None:
                self.sim.recorder.end_session()
            self.sim.close()
        self._build()

    def _loop(self) -> None:
        sim = self.sim
        assert sim is not None
        dt = sim.sim.dt
        last_emit = 0.0
        wall = time.time()
        try:
            while not self._stop.is_set() and sim.t < sim.profile.duration_s:
                target_sim_dt = (time.time() - wall) * self.speed
                wall = time.time()
                n = max(1, min(400, int(target_sim_dt / dt)))
                snap = None
                for _ in range(n):
                    snap = sim.step()
                if snap is not None and (snap["t_s"] - last_emit) >= 0.4:
                    last_emit = snap["t_s"]
                    with self.lock:
                        self.latest = snap
                    self._publish(snap)
                time.sleep(0.05)
        except Exception:
            self.error = traceback.format_exc()
            print("  simulator thread failed:\n" + self.error)
        finally:
            self.running = False
            if sim.recorder is not None:
                sim.recorder.flush()
                if sim.t >= sim.profile.duration_s:
                    sim.recorder.end_session()

    def _publish(self, snap: dict) -> None:
        dead = []
        for q in self.subscribers:
            try:
                q.put_nowait(snap)
            except queue.Full:
                dead.append(q)
        for q in dead:
            if q in self.subscribers:
                self.subscribers.remove(q)

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=64)
        self.subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        if q in self.subscribers:
            self.subscribers.remove(q)

    # -- control ------------------------------------------------------------
    def inject(self, spec: dict) -> dict:
        if self.sim is None:
            return {"ok": False, "error": "no simulator"}
        kind = spec.get("kind")
        if kind not in FAULT_CATALOGUE:
            return {"ok": False, "error": f"unknown fault '{kind}'"}
        ev = FaultEvent(
            kind=kind,
            onset_s=float(spec.get("onset_s", self.sim.t + 1.0)),
            severity=float(spec.get("severity", 0.4)),
            profile=str(spec.get("profile", "ramp")),
            duration_s=float(spec.get("duration_s", 180.0)),
            cylinder=(int(spec["cylinder"]) if spec.get("cylinder") else None),
            channel=(str(spec["channel"]) if spec.get("channel") else None),
            note=str(spec.get("note", "operator-injected")))
        self.sim.injector.add(ev)
        return {"ok": True, "event": ev.label(), "onset_s": ev.onset_s}

    def status(self) -> dict:
        s = self.sim
        return {"running": self.running, "speed": self.speed,
                "t_s": round(s.t, 1) if s else 0.0,
                "duration_s": round(s.profile.duration_s, 1) if s else 0.0,
                "profile": self.sim_cfg.profile, "scenario": self.sim_cfg.scenario,
                "initial_hours": self.sim_cfg.initial_hours,
                "recording": self.sim_cfg.record,
                "can": (s.bus.stats() if s and s.bus else {"transport": "direct"}),
                "can_load_pct": round(bus_load_percent(), 2),
                "model_trained": bool(s and s.twin.anomaly.trained),
                "active_faults": list(s.injector.active) if s else [],
                "error": self.error}


class Handler(BaseHTTPRequestHandler):
    service: TwinService = None            # type: ignore
    fdr_path: str = "data/fdr.sqlite"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):     # keep the console readable
        pass

    # -- helpers -------------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj: Any, code: int = 200) -> None:
        self._send(code, json.dumps(obj, default=str).encode(), "application/json")

    def _file(self, name: str, ctype: str) -> None:
        path = os.path.join(WEB_DIR, name)
        if not os.path.exists(path):
            self._send(404, b"not found", "text/plain")
            return
        with open(path, "rb") as f:
            self._send(200, f.read(), ctype)

    def _recorder(self) -> FlightRecorder:
        svc = Handler.service
        live = getattr(getattr(svc, "sim", None), "recorder", None)
        return FlightRecorder(live.path if live is not None else self.fdr_path)

    # -- routes ---------------------------------------------------------------
    def do_GET(self) -> None:
        u = urlparse(self.path)
        p, q = u.path, parse_qs(u.query)
        svc = Handler.service
        try:
            if p in ("/", "/index.html"):
                return self._file("index.html", "text/html; charset=utf-8")
            if p in ("/healthz", "/health", "/api/health"):
                return self._json({"ok": True, "running": svc.running,
                                   "model_trained": bool(svc.sim and svc.sim.twin.anomaly.trained)})
            if p == "/api/state":
                with svc.lock:
                    return self._json({"snapshot": svc.latest, "status": svc.status()})
            if p == "/api/status":
                return self._json(svc.status())
            if p == "/api/config":
                cfg = svc.sim.cfg if svc.sim else TwinConfig()
                return self._json({
                    "engine": cfg.to_dict()["engine"], "limits": cfg.to_dict()["limits"],
                    "sensors": sorted(cfg.sensors),
                    "can": {"bitrate": cfg.can_bitrate, "load_pct": round(bus_load_percent(), 2),
                            "pgns": [{"pgn": f"0x{k:04X}", "name": v.name, "rate_hz": v.rate_hz,
                                      "signals": [s.name for s in v.signals]}
                                     for k, v in PGN_MAP.items()]}})
            if p == "/api/profiles":
                return self._json(sorted(PROFILES))
            if p == "/api/scenarios":
                return self._json({"scenarios": FaultInjector.scenario_names(),
                                   "faults": {k: v for k, v in FAULT_CATALOGUE.items()}})
            if p == "/api/sessions":
                r = self._recorder()
                try:
                    return self._json(r.sessions())
                finally:
                    r.close()
            if p.startswith("/api/session/"):
                sid = int(p.rsplit("/", 1)[-1])
                r = self._recorder()
                try:
                    stride = int(q.get("stride", ["1"])[0])
                    rows = r.read(sid)
                    return self._json({"summary": r.summary(sid),
                                       "events": r.events(sid),
                                       "samples": rows[::max(1, stride)]})
                finally:
                    r.close()
            if p.startswith("/api/report/"):
                sid = int(p.rsplit("/", 1)[-1])
                r = self._recorder()
                try:
                    return self._json(_mission_report(r, sid))
                finally:
                    r.close()
            if p == "/api/stream":
                return self._stream()
            return self._send(404, b'{"error":"not found"}')
        except Exception as e:
            return self._json({"error": str(e), "trace": traceback.format_exc()}, 500)

    def do_POST(self) -> None:
        u = urlparse(self.path)
        svc = Handler.service
        n = int(self.headers.get("Content-Length", "0") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._json({"error": "bad json"}, 400)
        try:
            if u.path == "/api/control":
                act = body.get("action")
                if act == "run":
                    svc.start()
                elif act == "pause":
                    svc.pause()
                elif act == "speed":
                    svc.speed = max(0.1, min(2000.0, float(body.get("value", 20))))
                elif act == "reset":
                    svc.reset(**{k: v for k, v in body.items()
                                 if k in ("profile", "scenario", "initial_hours",
                                          "record", "seed", "degradation_accel")})
                elif act == "inject":
                    return self._json(svc.inject(body.get("fault", {})))
                elif act == "service":
                    if svc.sim:
                        msg = svc.sim.degradation.service(svc.sim.base_health,
                                                          body.get("item", "oil_change"))
                        return self._json({"ok": True, "message": msg})
                else:
                    return self._json({"error": f"unknown action '{act}'"}, 400)
                return self._json({"ok": True, "status": svc.status()})
            return self._send(404, b'{"error":"not found"}')
        except Exception as e:
            return self._json({"error": str(e), "trace": traceback.format_exc()}, 500)

    def _stream(self) -> None:
        svc = Handler.service
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")     # nginx / most PaaS proxies
        self.end_headers()
        q = svc.subscribe()
        try:
            with svc.lock:
                if svc.latest:
                    self.wfile.write(f"data: {json.dumps(svc.latest, default=str)}\n\n".encode())
                    self.wfile.flush()
            while True:
                try:
                    snap = q.get(timeout=10.0)
                    payload = json.dumps(snap, default=str)
                    self.wfile.write(f"data: {payload}\n\n".encode())
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            svc.unsubscribe(q)


def _mission_report(rec: FlightRecorder, sid: int) -> dict:
    """Mission-wise health report: what the sortie cost the engine.

    Aggregates come from the 2 Hz continuous tier; the structured sections - RUL,
    ranked hypotheses, advisories - come from the once-a-minute summary tier,
    which is where they are recorded.
    """
    rows = rec.read(sid)
    if not rows:
        return {"session_id": sid, "error": "no data"}
    first, last = rows[0], rows[-1]
    summaries = rec.read(sid, rate="summary")
    last_full = summaries[-1] if summaries else {}
    phases: Dict[str, float] = {}
    prev_t = rows[0]["t_s"]
    for r in rows:
        ph = r.get("phase", "unknown")
        phases[ph] = phases.get(ph, 0.0) + (r["t_s"] - prev_t)
        prev_t = r["t_s"]
    def idx(r, k):
        return r.get("health", {}).get("subsystems", {}).get(k)
    subs = list(last.get("health", {}).get("subsystems", {}))
    delta = {k: round((idx(last, k) or 0) - (idx(first, k) or 0), 2) for k in subs}
    fuel = 0.0
    prev_t = rows[0]["t_s"]
    for r in rows:
        ff = (r.get("meas", {}) or {}).get("fuel_flow_lph") or 0.0
        fuel += ff * (r["t_s"] - prev_t) / 3600.0
        prev_t = r["t_s"]
    s = rec.summary(sid)
    return {
        "session_id": sid,
        "duration_h": round(last["t_s"] / 3600.0, 2),
        "engine_hours_start": first.get("engine_hours"),
        "engine_hours_end": last.get("engine_hours"),
        "fuel_burn_l": round(fuel, 1),
        "phase_seconds": {k: round(v, 1) for k, v in phases.items()},
        "health_start": first.get("health", {}),
        "health_end": last.get("health", {}),
        "health_delta": delta,
        "peaks": s.get("aggregates", {}),
        "events": s.get("events", []),
        "event_counts": s.get("event_counts", {}),
        "rul_end": (last_full.get("rul", {}) or {}).get("engine", {}),
        "advisories_end": last_full.get("advisories", []),
        "diagnostics_end": last_full.get("diagnostics", {}),
        "storage": rec.storage(sid),
    }


def serve(host: str = "127.0.0.1", port: int = 8080,
          sim_cfg: Optional[SimConfig] = None,
          model_path: Optional[str] = "data/models.json",
          autostart: bool = True) -> None:
    cfg = sim_cfg or SimConfig(record=True)
    cfg.fdr_path = resolve(cfg.fdr_path)
    model_path = resolve(model_path) if model_path else model_path
    svc = TwinService(cfg, model_path)
    Handler.service = svc
    Handler.fdr_path = cfg.fdr_path
    ThreadingHTTPServer.daemon_threads = True
    ThreadingHTTPServer.allow_reuse_address = True
    httpd = ThreadingHTTPServer((host, port), Handler)
    if autostart:
        svc.start()
    print(f"  Digital Twin GCS on http://{host}:{port}")
    print(f"  profile={cfg.profile}  scenario={cfg.scenario}  "
          f"engine_hours={cfg.initial_hours}  model={'loaded' if svc.sim.twin.anomaly.trained else 'untrained'}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  shutting down")
    finally:
        svc.pause()
        if svc.sim:
            svc.sim.close()
        httpd.server_close()
