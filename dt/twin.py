"""
Digital Twin core framework.

Pipeline, executed once per synchronisation tick:

    CAN frames -> signals -> reference model -> UKF sync -> residual features
              -> anomaly engine -> condition indicators -> health indices
              -> fault isolation -> RUL -> advisories -> snapshot -> recorder

The *reference model* is a pristine copy of the physics model clamped to the
measured operating point. Comparing the real engine against a new engine at the
same rpm, MAP, altitude and temperature is what removes the operating point from
the problem: every index the health layer consumes is a ratio against that
reference, and is therefore comparable across the whole flight envelope.

The twin is transport-agnostic and side-effect free apart from the recorder, so
the identical object serves live GCS telemetry, a test-rig harness, and offline
mission replay.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .advisory import AdvisoryEngine
from .anomaly import AnomalyEngine, regime_of
from .atmosphere import AtmState, isa
from .canbus import BaseCanBus, VirtualCanBus
from .config import Limits, SUBSYSTEMS, TwinConfig
from .degradation import DegradationModel
from .diagnostics import FaultIsolator
from .engine_model import Command, HealthParams, Outputs, PistonEngine
from .estimator import UnscentedKalmanFilter
from .health import HealthMonitor
from .recorder import FlightRecorder
from .rul import RULEstimator
from .sensors import SensorSuite


# --------------------------------------------------------------------------
FEATURES: List[str] = [
    "r_rpm", "r_map", "r_iat", "r_oil_press", "r_oil_temp", "r_coolant",
    "r_fuel_flow", "r_vib", "r_bus_v",
    "r_cht1", "r_cht2", "r_cht3", "r_cht4",
    "egt_dev1", "egt_dev2", "egt_dev3", "egt_dev4",
    "cht_spread", "ve_index", "ff_index", "power_index",
    "cooling_eff", "oil_press_index", "comp_eff_index",
    "vib_half", "vib_1x", "boost_err", "knock", "cool_reserve",
]


class ReferenceModel:
    """A new-engine twin held at the measured operating point.

    Rather than integrating freely (which would drift away from the real engine
    and destroy the comparison), the reference is clamped: measured rpm and MAP
    are written into its state each tick, while its thermal states are allowed to
    evolve under its own healthy heat balance. The difference between its
    temperatures and the measured ones is therefore attributable to degradation,
    not to a different operating point.
    """

    def __init__(self, cfg: TwinConfig):
        self.engine = PistonEngine(cfg.engine, HealthParams(n_cyl=cfg.engine.n_cyl))
        self.engine.running = True
        self._primed = False

    def prime(self, atm: AtmState, meas: Dict[str, Optional[float]]) -> None:
        st = self.engine.state
        n = len(st.cht_c)
        st.coolant_c = meas.get("coolant_temp_c") or atm.temp_c
        st.oil_c = meas.get("oil_temp_c") or atm.temp_c
        for i in range(n):
            st.cht_c[i] = meas.get(f"cht{i+1}_c") or atm.temp_c
            st.egt_c[i] = meas.get(f"egt{i+1}_c") or atm.temp_c
        self._primed = True

    def step(self, dt: float, cmd: Command, atm: AtmState, tas_ms: float,
             meas: Dict[str, Optional[float]]) -> Outputs:
        if not self._primed:
            self.prime(atm, meas)
        st = self.engine.state
        if meas.get("rpm") is not None:
            st.rpm = max(0.0, meas["rpm"])
        if meas.get("map_kpa") is not None:
            st.map_kpa = meas["map_kpa"]
        self.engine.running = st.rpm > 200.0
        out = self.engine.step(dt, cmd, atm, tas_ms)
        # Re-clamp: the model's own crank/manifold dynamics must not fight the
        # measurement, they are only there to produce consistent derived terms.
        if meas.get("rpm") is not None:
            st.rpm = max(0.0, meas["rpm"])
            out.rpm = st.rpm
        if meas.get("map_kpa") is not None:
            st.map_kpa = meas["map_kpa"]
            out.map_kpa = st.map_kpa
        return out


class SteadyStateDetector:
    """Flags a stabilised operating point.

    Engine health trending is only valid between comparable operating points.
    Fleet health-monitoring practice is to take one stabilised cruise snapshot
    per flight rather than to trend every sample, precisely because a throttle
    movement otherwise looks like a step change in condition. The same gate also
    protects fault isolation: a large innovation during a throttle slam is model
    transient error, not a lying sensor.
    """

    # Bands chosen so that a level cruise or loiter qualifies and a climb does
    # not: 40 m over the 30 s window is a 1.3 m/s vertical rate, essentially
    # level, whereas a MALE climb runs 2-8 m/s.
    BANDS = (("rpm", 70.0), ("map_kpa", 4.0), ("throttle_pct", 3.0),
             ("altitude_m", 40.0), ("oat_c", 1.0))

    def __init__(self, window_s: float = 30.0, sample_hz: float = 2.0):
        import collections
        self.n = max(4, int(window_s * sample_hz))
        self.period = 1.0 / sample_hz
        self.buf = collections.deque(maxlen=self.n)
        self._next = 0.0
        self.steady = False
        self.reason = "initialising"

    def update(self, t_s: float, rpm: float, map_kpa: float, thr: float,
               altitude_m: float = 0.0, oat_c: float = 15.0) -> bool:
        if t_s + 1e-9 >= self._next:
            self._next = t_s + self.period
            self.buf.append((rpm, map_kpa, thr, altitude_m, oat_c))
        if len(self.buf) < self.n:
            self.steady = False
            self.reason = "window not full"
            return False
        self.reason = ""
        for i, (name, band) in enumerate(self.BANDS):
            r = max(b[i] for b in self.buf) - min(b[i] for b in self.buf)
            if r > band:
                # A constant-power climb holds rpm, MAP and throttle perfectly
                # steady while the atmosphere changes underneath it. Judging
                # steadiness from the engine alone lets the whole climb count as
                # a stabilised point, and every altitude-driven change of the
                # indices is then read as degradation.
                self.reason = f"{name} varying by {r:.1f}"
                self.steady = False
                return False
        self.steady = True
        return True


@dataclass
class Exceedance:
    channel: str
    subsystem: str
    unit: str
    value: float
    limit: float
    severity: str            # "caution" | "limit"
    since_s: float
    duration_s: float = 0.0
    flight_critical: bool = False

    def to_dict(self) -> dict:
        return {"channel": self.channel, "subsystem": self.subsystem, "unit": self.unit,
                "value": round(self.value, 1), "limit": round(self.limit, 1),
                "severity": self.severity, "duration_s": round(self.duration_s, 1),
                "flight_critical": self.flight_critical}


class ExceedanceMonitor:
    """The conventional threshold layer, retained as an independent safety net.

    A dwell time is required before an exceedance latches, so sensor spikes and
    single dropped frames do not raise nuisance alerts on the operator display.
    """

    RULES = [
        # channel, subsystem, unit, caution, limit, dwell_s, flight_critical
        ("cht_max_c",      "cooling",     "degC", "cht_caution_c",  "cht_c",        3.0, True),
        ("egt_max_c",      "combustion",  "degC", "egt_caution_c",  "egt_c",        3.0, True),
        ("oil_temp_c",     "lubrication", "degC", "oil_temp_caution_c", "oil_temp_c", 5.0, True),
        ("coolant_temp_c", "cooling",     "degC", None,             "coolant_temp_c", 5.0, True),
        ("rpm",            "mechanical",  "rpm",  None,             "rpm_max",      1.0, True),
        ("map_kpa",        "air_path",    "kPa",  None,             "map_kpa_max",  2.0, False),
        ("vib_rms_g",      "mechanical",  "g",    "vib_caution_g",  "vib_rms_g",    2.0, False),
    ]

    def __init__(self, limits: Limits):
        self.limits = limits
        self.active: Dict[str, Exceedance] = {}
        self.history: List[Exceedance] = []

    def update(self, values: Dict[str, float], t_s: float) -> List[Exceedance]:
        L = self.limits
        pending: Dict[str, Exceedance] = {}
        for ch, sub, unit, caution_attr, limit_attr, dwell, crit in self.RULES:
            v = values.get(ch)
            if v is None:
                continue
            lim = getattr(L, limit_attr)
            caution = getattr(L, caution_attr) if caution_attr else None
            sev = None
            if v >= lim:
                sev = "limit"
                ref = lim
            elif caution is not None and v >= caution:
                sev = "caution"
                ref = caution
            if sev is None:
                continue
            prev = self.active.get(ch)
            since = prev.since_s if prev and prev.severity == sev else t_s
            e = Exceedance(ch, sub, unit, v, ref, sev, since, t_s - since, crit)
            # Dwell applies to limits as well as cautions: real overspeed and
            # overtemperature detection requires the condition to persist, or
            # every governor transient becomes an abort.
            if e.duration_s >= dwell:
                pending[ch] = e

        for ch, e in pending.items():
            if ch not in self.active:
                self.history.append(e)
        keep = {k: v for k, v in self.active.items()
                if k not in [r[0] for r in self.RULES] and k not in pending}
        self.active = {**pending}
        self._carry = keep
        # Low-side rules go through the same dwell logic, with a small
        # hysteresis so a channel sitting exactly on its limit does not chatter.
        rpm = values.get("rpm", 0.0)
        for key, ch, sub, unit, limit, sev, min_rpm, hyst, crit in (
                ("oil_press_kpa", "oil_press_kpa", "lubrication", "kPa",
                 L.oil_press_kpa_min, "limit", 2500.0, 5.0, True),
                ("oil_temp_low", "oil_temp_c", "lubrication", "degC",
                 L.oil_temp_min_c, "caution", 3000.0, 2.0, False),
                ("bus_v", "bus_v", "electrical", "V",
                 L.bus_voltage_min, "caution", 3000.0, 0.15, False)):
            v = values.get(ch)
            if v is None or rpm <= min_rpm:
                continue
            was = key in self.active
            trip = limit if not was else limit + hyst      # hysteresis on release
            if v >= trip:
                continue
            prev = self.active.get(key)
            since = prev.since_s if prev else t_s
            e = Exceedance(ch, sub, unit, v, limit, sev, since, t_s - since, crit)
            if e.duration_s >= (0.0 if sev == "limit" else 5.0):
                if not was:
                    self.history.append(e)
                self.active[key] = e
        return list(self.active.values())


# --------------------------------------------------------------------------
class DigitalTwin:
    """The synchronised virtual engine and its analytics stack."""

    def __init__(self, cfg: Optional[TwinConfig] = None,
                 anomaly: Optional[AnomalyEngine] = None,
                 recorder: Optional[FlightRecorder] = None,
                 tbo_hours: float = 1200.0):
        self.cfg = cfg or TwinConfig()
        n = self.cfg.engine.n_cyl
        self.reference = ReferenceModel(self.cfg)
        self.health = HealthMonitor(n, self.cfg.limits)
        self.isolator = FaultIsolator(n)
        self.rul = RULEstimator(tbo_hours=tbo_hours)
        self.advisor = AdvisoryEngine(self.cfg.limits)
        self.exceedance = ExceedanceMonitor(self.cfg.limits)
        self.anomaly = anomaly or AnomalyEngine(FEATURES)
        self.recorder = recorder
        self.engine_hours = 0.0
        # Ratio of accrued flight hours to elapsed simulation time. It is 1.0 in
        # service and on ordinary mission runs; a lifecycle study compresses wear
        # and must compress the twin's hour counter and trend clock by the same
        # factor, or the RUL filter is fed a slope in the wrong units.
        self.time_scale = 1.0
        self.t_s = 0.0
        self.regime = "off"
        self.last_snapshot: dict = {}
        self.alerts: List[dict] = []
        self._alert_keys: set = set()
        self._ukf: Optional[UnscentedKalmanFilter] = None
        self._ukf_names = ["rpm", "map_kpa"] + [f"cht{i}_c" for i in range(1, n + 1)] \
                          + ["coolant_temp_c", "oil_temp_c"]
        self.analytics_valid = False
        self.steady = SteadyStateDetector()
        self._resid_bias: Dict[str, float] = {}
        self._last_env: Tuple[AtmState, Command, float] = (isa(0), Command(), 45.0)
        self._degradation_view = {"total_hours": 0.0, "oil_hours": 0.0}
        self._prev_meas: Dict[str, Optional[float]] = {}
        self._frozen_counts: Dict[str, int] = {}
        self.feature_log: List[List[float]] = []
        self.collect_features = False

    # -- UKF wiring ---------------------------------------------------------
    def _build_ukf(self, atm: AtmState) -> None:
        cfg = self.cfg
        n = cfg.engine.n_cyl
        shadow = PistonEngine(cfg.engine, HealthParams(n_cyl=n))
        shadow.running = True
        names = self._ukf_names

        def fx(x: List[float], dt: float, u) -> List[float]:
            atm_u, cmd_u, tas_u, running_u = u
            st = shadow.state
            st.rpm = max(0.0, x[0])
            st.map_kpa = max(5.0, x[1])
            for i in range(n):
                st.cht_c[i] = x[2 + i]
            st.coolant_c = x[2 + n]
            st.oil_c = x[3 + n]
            # The plant's run state is observed, not inferred: without this the
            # shadow model stalls at zero speed and the filter cannot follow.
            shadow.running = running_u
            shadow.step(dt, cmd_u, atm_u, tas_u)
            return [st.rpm, st.map_kpa, *st.cht_c, st.coolant_c, st.oil_c]

        def hx(x: List[float], u) -> List[float]:
            return list(x)          # every filtered state is directly measured

        q = [9.0, 0.9] + [0.02] * n + [0.02, 0.02]
        self._ukf = UnscentedKalmanFilter(len(names), fx, hx, q)
        self._ukf.x = [1400.0, atm.press_kpa] + [atm.temp_c] * n + [atm.temp_c, atm.temp_c]
        self._ukf.P = [[(50.0 if i == j and i < 2 else (9.0 if i == j else 0.0))
                        for j in range(len(names))] for i in range(len(names))]

    def _seed_ukf(self, signals: Dict[str, Optional[float]], atm: AtmState) -> None:
        for i, n in enumerate(self._ukf_names):
            v = signals.get(n)
            if v is not None:
                self._ukf.x[i] = v

    def _r_diag(self) -> List[float]:
        s = self.cfg.sensors
        return [max(s[n].sigma, 1e-3) ** 2 if n in s else 1.0 for n in self._ukf_names]

    # -- main tick ------------------------------------------------------------
    def sync(self, signals: Dict[str, Optional[float]], dt: float,
             atm: AtmState, cmd: Command, tas_ms: float,
             t_s: Optional[float] = None,
             context: Optional[dict] = None) -> dict:
        """Advance the twin by one tick against a fresh set of engine signals.

        `context` is merged into the snapshot before it reaches the recorder, so
        labels the twin does not compute for itself - mission phase, tail number,
        sortie identifier - are present in the replay rather than being attached
        afterwards and lost.
        """
        self.t_s = self.t_s + dt if t_s is None else t_s
        self._last_env = (atm, cmd, tas_ms)
        rpm = signals.get("rpm") or 0.0
        running = rpm > 300.0
        if self._ukf is None:
            self._build_ukf(atm)
            self._seed_ukf(signals, atm)
        if running:
            self.engine_hours += dt / 3600.0 * self.time_scale

        # 1. reference (new-engine) prediction at the measured operating point
        ref = self.reference.step(dt, cmd, atm, tas_ms, signals)

        # 2. state estimation
        u = (atm, cmd, tas_ms, running)
        self._ukf.predict(dt, u)
        z = [signals.get(n) for n in self._ukf_names]
        self._ukf.update(z, self._r_diag(), u)
        est = dict(zip(self._ukf_names, self._ukf.x))
        resid = dict(zip(self._ukf_names, self._ukf.last_innovation)) \
            if self._ukf.last_innovation else {}

        # 3. physics-normalised indices
        analytics_valid_pre = running and rpm >= 2200.0
        steady = self.steady.update(self.t_s, rpm, signals.get("map_kpa") or 0.0,
                                    signals.get("throttle_pct") or 0.0,
                                    atm.altitude_m, atm.temp_c)
        extras = self._indices(signals, ref, atm, cmd, tas_ms, resid)
        self.regime = regime_of(rpm, extras.get("load", 0.0), atm.altitude_m)

        # 4. anomaly detection on the residual feature vector
        feats = self._features(signals, ref, resid, extras)
        if self.collect_features and rpm >= 2200.0 and self.regime not in ("off", "idle"):
            self.feature_log.append(feats)
        # Scoring an engine that is not running, or is below the governed
        # envelope, would feed the detectors a feature vector the reference model
        # cannot normalise - and it is exactly where a spurious alert would land
        # on the operator's display during start-up.
        if self.anomaly.trained and analytics_valid_pre:
            anom = self.anomaly.score(feats, steady=(analytics_valid_pre and steady),
                                      t_s=self.t_s, regime=self.regime)
        else:
            anom = {"score": 0.0, "raw_score": 0.0,
                    "status": "untrained" if not self.anomaly.trained else "warming_up",
                    "contributors": [], "drifting": []}

        # 5. condition indicators and health indices.
        #    The reference-ratio indices are only meaningful inside the governed
        #    envelope; at idle and during cranking they are held rather than
        #    allowed to inject noise into the trends that drive RUL.
        analytics_valid = analytics_valid_pre
        self.analytics_valid = analytics_valid
        # A genuine sensor fault biases its innovation; a throttle transient
        # produces a large but zero-mean one. Isolation therefore keys on the
        # slow mean of each innovation, not its instantaneous value.
        for k, v in resid.items():
            self._resid_bias[k] = self._resid_bias.get(k, 0.0) + 0.01 * (v - self._resid_bias.get(k, 0.0))
        pred = {n: ref_v for n, ref_v in self._ref_channels(ref).items()}
        ind = self.health.compute_indicators(signals, pred, resid, extras)
        indices = self.health.update(ind, anom, analytics_valid)

        # 6. threshold layer
        peak = self._peaks(signals)
        exc = self.exceedance.update({**{k: v for k, v in signals.items() if v is not None},
                                      **peak}, self.t_s)

        # 7. fault isolation
        if analytics_valid and steady:
            symptoms = self.isolator.symptoms(ind, signals, self._resid_bias, extras)
            hyps = self.isolator.isolate(symptoms)
        elif analytics_valid:
            symptoms = self.isolator.symptoms(ind, signals, {}, extras)
            hyps = self.isolator.isolate(symptoms)
        else:
            symptoms, hyps = self.isolator.last_symptoms, self.isolator.last_hypotheses

        # 8. RUL and advisories
        rul = self.rul.update(indices,
                              (dt / 3600.0 * self.time_scale)
                              if (analytics_valid and steady) else 0.0,
                              self.engine_hours)
        adv = self.advisor.generate(self.health.report(), self.rul.report(),
                                    [h.to_dict() for h in hyps],
                                    [e.to_dict() for e in exc],
                                    self._degradation_view)
        mission = self.advisor.mission_recommendation(adv, self.health.report())

        snap = self._snapshot(signals, ref, est, resid, ind, extras, anom, indices,
                              exc, hyps, rul, adv, mission, peak, running, atm, cmd)
        self._raise_alerts(exc, hyps, anom, adv)
        snap["alerts"] = self.alerts[-25:]
        if context:
            snap.update(context)
        self.last_snapshot = snap
        if self.recorder is not None:
            self.recorder.record(self.t_s, snap)
        self._prev_meas = dict(signals)
        return snap

    # -- derived indices --------------------------------------------------------
    def _indices(self, m: Dict[str, Optional[float]], ref: Outputs, atm: AtmState,
                 cmd: Command, tas_ms: float, resid: Dict[str, float]) -> Dict[str, float]:
        n = self.cfg.engine.n_cyl
        g = lambda k, d=0.0: (m.get(k) if m.get(k) is not None else d)
        rpm = g("rpm")
        e: Dict[str, float] = {}
        e["load"] = min(1.2, g("map_kpa") / max(self.cfg.engine.boost_target_kpa, 1.0))

        # Breathing: the ECU meters fuel to trapped air, so measured fuel flow
        # against the reference at the same rpm/MAP is an airflow proxy.
        ff, ff_ref = g("fuel_flow_lph"), ref.fuel_flow_lph
        e["ve_index"] = (ff / ff_ref) if ff_ref > 0.5 else 1.0

        # Delivered power, from the telemetered governor blade angle. Power
        # absorbed by a constant-speed propeller is k(beta) * sigma * omega^3,
        # so blade angle plus speed plus air density gives shaft power without a
        # torque sensor. Comparing it against the power a new engine would make
        # at this rpm and MAP is the cleanest single measure of lost output.
        sp = self.cfg.engine
        beta = g("prop_load_pct", -1.0)
        if beta >= 0.0 and rpm > 1500:
            k = sp.prop_k * (sp.prop_pitch_min
                             + (beta / 100.0) * (sp.prop_pitch_max - sp.prop_pitch_min))
            p_est = k * atm.sigma * (rpm * 2 * math.pi / 60.0) ** 3 / 1000.0
            e["power_kw_est"] = p_est
            e["power_index"] = (p_est / ref.power_kw) if ref.power_kw > 1.0 else 1.0
        else:
            e["power_kw_est"] = ref.power_kw
            e["power_index"] = 1.0

        # Specific-consumption consistency: fuel actually burned per unit power.
        e["ff_index"] = (e["ve_index"] / max(e["power_index"], 1e-3))
        e["friction_index"] = max(0.5, min(2.0, e["ff_index"]))

        # Boost tracking error: measured MAP against the pressure a healthy air
        # path would reach at this throttle setting and altitude.
        healthy_map = getattr(self.reference.engine, "last_map_target", None)
        e["boost_error_kpa"] = (g("map_kpa") - healthy_map) if healthy_map else 0.0
        e["boost_target_kpa"] = healthy_map or 0.0
        oat = g("oat_c", atm.temp_c)
        d_ref, d_meas = ref.iat_c - oat, g("iat_c") - oat
        # Compressor efficiency is inferred from temperature rise across the
        # charge path. Below a useful pressure ratio the rise is comparable to
        # the sensor noise, so the index is reported as nominal rather than as
        # a noisy number the trend filters would chase.
        e["comp_eff_index"] = ((d_ref / d_meas)
                               if (d_meas > 6.0 and ref.boost_pr > 1.18) else 1.0)
        e["iat_model_error_k"] = g("iat_c") - ref.iat_c

        # Cooling: at the same heat input, a higher CHT rise above ambient means
        # less heat-rejection capability.
        cht = [g(f"cht{i}_c", -99) for i in range(1, n + 1)]
        cht_v = [c for c in cht if c > -50]
        cht_mean = sum(cht_v) / len(cht_v) if cht_v else oat
        r_rise = max(1.0, sum(ref.cht_c) / n - oat)
        m_rise = max(1.0, cht_mean - oat)
        # Loss of heat-rejection *capability*, which appears long before any
        # temperature limit is approached - at altitude a badly fouled radiator
        # may never push CHT past its limit at all, yet the margin has gone.
        e["cooling_temp_index"] = min(1.6, r_rise / m_rise) if rpm > 1500 else 1.0
        e["cht_model_error_c"] = cht_mean - sum(ref.cht_c) / n

        # Remaining regulator travel is the sensitive measure of cooling health.
        # A degraded radiator is compensated by the thermostat opening further,
        # so temperature barely moves until the regulator saturates - at which
        # point the margin is already gone. Comparing the shutter position this
        # engine needs against the position a healthy one would need exposes the
        # loss while there is still reserve to lose.
        # The comparison is only well posed while the *reference* still has
        # travel to spare. On a hot-day full-power climb a healthy engine also
        # runs its shutter wide open, and a ratio of two saturated positions
        # says nothing; there the temperature index carries the assessment.
        v_meas = g("coolant_valve_pct", -1.0)
        v_ref = ref.coolant_valve_pct
        if v_meas >= 0.0 and rpm > 2000 and v_ref < 92.0:
            e["cooling_reserve_index"] = max(0.0, min(1.5, (100.0 - v_meas)
                                                      / max(100.0 - v_ref, 20.0)))
        else:
            e["cooling_reserve_index"] = 1.0
        e["coolant_valve_pct"] = max(0.0, v_meas)
        # The health layer consumes the weaker of the two: a saturated regulator
        # and a raised temperature are both loss of cooling capability.
        e["cooling_effectiveness"] = min(e["cooling_temp_index"],
                                         e["cooling_reserve_index"])

        # Per-cylinder CHT against the reference model - the analytical
        # redundancy a drifting thermocouple has no way to satisfy.
        e["cht_max_model_dev_c"] = max(
            (abs(g(f"cht{i+1}_c", ref.cht_c[i]) - ref.cht_c[i]) for i in range(n)),
            default=0.0)

        # The healthy oil-pressure reference must be evaluated at the *measured*
        # oil temperature, otherwise a hot oil charge is indistinguishable from a
        # worn pump.
        op = g("oil_press_kpa")
        op_ref = self.reference.engine.oil_pressure_at(rpm, g("oil_temp_c", 90.0))
        e["oil_press_ref_kpa"] = op_ref
        e["oil_press_index"] = (op / op_ref) if op_ref > 20 else 1.0

        # Per-cylinder fuelling balance from EGT dispersion.
        egt = [g(f"egt{i}_c", -99) for i in range(1, n + 1)]
        egt_v = [x for x in egt if x > 0]
        if len(egt_v) >= 2:
            mu = sum(egt_v) / len(egt_v)
            e["injector_balance"] = (math.sqrt(sum((x - mu) ** 2 for x in egt_v)
                                               / len(egt_v)) / max(mu, 1.0))
        else:
            e["injector_balance"] = 0.0
        e["lambda_est"] = ref.lambda_avg * (1.0 / max(e["ve_index"], 0.2)) \
            if e["ve_index"] > 0 else ref.lambda_avg

        # Order-resolved vibration comes straight off the bus. A four-stroke
        # repeats its firing sequence every two revolutions, so a missing
        # combustion event lands on the 0.5 order while rotating imbalance lands
        # on 1x - the separation that distinguishes a misfiring cylinder from a
        # worn bearing.
        e["vib_halforder_g"] = g("vib_half_g")
        e["vib_1x_g"] = g("vib_1x_g")
        e["vib_broadband_g"] = g("vib_broad_g")
        e["vib_half_excess"] = g("vib_half_g") - ref.vib_halforder_g
        e["vib_1x_excess"] = g("vib_1x_g") - ref.vib_1x_g
        # Detonation is read from the knock sensor, not inferred from the
        # reference model - the reference always has in-spec fuel and would
        # therefore never report knock.
        e["knock_index"] = g("knock_level") / 100.0
        e["blowby_index"] = 1.0 + 3.0 * max(0.0, 1.0 - e["ve_index"])

        alt_exp = max(1.0, abs(ref.alt_current_a))
        e["alt_index"] = min(1.5, abs(g("alt_current_a")) / alt_exp) if rpm > 2500 else 1.0

        # Frozen-channel detector: a live channel that has not moved at all.
        frozen = 0
        for k, v in m.items():
            if v is None or k not in self._prev_meas:
                continue
            pv = self._prev_meas.get(k)
            if pv is not None and abs(v - pv) < 1e-9 and rpm > 2000:
                self._frozen_counts[k] = self._frozen_counts.get(k, 0) + 1
            else:
                self._frozen_counts[k] = 0
            if self._frozen_counts.get(k, 0) > 120:
                frozen += 1
        e["frozen_channels"] = float(frozen)
        return e

    def _ref_channels(self, ref: Outputs) -> Dict[str, float]:
        d = {"rpm": ref.rpm, "map_kpa": ref.map_kpa, "iat_c": ref.iat_c,
             "oil_press_kpa": ref.oil_press_kpa, "oil_temp_c": ref.oil_temp_c,
             "coolant_temp_c": ref.coolant_c, "fuel_flow_lph": ref.fuel_flow_lph,
             "vib_rms_g": ref.vib_rms_g, "bus_v": ref.bus_v,
             "alt_current_a": ref.alt_current_a,
             "prop_load_pct": ref.prop_load_pct, "throttle_pct": ref.throttle_pct,
             "coolant_valve_pct": ref.coolant_valve_pct, "knock_level": ref.knock_level}
        for i, v in enumerate(ref.cht_c, 1):
            d[f"cht{i}_c"] = v
        for i, v in enumerate(ref.egt_c, 1):
            d[f"egt{i}_c"] = v
        return d

    def _features(self, m: Dict[str, Optional[float]], ref: Outputs,
                  resid: Dict[str, float], e: Dict[str, float]) -> List[float]:
        n = self.cfg.engine.n_cyl
        g = lambda k, d=0.0: (m.get(k) if m.get(k) is not None else d)
        egt = [g(f"egt{i}_c", 0.0) for i in range(1, n + 1)]
        egt_v = [x for x in egt if x > 0]
        mu = sum(egt_v) / len(egt_v) if egt_v else 1.0
        cht = [g(f"cht{i}_c", 0.0) for i in range(1, n + 1)]
        cht_v = [x for x in cht if x > -50]

        f = [
            resid.get("rpm", 0.0), resid.get("map_kpa", 0.0),
            (g("iat_c") - ref.iat_c) / 10.0,
            resid.get("oil_temp_c", 0.0) * 0 + (g("oil_press_kpa") - ref.oil_press_kpa) / 40.0,
            resid.get("oil_temp_c", 0.0), resid.get("coolant_temp_c", 0.0),
            (g("fuel_flow_lph") - ref.fuel_flow_lph) / 1.5,
            (g("vib_rms_g") - ref.vib_rms_g) / 0.2,
            (g("bus_v") - ref.bus_v) / 0.2,
        ]
        f += [resid.get(f"cht{i}_c", 0.0) for i in range(1, n + 1)]
        f += [(x - mu) / 25.0 for x in egt]
        f += [
            (max(cht_v) - min(cht_v)) / 6.0 if len(cht_v) > 1 else 0.0,
            e.get("ve_index", 1.0), e.get("ff_index", 1.0), e.get("power_index", 1.0),
            e.get("cooling_effectiveness", 1.0), e.get("oil_press_index", 1.0),
            e.get("comp_eff_index", 1.0),
            e.get("vib_halforder_g", 0.0), e.get("vib_1x_g", 0.0),
            e.get("boost_error_kpa", 0.0) / 5.0, e.get("knock_index", 0.0),
            e.get("cooling_reserve_index", 1.0),
        ]
        return [0.0 if (x != x or x in (float("inf"), float("-inf"))) else x for x in f]

    def _peaks(self, m: Dict[str, Optional[float]]) -> Dict[str, float]:
        n = self.cfg.engine.n_cyl
        cht = [m.get(f"cht{i}_c") for i in range(1, n + 1)]
        egt = [m.get(f"egt{i}_c") for i in range(1, n + 1)]
        cht = [c for c in cht if c is not None]
        egt = [x for x in egt if x is not None]
        return {"cht_max_c": max(cht) if cht else 0.0,
                "egt_max_c": max(egt) if egt else 0.0}

    # -- alerting ----------------------------------------------------------------
    def _raise_alerts(self, exc, hyps, anom, adv) -> None:
        if not self.analytics_valid:
            # Threshold exceedances still latch below the governed envelope; the
            # predictive layers do not speak until they have valid inputs.
            hyps, anom, adv = [], {"status": "warming_up"}, []
        new = []
        for e in exc:
            key = f"exc:{e.channel}:{e.severity}"
            if key not in self._alert_keys:
                self._alert_keys.add(key)
                new.append({"t_s": round(self.t_s, 1), "level": "warning" if e.severity == "limit" else "caution",
                            "source": "threshold",
                            "text": f"{e.channel} {e.severity}: {e.value:.1f} {e.unit} "
                                    f"vs {e.limit:.1f} {e.unit}"})
        # Only the leading hypothesis reaches the operator. Overlapping
        # signatures always leave a tail of runners-up whose cylinder
        # attribution moves around; alerting on all of them turns one fault into
        # a stream of contradictory messages.
        for h in hyps[:1]:
            if h.confidence < 0.55:
                continue
            key = f"fdi:{h.fault}"
            if key not in self._alert_keys:
                self._alert_keys.add(key)
                where = f" cyl {h.cylinder}" if h.cylinder else (f" [{h.channel}]" if h.channel else "")
                new.append({"t_s": round(self.t_s, 1), "level": "caution", "source": "fdi",
                            "text": f"Probable {h.fault.replace('_',' ')}{where} "
                                    f"({h.confidence:.0%} confidence)"})
        if anom.get("status") in ("anomaly", "severe"):
            key = f"anom:{anom['status']}"
            if key not in self._alert_keys:
                self._alert_keys.add(key)
                top = anom.get("contributors", [{}])[0].get("feature", "?")
                new.append({"t_s": round(self.t_s, 1),
                            "level": "warning" if anom["status"] == "severe" else "caution",
                            "source": "ml",
                            "text": f"Anomaly score {anom['score']:.2f} ({anom['status']}), "
                                    f"led by {top}"})
        for a in adv:
            if a.urgency in ("ABORT", "IMMEDIATE"):
                # Keyed on subsystem and urgency rather than on the full title, so
                # a hypothesis that migrates between cylinders raises one alert
                # rather than one per cylinder.
                key = f"adv:{a.urgency}:{a.subsystem}"
                if key not in self._alert_keys:
                    self._alert_keys.add(key)
                    new.append({"t_s": round(self.t_s, 1), "level": "warning",
                                "source": "advisory", "text": f"{a.urgency}: {a.title}"})
        for a in new:
            self.alerts.append(a)
            if self.recorder is not None:
                self.recorder.log_event(self.t_s, a["source"], a["level"], a["text"],
                                        dump_ring=(a["level"] == "warning"))

    # -- snapshot ------------------------------------------------------------------
    def _snapshot(self, m, ref, est, resid, ind, extras, anom, indices, exc, hyps,
                  rul, adv, mission, peak, running, atm, cmd) -> dict:
        n = self.cfg.engine.n_cyl
        return {
            "t_s": round(self.t_s, 2),
            "wall_utc": round(time.time(), 2),
            "running": running,
            "analytics_valid": self.analytics_valid,
            "steady_state": self.steady.steady,
            "steady_reason": self.steady.reason,
            "regime": self.regime,
            "engine_hours": round(self.engine_hours, 3),
            "env": {"altitude_m": round(atm.altitude_m, 1), "oat_c": round(atm.temp_c, 1),
                    "amb_press_kpa": round(atm.press_kpa, 2), "density_ratio": round(atm.sigma, 3),
                    "isa_dev_k": round(atm.isa_dev_k, 1)},
            "cmd": {"throttle_pct": round(cmd.throttle_pct, 1),
                    "rpm_target": round(cmd.rpm_target, 0),
                    "elec_load_w": round(cmd.elec_load_w, 0)},
            "meas": {k: (None if v is None else round(v, 2)) for k, v in m.items()},
            "ref": {k: round(v, 2) for k, v in self._ref_channels(ref).items()},
            "est": {k: round(v, 2) for k, v in est.items()},
            "est_sigma": {k: round(v, 3) for k, v in zip(self._ukf_names, self._ukf.std)},
            "resid": {k: round(v, 3) for k, v in resid.items()},
            "resid_bias": {k: round(v, 3) for k, v in self._resid_bias.items()},
            "peak": {k: round(v, 1) for k, v in peak.items()},
            "derived": {
                "power_kw": round(extras.get("power_kw_est", ref.power_kw), 2),
                "power_ref_kw": round(ref.power_kw, 2),
                "torque_nm": round(ref.torque_nm, 1),
                "bsfc_g_kwh": round(ref.bsfc_g_kwh * extras.get("ff_index", 1.0), 0),
                "ve_index": round(extras.get("ve_index", 1.0), 4),
                "power_index": round(extras.get("power_index", 1.0), 4),
                "ff_index": round(extras.get("ff_index", 1.0), 4),
                "cooling_effectiveness": round(extras.get("cooling_effectiveness", 1.0), 4),
                "cooling_reserve_index": round(extras.get("cooling_reserve_index", 1.0), 4),
                "cooling_temp_index": round(extras.get("cooling_temp_index", 1.0), 4),
                "coolant_valve_pct": round(extras.get("coolant_valve_pct", 0.0), 1),
                "oil_press_index": round(extras.get("oil_press_index", 1.0), 4),
                "oil_press_ref_kpa": round(extras.get("oil_press_ref_kpa", 0.0), 1),
                "comp_eff_index": round(extras.get("comp_eff_index", 1.0), 4),
                "vib_halforder_g": round(extras.get("vib_halforder_g", 0.0), 3),
                "vib_1x_g": round(extras.get("vib_1x_g", 0.0), 3),
                "vib_broadband_g": round(extras.get("vib_broadband_g", 0.0), 3),
                "knock_index": round(extras.get("knock_index", 0.0), 3),
                "boost_error_kpa": round(extras.get("boost_error_kpa", 0.0), 2),
                "boost_target_kpa": round(extras.get("boost_target_kpa", 0.0), 1),
                "injector_balance": round(extras.get("injector_balance", 0.0), 4),
                "lambda_est": round(extras.get("lambda_est", 1.0), 3),
                "prop_rpm": round(ref.prop_rpm, 0),
            },
            "anomaly": {k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in anom.items() if k != "per_feature"},
            "health": self.health.report(),
            "exceedances": [e.to_dict() for e in exc],
            "diagnostics": self.isolator.report(),
            "rul": self.rul.report(),
            "advisories": [a.to_dict() for a in adv[:8]],
            "mission": mission,
            "degradation": self._degradation_view,
        }

    def set_degradation_view(self, d: dict) -> None:
        """Hours-based counters supplied by the operator's maintenance record
        (or, in simulation, by the degradation model)."""
        self._degradation_view = d
