"""
Closed-loop simulation harness.

Wires the full chain exactly as it would be wired on the aircraft:

    plant engine -> sensors -> CAN frames -> [bus] -> decode -> Digital Twin

The twin never touches the plant object or its ground-truth health state; it
only ever sees decoded CAN signals. That is what makes the accuracy numbers
produced by `scripts/evaluate.py` meaningful rather than circular.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional

from .anomaly import AnomalyEngine
from .canbus import BaseCanBus, VirtualCanBus, open_bus
from .config import TwinConfig
from .degradation import DegradationModel
from .engine_model import Command, HealthParams, PistonEngine
from .faults import FaultEvent, FaultInjector
from .missions import MissionProfile, get_profile
from .recorder import FlightRecorder
from .sensors import SensorSuite
from .twin import DigitalTwin, FEATURES


@dataclass
class SimConfig:
    profile: str = "isr_sortie"
    scenario: str = "healthy"
    dt: float = 0.05
    seed: int = 42
    initial_hours: float = 380.0        # engine time already on the log
    degradation_accel: float = 1.0      # >1 compresses wear for lifecycle studies
    use_can: bool = True
    prefer_socketcan: bool = False
    record: bool = False
    fdr_path: str = "data/fdr.sqlite"
    tbo_hours: float = 1200.0
    sample_stride: int = 20             # snapshots emitted every N steps


class MissionSimulator:
    def __init__(self, sim: Optional[SimConfig] = None,
                 cfg: Optional[TwinConfig] = None,
                 anomaly: Optional[AnomalyEngine] = None,
                 extra_faults: Optional[List[FaultEvent]] = None):
        self.sim = sim or SimConfig()
        self.cfg = cfg or TwinConfig()
        n = self.cfg.engine.n_cyl
        self.rng = random.Random(self.sim.seed)

        # --- plant ---------------------------------------------------------
        self.base_health = HealthParams(n_cyl=n)
        self.plant = PistonEngine(self.cfg.engine, self.base_health.copy(), seed=self.sim.seed)
        self.degradation = DegradationModel(accel=self.sim.degradation_accel)
        self.degradation.total_hours = self.sim.initial_hours
        self.degradation.oil_hours = self.sim.initial_hours % 100.0
        self._preage(self.sim.initial_hours)

        events = FaultInjector.scenario(self.sim.scenario, n) + list(extra_faults or [])
        self.injector = FaultInjector(events, seed=self.sim.seed + 1)

        # --- acquisition ----------------------------------------------------
        self.sensors = SensorSuite(self.cfg.sensors, seed=self.sim.seed + 2)
        self.bus: Optional[BaseCanBus] = (
            open_bus(self.cfg.can_channel, self.sim.prefer_socketcan)
            if self.sim.use_can else None)

        # --- twin -------------------------------------------------------------
        self.recorder = FlightRecorder(self.sim.fdr_path) if self.sim.record else None
        self.twin = DigitalTwin(self.cfg, anomaly, self.recorder,
                                tbo_hours=self.sim.tbo_hours)
        self.twin.engine_hours = self.sim.initial_hours
        # Keep the twin's flight-hour clock on the same scale as the wear model.
        self.twin.time_scale = self.sim.degradation_accel
        self.profile: MissionProfile = get_profile(self.sim.profile)
        self.t = 0.0
        self.step_count = 0

    # -- pre-ageing --------------------------------------------------------
    OIL_INTERVAL_H = 100.0
    PLUG_INTERVAL_H = 200.0
    BATTERY_INTERVAL_H = 500.0

    def _preage(self, hours: float) -> None:
        """Age the engine to `hours` on a representative cruise duty cycle,
        applying the scheduled maintenance it would have received on the way.

        Skipping the maintenance schedule is what makes naive ageing models
        absurd: an engine at 380 h has had its oil changed three times and its
        plugs replaced once, so it must not carry 380 h of consumable wear.
        """
        if hours <= 0:
            return
        from .atmosphere import isa
        atm = isa(5000)
        e = PistonEngine(self.cfg.engine, self.base_health, seed=self.sim.seed)
        e.state.oil_c = 100.0
        e.state.coolant_c = 95.0
        e.state.cht_c = [112.0] * self.cfg.engine.n_cyl
        e.start(atm)
        cmd = Command(throttle_pct=72, rpm_target=5000)
        for _ in range(60):
            o = e.step(0.5, cmd, atm, 50.0)

        d = DegradationModel(accel=1.0)
        step_h, done = 5.0, 0.0
        while done < hours:
            block = min(step_h, hours - done)
            d.advance(self.base_health, o, block * 3600.0, self.cfg.engine.rated_rpm)
            done += block
            if int(done // self.OIL_INTERVAL_H) > int((done - block) // self.OIL_INTERVAL_H):
                d.service(self.base_health, "oil_change")
            if int(done // self.PLUG_INTERVAL_H) > int((done - block) // self.PLUG_INTERVAL_H):
                d.service(self.base_health, "spark_plugs")
            if int(done // self.BATTERY_INTERVAL_H) > int((done - block) // self.BATTERY_INTERVAL_H):
                # The battery is a life-limited item, not a wear-to-failure one.
                self.base_health.battery_soh = 1.0
        d.total_hours = hours
        self.degradation.oil_hours = hours % self.OIL_INTERVAL_H
        self.plug_hours = hours % self.PLUG_INTERVAL_H
        self.plant.health = self.base_health.copy()

    # -- one step -----------------------------------------------------------
    def step(self) -> Optional[dict]:
        dt = self.sim.dt
        atm, cmd, tas, phase = self.profile.sample(self.t)

        # start the engine once the profile calls for power
        if not self.plant.running:
            if self.t < 1.0:
                self.plant.soak(atm)
            if cmd.throttle_pct > 5.0 and self.t > 2.0:
                self.plant.start(atm)

        # faults compose on top of the slowly degrading baseline
        self.plant.health = self.injector.apply(self.base_health.copy(), self.t)
        o = self.plant.step(dt, cmd, atm, tas)
        self.degradation.advance(self.base_health, o, dt, self.cfg.engine.rated_rpm)

        # acquisition
        meas = self.sensors.measure(o, atm, self.t, dt / 3600.0,
                                    self.injector.sensor_faults, tas)
        if self.bus is not None:
            self.bus.publish(meas, self.t)
            signals = BaseCanBus.frames_to_signals(self.bus.recv())
            # channels not carried by a PGN due this tick keep their last value
            merged = dict(getattr(self, "_last_signals", {}))
            merged.update(signals)
            signals = merged
            self._last_signals = dict(signals)
        else:
            signals = meas

        self.twin.set_degradation_view(self.degradation.snapshot())
        snap = self.twin.sync(signals, dt, atm, cmd, tas, t_s=self.t,
                              context={"phase": phase, "truth": self._truth(o)})
        self.t += dt
        self.step_count += 1
        return snap

    def _truth(self, o) -> dict:
        """Ground truth, recorded for scoring only. The twin never reads this."""
        h = self.plant.health
        return {"active_faults": list(self.injector.active),
                "power_kw": round(o.power_kw, 2),
                "lambda_avg": round(o.lambda_avg, 3),
                "misfire_rate": [round(x, 3) for x in o.misfire_rate],
                "health": {"ring_seal": round(h.ring_seal, 4),
                           "valve_seal": round(h.valve_seal, 4),
                           "cooling_ua": round(h.cooling_ua, 4),
                           "oil_pump": round(h.oil_pump, 4),
                           "bearing_clearance": round(h.bearing_clearance, 4),
                           "turbo_efficiency": round(h.turbo_efficiency, 4),
                           "injector_flow": [round(x, 4) for x in h.injector_flow],
                           "ignition_energy": [round(x, 4) for x in h.ignition_energy]}}

    # -- run -----------------------------------------------------------------
    def run(self, duration_s: Optional[float] = None,
            on_sample: Optional[Callable[[dict], None]] = None,
            progress: bool = False) -> List[dict]:
        end = duration_s if duration_s is not None else self.profile.duration_s
        if self.recorder is not None:
            self.recorder.start_session(
                f"{self.sim.profile}/{self.sim.scenario}", self.sim.profile,
                self.sim.scenario, config=self.cfg.to_dict())
        samples: List[dict] = []
        next_pct = 10
        while self.t < end:
            snap = self.step()
            if snap is not None and self.step_count % self.sim.sample_stride == 0:
                samples.append(snap)
                if on_sample:
                    on_sample(snap)
            if progress:
                pct = 100.0 * self.t / max(end, 1e-9)
                if pct >= next_pct:
                    print(f"    {next_pct:3d}%  t={self.t/60:6.1f} min  "
                          f"health={snap['health']['overall']:5.1f}  "
                          f"anomaly={snap['anomaly'].get('score',0):.2f}")
                    next_pct += 10
        if self.recorder is not None:
            self.recorder.end_session()
        return samples

    def iter_run(self, duration_s: Optional[float] = None) -> Iterator[dict]:
        end = duration_s if duration_s is not None else self.profile.duration_s
        if self.recorder is not None:
            self.recorder.start_session(
                f"{self.sim.profile}/{self.sim.scenario}", self.sim.profile,
                self.sim.scenario, config=self.cfg.to_dict())
        while self.t < end:
            snap = self.step()
            if self.step_count % self.sim.sample_stride == 0:
                yield snap
        if self.recorder is not None:
            self.recorder.end_session()

    def close(self) -> None:
        if self.bus is not None:
            self.bus.close()
        if self.recorder is not None:
            self.recorder.close()
