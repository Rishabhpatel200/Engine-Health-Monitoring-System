"""
Measurement chain: truth -> sensor -> ECU frame.

Applies, in order, the effects a real acquisition path introduces:
  1. transducer noise (white, per-channel sigma)
  2. calibration bias and in-service drift (accumulated per flight hour)
  3. injected sensor faults (drift / bias / stuck / dropout)
  4. ADC quantisation and range saturation
  5. channel update rate (zero-order hold between CAN updates)

Keeping this separate from the plant is what lets the twin distinguish a real
overheat from a lying thermocouple: the estimator sees only this output.
"""

from __future__ import annotations

import random
from typing import Dict, Optional

from .config import SensorSpec
from .engine_model import Outputs


class SensorSuite:
    def __init__(self, specs: Dict[str, SensorSpec], seed: int = 11,
                 enable_noise: bool = True):
        self.specs = specs
        self.rng = random.Random(seed)
        self.enable_noise = enable_noise
        # Each unit has a fixed as-built calibration bias, drawn once.
        self.bias: Dict[str, float] = {
            k: self.rng.gauss(0.0, v.sigma * 0.6) for k, v in specs.items()}
        self.drift: Dict[str, float] = {k: 0.0 for k in specs}
        self._held: Dict[str, float] = {}
        self._next_update: Dict[str, float] = {k: 0.0 for k in specs}
        self.dropped: Dict[str, bool] = {k: False for k in specs}

    # -- truth extraction ----------------------------------------------------
    @staticmethod
    def truth(o: Outputs, atm, tas_ms: float = 45.0) -> Dict[str, float]:
        t = {
            "rpm": o.rpm,
            "map_kpa": o.map_kpa,
            "iat_c": o.iat_c,
            "oil_press_kpa": o.oil_press_kpa,
            "oil_temp_c": o.oil_temp_c,
            "coolant_temp_c": o.coolant_c,
            "fuel_flow_lph": o.fuel_flow_lph,
            "vib_rms_g": o.vib_rms_g,
            "vib_half_g": o.vib_halforder_g,
            "vib_1x_g": o.vib_1x_g,
            "vib_broad_g": o.vib_broadband_g,
            "knock_level": o.knock_level,
            "coolant_valve_pct": o.coolant_valve_pct,
            "bus_v": o.bus_v,
            "alt_current_a": o.alt_current_a,
            "throttle_pct": o.throttle_pct,
            "prop_load_pct": o.prop_load_pct,
            "amb_press_kpa": atm.press_kpa,
            "oat_c": atm.temp_c,
        }
        for i, v in enumerate(o.cht_c, 1):
            t[f"cht{i}_c"] = v
        for i, v in enumerate(o.egt_c, 1):
            t[f"egt{i}_c"] = v
        return t

    # -- measurement ---------------------------------------------------------
    def measure(self, o: Outputs, atm, t_s: float, dt_hours: float,
                faults: Optional[Dict[str, dict]] = None,
                tas_ms: float = 45.0) -> Dict[str, Optional[float]]:
        faults = faults or {}
        truth = self.truth(o, atm, tas_ms)
        out: Dict[str, Optional[float]] = {}

        for name, spec in self.specs.items():
            if name not in truth:
                continue
            # (5) channel update rate - hold the previous sample between updates
            if t_s < self._next_update[name] and name in self._held:
                out[name] = None if self.dropped[name] else self._held[name]
                continue
            self._next_update[name] = t_s + 1.0 / max(spec.rate_hz, 1e-6)

            v = truth[name]
            # (2) in-service drift accumulates with flight hours
            self.drift[name] += spec.drift_per_hr * dt_hours
            v += self.bias[name] + self.drift[name]

            # (1) transducer noise
            if self.enable_noise and spec.sigma > 0:
                v += self.rng.gauss(0.0, spec.sigma)

            # (3) injected sensor faults
            f = faults.get(name)
            self.dropped[name] = False
            if f:
                kind, sev = f["kind"], f["severity"]
                if kind == "sensor_drift":
                    v += sev
                elif kind == "sensor_bias":
                    v += sev
                elif kind == "sensor_stuck":
                    v = self._held.get(name, v)
                elif kind == "sensor_dropout":
                    if self.rng.random() < sev:
                        self.dropped[name] = True

            # (4) quantisation and range saturation
            if spec.resolution > 0:
                v = round(v / spec.resolution) * spec.resolution
            v = max(spec.range_lo, min(spec.range_hi, v))

            self._held[name] = v
            out[name] = None if self.dropped[name] else v

        return out

    def drift_report(self) -> Dict[str, float]:
        return {k: round(v, 3) for k, v in self.drift.items() if abs(v) > 1e-6}
