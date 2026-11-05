"""
Health monitoring: observable condition indicators and subsystem health indices.

Everything here is computed from what the twin can actually *see* - measurements
plus model predictions - never from the simulator's ground-truth wear state.
That distinction matters: it is what makes the same code valid on a real engine.

Each subsystem gets a health index on 0-100 with an explicit confidence, so the
maintainer can tell "this engine is degrading" from "the twin has not yet seen
enough of this regime to judge".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

from .anomaly import EWMA
from .config import Limits, SUBSYSTEMS

# Which subsystem each anomaly feature speaks for. The fused anomaly score is
# strong evidence that *something* is wrong, but applying it uniformly would
# drag all eight indices to the same number and destroy exactly the localisation
# the operator needs. Routing it by contributor share keeps the score's
# sensitivity while preserving where it points.
FEATURE_SUBSYSTEM: Dict[str, str] = {
    "r_rpm": "mechanical", "r_map": "air_path", "r_iat": "air_path",
    "boost_err": "air_path", "comp_eff_index": "air_path", "ve_index": "air_path",
    "r_oil_press": "lubrication", "oil_press_index": "lubrication",
    "r_oil_temp": "lubrication",
    "r_coolant": "cooling", "cooling_eff": "cooling", "cool_reserve": "cooling",
    "cht_spread": "cooling",
    "r_cht1": "cooling", "r_cht2": "cooling", "r_cht3": "cooling", "r_cht4": "cooling",
    "r_fuel_flow": "fuel_injection", "ff_index": "fuel_injection",
    "egt_dev1": "fuel_injection", "egt_dev2": "fuel_injection",
    "egt_dev3": "fuel_injection", "egt_dev4": "fuel_injection",
    "r_vib": "mechanical", "vib_1x": "mechanical", "power_index": "mechanical",
    "vib_half": "combustion", "knock": "combustion",
    "r_bus_v": "electrical",
}


def _clip(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def _margin_index(value: float, caution: float, limit: float,
                  higher_is_worse: bool = True) -> float:
    """Map a measurement to 0-100 against its caution and limit values.
    100 at or below caution, 0 at the certified limit.

    The reference-ratio indices are scored against *serviceability*, not against
    a zero-time engine. An engine three quarters of the way to overhaul has
    legitimately lost around 12 % of its breathing and 7 % of its power; scoring
    those against a new engine would leave every mid-life airframe permanently
    reading 70 and raising standing advisories, which is how a health monitoring
    system teaches its users to ignore it. The caution points below are set to
    the wear a serviceable engine may carry, and the limit points to the wear at
    which it may not.
    """
    if not higher_is_worse:
        value, caution, limit = -value, -caution, -limit
    if value <= caution:
        return 100.0
    if value >= limit:
        return 0.0
    return 100.0 * (limit - value) / max(limit - caution, 1e-6)


@dataclass
class Indicator:
    name: str
    unit: str
    subsystem: str
    value: float = 0.0
    baseline: float = 0.0
    sigma: float = 1.0
    description: str = ""

    @property
    def z(self) -> float:
        return (self.value - self.baseline) / max(self.sigma, 1e-9)


class HealthMonitor:
    def __init__(self, n_cyl: int = 4, limits: Optional[Limits] = None):
        self.n_cyl = n_cyl
        self.limits = limits or Limits()
        self.indices: Dict[str, float] = {s: 100.0 for s in SUBSYSTEMS}
        self.confidence: Dict[str, float] = {s: 0.0 for s in SUBSYSTEMS}
        self._smooth: Dict[str, EWMA] = {}
        self._idx_smooth: Dict[str, EWMA] = {s: EWMA(0.02, 100.0) for s in SUBSYSTEMS}
        self._samples = 0

    # -- indicator computation ---------------------------------------------
    def compute_indicators(self, meas: Dict[str, Optional[float]],
                           pred: Dict[str, float],
                           resid: Dict[str, float],
                           extras: Dict[str, float]) -> Dict[str, float]:
        """Build the condition-indicator set.

        `meas`   measured channels (may contain None on dropout)
        `pred`   twin predictions for the same channels at this operating point
        `resid`  normalised innovations from the UKF
        `extras` derived quantities (VE index, order-resolved vibration, ...)
        """
        n = self.n_cyl
        g = lambda k, d=0.0: (meas.get(k) if meas.get(k) is not None else d)

        cht = [g(f"cht{i}_c") for i in range(1, n + 1)]
        egt = [g(f"egt{i}_c") for i in range(1, n + 1)]
        cht_valid = [c for c in cht if c > -50]
        egt_valid = [e for e in egt if e > 0]
        rpm = g("rpm")
        ind: Dict[str, float] = {}

        # --- combustion -----------------------------------------------------
        if len(egt_valid) >= 2:
            m = sum(egt_valid) / len(egt_valid)
            ind["egt_spread_c"] = max(egt_valid) - min(egt_valid)
            ind["egt_mean_c"] = m
            ind["egt_max_dev_c"] = max(abs(e - m) for e in egt_valid)
        ind["misfire_order_g"] = extras.get("vib_halforder_g", 0.0)
        ind["knock_index"] = extras.get("knock_index", 0.0)

        # --- air path -------------------------------------------------------
        ind["ve_index"] = extras.get("ve_index", 1.0)          # measured/expected
        ind["boost_error_kpa"] = extras.get("boost_error_kpa", 0.0)
        ind["iat_rise_k"] = g("iat_c") - g("oat_c")
        ind["comp_eff_index"] = extras.get("comp_eff_index", 1.0)

        # --- fuel injection --------------------------------------------------
        ind["fuel_flow_index"] = extras.get("fuel_flow_index", 1.0)
        ind["injector_balance"] = extras.get("injector_balance", 0.0)
        ind["lambda_est"] = extras.get("lambda_est", 1.0)

        # --- cooling ---------------------------------------------------------
        if cht_valid:
            ind["cht_max_c"] = max(cht_valid)
            ind["cht_spread_c"] = max(cht_valid) - min(cht_valid)
        ind["coolant_temp_c"] = g("coolant_temp_c")
        ind["cooling_effectiveness"] = extras.get("cooling_effectiveness", 1.0)
        ind["cht_model_error_c"] = extras.get("cht_model_error_c", 0.0)

        # --- lubrication -----------------------------------------------------
        ind["oil_press_kpa"] = g("oil_press_kpa")
        ind["oil_temp_c"] = g("oil_temp_c")
        ind["oil_press_index"] = extras.get("oil_press_index", 1.0)
        ind["oil_dp_per_krpm"] = (g("oil_press_kpa") / max(rpm / 1000.0, 0.2)) if rpm > 500 else 0.0

        # --- mechanical -------------------------------------------------------
        ind["vib_rms_g"] = g("vib_rms_g")
        ind["vib_1x_g"] = extras.get("vib_1x_g", 0.0)
        ind["vib_broadband_g"] = extras.get("vib_broadband_g", 0.0)
        ind["friction_index"] = extras.get("friction_index", 1.0)

        # --- electrical --------------------------------------------------------
        ind["bus_v"] = g("bus_v", 13.8)
        ind["alt_current_a"] = g("alt_current_a")
        ind["alt_index"] = extras.get("alt_index", 1.0)

        # --- sensing ------------------------------------------------------------
        ind["max_resid_sigma"] = max((abs(v) for v in resid.values()), default=0.0)
        ind["resid_rms"] = (math.sqrt(sum(v * v for v in resid.values()) / len(resid))
                            if resid else 0.0)
        ind["dropout_count"] = float(sum(1 for v in meas.values() if v is None))

        for k, v in ind.items():
            if k not in self._smooth:
                self._smooth[k] = EWMA(0.05, v)
            ind[k] = self._smooth[k].update(v)
        return ind

    # -- subsystem health indices --------------------------------------------
    def update(self, ind: Dict[str, float], anomaly: dict, running: bool) -> Dict[str, float]:
        if not running:
            return self.indices
        self._samples += 1
        L = self.limits
        raw: Dict[str, float] = {}

        # combustion: EGT balance, misfire energy, knock margin, mean EGT margin
        raw["combustion"] = _clip(
            0.35 * _margin_index(ind.get("egt_max_dev_c", 0.0), 22.0, 95.0)
            + 0.30 * _margin_index(ind.get("misfire_order_g", 0.0), 0.12, 0.95)
            + 0.20 * _margin_index(ind.get("knock_index", 0.0), 0.06, 0.55)
            + 0.15 * _margin_index(ind.get("egt_mean_c", 0.0), L.egt_caution_c, L.egt_c))

        # air path: breathing efficiency, boost tracking, compressor efficiency
        raw["air_path"] = _clip(
            0.45 * _margin_index(ind.get("ve_index", 1.0), 0.90, 0.72, higher_is_worse=False)
            + 0.30 * _margin_index(abs(ind.get("boost_error_kpa", 0.0)), 6.0, 25.0)
            + 0.25 * _margin_index(ind.get("comp_eff_index", 1.0), 0.92, 0.70, higher_is_worse=False))

        # fuel injection: delivered flow vs model, cylinder-to-cylinder balance
        raw["fuel_injection"] = _clip(
            0.45 * _margin_index(abs(1.0 - ind.get("fuel_flow_index", 1.0)), 0.06, 0.25)
            + 0.40 * _margin_index(ind.get("injector_balance", 0.0), 0.035, 0.20)
            + 0.15 * _margin_index(abs(ind.get("lambda_est", 1.0) - 0.95), 0.12, 0.45))

        # cooling: temperature margins plus loss of heat-rejection capability
        raw["cooling"] = _clip(
            0.40 * _margin_index(ind.get("cht_max_c", 0.0), L.cht_caution_c, L.cht_c)
            + 0.25 * _margin_index(ind.get("coolant_temp_c", 0.0), L.coolant_temp_c - 15, L.coolant_temp_c)
            + 0.25 * _margin_index(ind.get("cooling_effectiveness", 1.0), 0.88, 0.50, higher_is_worse=False)
            + 0.10 * _margin_index(abs(ind.get("cht_model_error_c", 0.0)), 4.0, 25.0))

        # lubrication: pressure delivery vs model, oil temperature margin
        raw["lubrication"] = _clip(
            0.50 * _margin_index(ind.get("oil_press_index", 1.0), 0.90, 0.60, higher_is_worse=False)
            + 0.30 * _margin_index(ind.get("oil_temp_c", 0.0), L.oil_temp_caution_c, L.oil_temp_c)
            + 0.20 * _margin_index(ind.get("oil_press_kpa", 500.0),
                                   L.oil_press_kpa_min + 60, L.oil_press_kpa_min,
                                   higher_is_worse=False))

        # mechanical: order-resolved vibration and parasitic loss
        raw["mechanical"] = _clip(
            0.40 * _margin_index(ind.get("vib_1x_g", 0.0), 0.55, 1.9)
            + 0.25 * _margin_index(ind.get("vib_rms_g", 0.0), L.vib_caution_g, L.vib_rms_g)
            + 0.20 * _margin_index(ind.get("vib_broadband_g", 0.0), 0.75, 2.2)
            + 0.15 * _margin_index(ind.get("friction_index", 1.0), 1.12, 1.45))

        # electrical: bus regulation and alternator delivery
        v = ind.get("bus_v", 13.8)
        v_pen = max(0.0, L.bus_voltage_min - v) + max(0.0, v - L.bus_voltage_max)
        raw["electrical"] = _clip(
            0.55 * _margin_index(v_pen, 0.05, 1.4)
            + 0.45 * _margin_index(ind.get("alt_index", 1.0), 0.92, 0.45, higher_is_worse=False))

        # sensing: residual behaviour and frame loss
        raw["sensing"] = _clip(
            0.55 * _margin_index(ind.get("max_resid_sigma", 0.0), 3.2, 9.0)
            + 0.25 * _margin_index(ind.get("resid_rms", 0.0), 1.3, 4.0)
            + 0.20 * _margin_index(ind.get("dropout_count", 0.0), 0.5, 5.0))

        # The fused anomaly score penalises the subsystems its own contributors
        # point at, in proportion to their share. Something can be wrong even
        # when no single indicator has moved far enough on its own, but that
        # evidence should land where it belongs.
        excess = max(0.0, anomaly.get("score", 0.0) - 1.2)
        budget = min(45.0, 15.0 * excess)
        share: Dict[str, float] = {k: 0.0 for k in SUBSYSTEMS}
        total = 0.0
        for c in anomaly.get("contributors", []):
            sub = FEATURE_SUBSYSTEM.get(c.get("feature", ""))
            if sub:
                share[sub] += max(0.0, c.get("share", 0.0))
                total += max(0.0, c.get("share", 0.0))
        for s in SUBSYSTEMS:
            frac = (share[s] / total) if total > 1e-9 else (1.0 / len(SUBSYSTEMS))
            # A concentrated anomaly takes most of the budget; a diffuse one is
            # spread thin, which is the correct signal in both cases.
            cap = 100.0 - budget * min(1.0, frac * 2.5)
            self.indices[s] = round(self._idx_smooth[s].update(min(raw[s], cap)), 1)
            self.confidence[s] = round(min(1.0, self._samples / 600.0), 2)
        return self.indices

    @property
    def overall(self) -> float:
        """Overall index is weighted toward the *worst* subsystem: a plain
        average would hide a single failing subsystem behind seven healthy ones."""
        vals = list(self.indices.values())
        return round(0.7 * min(vals) + 0.3 * (sum(vals) / len(vals)), 1)

    def report(self) -> dict:
        return {"overall": self.overall,
                "subsystems": dict(self.indices),
                "confidence": dict(self.confidence)}
