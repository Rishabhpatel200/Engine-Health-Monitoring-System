"""
Cumulative degradation model.

Wear is driven by *severity*, not by calendar time: an hour at 135 C CHT and
95 % load consumes far more life than an hour of loitering at FL200. Each
mechanism has its own severity exponent, taken from the usual accelerated-life
forms (Arrhenius for thermal/oxidative processes, Archard-like load-speed
products for mechanical wear).

The state carried here *is* the `HealthParams` object handed to the engine
model, so the degraded twin and the healthy twin run identical equations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List

from .engine_model import HealthParams, Outputs


ARRHENIUS_EA_R = 6800.0     # K, effective activation temperature for oil/seals


def arrhenius(temp_c: float, ref_c: float = 100.0) -> float:
    """Acceleration factor relative to a reference temperature."""
    t, tr = temp_c + 273.15, ref_c + 273.15
    return math.exp(ARRHENIUS_EA_R * (1.0 / tr - 1.0 / t))


@dataclass
class WearRates:
    """Nominal loss per flight hour at unit severity.

    Calibrated so that an engine reaching a 1200 h TBO on a representative MALE
    duty cycle has lost roughly 6-8 % of rated power, 15-20 % of cooling
    capacity and 0.25 of bearing clearance - consistent with published overhaul
    findings for this engine class. Consumables (spark plugs, oil) degrade on
    their own service intervals rather than on TBO.
    """
    ring_seal: float = 3.2e-5
    valve_seal: float = 4.3e-5
    injector_fouling: float = 5.6e-5
    plug_erosion: float = 7.0e-4     # plugs are a ~200 h consumable
    cooling_fouling: float = 1.7e-4
    oil_cooler_fouling: float = 1.25e-4
    oil_pump: float = 7.0e-5
    bearing_growth: float = 1.6e-4
    oil_shear: float = 2.5e-3        # oil charge is a ~100 h consumable
    turbo: float = 1.0e-4
    alternator: float = 1.25e-4
    battery: float = 2.1e-4


@dataclass
class DegradationModel:
    rates: WearRates = field(default_factory=WearRates)
    accel: float = 1.0               # demo time-compression factor
    oil_hours: float = 0.0           # hours since last oil change
    total_hours: float = 0.0
    cold_starts: int = 0
    knock_hours: float = 0.0
    overtemp_hours: float = 0.0
    _severity_ewma: Dict[str, float] = field(default_factory=dict)

    # -- severity factors ---------------------------------------------------
    def _severity(self, o: Outputs, rated_rpm: float) -> Dict[str, float]:
        cht = sum(o.cht_c) / max(len(o.cht_c), 1)
        egt = sum(o.egt_c) / max(len(o.egt_c), 1)
        speed = o.rpm / max(rated_rpm, 1.0)
        load = max(0.0, min(1.4, o.bmep_kpa / 1000.0))
        s = {
            # ring/bore: load x speed, plus a large penalty for detonation
            "ring": (0.25 + load ** 1.6) * (0.3 + speed ** 1.3) * (1.0 + 3.5 * o.knock_index)
                    * arrhenius(cht, 110.0) ** 0.5,
            # valve seats: exhaust temperature dominated
            "valve": arrhenius(egt, 780.0) ** 0.6 * (0.3 + speed),
            # injector tip coking: hot, lean, low-flow conditions form deposits
            "inj": arrhenius(cht, 110.0) ** 0.4 * (1.0 + 1.2 * max(0.0, o.lambda_avg - 0.95))
                   * (0.4 + 0.6 * load),
            # spark plug erosion: energy per firing event x firing rate
            "plug": (0.3 + speed) * (0.5 + load) * (1.0 + 2.0 * o.knock_index),
            # radiator/oil-cooler fouling: ambient, essentially time-based
            "cool": 1.0,
            # pump and bearing wear: speed x load, hurt badly by thin hot oil
            "pump": (0.2 + speed ** 1.5) * (1.0 + 1.5 * max(0.0, (o.oil_temp_c - 110.0) / 20.0)),
            "bearing": (0.2 + load ** 1.4) * (0.3 + speed ** 1.2)
                       * (1.0 + 2.2 * max(0.0, (110.0 - o.oil_press_kpa) / 60.0))
                       * arrhenius(o.oil_temp_c, 100.0) ** 0.5,
            # oil oxidation and shear
            "oil": arrhenius(o.oil_temp_c, 100.0) * (0.4 + 0.6 * speed),
            # turbo: hot-side thermal cycling
            "turbo": arrhenius(egt, 800.0) ** 0.7 * (0.2 + 0.8 * max(0.0, o.boost_pr - 1.05)),
            "alt": (0.3 + speed) * (0.4 + 0.6 * abs(o.alt_current_a) / 18.0),
            "batt": 1.0 + 2.0 * max(0.0, 0.4 - o.batt_soc),
        }
        return s

    # -- integration --------------------------------------------------------
    def advance(self, health: HealthParams, o: Outputs, dt_s: float,
                rated_rpm: float = 5800.0) -> None:
        """Advance every wear state by `dt_s` seconds of operation."""
        if o.rpm < 200.0:
            return
        h_elapsed = dt_s / 3600.0 * self.accel
        self.total_hours += h_elapsed
        self.oil_hours += h_elapsed
        if o.knock_index > 0.25:
            self.knock_hours += h_elapsed
        if max(o.cht_c) > 130.0 or o.oil_temp_c > 125.0:
            self.overtemp_hours += h_elapsed

        s = self._severity(o, rated_rpm)
        r = self.rates
        n = len(health.injector_flow)

        health.ring_seal = max(0.30, health.ring_seal - r.ring_seal * s["ring"] * h_elapsed)
        health.valve_seal = max(0.30, health.valve_seal - r.valve_seal * s["valve"] * h_elapsed)
        health.cooling_ua = max(0.35, health.cooling_ua - r.cooling_fouling * s["cool"] * h_elapsed)
        health.oil_cooler_ua = max(0.35, health.oil_cooler_ua - r.oil_cooler_fouling * s["cool"] * h_elapsed)
        health.oil_pump = max(0.30, health.oil_pump - r.oil_pump * s["pump"] * h_elapsed)
        health.bearing_clearance = min(2.5, health.bearing_clearance
                                       + r.bearing_growth * s["bearing"] * h_elapsed)
        health.turbo_efficiency = max(0.40, health.turbo_efficiency - r.turbo * s["turbo"] * h_elapsed)
        health.alternator = max(0.0, health.alternator - r.alternator * s["alt"] * h_elapsed)
        health.battery_soh = max(0.20, health.battery_soh - r.battery * s["batt"] * h_elapsed)

        # Oil viscosity falls with shear and oxidation; it is restored, along
        # with the oil-hours counter, by a service action.
        health.oil_viscosity = max(0.45, health.oil_viscosity
                                   - r.oil_shear * s["oil"] * h_elapsed)

        # Per-cylinder items degrade at slightly different rates - the spread is
        # exactly what the per-cylinder residual bank keys on.
        for i in range(n):
            bias = 1.0 + 0.12 * math.sin(1.7 * i + 0.4)
            health.injector_flow[i] = max(0.35, health.injector_flow[i]
                                          - r.injector_fouling * s["inj"] * bias * h_elapsed)
            health.ignition_energy[i] = max(0.20, health.ignition_energy[i]
                                            - r.plug_erosion * s["plug"] * bias * h_elapsed)

    # -- maintenance actions -------------------------------------------------
    def service(self, health: HealthParams, action: str) -> str:
        """Apply a maintenance action and report what it restored."""
        a = action.lower()
        if a in ("oil_change", "oil"):
            health.oil_viscosity = 1.0
            self.oil_hours = 0.0
            return "Oil and filter replaced; viscosity restored."
        if a in ("spark_plugs", "plugs"):
            health.ignition_energy = [1.0] * len(health.ignition_energy)
            return "Spark plugs replaced."
        if a in ("injectors", "injector_clean"):
            health.injector_flow = [1.0] * len(health.injector_flow)
            return "Injectors ultrasonically cleaned and flow-matched."
        if a in ("cooling_clean", "radiator"):
            health.cooling_ua = 1.0
            health.oil_cooler_ua = 1.0
            return "Radiator and oil cooler cleaned."
        if a in ("top_overhaul", "top_end"):
            health.ring_seal = 1.0
            health.valve_seal = 1.0
            health.ignition_energy = [1.0] * len(health.ignition_energy)
            return "Top overhaul: rings, valves and plugs renewed."
        if a in ("overhaul", "tbo"):
            fresh = HealthParams(n_cyl=len(health.injector_flow))
            for k, v in fresh.__dict__.items():
                setattr(health, k, list(v) if isinstance(v, list) else v)
            self.total_hours = 0.0
            self.oil_hours = 0.0
            self.knock_hours = 0.0
            self.overtemp_hours = 0.0
            return "Engine overhauled to zero-time."
        return f"Unknown maintenance action: {action}"

    def snapshot(self) -> dict:
        return {"total_hours": round(self.total_hours, 2),
                "oil_hours": round(self.oil_hours, 2),
                "knock_hours": round(self.knock_hours, 3),
                "overtemp_hours": round(self.overtemp_hours, 3)}
