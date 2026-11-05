"""
Fault library and injection scheduler.

Faults are declarative: a catalogue entry names the affected physical
parameter, the shape of its progression (step / ramp / exponential /
intermittent), and the ground-truth label used to score the detector. This
keeps the fault simulator honest - nothing here tells the detector what is
wrong, it only perturbs the plant.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .engine_model import HealthParams


# --------------------------------------------------------------------------
FAULT_CATALOGUE: Dict[str, dict] = {
    "misfire":            {"subsystem": "combustion",      "target": "ignition_energy",
                           "per_cylinder": True,  "severity_desc": "fraction of spark energy lost"},
    "injector_clog":      {"subsystem": "fuel_injection",  "target": "injector_flow",
                           "per_cylinder": True,  "severity_desc": "fraction of rated flow lost"},
    "injector_leak":      {"subsystem": "fuel_injection",  "target": "injector_flow",
                           "per_cylinder": True,  "severity_desc": "fraction of over-fuelling"},
    "cooling_degradation":{"subsystem": "cooling",         "target": "cooling_ua",
                           "per_cylinder": False, "severity_desc": "fraction of heat-rejection capacity lost"},
    "coolant_loss":       {"subsystem": "cooling",         "target": "cooling_ua",
                           "per_cylinder": False, "severity_desc": "fraction of coolant lost"},
    "oil_cooler_block":   {"subsystem": "cooling",         "target": "oil_cooler_ua",
                           "per_cylinder": False, "severity_desc": "fraction of oil-cooler face blocked"},
    "oil_pump_wear":      {"subsystem": "lubrication",     "target": "oil_pump",
                           "per_cylinder": False, "severity_desc": "fraction of pump delivery lost"},
    "oil_dilution":       {"subsystem": "lubrication",     "target": "oil_viscosity",
                           "per_cylinder": False, "severity_desc": "fraction of viscosity lost"},
    "bearing_wear":       {"subsystem": "mechanical",      "target": "bearing_clearance",
                           "per_cylinder": False, "severity_desc": "clearance growth above nominal"},
    "mount_degradation":  {"subsystem": "mechanical",      "target": "mount_stiffness",
                           "per_cylinder": False, "severity_desc": "fraction of mount stiffness lost"},
    "turbo_degradation":  {"subsystem": "air_path",        "target": "turbo_efficiency",
                           "per_cylinder": False, "severity_desc": "fraction of compressor efficiency lost"},
    "wastegate_stuck":    {"subsystem": "air_path",        "target": "wastegate_authority",
                           "per_cylinder": False, "severity_desc": "fraction of boost authority lost"},
    "ring_wear":          {"subsystem": "mechanical",      "target": "ring_seal",
                           "per_cylinder": False, "severity_desc": "fraction of ring seal lost"},
    "valve_recession":    {"subsystem": "mechanical",      "target": "valve_seal",
                           "per_cylinder": False, "severity_desc": "fraction of valve seal lost"},
    "bad_fuel":           {"subsystem": "combustion",      "target": "fuel_octane",
                           "per_cylinder": False, "severity_desc": "octane numbers below spec"},
    "alternator_fault":   {"subsystem": "electrical",      "target": "alternator",
                           "per_cylinder": False, "severity_desc": "fraction of output lost"},
    "battery_ageing":     {"subsystem": "electrical",      "target": "battery_soh",
                           "per_cylinder": False, "severity_desc": "fraction of capacity lost"},
    # Sensor faults act on the measurement chain, not the plant.
    "sensor_drift":       {"subsystem": "sensing",         "target": "__sensor__",
                           "per_cylinder": False, "severity_desc": "drift in engineering units per hour"},
    "sensor_bias":        {"subsystem": "sensing",         "target": "__sensor__",
                           "per_cylinder": False, "severity_desc": "fixed offset in engineering units"},
    "sensor_stuck":       {"subsystem": "sensing",         "target": "__sensor__",
                           "per_cylinder": False, "severity_desc": "channel frozen at last value"},
    "sensor_dropout":     {"subsystem": "sensing",         "target": "__sensor__",
                           "per_cylinder": False, "severity_desc": "fraction of frames lost"},
}

PROFILES = ("step", "ramp", "exponential", "intermittent")


@dataclass
class FaultEvent:
    kind: str
    onset_s: float                    # mission time at which the fault starts
    severity: float                   # final severity, units per the catalogue
    profile: str = "ramp"
    duration_s: float = 300.0         # time to reach full severity
    cylinder: Optional[int] = None    # 1-based, for per-cylinder faults
    channel: Optional[str] = None     # for sensor faults
    duty: float = 0.15                # intermittent: fraction of time active
    period_s: float = 20.0            # intermittent: cycle period
    note: str = ""

    def progress(self, t: float, rng: random.Random) -> float:
        """Severity actually applied at mission time `t`."""
        if t < self.onset_s:
            return 0.0
        x = t - self.onset_s
        if self.profile == "step":
            f = 1.0
        elif self.profile == "ramp":
            f = min(1.0, x / max(self.duration_s, 1e-6))
        elif self.profile == "exponential":
            f = 1.0 - math.exp(-3.0 * x / max(self.duration_s, 1e-6))
        elif self.profile == "intermittent":
            phase = (x % max(self.period_s, 1e-6)) / max(self.period_s, 1e-6)
            f = 1.0 if phase < self.duty else 0.0
        else:
            f = 1.0
        return self.severity * f

    @property
    def subsystem(self) -> str:
        return FAULT_CATALOGUE.get(self.kind, {}).get("subsystem", "unknown")

    def label(self) -> str:
        if self.cylinder:
            return f"{self.kind}#cyl{self.cylinder}"
        if self.channel:
            return f"{self.kind}#{self.channel}"
        return self.kind


class FaultInjector:
    """Applies scheduled faults on top of the slowly degrading health state.

    `apply()` is called once per simulation step with a *pristine copy* of the
    degradation health vector, so faults compose with wear without accumulating
    twice.
    """

    def __init__(self, events: Optional[List[FaultEvent]] = None, seed: int = 7):
        self.events: List[FaultEvent] = list(events or [])
        self.rng = random.Random(seed)
        self.sensor_faults: Dict[str, dict] = {}
        self.active: List[str] = []

    def add(self, event: FaultEvent) -> None:
        self.events.append(event)

    def clear(self) -> None:
        self.events.clear()
        self.sensor_faults.clear()
        self.active.clear()

    def apply(self, health: HealthParams, t: float) -> HealthParams:
        self.active = []
        self.sensor_faults = {}
        for ev in self.events:
            sev = ev.progress(t, self.rng)
            if sev <= 0.0:
                continue
            self.active.append(ev.label())
            meta = FAULT_CATALOGUE.get(ev.kind)
            if meta is None:
                continue
            tgt = meta["target"]
            if tgt == "__sensor__":
                if ev.channel:
                    self.sensor_faults[ev.channel] = {"kind": ev.kind, "severity": sev}
                continue
            if meta["per_cylinder"]:
                idx = (ev.cylinder or 1) - 1
                arr = getattr(health, tgt)
                if 0 <= idx < len(arr):
                    if ev.kind == "injector_leak":
                        arr[idx] = min(2.5, arr[idx] * (1.0 + sev))
                    else:
                        arr[idx] = max(0.0, arr[idx] * (1.0 - sev))
            else:
                cur = getattr(health, tgt)
                if tgt == "bearing_clearance":
                    setattr(health, tgt, min(3.0, cur + sev))
                elif tgt == "fuel_octane":
                    setattr(health, tgt, max(70.0, cur - sev))
                else:
                    setattr(health, tgt, max(0.0, cur * (1.0 - sev)))
        return health

    # -- convenience constructors used by the demo scenarios ----------------
    @staticmethod
    def scenario(name: str, n_cyl: int = 4) -> List[FaultEvent]:
        """Named demonstration scenarios with realistic onsets and rates."""
        s = name.lower()
        if s == "healthy":
            return []
        if s == "progressive_misfire":
            return [FaultEvent("misfire", 900.0, 0.85, "exponential", 1500.0, cylinder=3,
                               note="Cyl 3 spark plug erosion progressing to hard misfire")]
        if s == "injector_fouling":
            return [FaultEvent("injector_clog", 600.0, 0.32, "ramp", 2400.0, cylinder=2,
                               note="Cyl 2 injector tip coking, 32 % flow loss")]
        if s == "cooling_loss":
            return [FaultEvent("cooling_degradation", 1200.0, 0.55, "ramp", 1800.0,
                               note="Radiator core progressively blocked by debris"),
                    FaultEvent("oil_cooler_block", 1500.0, 0.30, "ramp", 1800.0,
                               note="Oil cooler face partially obstructed")]
        if s == "lubrication_failure":
            return [FaultEvent("oil_pump_wear", 800.0, 0.45, "exponential", 2000.0,
                               note="Oil pump gear wear, delivery falling"),
                    FaultEvent("bearing_wear", 1600.0, 0.55, "ramp", 2400.0,
                               note="Main bearing clearance growth")]
        if s == "sensor_drift":
            return [FaultEvent("sensor_drift", 500.0, 22.0, "ramp", 2600.0, channel="cht2_c",
                               note="CHT2 thermocouple junction drifting high"),
                    FaultEvent("sensor_stuck", 2600.0, 1.0, "step", 1.0, channel="oil_press_kpa",
                               note="Oil pressure transducer frozen")]
        if s == "detonation":
            return [FaultEvent("bad_fuel", 400.0, 14.0, "step", 1.0,
                               note="Off-spec fuel, 14 octane numbers low"),
                    FaultEvent("cooling_degradation", 400.0, 0.25, "ramp", 1200.0,
                               note="Reduced cooling raising charge temperature")]
        if s == "turbo_decay":
            return [FaultEvent("turbo_degradation", 700.0, 0.35, "ramp", 3000.0,
                               note="Compressor wheel erosion and shaft-seal leakage")]
        if s == "combined":
            return (FaultInjector.scenario("progressive_misfire")
                    + FaultInjector.scenario("cooling_loss")
                    + [FaultEvent("sensor_drift", 2000.0, 14.0, "ramp", 2000.0, channel="egt1_c",
                                  note="EGT1 probe drift")])
        raise ValueError(f"unknown scenario: {name}")

    @staticmethod
    def scenario_names() -> List[str]:
        return ["healthy", "progressive_misfire", "injector_fouling", "cooling_loss",
                "lubrication_failure", "sensor_drift", "detonation", "turbo_decay", "combined"]
