"""
Mission profiles and environment scheduling.

A MALE sortie is 20-30 h long and spends most of it in a narrow loiter regime,
punctuated by the few conditions that actually consume engine life: the full-
power climb, hot-and-high departures, and throttle transients during handovers
and weather avoidance. The profiles below reproduce those phases so the twin
can be exercised - and replayed - against representative duty.

Each profile is a piecewise schedule; `sample(t)` interpolates it into the
environment and command vector the plant and the twin both consume.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .atmosphere import AtmState, isa
from .engine_model import Command


@dataclass
class Waypoint:
    t_s: float
    altitude_m: float
    tas_ms: float
    throttle_pct: float
    rpm_target: float
    phase: str = "cruise"
    elec_load_w: float = 140.0


@dataclass
class MissionProfile:
    name: str
    description: str
    waypoints: List[Waypoint]
    isa_dev_k: float = 0.0
    humidity: float = 0.30
    turbulence: float = 0.0          # RMS throttle/airspeed disturbance
    time_compression: float = 1.0    # wall-clock seconds per mission second

    @property
    def duration_s(self) -> float:
        return self.waypoints[-1].t_s if self.waypoints else 0.0

    def sample(self, t: float) -> Tuple[AtmState, Command, float, str]:
        wps = self.waypoints
        t = max(0.0, min(t, self.duration_s))
        i = 0
        while i < len(wps) - 1 and wps[i + 1].t_s <= t:
            i += 1
        a = wps[i]
        b = wps[min(i + 1, len(wps) - 1)]
        span = max(b.t_s - a.t_s, 1e-6)
        f = max(0.0, min(1.0, (t - a.t_s) / span))
        lerp = lambda x, y: x + (y - x) * f

        alt = lerp(a.altitude_m, b.altitude_m)
        tas = lerp(a.tas_ms, b.tas_ms)
        thr = lerp(a.throttle_pct, b.throttle_pct)
        rpm = lerp(a.rpm_target, b.rpm_target)
        load = lerp(a.elec_load_w, b.elec_load_w)

        if self.turbulence > 0.0:
            # deterministic pseudo-turbulence: sum of incommensurate sinusoids
            d = (math.sin(0.37 * t) + 0.6 * math.sin(1.13 * t + 1.1)
                 + 0.35 * math.sin(2.71 * t + 0.3))
            thr = max(0.0, min(100.0, thr + self.turbulence * 6.0 * d))
            tas = max(5.0, tas + self.turbulence * 3.0 * d)

        # ISA deviation eases off with altitude, as real hot days do
        dev = self.isa_dev_k * max(0.25, 1.0 - alt / 12000.0)
        atm = isa(alt, dev, self.humidity)
        cmd = Command(throttle_pct=thr, rpm_target=rpm, elec_load_w=load)
        return atm, cmd, tas, a.phase


# --------------------------------------------------------------------------
def isr_sortie(loiter_hours: float = 4.0) -> MissionProfile:
    """Representative ISR sortie: taxi, climb to FL200, transit, loiter, recover."""
    L = loiter_hours * 3600.0
    wps = [
        Waypoint(0,      0,    0,  8,  1400, "ground_idle", 90),
        Waypoint(120,    0,    0, 35,  2600, "runup",       110),
        Waypoint(180,    0,   28,100,  5800, "takeoff",     150),
        Waypoint(300,   400,  38,100,  5800, "climb",       150),
        Waypoint(1500, 4000,  48, 98,  5600, "climb",       150),
        Waypoint(2700, 6100,  52, 96,  5500, "climb",       160),
        Waypoint(3300, 6100,  55, 72,  5100, "transit",     160),
        Waypoint(3300 + L,      6100, 52, 62, 4900, "loiter", 210),
        Waypoint(3300 + L + 900, 3000, 58, 40, 4600, "descent", 180),
        Waypoint(3300 + L + 1500,  600, 45, 45, 4800, "approach", 160),
        Waypoint(3300 + L + 1700,    0, 20, 12, 1600, "landing", 120),
        Waypoint(3300 + L + 1900,    0,  0,  8, 1400, "shutdown", 90),
    ]
    return MissionProfile("isr_sortie",
                          f"{loiter_hours:.0f} h ISR sortie: climb to 6100 m, loiter, recover",
                          wps, isa_dev_k=5.0, turbulence=0.15)


def high_altitude() -> MissionProfile:
    """Sustained operation above the turbocharger critical altitude."""
    wps = [
        Waypoint(0,       0,  28,100, 5800, "takeoff", 150),
        Waypoint(900,  4900,  50,100, 5700, "climb",   160),
        Waypoint(2400, 7600,  55,100, 5400, "climb",   170),
        Waypoint(3000, 7600,  55,100, 5300, "high_cruise", 200),
        Waypoint(6000, 7600,  55,100, 5300, "high_cruise", 200),
        Waypoint(7200, 7600,  55, 85, 5100, "high_loiter", 200),
    ]
    return MissionProfile("high_altitude",
                          "Climb above the 4900 m critical altitude and hold FL250",
                          wps, isa_dev_k=0.0, turbulence=0.10)


def endurance() -> MissionProfile:
    """Long loiter - the regime that dominates cumulative wear on a MALE UAV."""
    wps = [
        Waypoint(0,       0,  28,100, 5800, "takeoff", 150),
        Waypoint(1200, 5500,  50, 96, 5500, "climb",   160),
        Waypoint(1800, 5500,  52, 58, 4800, "loiter",  220),
        Waypoint(20000, 5800, 50, 56, 4750, "loiter",  220),
        Waypoint(40000, 6100, 50, 57, 4800, "loiter",  220),
        Waypoint(60000, 6100, 50, 58, 4850, "loiter",  220),
        Waypoint(62000,  500, 55, 35, 4500, "descent", 180),
        Waypoint(63000,    0,  0, 10, 1400, "shutdown", 100),
    ]
    return MissionProfile("endurance", "17.5 h endurance loiter profile",
                          wps, isa_dev_k=3.0, turbulence=0.12)


def hot_weather() -> MissionProfile:
    """ISA+35 desert operation - the cooling-system stress case."""
    wps = [
        Waypoint(0,     600,   0, 10, 1400, "ground_idle", 90),
        Waypoint(600,   600,   0, 30, 2400, "hot_soak",    110),
        Waypoint(700,   600,  28,100, 5800, "takeoff",     150),
        Waypoint(760,   700,  40,100, 5800, "climb",       150),
        Waypoint(1600, 2500,  46,100, 5700, "climb",       160),
        Waypoint(2400, 4000,  48, 92, 5500, "climb",       160),
        Waypoint(3000, 4000,  50, 70, 5000, "cruise",      190),
        Waypoint(5400, 4000,  50, 68, 4950, "loiter",      190),
    ]
    return MissionProfile("hot_weather", "ISA+35 desert departure and cruise",
                          wps, isa_dev_k=35.0, humidity=0.10, turbulence=0.18)


def throttle_transients() -> MissionProfile:
    """Test-rig style throttle slam sequence for transient model validation."""
    wps: List[Waypoint] = [Waypoint(0, 1000, 40, 30, 4200, "warmup", 140)]
    t = 300.0
    for i, (lo, hi) in enumerate([(25, 100), (100, 20), (20, 90), (90, 35),
                                  (35, 100), (100, 55), (55, 15), (15, 100)]):
        wps.append(Waypoint(t,       1000, 45, lo, 4200 + 200 * (i % 3), "steady", 140))
        wps.append(Waypoint(t + 1.0, 1000, 45, hi, 5600, "slam", 140))
        wps.append(Waypoint(t + 60,  1000, 45, hi, 5600, "settle", 140))
        t += 120.0
    wps.append(Waypoint(t, 1000, 40, 25, 4000, "cooldown", 120))
    return MissionProfile("throttle_transients",
                          "Repeated throttle slams for transient response validation",
                          wps, turbulence=0.0)


def cold_high() -> MissionProfile:
    """ISA-20 high-altitude launch: the cold-start and oil-viscosity case."""
    wps = [
        Waypoint(0,    3000,  0,  8, 1400, "cold_start", 90),
        Waypoint(900,  3000,  0, 25, 2200, "warmup",     110),
        Waypoint(1000, 3000, 28,100, 5800, "takeoff",    150),
        Waypoint(2000, 6000, 48,100, 5600, "climb",      160),
        Waypoint(3600, 6000, 52, 65, 4900, "loiter",     200),
    ]
    return MissionProfile("cold_high", "ISA-20 high-elevation cold start and climb",
                          wps, isa_dev_k=-20.0, humidity=0.05, turbulence=0.08)


def envelope_sweep(isa_dev_k: float = 0.0, dwell_s: float = 45.0) -> MissionProfile:
    """Test-rig style sweep of the operating envelope.

    Baseline identification is a rig activity, not a flight activity: holding a
    grid of altitude and power points for long enough to stabilise gives the
    learners even coverage of the envelope in a few hundred seconds. Training on
    whole sorties instead biases the baseline toward whichever phase happened to
    last longest, which is how a health model ends up flagging a legitimate
    high-altitude loiter as an anomaly.
    """
    wps = [Waypoint(0, 500, 30, 25, 3000, "warmup", 130),
           Waypoint(180, 500, 35, 55, 4400, "warmup", 140)]
    t = 200.0
    for alt in (500, 3000, 5000, 6100, 7600):
        for thr, rpm in ((45, 4300), (60, 4700), (75, 5100), (90, 5500), (100, 5800)):
            wps.append(Waypoint(t, alt, 40 + alt / 400.0, thr, rpm, "sweep", 160))
            t += dwell_s
    wps.append(Waypoint(t, 1000, 45, 35, 4000, "cooldown", 130))
    return MissionProfile("envelope_sweep",
                          f"Envelope sweep at ISA{isa_dev_k:+.0f} for baseline identification",
                          wps, isa_dev_k=isa_dev_k, turbulence=0.05)


PROFILES = {
    "envelope_sweep": envelope_sweep,
    "isr_sortie": isr_sortie,
    "high_altitude": high_altitude,
    "endurance": endurance,
    "hot_weather": hot_weather,
    "throttle_transients": throttle_transients,
    "cold_high": cold_high,
}


def get_profile(name: str, **kw) -> MissionProfile:
    if name not in PROFILES:
        raise ValueError(f"unknown mission profile '{name}'; "
                         f"available: {', '.join(sorted(PROFILES))}")
    return PROFILES[name](**kw)
