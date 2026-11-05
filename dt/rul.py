"""
Remaining Useful Life estimation.

Method
------
Each monitored health index is tracked by a two-state Kalman filter over
[level, slope]. This is the discrete local-linear-trend model: it is the
minimum-variance estimator of a noisy degradation trajectory, it gives the
slope (degradation *rate*) as a first-class estimate rather than a finite
difference, and - crucially for maintenance planning - it carries a covariance
that propagates into an honest confidence interval on the crossing time.

RUL is the time until the projected index reaches its action threshold. The
band is obtained by propagating the level/slope covariance forward, so a
freshly-observed engine reports a wide band rather than a falsely precise one.

Two guards keep the output trustworthy:
  * a slope must be statistically significant (|slope| > 2 sigma) before any
    finite RUL is published; otherwise the answer is "no significant trend";
  * RUL is capped at the remaining time to the scheduled overhaul, because no
    trend extrapolation should ever recommend flying past TBO.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .config import SUBSYSTEMS


@dataclass
class TrendState:
    level: float = 100.0
    slope: float = 0.0          # index points per flight hour
    P: List[List[float]] = field(default_factory=lambda: [[25.0, 0.0], [0.0, 4.0]])
    n: int = 0


class TrendKalman:
    """Local linear trend filter on a health index versus flight hours."""

    def __init__(self, q_level: float = 1e-4, q_slope: float = 4e-5, r: float = 4.0):
        self.q_level, self.q_slope, self.r = q_level, q_slope, r
        self.s = TrendState()

    def update(self, y: float, dt_hours: float) -> TrendState:
        s = self.s
        dt = max(dt_hours, 1e-9)
        # predict:  level += slope*dt ; slope constant
        level = s.level + s.slope * dt
        slope = s.slope
        P = s.P
        p00 = P[0][0] + dt * (P[1][0] + P[0][1]) + dt * dt * P[1][1] + self.q_level * dt
        p01 = P[0][1] + dt * P[1][1]
        p10 = P[1][0] + dt * P[1][1]
        p11 = P[1][1] + self.q_slope * dt
        # update with H = [1, 0]
        sK = p00 + self.r
        k0, k1 = p00 / sK, p10 / sK
        innov = y - level
        s.level = level + k0 * innov
        s.slope = slope + k1 * innov
        s.P = [[(1 - k0) * p00, (1 - k0) * p01],
               [p10 - k1 * p00, p11 - k1 * p01]]
        s.n += 1
        return s

    @property
    def slope_sigma(self) -> float:
        return math.sqrt(max(self.s.P[1][1], 0.0))

    @property
    def level_sigma(self) -> float:
        return math.sqrt(max(self.s.P[0][0], 0.0))


@dataclass
class RULEstimate:
    subsystem: str
    rul_hours: Optional[float]
    lower_hours: Optional[float]
    upper_hours: Optional[float]
    level: float
    slope_per_hour: float
    threshold: float
    confidence: float
    basis: str

    def to_dict(self) -> dict:
        r = lambda v: (None if v is None else round(v, 1))
        return {"subsystem": self.subsystem, "rul_hours": r(self.rul_hours),
                "lower_hours": r(self.lower_hours), "upper_hours": r(self.upper_hours),
                "level": round(self.level, 1),
                "slope_per_hour": round(self.slope_per_hour, 4),
                "threshold": self.threshold,
                "confidence": round(self.confidence, 2), "basis": self.basis}


class RULEstimator:
    """Per-subsystem RUL plus a rolled-up engine-level figure."""

    # Health index at which the subsystem must be actioned before further flight.
    THRESHOLDS: Dict[str, float] = {
        "combustion": 55.0, "air_path": 55.0, "fuel_injection": 55.0,
        "cooling": 60.0, "lubrication": 65.0, "mechanical": 65.0,
        "electrical": 55.0, "sensing": 50.0,
    }

    def __init__(self, tbo_hours: float = 1200.0, z: float = 1.645,
                 trend_period_h: float = 1.0 / 60.0, warmup_h: float = 5.0 / 60.0):
        self.tbo_hours = tbo_hours
        self.z = z                        # 90 % two-sided band
        self.trend_period_h = trend_period_h
        self.warmup_h = warmup_h
        self.filters: Dict[str, TrendKalman] = {s: TrendKalman() for s in SUBSYSTEMS}
        self.last: Dict[str, RULEstimate] = {}
        self.hours = 0.0
        self._buf: Dict[str, List[float]] = {}
        self._accum_h = 0.0
        self._valid_h = 0.0

    def update(self, indices: Dict[str, float], dt_hours: float,
               engine_hours: float) -> Dict[str, RULEstimate]:
        """Accumulate health indices and step the trend filters once per
        `trend_period_h`.

        Trending is deliberately decoupled from the 20 Hz synchronisation rate.
        Degradation is an hours-scale process; driving an hours-scale trend
        filter at sample rate makes it fit warm-up transients and throttle
        movements, which is how predictive systems end up crying wolf. Each
        trend point is the *median* of its interval, so a single spike cannot
        move the slope.
        """
        self.hours = engine_hours
        for sub, idx in indices.items():
            self._buf.setdefault(sub, []).append(idx)
        self._accum_h += dt_hours
        if dt_hours > 0:
            self._valid_h += dt_hours

        if self._accum_h >= self.trend_period_h and self._valid_h > self.warmup_h:
            for sub, vals in self._buf.items():
                if not vals:
                    continue
                v = sorted(vals)
                med = v[len(v) // 2]
                f = self.filters.setdefault(sub, TrendKalman())
                st = f.update(med, self._accum_h)
                self.last[sub] = self._project(sub, f, st, self.THRESHOLDS.get(sub, 60.0))
            self._buf = {}
            self._accum_h = 0.0
        elif self._accum_h >= self.trend_period_h:
            self._buf = {}
            self._accum_h = 0.0

        if not self.last:
            for sub, idx in indices.items():
                self.last[sub] = RULEstimate(sub, None, None, None, idx, 0.0,
                                             self.THRESHOLDS.get(sub, 60.0), 0.0,
                                             "warm-up: insufficient observation time")
        return self.last

    def _project(self, sub: str, f: TrendKalman, st: TrendState,
                 thr: float) -> RULEstimate:
        slope, sig = st.slope, f.slope_sigma
        margin = st.level - thr
        # Two hours of one-minute trend points before the estimate is
        # considered fully converged.
        conf = min(1.0, st.n / 120.0)
        tbo_left = max(0.0, self.tbo_hours - self.hours)

        if margin <= 0.0:
            return RULEstimate(sub, 0.0, 0.0, 0.0, st.level, slope, thr,
                               conf, "threshold already breached")
        if slope >= -1e-6 or abs(slope) < 2.0 * sig:
            return RULEstimate(sub, None, None, None, st.level, slope, thr, conf,
                               "no statistically significant degradation trend")

        rul = margin / (-slope)
        # Propagate uncertainty in both level and slope to the crossing time.
        s_lo = min(-1e-9, slope + self.z * sig)     # slower decay -> longer life
        s_hi = slope - self.z * sig                 # faster decay -> shorter life
        m_lo = max(1e-6, margin - self.z * f.level_sigma)
        m_hi = margin + self.z * f.level_sigma
        upper = m_hi / (-s_lo)
        lower = m_lo / (-s_hi)
        rul = min(rul, tbo_left)
        upper = min(upper, tbo_left)
        lower = min(lower, upper)
        return RULEstimate(sub, rul, lower, upper, st.level, slope, thr, conf,
                           "local-linear-trend extrapolation of the health index")

    # -- roll-up ----------------------------------------------------------------
    def engine_rul(self) -> dict:
        finite = [(s, e) for s, e in self.last.items() if e.rul_hours is not None]
        if not finite:
            return {"rul_hours": None, "driver": None,
                    "basis": "no subsystem shows a significant degradation trend",
                    "tbo_remaining_hours": round(max(0.0, self.tbo_hours - self.hours), 1)}
        sub, e = min(finite, key=lambda t: t[1].rul_hours)
        return {"rul_hours": round(e.rul_hours, 1),
                "lower_hours": round(e.lower_hours, 1) if e.lower_hours is not None else None,
                "upper_hours": round(e.upper_hours, 1) if e.upper_hours is not None else None,
                "driver": sub, "confidence": round(e.confidence, 2),
                "basis": f"limited by {sub} (index {e.level:.1f}, "
                         f"{e.slope_per_hour:+.4f} pts/h, threshold {e.threshold:.0f})",
                "tbo_remaining_hours": round(max(0.0, self.tbo_hours - self.hours), 1)}

    def report(self) -> dict:
        return {"engine": self.engine_rul(),
                "subsystems": {s: e.to_dict() for s, e in self.last.items()}}
