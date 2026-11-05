"""
Unscented Kalman filter over the physics model.

The DT does not merely display telemetry - it runs the virtual engine forward
and fuses it with the measurements. That gives three things a threshold system
cannot provide:

  * estimates of states that are not instrumented (per-cylinder IMEP, trapped
    mass, true CHT behind a drifting thermocouple),
  * innovation (residual) sequences that are white and unit-variance on a
    healthy engine - the statistically correct basis for anomaly detection,
  * graceful handling of dropped CAN frames, since the prediction continues
    when a measurement is unavailable.

An unscented filter is used rather than an EKF because the plant contains
saturations (wastegate, relief valve, pitch stops) that are not differentiable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from . import linalg as la


@dataclass
class UKFConfig:
    alpha: float = 1e-3        # sigma-point spread
    beta: float = 2.0          # 2 is optimal for Gaussian priors
    kappa: float = 0.0


class UnscentedKalmanFilter:
    def __init__(self, n_x: int, fx: Callable[[List[float], float, object], List[float]],
                 hx: Callable[[List[float], object], List[float]],
                 q_diag: Sequence[float], cfg: Optional[UKFConfig] = None):
        self.n = n_x
        self.fx = fx
        self.hx = hx
        self.cfg = cfg or UKFConfig()
        self.x: List[float] = [0.0] * n_x
        self.P: la.Matrix = la.eye(n_x, 1.0)
        self.Q: la.Matrix = [[q_diag[i] if i == j else 0.0 for j in range(n_x)]
                             for i in range(n_x)]
        a, k = self.cfg.alpha, self.cfg.kappa
        self.lam = a * a * (n_x + k) - n_x
        c = n_x + self.lam
        self.wm = [self.lam / c] + [1.0 / (2.0 * c)] * (2 * n_x)
        self.wc = [self.lam / c + (1.0 - a * a + self.cfg.beta)] + [1.0 / (2.0 * c)] * (2 * n_x)
        self._c = c
        self.last_innovation: List[float] = []
        self.last_nis: float = 0.0
        self.last_S_inv: Optional[la.Matrix] = None

    def _sigma_points(self) -> List[List[float]]:
        l = la.cholesky(la.symmetrise(self.P))
        s = math.sqrt(self._c)
        pts = [list(self.x)]
        for i in range(self.n):
            col = [l[r][i] * s for r in range(self.n)]
            pts.append([self.x[r] + col[r] for r in range(self.n)])
            pts.append([self.x[r] - col[r] for r in range(self.n)])
        return pts

    def predict(self, dt: float, u: object) -> None:
        pts = [self.fx(p, dt, u) for p in self._sigma_points()]
        x = [sum(self.wm[i] * pts[i][j] for i in range(len(pts))) for j in range(self.n)]
        P = la.zeros(self.n, self.n)
        for i, p in enumerate(pts):
            d = [p[j] - x[j] for j in range(self.n)]
            w = self.wc[i]
            for r in range(self.n):
                for c in range(self.n):
                    P[r][c] += w * d[r] * d[c]
        self.x = x
        self.P = la.symmetrise(la.add(P, self.Q))

    def update(self, z: List[Optional[float]], r_diag: Sequence[float], u: object) -> None:
        """Fuse a measurement. `None` entries (dropped CAN signals) are skipped,
        so the filter degrades to pure prediction on those channels."""
        idx = [i for i, v in enumerate(z) if v is not None]
        if not idx:
            self.last_innovation = []
            self.last_nis = 0.0
            return
        # Sigma points are regenerated from the *post-process-noise* covariance.
        # Reusing the propagated set would omit Q from the innovation covariance,
        # collapsing the Kalman gain whenever the model is locally deterministic.
        pts = self._sigma_points()
        zs = [[self.hx(p, u)[i] for i in idx] for p in pts]
        m = len(idx)
        zhat = [sum(self.wm[k] * zs[k][j] for k in range(len(zs))) for j in range(m)]

        S = [[(r_diag[idx[i]] if i == j else 0.0) for j in range(m)] for i in range(m)]
        C = la.zeros(self.n, m)
        for k, zk in enumerate(zs):
            dz = [zk[j] - zhat[j] for j in range(m)]
            dx = [pts[k][j] - self.x[j] for j in range(self.n)]
            w = self.wc[k]
            for r in range(m):
                for c in range(m):
                    S[r][c] += w * dz[r] * dz[c]
            for r in range(self.n):
                for c in range(m):
                    C[r][c] += w * dx[r] * dz[c]

        S = la.symmetrise(S)
        S_inv = la.spd_inverse(S)
        K = la.matmul(C, S_inv)
        y = [z[idx[j]] - zhat[j] for j in range(m)]

        self.x = [self.x[i] + sum(K[i][j] * y[j] for j in range(m)) for i in range(self.n)]
        KS = la.matmul(K, S)
        self.P = la.symmetrise([[self.P[i][j] - sum(KS[i][k] * K[j][k] for k in range(m))
                                 for j in range(self.n)] for i in range(self.n)])

        # Normalised innovation: unit-variance and white on a healthy engine.
        full = [0.0] * len(z)
        for j, i in enumerate(idx):
            full[i] = y[j] / math.sqrt(max(S[j][j], 1e-12))
        self.last_innovation = full
        self.last_nis = sum(y[i] * S_inv[i][j] * y[j] for i in range(m) for j in range(m))
        self.last_S_inv = S_inv
        self._active_idx = idx

    @property
    def std(self) -> List[float]:
        return [math.sqrt(max(self.P[i][i], 0.0)) for i in range(self.n)]
