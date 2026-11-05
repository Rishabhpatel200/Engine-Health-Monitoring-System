"""
AI/ML layer: anomaly detection with physics-informed features.

Design principle
----------------
Raw engine telemetry is dominated by the operating point: CHT at 5800 rpm on a
hot day is nothing like CHT at flight idle, and a purely data-driven detector
spends all its capacity learning that. So the feature vector fed to the
learners is *already* physics-normalised - it is built mostly from model-versus-
measurement residuals and from dimensionless performance indices, which are
approximately invariant across the flight envelope. The learners then only have
to model what physics could not explain.

Three complementary detectors run in parallel and are fused:

  PCA monitor      Hotelling T^2 (variation inside the healthy subspace) and
                   SPE/Q (variation orthogonal to it). Contributions decompose
                   either statistic per feature, which is what makes the alert
                   explainable rather than a bare score.
  Autoencoder      a small bottlenecked MLP catching nonlinear couplings the
                   linear subspace misses; per-feature reconstruction error is
                   again the explanation.
  CUSUM / EWMA     sequential change detection on each residual, tuned to catch
                   slow drift that never trips an instantaneous threshold - the
                   exact failure mode of conventional limit-based monitoring.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import linalg as la


# --------------------------------------------------------------------------
# Operating-regime bucketing
# --------------------------------------------------------------------------
def regime_of(rpm: float, load: float, altitude_m: float) -> str:
    """Coarse operating regime. Baselines are kept per regime so that a
    transient throttle movement is not mistaken for a fault."""
    if rpm < 800:
        return "off"
    if rpm < 2200:
        return "idle"
    band = "low" if load < 0.45 else ("mid" if load < 0.78 else "high")
    alt = "sl" if altitude_m < 2500 else ("med" if altitude_m < 6000 else "high")
    return f"{band}_{alt}"


# --------------------------------------------------------------------------
# Online statistics
# --------------------------------------------------------------------------
class OnlineStats:
    """Welford mean/variance with an optional forgetting factor."""

    def __init__(self, n: int, forget: float = 0.0):
        self.n_feat = n
        self.count = 0
        self.mean = [0.0] * n
        self.m2 = [0.0] * n
        self.forget = forget

    def update(self, x: Sequence[float]) -> None:
        self.count += 1
        for i, v in enumerate(x):
            d = v - self.mean[i]
            self.mean[i] += d / self.count
            self.m2[i] += d * (v - self.mean[i])

    def std(self) -> List[float]:
        if self.count < 2:
            return [1.0] * self.n_feat
        return [max(math.sqrt(self.m2[i] / (self.count - 1)), 1e-6)
                for i in range(self.n_feat)]

    def normalise(self, x: Sequence[float],
                  scale: Optional[Sequence[float]] = None) -> List[float]:
        """Standardise against this baseline's mean. An external `scale` may be
        supplied when the reference mean and the reference spread should come
        from different populations."""
        s = scale if scale is not None else self.std()
        return [(x[i] - self.mean[i]) / max(s[i], 1e-9) for i in range(self.n_feat)]


class EWMA:
    def __init__(self, lam: float = 0.05, target: float = 0.0):
        self.lam = lam
        self.value = target

    def update(self, x: float) -> float:
        self.value += self.lam * (x - self.value)
        return self.value


class CUSUM:
    """Two-sided cumulative sum for a standardised sequence.

    `k` is the slack (half the smallest shift worth detecting, in sigma) and
    `h` the decision interval. With k=0.5, h=6 the average run length to a
    false alarm is on the order of 10^3 samples while a 1-sigma step is caught
    in ~10 - which is why slow sensor drift is picked up long before any
    fixed limit is reached.
    """

    def __init__(self, k: float = 0.6, h: float = 6.0):
        self.k, self.h = k, h
        self.hi = 0.0
        self.lo = 0.0
        self.alarm = 0

    def update(self, z: float) -> int:
        self.hi = max(0.0, self.hi + z - self.k)
        self.lo = max(0.0, self.lo - z - self.k)
        if self.hi > self.h:
            self.alarm = 1
        elif self.lo > self.h:
            self.alarm = -1
        else:
            self.alarm = 0
        return self.alarm

    @property
    def score(self) -> float:
        return max(self.hi, self.lo) / self.h

    def reset(self) -> None:
        self.hi = self.lo = 0.0
        self.alarm = 0


# --------------------------------------------------------------------------
# PCA-based multivariate monitor
# --------------------------------------------------------------------------
@dataclass
class PCAMonitor:
    names: List[str] = field(default_factory=list)
    mean: List[float] = field(default_factory=list)
    scale: List[float] = field(default_factory=list)
    loadings: List[List[float]] = field(default_factory=list)   # d x k
    eigenvalues: List[float] = field(default_factory=list)
    t2_limit: float = 0.0
    spe_limit: float = 0.0
    n_train: int = 0

    def fit(self, X: List[List[float]], names: List[str],
            var_explained: float = 0.92, alpha: float = 0.99) -> "PCAMonitor":
        n, d = len(X), len(X[0])
        self.names = list(names)
        self.mean = [sum(row[j] for row in X) / n for j in range(d)]
        self.scale = [max(math.sqrt(sum((row[j] - self.mean[j]) ** 2 for row in X) / max(n - 1, 1)), 1e-6)
                      for j in range(d)]
        Z = [[(row[j] - self.mean[j]) / self.scale[j] for j in range(d)] for row in X]
        C = la.zeros(d, d)
        for row in Z:
            for i in range(d):
                ri = row[i]
                for j in range(i, d):
                    C[i][j] += ri * row[j]
        for i in range(d):
            for j in range(i, d):
                C[i][j] /= max(n - 1, 1)
                C[j][i] = C[i][j]
        vals, vecs = la.jacobi_eig(C)
        total = sum(max(v, 0.0) for v in vals) or 1.0
        acc, k = 0.0, 0
        while k < d and acc / total < var_explained:
            acc += max(vals[k], 0.0)
            k += 1
        k = max(1, min(k, d - 1))
        self.eigenvalues = [max(vals[i], 1e-9) for i in range(k)]
        self.loadings = [[vecs[r][c] for c in range(k)] for r in range(d)]
        self.n_train = n

        t2s, spes = [], []
        for z in Z:
            t2, spe, _, _ = self._stats(z)
            t2s.append(t2)
            spes.append(spe)
        self.t2_limit = self._quantile(t2s, alpha)
        self.spe_limit = self._quantile(spes, alpha)
        return self

    @staticmethod
    def _quantile(v: List[float], q: float) -> float:
        s = sorted(v)
        if not s:
            return 1.0
        i = min(len(s) - 1, int(q * (len(s) - 1)))
        return max(s[i], 1e-9)

    def _stats(self, z: List[float]) -> Tuple[float, float, List[float], List[float]]:
        d, k = len(z), len(self.eigenvalues)
        t = [sum(z[r] * self.loadings[r][c] for r in range(d)) for c in range(k)]
        zhat = [sum(t[c] * self.loadings[r][c] for c in range(k)) for r in range(d)]
        e = [z[r] - zhat[r] for r in range(d)]
        t2 = sum(t[c] ** 2 / self.eigenvalues[c] for c in range(k))
        spe = sum(v * v for v in e)
        # Per-feature contributions - the explanation behind the score.
        spe_contrib = [v * v for v in e]
        t2_contrib = [abs(z[r] * sum(self.loadings[r][c] * t[c] / self.eigenvalues[c]
                                     for c in range(k))) for r in range(d)]
        return t2, spe, t2_contrib, spe_contrib

    def score(self, x: Sequence[float]) -> dict:
        z = [(x[j] - self.mean[j]) / self.scale[j] for j in range(len(x))]
        t2, spe, t2c, spec = self._stats(z)
        return {"t2": t2, "spe": spe,
                "t2_norm": t2 / max(self.t2_limit, 1e-9),
                "spe_norm": spe / max(self.spe_limit, 1e-9),
                "t2_contrib": t2c, "spe_contrib": spec}

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in
                ("names", "mean", "scale", "loadings", "eigenvalues",
                 "t2_limit", "spe_limit", "n_train")}

    @staticmethod
    def from_dict(d: dict) -> "PCAMonitor":
        m = PCAMonitor()
        for k, v in d.items():
            setattr(m, k, v)
        return m


# --------------------------------------------------------------------------
# Autoencoder (pure Python, tanh hidden units, linear output)
# --------------------------------------------------------------------------
class Autoencoder:
    def __init__(self, dims: List[int], seed: int = 5):
        self.dims = dims
        rng = random.Random(seed)
        self.W: List[List[List[float]]] = []
        self.b: List[List[float]] = []
        for i in range(len(dims) - 1):
            fan_in, fan_out = dims[i], dims[i + 1]
            lim = math.sqrt(6.0 / (fan_in + fan_out))
            self.W.append([[rng.uniform(-lim, lim) for _ in range(fan_out)]
                           for _ in range(fan_in)])
            self.b.append([0.0] * fan_out)
        self.mean: List[float] = [0.0] * dims[0]
        self.scale: List[float] = [1.0] * dims[0]
        self.err_mean: float = 0.0
        self.err_limit: float = 1.0
        self.feat_limit: List[float] = [1.0] * dims[0]

    # -- forward / backward -------------------------------------------------
    def _forward(self, x: List[float]):
        acts = [x]
        a = x
        n_layers = len(self.W)
        for li in range(n_layers):
            W, b = self.W[li], self.b[li]
            z = [b[j] + sum(a[i] * W[i][j] for i in range(len(a))) for j in range(len(b))]
            a = z if li == n_layers - 1 else [math.tanh(v) for v in z]
            acts.append(a)
        return acts

    def _train_batch(self, batch: List[List[float]], lr: float) -> float:
        n_layers = len(self.W)
        gW = [[[0.0] * len(self.W[l][0]) for _ in range(len(self.W[l]))] for l in range(n_layers)]
        gb = [[0.0] * len(self.b[l]) for l in range(n_layers)]
        loss = 0.0
        for x in batch:
            acts = self._forward(x)
            out = acts[-1]
            delta = [2.0 * (out[j] - x[j]) / len(x) for j in range(len(x))]
            loss += sum((out[j] - x[j]) ** 2 for j in range(len(x))) / len(x)
            for l in range(n_layers - 1, -1, -1):
                a_prev = acts[l]
                for j in range(len(delta)):
                    dj = delta[j]
                    gb[l][j] += dj
                    for i in range(len(a_prev)):
                        gW[l][i][j] += a_prev[i] * dj
                if l > 0:
                    new_delta = []
                    for i in range(len(a_prev)):
                        s = sum(self.W[l][i][j] * delta[j] for j in range(len(delta)))
                        new_delta.append(s * (1.0 - a_prev[i] ** 2))   # tanh'
                    delta = new_delta
        m = max(len(batch), 1)
        for l in range(n_layers):
            for i in range(len(gW[l])):
                Wi, gi = self.W[l][i], gW[l][i]
                for j in range(len(gi)):
                    Wi[j] -= lr * gi[j] / m
            for j in range(len(gb[l])):
                self.b[l][j] -= lr * gb[l][j] / m
        return loss / m

    def fit(self, X: List[List[float]], epochs: int = 30, lr: float = 0.05,
            batch: int = 32, seed: int = 5, verbose: bool = False) -> "Autoencoder":
        n, d = len(X), len(X[0])
        self.mean = [sum(r[j] for r in X) / n for j in range(d)]
        self.scale = [max(math.sqrt(sum((r[j] - self.mean[j]) ** 2 for r in X) / max(n - 1, 1)), 1e-6)
                      for j in range(d)]
        Z = [[(r[j] - self.mean[j]) / self.scale[j] for j in range(d)] for r in X]
        rng = random.Random(seed)
        idx = list(range(n))
        for ep in range(epochs):
            rng.shuffle(idx)
            tot = 0.0
            nb = 0
            for s in range(0, n, batch):
                bs = [Z[i] for i in idx[s:s + batch]]
                tot += self._train_batch(bs, lr * (0.5 ** (ep / max(epochs / 3, 1))))
                nb += 1
            if verbose and (ep % 5 == 0 or ep == epochs - 1):
                print(f"    autoencoder epoch {ep:3d}  mse={tot / max(nb,1):.5f}")
        errs, feat = [], [[] for _ in range(d)]
        for z in Z:
            out = self._forward(z)[-1]
            e = [(out[j] - z[j]) ** 2 for j in range(d)]
            errs.append(sum(e) / d)
            for j in range(d):
                feat[j].append(e[j])
        errs_s = sorted(errs)
        self.err_mean = sum(errs) / len(errs)
        self.err_limit = max(errs_s[int(0.995 * (len(errs_s) - 1))], 1e-9)
        self.feat_limit = [max(sorted(f)[int(0.995 * (len(f) - 1))], 1e-9) for f in feat]
        return self

    def score(self, x: Sequence[float]) -> dict:
        d = len(x)
        z = [(x[j] - self.mean[j]) / self.scale[j] for j in range(d)]
        out = self._forward(z)[-1]
        per = [(out[j] - z[j]) ** 2 for j in range(d)]
        err = sum(per) / d
        return {"recon_err": err, "recon_norm": err / max(self.err_limit, 1e-12),
                "per_feature": per,
                "per_feature_norm": [per[j] / max(self.feat_limit[j], 1e-12) for j in range(d)]}

    def to_dict(self) -> dict:
        return {"dims": self.dims, "W": self.W, "b": self.b, "mean": self.mean,
                "scale": self.scale, "err_mean": self.err_mean,
                "err_limit": self.err_limit, "feat_limit": self.feat_limit}

    @staticmethod
    def from_dict(d: dict) -> "Autoencoder":
        a = Autoencoder(d["dims"])
        for k in ("W", "b", "mean", "scale", "err_mean", "err_limit", "feat_limit"):
            setattr(a, k, d[k])
        return a


# --------------------------------------------------------------------------
# Fused engine
# --------------------------------------------------------------------------
class AnomalyEngine:
    """Fuses the three detectors and produces an explainable verdict."""

    WINSOR = 8.0             # feature z-scores are clipped before scoring
    BASELINE_SAMPLES = 120   # stabilised samples per regime before drift detection
    SEQ_PERIOD_S = 1.0       # sequential detectors update at 1 Hz, not 20 Hz

    def __init__(self, feature_names: List[str]):
        self.names = list(feature_names)
        self.pca: Optional[PCAMonitor] = None
        self.ae: Optional[Autoencoder] = None
        self.cusum: Dict[str, CUSUM] = {n: CUSUM() for n in feature_names}
        self.ewma: Dict[str, EWMA] = {n: EWMA(0.02) for n in feature_names}
        self.score_ewma = EWMA(0.08)
        self.trained = False
        self._resid_stats = OnlineStats(len(feature_names))
        # Per-airframe, per-regime adaptive baseline. The fleet-trained PCA and
        # autoencoder answer "is this engine like a healthy fleet engine"; the
        # sequential detector answers "has *this* engine changed since the start
        # of the sortie". Two references are needed because an engine that is
        # merely older than the training fleet is a legitimate mean shift.
        #
        # The baseline is kept per operating regime because several indices are
        # only approximately operating-point invariant: compressor efficiency
        # cannot be measured at low pressure ratio, and oil pressure carries no
        # pump information while the relief valve is open. Holding one baseline
        # across regimes makes a climb-to-cruise transition read as drift.
        self._tail: Dict[str, OnlineStats] = {}
        self._next_seq_t = 0.0

    def fit(self, X: List[List[float]], epochs: int = 25, verbose: bool = False) -> None:
        self.pca = PCAMonitor().fit(X, self.names)
        dims = self._ae_dims(len(self.names))
        self.ae = Autoencoder(dims).fit(X, epochs=epochs, verbose=verbose)
        for row in X:
            self._resid_stats.update(row)
        self.trained = True
        self.reset_sequential()

    @staticmethod
    def _ae_dims(d: int) -> List[int]:
        h = max(6, int(d * 0.6))
        z = max(3, int(d * 0.22))
        return [d, h, z, h, d]

    def reset_sequential(self) -> None:
        """Clear the per-airframe baseline and the sequential detectors.
        Called when the twin is bound to a different engine or a new sortie."""
        for c in self.cusum.values():
            c.reset()
        for e in self.ewma.values():
            e.value = 0.0
        self._tail = {}
        self._next_seq_t = 0.0
        self.score_ewma.value = 0.0

    def _winsorise(self, x: Sequence[float]) -> List[float]:
        """Clip against the fleet baseline before scoring, so a single throttle
        transient cannot dominate a quadratic statistic."""
        m, s = self._resid_stats.mean, self._resid_stats.std()
        return [max(m[i] - self.WINSOR * s[i], min(m[i] + self.WINSOR * s[i], x[i]))
                for i in range(len(x))]

    @property
    def _tail_ready(self) -> bool:
        return any(st.count >= self.BASELINE_SAMPLES for st in self._tail.values())

    def _regime_baseline(self, regime: str) -> Optional[OnlineStats]:
        st = self._tail.get(regime)
        return st if (st is not None and st.count >= self.BASELINE_SAMPLES) else None

    def score(self, x: Sequence[float], steady: bool = False,
              t_s: float = 0.0, regime: str = "unknown") -> dict:
        """Score one sample.

        `steady` must be true only at a stabilised operating point. The two
        static detectors run every sample, but the sequential ones do not: a
        CUSUM compares a value against a fixed reference, so running it through
        a climb or a throttle movement makes it accumulate legitimate operating-
        point changes until it saturates. It is therefore gated on steady state
        and decimated to 1 Hz, which is also the rate its slack and decision
        interval were designed around.
        """
        if not self.trained:
            return {"score": 0.0, "status": "untrained", "contributors": [],
                    "drifting": []}
        x = self._winsorise(x)
        p = self.pca.score(x)
        a = self.ae.score(x)

        # Sequential detection runs against this airframe's own baseline, learned
        # from its first stabilised samples rather than from the training fleet.
        drifting: List[dict] = []
        cus = {n: self.cusum[n].score for n in self.names}
        base = self._regime_baseline(regime)
        if steady and t_s + 1e-9 >= self._next_seq_t:
            self._next_seq_t = t_s + self.SEQ_PERIOD_S
            st = self._tail.setdefault(regime, OnlineStats(len(self.names)))
            if base is None:
                # Only adopt a sample into this airframe's baseline while the
                # fleet-trained detectors agree the engine looks healthy.
                # Without this gate an aircraft that launches with a fault
                # already present learns the faulted state as its own normal,
                # and the sequential detector is blind to it for the rest of the
                # sortie - the characteristic failure of adaptive baselining.
                if max(p["t2_norm"], p["spe_norm"], a["recon_norm"]) < 1.0:
                    st.update(x)
            else:
                # Reference mean from this airframe in this regime; reference
                # *spread* from the training fleet. Using the airframe's own
                # spread would take the scale from a single operating point, and
                # every later power change would read as many sigma of drift.
                z = base.normalise(x, scale=self._resid_stats.std())
                for i, n in enumerate(self.names):
                    fl = self.cusum[n].update(z[i])
                    self.ewma[n].update(z[i])
                    cus[n] = self.cusum[n].score
                    if fl != 0:
                        drifting.append({"feature": n,
                                         "direction": "high" if fl > 0 else "low",
                                         "cusum": round(self.cusum[n].score, 2),
                                         "ewma_sigma": round(self.ewma[n].value, 2)})
        drift_score = (min(4.0, max(cus.values())) if (cus and self._tail_ready) else 0.0)

        # Fusion: the maximum over the static detectors keeps a single strong
        # one from being diluted, the mean rewards agreement, and drift enters
        # additively and bounded - slow drift should raise a watch item, not
        # by itself declare a severe fault.
        static = [p["t2_norm"], p["spe_norm"], a["recon_norm"]]
        fused = (0.65 * max(static) + 0.35 * (sum(static) / len(static))
                 + min(1.2, 0.4 * drift_score))
        smooth = self.score_ewma.update(fused)

        zref = (base.normalise(x, scale=self._resid_stats.std())
                if base is not None else self._resid_stats.normalise(x))
        contrib = []
        for i, n in enumerate(self.names):
            c = (0.45 * p["spe_contrib"][i] / max(sum(p["spe_contrib"]), 1e-9)
                 + 0.35 * a["per_feature_norm"][i] / max(sum(a["per_feature_norm"]), 1e-9)
                 + 0.20 * cus[n] / max(sum(cus.values()), 1e-9))
            contrib.append((n, c, zref[i]))
        contrib.sort(key=lambda t: -t[1])

        # Bands are set from the measured healthy distribution: a healthy engine
        # the models have never seen sits at a 99th percentile of roughly 1.2.
        if smooth < 1.2:
            status = "normal"
        elif smooth < 2.0:
            status = "watch"
        elif smooth < 3.5:
            status = "anomaly"
        else:
            status = "severe"

        return {
            "score": smooth, "raw_score": fused, "status": status,
            "t2": p["t2"], "t2_norm": p["t2_norm"],
            "spe": p["spe"], "spe_norm": p["spe_norm"],
            "recon_norm": a["recon_norm"], "drift_score": drift_score,
            "baseline_ready": base is not None,
            "baseline_regimes": {k: v.count for k, v in self._tail.items()},
            "baseline_regime": regime,
            "drifting": drifting,
            "contributors": [{"feature": n, "share": round(c, 4), "sigma": round(sg, 2)}
                             for n, c, sg in contrib[:6]],
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"names": self.names, "pca": self.pca.to_dict(),
                       "ae": self.ae.to_dict(),
                       "resid_mean": self._resid_stats.mean,
                       "resid_m2": self._resid_stats.m2,
                       "resid_count": self._resid_stats.count}, f)

    @staticmethod
    def load(path: str) -> "AnomalyEngine":
        with open(path) as f:
            d = json.load(f)
        e = AnomalyEngine(d["names"])
        e.pca = PCAMonitor.from_dict(d["pca"])
        e.ae = Autoencoder.from_dict(d["ae"])
        e._resid_stats.mean = d["resid_mean"]
        e._resid_stats.m2 = d["resid_m2"]
        e._resid_stats.count = d["resid_count"]
        e.trained = True
        e.reset_sequential()
        return e
