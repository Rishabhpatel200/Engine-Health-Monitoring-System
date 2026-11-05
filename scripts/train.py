#!/usr/bin/env python3
"""
Train the anomaly-detection models on healthy-engine data.

Training data is deliberately drawn from *several* mission profiles, engine
ages and atmospheres. A model trained on one profile learns that profile, not
health: it will then flag a legitimate high-altitude loiter as an anomaly. The
spread here is what makes the detector transferable across the envelope.

    python3 scripts/train.py [--samples 4000] [--epochs 25] [--out data/models.json]
"""
import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dt.anomaly import AnomalyEngine
from dt.simulator import MissionSimulator, SimConfig
from dt.twin import FEATURES

# Baseline identification runs over engine ages and atmospheres, so the models
# learn health rather than a particular airframe or a particular day.
TRAIN_SET = [
    # profile,             hours, ISA dev, cap_s, seed
    ("envelope_sweep",      120,    0,   1500, 11),
    ("envelope_sweep",      450,  -20,   1500, 12),
    ("envelope_sweep",      780,  +30,   1500, 13),
    ("envelope_sweep",     1050,  +12,   1500, 14),
    ("isr_sortie",          300,    0,   2400, 15),
    ("hot_weather",         600,    0,   3000, 16),
    ("throttle_transients", 400,    0,   1200, 17),
]


def collect(target: int, verbose: bool = True):
    """Run each condition to completion, then decimate evenly across it.

    Decimating *after* the run is what gives even coverage: taking the first N
    samples instead would hand the models nothing but the climb.
    """
    rows = []
    per = max(200, target // len(TRAIN_SET))
    for prof, hours, isa_dev, cap_s, seed in TRAIN_SET:
        t0 = time.time()
        sim = MissionSimulator(SimConfig(profile=prof, scenario="healthy",
                                         initial_hours=hours, seed=seed,
                                         use_can=True, record=False))
        if isa_dev:
            sim.profile.isa_dev_k = isa_dev
        sim.twin.collect_features = True
        end = min(sim.profile.duration_s, cap_s)
        while sim.t < end:
            sim.step()
        got = sim.twin.feature_log
        step = max(1, len(got) // per)
        take = got[::step][:per]
        rows.extend(take)
        if verbose:
            print(f"  {prof:20s} {hours:4d} h  ISA{isa_dev:+3d}  "
                  f"{sim.t:5.0f} s  ->  {len(take):4d} of {len(got)} samples "
                  f"({time.time()-t0:.1f} s)")
        sim.close()
    random.Random(3).shuffle(rows)
    return rows[:target]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=3500)
    ap.add_argument("--epochs", type=int, default=22)
    ap.add_argument("--out", default="data/models.json")
    a = ap.parse_args()

    print(f"Collecting healthy-engine training data ({len(FEATURES)} features)…")
    X = collect(a.samples)
    print(f"  total {len(X)} samples\n")

    print("Fitting PCA monitor and autoencoder…")
    t0 = time.time()
    eng = AnomalyEngine(FEATURES)
    eng.fit(X, epochs=a.epochs, verbose=True)
    print(f"  fit completed in {time.time()-t0:.1f} s")
    print(f"  PCA: {len(eng.pca.eigenvalues)} of {len(FEATURES)} components retained, "
          f"T² limit {eng.pca.t2_limit:.2f}, SPE limit {eng.pca.spe_limit:.3f}")
    print(f"  autoencoder: {'-'.join(map(str, eng.ae.dims))}, "
          f"reconstruction limit {eng.ae.err_limit:.5f}")

    # Held-out false-alarm check: profiles, ages and atmospheres the models have
    # not seen. Each engine is scored with a fresh sequential state, because the
    # CUSUMs are per-airframe and must not be carried across aircraft.
    print("\nFalse-alarm check on unseen healthy engines…")
    total, alarms = 0, 0
    for prof, hours, seed in [("isr_sortie", 430, 91), ("hot_weather", 260, 92),
                              ("high_altitude", 810, 93), ("endurance", 150, 94),
                              ("cold_high", 990, 95)]:
        sim = MissionSimulator(SimConfig(profile=prof, scenario="healthy",
                                         initial_hours=hours, seed=seed),
                               anomaly=None)
        sim.twin.anomaly = eng
        eng.reset_sequential()
        sim.twin.collect_features = True
        end = min(sim.profile.duration_s, 2400)
        while sim.t < end:
            sim.step()
        sc = [eng.score(x)["raw_score"] for x in sim.twin.feature_log[::5]]
        t2 = [eng.pca.score(eng._winsorise(x))["t2_norm"] for x in sim.twin.feature_log[::25]]
        spe = [eng.pca.score(eng._winsorise(x))["spe_norm"] for x in sim.twin.feature_log[::25]]
        ae = [eng.ae.score(eng._winsorise(x))["recon_norm"] for x in sim.twin.feature_log[::25]]
        n = sum(1 for v in sc if v > 2.0)
        q = lambda v: sorted(v)[int(.99 * (len(v) - 1))]
        print(f"  {prof:16s} {hours:4d} h  {len(sc):5d} samples  mean {sum(sc)/max(len(sc),1):6.3f}"
              f"  p99 {q(sc):6.3f}   [T² {q(t2):5.2f}  SPE {q(spe):5.2f}  AE {q(ae):5.2f}]"
              f"  false alarms {n/max(len(sc),1):6.2%}")
        total += len(sc); alarms += n
        sim.close()
    eng.reset_sequential()
    print(f"  overall false-alarm rate above 2.0 = {alarms/max(total,1):.2%}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    eng.save(a.out)
    print(f"\nSaved to {a.out} ({os.path.getsize(a.out)/1024:.0f} kB)")


if __name__ == "__main__":
    main()
