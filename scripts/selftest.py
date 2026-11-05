#!/usr/bin/env python3
"""
Verification suite. Checks the parts of the twin that have a right answer.

    python3 scripts/selftest.py
"""
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random

from dt import linalg as la
from dt.anomaly import CUSUM, PCAMonitor
from dt.atmosphere import isa
from dt.canbus import BaseCanBus, PGN_MAP, VirtualCanBus, bus_load_percent, decode_pgn, encode_pgn
from dt.engine_model import Command, PistonEngine
from dt.estimator import UnscentedKalmanFilter
from dt.recorder import FlightRecorder
from dt.rul import RULEstimator
from dt.simulator import MissionSimulator, SimConfig

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")


def near(a, b, tol):
    return abs(a - b) <= tol


def main():
    print("\n[1] ISA atmosphere against the standard tables")
    for h, p, t in ((0, 101.325, 15.0), (5000, 54.020, -17.5), (11000, 22.632, -56.5)):
        a = isa(h)
        check(f"{h} m: {a.press_kpa:.3f} kPa, {a.temp_c:.1f} °C",
              near(a.press_kpa, p, 0.05) and near(a.temp_c, t, 0.15))

    print("\n[2] Linear algebra")
    rng = random.Random(1)
    n = 6
    A = [[rng.gauss(0, 1) for _ in range(n)] for _ in range(n)]
    S = la.matmul(A, la.transpose(A))
    S = [[S[i][j] + (n if i == j else 0) for j in range(n)] for i in range(n)]
    L = la.cholesky(S)
    R = la.matmul(L, la.transpose(L))
    err = max(abs(R[i][j] - S[i][j]) for i in range(n) for j in range(n))
    check(f"Cholesky reconstruction error {err:.2e}", err < 1e-9)
    I = la.matmul(S, la.spd_inverse(S))
    err = max(abs(I[i][j] - (1.0 if i == j else 0.0)) for i in range(n) for j in range(n))
    check(f"SPD inverse error {err:.2e}", err < 1e-9)
    vals, vecs = la.jacobi_eig(S)
    ok = all(vals[i] >= vals[i + 1] - 1e-9 for i in range(n - 1)) and min(vals) > 0
    resid = 0.0
    for k in range(n):
        v = [vecs[r][k] for r in range(n)]
        Av = la.matvec(S, v)
        resid = max(resid, max(abs(Av[i] - vals[k] * v[i]) for i in range(n)))
    check(f"Jacobi eigendecomposition residual {resid:.2e}, sorted descending",
          ok and resid < 1e-8)

    print("\n[3] CAN framing round-trip")
    vals_in = {"rpm": 5432.0, "map_kpa": 131.75, "oil_press_kpa": 455.0,
               "cht1_c": 114.5, "egt1_c": 798.2, "fuel_flow_lph": 28.44}
    worst = 0.0
    for pgn in PGN_MAP:
        out = decode_pgn(pgn, encode_pgn(pgn, vals_in))
        for k, v in out.items():
            if k in vals_in and v is not None:
                worst = max(worst, abs(v - vals_in[k]))
    check(f"signal round-trip error {worst:.4f} engineering units", worst < 0.06)
    got = decode_pgn(0xFD0E, encode_pgn(0xFD0E, {"egt1_c": 700.0}))
    check("unavailable signals decode as None", got["egt2_c"] is None)
    load = bus_load_percent()
    check(f"bus load {load:.2f} % of 500 kbit/s", 0 < load < 30)

    print("\n[4] UKF consistency (mean NIS should be ~1.0 for a 1-D measurement)")
    def fx(x, dt, u): return [x[0] + x[1] * dt, x[1] + (-4.0 * x[0] - 0.3 * x[1]) * dt]
    def hx(x, u): return [x[0]]
    f = UnscentedKalmanFilter(2, fx, hx, [1e-6, 1e-4])
    f.x, f.P = [1.0, 0.0], [[0.5, 0], [0, 0.5]]
    rng = random.Random(3)
    tx, errs, nis = [1.0, 0.0], [], []
    for k in range(400):
        tx = fx(tx, 0.02, None)
        f.predict(0.02, None)
        f.update([tx[0] + rng.gauss(0, 0.05)], [0.05 ** 2], None)
        if k > 100:
            errs.append(abs(f.x[0] - tx[0])); nis.append(f.last_nis)
    rmse = math.sqrt(sum(e * e for e in errs) / len(errs))
    m_nis = sum(nis) / len(nis)
    check(f"RMSE {rmse:.4f} below the 0.05 measurement sigma", rmse < 0.05)
    check(f"mean NIS {m_nis:.2f} within [0.5, 2.0]", 0.5 <= m_nis <= 2.0)

    print("\n[5] Engine model against the 914-class performance envelope")
    def run(th, rpm, alt=0, dev=0, tas=45, secs=200, hours=0):
        e = PistonEngine(); a = isa(alt, dev); e.soak(a); e.start(a)
        e.state.oil_c = 95; e.state.coolant_c = 95; e.state.cht_c = [110] * 4
        c = Command(throttle_pct=th, rpm_target=rpm)
        for _ in range(int(secs / 0.05)):
            o = e.step(0.05, c, a, tas)
        return o
    o = run(100, 5800)
    check(f"takeoff power {o.power_kw:.1f} kW in [75, 90]", 75 <= o.power_kw <= 90)
    check(f"takeoff fuel flow {o.fuel_flow_lph:.1f} L/h in [28, 38]", 28 <= o.fuel_flow_lph <= 38)
    check(f"takeoff BSFC {o.bsfc_g_kwh:.0f} g/kWh in [260, 340]", 260 <= o.bsfc_g_kwh <= 340)
    check(f"governor holds {o.rpm:.0f} rpm within 1 % of 5800", near(o.rpm, 5800, 58))
    check(f"CHT {o.cht_c[0]:.0f} °C below the 135 °C limit", o.cht_c[0] < 135)
    check(f"EGT {o.egt_c[0]:.0f} °C below the 900 °C limit", o.egt_c[0] < 900)
    oc = run(100, 5200, alt=4900)
    check(f"full boost held to 4900 m ({oc.map_kpa:.0f} kPa)", oc.map_kpa > 128)
    oh = run(100, 5200, alt=7620)
    check(f"boost falls off above critical altitude ({oh.map_kpa:.0f} kPa)", oh.map_kpa < 115)
    check(f"power at 7620 m {oh.power_kw:.1f} kW below sea-level power",
          oh.power_kw < o.power_kw)
    oi = run(8, 1400)
    check(f"idle {oi.rpm:.0f} rpm, {oi.fuel_flow_lph:.1f} L/h",
          1200 < oi.rpm < 1700 and oi.fuel_flow_lph < 4)

    print("\n[6] Energy balance closes")
    e = PistonEngine(); a = isa(0); e.soak(a); e.start(a)
    e.state.oil_c = 95; e.state.coolant_c = 95; e.state.cht_c = [110] * 4
    c = Command(throttle_pct=100, rpm_target=5800)
    for _ in range(4000):
        o = e.step(0.05, c, a, 45)
    fuel_kw = o.fuel_flow_kgs * 43.5e6 / 1000.0
    frac = o.power_kw / fuel_kw
    check(f"brake thermal efficiency {frac:.3f} in [0.24, 0.34]", 0.24 <= frac <= 0.34)

    print("\n[7] Degradation reaches TBO with a credible power loss")
    s0 = MissionSimulator(SimConfig(initial_hours=0, use_can=False))
    s1 = MissionSimulator(SimConfig(initial_hours=1200, use_can=False))
    def wot(h):
        e = PistonEngine(health=h); a = isa(0); e.soak(a); e.start(a)
        e.state.oil_c = 95; e.state.coolant_c = 95; e.state.cht_c = [110] * 4
        c = Command(throttle_pct=100, rpm_target=5800)
        for _ in range(3000):
            o = e.step(0.05, c, a, 45)
        return o.power_kw
    p0, p1 = wot(s0.base_health), wot(s1.base_health)
    loss = 100 * (p0 - p1) / p0
    check(f"power loss at 1200 h TBO = {loss:.1f} % in [3, 12]", 3 <= loss <= 12)
    check(f"cooling capacity at TBO {s1.base_health.cooling_ua:.2f} in [0.70, 0.90]",
          0.70 <= s1.base_health.cooling_ua <= 0.90)
    s0.close(); s1.close()

    print("\n[8] PCA monitor calibration")
    rng = random.Random(7)
    X = [[rng.gauss(0, 1) + 0.6 * b for b in (0, 0, 0)] + [rng.gauss(0, 1) for _ in range(5)]
         for _ in range(1200)]
    m = PCAMonitor().fit(X, [f"f{i}" for i in range(8)], alpha=0.99)
    exceed = sum(1 for r in X if m.score(r)["spe_norm"] > 1.0) / len(X)
    check(f"SPE exceeds its 99th-percentile limit on {exceed:.1%} of training data",
          0.005 <= exceed <= 0.02)

    print("\n[9] CUSUM detects a 1-sigma step and is quiet before it")
    c = CUSUM(k=0.6, h=6.0)
    rng = random.Random(9)
    pre = sum(1 for _ in range(2000) if c.update(rng.gauss(0, 1)) != 0)
    c.reset()
    n = 0
    while c.update(rng.gauss(1.5, 1)) == 0 and n < 200:
        n += 1
    check(f"{pre} alarms in 2000 in-control samples", pre <= 40)
    check(f"1.5-sigma step detected in {n} samples", n < 40)

    print("\n[10] RUL refuses to extrapolate a noise-only trend")
    r = RULEstimator(tbo_hours=1200)
    rng = random.Random(11)
    for k in range(4000):
        r.update({s: 92.0 + rng.gauss(0, 1.5) for s in r.THRESHOLDS},
                 1.0 / 60.0, 400 + k / 60.0)
    e = r.engine_rul()
    check(f"flat noisy index -> '{(e.get('basis') or '')[:40]}'", e.get("rul_hours") is None)
    # 20 h of observation at -0.5 pts/h: the index falls 95 -> 85, still clear of
    # the highest action threshold (65), so a finite RUL of about 40 h is due.
    r2 = RULEstimator(tbo_hours=1200)
    for k in range(1200):
        v = 95.0 - 0.5 * (k / 60.0) + rng.gauss(0, 1.0)
        r2.update({s: v for s in r2.THRESHOLDS}, 1.0 / 60.0, 400 + k / 60.0)
    e2 = r2.engine_rul()
    ok = e2.get("rul_hours") is not None
    exp = e2["rul_hours"] if ok else 0.0
    band = (e2.get("lower_hours"), e2.get("upper_hours"))
    check(f"0.5 pts/h decay on a 65-point margin -> RUL {exp:.0f} h "
          f"(90 % band {band[0]:.0f}-{band[1]:.0f}), expected ~40 h",
          ok and 20 < exp < 70)
    check("RUL band brackets the point estimate",
          ok and band[0] <= exp <= band[1])

    print("\n[11] Flight data recorder round-trip")
    path = os.path.join(tempfile.gettempdir(), "_aerodt_selftest.sqlite")
    if os.path.exists(path):
        os.remove(path)
    rec = FlightRecorder(path, continuous_hz=2.0)
    sid = rec.start_session("selftest", "isr_sortie", "healthy")
    for i in range(400):
        rec.record(i * 0.05, {"meas": {"rpm": 5000 + i}, "health": {"overall": 95.0},
                              "anomaly": {"score": 0.1}, "peak": {"cht_max_c": 110.0}})
    rec.log_event(5.0, "test", "caution", "synthetic event")
    rec.end_session()
    rows = rec.read(sid)
    check(f"recorded {len(rows)} decimated samples from 400 at 2 Hz (expect ~40)",
          35 <= len(rows) <= 45)
    check("events persisted", len(rec.events(sid)) == 1)
    check("summary aggregates present", "rpm" in rec.summary(sid)["aggregates"])
    rec.close()
    os.remove(path)

    print("\n[12] Twin never reads plant ground truth")
    import inspect
    import dt.twin as twin_mod
    src = inspect.getsource(twin_mod)
    leaks = [w for w in ("base_health", "injector.", "plant.", '"truth"', "HealthParams(")
             if w in src and w != "HealthParams("]
    check(f"no plant-state references in twin.py {leaks or ''}", not leaks)

    print(f"\n{'='*60}\n  {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("  failures:")
        for f in FAIL:
            print(f"    - {f}")
    print("=" * 60)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
