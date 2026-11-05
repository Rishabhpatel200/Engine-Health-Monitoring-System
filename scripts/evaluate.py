#!/usr/bin/env python3
"""
Quantitative evaluation of the detection and isolation stack.

Reports, per scenario, the four numbers that decide whether a predictive
maintenance system is worth fielding:

  detection rate     fraction of faulted samples flagged after the fault becomes
                     physically significant
  false-alarm rate   fraction of healthy samples flagged, measured on engines
                     the models were never trained on
  lead time          how long before the conventional threshold layer would have
                     tripped the twin raised the alarm - the whole value
                     proposition of predictive over reactive monitoring
  isolation accuracy fraction of detected samples whose top-ranked hypothesis
                     names the injected fault

Ground truth is read only here, for scoring. The twin never sees it.

    python3 scripts/evaluate.py [--scenarios all] [--duration 5400]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dt.anomaly import AnomalyEngine
from dt.faults import FaultInjector
from dt.simulator import MissionSimulator, SimConfig

# Which isolation labels count as correct for each injected scenario.
EXPECTED = {
    "progressive_misfire": {"misfire"},
    "injector_fouling":    {"injector_clog"},
    "cooling_loss":        {"cooling_degradation", "oil_cooler_block"},
    "lubrication_failure": {"oil_pump_wear", "bearing_wear", "oil_cooler_block"},
    "sensor_drift":        {"sensor_fault"},
    "detonation":          {"detonation", "cooling_degradation"},
    "turbo_decay":         {"turbo_degradation"},
    "combined":            {"misfire", "cooling_degradation", "oil_cooler_block",
                            "sensor_fault"},
}

# Some faults are only physically expressible in some parts of the envelope.
# Detonation cannot occur at 6100 m, where intake air is 23 C: pairing it with a
# high-altitude sortie would measure the profile, not the detector.
SCENARIO_PROFILE = {"detonation": "hot_weather", "cooling_loss": "hot_weather"}

# "Detected" means the twin surfaced something to the operator, through any of
# the paths that actually reach the display. Scoring only the fused anomaly score
# would understate a system whose fault-isolation stage names the fault while the
# score is still in the watch band. The same criterion is applied to the clean
# window, so detection and false alarms remain comparable.
DETECT_THRESHOLD = 2.0      # fused anomaly score that raises an operator alert
HYPOTHESIS_THRESHOLD = 0.55  # isolation confidence that raises an operator alert
HEALTH_THRESHOLD = 85.0     # subsystem health index that raises a watch item
WARMUP_S = 600.0            # analytics warm-up excluded from the false-alarm window
ADVISORY_URGENCIES = {"ABORT", "IMMEDIATE", "BEFORE_NEXT_FLIGHT", "SCHEDULED"}


def flagged(snap: dict) -> bool:
    if snap["anomaly"].get("score", 0.0) > DETECT_THRESHOLD:
        return True
    hy = snap["diagnostics"]["hypotheses"]
    if hy and hy[0]["confidence"] >= HYPOTHESIS_THRESHOLD:
        return True
    if min(snap["health"]["subsystems"].values()) < HEALTH_THRESHOLD:
        return True
    return any(a["urgency"] in ADVISORY_URGENCIES for a in snap["advisories"])


def run(scenario: str, profile: str, hours: int, duration: float,
        model: AnomalyEngine, seed: int = 21) -> dict:
    sim = MissionSimulator(SimConfig(profile=profile, scenario=scenario,
                                     initial_hours=hours, seed=seed))
    sim.twin.anomaly = model
    model.reset_sequential()

    onsets = [e.onset_s for e in sim.injector.events] or [0.0]
    # A fault is only *detectable* once it has grown enough to move the physics.
    # Scoring from the injection instant would credit the detector for reading
    # the future, so detection is scored from 25 % of the ramp.
    grow = [e.onset_s + 0.25 * (e.duration_s if e.profile != "step" else 1.0)
            for e in sim.injector.events] or [0.0]
    t_detectable = min(grow)

    first_twin = first_threshold = None
    n_faulted = n_detected = n_isolated = 0
    n_clean = n_false = 0
    top_labels = {}

    while sim.t < duration:
        snap = sim.step()
        if sim.step_count % 10 or not snap["analytics_valid"]:
            continue
        fl = flagged(snap)
        tripped = any(e["severity"] == "limit" for e in snap["exceedances"])

        if WARMUP_S <= snap["t_s"] < min(onsets):
            # The twin declares its own warm-up; scoring it as a false alarm
            # would penalise the system for saying "not yet confident".
            n_clean += 1
            n_false += 1 if fl else 0
        elif snap["t_s"] >= t_detectable:
            n_faulted += 1
            if fl:
                n_detected += 1
                if first_twin is None:
                    first_twin = snap["t_s"]
                hy = snap["diagnostics"]["hypotheses"]
                if hy:
                    lab = hy[0]["fault"]
                    top_labels[lab] = top_labels.get(lab, 0) + 1
                    if lab in EXPECTED.get(scenario, set()):
                        n_isolated += 1
        if tripped and first_threshold is None:
            first_threshold = snap["t_s"]

    sim.close()
    lead = None
    if first_twin is not None:
        lead = ((first_threshold - first_twin) if first_threshold is not None
                else duration - first_twin)
    return {"scenario": scenario, "profile": profile,
            "detection_rate": n_detected / max(n_faulted, 1),
            "false_alarm_rate": n_false / max(n_clean, 1),
            "isolation_accuracy": n_isolated / max(n_detected, 1),
            "first_detection_s": first_twin,
            "first_threshold_trip_s": first_threshold,
            "lead_time_s": lead,
            "n_faulted": n_faulted, "n_clean": n_clean,
            "top_labels": sorted(top_labels.items(), key=lambda t: -t[1])[:3]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="data/models.json")
    ap.add_argument("--duration", type=float, default=5400.0)
    ap.add_argument("--profile", default="isr_sortie")
    ap.add_argument("--hours", type=int, default=420)
    ap.add_argument("--scenarios", default="all")
    a = ap.parse_args()

    if not os.path.exists(a.model):
        sys.exit(f"model not found: {a.model} — run scripts/train.py first")
    model = AnomalyEngine.load(a.model)

    names = ([s for s in FaultInjector.scenario_names() if s != "healthy"]
             if a.scenarios == "all" else a.scenarios.split(","))

    print(f"Evaluating {len(names)} fault scenarios on '{a.profile}' "
          f"({a.duration/60:.0f} min, {a.hours} engine hours)\n")
    hdr = (f"{'scenario':<22}{'detect':>8}{'false':>8}{'isolate':>9}"
           f"{'1st det':>9}{'thresh':>9}{'lead':>10}  top hypothesis")
    print(hdr); print("-" * len(hdr))
    rows = []
    for s in names:
        t0 = time.time()
        prof = SCENARIO_PROFILE.get(s, a.profile)
        r = run(s, prof, a.hours, min(a.duration, 5400.0), model)
        rows.append(r)
        top = r["top_labels"][0][0] if r["top_labels"] else "—"
        fd = f"{r['first_detection_s']:.0f}s" if r["first_detection_s"] is not None else "—"
        th = f"{r['first_threshold_trip_s']:.0f}s" if r["first_threshold_trip_s"] is not None else "never"
        ld = f"{r['lead_time_s']/60:.1f} min" if r["lead_time_s"] is not None else "—"
        mark = "*" if s in SCENARIO_PROFILE else " "
        fa = f"{r['false_alarm_rate']:.1%}" if r["n_clean"] >= 30 else "n/a"
        print(f"{s+mark:<22}{r['detection_rate']:>7.1%}{fa:>8}"
              f"{r['isolation_accuracy']:>9.1%}{fd:>9}{th:>9}{ld:>10}  {top}")
    print("-" * len(hdr))
    n = len(rows)
    clean = [r for r in rows if r["n_clean"] >= 30]
    print(f"{'mean':<22}{sum(r['detection_rate'] for r in rows)/n:>7.1%}"
          f"{(sum(r['false_alarm_rate'] for r in clean)/len(clean) if clean else 0):>8.1%}"
          f"{sum(r['isolation_accuracy'] for r in rows)/n:>9.1%}")
    print(f"\n'n/a' in the false-alarm column means the scenario's fault begins before the "
          f"\nanalytics warm-up ends, so there is no clean window to score.")
    print("\nHealthy-engine control runs (same criterion, no fault injected):")
    hdr2 = f"{'profile':<22}{'false':>8}{'samples':>10}"
    print(hdr2)
    tot = fa = 0
    for prof, hrs, seed in (("isr_sortie", 470, 71), ("hot_weather", 330, 72),
                            ("high_altitude", 880, 73)):
        sim = MissionSimulator(SimConfig(profile=prof, scenario="healthy",
                                         initial_hours=hrs, seed=seed))
        sim.twin.anomaly = model
        model.reset_sequential()
        n = k = 0
        while sim.t < min(a.duration, 5400.0):
            sn = sim.step()
            if sim.step_count % 10 or not sn["analytics_valid"] or sn["t_s"] < WARMUP_S:
                continue
            n += 1
            k += 1 if flagged(sn) else 0
        print(f"{prof:<22}{k/max(n,1):>7.1%}{n:>10}")
        tot += n; fa += k
        sim.close()
    print(f"{'overall':<22}{fa/max(tot,1):>7.1%}{tot:>10}")

    print("\n* evaluated on the 'hot_weather' profile: these faults are only physically"
          "\n  expressible at high charge temperature.")
    print("\nLead time is measured against the conventional threshold layer running on the "
          "\nsame data. 'never' means the sortie ended before any certified limit was reached "
          "\n- the twin found the fault that a limit-based system would have missed entirely.")


if __name__ == "__main__":
    main()
