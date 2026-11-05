#!/usr/bin/env python3
"""
End-to-end demonstration: run a sortie with an injected fault and narrate what
the twin concludes, when, and on what evidence.

    python3 scripts/demo.py --scenario progressive_misfire
    python3 scripts/demo.py --scenario cooling_loss --profile hot_weather
    python3 scripts/demo.py --list
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dt.anomaly import AnomalyEngine
from dt.faults import FaultInjector
from dt.missions import PROFILES
from dt.simulator import MissionSimulator, SimConfig


def bar(v, width=26, lo=0.0, hi=100.0):
    n = int(width * max(0.0, min(1.0, (v - lo) / (hi - lo))))
    return "█" * n + "·" * (width - n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="progressive_misfire")
    ap.add_argument("--profile", default="isr_sortie")
    ap.add_argument("--hours", type=int, default=420)
    ap.add_argument("--duration", type=float, default=5400)
    ap.add_argument("--model", default="data/models.json")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list:
        print("scenarios:", ", ".join(FaultInjector.scenario_names()))
        print("profiles: ", ", ".join(sorted(PROFILES)))
        return

    model = AnomalyEngine.load(a.model) if os.path.exists(a.model) else None
    if model is None:
        print(f"! {a.model} not found — running with physics and rules only "
              f"(run scripts/train.py to enable the ML layer)\n")
    sim = MissionSimulator(SimConfig(profile=a.profile, scenario=a.scenario,
                                     initial_hours=a.hours, record=a.record),
                           anomaly=model)
    print(f"Engine   : {sim.cfg.engine.name}")
    print(f"Profile  : {sim.profile.name} — {sim.profile.description}")
    print(f"Scenario : {a.scenario}")
    for e in sim.injector.events:
        print(f"  injected: {e.label():28s} onset t+{e.onset_s:.0f}s  "
              f"{e.profile} to {e.severity:.2f} over {e.duration_s:.0f}s")
        if e.note:
            print(f"            {e.note}")
    print(f"CAN      : {sim.bus.stats()['transport']}\n")

    milestones = []
    seen = set()
    next_print = 0.0
    hdr = f"{'t (s)':>7} {'phase':<12} {'health':>6} {'anom':>6} {'CHTmx':>6} {'EGTmx':>6} {'vib½':>5} {'oilP':>5}  top hypothesis"
    print(hdr); print("-" * len(hdr))
    while sim.t < a.duration:
        s = sim.step()
        if s["t_s"] >= next_print:
            next_print += 300.0
            hy = s["diagnostics"]["hypotheses"]
            top = (f"{hy[0]['fault']}"
                   + (f" cyl{hy[0]['cylinder']}" if hy[0].get("cylinder") else "")
                   + (f" [{hy[0]['channel']}]" if hy[0].get("channel") else "")
                   + f" {hy[0]['confidence']:.0%}") if hy else "—"
            print(f"{s['t_s']:>7.0f} {s['phase']:<12} {s['health']['overall']:>6.1f} "
                  f"{s['anomaly'].get('score',0):>6.2f} {s['peak']['cht_max_c']:>6.1f} "
                  f"{s['peak']['egt_max_c']:>6.0f} {s['derived']['vib_halforder_g']:>5.2f} "
                  f"{s['meas'].get('oil_press_kpa') or 0:>5.0f}  {top}")
        for al in s["alerts"]:
            k = al["text"]
            if k not in seen:
                seen.add(k)
                milestones.append((al["t_s"], al["level"], al["source"], k))

    print("\n── Detection timeline " + "─" * 46)
    for t, lvl, src, txt in milestones:
        print(f"  t+{t:>7.0f}s  [{src:<9}] {lvl:<8} {txt}")
    if not milestones:
        print("  (no alerts raised)")

    s = sim.last_state = sim.twin.last_snapshot
    print("\n── Final health indices " + "─" * 44)
    for k, v in s["health"]["subsystems"].items():
        cf = s["health"]["confidence"].get(k, 0)
        print(f"  {k:<16} {bar(v)} {v:5.1f}   confidence {cf:.0%}")
    print(f"  {'OVERALL':<16} {bar(s['health']['overall'])} {s['health']['overall']:5.1f}")

    print("\n── Remaining useful life " + "─" * 43)
    e = s["rul"]["engine"]
    if e.get("rul_hours") is not None:
        print(f"  {e['rul_hours']:.1f} h  (90 % band {e['lower_hours']:.0f}–{e['upper_hours']:.0f} h)")
    else:
        print("  no statistically significant degradation trend")
    print(f"  {e['basis']}")
    print(f"  hours to overhaul: {e['tbo_remaining_hours']:.0f}")

    print("\n── Maintenance advisories " + "─" * 42)
    for adv in s["advisories"][:6]:
        due = f" (due in {adv['due_hours']:.1f} h)" if adv.get("due_hours") is not None else ""
        print(f"  [{adv['urgency']:<18}] {adv['title']}{due}")
        print(f"      {adv['detail']}")
        for ev in adv.get("evidence", [])[:3]:
            print(f"        · {ev}")
        print(f"      → {adv['action']}")
    if not s["advisories"]:
        print("  none")

    print(f"\n── Mission recommendation " + "─" * 42)
    print(f"  {s['mission']['decision']}: {s['mission']['rationale']}")

    gt = s.get("truth", {})
    print("\n── Ground truth (scoring only; the twin never reads this) " + "─" * 11)
    print(f"  active faults : {', '.join(gt.get('active_faults', [])) or 'none'}")
    print(f"  true power    : {gt.get('power_kw')} kW")
    h = gt.get("health", {})
    print(f"  ignition      : {h.get('ignition_energy')}")
    print(f"  injectors     : {h.get('injector_flow')}")
    print(f"  cooling UA    : {h.get('cooling_ua')}   oil pump: {h.get('oil_pump')}   "
          f"bearing: {h.get('bearing_clearance')}")
    sim.close()


if __name__ == "__main__":
    main()
