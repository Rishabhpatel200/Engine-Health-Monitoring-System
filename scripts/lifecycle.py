#!/usr/bin/env python3
"""
Accelerated life-cycle campaign.

Degradation is an hours-scale process, so a single sortie shows almost none of
it — which is correct, and is why `degradation_accel` defaults to 1.0. This
script compresses wear so that the *trend-tracking and RUL* machinery can be
exercised end to end over hundreds of equivalent engine hours, and prints how
the health indices and the RUL estimate evolve sortie by sortie.

    python3 scripts/lifecycle.py [--sorties 12] [--accel 250]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional

from dt.anomaly import AnomalyEngine
from dt.rul import RULEstimator
from dt.simulator import MissionSimulator, SimConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sorties", type=int, default=6)
    ap.add_argument("--accel", type=float, default=130.0)
    ap.add_argument("--profile", default="endurance")
    ap.add_argument("--duration", type=float, default=3600.0)
    ap.add_argument("--start-hours", type=int, default=400)
    ap.add_argument("--model", default="data/models.json")
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()

    model = AnomalyEngine.load(a.model) if os.path.exists(a.model) else None
    hours = float(a.start_hours)
    # The degradation trend belongs to the airframe, not to the sortie. A depot
    # health record carries it forward, so the same RUL estimator is handed to
    # each successive sortie's twin rather than being rebuilt - which is also
    # the only way a slope measured in flight hours can ever become significant.
    trend: Optional[RULEstimator] = None
    print(f"Accelerated campaign: {a.sorties} sorties of {a.duration/60:.0f} min "
          f"at {a.accel:.0f}x wear, from {hours:.0f} engine hours\n")
    hdr = (f"{'sortie':>6}{'hours':>8}{'overall':>8}{'comb':>6}{'air':>6}{'fuel':>6}"
           f"{'cool':>6}{'lube':>6}{'mech':>6}{'elec':>6}{'sens':>6}"
           f"{'RUL h':>8}{'90 % band':>13}  driver")
    print(hdr); print("-" * len(hdr))

    for k in range(a.sorties):
        sim = MissionSimulator(SimConfig(profile=a.profile, scenario="healthy",
                                         initial_hours=int(hours),
                                         degradation_accel=a.accel,
                                         record=a.record, seed=40 + k),
                               anomaly=model)
        if trend is not None:
            sim.twin.rul = trend
        trend = sim.twin.rul
        if model:
            model.reset_sequential()
        while sim.t < a.duration:
            sim.step()
        s = sim.twin.last_snapshot
        h = s["health"]["subsystems"]
        e = s["rul"]["engine"]
        rul = f"{e['rul_hours']:.0f}" if e.get("rul_hours") is not None else "—"
        band = (f"{e['lower_hours']:.0f}–{e['upper_hours']:.0f}"
                if e.get("lower_hours") is not None else "—")
        print(f"{k+1:>6}{s['engine_hours']:>8.0f}{s['health']['overall']:>8.1f}"
              f"{h['combustion']:>6.0f}{h['air_path']:>6.0f}{h['fuel_injection']:>6.0f}"
              f"{h['cooling']:>6.0f}{h['lubrication']:>6.0f}{h['mechanical']:>6.0f}"
              f"{h['electrical']:>6.0f}{h['sensing']:>6.0f}"
              f"{rul:>8}{band:>13}  {e.get('driver') or ''}")
        # Carry the accelerated wear clock, not the wall clock, into the next
        # sortie - otherwise every sortie re-ages a fresh engine to the same
        # number of hours and the campaign never advances.
        hours = max(s["engine_hours"], sim.degradation.total_hours)
        sim.close()

    print("\nRUL narrows as the trend accumulates evidence and stays blank while the slope"
          "\nis not yet statistically distinguishable from zero. Sawtooth in the combustion"
          "\nand fuel columns is the scheduled replacement of consumables - spark plugs every"
          "\n200 h, oil every 100 h - not noise in the estimate.")


if __name__ == "__main__":
    main()
