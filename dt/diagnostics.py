"""
Fault detection and isolation.

Detection says *something* is wrong; isolation says *what*. The isolation stage
uses a structured-residual (signature matrix) scheme: each candidate fault has a
prescribed pattern of symptoms it must produce, must not produce, or is
indifferent to. Comparing the observed symptom vector against every column of
the matrix yields a ranked, auditable hypothesis list.

This is deliberately not a black-box classifier. A maintainer can read the
evidence line by line and disagree with it - which is the difference between an
advisory a squadron will act on and one it will learn to ignore.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# --------------------------------------------------------------------------
# Symptom extraction: continuous indicators -> soft binary symptoms in [0, 1]
# --------------------------------------------------------------------------
SYMPTOMS: Dict[str, str] = {
    "egt_one_low":        "one cylinder EGT well below the others",
    "egt_one_high":       "one cylinder EGT well above the others",
    "egt_all_high":       "all EGTs elevated",
    "cht_one_high":       "one cylinder CHT above the others",
    "cht_all_high":       "all CHTs elevated together",
    "coolant_high":       "coolant temperature elevated",
    "oil_temp_high":      "oil temperature elevated",
    "oil_press_low":      "oil pressure below model expectation",
    "vib_half_high":      "0.5-order vibration elevated (firing-event periodicity)",
    "vib_1x_high":        "1x vibration elevated (rotating imbalance)",
    "vib_broad_high":     "broadband vibration elevated",
    "fuel_flow_high":     "fuel flow above model expectation",
    "fuel_flow_low":      "fuel flow below model expectation",
    "ve_low":             "volumetric efficiency below model expectation",
    "boost_low":          "manifold pressure below commanded boost",
    "iat_high":           "intake air temperature above model expectation",
    "knock_high":         "knock index elevated",
    "power_low":          "brake power below model expectation",
    "single_channel_resid": "one measurement channel inconsistent with all others",
    "channel_frozen":     "a channel has stopped changing",
    "volt_low":           "bus voltage below regulation",
    "blowby_high":        "crankcase blowby above expectation (inferred)",
    "cooling_capacity_low": "heat-rejection capability below model expectation",
    "cht_vs_model_high":  "one CHT channel disagrees with the model while the others agree",
    "egt_one_deviant":    "one cylinder EGT displaced from the others in either direction",
}

# Signature matrix. Positive = symptom expected, negative = symptom must be
# absent, 0 = uninformative. The magnitude is a weight: +2 marks the symptom
# that actually discriminates this fault from its neighbours (0.5-order
# vibration for misfire, 1x for bearing wear), so a fault cannot be declared on
# circumstantial evidence alone. Weights are sum-normalised at scoring time.
SIGNATURES: Dict[str, Dict[str, int]] = {
    # Misfire and injector clogging both cool one cylinder and both raise the
    # 0.5 order, so EGT direction cannot separate them: at a large flow loss the
    # starved cylinder is pushed lean of peak EGT and runs cold, exactly like a
    # misfire. Total fuel flow is what separates them - a misfiring cylinder is
    # still being fuelled, a clogged one is not.
    # A misfiring cylinder is still being fuelled - the charge simply does not
    # burn - so total fuel flow is unchanged. A starved cylinder receives less
    # fuel, so measured flow falls. Both cool the affected cylinder and both
    # raise the 0.5 order, which is why EGT direction cannot separate them and
    # delivered fuel can.
    "misfire": {
        "egt_one_low": +2, "vib_half_high": +2, "ve_low": -2,
        "cht_one_high": -1, "power_low": +1, "vib_1x_high": -1,
        "cooling_capacity_low": -1,
    },
    "injector_clog": {
        "egt_one_deviant": +2, "ve_low": +2, "power_low": +1,
        "vib_half_high": +1, "vib_1x_high": -1, "boost_low": -2,
        "cooling_capacity_low": -1, "cht_vs_model_high": -1,
    },
    "injector_leak": {
        "egt_one_low": +1, "fuel_flow_high": +1, "vib_half_high": +1,
        "cht_one_high": -1,
    },
    "cooling_degradation": {
        "cooling_capacity_low": +2, "cht_all_high": +1, "coolant_high": +1,
        "oil_temp_high": +1, "knock_high": +1, "cht_one_high": -1,
        "egt_one_low": -1, "vib_1x_high": -1,
    },
    "oil_cooler_block": {
        "oil_temp_high": +2, "oil_press_low": +1, "coolant_high": -1,
        "cooling_capacity_low": -1, "vib_1x_high": -1,
    },
    "oil_pump_wear": {
        "oil_press_low": +2, "oil_temp_high": +1, "vib_1x_high": -1,
        "cht_all_high": -1, "coolant_high": -1, "ve_low": -1,
    },
    "bearing_wear": {
        "vib_1x_high": +2, "oil_press_low": +1, "oil_temp_high": +1,
        "power_low": +1, "vib_half_high": -1, "egt_one_low": -1,
    },
    # Ring wear and a loss of boost both reduce breathing. The blowby term is
    # inferred from the breathing index rather than measured, so it is weighted
    # low - it would otherwise count the same evidence twice.
    # Ring wear reduces breathing on *every* cylinder, so it must be rejected
    # whenever the loss is localised to one - otherwise it absorbs every
    # single-cylinder fault as a plausible whole-engine explanation.
    "ring_wear": {
        "ve_low": +2, "power_low": +1, "blowby_high": +1,
        "egt_one_deviant": -2, "vib_half_high": -1, "boost_low": -2,
        "cooling_capacity_low": -1, "cht_vs_model_high": -1,
    },
    "turbo_degradation": {
        "boost_low": +2, "iat_high": +1, "ve_low": +1, "power_low": +1,
        "vib_half_high": -1, "oil_press_low": -1,
        "cooling_capacity_low": -1, "egt_one_deviant": -1,
    },
    "detonation": {
        "knock_high": +2, "vib_broad_high": +1, "cht_all_high": +1,
        "power_low": +1, "vib_half_high": -1, "vib_1x_high": -1,
        "fuel_flow_low": -1,
    },
    # An instrument fault shows a channel disagreeing with everything that
    # should corroborate it, and *nothing else* moving. The negatives carry as
    # much weight as the positives here: it is their absence that distinguishes
    # a lying thermocouple from a genuinely hot cylinder.
    "sensor_fault": {
        "cht_vs_model_high": +2, "single_channel_resid": +2,
        "channel_frozen": +1, "cht_one_high": +1,
        "egt_one_deviant": -2, "vib_half_high": -1, "vib_1x_high": -1,
        "power_low": -1, "ve_low": -1, "cooling_capacity_low": -1,
        "oil_press_low": -1,
    },
    "alternator_fault": {
        "volt_low": +1, "vib_1x_high": -1, "oil_press_low": -1,
        "cht_all_high": -1,
    },
}

FAULT_ACTIONS: Dict[str, str] = {
    "misfire": "Borescope and replace the spark plug / ignition lead on the identified cylinder; verify coil primary resistance.",
    "injector_clog": "Remove and ultrasonically clean the identified injector; flow-match the set before refit.",
    "injector_leak": "Replace the identified injector; check rail pressure regulator.",
    "cooling_degradation": "Inspect and clean the radiator core and cowl ducting; check coolant level, cap pressure and thermostat.",
    "oil_cooler_block": "Clean the oil cooler matrix; verify the oil thermostat bypass.",
    "oil_pump_wear": "Check oil pump end-float and relief valve; inspect the pickup screen.",
    "bearing_wear": "Sample and spectro-analyse oil for bearing metals; schedule bottom-end inspection.",
    "ring_wear": "Perform a differential compression and leak-down check; measure crankcase blowby.",
    "turbo_degradation": "Inspect compressor wheel and shaft play; check wastegate linkage and intercooler for fouling.",
    "detonation": "Verify fuel grade and octane; check charge temperature and ignition timing; reduce boost until rectified.",
    "sensor_fault": "Verify the identified channel against its redundant source; check connector, shield and cold-junction compensation.",
    "alternator_fault": "Check alternator drive belt, regulator output and field winding continuity.",
}


def _soft(value: float, on: float, full: float) -> float:
    """Continuous membership: 0 below `on`, 1 at or above `full`."""
    if full == on:
        return 1.0 if value >= on else 0.0
    x = (value - on) / (full - on)
    return max(0.0, min(1.0, x))


@dataclass
class Hypothesis:
    fault: str
    confidence: float
    evidence: List[str] = field(default_factory=list)
    against: List[str] = field(default_factory=list)
    cylinder: Optional[int] = None
    channel: Optional[str] = None
    action: str = ""

    def to_dict(self) -> dict:
        return {"fault": self.fault, "confidence": round(self.confidence, 3),
                "cylinder": self.cylinder, "channel": self.channel,
                "evidence": self.evidence, "against": self.against,
                "action": self.action}


class FaultIsolator:
    def __init__(self, n_cyl: int = 4):
        self.n_cyl = n_cyl
        self.last_symptoms: Dict[str, float] = {}
        self.last_hypotheses: List[Hypothesis] = []

    # -- symptoms ------------------------------------------------------------
    def symptoms(self, ind: Dict[str, float], meas: Dict[str, Optional[float]],
                 resid: Dict[str, float], extras: Dict[str, float]) -> Dict[str, float]:
        n = self.n_cyl
        egt = [meas.get(f"egt{i}_c") for i in range(1, n + 1)]
        cht = [meas.get(f"cht{i}_c") for i in range(1, n + 1)]
        egt_v = [(i, v) for i, v in enumerate(egt, 1) if v is not None and v > 0]
        cht_v = [(i, v) for i, v in enumerate(cht, 1) if v is not None and v > -50]

        s: Dict[str, float] = {k: 0.0 for k in SYMPTOMS}
        self.worst_cyl_egt_low = self.worst_cyl_egt_high = self.worst_cyl_cht = None

        if len(egt_v) >= 3:
            m = sum(v for _, v in egt_v) / len(egt_v)
            lo_i, lo_v = min(egt_v, key=lambda t: t[1])
            hi_i, hi_v = max(egt_v, key=lambda t: t[1])
            s["egt_one_low"] = _soft(m - lo_v, 38.0, 120.0)
            s["egt_one_high"] = _soft(hi_v - m, 38.0, 120.0)
            s["egt_one_deviant"] = max(s["egt_one_low"], s["egt_one_high"])
            s["egt_all_high"] = _soft(m, 820.0, 900.0)
            self.worst_cyl_egt_low, self.worst_cyl_egt_high = lo_i, hi_i
        if len(cht_v) >= 3:
            m = sum(v for _, v in cht_v) / len(cht_v)
            hi_i, hi_v = max(cht_v, key=lambda t: t[1])
            s["cht_one_high"] = _soft(hi_v - m, 8.0, 30.0)
            s["cht_all_high"] = _soft(m, 118.0, 135.0)
            self.worst_cyl_cht = hi_i

        s["coolant_high"] = _soft(ind.get("coolant_temp_c", 90.0), 105.0, 120.0)
        s["oil_temp_high"] = _soft(ind.get("oil_temp_c", 90.0), 115.0, 130.0)
        s["oil_press_low"] = _soft(1.0 - ind.get("oil_press_index", 1.0), 0.05, 0.35)
        s["vib_half_high"] = _soft(ind.get("misfire_order_g", 0.0), 0.14, 0.85)
        s["vib_1x_high"] = _soft(ind.get("vib_1x_g", 0.0), 0.55, 1.7)
        s["vib_broad_high"] = _soft(ind.get("vib_broadband_g", 0.0), 0.80, 2.0)
        ffi = ind.get("fuel_flow_index", 1.0)
        s["fuel_flow_high"] = _soft(ffi - 1.0, 0.04, 0.25)
        s["fuel_flow_low"] = _soft(1.0 - ffi, 0.04, 0.25)
        s["ve_low"] = _soft(1.0 - ind.get("ve_index", 1.0), 0.09, 0.28)
        s["boost_low"] = _soft(-min(0.0, ind.get("boost_error_kpa", 0.0)), 2.5, 18.0)
        s["iat_high"] = _soft(extras.get("iat_model_error_k", 0.0), 5.0, 28.0)
        s["knock_high"] = _soft(ind.get("knock_index", 0.0), 0.10, 0.55)
        s["cooling_capacity_low"] = _soft(1.0 - ind.get("cooling_effectiveness", 1.0),
                                          0.10, 0.60)
        s["cht_vs_model_high"] = _soft(extras.get("cht_max_model_dev_c", 0.0), 9.0, 32.0)
        s["power_low"] = _soft(1.0 - extras.get("power_index", 1.0), 0.08, 0.30)
        s["volt_low"] = _soft(13.0 - ind.get("bus_v", 13.8), 0.4, 1.6)
        s["blowby_high"] = _soft(extras.get("blowby_index", 1.0) - 1.0, 0.35, 2.0)

        # A single channel far outside its innovation bound while the rest are
        # clean is the signature of an instrument fault rather than an engine
        # fault - the twin's most valuable discrimination.
        if resid:
            items = sorted(resid.items(), key=lambda kv: -abs(kv[1]))
            top_name, top = items[0]
            others = [abs(v) for _, v in items[1:]]
            rest = (sum(others) / len(others)) if others else 0.0
            s["single_channel_resid"] = _soft(abs(top) - 2.2 * rest, 1.8, 6.0)
            self.suspect_channel = top_name if s["single_channel_resid"] > 0.15 else None
        else:
            self.suspect_channel = None
        s["channel_frozen"] = _soft(extras.get("frozen_channels", 0.0), 0.5, 2.0)

        self.last_symptoms = s
        return s

    # -- isolation ------------------------------------------------------------
    def isolate(self, s: Dict[str, float], min_conf: float = 0.18) -> List[Hypothesis]:
        hyps: List[Hypothesis] = []
        for fault, sig in SIGNATURES.items():
            pos = {k: v for k, v in sig.items() if v > 0}
            neg = {k: -v for k, v in sig.items() if v < 0}
            if not pos:
                continue
            wp = sum(pos.values())
            support = sum(s.get(k, 0.0) * w for k, w in pos.items()) / wp
            wn = sum(neg.values())
            conflict = (sum(s.get(k, 0.0) * w for k, w in neg.items()) / wn) if wn else 0.0
            conf = support * (1.0 - 0.75 * conflict)
            if conf < min_conf:
                continue
            h = Hypothesis(
                fault=fault, confidence=conf,
                evidence=[f"{SYMPTOMS[k]} ({s.get(k,0):.0%})" for k in pos if s.get(k, 0) > 0.2],
                against=[f"{SYMPTOMS[k]} ({s.get(k,0):.0%})" for k in neg if s.get(k, 0) > 0.3],
                action=FAULT_ACTIONS.get(fault, ""))
            if fault == "misfire" or fault == "injector_leak":
                h.cylinder = getattr(self, "worst_cyl_egt_low", None)
            elif fault == "injector_clog":
                h.cylinder = getattr(self, "worst_cyl_egt_high", None)
            elif fault == "sensor_fault":
                h.channel = (getattr(self, "suspect_channel", None)
                             or (f"cht{self.worst_cyl_cht}_c"
                                 if s.get("cht_vs_model_high", 0) > 0.3
                                 and self.worst_cyl_cht else None))
            hyps.append(h)

        # Normalise into a comparable ranking without forcing them to sum to 1:
        # "nothing is wrong" must remain a possible outcome.
        hyps.sort(key=lambda h: -h.confidence)
        self.last_hypotheses = hyps[:5]
        return self.last_hypotheses

    def report(self) -> dict:
        return {"symptoms": {k: round(v, 3) for k, v in self.last_symptoms.items() if v > 0.05},
                "hypotheses": [h.to_dict() for h in self.last_hypotheses]}
