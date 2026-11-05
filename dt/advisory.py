"""
Autonomous maintenance advisory.

Turns health indices, RUL and ranked fault hypotheses into ordered, actionable
work items with a stated urgency and the evidence that produced them. Advisories
carry a mission-impact judgement as well as a maintenance one, because the two
decisions - "can this sortie continue" and "what does the line crew do tonight"
- are made by different people from the same data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .config import Limits

URGENCY_ORDER = {"ABORT": 0, "IMMEDIATE": 1, "BEFORE_NEXT_FLIGHT": 2,
                 "SCHEDULED": 3, "MONITOR": 4}


@dataclass
class Advisory:
    urgency: str
    subsystem: str
    title: str
    detail: str
    action: str
    evidence: List[str] = field(default_factory=list)
    due_hours: Optional[float] = None
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {"urgency": self.urgency, "subsystem": self.subsystem,
                "title": self.title, "detail": self.detail, "action": self.action,
                "evidence": self.evidence,
                "due_hours": None if self.due_hours is None else round(self.due_hours, 1),
                "confidence": round(self.confidence, 2)}


class AdvisoryEngine:
    def __init__(self, limits: Optional[Limits] = None, oil_change_hours: float = 100.0):
        self.limits = limits or Limits()
        self.oil_change_hours = oil_change_hours

    def generate(self, health: dict, rul: dict, hypotheses: List[dict],
                 exceedances: List[dict], degradation: dict,
                 mission_hours_remaining: float = 6.0) -> List[Advisory]:
        adv: List[Advisory] = []
        idx = health.get("subsystems", {})

        # 1. Certified-limit exceedances always outrank predictive findings.
        for ex in exceedances:
            if ex["severity"] == "limit":
                adv.append(Advisory(
                    "ABORT" if ex.get("flight_critical") else "IMMEDIATE",
                    ex["subsystem"],
                    f"{ex['channel']} exceeded its certified limit",
                    f"{ex['channel']} = {ex['value']:.1f} {ex['unit']} against a limit of "
                    f"{ex['limit']:.1f} {ex['unit']}.",
                    "Reduce power and, if the exceedance persists, terminate the sortie "
                    "and recover to the nearest suitable field.",
                    [f"sustained for {ex.get('duration_s', 0):.0f} s"], 0.0, 1.0))

        # 2. Fault hypotheses above a usable confidence.
        #
        # Only the leading explanation and anything genuinely competitive with it
        # are turned into advisories. Faults with overlapping signatures always
        # produce a tail of near-miss hypotheses, and their cylinder attribution
        # flickers between cylinders as the temperatures move; promoting all of
        # them fills the operator's display with contradictory work orders for
        # what is one fault.
        top = hypotheses[0]["confidence"] if hypotheses else 0.0
        for h in hypotheses[:2]:
            c = h["confidence"]
            if c < 0.30 or c < 0.88 * top:
                continue
            where = ""
            if h.get("cylinder"):
                where = f" on cylinder {h['cylinder']}"
            elif h.get("channel"):
                where = f" on channel {h['channel']}"
            urgency = ("IMMEDIATE" if c > 0.75 else
                       "BEFORE_NEXT_FLIGHT" if c > 0.5 else "SCHEDULED")
            if h["fault"] in ("sensor_fault",):
                urgency = "BEFORE_NEXT_FLIGHT" if c > 0.6 else "SCHEDULED"
            adv.append(Advisory(
                urgency, _subsystem_of(h["fault"]),
                f"Probable {h['fault'].replace('_', ' ')}{where}",
                f"Structured-residual isolation ranks this hypothesis at {c:.0%} "
                f"confidence against the observed symptom pattern.",
                h.get("action", ""), h.get("evidence", [])[:4], None, c))

        # 3. RUL-driven scheduling.
        eng = rul.get("engine", {})
        for sub, r in rul.get("subsystems", {}).items():
            rh = r.get("rul_hours")
            if rh is None:
                continue
            # A trend seen for only a few minutes must not abort a sortie. The
            # urgency a RUL estimate can command is capped by its confidence.
            conf = r.get("confidence", 0.0)
            if conf < 0.35 and rh < 25:
                adv.append(Advisory(
                    "MONITOR", sub, f"{sub.replace('_',' ').title()} trending down",
                    f"Projected {rh:.1f} h to threshold, but the trend has only "
                    f"{conf:.0%} confidence; continuing to observe.",
                    _rul_action(sub),
                    [f"health index {r['level']:.1f}, "
                     f"{r['slope_per_hour']:+.4f} pts/h"], rh, conf))
                continue
            if rh <= mission_hours_remaining:
                urgency = "ABORT"
                detail = (f"{sub} is projected to reach its action threshold "
                          f"({r['threshold']:.0f}) in {rh:.1f} h, inside the "
                          f"{mission_hours_remaining:.1f} h remaining on this sortie.")
            elif rh < 10:
                urgency, detail = "IMMEDIATE", f"{sub} has {rh:.1f} h of usable life remaining."
            elif rh < 25:
                urgency, detail = "BEFORE_NEXT_FLIGHT", f"{sub} has {rh:.1f} h remaining."
            elif rh < 80:
                urgency, detail = "SCHEDULED", f"{sub} has {rh:.1f} h remaining."
            else:
                continue
            band = ""
            if r.get("lower_hours") is not None:
                band = f" (90 % band {r['lower_hours']:.0f}-{r['upper_hours']:.0f} h)"
            adv.append(Advisory(
                urgency, sub, f"{sub.replace('_', ' ').title()} life limit approaching",
                detail + band, _rul_action(sub),
                [f"health index {r['level']:.1f}, trending {r['slope_per_hour']:+.4f} pts/h"],
                rh, r.get("confidence", 0.0)))

        # 4. Calendar/hours-based servicing that needs no fault at all.
        oil_left = self.oil_change_hours - degradation.get("oil_hours", 0.0)
        if oil_left < 15:
            adv.append(Advisory(
                "BEFORE_NEXT_FLIGHT" if oil_left < 5 else "SCHEDULED", "lubrication",
                "Oil and filter change due",
                f"{degradation.get('oil_hours', 0):.1f} h on the current oil charge "
                f"against a {self.oil_change_hours:.0f} h interval.",
                "Drain, replace filter, refill and sample the used oil for spectrographic analysis.",
                [], max(0.0, oil_left), 1.0))

        # 5. Watch items: subsystems that are down but not yet trending.
        for sub, v in idx.items():
            if v < 80 and not any(a.subsystem == sub for a in adv):
                adv.append(Advisory(
                    "MONITOR", sub, f"{sub.replace('_', ' ').title()} health index reduced",
                    f"Index at {v:.1f} with no significant trend yet established.",
                    "Continue monitoring; re-assess after the next sortie.",
                    [], None, health.get("confidence", {}).get(sub, 0.0)))

        adv.sort(key=lambda a: (URGENCY_ORDER.get(a.urgency, 9), -a.confidence))
        return adv

    def mission_recommendation(self, adv: List[Advisory], health: dict) -> dict:
        if any(a.urgency == "ABORT" for a in adv):
            return {"decision": "ABORT",
                    "rationale": "A certified limit has been exceeded or a subsystem is "
                                 "projected to fail within the remaining sortie time.",
                    "colour": "red"}
        if any(a.urgency == "IMMEDIATE" for a in adv):
            return {"decision": "CURTAIL",
                    "rationale": "Return to base at reduced power; a fault requiring "
                                 "immediate rectification has been isolated.",
                    "colour": "amber"}
        if health.get("overall", 100) < 75:
            return {"decision": "CONTINUE_WITH_RESTRICTION",
                    "rationale": "Continue the sortie but avoid sustained maximum "
                                 "continuous power and monitor the affected subsystem.",
                    "colour": "amber"}
        return {"decision": "CONTINUE",
                "rationale": "All subsystems within limits with no significant "
                             "degradation trend.",
                "colour": "green"}


def _subsystem_of(fault: str) -> str:
    m = {"misfire": "combustion", "detonation": "combustion",
         "injector_clog": "fuel_injection", "injector_leak": "fuel_injection",
         "cooling_degradation": "cooling", "oil_cooler_block": "cooling",
         "oil_pump_wear": "lubrication", "bearing_wear": "mechanical",
         "ring_wear": "mechanical", "turbo_degradation": "air_path",
         "sensor_fault": "sensing", "alternator_fault": "electrical"}
    return m.get(fault, "mechanical")


def _rul_action(sub: str) -> str:
    return {
        "combustion": "Plan an ignition-system service: plugs, leads and coil check.",
        "air_path": "Plan a turbocharger and induction-system inspection.",
        "fuel_injection": "Plan injector removal, cleaning and flow matching.",
        "cooling": "Plan a cooling-system service: core cleaning, coolant renewal, thermostat check.",
        "lubrication": "Plan an oil-system service and pump/relief-valve inspection.",
        "mechanical": "Plan a compression and leak-down check with oil spectrographic analysis.",
        "electrical": "Plan an alternator and battery capacity check.",
        "sensing": "Plan sensor calibration and harness inspection.",
    }.get(sub, "Plan a scheduled inspection of the affected subsystem.")
