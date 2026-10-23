"""Deterministic combination of independent visual and verified text evidence."""
from copy import deepcopy
import math
from .policy import decide, DecisionPolicy, _rank
from .threat_rule import threat_rule
from src.labels import TYPE_CLASSES, INJURY_CLASSES, RISK_CLASSES


def verified_text(semantic, text):
    quote = semantic.get("quote", "")
    return (semantic.get("status") == "ok" and semantic.get("verdict") == "violation"
            and isinstance(quote, str) and len(quote.strip()) >= 3 and quote in text)


def combine(evidence, policy=None, vision=None, semantic=None, text="", text_gate=False):
    policy = policy or DecisionPolicy()
    filtered = deepcopy(evidence)
    scores = evidence.get("type", {}).get("scores", {})
    binary = evidence.get("blood_binary") or {}
    binary_negative = False
    if binary.get("trained"):
        score, threshold = binary.get("p_blood"), binary.get("threshold")
        if isinstance(score, (int, float)) and isinstance(threshold, (int, float)):
            binary_negative = math.isfinite(score) and 0 <= score < threshold <= 1
    if binary_negative:
        filtered["injury"] = {"scores": {label: float(i == 0) for i, label in enumerate(INJURY_CLASSES)}}
        if scores.get(TYPE_CLASSES[2], 0) < .9:
            filtered["type"]["labels"] = [x for x in filtered["type"].get("labels", []) if x != TYPE_CLASSES[2]]
    decision = decide(filtered, policy, vision)
    decision["base_action"] = decision["action"]
    decision["policy_version"] = "2026-10-23-local-evidence-v1"
    decision["binary_negative"] = binary_negative
    original_rank = _rank(evidence.get("injury", {}), INJURY_CLASSES)
    if original_rank:
        decision["injury_candidate"].update(label=original_rank[0][1], score=original_rank[0][0])
        if binary_negative:
            decision["injury_candidate"]["accepted"] = bool(vision and vision.get("status") == "ok" and vision.get("verdict") == "supported")
    valid = (evidence.get("trained") and _rank(evidence.get("risk", {}), RISK_CLASSES)
             and original_rank and _rank(evidence.get("type", {}), TYPE_CLASSES))
    decision.update(review_hint=False, review_reason="", text_violation=False, rule_id=None)
    if not valid:
        return decision
    strong = [name for name, threshold in zip(TYPE_CLASSES[1:], [.5, .9, .7])
              if scores[name] >= threshold and scores[name] >= scores[TYPE_CLASSES[0]]]
    if decision["action"] == "allow" and strong:
        decision.update(action="limit", uncertain=False, category="、".join(strong),
                        reason="独立暴力类型分数达到处置门槛，采用限流")
    text_hit = text_gate and verified_text(semantic or {}, text)
    rule = threat_rule(text) if text_gate else None
    decision.update(text_violation=bool(text_hit), rule_id=rule["rule_id"] if rule else None)
    if text_hit or rule:
        if decision["action"] == "allow":
            decision["action"] = "limit"
        decision.update(uncertain=False, reason="图中文字构成人身威胁，按独立文字证据处置")
        decision["category"] = "文字威胁" if decision["category"] == "无达到处置门槛的违规类别" else decision["category"] + "、文字威胁"
    return decision


def weak_weapon_gate(evidence, decision):
    scores = evidence.get("type", {}).get("scores", {})
    if not _rank(evidence.get("type", {}), TYPE_CLASSES):
        return False
    none, direct, aftermath, weapon = (scores[x] for x in TYPE_CLASSES)
    return (decision["action"] != "allow" and decision["base_action"] == "allow"
            and not decision["text_violation"] and not decision["rule_id"]
            and .7 <= weapon < .9 and none >= .45 and direct < .5 and aftermath < .9)
