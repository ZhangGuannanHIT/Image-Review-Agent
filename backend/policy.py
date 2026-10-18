"""演示阶段处置规则；阈值是业务门槛，不是经过标定的准确率保证。"""
from dataclasses import dataclass
import math

from src.labels import RISK_CLASSES, INJURY_CLASSES, TYPE_CLASSES


@dataclass(frozen=True)
class DecisionPolicy:
    min_injury_confidence: float = 0.75
    min_injury_margin: float = 0.20

    def __post_init__(self):
        for value in (self.min_injury_confidence, self.min_injury_margin):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("处置门槛必须位于 [0, 1]")


def _rank(task, classes):
    scores = task.get("scores", {})
    if set(scores) != set(classes):
        return []
    values = [float(scores[label]) for label in classes]
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        return []
    return sorted(zip(values, classes), reverse=True)


def decide(evidence: dict, policy: DecisionPolicy, vision: dict | None = None) -> dict:
    """风险保留原始 argmax；类型沿用部署阈值，不确定按用户策略放行。"""
    risk_rank = _rank(evidence.get("risk", {}), RISK_CLASSES)
    injury_rank = _rank(evidence.get("injury", {}), INJURY_CLASSES)
    risk = risk_rank[0][1] if risk_rank else None
    injury = None
    if (injury_rank and injury_rank[0][0] >= policy.min_injury_confidence
            and injury_rank[0][0] - injury_rank[1][0] >= policy.min_injury_margin):
        injury = injury_rank[0][1]
    types = evidence.get("type", {})
    scores, thresholds = types.get("scores", {}), types.get("thresholds", {})
    # labels 由引擎在未四舍五入的概率上按逐类阈值产生，保留该边界判定。
    selected = [name for name in types.get("labels", [])
                if name in TYPE_CLASSES and name in thresholds]
    active = [name for name in selected if name != TYPE_CLASSES[0]]
    conflict = TYPE_CLASSES[0] in selected and TYPE_CLASSES[1] in selected
    category = "、".join(active + ([injury] if injury and injury != INJURY_CLASSES[0] else []))
    action, uncertain = "allow", True
    reason = "风险候选缺少达到门槛的内容类别支持，不足以据此限制内容"
    if not evidence.get("trained"):
        reason = "分类头未训练，预测不能用于违规处置"
    elif not risk_rank or not injury_rank:
        reason = "分类分数缺失或无效，无法完成证据核对"
    elif conflict:
        reason = "无正在进行的暴力与攻击进行中同时命中，行为证据存在分歧"
    elif risk == RISK_CLASSES[4]:
        reason = "风险头最高候选为 X，无法确定内容等级"
    elif risk == RISK_CLASSES[0]:
        if active or (injury and injury != INJURY_CLASSES[0]):
            reason = "风险头倾向安全，但内容头有达到门槛的违规候选，暂不据此升级处置"
        else:
            uncertain = False
            reason = "风险头最高候选为安全，内容类别未提供足以升级处置的证据"
    elif risk in RISK_CLASSES[1:3] and category:
        action, uncertain = "limit", False
        reason = "风险头倾向轻中度违规，且暴力类型或血腥类别提供支持"
    elif risk == RISK_CLASSES[3] and (injury == INJURY_CLASSES[3] or TYPE_CLASSES[1] in active):
        action, uncertain = "block", False
        reason = "风险头倾向严重违规，且严重血腥或攻击进行中提供支持"
    elif risk == RISK_CLASSES[3] and category:
        action, uncertain = "limit", False
        reason = "有违规类别支持，但缺少严重血腥或攻击进行中的证据，采用限流而非下架"
    review_level = None
    if vision and vision.get("trigger", {}).get("eligible"):
        if vision.get("status") == "ok" and vision.get("verdict") == "not_supported":
            injury = None
            category = "、".join(active)
            if not active and not conflict:
                action, uncertain, review_level = "allow", False, RISK_CLASSES[0]
                reason = "视觉复核未发现血腥，且无独立暴力类型命中"
            else:
                reason += "；血腥复核为阴性，独立暴力类型判断保留"
        elif vision.get("status") == "ok" and vision.get("verdict") == "supported":
            severity = vision.get("severity")
            index = {"mild": 1, "moderate": 2, "severe": 3}.get(severity)
            if index is not None:
                injury = INJURY_CLASSES[index]
                category = "、".join(active + [injury])
                if action != "block":
                    action = "block" if index == 3 else "limit"
                uncertain, review_level = False, RISK_CLASSES[3 if action == "block" else index]
                reason = "视觉复核支持血腥，按复核程度处置，并保留独立暴力判断"
        elif not active:
            action, uncertain = "allow", True
            if vision.get("status") == "ok":
                reason = "视觉复核未得出明确结论"
            elif vision.get("stage") == "response_validation" or vision.get("error") == "ValidationError":
                reason = "视觉复核返回格式不符合要求，未取得有效的血腥复核结论"
            else:
                reason = "视觉复核请求未完成，未取得有效的血腥复核结论"
    if uncertain:
        reason += "；按不确定放行策略处理"
    return {"policy_version": "2026-10-18-v3", "action": action,
            "level": review_level or risk or "无有效预测", "uncertain": uncertain,
            "vision_review": vision,
            "category": category or "无达到处置门槛的违规类别", "reason": reason,
            "risk_candidate": {"label": risk, "score": risk_rank[0][0] if risk_rank else None},
            "injury_candidate": {"label": injury_rank[0][1] if injury_rank else None,
                                 "score": injury_rank[0][0] if injury_rank else None,
                                 "accepted": injury is not None},
            "type_candidates": [{"label": name, "score": scores.get(name),
                                 "threshold": thresholds.get(name)} for name in selected]}


def render(decision: dict) -> str:
    action = {"allow": "放行", "limit": "限流", "block": "下架"}[decision["action"]]
    def candidate(item):
        if item["label"] is None:
            return "无有效预测"
        return f"{item['label']}（分数 {item['score']:.3f}）"
    risk = candidate(decision["risk_candidate"])
    injury = candidate(decision["injury_candidate"])
    types = "、".join(f"{x['label']}（{x['score']:.3f}）" for x in decision["type_candidates"])
    status = "证据不确定，暂不认定违规" if decision["uncertain"] else "已生成处置建议"
    review = decision.get("vision_review")
    review_text = ""
    if review and review.get("trigger", {}).get("eligible"):
        label = {"supported": "支持血腥", "not_supported": "未发现血腥", "uncertain": "无法判断"}.get(review.get("verdict"), "无法判断")
        if review.get("status") == "ok":
            if review.get("verdict") == "supported":
                label = {"mild": "轻微血腥", "moderate": "中度血腥", "severe": "严重血腥"}.get(review.get("severity"), "支持血腥")
            review_text = f"血腥复核：{label}。"
    return (f"审核结果：{status}；处置：{action}。"
            f"CLIP 风险候选：{risk}；暴力类型：{types or '无命中标签'}；血腥候选：{injury}。"
            f"{review_text}依据：{decision['reason']}。")
