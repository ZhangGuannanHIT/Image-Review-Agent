"""演示阶段处置规则；阈值是业务门槛，不是经过标定的准确率保证。"""
from dataclasses import dataclass
import math

from src.labels import RISK_CLASSES, INJURY_CLASSES, TYPE_CLASSES, QUALITY_CLASSES


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
    """风险保留原始 argmax；类型沿用部署阈值，高不确定性进入人工复核。"""
    risk_rank = _rank(evidence.get("risk", {}), RISK_CLASSES)
    injury_rank = _rank(evidence.get("injury", {}), INJURY_CLASSES)
    risk = risk_rank[0][1] if risk_rank else None
    injury = None
    if (injury_rank and injury_rank[0][0] >= policy.min_injury_confidence
            and injury_rank[0][0] - injury_rank[1][0] >= policy.min_injury_margin):
        injury = injury_rank[0][1]
    binary = evidence.get("blood_binary")
    binary_supported = None
    binary_reason = ""
    if isinstance(binary, dict):
        try:
            binary_score = float(binary["p_blood"])
            binary_threshold = float(binary["threshold"])
        except (KeyError, TypeError, ValueError):
            binary_score = binary_threshold = None
        if (
            binary_score is not None
            and binary_threshold is not None
            and math.isfinite(binary_score)
            and math.isfinite(binary_threshold)
            and 0 <= binary_score <= 1
            and 0 <= binary_threshold <= 1
        ):
            binary_supported = binary_score >= binary_threshold
            if not binary_supported and injury not in {None, INJURY_CLASSES[0]}:
                injury = None
                binary_reason = "二分类血腥头未达到校准阈值，已否决四分类血腥候选"
            elif binary_supported:
                binary_reason = "二分类血腥头达到校准阈值"
    types = evidence.get("type", {})
    scores, thresholds = types.get("scores", {}), types.get("thresholds", {})
    quality = evidence.get("quality", {})
    quality_labels = [
        label for label in quality.get("labels", []) if label in QUALITY_CLASSES
    ]
    quality_review_labels = {
        "严重模糊", "文字扭曲形变", "故意遮挡（马赛克/黑条/贴纸）",
    }
    quality_needs_review = bool(set(quality_labels) & quality_review_labels)
    text_evidence = evidence.get("text_evidence", {}) or {}
    scene_evidence = evidence.get("scene_concepts", {}) or {}
    knowledge_hits = evidence.get("search_knowledge", []) or []
    # labels 由引擎在未四舍五入的概率上按逐类阈值产生，保留该边界判定。
    selected = [name for name in types.get("labels", [])
                if name in TYPE_CLASSES and name in thresholds]
    active = [name for name in selected if name != TYPE_CLASSES[0]]
    binary_conflict = (
        binary_supported is False and TYPE_CLASSES[2] in active
    )
    independent_active = [
        name for name in active
        if name != TYPE_CLASSES[2] or binary_supported is not False
    ]
    conflict = TYPE_CLASSES[0] in selected and TYPE_CLASSES[1] in selected
    category = "、".join(
        independent_active
        + ([injury] if injury and injury != INJURY_CLASSES[0] else [])
    )
    action, uncertain = "allow", True
    reason = "风险候选缺少达到门槛的内容类别支持，不足以据此限制内容"
    if not evidence.get("trained"):
        reason = "分类头未训练，预测不能用于违规处置"
    elif not risk_rank or not injury_rank:
        reason = "分类分数缺失或无效，无法完成证据核对"
    elif conflict:
        reason = "无正在进行的暴力与攻击进行中同时命中，行为证据存在分歧"
    elif risk == RISK_CLASSES[4]:
        action, uncertain = "review", False
        reason = "风险头最高候选为 X，图片内容无法可靠判定，需要人工复核"
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
    if binary_conflict and not independent_active:
        if risk == RISK_CLASSES[0]:
            action, uncertain = "allow", False
            reason = "二分类血腥头未支持血腥，且无独立暴力类型命中"
        else:
            action, uncertain = "review", False
            reason = "二分类血腥头与风险/后果展示候选冲突，需要人工复核"
    if quality_needs_review:
        reason += "；图像质量退化只作为证据可靠性提示，不单独改变处置动作"
    text_status = text_evidence.get("status")
    text_high_risk = bool(text_evidence.get("high_risk_rule_ids"))
    text_safe_context = bool(text_evidence.get("safe_context_rule_ids"))
    image_risk_indicator = bool(
        active or binary_supported or risk in RISK_CLASSES[1:4]
    )
    scene_safe_context = bool(scene_evidence.get("safe_context_matched"))
    scene_safe_keys = {
        str(item.get("key", ""))
        for item in scene_evidence.get("safe_contexts", [])
    }
    safe_context_confirmed = bool(
        scene_safe_context and text_safe_context and image_risk_indicator
    )
    editorial_mitigation = bool(
        {"严重模糊", "故意遮挡（马赛克/黑条/贴纸）"} & set(quality_labels)
    )
    news_shell_suspect = bool(
        safe_context_confirmed
        and "news_legal_context" in scene_safe_keys
        and not editorial_mitigation
        and (
            risk == RISK_CLASSES[3]
            or injury == INJURY_CLASSES[3]
        )
    )
    unreadable_text_conflict = bool(
        "文字扭曲形变" in quality_labels
        and text_status in {"empty", "partial", "failed"}
        and risk in RISK_CLASSES[1:4]
    )
    if news_shell_suspect:
        action, uncertain = "review", False
        reason = "新闻语境与未加工的血腥暴力画面冲突，疑似新闻壳绕过，需要人工复核"
    elif safe_context_confirmed and action != "review":
        action, uncertain = "limit", False
        reason = "场景 Prompt 与 OCR/RAG 共同确认新闻/医学等安全语境，但视觉仍存在违规候选，采用限流而非下架或放行"
    elif unreadable_text_conflict and action == "allow":
        action, uncertain = "review", False
        reason = "quality 命中文字扭曲，但 OCR 无文字或文本残缺，且风险头非安全，需要人工复核"
    elif text_high_risk and action == "allow":
        action, uncertain = "limit", False
        reason = "OCR 文本命中明确高风险规则，图文证据需要限制传播"
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
        action = "review"
        reason += "；证据存在高不确定性，进入人工复核"
    if binary_reason:
        reason += f"；{binary_reason}"
    return {"policy_version": "2026-10-23-v4", "action": action,
            "level": review_level or risk or "无有效预测", "uncertain": uncertain,
            "vision_review": vision,
            "category": category or "无达到处置门槛的违规类别", "reason": reason,
            "risk_candidate": {"label": risk, "score": risk_rank[0][0] if risk_rank else None},
            "injury_candidate": {"label": injury_rank[0][1] if injury_rank else None,
                                 "score": injury_rank[0][0] if injury_rank else None,
                                 "accepted": injury is not None},
            "blood_binary_candidate": {
                "score": binary.get("p_blood") if isinstance(binary, dict) else None,
                "threshold": binary.get("threshold") if isinstance(binary, dict) else None,
                "supported": binary_supported,
            },
            "quality_candidate": {
                "labels": quality_labels,
                "needs_review": quality_needs_review,
                "action_effect": "evidence_warning_only",
            },
            "text_evidence": text_evidence,
            "search_knowledge_hits": knowledge_hits,
            "scene_context": {
                "risk_scenes": scene_evidence.get("risk_scenes", []),
                "safe_contexts": scene_evidence.get("safe_contexts", []),
                "context_conflict": safe_context_confirmed,
                "safe_context_confirmed": safe_context_confirmed,
                "news_shell_suspect": news_shell_suspect,
                "knowledge_hits": scene_evidence.get("knowledge_hits", []),
            },
            "type_candidates": [{"label": name, "score": scores.get(name),
                                 "threshold": thresholds.get(name)} for name in selected]}


def render(decision: dict) -> str:
    action = {"allow": "放行", "limit": "限流", "block": "下架", "review": "人工复核"}[decision["action"]]
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
    binary = decision.get("blood_binary_candidate") or {}
    binary_text = ""
    if binary.get("score") is not None:
        binary_label = "支持血腥" if binary.get("supported") else "未支持血腥"
        binary_text = (
            f"血腥二分类：{binary_label}"
            f"（{binary['score']:.3f}，阈值 {binary['threshold']:.3f}）。"
        )
    quality = decision.get("quality_candidate") or {}
    quality_text = ""
    if quality.get("labels"):
        quality_text = "图像质量：" + "、".join(quality["labels"]) + "。"
    text = decision.get("text_evidence") or {}
    text_text = ""
    if text:
        status_text = {
            "not_available": "未执行",
            "failed": "失败",
            "empty": "无文字",
            "partial": "文本残缺",
            "readable": "文本可读",
        }.get(text.get("status"), "未知")
        rules = text.get("high_risk_rule_ids", []) + text.get("safe_context_rule_ids", [])
        rules_text = f"，规则 {'/'.join(rules)}" if rules else ""
        text_text = f"OCR：{status_text}{rules_text}。"
    scene = decision.get("scene_context") or {}
    scene_text = ""
    if scene.get("risk_scenes") or scene.get("safe_contexts"):
        risk_titles = "、".join(
            item.get("title", item.get("key", ""))
            for item in scene.get("risk_scenes", [])[:2]
        )
        safe_titles = "、".join(
            item.get("title", item.get("key", ""))
            for item in scene.get("safe_contexts", [])[:2]
        )
        scene_text = (
            f"场景候选：{risk_titles or '无'}；"
            f"安全语境候选：{safe_titles or '无'}。"
        )
        if scene.get("context_conflict"):
            scene_text = "场景冲突：" + scene_text.removeprefix("场景候选：")
    knowledge = decision.get("search_knowledge_hits") or []
    knowledge_text = ""
    if knowledge:
        knowledge_text = "RAG 规则：" + "、".join(
            str(item.get("id")) for item in knowledge[:5] if item.get("id")
        ) + "。"
    return (f"审核结果：{status}；处置：{action}。"
            f"CLIP 风险候选：{risk}；暴力类型：{types or '无命中标签'}；血腥候选：{injury}。"
            f"{binary_text}{quality_text}{text_text}{scene_text}{knowledge_text}{review_text}依据：{decision['reason']}。")
