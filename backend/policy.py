"""演示阶段处置规则；阈值是业务门槛，不是经过标定的准确率保证。"""
from dataclasses import dataclass
import math

from src.labels import RISK_CLASSES, INJURY_CLASSES, TYPE_CLASSES, QUALITY_CLASSES


@dataclass(frozen=True)
class DecisionPolicy:
    min_injury_confidence: float = 0.75
    min_injury_margin: float = 0.20
    # Strictly below threshold by this absolute margin is a clear binary
    # negative. Values inside the hysteresis band remain uncertain.
    binary_clear_margin: float = 0.05
    # Real images only receive the narrow L1 relaxation. L2/L3 remain
    # conservative so the AI-style policy cannot silently affect real photos.
    real_l1_relax_max_confidence: float = 0.65
    aggressive_anime_relaxation: bool = True

    def __post_init__(self):
        for value in (
            self.min_injury_confidence,
            self.min_injury_margin,
            self.binary_clear_margin,
            self.real_l1_relax_max_confidence,
        ):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("处置门槛必须位于 [0, 1]")
        if not isinstance(self.aggressive_anime_relaxation, bool):
            raise ValueError("aggressive_anime_relaxation 必须是布尔值")


def _rank(task, classes):
    scores = task.get("scores", {})
    if set(scores) != set(classes):
        return []
    values = [float(scores[label]) for label in classes]
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        return []
    return sorted(zip(values, classes), reverse=True)


def decide(evidence: dict, policy: DecisionPolicy, vision: dict | None = None) -> dict:
    """Derive an auditable effective risk without overwriting the raw head.

    The model heads are independent estimators. A clear binary negative can
    invalidate blood support, but it cannot rewrite the risk head mathematically.
    This function therefore keeps ``risk_raw`` and emits a separate policy-level
    ``effective_risk``.
    """
    risk_rank = _rank(evidence.get("risk", {}), RISK_CLASSES)
    injury_rank = _rank(evidence.get("injury", {}), INJURY_CLASSES)
    risk = risk_rank[0][1] if risk_rank else None
    risk_confidence = risk_rank[0][0] if risk_rank else None
    injury = None
    if (injury_rank and injury_rank[0][0] >= policy.min_injury_confidence
            and injury_rank[0][0] - injury_rank[1][0] >= policy.min_injury_margin):
        injury = injury_rank[0][1]
    binary = evidence.get("blood_binary")
    binary_supported = None
    binary_state = "unknown"
    binary_score = None
    binary_threshold = None
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
            clear_negative_ceiling = binary_threshold - policy.binary_clear_margin
            if binary_supported:
                binary_state = "positive"
                binary_reason = "二分类血腥头达到校准阈值"
            elif binary_score <= clear_negative_ceiling:
                binary_state = "clear_negative"
                binary_reason = "二分类血腥头明确低于校准阈值"
            else:
                binary_state = "uncertain"
                binary_reason = "二分类血腥头位于阈值滞回区"
            if binary_state != "positive" and injury not in {None, INJURY_CLASSES[0]}:
                injury = None
                if binary_state == "clear_negative":
                    binary_reason += "，已否决四分类血腥候选"
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
    text_detection = evidence.get("text_detection", {}) or {}
    scene_evidence = evidence.get("scene_concepts", {}) or {}
    knowledge_hits = evidence.get("search_knowledge", []) or []
    # labels 由引擎在未四舍五入的概率上按逐类阈值产生，保留该边界判定。
    selected = [name for name in types.get("labels", [])
                if name in TYPE_CLASSES and name in thresholds]
    active = [name for name in selected if name != TYPE_CLASSES[0]]
    binary_conflict = (
        binary_state in {"clear_negative", "uncertain"} and TYPE_CLASSES[2] in active
    )
    independent_active = [
        name for name in active
        if name != TYPE_CLASSES[2] or binary_state == "positive"
    ]
    conflict = TYPE_CLASSES[0] in selected and TYPE_CLASSES[1] in selected
    has_direct_violence = TYPE_CLASSES[1] in independent_active
    has_threat = TYPE_CLASSES[3] in independent_active
    has_independent_violence = bool(independent_active)
    category = "、".join(
        independent_active
        + ([injury] if injury and injury != INJURY_CLASSES[0] else [])
    )
    action = "allow"
    uncertain = False
    reason = "视觉与辅助证据未形成可执行的违规支持"
    adjustments: list[str] = []
    conflict_type = "none"

    text_status = text_evidence.get("status", "not_available")
    text_high_risk = bool(text_evidence.get("grounded_violation"))
    text_safe_context = bool(text_evidence.get("safe_context_rule_ids"))
    semantic_review = text_evidence.get("semantic_review", {}) or {}
    text_detected = bool(
        text_detection.get("has_text")
        or text_detection.get("status") == "detected"
    )
    text_failed = text_detected and text_status == "failed"
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
    domain_evidence = scene_evidence.get("domain", {}) or {}
    style_router = evidence.get("anime_style")
    if isinstance(style_router, dict):
        anime_matched = bool(style_router.get("is_anime"))
        anime_source = "style_router"
    else:
        anime_matched = bool(
            domain_evidence.get("anime_matched")
            or domain_evidence.get("effective") == "anime_illustration"
        )
        anime_source = "scene_prompt"
    confounder_matched = bool(scene_evidence.get("confounder_matched"))
    fiction_context_score = max(
        [
            float(item.get("score", -1))
            for item in scene_evidence.get("safe_contexts", [])
            if item.get("key") == "fiction_context"
        ]
        or [-1.0]
    )
    editorial_mitigation = bool(
        {"严重模糊", "故意遮挡（马赛克/黑条/贴纸）"} & set(quality_labels)
    )
    news_shell_suspect = bool(
        safe_context_confirmed
        and "news_legal_context" in scene_safe_keys
        and not editorial_mitigation
        and (
            binary_state == "positive"
            or injury == INJURY_CLASSES[3]
            or risk == RISK_CLASSES[3]
        )
    )
    quality_explains_risk = bool(
        quality_needs_review
        and not has_independent_violence
        and binary_state != "positive"
    )
    pseudo_binary_conflict = bool(
        binary_state == "clear_negative"
        and risk in RISK_CLASSES[1:3]
        and not has_independent_violence
        and not text_high_risk
    )
    anime_relaxation = bool(
        policy.aggressive_anime_relaxation
        and pseudo_binary_conflict
        and anime_matched
    )
    real_l1_relaxation = bool(
        pseudo_binary_conflict
        and risk == RISK_CLASSES[1]
        and risk_confidence is not None
        and risk_confidence < policy.real_l1_relax_max_confidence
        and not anime_matched
        and not editorial_mitigation
        and not text_detected
    )
    quality_relaxation = bool(
        pseudo_binary_conflict
        and quality_explains_risk
        and anime_matched
    )
    confounder_relaxation = bool(
        pseudo_binary_conflict
        and confounder_matched
        and (
            anime_matched
            or (
                risk == RISK_CLASSES[1]
                and risk_confidence is not None
                and risk_confidence < policy.real_l1_relax_max_confidence
            )
        )
    )

    if not evidence.get("trained"):
        reason = "分类头未训练，预测不能用于违规处置"
        action, uncertain, conflict_type = "review", True, "model_not_trained"
    elif not risk_rank or not injury_rank:
        reason = "分类分数缺失或无效，无法完成证据核对"
        action, uncertain, conflict_type = "review", True, "missing_scores"
    elif conflict:
        reason = "无正在进行的暴力与攻击进行中同时命中，行为证据存在分歧"
        action, uncertain, conflict_type = "review", True, "behavior_conflict"
    elif text_high_risk:
        action, uncertain = "limit", False
        reason = "OCR 原文逐字证据或窄条件规则确认文字威胁，独立支持限流"
    elif risk == RISK_CLASSES[4]:
        action, uncertain = "review", False
        reason = "风险头最高候选为 X，图片内容无法可靠判定，需要人工复核"
        conflict_type = "risk_x"
    elif binary_state == "positive":
        if injury == INJURY_CLASSES[3] and risk == RISK_CLASSES[3]:
            action, uncertain = "block", False
            reason = "binary 阳性与严重血腥候选一致，风险头为 L3"
        elif independent_active or (injury and injury != INJURY_CLASSES[0]):
            action, uncertain = "limit", False
            reason = "binary 阳性并存在独立暴力或血腥支持，采用限流"
        elif risk == RISK_CLASSES[3]:
            action, uncertain, conflict_type = "review", False, "severe_without_support"
            reason = "风险头为 L3，但 binary 阳性不足以单独确认严重等级"
        elif risk in RISK_CLASSES[1:3]:
            action, uncertain = "limit", False
            reason = "风险头非 L0，且 binary 阳性提供血腥存在性支持"
        else:
            action, uncertain = "allow", False
            reason = "binary 阳性未伴随达到门槛的类别支持，保留风险原值"
    elif binary_state == "clear_negative":
        if has_direct_violence or has_threat:
            action = "block" if risk == RISK_CLASSES[3] and has_direct_violence else "limit"
            uncertain = False
            reason = "binary 阴性只否决血腥维度，独立暴力类型仍支持处置"
        elif risk == RISK_CLASSES[3]:
            action, uncertain, conflict_type = "review", False, "severe_without_content"
            reason = "风险头为 L3，但 binary 明确阴性和独立内容证据不足，不能自动降级"
        elif risk in RISK_CLASSES[1:3]:
            if (
                anime_relaxation
                or real_l1_relaxation
                or quality_relaxation
                or confounder_relaxation
                or (
                    safe_context_confirmed
                    and not has_independent_violence
                    and not text_high_risk
                )
            ):
                action, uncertain = "allow", False
                conflict_type = "pseudo_binary_conflict"
                reason = "binary 明确阴性；风险/后果展示矛盾缺少独立内容支持，按有效风险降级"
                if anime_relaxation:
                    adjustments.append("动漫/插画域 Prompt 命中，解除风格与血腥的假冲突")
                if real_l1_relaxation:
                    adjustments.append("真实域仅对低置信 L1 解除假冲突")
                if quality_relaxation:
                    adjustments.append("质量提示解释了风险头的不稳定候选")
                if confounder_relaxation:
                    adjustments.append("场景 Prompt 命中红色物体/颜料等安全混淆项")
                if safe_context_confirmed:
                    adjustments.append("场景 Prompt 与 OCR/RAG 共同确认安全语境")
            else:
                action, uncertain, conflict_type = "review", False, "risk_without_support"
                reason = "binary 阴性后风险候选缺少独立内容支持，域证据不足以自动降级"
        elif risk == RISK_CLASSES[0]:
            action, uncertain = "allow", False
            reason = "风险头为 L0，binary 明确阴性且无独立违规支持"
    else:
        # Missing or near-threshold binary evidence must not become an
        # aggressive downgrade path.
        if risk == RISK_CLASSES[3] and injury == INJURY_CLASSES[3]:
            action, uncertain = "block", False
            reason = "风险头为 L3，且高置信严重血腥候选提供支持"
        elif has_direct_violence or has_threat:
            action = "block" if risk == RISK_CLASSES[3] and has_direct_violence else "limit"
            uncertain = False
            reason = "binary 结论不确定，但独立暴力类型仍支持处置"
        elif risk == RISK_CLASSES[3] or risk in RISK_CLASSES[1:3]:
            action, uncertain = "review", True
            conflict_type = "binary_uncertain" if binary_state == "uncertain" else "risk_without_support"
            reason = "binary 不足或缺失，无法确认血腥支持是否成立"
        else:
            action, uncertain = "allow", False
            reason = "风险头为 L0，未发现独立违规支持"

    text_unverified = bool(
        text_detected
        and text_status in {"not_available", "failed"}
    )
    real_quality_guard = bool(
        not anime_matched
        and (
            {"严重模糊", "故意遮挡（马赛克/黑条/贴纸）"} & set(quality_labels)
            or (
                "文字扭曲形变" in quality_labels
                and text_detected
                and text_status != "readable"
            )
        )
    )
    if (
        real_quality_guard
        and action == "allow"
        and risk not in {RISK_CLASSES[0], RISK_CLASSES[3]}
    ):
        action, uncertain = "review", False
        conflict_type = "real_quality_unverified"
        reason = "真实/未确认风格域存在严重模糊或遮挡，不能依赖低风险候选自动放行"

    if quality_needs_review:
        reason += "；图像质量退化作为证据可靠性提示"
    unreadable_text_conflict = bool(
        "文字扭曲形变" in quality_labels
        and text_status in {"empty", "partial", "failed"}
        and risk in RISK_CLASSES[1:4]
    )
    if news_shell_suspect:
        action, uncertain = "review", False
        conflict_type = "news_shell_suspect"
        reason = "新闻语境与未加工的血腥暴力画面冲突，疑似新闻壳绕过，需要人工复核"
    elif safe_context_confirmed and action in {"limit", "block"}:
        action, uncertain = "limit", False
        reason = "场景 Prompt 与 OCR/RAG 共同确认新闻/医学等安全语境，但视觉仍存在违规候选，采用限流而非下架或放行"
        adjustments.append("确认安全语境将下架风险上限降为限流")
    elif unreadable_text_conflict and action == "allow":
        action, uncertain = "review", False
        conflict_type = "text_unreadable"
        reason = "quality 命中文字扭曲，但 OCR 无文字或文本残缺，且风险头非安全，需要人工复核"
    review_level = None
    if vision and vision.get("trigger", {}).get("eligible"):
        if vision.get("status") == "ok" and vision.get("verdict") == "not_supported":
            injury = None
            category = "、".join(active)
            if not independent_active and not conflict and binary_state != "positive":
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
        elif not independent_active:
            action, uncertain = "allow", True
            if vision.get("status") == "ok":
                reason = "视觉复核未得出明确结论"
            elif vision.get("stage") == "response_validation" or vision.get("error") == "ValidationError":
                reason = "视觉复核返回格式不符合要求，未取得有效的血腥复核结论"
            else:
                reason = "视觉复核请求未完成，未取得有效的血腥复核结论"
    # Optional image review must not bypass unresolved text or known context
    # hazards. Reapply these guards after the review reconciliation.
    if text_failed and action != "block":
        action, uncertain = "review", False
        conflict_type = "ocr_failed_after_detection"
        reason = "文字检测发现候选区域，但 OCR 失败，不能据此确认安全"
    elif text_unverified and action == "allow":
        action, uncertain = "review", False
        conflict_type = "ocr_not_completed"
        reason = "文字检测发现候选区域，但本次没有可用的 OCR 结论"
    elif (
        action == "allow"
        and text_evidence.get("high_risk_rule_ids")
        and not text_high_risk
        and semantic_review.get("verdict") != "safe"
    ):
        action, uncertain = "review", False
        conflict_type = "text_semantic_unresolved"
        reason = "RAG 命中高风险文字规则，但缺少可靠逐字语义结论，需要人工核对"
    elif unreadable_text_conflict and action == "allow":
        action, uncertain = "review", False
        conflict_type = "text_unreadable"
        reason = "quality 命中文字扭曲，但 OCR 无文字或文本残缺，且风险头非安全，需要人工复核"
    elif news_shell_suspect and action != "block":
        action, uncertain = "review", False
        conflict_type = "news_shell_suspect"
        reason = "新闻语境与未加工的血腥暴力画面冲突，疑似新闻壳绕过，需要人工复核"
    if (
        action == "review"
        and risk == RISK_CLASSES[3]
        and anime_matched
        and binary_score is not None
        and binary_score < 0.80
        and fiction_context_score >= 0.06
        and injury in {None, INJURY_CLASSES[1]}
        and not has_direct_violence
        and not has_threat
        and not text_high_risk
    ):
        action, uncertain = "limit", False
        review_level = RISK_CLASSES[1]
        conflict_type = "anime_severe_style_cap"
        reason = "二次元虚构语境与高风险候选冲突，但仅见轻微 injury 且无直接攻击或明确威胁；按轻度限流处理"
        adjustments.append("动漫风格夸张导致严重风险候选缺少关系证据，降为 L1 限流")
    if (
        risk == RISK_CLASSES[1]
        and anime_matched
        and fiction_context_score >= 0.035
        and TYPE_CLASSES[2] not in independent_active
        and not text_high_risk
        and action in {"limit", "review"}
    ):
        action, uncertain = "allow", False
        review_level = RISK_CLASSES[0]
        conflict_type = "anime_l1_fiction_pose"
        reason = "虚构动画语境明确，且不存在后果展示或 grounded 文字威胁，按无冲突低风险候选放行"
        adjustments.append("动画战斗 pose 导致的 L1 假阳性已解除")
    if uncertain:
        action = "review"
        reason += "；证据存在高不确定性，进入人工复核"
    if binary_reason:
        reason += f"；{binary_reason}"
    effective_risk = review_level or risk or "无有效预测"
    if action == "allow":
        effective_risk = RISK_CLASSES[0]
    elif action == "block":
        effective_risk = RISK_CLASSES[3]
    elif action == "limit" and effective_risk == RISK_CLASSES[0]:
        effective_risk = RISK_CLASSES[2] if text_high_risk else RISK_CLASSES[1]
    return {"policy_version": "2026-10-26-effective-v5", "action": action,
            "level": effective_risk, "effective_risk": effective_risk,
            "risk_raw": risk, "uncertain": uncertain, "conflict_type": conflict_type,
            "vision_review": vision,
            "category": category or "无达到处置门槛的违规类别", "reason": reason,
            "adjustments": adjustments,
            "binary_state": binary_state,
            "risk_candidate": {"label": risk, "score": risk_rank[0][0] if risk_rank else None},
            "effective_risk_candidate": {
                "label": effective_risk,
                "score": risk_rank[0][0] if risk_rank else None,
                "anime_matched": anime_matched,
                "anime_source": anime_source,
                "pseudo_binary_conflict": pseudo_binary_conflict,
            },
            "injury_candidate": {"label": injury_rank[0][1] if injury_rank else None,
                                 "score": injury_rank[0][0] if injury_rank else None,
                                 "accepted": injury is not None},
            "blood_binary_candidate": {
                "score": binary.get("p_blood") if isinstance(binary, dict) else None,
                "threshold": binary.get("threshold") if isinstance(binary, dict) else None,
                "supported": binary_supported,
                "state": binary_state,
            },
            "quality_candidate": {
                "labels": quality_labels,
                "needs_review": quality_needs_review,
                "action_effect": "evidence_warning_only",
            },
            "text_detection": text_detection,
            "text_evidence": text_evidence,
            "search_knowledge_hits": knowledge_hits,
            "scene_context": {
                "risk_scenes": scene_evidence.get("risk_scenes", []),
                "safe_contexts": scene_evidence.get("safe_contexts", []),
                "confounders": scene_evidence.get("confounders", []),
                "domain": domain_evidence,
                "style_router": style_router,
                "context_conflict": safe_context_confirmed,
                "safe_context_confirmed": safe_context_confirmed,
                "confounder_matched": confounder_matched,
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
    effective = decision.get("risk_raw") or decision["risk_candidate"]["label"]
    effective_label = decision.get("effective_risk") or decision.get("level")
    if effective_label and effective_label != effective:
        effective_text = f"有效风险：{effective_label}。"
    else:
        effective_text = ""
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
    domain = (scene.get("domain") or {}).get("effective")
    domain_text = "；域：二次元/插画" if domain == "anime_illustration" else ""
    return (f"审核结果：{status}；处置：{action}。"
            f"CLIP 风险候选：{risk}；暴力类型：{types or '无命中标签'}；血腥候选：{injury}。"
            f"{effective_text}{binary_text}{quality_text}{text_text}{scene_text}"
            f"{domain_text}{knowledge_text}{review_text}依据：{decision['reason']}。")
