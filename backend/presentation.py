"""Build the bounded, review-facing API payload from internal evidence."""

from __future__ import annotations

import re
from typing import Any

from .text_evidence import HIGH_RISK_RULE_IDS, SAFE_CONTEXT_RULE_IDS


ACTION_LABELS = {
    "allow": "放行",
    "limit": "限流",
    "review": "人工复核",
    "block": "下架",
    "unavailable": "流程不可用",
}

STATUS_LABELS = {
    "ok": "已完成",
    "detected": "已检出",
    "empty": "未检出",
    "skipped": "已跳过",
    "failed": "执行失败",
    "invalid": "结果无效",
    "not_available": "不可用",
    "unavailable": "不可用",
    "disabled": "已关闭",
    "uncertain": "不确定",
    "fallback": "已降级",
    "template": "程序模板",
    "matched": "已命中",
}

OUTCOME_LABELS = {
    "safe": "支持安全",
    "risk": "发现风险",
    "mixed": "证据冲突",
    "neutral": "中性证据",
    "unknown": "无法判断",
    "not_applicable": "不适用",
}

TIMING_LABELS = {
    "service.image_persist": ("原图审计写盘", "service"),
    "workflow.total": ("固定工作流", "workflow"),
    "analyze_image": ("FG-CLIP2 视觉推理", "model"),
    "review_blood": ("血腥视觉复核", "model"),
    "detect_text": ("文字区域检测", "text"),
    "ocr_image": ("OCR 文字识别", "text"),
    "text.rag": ("OCR 规则检索", "text"),
    "text.semantic": ("文字语义复核", "model"),
    "match_scene_concepts": ("场景 Prompt 匹配", "model"),
    "policy.build_evidence": ("证据包整理", "policy"),
    "policy.decision": ("有效风险裁决", "policy"),
    "policy.anime_gate": ("虚拟视觉门控", "policy"),
    "policy.anime_visual": ("虚拟视觉复核", "model"),
    "explanation": ("LLM 解释生成", "model"),
    "compose": ("最终建议组装", "policy"),
    "service.total": ("后端总处理", "service"),
}


def _duration(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return max(0, int(round(float(value))))
    except (TypeError, ValueError):
        return None


def _score(value: Any) -> float | None:
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None


def _status(value: Any, default: str = "not_available") -> str:
    normalized = str(value or default)
    return normalized if normalized in STATUS_LABELS else normalized


def _tool(
    tool_id: str,
    label: str,
    *,
    status: str,
    outcome: str,
    summary: str,
    duration_ms: int | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": tool_id,
        "label": label,
        "status": _status(status),
        "status_label": STATUS_LABELS.get(_status(status), _status(status)),
        "outcome": outcome,
        "outcome_label": OUTCOME_LABELS.get(outcome, outcome),
        "summary": summary,
        "duration_ms": duration_ms,
        "details": details or {},
    }


def _risk_tone(action: str) -> str:
    return {
        "allow": "safe",
        "limit": "warning",
        "review": "review",
        "block": "danger",
        "unavailable": "failed",
    }.get(action, "neutral")


def _finding_for_risk(label: str | None) -> str:
    if label == "L0_安全":
        return "safe"
    if label in {"L1_轻微违规", "L2_中度违规", "L3_严重违规"}:
        return "risk"
    return "unknown"


def _tool_status_from_step(
    steps: list[dict[str, Any]],
    name: str,
) -> tuple[str, str]:
    for step in steps:
        if step.get("name") != name:
            continue
        if step.get("type") == "tool_error":
            return "failed", str(step.get("error") or "执行失败")
        if step.get("type") == "tool_skip":
            return "skipped", str(step.get("reason") or "未触发")
        if step.get("type") == "tool_call":
            return "ok", ""
    return "not_available", ""


def _compact_reason_parts(reason: str) -> list[str]:
    parts = [
        re.sub(r"\s+", " ", part).strip("。 ")
        for part in re.split(r"[；;]", reason or "")
        if part.strip()
    ]
    return [part for part in parts if part]


def _primary_reasons(decision: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    candidates.extend(str(item) for item in (decision.get("adjustments") or []))
    candidates.extend(_compact_reason_parts(str(decision.get("reason") or "")))
    deduplicated: list[str] = []
    for item in candidates:
        item = re.sub(r"\s+", " ", item).strip()
        if not item or item in deduplicated:
            continue
        deduplicated.append(item[:120])
        if len(deduplicated) == 2:
            break
    return deduplicated


def _domain(evidence: dict[str, Any]) -> tuple[str, str, float | None, str]:
    style = evidence.get("anime_style")
    if isinstance(style, dict) and style:
        score = _score(style.get("score"))
        if bool(style.get("is_anime")):
            return "virtual", "虚拟（二次元 / 插画 / 游戏场景等）", score, "style_router"
        return "real", "真实", score, "style_router"
    domain = (evidence.get("scene_concepts") or {}).get("domain") or {}
    score = _score(domain.get("anime_score"))
    if domain.get("effective") == "anime_illustration" or domain.get("anime_matched"):
        return "virtual", "虚拟（二次元 / 插画 / 游戏场景等）", score, "scene_prompt"
    return "real", "真实", score, "default"


def _model_heads(evidence: dict[str, Any]) -> dict[str, Any]:
    heads = evidence.get("model_heads")
    return heads if isinstance(heads, dict) else evidence


def _risk_tool(evidence: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    risk = _model_heads(evidence).get("risk") or {}
    label = str(risk.get("label") or decision.get("risk_raw") or "无有效预测")
    confidence = _score(risk.get("confidence"))
    return _tool(
        "risk_head",
        "风险等级头",
        status="ok" if risk.get("scores") else "failed",
        outcome=_finding_for_risk(label),
        summary=f"原始候选 {label}" + (
            f"，置信度 {confidence:.3f}" if confidence is not None else ""
        ),
        details={
            "label": label,
            "confidence": confidence,
            "scores": risk.get("scores") or {},
        },
    )


def _type_tool(evidence: dict[str, Any]) -> dict[str, Any]:
    task = _model_heads(evidence).get("type") or {}
    labels = list(task.get("labels") or [])
    active = [label for label in labels if label != "无暴力"]
    if active:
        outcome = "risk"
    elif labels:
        outcome = "safe"
    else:
        outcome = "neutral"
    return _tool(
        "type_head",
        "暴力类型头",
        status="ok" if task.get("scores") else "failed",
        outcome=outcome,
        summary="命中：" + ("、".join(active) if active else "无达到阈值的风险类型"),
        details={
            "labels": labels,
            "scores": task.get("scores") or {},
            "thresholds": task.get("thresholds") or {},
        },
    )


def _injury_tool(evidence: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    task = _model_heads(evidence).get("injury") or {}
    label = str(task.get("label") or "无有效预测")
    accepted = (decision.get("injury_candidate") or {}).get("accepted")
    outcome = "safe" if label == "无血腥" else "risk"
    if accepted is False and label != "无血腥":
        outcome = "mixed"
    return _tool(
        "injury_head",
        "血腥严重度头",
        status="ok" if task.get("scores") else "failed",
        outcome=outcome,
        summary=(
            f"候选 {label}"
            + ("，已作为证据采纳" if accepted is True else "，未直接采纳")
        ),
        details={
            "label": label,
            "confidence": _score(task.get("confidence")),
            "accepted": accepted,
            "scores": task.get("scores") or {},
        },
    )


def _quality_tool(evidence: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    quality = _model_heads(evidence).get("quality") or {}
    labels = list(quality.get("labels") or [])
    needs_review = bool((decision.get("quality_candidate") or {}).get("needs_review"))
    return _tool(
        "quality_fusion",
        "质量融合头",
        status="ok" if quality.get("scores") else "failed",
        outcome="mixed" if needs_review else "neutral",
        summary="、".join(labels) if labels else "无达到阈值的质量标签",
        details={
            "labels": labels,
            "scores": quality.get("scores") or {},
            "thresholds": quality.get("thresholds") or {},
            "needs_review": needs_review,
            "source": quality.get("source", ""),
        },
    )


def _binary_tool(evidence: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    binary = evidence.get("blood_binary") or {}
    state = str(decision.get("binary_state") or "unknown")
    score = _score(binary.get("p_blood"))
    threshold = _score(binary.get("threshold"))
    if state == "clear_negative":
        outcome = "safe"
    elif state == "positive":
        outcome = "risk"
    elif state == "uncertain":
        outcome = "mixed"
    else:
        outcome = "unknown"
    summary = {
        "clear_negative": "明确无血腥支持",
        "positive": "支持存在血腥",
        "uncertain": "位于阈值滞回区",
        "unknown": "未取得有效结论",
    }.get(state, state)
    return _tool(
        "blood_binary",
        "血腥二分类头",
        status="ok" if binary else "failed",
        outcome=outcome,
        summary=summary,
        details={
            "p_blood": score,
            "threshold": threshold,
            "state": state,
            "supported": (decision.get("blood_binary_candidate") or {}).get(
                "supported"
            ),
        },
    )


def _anime_tool(evidence: dict[str, Any]) -> dict[str, Any]:
    style = evidence.get("anime_style")
    if not isinstance(style, dict) or not style:
        domain = (evidence.get("scene_concepts") or {}).get("domain") or {}
        score = _score(domain.get("anime_score"))
        matched = bool(domain.get("anime_matched"))
        return _tool(
            "anime_router",
            "虚拟风格路由",
            status="skipped",
            outcome="unknown",
            summary="风格路由未加载，使用场景 Prompt 域候选",
            details={
                "score": score,
                "is_anime": matched,
                "source": "scene_prompt",
            },
        )
    score = _score(style.get("score"))
    threshold = _score(style.get("threshold"))
    is_anime = bool(style.get("is_anime"))
    return _tool(
        "anime_router",
        "虚拟风格路由",
        status="ok",
        outcome="neutral",
        summary=(
            f"{'判定为虚拟内容' if is_anime else '未判定为虚拟内容'}"
            + (f"，分数 {score:.3f}" if score is not None else "")
        ),
        details={
            "score": score,
            "threshold": threshold,
            "is_anime": is_anime,
            "source": style.get("source", ""),
        },
    )


def _scene_tool(evidence: dict[str, Any], steps: list[dict[str, Any]],
                 timings: dict[str, Any]) -> dict[str, Any]:
    scene = evidence.get("scene_concepts")
    if (
        not isinstance(scene, dict)
        or not scene
        or scene.get("status") == "not_called"
    ):
        status, reason = _tool_status_from_step(steps, "match_scene_concepts")
        return _tool(
            "scene_prompt",
            "场景与语境 Prompt",
            status=status,
            outcome="not_applicable",
            summary="未执行场景匹配" + (f"：{reason}" if reason else ""),
            duration_ms=_duration(timings.get("match_scene_concepts")),
        )
    domain = scene.get("domain") or {}
    risk_scenes = list(scene.get("risk_scenes") or [])
    safe_contexts = list(scene.get("safe_contexts") or [])
    confounders = list(scene.get("confounders") or [])
    routing_signals = list(scene.get("routing_signals") or [])
    matched = bool(scene.get("risk_matched") or scene.get("safe_context_matched"))
    return _tool(
        "scene_prompt",
        "场景与语境 Prompt",
        status=str(scene.get("status") or ("matched" if matched else "empty")),
        outcome="mixed" if scene.get("context_conflict") else (
            "risk" if scene.get("risk_matched") else "safe"
        ),
        summary=(
            f"域：{domain.get('effective', 'unknown')}；"
            f"风险场景 {len(risk_scenes)}；安全语境 {len(safe_contexts)}；"
            f"混淆项 {len(confounders)}；路由信号 {len(routing_signals)}"
        ),
        duration_ms=_duration(timings.get("match_scene_concepts")),
        details={
            "domain": domain,
            "risk_scenes": risk_scenes,
            "safe_contexts": safe_contexts,
            "confounders": confounders,
            "routing_signals": routing_signals,
            "routing_scores": scene.get("routing_scores") or {},
            "game_ui_score": _score(scene.get("game_ui_score")),
            "game_ui_trigger_threshold": _score(
                scene.get("game_ui_trigger_threshold")
            ),
            "game_ui_matched": bool(scene.get("game_ui_matched")),
            "risk_margin": _score(scene.get("risk_margin")),
            "context_conflict": bool(scene.get("context_conflict")),
            "knowledge_hits": scene.get("knowledge_hits") or [],
        },
    )


def _detector_tool(evidence: dict[str, Any], steps: list[dict[str, Any]],
                   timings: dict[str, Any]) -> dict[str, Any]:
    detection = evidence.get("text_detection") or {}
    status = str(detection.get("status") or "not_available")
    has_text = bool(detection.get("has_text"))
    outcome = "risk" if has_text else ("safe" if status == "empty" else "unknown")
    summary = (
        f"发现 {int(detection.get('box_count', 0) or 0)} 个文字候选区域"
        if has_text else "未发现文字候选区域"
    )
    if status == "failed":
        summary = "文字检测失败，程序保守地执行 OCR"
    return _tool(
        "text_detector",
        "轻量文字检测",
        status=_status(status),
        outcome=outcome,
        summary=summary,
        duration_ms=_duration(timings.get("detect_text")),
        details={
            "has_text": has_text,
            "box_count": int(detection.get("box_count", 0) or 0),
            "confidence": _score(detection.get("confidence")),
            "method": detection.get("method", ""),
            "error": detection.get("error"),
        },
    )


def _ocr_tool(evidence: dict[str, Any], cache: dict[str, Any],
              steps: list[dict[str, Any]], timings: dict[str, Any],
              redact_text) -> dict[str, Any]:
    ocr = cache.get("ocr_image")
    text_evidence = (
        evidence.get("text_evidence")
        or evidence.get("ocr")
        or {}
    )
    if not isinstance(ocr, dict):
        status, reason = _tool_status_from_step(steps, "ocr_image")
        return _tool(
            "ocr",
            "OCR 文字识别",
            status=status,
            outcome="not_applicable",
            summary="未执行 OCR" + (f"：{reason}" if reason else ""),
            duration_ms=_duration(timings.get("ocr_image")),
        )
    success = bool(ocr.get("success"))
    texts = [str(item) for item in (ocr.get("texts") or []) if str(item).strip()]
    preview = redact_text(" ".join(texts))[:500]
    return _tool(
        "ocr",
        "OCR 文字识别",
        status="ok" if success else "failed",
        outcome="neutral" if success else "unknown",
        summary=(
            f"识别 {len(texts)} 段文字，平均置信度 "
            f"{_score(ocr.get('mean_confidence'))}"
            if success else f"OCR 失败：{ocr.get('error', 'unknown')}"
        ),
        duration_ms=_duration(timings.get("ocr_image")),
        details={
            "success": success,
            "available": bool(ocr.get("available", True)),
            "provider": ocr.get("provider", ""),
            "text_count": len(texts),
            "text_preview": preview,
            "mean_confidence": _score(ocr.get("mean_confidence")),
            "request_id": ocr.get("request_id", ""),
            "error": ocr.get("error"),
            "text_evidence": {
                "status": text_evidence.get("status"),
                "char_count": text_evidence.get("char_count"),
                "high_risk_rule_ids": text_evidence.get("high_risk_rule_ids", []),
                "safe_context_rule_ids": text_evidence.get("safe_context_rule_ids", []),
            },
        },
    )


def _text_semantic_tool(evidence: dict[str, Any], timings: dict[str, Any],
                        knowledge_hits: list[dict[str, Any]]) -> dict[str, Any]:
    text = evidence.get("text_evidence") or evidence.get("ocr") or {}
    semantic = text.get("semantic_review") or {}
    high_risk = list(text.get("high_risk_rule_ids") or [])
    safe_context = list(text.get("safe_context_rule_ids") or [])
    grounded = bool(text.get("grounded_violation"))
    status = str(semantic.get("status") or text.get("status") or "not_available")
    verdict = str(semantic.get("verdict") or "uncertain")
    if grounded or verdict == "violation":
        outcome = "risk"
    elif verdict == "safe":
        outcome = "safe"
    elif high_risk:
        outcome = "mixed"
    else:
        outcome = "neutral"
    summary = {
        "violation": "逐字证据确认文字违规",
        "safe": "未确认文字违规",
        "uncertain": "文字语义无法可靠确认",
    }.get(verdict, "未取得文字语义结论")
    if high_risk:
        summary += f"；高风险规则 {', '.join(high_risk)}"
    return _tool(
        "text_semantic_rag",
        "文字语义与 RAG",
        status=_status(status),
        outcome=outcome,
        summary=summary,
        duration_ms=(
            (_duration(timings.get("text.rag")) or 0)
            + (_duration(timings.get("text.semantic")) or 0)
        ),
        details={
            "text_status": text.get("status"),
            "semantic_status": semantic.get("status"),
            "verdict": verdict,
            "grounded_violation": grounded,
            "high_risk_rule_ids": high_risk,
            "safe_context_rule_ids": safe_context,
            "prompt_version": semantic.get("prompt_version"),
            "error": semantic.get("error"),
            "knowledge_hits": knowledge_hits,
            "high_risk_rule_catalog": sorted(HIGH_RISK_RULE_IDS),
            "safe_context_rule_catalog": sorted(SAFE_CONTEXT_RULE_IDS),
        },
    )


def _anime_visual_tool(log: dict[str, Any], timings: dict[str, Any]) -> dict[str, Any]:
    gate = log.get("anime_visual_gate") or {}
    review = log.get("anime_visual_review")
    decision = log.get("decision") or {}
    if not gate:
        return _tool(
            "anime_visual_review",
            "虚拟视觉证据复核",
            status="not_available",
            outcome="not_applicable",
            summary="未生成虚拟视觉门控结果",
        )
    if not gate.get("eligible"):
        return _tool(
            "anime_visual_review",
            "虚拟视觉证据复核",
            status="skipped",
            outcome="not_applicable",
            summary=f"门控未通过：{gate.get('reason', 'unknown')}",
            details={"gate": gate},
        )
    if not isinstance(review, dict):
        return _tool(
            "anime_visual_review",
            "虚拟视觉证据复核",
            status="not_available",
            outcome="unknown",
            summary="满足门控，但本次没有可用的视觉模型结果",
            details={"gate": gate},
        )
    status = str(review.get("status") or "uncertain")
    if status != "ok":
        return _tool(
            "anime_visual_review",
            "虚拟视觉证据复核",
            status="failed",
            outcome="unknown",
            summary="视觉模型调用失败，保留程序原动作",
            duration_ms=_duration(review.get("elapsed_ms")),
            details={"gate": gate, "review": review},
        )
    injury = review.get("injury")
    attack = review.get("attack")
    threat = review.get("threat")
    risky = attack == "present" or threat == "present" or injury in {
        "moderate", "severe",
    }
    uncertain = any(
        review.get(key) == "uncertain"
        for key in ("injury", "attack", "threat", "weapon_visible")
    )
    outcome = "risk" if risky else ("mixed" if uncertain else "safe")
    return _tool(
        "anime_visual_review",
        "虚拟视觉证据复核",
        status="ok",
        outcome=outcome,
        summary="；".join(
            [
                f"伤害 {injury}",
                f"攻击 {attack}",
                f"威胁 {threat}",
                f"效果 {decision.get('anime_visual_effect', '未记录')}",
            ]
        ),
        duration_ms=_duration(review.get("elapsed_ms")),
        details={"gate": gate, "review": review},
    )


def _blood_visual_tool(
    cache: dict[str, Any],
    timings: dict[str, Any],
) -> dict[str, Any] | None:
    review = cache.get("review_blood")
    if not isinstance(review, dict):
        # The production fixed workflow does not use this legacy optional tool.
        # Do not create a placeholder that makes it look like part of the run.
        return None
    status = str(review.get("status") or "uncertain")
    if status == "skipped":
        return _tool(
            "blood_visual_review",
            "血腥视觉复核",
            status="skipped",
            outcome="not_applicable",
            summary="未满足血腥视觉复核门控",
            details=review,
        )
    verdict = str(review.get("verdict") or "uncertain")
    outcome = {
        "supported": "risk",
        "not_supported": "safe",
        "uncertain": "mixed",
    }.get(verdict, "unknown")
    return _tool(
        "blood_visual_review",
        "血腥视觉复核",
        status="ok" if status == "ok" else "failed",
        outcome=outcome,
        summary=f"结论：{verdict}；严重度：{review.get('severity', 'unknown')}",
        duration_ms=_duration(review.get("elapsed_ms") or timings.get("review_blood")),
        details=review,
    )


def _policy_tool(decision: dict[str, Any], action: str,
                 timings: dict[str, Any]) -> dict[str, Any]:
    summary = (
        f"{decision.get('risk_raw') or '无原始风险'} → "
        f"{decision.get('effective_risk') or '无有效风险'}；"
        f"最终动作 {ACTION_LABELS.get(action, action)}"
    )
    if action == "review" and decision.get("recommended_action_label"):
        summary += f"；{decision['recommended_action_label']}"
    return _tool(
        "effective_policy",
        "有效风险与最终裁决",
        status="ok" if decision else "failed",
        outcome={
            "allow": "safe",
            "limit": "risk",
            "review": "mixed",
            "block": "risk",
            "unavailable": "unknown",
        }.get(action, "unknown"),
        summary=summary,
        duration_ms=_duration(timings.get("policy.decision")),
        details={
            "policy_version": decision.get("policy_version"),
            "conflict_type": decision.get("conflict_type"),
            "uncertain": decision.get("uncertain"),
            "adjustments": decision.get("adjustments") or [],
            "binary_state": decision.get("binary_state"),
            "recommended_action": decision.get("recommended_action"),
            "review_hint": bool(decision.get("review_hint")),
            "review_reason": decision.get("review_reason"),
        },
    )


def _explanation_tool(log: dict[str, Any], timings: dict[str, Any]) -> dict[str, Any]:
    explanation = log.get("final_explanation") or {}
    status = str(explanation.get("status") or "not_available")
    source = str(explanation.get("source") or "")
    if status == "ok":
        outcome = "neutral"
        summary = "LLM 在动作锁定后按固定模板填写了说明"
    elif status == "template":
        outcome = "neutral"
        summary = "未调用 LLM，由程序固定模板直接生成说明"
    elif status == "fallback":
        outcome = "unknown"
        summary = "LLM 解释无效，已使用确定性模板"
    elif status == "disabled":
        outcome = "not_applicable"
        summary = "LLM 解释功能已关闭"
    else:
        outcome = "unknown"
        summary = "没有可用的解释结果"
    return _tool(
        "explanation",
        "LLM 解释生成",
        status=_status(status),
        outcome=outcome,
        summary=summary,
        duration_ms=_duration(timings.get("explanation")),
        details={
            "prompt_version": explanation.get("prompt_version"),
            "source": source,
            "template_version": explanation.get("template_version"),
            "skip_reason": explanation.get("skip_reason"),
            "error": explanation.get("error"),
            "reason": explanation.get("reason"),
        },
    )


def _group(
    group_id: str,
    label: str,
    description: str,
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "id": group_id,
        "label": label,
        "description": description,
        "tools": tools,
    }


def build_frontend_payload(
    log: dict[str, Any],
    cache: dict[str, Any],
    timings: dict[str, Any],
    *,
    redact_text,
) -> dict[str, Any]:
    """Return the curated frontend contract without secrets or internal paths."""
    evidence = log.get("decision_evidence") or {}
    decision = log.get("decision") or {}
    action = str(decision.get("action") or "unavailable")
    steps = list(log.get("steps") or [])
    knowledge_hits = list(log.get("search_knowledge_hits") or [])
    domain, domain_label, domain_score, domain_source = _domain(evidence)
    primary_reasons = _primary_reasons(decision)

    basic_tools = [
        _risk_tool(evidence, decision),
        _type_tool(evidence),
        _injury_tool(evidence, decision),
        _quality_tool(evidence, decision),
    ]
    specialized_tools = [
        _binary_tool(evidence, decision),
        _anime_tool(evidence),
        _scene_tool(evidence, steps, timings),
    ]
    text_tools = [
        _detector_tool(evidence, steps, timings),
        _ocr_tool(evidence, cache, steps, timings, redact_text),
        _text_semantic_tool(evidence, timings, knowledge_hits),
    ]
    branch_tools = [
        _anime_visual_tool(log, timings),
        _policy_tool(decision, action, timings),
        _explanation_tool(log, timings),
    ]
    blood_visual = _blood_visual_tool(cache, timings)
    if blood_visual is not None:
        branch_tools.insert(0, blood_visual)

    timing_stages = []
    for key, value in timings.items():
        duration_ms = _duration(value)
        if duration_ms is None:
            continue
        label, category = TIMING_LABELS.get(key, (key, "other"))
        timing_stages.append({
            "id": key,
            "label": label,
            "category": category,
            "duration_ms": duration_ms,
        })
    timing_stages.sort(
        key=lambda item: (
            item["id"] != "service.total",
            -item["duration_ms"],
        )
    )

    return {
        "schema_version": "2026-10-27-review-v1",
        "result": {
            "action": action,
            "action_label": ACTION_LABELS.get(action, action),
            "tone": _risk_tone(action),
            "effective_risk": decision.get("effective_risk") or "无有效预测",
            "risk_raw": decision.get("risk_raw"),
            "domain": domain,
            "domain_label": domain_label,
            "domain_score": domain_score,
            "domain_source": domain_source,
            "confidence": _score(
                (_model_heads(evidence).get("risk") or {}).get("confidence")
            ),
            "reason": decision.get("reason") or "没有可用的程序裁决理由",
            "primary_reasons": primary_reasons,
            "adjustments": list(decision.get("adjustments") or []),
            "conflict_type": decision.get("conflict_type") or "none",
            "uncertain": bool(decision.get("uncertain")),
            "policy_version": decision.get("policy_version"),
            "suggestion_source": log.get("suggestion_source", ""),
            "degraded": bool(log.get("degraded")),
            "recommended_action": decision.get("recommended_action") or action,
            "recommended_action_label": decision.get(
                "recommended_action_label"
            ) or ACTION_LABELS.get(action, action),
            "review_hint": bool(decision.get("review_hint")),
            "review_reason": decision.get("review_reason") or "",
        },
        "groups": [
            _group(
                "basic_heads",
                "CLIP 基础头",
                "FG-CLIP2 输出的风险、暴力类型、血腥严重度和质量证据。",
                basic_tools,
            ),
            _group(
                "specialized_context",
                "专项模型与语境",
                "独立血腥二分类、虚拟风格路由以及场景语义 Prompt。",
                specialized_tools,
            ),
            _group(
                "text_rules",
                "文字与规则",
                "文字区域检测、OCR 原文证据、逐字语义校验和本地 RAG。",
                text_tools,
            ),
            _group(
                "branch_policy",
                "分支与最终裁决",
                "虚拟视觉复核、有效风险调整和动作锁定的程序依据。",
                branch_tools,
            ),
        ],
        "timing": {
            "total_ms": _duration(timings.get("service.total"))
            or _duration(log.get("elapsed_ms"))
            or 0,
            "stages": timing_stages,
        },
    }
