# -*- coding: utf-8 -*-
"""Program-first explanations with an optional fixed-schema LLM renderer."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field


EXPLANATION_PROMPT_VERSION = "2026-10-27-template-v2"
EXPLANATION_TEMPLATE_VERSION = "program-explanation-v1"
ACTION_LABELS = {
    "allow": "放行",
    "limit": "限流",
    "review": "人工复核",
    "block": "下架",
    "unavailable": "流程不可用",
}

EXPLANATION_SYSTEM_PROMPT = """你是图片审核解释生成器，不是决策器。
程序已经完成最终处置，动作、有效风险和建议动作都不可修改。

只能根据输入中的固定模板和事实摘要，填写以下三个字段：
- summary：一句话说明处置依据；
- evidence_note：一句话概括最关键的支持或冲突证据；
- limitation：如有质量、文字或调用限制，用一句话说明，否则为空字符串。

要求：
1. 不新增图片中未提供的事实，不把候选分数写成亲眼观察。
2. 不输出放行、限流、人工复核、下架等动作词，程序会统一显示动作。
3. 不复述内部字段名，不堆砌全部分数。
4. 不复述固定模板中不存在的数字、规则编号或对象。
5. 只输出 JSON 对象：
{"summary":"...","evidence_note":"...","limitation":"..."}
"""


class ExplanationDraft(BaseModel):
    summary: str = Field(min_length=8, max_length=220)
    evidence_note: str = Field(min_length=8, max_length=220)
    limitation: str = Field(default="", max_length=180)


class ExplanationValidationError(ValueError):
    """The generated explanation violated the fixed template."""


def _message_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts).strip()
    return str(content or "").strip()


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ExplanationValidationError("invalid_json") from exc
        try:
            payload = json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError as nested:
            raise ExplanationValidationError("invalid_json") from nested
    if not isinstance(payload, dict):
        raise ExplanationValidationError("json_object_required")
    return payload


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def validate_explanation_fields(draft: ExplanationDraft) -> ExplanationDraft:
    values = {
        "summary": _normalize(draft.summary),
        "evidence_note": _normalize(draft.evidence_note),
        "limitation": _normalize(draft.limitation),
    }
    combined = "".join(values.values())
    if any(label in combined for label in ACTION_LABELS.values()):
        raise ExplanationValidationError("action_word_forbidden")
    if "以上为模型预测" in combined:
        raise ExplanationValidationError("fixed_footer_forbidden")
    return ExplanationDraft(**values)


def _summary_for_action(action: str) -> str:
    return {
        "allow": "程序综合视觉与辅助证据后，未形成达到自动处置门槛的违规支持。",
        "limit": "程序综合多源证据后，形成了轻度违规支持并采用限制措施。",
        "review": "现有证据存在无法自动消解的冲突或不确定性，需要人工确认。",
        "block": "多源证据形成了严重违规支持，程序采用最高等级处置。",
        "unavailable": "审核流程未取得必需视觉证据，无法形成内容结论。",
    }.get(action, "程序已完成多源证据核对。")


def _evidence_note(
    decision: dict[str, Any],
    decision_evidence: dict[str, Any] | None,
) -> str:
    bits: list[str] = []
    vision = decision.get("vision_review")
    if isinstance(vision, dict) and vision.get("status") == "ok":
        verdict = vision.get("verdict")
        if verdict == "not_supported":
            bits.append("血腥复核：未发现血腥")
        elif verdict == "supported":
            bits.append(
                f"血腥复核：支持{vision.get('severity', '存在')}血腥证据"
            )
        elif verdict == "uncertain":
            bits.append("血腥复核：未能形成明确结论")
    effective = decision.get("effective_risk")
    risk_raw = decision.get("risk_raw")
    if effective and risk_raw and effective != risk_raw:
        bits.append(f"有效风险由 {risk_raw} 调整为 {effective}")
    elif effective:
        bits.append(f"有效风险为 {effective}")

    binary_state = decision.get("binary_state")
    if binary_state == "clear_negative":
        bits.append("血腥二分类明确阴性")
    elif binary_state == "positive":
        bits.append("血腥二分类提供阳性支持")
    elif binary_state == "uncertain":
        bits.append("血腥二分类位于不确定区间")

    text = (decision_evidence or {}).get("ocr") or {}
    if text.get("grounded_violation"):
        bits.append("OCR 原文存在逐字核验的文字违规证据")
    elif text.get("high_risk_rule_ids"):
        bits.append("文字证据命中高风险规则但结论仍需结合语义")
    elif text.get("status") in {"failed", "partial"}:
        bits.append("文字证据读取不完整")

    adjustments = list(decision.get("adjustments") or [])
    if adjustments:
        bits.append(str(adjustments[0]))
    if not bits:
        bits.append("视觉候选与辅助证据已完成核对")
    return "；".join(bits[:2]) + "。"


def _limitation(decision: dict[str, Any]) -> str:
    quality = decision.get("quality_candidate") or {}
    labels = list(quality.get("labels") or [])
    limitations = [
        label for label in labels
        if label in {"严重模糊", "文字扭曲形变", "故意遮挡（马赛克/黑条/贴纸）"}
    ]
    text = decision.get("text_evidence") or {}
    if text.get("status") in {"failed", "partial"} and "文字证据读取不完整" not in limitations:
        limitations.append("文字证据读取不完整")
    if not limitations:
        return ""
    return "质量或文字条件存在限制：" + "、".join(limitations[:2]) + "。"


def build_program_explanation(
    decision: dict[str, Any],
    decision_evidence: dict[str, Any] | None = None,
    *,
    skip_reason: str = "",
) -> dict[str, Any]:
    """Build the authoritative explanation without any model call."""
    result = {
        "status": "template",
        "source": "program_template",
        "template_version": EXPLANATION_TEMPLATE_VERSION,
        "summary": _summary_for_action(str(decision.get("action") or "unavailable")),
        "evidence_note": _evidence_note(decision, decision_evidence),
        "limitation": _limitation(decision),
        "review_hint": bool(decision.get("review_hint")),
        "review_suggestion": decision.get("recommended_action_label", ""),
        "skip_reason": skip_reason,
    }
    reason = str(decision.get("reason") or "").strip()
    if reason:
        result["program_reason"] = reason[:500]
    if decision.get("review_reason"):
        result["review_reason"] = str(decision["review_reason"])[:300]
    return result


def _llm_payload(
    template: dict[str, Any],
    decision: dict[str, Any],
    decision_evidence: dict[str, Any] | None,
) -> dict[str, Any]:
    text = (decision_evidence or {}).get("ocr") or {}
    quality = (decision_evidence or {}).get("model_heads", {}).get("quality") or {}
    return {
        "fixed_template": {
            key: template.get(key, "")
            for key in ("summary", "evidence_note", "limitation")
        },
        "facts": {
            "action_label": ACTION_LABELS.get(decision.get("action"), "未知"),
            "effective_risk": decision.get("effective_risk"),
            "risk_raw": decision.get("risk_raw"),
            "binary_state": decision.get("binary_state"),
            "conflict_type": decision.get("conflict_type"),
            "adjustments": list(decision.get("adjustments") or [])[:2],
            "quality_labels": list(quality.get("labels") or [])[:3],
            "text_status": text.get("status"),
            "text_verdict": (text.get("semantic_review") or {}).get("verdict"),
            "text_grounded": bool(text.get("grounded_violation")),
            "recommended_action_label": decision.get("recommended_action_label"),
            "review_reason": decision.get("review_reason"),
        },
    }


def generate_final_explanation(
    llm_model: Any,
    decision_evidence: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    """Fill the fixed explanation template; failures fall back to program text."""
    template = build_program_explanation(decision, decision_evidence)
    payload = _llm_payload(template, decision, decision_evidence)
    messages = [
        SystemMessage(content=EXPLANATION_SYSTEM_PROMPT),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False, default=str)),
    ]
    try:
        response = llm_model.invoke(messages)
        if getattr(response, "tool_calls", None):
            raise ExplanationValidationError("tool_call_forbidden")
        draft = validate_explanation_fields(
            ExplanationDraft.model_validate(
                _parse_json_object(_message_text(response))
            )
        )
        return {
            **template,
            "status": "ok",
            "source": "llm_template",
            "prompt_version": EXPLANATION_PROMPT_VERSION,
            **draft.model_dump(),
        }
    except Exception as exc:  # noqa: BLE001
        reason = str(exc) if isinstance(exc, ExplanationValidationError) else ""
        return {
            **template,
            "status": "fallback",
            "source": "program_template_fallback",
            "prompt_version": EXPLANATION_PROMPT_VERSION,
            "error": type(exc).__name__,
            "reason": reason,
        }


def compose_final_suggestion(
    decision: dict[str, Any],
    explanation_result: dict[str, Any],
) -> str:
    """Compose the authoritative action and the fixed explanation fields."""
    action = ACTION_LABELS.get(str(decision.get("action")), "未知")
    status = (
        "证据不确定，暂不认定违规"
        if decision.get("uncertain")
        else "已生成处置建议"
    )
    summary = _normalize(explanation_result.get("summary", ""))
    evidence_note = _normalize(explanation_result.get("evidence_note", ""))
    limitation = _normalize(explanation_result.get("limitation", ""))
    review_reason = _normalize(explanation_result.get("review_reason", ""))
    if not summary:
        summary = _summary_for_action(str(decision.get("action") or "unavailable"))
    if not evidence_note:
        evidence_note = str(decision.get("reason") or "程序已完成证据核对。").strip()
    details = "".join(
        item if not item or item.endswith(("。", "！", "？"))
        else item + "。"
        for item in (evidence_note, limitation)
    )
    review = ""
    recommended = explanation_result.get("review_suggestion")
    if decision.get("action") == "review" and recommended:
        review = f"复核提示：{recommended}。" + (
            review_reason
            if not review_reason or review_reason.endswith(("。", "！", "？"))
            else review_reason + "。"
        )
    return (
        f"审核结果：{status}；处置：{action}。"
        f"{summary}{details}{review}"
    )
