# -*- coding: utf-8 -*-
"""在程序完成处置后，用完整证据生成不可改变动作的解释。"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from .policy import render


EXPLANATION_PROMPT_VERSION = "2026-10-23-v1"
ACTION_LABELS = {
    "allow": "放行",
    "limit": "限流",
    "review": "人工复核",
    "block": "下架",
}

EXPLANATION_SYSTEM_PROMPT = """你是图片审核结果解释生成器，不是决策器。
程序已经根据结构化证据完成最终处置，该处置不可修改、不可质疑、不可重新裁决。

你的任务：
1. 只能使用输入中的 decision_evidence 和 program_decision，不得补充图片中未提供的事实。
2. 将模型输出表述为候选证据或预测，不得说成已经亲眼确认的具体画面。
3. 概括支持程序处置的主要证据，并说明证据冲突、质量限制或不确定性。
4. OCR 未执行、无文字、失败或残缺均不能推出图片一定没有文字。
5. RAG 无命中不代表安全；场景候选单独出现不代表安全语境已经确认。
6. 不得自行输出“放行、限流、人工复核、下架”中的任何动作词，程序会统一添加权威处置。
7. 使用 2 至 4 句、面向审核运营的中文，避免堆砌全部分数和内部字段名。

只输出一个 JSON 对象，不要使用 Markdown：
{"explanation":"解释正文"}
"""


class ExplanationDraft(BaseModel):
    explanation: str


class ExplanationValidationError(ValueError):
    """解释未通过程序约束。"""


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


def validate_explanation(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not 20 <= len(normalized) <= 500:
        raise ExplanationValidationError("invalid_length")
    if any(label in normalized for label in ACTION_LABELS.values()):
        raise ExplanationValidationError("action_word_forbidden")
    if "以上为模型预测" in normalized:
        raise ExplanationValidationError("fixed_footer_forbidden")
    return normalized


def generate_final_explanation(
    llm_model: Any,
    decision_evidence: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    """调用无工具 LLM 生成解释；任何异常都返回可审计的回退状态。"""
    payload = {
        "decision_evidence": decision_evidence,
        "program_decision": decision,
    }
    messages = [
        SystemMessage(content=EXPLANATION_SYSTEM_PROMPT),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False, default=str)),
    ]
    try:
        response = llm_model.invoke(messages)
        if getattr(response, "tool_calls", None):
            raise ExplanationValidationError("tool_call_forbidden")
        draft = ExplanationDraft.model_validate(
            _parse_json_object(_message_text(response))
        )
        explanation = validate_explanation(draft.explanation)
        return {
            "status": "ok",
            "prompt_version": EXPLANATION_PROMPT_VERSION,
            "explanation": explanation,
        }
    except Exception as exc:  # noqa: BLE001
        reason = str(exc) if isinstance(exc, ExplanationValidationError) else ""
        return {
            "status": "fallback",
            "prompt_version": EXPLANATION_PROMPT_VERSION,
            "error": type(exc).__name__,
            "reason": reason,
        }


def compose_final_suggestion(
    decision: dict[str, Any],
    explanation_result: dict[str, Any],
) -> str:
    """拼接程序权威处置和 LLM 解释；失败时使用确定性模板。"""
    if explanation_result.get("status") != "ok":
        return render(decision)
    action = ACTION_LABELS[decision["action"]]
    status = (
        "证据不确定，暂不认定违规"
        if decision.get("uncertain")
        else "已生成处置建议"
    )
    explanation = str(explanation_result["explanation"]).strip()
    return (
        f"审核结果：{status}；处置：{action}。"
        f"{explanation}程序依据：{decision['reason']}。"
    )
