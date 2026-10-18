# -*- coding: utf-8 -*-
"""从 agent 运行结果中提取 suggestion 文本（含降级）。

优先读取受控状态图写入的 suggestion，再兼容最后一条 AI 消息。
"""

from __future__ import annotations

from typing import Any

FALLBACK_SUGGESTION = "未能生成审核建议，请稍后重试。"


def extract_suggestion(result: dict[str, Any]) -> str:
    """从 langgraph invoke 结果里抽取 suggestion。"""
    suggestion = str(result.get("suggestion", "") or "").strip()
    if suggestion:
        return suggestion
    messages = result.get("messages", []) or []
    for m in reversed(messages):
        if getattr(m, "type", "") == "ai" and getattr(m, "content", ""):
            return str(m.content)

    return FALLBACK_SUGGESTION
