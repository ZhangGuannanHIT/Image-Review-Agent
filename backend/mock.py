# -*- coding: utf-8 -*-
"""开发与测试专用 Mock；不得用于正式指标或冒充真实审核。"""

from __future__ import annotations

import itertools
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from src.engine import ModelOutput
from src.labels import QUALITY_CLASSES


class MockReviewEngine:
    is_mock = True
    loaded = True

    def predict(self, image: bytes) -> ModelOutput:
        return ModelOutput(
            risk={"label": "L2_中度违规", "index": 2, "confidence": 0.82, "scores": {}},
            type={"labels": ["直接暴力（攻击进行中）"], "scores": {}, "thresholds": {}},
            injury={"label": "轻微（微量血迹/淤青）", "index": 1, "confidence": 0.71, "scores": {}},
            quality={
                "labels": ["清晰"],
                "scores": {label: 0.93 if label == "清晰" else 0.07 for label in QUALITY_CLASSES},
                "thresholds": {label: 0.5 for label in QUALITY_CLASSES},
            },
            conf=0.71,
            trained=False,
        )


class MockOCRClient:
    def __init__(self, texts: list[str] | None = None):
        self.texts = texts or []

    def recognize(self, image_bytes: bytes) -> dict[str, Any]:
        return {
            "success": True, "available": True, "provider": "mock",
            "model": "mock-ocr", "texts": list(self.texts), "boxes": [],
            "box_format": "normalized_xyxy", "request_id": "", "error": None,
        }


class RuleBasedMockChatModel:
    """最小工具调用模型：先调用 CLIP，再提交固定建议。"""

    def __init__(self):
        self._ids = itertools.count(1)

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        clip_messages = [m for m in tool_messages if getattr(m, "name", "") == "analyze_image"]
        if not clip_messages:
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": "analyze_image", "args": {},
                    "id": f"mock-{next(self._ids)}", "type": "tool_call",
                }],
            )
        suggestion = "该图片模拟判定为中度违规，涉及直接暴力，建议下架并由运营复核。"
        return AIMessage(
            content="",
            tool_calls=[{
                "name": "finalize_review", "args": {"suggestion": suggestion},
                "id": f"mock-{next(self._ids)}", "type": "tool_call",
            }],
        )
