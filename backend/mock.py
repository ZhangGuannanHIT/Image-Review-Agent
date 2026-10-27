# -*- coding: utf-8 -*-
"""开发与测试专用 Mock；不得用于正式指标或冒充真实审核。"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

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
    """Minimal explanation-only model used by local tests."""

    def invoke(self, messages):
        return AIMessage(
            content=(
                '{"explanation":"视觉候选与辅助证据已完成核对，'
                '当前结论由程序处置规则生成。"}'
            )
        )
