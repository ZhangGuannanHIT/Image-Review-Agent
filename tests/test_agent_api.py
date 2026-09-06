# -*- coding: utf-8 -*-

import io
import tempfile
import unittest

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from PIL import Image

from backend.api import app, init_service
from backend.mock import MockOCRClient, MockReviewEngine, RuleBasedMockChatModel
from backend.service import ReviewService


class FinalizeFirstModel:
    """先违规提交，用于证明调用守卫会拒绝并要求 CLIP。"""

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        has_clip = any(isinstance(m, ToolMessage) and m.name == "analyze_image" for m in messages)
        was_rejected = any(
            isinstance(m, ToolMessage) and "调用守卫拒绝" in str(m.content) for m in messages
        )
        if has_clip:
            return AIMessage(content="", tool_calls=[{
                "name": "finalize_review", "args": {"suggestion": "守卫测试通过，建议人工复核。"},
                "id": "final-2", "type": "tool_call",
            }])
        if was_rejected:
            return AIMessage(content="", tool_calls=[{
                "name": "analyze_image", "args": {}, "id": "clip-1", "type": "tool_call",
            }])
        return AIMessage(content="", tool_calls=[{
            "name": "finalize_review", "args": {"suggestion": "不应被接受"},
            "id": "final-1", "type": "tool_call",
        }])


def image_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(buffer, format="PNG")
    return buffer.getvalue()


class AgentAndAPITests(unittest.TestCase):
    def make_service(self, model=None):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        return ReviewService(
            model or RuleBasedMockChatModel(), MockReviewEngine(), MockOCRClient(),
            log_dir=self.temp.name, max_rounds=5, mock_mode=True,
        )

    def test_clip_guard_rejects_early_finalize(self):
        service = self.make_service(FinalizeFirstModel())
        result = service.review(image_bytes(), "test.png")
        self.assertEqual(result.suggestion, "守卫测试通过，建议人工复核。")

    def test_api_contract_and_validation(self):
        init_service(self.make_service(), max_image_mb=1)
        client = TestClient(app)
        response = client.post("/api/review", files={"image": ("test.png", image_bytes(), "image/png")})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.json()), {"id", "suggestion"})
        bad = client.post("/api/review", files={"image": ("bad.jpg", b"not-image", "image/jpeg")})
        self.assertEqual(bad.status_code, 400)
        self.assertEqual(set(bad.json()), {"code", "message", "id"})


if __name__ == "__main__":
    unittest.main()
