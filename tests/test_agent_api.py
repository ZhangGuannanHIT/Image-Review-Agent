# -*- coding: utf-8 -*-

import io
import json
from pathlib import Path
from unittest.mock import patch
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


class RagTraceModel:
    """按 CLIP、OCR、两次 RAG、提交顺序执行，用于检查日志审计与脱敏。"""

    queries = ["明确暴力威胁，联系电话13800138000", "暴力画面与文字冲突"]

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        tool_names = [m.name for m in messages if isinstance(m, ToolMessage)]
        if "analyze_image" not in tool_names:
            return AIMessage(content="", tool_calls=[{
                "name": "analyze_image", "args": {}, "id": "clip-log", "type": "tool_call",
            }])
        if "ocr_image" not in tool_names:
            return AIMessage(content="", tool_calls=[{
                "name": "ocr_image", "args": {}, "id": "ocr-log", "type": "tool_call",
            }])
        rag_calls = tool_names.count("search_knowledge")
        if rag_calls < len(self.queries):
            return AIMessage(content="", tool_calls=[{
                "name": "search_knowledge", "args": {"query": self.queries[rag_calls]},
                "id": f"rag-log-{rag_calls}", "type": "tool_call",
            }])
        return AIMessage(content="", tool_calls=[{
            "name": "finalize_review", "args": {"suggestion": "建议联系13800138000进行人工复核。"},
            "id": "final-log", "type": "tool_call",
        }])


class FailingRetriever:
    def __call__(self, query, k=5):
        raise RuntimeError("RAG unavailable")


class RagFailureModel:
    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        tool_names = [m.name for m in messages if isinstance(m, ToolMessage)]
        if "analyze_image" not in tool_names:
            return AIMessage(content="", tool_calls=[{
                "name": "analyze_image", "args": {}, "id": "clip-fail", "type": "tool_call",
            }])
        if "search_knowledge" not in tool_names:
            return AIMessage(content="", tool_calls=[{
                "name": "search_knowledge", "args": {"query": "暴力规则"},
                "id": "rag-fail", "type": "tool_call",
            }])
        return AIMessage(content="", tool_calls=[{
            "name": "finalize_review", "args": {"suggestion": "RAG不可用，建议人工复核。"},
            "id": "final-fail", "type": "tool_call",
        }])


class AnalyzeThenContinueModel:
    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        tool_names = [m.name for m in messages if isinstance(m, ToolMessage)]
        if "analyze_image" not in tool_names:
            return AIMessage(content="", tool_calls=[{
                "name": "analyze_image", "args": {}, "id": "clip-short", "type": "tool_call",
            }])
        return AIMessage(content="", tool_calls=[{
            "name": "search_knowledge", "args": {"query": "应被短路跳过"},
            "id": "rag-short", "type": "tool_call",
        }])


class L0Engine:
    trained = True

    def predict(self, image):
        from tests.test_text_evidence import evidence

        return evidence(0)


def image_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(buffer, format="PNG")
    return buffer.getvalue()


class AgentAndAPITests(unittest.TestCase):
    def test_separate_local_frontend_cors(self):
        client = TestClient(app)
        for origin in ["http://127.0.0.1:5500", "http://localhost:5500"]:
            response = client.options("/api/review", headers={
                "Origin": origin, "Access-Control-Request-Method": "POST"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["access-control-allow-origin"], origin)
        denied = client.options("/api/review", headers={
            "Origin": "http://example.com", "Access-Control-Request-Method": "POST"})
        self.assertNotIn("access-control-allow-origin", denied.headers)

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
        self.assertIn("处置：人工复核", result.suggestion)
        import json
        from pathlib import Path
        log = json.loads((Path(self.temp.name) / "review.log.jsonl").read_text(encoding="utf-8"))
        self.assertTrue(any(step["type"] == "guard_reject" for step in log["steps"]))
        self.assertTrue(log["clip_succeeded"])
        self.assertEqual(log["agent_suggestion"], "守卫测试通过，建议人工复核。")

    def test_api_contract_and_validation(self):
        init_service(self.make_service(), max_image_mb=1)
        client = TestClient(app)
        response = client.post("/api/review", files={"image": ("test.png", image_bytes(), "image/png")})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.json()), {"id", "suggestion"})
        bad = client.post("/api/review", files={"image": ("bad.jpg", b"not-image", "image/jpeg")})
        self.assertEqual(bad.status_code, 400)
        self.assertEqual(set(bad.json()), {"code", "message", "id"})


    def test_decompression_bomb_is_invalid_image(self):
        init_service(self.make_service(), max_image_mb=1)
        client = TestClient(app)
        from PIL.Image import DecompressionBombError
        with patch("backend.api.Image.open", side_effect=DecompressionBombError("too many pixels")):
            response = client.post(
                "/api/review", files={"image": ("huge.png", image_bytes(), "image/png")}
            )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["code"], 40001)
        self.assertIn("像素尺寸过大", response.json()["message"])

    def test_static_frontend_is_served_same_origin(self):
        init_service(self.make_service(), max_image_mb=1)
        client = TestClient(app)
        page = client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("ImageGuard", page.text)
        self.assertIn('href="./styles.css"', page.text)
        self.assertIn('href="./landing.css"', page.text)
        self.assertIn('src="./app.js"', page.text)
        css = client.get("/styles.css")
        self.assertEqual(css.status_code, 200)
        self.assertIn("text/css", css.headers["content-type"])
        landing_css = client.get("/landing.css")
        self.assertEqual(landing_css.status_code, 200)
        self.assertIn("landing-hero", landing_css.text)
        js = client.get("/app.js")
        self.assertEqual(js.status_code, 200)
        self.assertIn("api/review", js.text)
        declared_ids = set(__import__("re").findall(r'id="([^"]+)"', page.text))
        required_ids = set(__import__("re").findall(r'"([A-Za-z][A-Za-z0-9]+)"', js.text.split("var el = {};", 1)[1].split(".forEach", 1)[0]))
        self.assertFalse(required_ids - declared_ids)
        # 健康检查与审核接口不受静态路由影响
        health = client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertIn("service_ready", health.json())

    def test_rag_query_is_not_written_to_log(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        service = ReviewService(
            RagTraceModel(),
            MockReviewEngine(),
            MockOCRClient(["联系电话13800138000"]),
            retriever=lambda query, k=5: [{
                "id": "REG-TEST", "type": "regulation", "score": 0.8,
                "title": "测试规则", "content": "测试正文",
                "source": {"locator_status": "site_search_only"},
            }],
            log_dir=temp.name,
            max_rounds=5,
            mock_mode=True,
            rag_enabled=True,
        )
        response = service.review(image_bytes(), "用户13800138000.png")
        self.assertIn("处置：", response.suggestion)
        # The Agent result is retained for audit; main policy still decides the response.
        log_text = (Path(temp.name) / "review.log.jsonl").read_text(encoding="utf-8")
        for query in RagTraceModel.queries:
            self.assertNotIn(query, log_text)
        self.assertNotIn("13800138000", log_text)
        self.assertNotIn("测试正文", log_text)
        record = json.loads(log_text)
        cache_keys = [key for key in record["tool_cache_keys"] if key.startswith("search_knowledge:")]
        self.assertEqual(len(cache_keys), 2)
        self.assertEqual(record["tool_results"]["ocr_image"]["text_count"], 1)
        self.assertEqual(len(record["tool_results"]["search_knowledge"]), 2)
        self.assertEqual(
            {match["id"] for matches in record["tool_results"]["search_knowledge"].values() for match in matches},
            {"REG-TEST"},
        )

    def test_optional_tool_failure_marks_review_degraded(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        service = ReviewService(
            RagFailureModel(), MockReviewEngine(), MockOCRClient(),
            retriever=FailingRetriever(), log_dir=temp.name, mock_mode=True, rag_enabled=True,
        )
        service.review(image_bytes(), "test.png")
        record = json.loads((Path(temp.name) / "review.log.jsonl").read_text(encoding="utf-8"))
        self.assertTrue(record["degraded"])
        self.assertTrue(any(step["type"] == "tool_error" for step in record["steps"]))

    def test_l0_short_circuits_optional_tools(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        service = ReviewService(
            AnalyzeThenContinueModel(), L0Engine(), MockOCRClient(),
            log_dir=temp.name, mock_mode=False, rag_enabled=True,
        )
        result = service.review(image_bytes(), "l0.png")
        self.assertIn("处置：放行", result.suggestion)
        record = json.loads((Path(temp.name) / "review.log.jsonl").read_text(encoding="utf-8"))
        called = [step["name"] for step in record["steps"] if step["type"] == "tool_call"]
        self.assertEqual(called, ["analyze_image"])
        self.assertTrue(record["finalized"])


if __name__ == "__main__":
    unittest.main()
