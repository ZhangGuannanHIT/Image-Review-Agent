# -*- coding: utf-8 -*-

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from backend.api import app, init_service
from backend.mock import MockOCRClient, MockReviewEngine, RuleBasedMockChatModel
from backend.service import ReviewService
from tests.test_review_policy import evidence


class StaticDetector:
    method = "static-test"

    def __init__(self, has_text=False, fail=False):
        self.has_text = has_text
        self.fail = fail
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        if self.fail:
            raise RuntimeError("detector failed")
        return {
            "status": "detected" if self.has_text else "empty",
            "has_text": self.has_text,
            "boxes": [[0.1, 0.2, 0.8, 0.4]] if self.has_text else [],
            "box_count": 1 if self.has_text else 0,
            "confidence": 0.8 if self.has_text else 0.0,
            "method": self.method,
        }


class CountingOCR(MockOCRClient):
    def __init__(self, texts=None):
        super().__init__(texts)
        self.calls = 0

    def recognize(self, image_bytes):
        self.calls += 1
        return super().recognize(image_bytes)


class StaticSemanticReviewer:
    def review(self, text, references=None):
        if "I WILL KILL YOU" in text:
            return {
                "status": "ok",
                "verdict": "violation",
                "prompt_version": "test",
                "quote": "I WILL KILL YOU",
                "target": "viewer",
                "harm_action": "kill",
                "stance": "threat",
                "reason": "明确威胁",
            }
        return {
            "status": "ok",
            "verdict": "safe",
            "prompt_version": "test",
            "quote": "",
            "target": "",
            "harm_action": "",
            "stance": "",
            "reason": "普通文字",
        }


class StaticAnimeVisualReviewer:
    def __init__(self):
        self.calls = 0

    def review(self, image):
        self.calls += 1
        return {
            "status": "ok",
            "injury": "absent",
            "attack": "absent",
            "threat": "absent",
            "weapon_visible": "absent",
            "appearance": "cartoon",
            "threat_basis": "none",
            "threat_target": "",
            "weapon_pose": "",
            "reason": "动画战斗 pose，无实际伤害关系",
        }


class FixedEngine:
    trained = True
    ckpt_dir = ""
    type_thresholds = {}
    quality_thresholds = {}

    def __init__(self, item):
        self.item = item

    def predict(self, image):
        return self.item


class FailingEngine(FixedEngine):
    def predict(self, image):
        raise RuntimeError("CLIP unavailable")


def image_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(buffer, format="PNG")
    return buffer.getvalue()


class FixedWorkflowAndAPITests(unittest.TestCase):
    def make_service(self, **kwargs):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        kwargs.setdefault("text_detector", StaticDetector(False))
        kwargs.setdefault("text_semantic_reviewer", StaticSemanticReviewer())
        service = ReviewService(
            RuleBasedMockChatModel(),
            kwargs.pop("engine", MockReviewEngine()),
            kwargs.pop("ocr_client", MockOCRClient()),
            log_dir=temporary.name,
            mock_mode=True,
            **kwargs,
        )
        return service, temporary

    def test_service_has_fixed_workflow_and_no_react_agent(self):
        service, _ = self.make_service()
        self.assertTrue(service.ready)
        self.assertTrue(hasattr(service, "workflow"))
        self.assertFalse(hasattr(service, "agent"))

    def test_detector_empty_skips_ocr(self):
        detector = StaticDetector(False)
        ocr = CountingOCR(["should not be read"])
        service, directory = self.make_service(
            engine=FixedEngine(evidence(0)),
            text_detector=detector,
            ocr_client=ocr,
        )
        result = service.review(image_bytes(), "empty.png")
        self.assertIn("处置：放行", result.suggestion)
        self.assertEqual(ocr.calls, 0)
        self.assertEqual(detector.calls, 1)
        record = json.loads(
            (Path(directory.name) / "review.log.jsonl").read_text(encoding="utf-8")
        )
        self.assertTrue(any(
            step["type"] == "tool_skip" and step["name"] == "ocr_image"
            for step in record["steps"]
        ))

    def test_detector_detects_text_even_for_l0_and_runs_ocr_and_rag(self):
        ocr = CountingOCR(["I WILL KILL YOU"])
        service, directory = self.make_service(
            engine=FixedEngine(evidence(0)),
            text_detector=StaticDetector(True),
            ocr_client=ocr,
            retriever=lambda query, k=5: [{
                "id": "R006",
                "type": "regulation",
                "score": 0.8,
                "verdict": "明确死亡威胁",
                "source": {"locator_status": "attachment_only"},
            }],
        )
        result = service.review(image_bytes(), "text.png")
        self.assertIn("处置：限流", result.suggestion)
        self.assertEqual(ocr.calls, 1)
        record = json.loads(
            (Path(directory.name) / "review.log.jsonl").read_text(encoding="utf-8")
        )
        self.assertEqual(record["decision"]["text_evidence"]["high_risk_rule_ids"], ["R006"])
        self.assertNotIn("I WILL KILL YOU", json.dumps(record, ensure_ascii=False))
        self.assertEqual(len(record["decision"]["text_evidence"]["text_sha256"]), 64)

    def test_detector_failure_runs_ocr_and_marks_degraded(self):
        ocr = CountingOCR(["ordinary text"])
        service, directory = self.make_service(
            engine=FixedEngine(evidence(0)),
            text_detector=StaticDetector(fail=True),
            ocr_client=ocr,
        )
        service.review(image_bytes(), "failed-detector.png")
        self.assertEqual(ocr.calls, 1)
        record = json.loads(
            (Path(directory.name) / "review.log.jsonl").read_text(encoding="utf-8")
        )
        self.assertTrue(record["degraded"])
        self.assertEqual(record["text_detection"]["status"], "failed")

    def test_narrow_conditional_threat_rule_can_limit_without_llm_violation(self):
        service, _ = self.make_service(
            engine=FixedEngine(evidence(0)),
            text_detector=StaticDetector(True),
            ocr_client=CountingOCR(["PLEASE LOOT! I LOVE TO SHOOT"]),
        )
        result = service.review(image_bytes(), "conditional-threat.png")
        self.assertIn("处置：限流", result.suggestion)

    def test_anime_visual_review_withdraws_anime_limit_only(self):
        item = evidence(2, 2, .8, ["直接暴力（攻击进行中）"])
        item["anime_style"] = {
            "score": 0.99,
            "threshold": 0.90,
            "is_anime": True,
        }
        reviewer = StaticAnimeVisualReviewer()
        service, directory = self.make_service(
            engine=FixedEngine(item),
            anime_visual_reviewer=reviewer,
            anime_visual_enabled=True,
        )
        result = service.review(image_bytes(), "anime.png")
        self.assertIn("处置：放行", result.suggestion)
        self.assertEqual(reviewer.calls, 1)
        record = json.loads(
            (Path(directory.name) / "review.log.jsonl").read_text(encoding="utf-8")
        )
        self.assertEqual(
            record["decision"]["anime_visual_effect"], "withdraw_to_allow"
        )

    def test_real_high_confidence_route_does_not_call_anime_visual_review(self):
        item = evidence(2, 2, .8, ["直接暴力（攻击进行中）"])
        item["anime_style"] = {
            "score": 0.01,
            "threshold": 0.90,
            "is_anime": False,
        }
        reviewer = StaticAnimeVisualReviewer()
        service, _ = self.make_service(
            engine=FixedEngine(item),
            anime_visual_reviewer=reviewer,
            anime_visual_enabled=True,
        )
        service.review(image_bytes(), "real.png")
        self.assertEqual(reviewer.calls, 0)

    def test_fixed_workflow_order_and_non_l0_scene_gate(self):
        class Scorer:
            calls = 0

            def score(self, image):
                self.calls += 1
                return {
                    "concepts": {
                        "direct_attack": {
                            "title": "直接攻击", "kind": "risk", "score": 0.02,
                            "max": 0.02, "rag_query": "", "rag_ids": [],
                        },
                        "anime_illustration": {
                            "title": "二次元", "kind": "domain", "score": 0.03,
                            "max": 0.03, "rag_query": "", "rag_ids": [],
                        },
                        "real_photo": {
                            "title": "真实照片", "kind": "domain", "score": -0.01,
                            "max": -0.01, "rag_query": "", "rag_ids": [],
                        },
                    }
                }

        scorer = Scorer()
        item = evidence(2, 2, .8, ["直接暴力（攻击进行中）"])
        service, directory = self.make_service(
            engine=FixedEngine(item),
            scene_scorer=scorer,
            scene_concepts_enabled=True,
        )
        service.review(image_bytes(), "sequence.png")
        record = json.loads(
            (Path(directory.name) / "review.log.jsonl").read_text(encoding="utf-8")
        )
        called = [step["name"] for step in record["steps"] if step["type"] == "tool_call"]
        self.assertEqual(called, ["analyze_image", "detect_text", "match_scene_concepts"])
        self.assertEqual(record["workflow_version"], "2026-10-26-fixed-evidence-v1")

    def test_clip_failure_does_not_run_other_tools(self):
        service, directory = self.make_service(
            engine=FailingEngine(evidence(0)),
            ocr_client=CountingOCR(["secret"]),
        )
        response = service.review(image_bytes(), "broken.png")
        self.assertIn("暂不可用", response.suggestion)
        record = json.loads(
            (Path(directory.name) / "review.log.jsonl").read_text(encoding="utf-8")
        )
        called = [step["name"] for step in record["steps"] if step["type"] == "tool_call"]
        self.assertEqual(called, [])

    def test_api_contract_and_validation(self):
        service, _ = self.make_service()
        init_service(service, max_image_mb=1)
        client = TestClient(app)
        response = client.post(
            "/api/review",
            files={"image": ("test.png", image_bytes(), "image/png")},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.json()), {"id", "suggestion"})
        bad = client.post(
            "/api/review",
            files={"image": ("bad.jpg", b"not-image", "image/jpeg")},
        )
        self.assertEqual(bad.status_code, 400)
        self.assertEqual(set(bad.json()), {"code", "message", "id"})

    def test_decompression_bomb_is_invalid_image(self):
        service, _ = self.make_service()
        init_service(service, max_image_mb=1)
        client = TestClient(app)
        from PIL.Image import DecompressionBombError
        with patch("backend.api.Image.open", side_effect=DecompressionBombError("too many pixels")):
            response = client.post(
                "/api/review",
                files={"image": ("huge.png", image_bytes(), "image/png")},
            )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["code"], 40001)
        self.assertIn("像素尺寸过大", response.json()["message"])

    def test_separate_local_frontend_cors_and_static_files(self):
        service, _ = self.make_service()
        init_service(service, max_image_mb=1)
        client = TestClient(app)
        for origin in ["http://127.0.0.1:5500", "http://localhost:5500"]:
            response = client.options(
                "/api/review",
                headers={
                    "Origin": origin,
                    "Access-Control-Request-Method": "POST",
                },
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["access-control-allow-origin"], origin)
        denied = client.options(
            "/api/review",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        self.assertNotIn("access-control-allow-origin", denied.headers)
        page = client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("ImageGuard", page.text)
        self.assertEqual(client.get("/styles.css").status_code, 200)
        self.assertEqual(client.get("/app.js").status_code, 200)
        health = client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertTrue(health.json()["service_ready"])


if __name__ == "__main__":
    unittest.main()
