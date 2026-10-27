# -*- coding: utf-8 -*-

import io
import json
import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from PIL import Image

from backend.explanation import EXPLANATION_PROMPT_VERSION
from backend.mock import RuleBasedMockChatModel
from backend.service import ReviewService
from tests.test_review_policy import evidence


def image_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), "white").save(buffer, format="PNG")
    return buffer.getvalue()


class BlockedEngine:
    trained = True

    def predict(self, image):
        return evidence(3, 3, 0.9)


class ExplanationAwareModel(RuleBasedMockChatModel):
    def __init__(self, explanation: str):
        super().__init__()
        self.explanation = explanation
        self.explanation_payloads = []

    def invoke(self, messages):
        system = next(
            (message for message in messages if isinstance(message, SystemMessage)),
            None,
        )
        if system and "解释生成器" in str(system.content):
            human = next(
                message for message in messages if isinstance(message, HumanMessage)
            )
            self.explanation_payloads.append(json.loads(str(human.content)))
            return AIMessage(
                content=json.dumps(
                    {"explanation": self.explanation}, ensure_ascii=False
                )
            )
        return super().invoke(messages)


class FinalExplanationTests(unittest.TestCase):
    def test_service_sends_full_evidence_and_program_keeps_action_authority(self):
        model = ExplanationAwareModel(
            "风险头与血腥辅助头方向一致，暴力类型也提供了独立支持，"
            "当前证据能够支撑该处置级别。"
        )
        with tempfile.TemporaryDirectory() as directory:
            service = ReviewService(
                model, BlockedEngine(), log_dir=directory, save_images=False
            )
            response = service.review(image_bytes(), "blocked.png")
            record = json.loads(
                (Path(directory) / "review.log.jsonl").read_text(encoding="utf-8")
            )

        self.assertIn("处置：下架", response.suggestion)
        self.assertIn("风险头与血腥辅助头", response.suggestion)
        self.assertEqual(record["decision"]["action"], "block")
        self.assertEqual(record["final_explanation"]["status"], "ok")
        self.assertEqual(
            record["final_explanation"]["prompt_version"],
            EXPLANATION_PROMPT_VERSION,
        )
        self.assertEqual(record["suggestion_source"], "llm_explanation")
        self.assertEqual(len(model.explanation_payloads), 1)
        payload = model.explanation_payloads[0]
        self.assertEqual(
            set(payload), {"decision_evidence", "program_decision"}
        )
        self.assertIn("model_heads", payload["decision_evidence"])
        self.assertEqual(
            payload["decision_evidence"]["lowlevel"]["status"], "not_called"
        )
        self.assertIn("text_detection", payload["decision_evidence"])
        self.assertEqual(payload["program_decision"]["action"], "block")

    def test_explanation_cannot_emit_another_action_and_falls_back(self):
        model = ExplanationAwareModel(
            "模型证据存在冲突，因此建议放行并继续观察后续结果。"
        )
        with tempfile.TemporaryDirectory() as directory:
            service = ReviewService(
                model, BlockedEngine(), log_dir=directory, save_images=False
            )
            response = service.review(image_bytes(), "blocked.png")
            record = json.loads(
                (Path(directory) / "review.log.jsonl").read_text(encoding="utf-8")
            )

        self.assertIn("处置：下架", response.suggestion)
        self.assertNotIn("建议放行", response.suggestion)
        self.assertEqual(record["final_explanation"]["status"], "fallback")
        self.assertEqual(
            record["final_explanation"]["reason"], "action_word_forbidden"
        )
        self.assertEqual(record["suggestion_source"], "deterministic_fallback")

    def test_explanation_can_be_disabled_without_changing_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            service = ReviewService(
                RuleBasedMockChatModel(),
                BlockedEngine(),
                log_dir=directory,
                save_images=False,
                explanation_enabled=False,
            )
            response = service.review(image_bytes(), "blocked.png")
            record = json.loads(
                (Path(directory) / "review.log.jsonl").read_text(encoding="utf-8")
            )

        self.assertIn("处置：下架", response.suggestion)
        self.assertEqual(record["final_explanation"]["status"], "disabled")
        self.assertEqual(record["suggestion_source"], "deterministic_fallback")


if __name__ == "__main__":
    unittest.main()
