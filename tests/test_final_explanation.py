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
                    {
                        "summary": self.explanation,
                        "evidence_note": "视觉候选与辅助证据方向一致。",
                        "limitation": "",
                    },
                    ensure_ascii=False,
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
        self.assertEqual(record["final_explanation"]["source"], "llm_template")
        self.assertEqual(
            record["final_explanation"]["prompt_version"],
            EXPLANATION_PROMPT_VERSION,
        )
        self.assertEqual(record["suggestion_source"], "llm_explanation")
        self.assertEqual(len(model.explanation_payloads), 1)
        payload = model.explanation_payloads[0]
        self.assertEqual(
            set(payload), {"fixed_template", "facts"}
        )
        self.assertIn("summary", payload["fixed_template"])
        self.assertEqual(payload["facts"]["action_label"], "下架")
        self.assertEqual(payload["facts"]["effective_risk"], "L3_严重违规")
        self.assertNotIn("decision_evidence", payload)

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
        self.assertEqual(
            record["final_explanation"]["source"],
            "program_template_fallback",
        )

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
        self.assertEqual(record["final_explanation"]["status"], "template")
        self.assertEqual(record["final_explanation"]["source"], "program_template")
        self.assertEqual(record["suggestion_source"], "program_template")
        self.assertEqual(
            record["final_explanation"]["skip_reason"],
            "llm_disabled",
        )

    def test_selective_mode_skips_llm_for_allow_and_uses_program_template(self):
        model = ExplanationAwareModel(
            "视觉候选与辅助证据方向一致，当前结论由程序规则生成。"
        )

        class AllowedEngine:
            trained = True

            def predict(self, image):
                return evidence(0)

        with tempfile.TemporaryDirectory() as directory:
            service = ReviewService(
                model,
                AllowedEngine(),
                log_dir=directory,
                save_images=False,
                explanation_mode="selective",
            )
            response = service.review(image_bytes(), "allowed.png")
            record = json.loads(
                (Path(directory) / "review.log.jsonl").read_text(encoding="utf-8")
            )

        self.assertIn("处置：放行", response.suggestion)
        self.assertEqual(len(model.explanation_payloads), 0)
        self.assertEqual(record["final_explanation"]["status"], "template")
        self.assertEqual(
            record["final_explanation"]["skip_reason"],
            "program_action",
        )
        self.assertEqual(record["suggestion_source"], "program_template")
        self.assertNotIn("explanation", record["timings_ms"])


if __name__ == "__main__":
    unittest.main()
