# -*- coding: utf-8 -*-

import unittest
import io

from PIL import Image

from backend.mock import MockOCRClient
from backend.policy import DecisionPolicy, decide
from backend.text_evidence import (
    assess_text_evidence, build_decision_evidence, classify_ocr_content,
    collect_knowledge_hits,
)
from backend.tools import (
    build_tools, is_l0_without_conflict, reset_current_image, set_current_image,
    tools_by_name,
)


def evidence(risk=0, active=None):
    from src.labels import INJURY_CLASSES, RISK_CLASSES, TYPE_CLASSES

    active = active or [TYPE_CLASSES[0]]
    return {
        "trained": True,
        "risk": {
            "label": RISK_CLASSES[risk],
            "confidence": 0.9,
            "scores": {label: (0.9 if i == risk else 0.1 / 4) for i, label in enumerate(RISK_CLASSES)},
        },
        "injury": {
            "label": INJURY_CLASSES[0],
            "scores": {INJURY_CLASSES[0]: 0.9, INJURY_CLASSES[1]: 0.05, INJURY_CLASSES[2]: 0.03, INJURY_CLASSES[3]: 0.02},
        },
        "type": {
            "labels": active,
            "thresholds": {label: 0.5 for label in TYPE_CLASSES},
            "scores": {label: (0.9 if label in active else 0.01) for label in TYPE_CLASSES},
        },
        "quality": {"labels": ["清晰"]},
    }


def png_bytes():
    output = io.BytesIO()
    Image.new("RGB", (16, 16), "white").save(output, format="PNG")
    return output.getvalue()


class TextEvidenceTests(unittest.TestCase):
    def test_low_confidence_l0_does_not_short_circuit(self):
        item = evidence(0)
        item["risk"]["confidence"] = 0.5
        self.assertFalse(is_l0_without_conflict(item))

    def test_l0_text_distortion_does_not_short_circuit(self):
        item = evidence(0)
        item["quality"]["labels"] = ["文字扭曲形变"]
        self.assertFalse(is_l0_without_conflict(item))

    def test_empty_ocr_is_not_text_distortion(self):
        result = assess_text_evidence({
            "success": True,
            "texts": [],
            "mean_confidence": None,
        })
        self.assertEqual(result["status"], "empty")
        self.assertFalse(result["has_text"])

    def test_single_character_is_partial(self):
        result = assess_text_evidence({
            "success": True,
            "texts": ["杀"],
            "mean_confidence": 0.95,
        })
        self.assertEqual(result["status"], "partial")

    def test_long_local_ocr_without_confidence_is_readable(self):
        result = assess_text_evidence({
            "success": True,
            "texts": ["local OCR text without confidence"],
            "mean_confidence": None,
        })
        self.assertEqual(result["status"], "readable")

    def test_local_ocr_caption_is_identified_separately_from_literal_text(self):
        caption = classify_ocr_content(
            "The image shows a bowl of tomato sauce and red paint."
        )
        self.assertTrue(caption["descriptive_caption"])
        self.assertTrue(caption["safe_food_scene"])
        self.assertTrue(caption["safe_red_liquid"])

        literal = classify_ocr_content("POLICE")
        self.assertFalse(literal["descriptive_caption"])
        self.assertFalse(literal["safe_food_scene"])

    def test_food_caption_is_not_safe_when_it_describes_visible_harm(self):
        caption = classify_ocr_content(
            "The image shows blood on a plate during an attack."
        )
        self.assertTrue(caption["descriptive_caption"])
        self.assertFalse(caption["safe_food_scene"])
        self.assertFalse(caption["safe_red_liquid"])

    def test_readable_text_gets_rag_rules(self):
        result = assess_text_evidence(
            {
                "success": True,
                "texts": ["I WILL HURT YOU"],
                "mean_confidence": 0.9,
            },
            retrieve=lambda query, k=5: [{"id": "R006", "type": "regulation", "score": 0.5, "verdict": "高风险"}],
        )
        self.assertEqual(result["status"], "readable")
        self.assertEqual(result["high_risk_rule_ids"], ["R006"])

    def test_ocr_safe_context_alone_does_not_override_visual_violation(self):
        item = evidence(2, active=["直接暴力（攻击进行中）"])
        item["text_evidence"] = {
            "status": "readable",
            "safe_context_rule_ids": ["R009"],
            "high_risk_rule_ids": [],
        }
        result = decide(item, DecisionPolicy())
        self.assertEqual(result["action"], "limit")

    def test_high_risk_ocr_text_limits_safe_image(self):
        item = evidence(0)
        item["text_evidence"] = {
            "status": "readable",
            "safe_context_rule_ids": [],
            "grounded_violation": True,
            "high_risk_rule_ids": ["R006"],
        }
        self.assertEqual(decide(item, DecisionPolicy())["action"], "limit")

    def test_rag_high_risk_hit_without_grounding_requires_review(self):
        item = evidence(0)
        item["text_evidence"] = {
            "status": "readable",
            "safe_context_rule_ids": [],
            "high_risk_rule_ids": ["R006"],
            "grounded_violation": False,
        }
        result = decide(item, DecisionPolicy())
        self.assertEqual(result["action"], "review")
        self.assertEqual(result["conflict_type"], "text_semantic_unresolved")

    def test_quality_text_distortion_alone_does_not_trigger_ocr(self):
        class Engine:
            def predict(self, image):
                return {
                    **evidence(2),
                    "quality": {"labels": ["文字扭曲形变"], "scores": {}, "thresholds": {}},
                }

        class Detector:
            method = "test-empty"

            def detect(self, image):
                return {
                    "status": "empty",
                    "has_text": False,
                    "boxes": [],
                    "box_count": 0,
                    "method": self.method,
                }

        token = set_current_image(png_bytes())
        try:
            tools = tools_by_name(build_tools(
                Engine(), MockOCRClient(["测试文字"]), text_detector=Detector()
            ))
            result = tools["ocr_image"].invoke({})
            self.assertIn("跳过 OCR", result)
            self.assertNotIn("测试文字", result)
        finally:
            reset_current_image(token)

    def test_collects_and_deduplicates_cached_knowledge(self):
        hits = collect_knowledge_hits({
            "search_knowledge:a": [
                {"id": "R006", "type": "regulation", "score": 0.4},
                {"id": "R009", "type": "ocr_boundary", "score": 0.3},
            ],
            "search_knowledge:b": [
                {"id": "R006", "type": "regulation", "score": 0.6},
            ],
            "other": [],
        })
        self.assertEqual([item["id"] for item in hits], ["R006", "R009"])
        self.assertEqual(hits[0]["score"], 0.6)

    def test_decision_evidence_bundle_keeps_all_sources(self):
        item = evidence(2)
        item["blood_binary"] = {"p_blood": 0.1, "threshold": 0.3}
        item["lowlevel"] = {"status": "ok", "features": {"edge_density": 0.2}}
        item["text_evidence"] = {"status": "empty"}
        item["scene_concepts"] = {"status": "skipped"}
        item["search_knowledge"] = [{"id": "R009", "score": 0.4}]
        bundle = build_decision_evidence(item)
        self.assertEqual(
            set(bundle), {
                "model_heads", "blood_binary", "anime_style",
                "text_detection", "lowlevel", "ocr", "scene_concepts",
                "search_knowledge",
            }
        )
        self.assertEqual(bundle["lowlevel"]["features"]["edge_density"], 0.2)
        self.assertEqual(bundle["ocr"]["status"], "empty")
        self.assertEqual(bundle["search_knowledge"][0]["id"], "R009")


if __name__ == "__main__":
    unittest.main()
