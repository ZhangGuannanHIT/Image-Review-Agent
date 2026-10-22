# -*- coding: utf-8 -*-

import unittest
import io

from PIL import Image

from backend.policy import DecisionPolicy, decide
from backend.scene_concepts import build_scene_evidence, load_scene_concepts
from backend.tools import build_tools, reset_current_image, set_current_image, tools_by_name


def fake_score_result():
    concepts = {}
    for key, title, kind, score in [
        ("weapon_threat", "武器威胁", "risk", 0.03),
        ("direct_attack", "直接攻击", "risk", 0.02),
        ("medical_context", "医学语境", "safe_context", 0.01),
        ("red_paint_liquid", "红色颜料", "confounder", 0.0),
        ("anime_illustration", "二次元", "domain", 0.01),
    ]:
        concepts[key] = {
            "title": title,
            "kind": kind,
            "score": score,
            "max": score,
            "rag_query": "武器威胁" if kind == "risk" else "医学语境",
            "rag_ids": ["R006"] if key == "weapon_threat" else (
                ["R009"] if key == "medical_context" else []
            ),
        }
    return {"concepts": concepts, "top": []}


def png_bytes():
    output = io.BytesIO()
    Image.new("RGB", (16, 16), "white").save(output, format="PNG")
    return output.getvalue()


class SceneConceptTests(unittest.TestCase):
    def test_prompt_bank_structure(self):
        concepts = load_scene_concepts()
        self.assertGreaterEqual(sum(len(item["prompts"]) for item in concepts), 25)
        self.assertLessEqual(sum(len(item["prompts"]) for item in concepts), 35)
        self.assertEqual(len({item["key"] for item in concepts}), len(concepts))
        kinds = {item["kind"] for item in concepts}
        self.assertEqual(kinds, {"risk", "safe_context", "confounder", "domain"})

    def test_scene_evidence_detects_conflict(self):
        result = build_scene_evidence(
            fake_score_result(),
            risk_margin_threshold=-0.001,
            safe_context_threshold=0.006,
            retrieve=lambda query, k=3: [
                {"id": "R006", "type": "regulation", "score": 0.5},
                {"id": "R015", "type": "regulation", "score": 0.9},
                {"id": "R009", "type": "ocr_boundary", "score": 0.5},
            ],
        )
        self.assertTrue(result["risk_matched"])
        self.assertTrue(result["safe_context_matched"])
        self.assertTrue(result["context_conflict"])
        self.assertEqual(
            {item["id"] for item in result["knowledge_hits"]}, {"R006", "R009"}
        )

    def test_scene_tool_skips_l0_before_scoring(self):
        class Scorer:
            calls = 0

            def score(self, image):
                self.calls += 1
                return fake_score_result()

        class Engine:
            def predict(self, image):
                from src.labels import INJURY_CLASSES, RISK_CLASSES, TYPE_CLASSES

                return {
                    "trained": True,
                    "risk": {
                        "label": "L0_安全",
                        "confidence": 0.9,
                        "scores": {label: (0.9 if i == 0 else 0.025) for i, label in enumerate(RISK_CLASSES)},
                    },
                    "injury": {
                        "label": INJURY_CLASSES[0],
                        "scores": {INJURY_CLASSES[0]: 0.9, INJURY_CLASSES[1]: 0.04, INJURY_CLASSES[2]: 0.03, INJURY_CLASSES[3]: 0.03},
                    },
                    "type": {
                        "labels": [TYPE_CLASSES[0]],
                        "thresholds": {label: 0.5 for label in TYPE_CLASSES},
                        "scores": {TYPE_CLASSES[0]: 0.9, TYPE_CLASSES[1]: 0.01, TYPE_CLASSES[2]: 0.01, TYPE_CLASSES[3]: 0.01},
                    },
                    "quality": {"labels": ["清晰"]},
                }

        scorer = Scorer()
        token = set_current_image(png_bytes())
        try:
            tools = tools_by_name(build_tools(Engine(), scene_scorer=scorer))
            tools["analyze_image"].invoke({})
            result = tools["match_scene_concepts"].invoke({})
            self.assertIn("risk_l0", result)
            self.assertEqual(scorer.calls, 0)
        finally:
            reset_current_image(token)

    def test_non_l0_auto_calls_scene_tool(self):
        class Scorer:
            calls = 0

            def score(self, image):
                self.calls += 1
                return fake_score_result()

        class Engine:
            def predict(self, image):
                from tests.test_text_evidence import evidence

                item = evidence(2, active=["直接暴力（攻击进行中）"])
                return item

        scorer = Scorer()
        token = set_current_image(png_bytes())
        try:
            tools = tools_by_name(build_tools(Engine(), scene_scorer=scorer))
            content = tools["analyze_image"].invoke({})
            self.assertIn("自动调用场景概念匹配", content)
            self.assertEqual(scorer.calls, 1)
        finally:
            reset_current_image(token)

    def scene_evidence(self, kind):
        return {
            "safe_context_matched": True,
            "safe_contexts": [{"key": kind, "title": kind, "score": 0.02}],
            "risk_scenes": [{"key": "direct_attack", "title": "直接攻击", "score": 0.03}],
        }

    def test_unconfirmed_scene_prompt_does_not_change_action(self):
        from tests.test_text_evidence import evidence

        item = evidence(3, active=["直接暴力（攻击进行中）"])
        item["scene_concepts"] = self.scene_evidence("news_legal_context")
        result = decide(item, DecisionPolicy())
        self.assertEqual(result["action"], "block")

    def test_confirmed_medical_context_downgrades_block_to_limit(self):
        from tests.test_text_evidence import evidence

        item = evidence(3, active=["直接暴力（攻击进行中）"])
        item["text_evidence"] = {
            "status": "readable",
            "safe_context_rule_ids": ["R009"],
            "high_risk_rule_ids": [],
        }
        item["scene_concepts"] = self.scene_evidence("medical_context")
        self.assertEqual(decide(item, DecisionPolicy())["action"], "limit")

    def test_confirmed_news_context_with_raw_gore_requires_review(self):
        from tests.test_text_evidence import evidence

        item = evidence(3, active=["直接暴力（攻击进行中）"])
        item["text_evidence"] = {
            "status": "readable",
            "safe_context_rule_ids": ["R009"],
            "high_risk_rule_ids": [],
        }
        item["scene_concepts"] = self.scene_evidence("news_legal_context")
        result = decide(item, DecisionPolicy())
        self.assertEqual(result["action"], "review")
        self.assertIn("新闻壳", result["reason"])


if __name__ == "__main__":
    unittest.main()
