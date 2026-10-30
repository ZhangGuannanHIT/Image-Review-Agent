# -*- coding: utf-8 -*-

import io
import unittest

from PIL import Image

from backend.policy import DecisionPolicy, decide
from backend.scene_concepts import build_scene_evidence
from src.labels import INJURY_CLASSES, RISK_CLASSES, TYPE_CLASSES


def evidence(risk=0, injury=0, confidence=0.9, active=None):
    def single(classes, index):
        return {
            "label": classes[index],
            "confidence": confidence,
            "scores": {
                label: confidence if i == index else (1 - confidence) / (len(classes) - 1)
                for i, label in enumerate(classes)
            },
        }

    active = active or [TYPE_CLASSES[0]]
    return {
        "trained": True,
        "risk": single(RISK_CLASSES, risk),
        "injury": single(INJURY_CLASSES, injury),
        "type": {
            "labels": active,
            "thresholds": dict(zip(TYPE_CLASSES, [.45, .6, .6, .9])),
            "scores": {
                name: 0.95 if name in active else 0.01 for name in TYPE_CLASSES
            },
        },
        "quality": {"labels": ["清晰"]},
    }


def binary(score: float, threshold: float = 0.30):
    return {
        "p_blood": score,
        "threshold": threshold,
        "label": "有血腥" if score >= threshold else "无血腥",
        "trained": True,
    }


def anime_scene(anime: bool = True):
    return {
        "safe_context_matched": False,
        "safe_contexts": [],
        "risk_scenes": [],
        "confounders": [],
        "confounder_matched": False,
        "domain": {
            "effective": "anime_illustration" if anime else "unknown",
            "anime_matched": anime,
            "anime_score": 0.03 if anime else -0.01,
            "real_score": -0.01 if anime else 0.01,
        },
    }


class EffectivePolicyTests(unittest.TestCase):
    def test_anime_style_false_blood_conflict_downgrades_without_erasing_raw_risk(self):
        item = evidence(2, 3, .6, [TYPE_CLASSES[0], TYPE_CLASSES[2]])
        item["blood_binary"] = binary(.02)
        item["scene_concepts"] = anime_scene(True)
        result = decide(item, DecisionPolicy())
        self.assertEqual(result["action"], "allow")
        self.assertEqual(result["risk_raw"], RISK_CLASSES[2])
        self.assertEqual(result["effective_risk"], RISK_CLASSES[0])
        self.assertEqual(result["conflict_type"], "pseudo_binary_conflict")
        self.assertEqual(result["binary_state"], "clear_negative")

    def test_anime_severe_style_false_positive_caps_to_l1_limit(self):
        item = evidence(3, 1, .6)
        item["blood_binary"] = binary(.67)
        scene = anime_scene(True)
        scene["safe_contexts"] = [{
            "key": "fiction_context",
            "title": "虚构语境",
            "score": 0.072,
        }]
        item["scene_concepts"] = scene
        result = decide(item, DecisionPolicy())
        self.assertEqual(result["action"], "limit")
        self.assertEqual(result["effective_risk"], RISK_CLASSES[1])
        self.assertEqual(result["conflict_type"], "anime_severe_style_cap")

    def test_anime_l1_fiction_pose_can_allow_without_aftermath(self):
        item = evidence(
            1, 1, .6, [TYPE_CLASSES[0], TYPE_CLASSES[1]]
        )
        item["blood_binary"] = binary(.97)
        scene = anime_scene(True)
        scene["safe_contexts"] = [{
            "key": "fiction_context",
            "title": "虚构语境",
            "score": 0.0351,
        }]
        item["scene_concepts"] = scene
        result = decide(item, DecisionPolicy())
        self.assertEqual(result["action"], "allow")
        self.assertEqual(result["conflict_type"], "anime_l1_fiction_pose")

    def test_anime_l1_fiction_cannot_hide_aftermath_candidate(self):
        item = evidence(1, 1, .6, [TYPE_CLASSES[2]])
        item["blood_binary"] = binary(.48)
        scene = anime_scene(True)
        scene["safe_contexts"] = [{
            "key": "fiction_context",
            "title": "虚构语境",
            "score": 0.04,
        }]
        item["scene_concepts"] = scene
        result = decide(item, DecisionPolicy())
        self.assertEqual(result["action"], "limit")

    def test_real_l2_without_independent_support_falls_back_to_limit(self):
        item = evidence(2, 3, .6, [TYPE_CLASSES[0], TYPE_CLASSES[2]])
        item["blood_binary"] = binary(.02)
        item["scene_concepts"] = anime_scene(False)
        result = decide(item, DecisionPolicy())
        self.assertEqual(result["action"], "limit")
        self.assertEqual(result["risk_raw"], RISK_CLASSES[2])
        self.assertEqual(result["conflict_type"], "risk_without_support")
        self.assertFalse(result["review_hint"])

    def test_review_with_severe_support_recommends_block_without_changing_action(self):
        item = evidence(3, 0, .8)
        item["blood_binary"] = binary(.9)
        result = decide(item, DecisionPolicy())
        self.assertEqual(result["action"], "review")
        self.assertEqual(result["recommended_action"], "block")
        self.assertEqual(result["recommended_action_label"], "建议下架")

    def test_real_l1_low_confidence_can_relax_in_narrow_gate(self):
        item = evidence(1, 1, .52)
        item["blood_binary"] = binary(.02)
        item["quality"]["labels"] = ["轻微模糊"]
        item["scene_concepts"] = anime_scene(False)
        result = decide(item, DecisionPolicy())
        self.assertEqual(result["action"], "allow")
        self.assertIn("真实域仅对低置信 L1", "".join(result["adjustments"]))

    def test_real_l1_severe_blur_falls_back_to_conservative_limit(self):
        item = evidence(1, 1, .52)
        item["blood_binary"] = binary(.02)
        item["quality"]["labels"] = ["严重模糊"]
        item["scene_concepts"] = anime_scene(False)
        result = decide(item, DecisionPolicy())
        self.assertEqual(result["action"], "limit")
        self.assertEqual(result["conflict_type"], "risk_without_support")

    def test_real_l0_quality_warning_does_not_force_review(self):
        item = evidence(0, 0, .9)
        item["blood_binary"] = binary(.02)
        item["quality"]["labels"] = [
            "严重模糊", "故意遮挡（马赛克/黑条/贴纸）"
        ]
        item["scene_concepts"] = anime_scene(False)
        result = decide(item, DecisionPolicy())
        self.assertEqual(result["action"], "allow")

    def test_scene_confounder_can_resolve_low_confidence_blood_like_texture(self):
        item = evidence(1, 2, .5, [TYPE_CLASSES[0], TYPE_CLASSES[2]])
        item["blood_binary"] = binary(.02)
        scene = anime_scene(False)
        scene["confounder_matched"] = True
        scene["confounders"] = [{
            "key": "red_paint_liquid",
            "title": "红色颜料",
            "score": 0.02,
        }]
        item["scene_concepts"] = scene
        result = decide(item, DecisionPolicy())
        self.assertEqual(result["action"], "allow")
        self.assertIn("安全混淆项", "".join(result["adjustments"]))

    def test_real_l2_scene_confounder_falls_back_to_conservative_limit(self):
        item = evidence(2, 2, .5, [TYPE_CLASSES[0], TYPE_CLASSES[2]])
        item["blood_binary"] = binary(.02)
        scene = anime_scene(False)
        scene["confounder_matched"] = True
        item["scene_concepts"] = scene
        result = decide(item, DecisionPolicy())
        self.assertEqual(result["action"], "limit")

    def test_ai_l3_never_downgrades_only_from_style_and_binary(self):
        item = evidence(3, 1, .9)
        item["blood_binary"] = binary(.02)
        item["scene_concepts"] = anime_scene(True)
        result = decide(item, DecisionPolicy())
        self.assertEqual(result["action"], "review")
        self.assertEqual(result["effective_risk"], RISK_CLASSES[3])

    def test_anime_l1_can_relax_with_clear_negative_and_domain(self):
        item = evidence(1, 1, .5)
        item["blood_binary"] = binary(.02)
        item["scene_concepts"] = anime_scene(True)
        result = decide(item, DecisionPolicy())
        self.assertEqual(result["action"], "allow")
        self.assertEqual(result["conflict_type"], "pseudo_binary_conflict")

    def test_near_threshold_binary_does_not_enter_aggressive_branch(self):
        item = evidence(2, 3, .8, [TYPE_CLASSES[0], TYPE_CLASSES[2]])
        item["blood_binary"] = binary(.27, .30)
        item["scene_concepts"] = anime_scene(True)
        result = decide(item, DecisionPolicy())
        self.assertEqual(result["binary_state"], "uncertain")
        self.assertEqual(result["action"], "review")

    def test_direct_violence_survives_binary_negative(self):
        item = evidence(3, 3, .9, [TYPE_CLASSES[1]])
        item["blood_binary"] = binary(.0)
        result = decide(item, DecisionPolicy())
        self.assertEqual(result["action"], "block")
        self.assertEqual(result["risk_raw"], RISK_CLASSES[3])

    def test_binary_positive_and_severe_supported_injury_blocks(self):
        item = evidence(3, 3, .9)
        item["blood_binary"] = binary(.9)
        result = decide(item, DecisionPolicy())
        self.assertEqual(result["action"], "block")
        self.assertEqual(result["effective_risk"], RISK_CLASSES[3])

    def test_high_risk_ocr_text_limits_even_when_risk_is_l0(self):
        item = evidence(0, 0, .9)
        item["blood_binary"] = binary(.0)
        item["text_evidence"] = {
            "status": "readable",
            "grounded_violation": True,
            "high_risk_rule_ids": ["R006"],
            "safe_context_rule_ids": [],
        }
        result = decide(item, DecisionPolicy())
        self.assertEqual(result["action"], "limit")
        self.assertEqual(result["effective_risk"], RISK_CLASSES[2])

    def test_detected_but_failed_ocr_does_not_override_safe_action(self):
        item = evidence(0, 0, .9)
        item["blood_binary"] = binary(.0)
        item["text_detection"] = {"status": "detected", "has_text": True}
        item["text_evidence"] = {
            "status": "failed",
            "high_risk_rule_ids": [],
            "content_profile": {},
        }
        result = decide(item, DecisionPolicy())
        self.assertEqual(result["action"], "allow")
        self.assertEqual(result["conflict_type"], "none")

    def test_risk_x_can_allow_when_all_independent_evidence_is_clear(self):
        item = evidence(4, 0, .8, [TYPE_CLASSES[0]])
        item["blood_binary"] = binary(.01)
        item["quality"]["labels"] = ["清晰", "文字扭曲形变"]
        scene = anime_scene(False)
        scene["risk_matched"] = False
        scene["risk_margin"] = 0.0
        item["scene_concepts"] = scene
        item["text_evidence"] = {
            "status": "empty",
            "grounded_violation": False,
            "high_risk_rule_ids": [],
            "content_profile": {},
        }
        result = decide(item, DecisionPolicy())
        self.assertEqual(result["action"], "allow")
        self.assertEqual(result["conflict_type"], "risk_x_multievidence_safe")

    def test_risk_x_stays_review_when_quality_is_occluded(self):
        item = evidence(4, 0, .8, [TYPE_CLASSES[0]])
        item["blood_binary"] = binary(.01)
        item["quality"]["labels"] = ["故意遮挡（马赛克/黑条/贴纸）"]
        scene = anime_scene(False)
        scene["risk_matched"] = False
        scene["risk_margin"] = 0.0
        item["scene_concepts"] = scene
        item["text_evidence"] = {
            "status": "empty",
            "grounded_violation": False,
            "high_risk_rule_ids": [],
            "content_profile": {},
        }
        result = decide(item, DecisionPolicy())
        self.assertEqual(result["action"], "review")
        self.assertEqual(result["conflict_type"], "risk_x")

    def test_red_liquid_semantic_match_can_release_l3_false_positive(self):
        item = evidence(3, 0, .8, [TYPE_CLASSES[0]])
        item["blood_binary"] = binary(.02)
        scene = anime_scene(False)
        scene["risk_matched"] = False
        scene["confounders"] = [{
            "key": "red_paint_liquid",
            "title": "红色酱汁或颜料",
            "score": 0.02,
        }]
        item["scene_concepts"] = scene
        item["text_evidence"] = {
            "status": "readable",
            "grounded_violation": False,
            "high_risk_rule_ids": [],
            "semantic_review": {"verdict": "safe"},
            "content_profile": {
                "descriptive_caption": True,
                "safe_food_scene": True,
                "safe_red_liquid": True,
            },
        }
        result = decide(item, DecisionPolicy())
        self.assertEqual(result["action"], "allow")
        self.assertEqual(result["conflict_type"], "red_liquid_semantic_match")

    def test_optional_visual_review_cannot_clear_unverified_text(self):
        item = evidence(0, 0, .9)
        item["blood_binary"] = binary(.0)
        item["text_detection"] = {"status": "detected", "has_text": True}
        item["text_evidence"] = {
            "status": "not_available",
            "high_risk_rule_ids": [],
        }
        vision = {
            "status": "ok",
            "verdict": "not_supported",
            "severity": "none",
            "trigger": {"eligible": True},
        }
        result = decide(item, DecisionPolicy(), vision)
        self.assertEqual(result["action"], "review")
        self.assertEqual(result["conflict_type"], "ocr_not_completed")

    def test_behavior_conflict_reviews(self):
        item = evidence(2, 0, .9, TYPE_CLASSES[:2])
        item["blood_binary"] = binary(.0)
        self.assertEqual(decide(item, DecisionPolicy())["action"], "review")

    def test_scene_risk_prompt_alone_cannot_assert_visual_relation(self):
        item = evidence(0, 0, .9)
        item["blood_binary"] = binary(.0)
        item["scene_concepts"] = anime_scene(True)
        item["scene_concepts"]["risk_matched"] = True
        item["scene_concepts"]["risk_scenes"] = [{
            "key": "weapon_threat",
            "title": "武器指向或明确威胁",
            "score": 0.03,
        }]
        result = decide(item, DecisionPolicy())
        self.assertEqual(result["action"], "allow")

    def test_missing_binary_does_not_take_aggressive_path(self):
        item = evidence(2, 3, .7)
        item["scene_concepts"] = anime_scene(True)
        self.assertEqual(decide(item, DecisionPolicy())["action"], "review")

    def test_scene_evidence_exposes_domain_instead_of_dropping_it(self):
        concepts = {
            "direct_attack": {
                "title": "直接攻击", "kind": "risk", "score": 0.01,
                "rag_query": "", "rag_ids": [],
            },
            "anime_illustration": {
                "title": "二次元", "kind": "domain", "score": 0.03,
                "rag_query": "", "rag_ids": [],
            },
            "real_photo": {
                "title": "真实照片", "kind": "domain", "score": -0.01,
                "rag_query": "", "rag_ids": [],
            },
        }
        result = build_scene_evidence(
            {"concepts": concepts},
            risk_margin_threshold=-0.001,
            safe_context_threshold=0.006,
            anime_score_threshold=0.015,
            domain_margin_threshold=0.01,
        )
        self.assertTrue(result["domain"]["anime_matched"])
        self.assertEqual(result["domain"]["effective"], "anime_illustration")

    def test_style_router_overrides_prompt_domain_for_anime_policy(self):
        item = evidence(2, 3, .6, [TYPE_CLASSES[0], TYPE_CLASSES[2]])
        item["blood_binary"] = binary(.02)
        item["scene_concepts"] = anime_scene(False)
        item["anime_style"] = {
            "score": 0.99,
            "threshold": 0.9,
            "is_anime": True,
        }
        result = decide(item, DecisionPolicy())
        self.assertEqual(result["action"], "allow")
        self.assertEqual(
            result["effective_risk_candidate"]["anime_source"],
            "style_router",
        )

    def test_style_router_false_positive_keeps_action_out_of_anime_policy(self):
        item = evidence(2, 3, .6, [TYPE_CLASSES[0], TYPE_CLASSES[2]])
        item["blood_binary"] = binary(.02)
        item["scene_concepts"] = anime_scene(True)
        item["anime_style"] = {
            "score": 0.05,
            "threshold": 0.9,
            "is_anime": False,
        }
        result = decide(item, DecisionPolicy())
        self.assertEqual(result["action"], "limit")


class ImageHelpers:
    @staticmethod
    def png_bytes():
        output = io.BytesIO()
        Image.new("RGB", (16, 16), "white").save(output, format="PNG")
        return output.getvalue()


if __name__ == "__main__":
    unittest.main()
