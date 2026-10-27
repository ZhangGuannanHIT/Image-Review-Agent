# -*- coding: utf-8 -*-

import json
from types import SimpleNamespace
import unittest

from pydantic import ValidationError

from backend.anime_vision import AnimeVisualEvidence, AnimeVisualReviewer
from backend.anime_vision_policy import anime_visual_gate, reconcile_anime_visual
from src.labels import RISK_CLASSES


def decision(action="limit"):
    return {
        "action": action,
        "risk_raw": RISK_CLASSES[2],
        "effective_risk": RISK_CLASSES[2],
        "level": RISK_CLASSES[2],
        "uncertain": False,
        "conflict_type": "risk_without_support",
        "reason": "候选",
    }


def evidence(score=0.99, grounded=False, high_rule=False, verdict="safe"):
    return {
        "anime_style": {"score": score, "is_anime": score >= 0.9},
        "text_evidence": {
            "grounded_violation": grounded,
            "high_risk_rule_ids": ["R006"] if high_rule else [],
            "semantic_review": {"verdict": verdict},
        },
    }


def visual(**overrides):
    value = {
        "status": "ok",
        "injury": "absent",
        "attack": "absent",
        "threat": "absent",
        "weapon_visible": "absent",
        "context": "animation",
        "appearance": "cartoon",
        "threat_basis": "none",
        "threat_target": "",
        "weapon_pose": "",
        "reason": "未见伤害或威胁",
    }
    value.update(overrides)
    return value


class AnimeVisionTests(unittest.TestCase):
    def test_gate_requires_high_confidence_anime_and_non_allow_action(self):
        self.assertTrue(anime_visual_gate(decision(), evidence())["eligible"])
        self.assertFalse(anime_visual_gate(
            decision("allow"), evidence()
        )["eligible"])
        self.assertFalse(anime_visual_gate(
            decision(), evidence(score=0.8)
        )["eligible"])
        self.assertFalse(anime_visual_gate(
            decision(), evidence(grounded=True)
        )["eligible"])

    def test_gate_blocks_unresolved_high_risk_text(self):
        result = anime_visual_gate(
            decision(), evidence(high_rule=True, verdict="uncertain")
        )
        self.assertFalse(result["eligible"])
        self.assertIn("text_grounding_unresolved", result["reason"])

    def test_absent_visual_evidence_withdraws_anime_limit(self):
        result = reconcile_anime_visual(decision(), evidence(), visual())
        self.assertEqual(result["action"], "allow")
        self.assertEqual(result["effective_risk"], RISK_CLASSES[0])

    def test_original_l3_with_absent_visual_evidence_stays_l1(self):
        item = decision()
        item["risk_raw"] = RISK_CLASSES[3]
        result = reconcile_anime_visual(item, evidence(), visual())
        self.assertEqual(result["action"], "limit")
        self.assertEqual(result["effective_risk"], RISK_CLASSES[1])

    def test_horror_style_without_relation_remains_l1(self):
        result = reconcile_anime_visual(
            decision(), evidence(), visual(appearance="horror")
        )
        self.assertEqual(result["action"], "limit")
        self.assertEqual(result["effective_risk"], RISK_CLASSES[1])

    def test_cartoon_injury_uncertainty_caps_to_l1(self):
        result = reconcile_anime_visual(
            decision(), evidence(), visual(injury="uncertain")
        )
        self.assertEqual(result["action"], "limit")
        self.assertEqual(result["effective_risk"], RISK_CLASSES[1])

    def test_injury_uncertainty_caps_to_l1_without_review(self):
        result = reconcile_anime_visual(
            decision(), evidence(), visual(
                injury="uncertain", appearance="ordinary"
            )
        )
        self.assertEqual(result["action"], "limit")
        self.assertEqual(result["effective_risk"], RISK_CLASSES[1])
        self.assertEqual(result["anime_visual_resolution"], "limit")

    def test_attack_uncertainty_caps_to_l1_without_review(self):
        result = reconcile_anime_visual(
            decision("review"), evidence(), visual(attack="uncertain")
        )
        self.assertEqual(result["action"], "limit")
        self.assertEqual(result["effective_risk"], RISK_CLASSES[1])

    def test_weapon_visible_without_threat_remains_l1(self):
        result = reconcile_anime_visual(
            decision(), evidence(), visual(weapon_visible="present")
        )
        self.assertEqual(result["action"], "limit")
        self.assertEqual(result["effective_risk"], RISK_CLASSES[1])

    def test_game_weapon_display_without_relation_can_allow(self):
        result = reconcile_anime_visual(
            decision(), evidence(), visual(
                weapon_visible="present", context="game_ui"
            )
        )
        self.assertEqual(result["action"], "allow")

    def test_game_attack_without_injury_or_target_threat_can_allow(self):
        result = reconcile_anime_visual(
            decision(), evidence(), visual(
                attack="present",
                weapon_visible="present",
                context="game_gameplay",
            )
        )
        self.assertEqual(result["action"], "allow")
        self.assertEqual(
            result["anime_visual_effect"], "withdraw_game_context"
        )

    def test_fictional_attack_without_injury_can_allow(self):
        item_evidence = evidence()
        item_evidence["scene_concepts"] = {
            "safe_contexts": [{
                "key": "fiction_context",
                "score": 0.05,
            }]
        }
        result = reconcile_anime_visual(
            decision(), item_evidence, visual(attack="present")
        )
        self.assertEqual(result["action"], "allow")

    def test_attack_does_not_withdraw_anime_restriction(self):
        result = reconcile_anime_visual(
            decision(), evidence(), visual(attack="present")
        )
        self.assertEqual(result["action"], "limit")
        self.assertEqual(result["anime_visual_resolution"], "keep")

    def test_grounded_text_cannot_be_withdrawn(self):
        result = reconcile_anime_visual(
            decision(), evidence(grounded=True), visual()
        )
        self.assertEqual(result["action"], "limit")
        self.assertEqual(result["anime_visual_effect"], "unchanged_grounded_text")

    def test_invalid_threat_relation_is_rejected(self):
        with self.assertRaises(ValidationError):
            AnimeVisualEvidence(
                injury="absent",
                attack="absent",
                threat="present",
                weapon_visible="present",
                context="animation",
                appearance="cartoon",
                threat_basis="none",
                threat_target="",
                weapon_pose="持刀站立",
                reason="没有明确关系",
            )

    def test_reviewer_parses_structured_response(self):
        class Model:
            def invoke(self, messages):
                payload = visual()
                payload.pop("status")
                return SimpleNamespace(
                    content=json.dumps(payload, ensure_ascii=False)
                )

        from PIL import Image
        import io

        buffer = io.BytesIO()
        Image.new("RGB", (16, 16), "white").save(buffer, format="PNG")
        result = AnimeVisualReviewer(Model()).review(buffer.getvalue())
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["injury"], "absent")


if __name__ == "__main__":
    unittest.main()
