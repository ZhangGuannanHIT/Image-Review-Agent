import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from langchain_core.messages import AIMessage

from backend.lowlevel import extract, to_text
from backend.ocr import _parse_response, GLMOCRConfig
from backend.policy import DecisionPolicy, decide, render
from backend.service import ReviewService
from backend.mock import RuleBasedMockChatModel
from backend.tools import text_evidence
from src.dataset import _is_conflict
from src.labels import RISK_CLASSES, INJURY_CLASSES, TYPE_CLASSES


def evidence(risk=0, injury=0, confidence=0.9, active=None):
    def single(classes, index):
        return {"label": classes[index], "confidence": confidence,
                "scores": {label: confidence if i == index else (1-confidence)/(len(classes)-1)
                           for i, label in enumerate(classes)}}
    active = active or [TYPE_CLASSES[0]]
    return {"trained": True, "risk": single(RISK_CLASSES, risk),
            "injury": single(INJURY_CLASSES, injury),
            "type": {"labels": active, "thresholds": dict(zip(TYPE_CLASSES, [.45, .6, .6, .9])), "scores": {name: 0.95 if name in active else 0.01
                                                     for name in TYPE_CLASSES}},
            "quality": {"labels": ["清晰"]}}


class Engine:
    trained = True

    def predict(self, image):
        return evidence()


class FailingAfterClip(RuleBasedMockChatModel):
    def invoke(self, messages):
        if any(getattr(m, "name", "") == "analyze_image" for m in messages):
            raise RuntimeError("LLM unavailable")
        return super().invoke(messages)


class PlainHumanAfterClip(RuleBasedMockChatModel):
    def invoke(self, messages):
        if any(getattr(m, "name", "") == "analyze_image" for m in messages):
            return AIMessage(content="请转人工复核，暂不放行。")
        return super().invoke(messages)


class PolicyTests(unittest.TestCase):
    def test_moderate_risk_with_calibrated_threat_is_not_blanket_allow(self):
        item = evidence(1, 0, .46, [TYPE_CLASSES[3]])
        item["type"]["thresholds"] = dict(zip(TYPE_CLASSES, [.45, .6, .6, .9]))
        result = decide(item, DecisionPolicy())
        self.assertEqual(result["action"], "limit")
        self.assertIn("L1_轻微违规", render(result))

    def test_calibrated_type_threshold_not_global_point_eight(self):
        item = evidence(2, 0, .58, [TYPE_CLASSES[1]])
        item["type"]["scores"][TYPE_CLASSES[1]] = .65
        item["type"]["thresholds"] = dict(zip(TYPE_CLASSES, [.45, .6, .6, .9]))
        self.assertEqual(decide(item, DecisionPolicy())["action"], "limit")

    def test_uncertain_preserves_actual_predictions(self):
        a = render(decide(evidence(2, 2, .52), DecisionPolicy()))
        b = render(decide(evidence(3, 3, .48), DecisionPolicy()))
        self.assertNotEqual(a, b)
        self.assertIn("L2_中度违规", a)
        self.assertIn("0.520", a)
        self.assertIn("L3_严重违规", b)

    def test_independent_dimensions(self):
        self.assertFalse(_is_conflict([TYPE_CLASSES[0], TYPE_CLASSES[2], TYPE_CLASSES[3]]))
        self.assertTrue(_is_conflict([TYPE_CLASSES[0], TYPE_CLASSES[1]]))
        self.assertEqual(decide(evidence(3, 3), DecisionPolicy())["action"], "block")

    def test_binary_blood_head_vetoes_unconfirmed_severity(self):
        item = evidence(2, 3, .9, [TYPE_CLASSES[2]])
        item["blood_binary"] = {
            "p_blood": 0.10,
            "threshold": 0.285,
            "label": "无血腥",
        }
        result = decide(item, DecisionPolicy())
        self.assertEqual(result["action"], "review")
        self.assertFalse(result["blood_binary_candidate"]["supported"])
        self.assertFalse(result["injury_candidate"]["accepted"])
        self.assertIn("冲突", result["reason"])
        self.assertIn("人工复核", render(result))

    def test_binary_blood_head_can_confirm_existing_severity(self):
        item = evidence(3, 3, .9)
        item["blood_binary"] = {
            "p_blood": 0.90,
            "threshold": 0.285,
            "label": "有血腥",
        }
        result = decide(item, DecisionPolicy())
        self.assertEqual(result["action"], "block")
        self.assertTrue(result["blood_binary_candidate"]["supported"])
        self.assertIn("血腥二分类", render(result))

    def test_uncertain_reviews_and_clear_actions_remain_decisive(self):
        for risk, injury, confidence, action in [
            (0, 0, .9, "allow"), (2, 2, .52, "review"), (3, 3, .48, "review"),
            (4, 0, .9, "review"), (1, 1, .9, "limit"), (2, 2, .9, "limit"),
            (3, 3, .9, "block"), (3, 0, .9, "review"), (0, 3, .9, "review"),
        ]:
            with self.subTest(risk=risk, injury=injury, confidence=confidence):
                result = decide(evidence(risk, injury, confidence), DecisionPolicy())
                self.assertEqual(result["action"], action)
                if action != "review":
                    self.assertNotIn("人工复核", render(result))

    def test_quality_degradation_stays_warning_but_uncertainty_reviews(self):
        item = evidence(2, 0, .5)
        item["quality"]["labels"] = ["严重模糊"]
        result = decide(item, DecisionPolicy())
        self.assertEqual(result["action"], "review")
        self.assertTrue(result["quality_candidate"]["needs_review"])
        self.assertEqual(result["quality_candidate"]["action_effect"], "evidence_warning_only")
        self.assertIn("高不确定性", result["reason"])
        self.assertIn("严重模糊", render(result))

    def test_clear_quality_does_not_change_normal_allow(self):
        self.assertEqual(decide(evidence(0, 0, .9), DecisionPolicy())["action"], "allow")

    def test_safe_risk_with_quality_degradation_does_not_force_review(self):
        item = evidence(0, 0, .9)
        item["quality"]["labels"] = ["严重模糊"]
        self.assertEqual(decide(item, DecisionPolicy())["action"], "allow")

    def test_unknown_risk_routes_to_review_without_quality_dependency(self):
        item = evidence(4, 0, .9)
        item["quality"]["labels"] = ["清晰"]
        self.assertEqual(decide(item, DecisionPolicy())["action"], "review")

    def test_conflict_and_missing_scores_review(self):
        item = evidence(3, 3, active=TYPE_CLASSES[:2])
        result = decide(item, DecisionPolicy())
        self.assertTrue(result["uncertain"])
        self.assertEqual(result["action"], "review")
        item = evidence(3, 3)
        item["risk"]["scores"] = {}
        result = decide(item, DecisionPolicy())
        self.assertTrue(result["uncertain"])
        self.assertEqual(result["action"], "review")

    def test_warm_color_and_dense_edges_do_not_assert_semantics(self):
        features = extract(Image.new("RGB", (64, 64), (205, 145, 125)))
        features["edge_density"] = .9
        self.assertEqual(features["red_ratio"], 1.0)
        output = to_text(features)
        self.assertNotIn("疑似血腥", output)
        self.assertNotIn("疑似打码", output)
        self.assertNotIn("疑似过度压缩", output)

    def test_ocr_placeholder_removed_but_real_text_kept(self):
        for source, expected in [("![](page=0,bbox=[2, 0, 1536, 1024])", []),
                                 ("文字\n![](page=0,bbox=[2, 0, 1536, 1024])", ["文字"])]:
            self.assertEqual(_parse_response({"md_results": source}, GLMOCRConfig())["texts"], expected)

    def test_full_scores_present_in_evidence(self):
        text = text_evidence(evidence(3, 3, .38))
        for label in INJURY_CLASSES:
            self.assertIn(label, text)
        self.assertIn("无正在进行", text)

    def test_all_agent_exits_use_policy_and_audit_image(self):
        buf = io.BytesIO()
        Image.new("RGB", (16, 16), "white").save(buf, format="PNG")
        for model, rounds in [(RuleBasedMockChatModel(), 5), (PlainHumanAfterClip(), 5),
                              (FailingAfterClip(), 5), (RuleBasedMockChatModel(), 1)]:
            with self.subTest(model=type(model).__name__, rounds=rounds), tempfile.TemporaryDirectory() as directory:
                service = ReviewService(model, Engine(), log_dir=directory, max_rounds=rounds)
                result = service.review(buf.getvalue(), "../../outside.png")
                self.assertIn("处置：放行", result.suggestion)
                self.assertNotIn("人工", result.suggestion)
                record = json.loads((Path(directory) / "review.log.jsonl").read_text(encoding="utf-8"))
                self.assertEqual(record["decision"]["action"], "allow")
                self.assertEqual(Path(record["image_path"]).read_bytes(), buf.getvalue())
                self.assertEqual(Path(record["image_path"]).parent, Path(directory) / "images")


if __name__ == "__main__":
    unittest.main()
