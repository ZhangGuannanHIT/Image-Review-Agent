import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock

from backend.vision import VisionGate, trigger, BloodReview, DeepSeekBloodReviewer, parse_review
from backend.tools import build_tools, tools_by_name, set_current_image, reset_current_image, get_request_context
from backend.policy import DecisionPolicy, decide, render
from tests.test_review_policy import evidence
from src.labels import TYPE_CLASSES, INJURY_CLASSES


class VisionTests(unittest.TestCase):
    def test_json_fence_and_invalid_response_diagnostics(self):
        from tests.test_workflow_api import image_bytes
        from langchain_core.messages import AIMessage
        payload = '{"verdict":"not_supported","severity":"none","observation":"红色背景"}'
        self.assertEqual(parse_review('```json\n' + payload + '\n```').verdict, 'not_supported')
        client = DeepSeekBloodReviewer.__new__(DeepSeekBloodReviewer)
        client.model_name = 'deepseek-v4-flash'
        client.model = Mock()
        client.model.invoke.return_value = AIMessage(content='{"verdict":"not_supported"}', response_metadata={'finish_reason':'stop'})
        result = client.review(image_bytes())
        self.assertEqual(result['stage'], 'response_validation')
        self.assertEqual(result['status'], 'failed')
        self.assertIn({'field':['severity'], 'type':'missing'}, result['validation_errors'])
        self.assertEqual(result['response_text'], '{"verdict":"not_supported"}')
        result['trigger'] = {'eligible':True}
        text = render(decide(evidence(4, 3, .6), DecisionPolicy(), result))
        self.assertEqual(text.count('格式不符合要求'), 1)
        self.assertNotIn('调用失败', text)

    def test_gate_requires_only_no_violence(self):
        for labels in [[TYPE_CLASSES[0], TYPE_CLASSES[2]], [TYPE_CLASSES[2]],
                       [TYPE_CLASSES[3]], [TYPE_CLASSES[1]], TYPE_CLASSES[:2], []]:
            with self.subTest(labels=labels):
                item = evidence(2, 2, .52)
                item['type']['labels'] = labels
                self.assertFalse(trigger(item, VisionGate())['eligible'])

    def test_fixed_workflow_reviews_blood_and_logs(self):
        from backend.service import ReviewService
        from backend.mock import RuleBasedMockChatModel
        from tests.test_workflow_api import image_bytes
        reviewer, engine = Mock(), Mock()
        engine.predict.return_value = evidence(2, 2, .52)
        engine.ckpt_dir = ""
        engine.type_thresholds = {}
        engine.quality_thresholds = {}
        reviewer.review.return_value = {"status": "ok", "verdict": "not_supported", "severity": "none", "observation": "只有红色背景"}
        with tempfile.TemporaryDirectory() as directory:
            service = ReviewService(
                RuleBasedMockChatModel(),
                engine,
                log_dir=directory,
                vision_client=reviewer,
                explanation_enabled=False,
            )
            result = service.review(image_bytes())
            record = json.loads((Path(directory)/"review.log.jsonl").read_text(encoding="utf-8"))
            called = [
                step["name"] for step in record["steps"]
                if step["type"] == "tool_call"
            ]
            self.assertEqual(called, ["analyze_image", "review_blood", "detect_text"])
            self.assertIn('血腥复核：未发现血腥', result.suggestion)
            self.assertIn('处置：放行', result.suggestion)
            self.assertNotIn('以上为模型预测', result.suggestion)
            reviewer.review.assert_called_once()

    def test_gate_only_ambiguous_bloody_candidate(self):
        gate = VisionGate()
        self.assertFalse(trigger(evidence(0, 0, .4), gate)["eligible"])
        self.assertTrue(trigger(evidence(2, 2, .64), gate)["eligible"])
        self.assertFalse(trigger(evidence(2, 2, .65), gate)["eligible"])
        self.assertFalse(trigger(evidence(3, 3, .9), gate)["eligible"])
        close = evidence(2, 2, .4)
        close["injury"]["scores"] = dict(zip(INJURY_CLASSES, [.1, .05, .45, .4]))
        self.assertTrue(trigger(close, gate)["eligible"])
        self.assertFalse(trigger({}, gate)["eligible"])

    def test_review_blood_tool_is_explicit_and_cached(self):
        reviewer = Mock()
        reviewer.review.return_value = {"status": "ok", "verdict": "not_supported", "severity": "none", "observation": "红色衣物"}
        engine = Mock()
        engine.predict.return_value = evidence(2, 2, .52)
        tools = tools_by_name(build_tools(engine, vision_client=reviewer))
        token = set_current_image(b"image")
        try:
            self.assertIn("skipped", tools["review_blood"].invoke({}))
            reviewer.review.assert_not_called()
            tools["analyze_image"].invoke({})
            result = tools["review_blood"].invoke({})
            self.assertIn("not_supported", result)
            tools["review_blood"].invoke({})
            tools["analyze_image"].invoke({})
            reviewer.review.assert_called_once_with(b"image")
        finally:
            reset_current_image(token)

    def test_noneligible_never_sends_image(self):
        for item in [evidence(0, 0, .4), evidence(3, 3, .9)]:
            reviewer, engine = Mock(), Mock()
            engine.predict.return_value = item
            tools = tools_by_name(build_tools(engine, vision_client=reviewer))
            token = set_current_image(b"image")
            try:
                tools["analyze_image"].invoke({})
                tools["review_blood"].invoke({})
                reviewer.review.assert_not_called()
            finally:
                reset_current_image(token)

    def test_failed_call_is_cached(self):
        reviewer, engine = Mock(), Mock()
        reviewer.review.side_effect = RuntimeError("secret must not leak")
        engine.predict.return_value = evidence(2, 2, .52)
        tools = tools_by_name(build_tools(engine, vision_client=reviewer))
        token = set_current_image(b"image")
        try:
            output = tools["analyze_image"].invoke({})
            tools["review_blood"].invoke({})
            reviewer.review.assert_called_once()
            self.assertNotIn("secret", output)
            self.assertEqual(get_request_context().cache["review_blood"]["status"], "failed")
        finally:
            reset_current_image(token)

    def test_review_changes_injury_not_violence(self):
        for verdict, severity, action in [("not_supported", "none", "allow"),
                                           ("supported", "moderate", "limit"),
                                           ("supported", "severe", "block"),
                                           ("uncertain", "unknown", "review")]:
            review = {"status": "ok", "verdict": verdict, "severity": severity,
                      "observation": "复核依据", "trigger": {"eligible": True}}
            result = decide(evidence(2, 2, .52), DecisionPolicy(), review)
            self.assertEqual(result["action"], action)
            text = render(result)
            self.assertIn("血腥复核", text)
            self.assertNotIn("复核依据", text)
            self.assertNotIn("以上为模型预测", text)
        violent = evidence(3, 2, .55, [TYPE_CLASSES[1]])
        negative = {"status": "ok", "verdict": "not_supported", "severity": "none", "trigger": {"eligible": True}}
        self.assertEqual(decide(violent, DecisionPolicy(), negative)["action"], "block")

    def test_inconsistent_review_rejected(self):
        with self.assertRaises(ValueError):
            BloodReview(verdict="not_supported", severity="severe", observation="不一致")

    def test_client_image_payload_and_no_clip_priming(self):
        from tests.test_workflow_api import image_bytes
        client = DeepSeekBloodReviewer.__new__(DeepSeekBloodReviewer)
        client.model_name = "deepseek-v4-flash"
        client.model = Mock()
        client.model.invoke.return_value.content = json.dumps({"verdict": "not_supported", "severity": "none", "observation": "白色背景"})
        self.assertEqual(client.review(image_bytes())["status"], "ok")
        messages = client.model.invoke.call_args.args[0]
        self.assertTrue(messages[1].content[1]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertNotIn("L3", str(messages))


if __name__ == "__main__":
    unittest.main()
