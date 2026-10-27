# -*- coding: utf-8 -*-

import json
import unittest

from langchain_core.messages import AIMessage

from backend.text_semantic import TextSemanticReviewer
from backend.threat_rule import threat_rule


class FakeModel:
    def __init__(self, payload):
        self.payload = payload

    def invoke(self, messages):
        return AIMessage(content=json.dumps(self.payload, ensure_ascii=False))


class TextSemanticTests(unittest.TestCase):
    def test_violation_requires_exact_ocr_quote(self):
        model = FakeModel({
            "verdict": "violation",
            "quote": "I WILL KILL YOU",
            "target": "viewer",
            "harm_action": "kill",
            "stance": "threat",
            "reason": "明确人身威胁",
        })
        result = TextSemanticReviewer(model).review("I WILL KILL YOU")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["verdict"], "violation")

    def test_invented_quote_is_rejected(self):
        model = FakeModel({
            "verdict": "violation",
            "quote": "I WILL KILL EVERYONE",
            "target": "everyone",
            "harm_action": "kill",
            "stance": "threat",
            "reason": "编造引文",
        })
        result = TextSemanticReviewer(model).review("I WILL KILL YOU")
        self.assertEqual(result["status"], "invalid")
        self.assertEqual(result["verdict"], "uncertain")

    def test_unreadable_text_cannot_be_safe(self):
        model = FakeModel({
            "verdict": "safe",
            "quote": "",
            "target": "",
            "harm_action": "",
            "stance": "",
            "reason": "读不清",
        })
        result = TextSemanticReviewer(model).review("[?] SHOOT")
        self.assertEqual(result["verdict"], "uncertain")

    def test_narrow_conditional_threat_rule(self):
        result = threat_rule("PLEASE LOOT! I LOVE TO SHOOT")
        self.assertEqual(
            result["rule_id"], "conditional_first_person_violence_v1"
        )
        self.assertIsNone(
            threat_rule("Please loot the snacks. I love to shoot hoops.")
        )
        self.assertIsNone(
            threat_rule("Do not say: please loot! I will shoot.")
        )


if __name__ == "__main__":
    unittest.main()
