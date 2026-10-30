# -*- coding: utf-8 -*-

import unittest

from src.evaluate_hard_probe_policy import aggregate


class PolicyProbeTests(unittest.TestCase):
    def test_aggregate_actions(self):
        rows = [
            {"domain": "ai", "has_blood": False, "action": "review", "uncertain": False},
            {"domain": "ai", "has_blood": True, "action": "limit", "uncertain": False},
            {"domain": "real", "has_blood": True, "action": "block", "uncertain": False},
        ]
        result = aggregate(rows, "domain")
        self.assertEqual(result["ai"]["actions"], {"review": 1, "limit": 1})
        self.assertEqual(result["real"]["actions"], {"block": 1})


if __name__ == "__main__":
    unittest.main()
