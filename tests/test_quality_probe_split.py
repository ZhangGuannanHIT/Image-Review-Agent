# -*- coding: utf-8 -*-

import unittest

from src.split_quality_probe import make_split, summarize


class QualityProbeSplitTests(unittest.TestCase):
    def test_split_is_deterministic_and_disjoint(self):
        rows = []
        for category in ("clear", "blur"):
            for origin in ("ai", "real"):
                for label in ("safe", "violation"):
                    for index in range(5):
                        rows.append({
                            "file": f"{category}/{origin}/{label}/{index}.png",
                            "category": category,
                            "origin": origin,
                            "label": label,
                            "style": "",
                            "color": "red",
                            "quality_tags": ["clear"],
                            "reviewer": "",
                            "source_url": "",
                            "license": "",
                        })
        dev_a, final_a = make_split(rows, 7)
        dev_b, final_b = make_split(rows, 7)
        self.assertEqual([row["file"] for row in dev_a], [row["file"] for row in dev_b])
        self.assertEqual([row["file"] for row in final_a], [row["file"] for row in final_b])
        self.assertFalse({row["file"] for row in dev_a} & {row["file"] for row in final_a})
        self.assertEqual(summarize(dev_a)["count"], 20)


if __name__ == "__main__":
    unittest.main()
