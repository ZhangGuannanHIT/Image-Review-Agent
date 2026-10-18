# -*- coding: utf-8 -*-

import copy
import json
import tempfile
import unittest
from pathlib import Path

from backend import rag


ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_PATH = ROOT / "knowledge" / "knowledge.json"
QUERY_PATH = ROOT / "knowledge" / "test_queries.json"


class KnowledgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items = rag.load_items(KNOWLEDGE_PATH)
        cls.retriever = rag.LocalKnowledgeRetriever(cls.items)
        cls.queries = json.loads(QUERY_PATH.read_text(encoding="utf-8"))

    def test_atomic_rules_cases_and_sources(self):
        ids = [item["id"] for item in self.items]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue({f"R{index:03d}" for index in range(1, 16)}.issubset(ids))
        self.assertEqual(sum(item["type"] == "verified_case" for item in self.items), 4)
        self.assertEqual(sum(item["type"] == "teaching_case" for item in self.items), 1)
        for item in self.items:
            source = item["source"]
            if "xxx" in source["locator"].casefold():
                self.assertNotEqual(source["locator_status"], "exact_url_verified")

    def test_calibration_query_count_and_groups(self):
        self.assertEqual(len(self.queries), 55)
        counts = {
            group: sum(case["group"] == group for case in self.queries)
            for group in ("violation", "boundary", "robustness", "noise")
        }
        self.assertEqual(counts, {
            "violation": 15,
            "boundary": 15,
            "robustness": 15,
            "noise": 10,
        })

    @staticmethod
    def _evaluate(retriever, queries):
        metrics = {
            "non_noise": 0,
            "top1_hits": 0,
            "top3_hits": 0,
            "noise": 0,
            "noise_rejected": 0,
            "boundary": 0,
            "boundary_conflicts": 0,
        }
        for case in queries:
            returned_ids = [item["id"] for item in retriever.retrieve(case["query"], k=3)]
            expected = set(case["expected_ids"])
            if case["group"] == "noise":
                metrics["noise"] += 1
                metrics["noise_rejected"] += int(not returned_ids)
                continue
            metrics["non_noise"] += 1
            metrics["top1_hits"] += int(bool(returned_ids) and returned_ids[0] in expected)
            metrics["top3_hits"] += int(bool(set(returned_ids) & expected))
            if case["group"] == "boundary":
                metrics["boundary"] += 1
                metrics["boundary_conflicts"] += int(bool(set(returned_ids) - expected))
        return metrics

    def test_threshold_030_calibration_balance(self):
        metrics = self._evaluate(self.retriever, self.queries)
        self.assertEqual(metrics, {
            "non_noise": 45,
            "top1_hits": 44,
            "top3_hits": 45,
            "noise": 10,
            "noise_rejected": 10,
            "boundary": 15,
            "boundary_conflicts": 1,
        })

    def test_threshold_031_starts_losing_top3_recall(self):
        stricter = rag.LocalKnowledgeRetriever(self.items, min_score=0.31)
        metrics = self._evaluate(stricter, self.queries)
        self.assertEqual(metrics["top3_hits"], 44)
        self.assertEqual(metrics["noise_rejected"], 10)

    def test_negative_examples_are_not_positive_matches(self):
        cases = {
            "医学解剖教材": "R005",
            "军事历史科普、合法刀具商品信息": "R002",
            "普通新闻报道": "CASE-VERIFIED-01",
        }
        for query, forbidden_id in cases.items():
            with self.subTest(query=query):
                self.assertNotIn(forbidden_id, [item["id"] for item in self.retriever.retrieve(query, k=5)])

        related = self.retriever.retrieve("动物保护组织反虐待宣传、公益曝光", k=5)
        animal_rule = next(item for item in related if item["id"] == "R015")
        self.assertIn("动物保护组织", animal_rule["negative_example"])

    def test_invalid_verified_url_is_rejected(self):
        item = copy.deepcopy(self.items[0])
        item["id"] = "INVALID-URL"
        item["source"]["locator_status"] = "exact_url_verified"
        item["source"]["locator"] = "https:///"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps([item], ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "精确来源地址无效"):
                rag.load_items(path)

    def test_default_knowledge_warmup(self):
        self.assertEqual(rag.warmup(), len(self.items))

    def test_empty_short_and_nonpositive_k_return_empty(self):
        self.assertEqual(self.retriever.retrieve("", k=5), [])
        self.assertEqual(self.retriever.retrieve("血", k=5), [])
        self.assertEqual(self.retriever.retrieve("明确暴力威胁", k=0), [])

    def test_results_are_limited_sorted_and_stable(self):
        first = self.retriever.retrieve("明确死亡威胁，我要杀了你", k=4)
        second = self.retriever.retrieve("明确死亡威胁，我要杀了你", k=4)
        self.assertLessEqual(len(first), 4)
        self.assertEqual([item["id"] for item in first], [item["id"] for item in second])
        self.assertEqual(
            [item["score"] for item in first],
            sorted((item["score"] for item in first), reverse=True),
        )
        self.assertTrue(all(isinstance(item["score"], float) for item in first))
        self.assertTrue(all("profile" in item and "desc" in item for item in first))

    def test_returned_records_do_not_mutate_knowledge(self):
        result = self.retriever.retrieve("OCR乱码单字杀没有上下文", k=1)[0]
        result["tags"].append("临时标签")
        original = next(item for item in self.retriever.items if item["id"] == result["id"])
        self.assertNotIn("临时标签", original["tags"])

    def test_knowledge_and_queries_are_valid_json(self):
        self.assertIsInstance(json.loads(KNOWLEDGE_PATH.read_text(encoding="utf-8")), list)
        self.assertIsInstance(json.loads(QUERY_PATH.read_text(encoding="utf-8")), list)


if __name__ == "__main__":
    unittest.main()
