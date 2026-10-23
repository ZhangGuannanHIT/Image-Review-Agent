# -*- coding: utf-8 -*-

import unittest

import torch

from src.binary_injury import BinaryInjuryHead, select_threshold


class BinaryInjuryTests(unittest.TestCase):
    def test_head_shapes(self):
        output = BinaryInjuryHead(embed_dim=8)(torch.randn(3, 8))
        self.assertEqual(tuple(output.shape), (3,))

    def test_threshold_respects_fpr_budget(self):
        labels = [False, False, False, False, True, True, True, True]
        scores = [0.05, 0.10, 0.20, 0.30, 0.60, 0.70, 0.80, 0.90]
        selected = select_threshold(labels, scores, max_false_positive_rate=0.25)
        self.assertTrue(selected["feasible"])
        self.assertLessEqual(selected["false_positive_rate"], 0.25)
        self.assertGreaterEqual(selected["sensitivity"], 0.75)

    def test_threshold_rejects_invalid_budget(self):
        with self.assertRaisesRegex(ValueError, "max_false_positive_rate"):
            select_threshold([False, True], [0.1, 0.9], 1.1)


if __name__ == "__main__":
    unittest.main()
