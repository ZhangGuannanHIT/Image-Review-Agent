# -*- coding: utf-8 -*-

import unittest

import torch

from src.ordinal_injury import (
    OrdinalInjuryHead, class_probabilities, cumulative_targets, severity_metrics,
)


class OrdinalInjuryTests(unittest.TestCase):
    def test_head_shapes(self):
        self.assertEqual(tuple(OrdinalInjuryHead(8)(torch.randn(3, 8)).shape), (3, 3))

    def test_cumulative_targets(self):
        result = cumulative_targets(torch.tensor([0, 1, 2, 3]))
        expected = torch.tensor([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1],
        ], dtype=torch.float32)
        torch.testing.assert_close(result, expected)

    def test_probabilities_sum_to_one(self):
        probabilities = class_probabilities(torch.randn(5, 3))
        torch.testing.assert_close(
            probabilities.sum(-1), torch.ones(5), atol=1e-6, rtol=0,
        )

    def test_qwk_and_severe_recall(self):
        metrics = severity_metrics([0, 1, 2, 3, 3], [0, 1, 2, 3, 2])
        self.assertGreater(metrics["qwk"], 0.7)
        self.assertEqual(metrics["severe_recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
