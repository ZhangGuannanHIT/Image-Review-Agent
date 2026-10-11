# -*- coding: utf-8 -*-

import io
import unittest

import torch
from PIL import Image

from src.dataset import _is_conflict
from src.engine import determine_max_patches
from src.losses import MultiTaskLoss, WeightedBCE, WeightedCE
from src.model import MultiTaskHead


class CoreTests(unittest.TestCase):
    def test_full_label_conflict(self):
        self.assertTrue(_is_conflict(["无暴力", "直接暴力（攻击进行中）"]))
        self.assertFalse(_is_conflict(["直接暴力（攻击进行中）"]))

    def test_type_missing_mask_keeps_graph(self):
        logits = torch.randn(2, 4, requires_grad=True)
        targets = torch.zeros(2, 4)
        loss = WeightedBCE()(logits, targets, torch.tensor([False, False]))
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_multitask_loss(self):
        criterion = MultiTaskLoss(focal_gamma=1.0)
        logits = {
            "risk": torch.randn(3, 5, requires_grad=True),
            "type": torch.randn(3, 4, requires_grad=True),
            "injury": torch.randn(3, 4, requires_grad=True),
            "quality": torch.randn(3, 5, requires_grad=True),
        }
        batch = {
            "risk_label": torch.tensor([0, 1, -1]),
            "type_target": torch.zeros(3, 4),
            "type_valid": torch.tensor([True, False, True]),
            "injury_label": torch.tensor([0, 1, 2]),
            "quality_target": torch.tensor([[1., 0., 0., 0., 0.], [0., 1., 0., 0., 1.], [0., 0., 0., 0., 0.]]),
            "quality_valid": torch.tensor([True, True, False]),
        }
        loss = criterion(logits, batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_head_shapes(self):
        output = MultiTaskHead()(torch.randn(2, 768))
        self.assertEqual(tuple(output["risk"].shape), (2, 5))
        self.assertEqual(tuple(output["type"].shape), (2, 4))
        self.assertEqual(tuple(output["quality"].shape), (2, 5))

    def test_patch_bucket(self):
        self.assertEqual(determine_max_patches(Image.new("RGB", (64, 64))), 128)
        self.assertEqual(determine_max_patches(Image.new("RGB", (512, 512))), 1024)


if __name__ == "__main__":
    unittest.main()
