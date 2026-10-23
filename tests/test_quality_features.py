# -*- coding: utf-8 -*-

import unittest

import numpy as np
from PIL import Image

from src.quality_features import FEATURE_NAMES, extract_quality_features
from src.quality_fusion import QualityFusionHead
import torch


class QualityFeatureTests(unittest.TestCase):
    def test_feature_shape_finite(self):
        features = extract_quality_features(Image.new("RGB", (64, 64), "gray"))
        self.assertEqual(features.shape, (len(FEATURE_NAMES),))
        self.assertTrue(np.isfinite(features).all())

    def test_fusion_head_shape(self):
        head = QualityFusionHead(embed_dim=8, feature_dim=4, hidden=16)
        logits = head(torch.randn(3, 8), torch.randn(3, 4))
        self.assertEqual(tuple(logits.shape), (3, 5))


if __name__ == "__main__":
    unittest.main()
