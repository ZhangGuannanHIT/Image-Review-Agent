# -*- coding: utf-8 -*-

import unittest

from src.evaluate_quality_probe import tag_vector
from src.labels import QUALITY_CLASSES


class QualityProbeEvalTests(unittest.TestCase):
    def test_tag_vector_maps_all_quality_tags(self):
        tags = ["severe_blur", "text_distortion"]
        vector = tag_vector(tags)
        self.assertEqual(len(vector), len(QUALITY_CLASSES))
        self.assertEqual(vector[QUALITY_CLASSES.index("严重模糊")], 1)
        self.assertEqual(vector[QUALITY_CLASSES.index("文字扭曲形变")], 1)
        self.assertEqual(sum(vector), 2)


if __name__ == "__main__":
    unittest.main()
