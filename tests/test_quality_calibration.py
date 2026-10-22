# -*- coding: utf-8 -*-

import unittest

from src.calibrate_quality_fusion import calibrate
from src.labels import QUALITY_CLASSES


class QualityCalibrationTests(unittest.TestCase):
    def test_thresholds_cover_all_classes(self):
        rows = [
            {
                "expected": [1, 0, 0, 0, 0],
                "scores": {label: 0.9 if label == "清晰" else 0.1 for label in QUALITY_CLASSES},
            },
            {
                "expected": [0, 0, 1, 0, 0],
                "scores": {label: 0.9 if label == "严重模糊" else 0.1 for label in QUALITY_CLASSES},
            },
        ]
        thresholds, details = calibrate(rows)
        self.assertEqual(set(thresholds), set(QUALITY_CLASSES))
        self.assertEqual(set(details), set(QUALITY_CLASSES))


if __name__ == "__main__":
    unittest.main()
