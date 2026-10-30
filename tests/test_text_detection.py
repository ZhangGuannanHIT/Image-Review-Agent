# -*- coding: utf-8 -*-

import unittest

import cv2
import numpy as np

from backend.text_detection import LightweightTextDetector


class LightweightTextDetectorTests(unittest.TestCase):
    @staticmethod
    def encoded(image: np.ndarray) -> bytes:
        ok, payload = cv2.imencode(".png", image)
        if not ok:
            raise AssertionError("PNG encoding failed")
        return payload.tobytes()

    def test_blank_image_has_no_text_regions(self):
        image = np.full((240, 320, 3), 255, dtype=np.uint8)
        result = LightweightTextDetector().detect(self.encoded(image))
        self.assertEqual(result["status"], "empty")
        self.assertFalse(result["has_text"])

    def test_text_line_is_detected_without_reading_it(self):
        image = np.full((240, 640, 3), 255, dtype=np.uint8)
        cv2.putText(
            image,
            "WILL HURT YOU",
            (24, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        result = LightweightTextDetector().detect(self.encoded(image))
        self.assertEqual(result["status"], "detected")
        self.assertTrue(result["has_text"])
        self.assertGreater(result["box_count"], 0)
        self.assertEqual(result["method"], "opencv_text_regions_v1")


if __name__ == "__main__":
    unittest.main()
