# -*- coding: utf-8 -*-

import unittest

from src.evaluate_ocr_probe import character_recall, normalize_text


class OCRProbeTests(unittest.TestCase):
    def test_normalize_text(self):
        self.assertEqual(normalize_text("TEA AND CAKE"), "teaandcake")
        self.assertEqual(normalize_text("可疑：炸弹！"), "可疑炸弹")

    def test_character_recall(self):
        self.assertEqual(character_recall("abc", "a b x"), 2 / 3)
        self.assertIsNone(character_recall("", "abc"))


if __name__ == "__main__":
    unittest.main()
