# -*- coding: utf-8 -*-

import tempfile
import unittest
from pathlib import Path

from src.hard_probe import binary_metrics, load_manifest, metrics_at_threshold


class HardProbeTests(unittest.TestCase):
    def test_manifest_loads_boolean_and_severity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            (root / "images" / "a.jpg").write_bytes(b"fixture")
            (root / "images" / "b.png").write_bytes(b"fixture")
            manifest = root / "manifest.csv"
            manifest.write_text(
                "file,folder,domain,red_level,has_blood,severity_truth\n"
                "a.jpg,f1,real,high,true,轻微（微量血迹/淤青）\n"
                "b.png,f2,ai_anime,low,0,\n",
                encoding="utf-8",
            )
            items = load_manifest(manifest, root / "images")
            self.assertEqual(len(items), 2)
            self.assertTrue(items[0].has_blood)
            self.assertFalse(items[1].has_blood)
            self.assertEqual(items[0].severity_truth, "轻微（微量血迹/淤青）")

    def test_manifest_rejects_duplicate_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.jpg").write_bytes(b"fixture")
            manifest = root / "manifest.csv"
            manifest.write_text(
                "file,folder,domain,red_level,has_blood\n"
                "a.jpg,f,real,high,1\n"
                "a.jpg,f,real,high,0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "重复"):
                load_manifest(manifest, root)

    def test_manifest_rejects_folder_label_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.jpg").write_bytes(b"fixture")
            manifest = root / "manifest.csv"
            manifest.write_text(
                "file,folder,domain,red_level,has_blood\n"
                "a.jpg,01_ai2_high_red_no_blood,ai_anime,high,1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "no_blood"):
                load_manifest(manifest, root)

    def test_binary_metrics_detect_perfect_separation(self):
        result = binary_metrics([False, False, True, True], [0.1, 0.2, 0.8, 0.9])
        self.assertEqual(result["auroc"], 1.0)
        self.assertEqual(result["auprc"], 1.0)
        self.assertEqual(result["thresholds"]["0.75"]["sensitivity"], 1.0)
        self.assertEqual(result["thresholds"]["0.75"]["false_positive_rate"], 0.0)

    def test_metrics_at_explicit_threshold(self):
        result = metrics_at_threshold(
            [False, False, True, True],
            [0.10, 0.40, 0.60, 0.90],
            0.50,
        )
        self.assertEqual(result["sensitivity"], 1.0)
        self.assertEqual(result["false_positive_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
