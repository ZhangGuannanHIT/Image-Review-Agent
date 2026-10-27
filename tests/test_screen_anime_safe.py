# -*- coding: utf-8 -*-

import csv
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw

from src.build_anime_review_queue import priority
from src.screen_anime_safe import build_manifest


class AnimeManifestTests(unittest.TestCase):
    @staticmethod
    def image(path: Path, reverse: bool = False):
        image = Image.new("RGB", (64, 64), "white")
        draw = ImageDraw.Draw(image)
        if reverse:
            draw.rectangle((32, 32, 64, 64), fill="black")
        else:
            draw.rectangle((0, 0, 32, 32), fill="black")
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)

    def test_manifest_marks_exact_duplicate_and_assigns_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "images"
            self.image(root / "清晰" / "a.png")
            self.image(root / "清晰" / "a_copy.png")
            self.image(root / "轻微模糊" / "b.png", reverse=True)
            output = Path(directory) / "manifest.csv"
            summary = build_manifest(root, output)
            with output.open("r", encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))
        self.assertEqual(summary["n_valid"], 3)
        self.assertEqual(summary["n_exact_duplicate"], 1)
        self.assertEqual(summary["n_selected"], 2)
        self.assertEqual({row["presumed_safe"] for row in rows}, {"1"})
        self.assertEqual({row["human_verified"] for row in rows}, {"0"})
        self.assertEqual({row["label_status"] for row in rows}, {"unreviewed"})
        self.assertTrue({row["split"] for row in rows}.issubset(
            {"dev", "validation", "test"}
        ))

    def test_review_queue_prioritises_l1_and_high_binary_l0(self):
        self.assertEqual(priority({"risk": "L1_轻微违规"}), 1)
        self.assertEqual(priority({"risk": "L0_安全", "binary_score": 0.8}), 2)
        self.assertEqual(priority({"risk": "X_无法判定"}), 3)
        self.assertEqual(priority({"risk": "L2_中度违规"}), 4)

    def test_label_studio_config_is_well_formed(self):
        path = Path(__file__).resolve().parents[1] / (
            "config/anime_annotation_labelstudio.xml"
        )
        root = ET.parse(path).getroot()
        self.assertEqual(root.tag, "View")
        names = {node.attrib.get("name") for node in root}
        self.assertTrue({
            "risk_level", "violence_type", "injury_detail", "quality",
            "style_domain", "scene_context", "note",
        }.issubset(names))


if __name__ == "__main__":
    unittest.main()
