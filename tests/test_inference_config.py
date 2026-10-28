# -*- coding: utf-8 -*-
"""单图推理配置契约测试：阈值必须覆盖全部多标签类别，且位于 [0, 1]。

本模块不依赖 torch，CPU 环境也能运行；它防止 config/inference.yaml 与
src/labels.py 的类别列表再次脱节（历史上 quality 阈值曾整块缺失）。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from src.cli import DEFAULT_CONFIG_PATH, load_inference_config
from src.labels import QUALITY_CLASSES, TYPE_CLASSES


class InferenceConfigTests(unittest.TestCase):
    def test_default_config_file_exists(self):
        self.assertTrue(Path(DEFAULT_CONFIG_PATH).is_file())

    def test_thresholds_cover_every_multilabel_class(self):
        config = load_inference_config()
        self.assertEqual(set(config["type_thresholds"]), set(TYPE_CLASSES))
        self.assertEqual(set(config["quality_thresholds"]), set(QUALITY_CLASSES))

    def test_thresholds_are_probabilities(self):
        config = load_inference_config()
        for group in ("type_thresholds", "quality_thresholds"):
            for label, value in config[group].items():
                with self.subTest(group=group, label=label):
                    self.assertGreaterEqual(float(value), 0.0)
                    self.assertLessEqual(float(value), 1.0)

    def test_model_paths_and_flags_present(self):
        config = load_inference_config()
        self.assertTrue(config["model_path"])
        self.assertIn("ckpt_dir", config)
        self.assertEqual(
            config.get("anime_style_router_dir"),
            "checkpoints/anime_style_router",
        )
        self.assertEqual(float(config.get("anime_style_min_score")), 0.90)
        self.assertIn("device", config)
        self.assertIn("lora", config)

    def test_backend_anime_visual_gate_is_explicit(self):
        import yaml
        backend = yaml.safe_load(
            (Path(__file__).resolve().parents[1] / "backend/config.yaml").read_text(
                encoding="utf-8"
            )
        )
        review = backend["anime_vision_review"]
        self.assertTrue(review["enabled"])
        self.assertEqual(float(review["min_style_score"]), 0.90)
        self.assertTrue(review["allow_safety_review"])
        self.assertGreaterEqual(int(review["max_tokens"]), 300)
        self.assertLessEqual(int(review["max_tokens"]), 512)
        self.assertEqual(int(review["max_retries"]), 1)
        self.assertTrue(review["json_response"])

    def test_backend_explanation_uses_selective_fixed_template(self):
        backend = yaml.safe_load(
            (Path(__file__).resolve().parents[1] / "backend/config.yaml").read_text(
                encoding="utf-8"
            )
        )
        features = backend["features"]
        self.assertEqual(features["llm_explanation_mode"], "selective")
        self.assertEqual(
            features["llm_explanation_actions"],
            ["review", "block"],
        )

    def test_missing_file_falls_back_to_defaults(self):
        self.assertEqual(load_inference_config("config/does-not-exist.yaml"), {})

    def test_non_mapping_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text(yaml.safe_dump(["not", "a", "mapping"]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "YAML 映射"):
                load_inference_config(str(path))


if __name__ == "__main__":
    unittest.main()
