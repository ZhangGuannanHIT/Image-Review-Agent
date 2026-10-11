# -*- coding: utf-8 -*-

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml
from PIL import Image

from src.dataset import Sample
from src.model import MultiTaskHead
from src.train import _quality_pos_weight, load_training_config, main


REPO_ROOT = Path(__file__).resolve().parents[1]


class TinyTrainingModel(torch.nn.Module):
    """验证真实训练控制流的 CPU 小模型，不加载基础模型。"""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.vision = torch.nn.Linear(3, 4)
        self.head = MultiTaskHead(embed_dim=4, hidden=8)

    def forward_vision(self, pixel_values, pixel_attention_mask, spatial_shapes):
        return self.vision(pixel_values)

    def forward(self, embeds):
        return self.head(embeds)

    def save_trainable(self, output_dir):
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        torch.save(self.head.state_dict(), destination / "head.pt")


class TrainingTests(unittest.TestCase):
    def test_smoke_default_selects_isolated_config(self):
        config, steps = load_training_config(smoke_steps=2, project_root=REPO_ROOT)
        self.assertEqual(steps, 2)
        self.assertEqual(config["data"]["split_path"], "data/smoke/split.json")
        self.assertEqual(config["output_dir"], "checkpoints/smoke")
        self.assertEqual(config["eval_dir"], "eval_results/smoke")
        self.assertEqual(config["optim"]["epochs"], 1)

    def test_explicit_smoke_config_defaults_to_one_step(self):
        _, steps = load_training_config("config/train.smoke.yaml", project_root=REPO_ROOT)
        self.assertEqual(steps, 1)

    def test_formal_default_is_unchanged(self):
        config, steps = load_training_config(project_root=REPO_ROOT)
        self.assertEqual(steps, 0)
        self.assertEqual(config["data"]["split_path"], "data/split.json")
        self.assertEqual(config["output_dir"], "checkpoints")

    def test_smoke_rejects_formal_config_and_negative_steps(self):
        with self.assertRaisesRegex(ValueError, "data.split_path"):
            load_training_config("config/train.yaml", 1, project_root=REPO_ROOT)
        with self.assertRaisesRegex(ValueError, "不能为负数"):
            load_training_config(smoke_steps=-1, project_root=REPO_ROOT)

    def test_each_smoke_output_must_stay_in_dedicated_directory(self):
        config, _ = load_training_config(smoke_steps=1, project_root=REPO_ROOT)
        cases = (
            ("data.split_path", "data/split.json"),
            ("data.split_path", "data/smoke/../split.json"),
            ("data.split_path", "data/smoke-other/split.json"),
            ("output_dir", "checkpoints"),
            ("eval_dir", "eval_results"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for field, value in cases:
                with self.subTest(field=field, value=value):
                    changed = copy.deepcopy(config)
                    if field == "data.split_path":
                        changed["data"]["split_path"] = value
                    else:
                        changed[field] = value
                    (root / "unsafe.yaml").write_text(yaml.safe_dump(changed), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, field):
                        load_training_config("unsafe.yaml", 1, project_root=root)
            self.assertFalse((root / "data").exists())
            self.assertFalse((root / "checkpoints").exists())
            self.assertFalse((root / "eval_results").exists())

    def test_quality_pos_weight_excludes_missing_labels(self):
        def sample(target, valid=True):
            return Sample("unused.png", 0, [0, 0, 0, 1], True, 0, target, valid)

        samples = [
            sample([0, 1, 0, 0, 1]), sample([0, 1, 0, 0, 1]),
            sample([1, 0, 0, 0, 0]), sample([0, 0, 0, 0, 0], False),
        ]
        torch.testing.assert_close(_quality_pos_weight(samples), torch.tensor([2., .5, 3., 3., .5]))
        torch.testing.assert_close(_quality_pos_weight(samples[-1:]), torch.ones(5))

    def test_smoke_main_trains_and_saves_only_in_isolated_directories(self):
        config, _ = load_training_config(smoke_steps=1, project_root=REPO_ROOT)

        def processor(*, images, **kwargs):
            count = len(images)
            return {
                "pixel_values": torch.ones(count, 3),
                "pixel_attention_mask": torch.ones(count, 1),
                "spatial_shapes": torch.ones(count, 2),
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config").mkdir()
            image_path = root / "sample.png"
            Image.new("RGB", (8, 8)).save(image_path)
            config["device"] = "cpu"
            config["data"]["json_path"] = "unused.json"
            config["data"]["image_root"] = "."
            config["data"]["split"] = [0.5, 0.25, 0.25]
            config["optim"]["amp"] = False
            (root / "config/train.smoke.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
            samples = [
                Sample(str(image_path), 0, [0, 0, 0, 1], True, 0, [0, 1, 0, 0, 1], True)
                for _ in range(8)
            ]
            # 预置正式划分和权重，检验内容不被冒烟覆盖。
            (root / "data").mkdir()
            (root / "checkpoints/best").mkdir(parents=True)
            (root / "eval_results").mkdir()
            formal_files = {
                root / "data/split.json": b"formal split",
                root / "checkpoints/best/head.pt": b"formal checkpoint",
                root / "eval_results/test_metrics.json": b"formal metrics",
            }
            for path, content in formal_files.items():
                path.write_bytes(content)
            with contextlib.chdir(root), \
                    patch("sys.argv", ["src.train", "--smoke-steps", "1"]), \
                    patch("src.train.parse_label_studio", return_value=(samples, {})), \
                    patch("src.train.ReviewModel", TinyTrainingModel), \
                    patch("src.train.AutoImageProcessor.from_pretrained", return_value=processor), \
                    contextlib.redirect_stdout(io.StringIO()):
                main()
            split = json.loads((root / "data/smoke/split.json").read_text(encoding="utf-8"))
            self.assertEqual(len(split["train"]), 4)
            self.assertTrue((root / "checkpoints/smoke/best/head.pt").is_file())
            metrics = json.loads((root / "checkpoints/smoke/best/metrics.json").read_text(encoding="utf-8"))
            self.assertIn("quality", metrics)
            for path, content in formal_files.items():
                self.assertEqual(path.read_bytes(), content)
            self.assertEqual({path.name for path in (root / "data").iterdir()}, {"split.json", "smoke"})
            self.assertEqual({path.name for path in (root / "checkpoints").iterdir()}, {"best", "smoke"})
            self.assertEqual({path.name for path in (root / "eval_results").iterdir()}, {"test_metrics.json"})


if __name__ == "__main__":
    unittest.main()
