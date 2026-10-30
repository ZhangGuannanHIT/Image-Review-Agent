# -*- coding: utf-8 -*-

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from PIL import Image
from torch import nn

from backend.mock import MockReviewEngine
from backend.tools import text_evidence
from src import cli, labels
from src.engine import ReviewEngine, _multi_label
from src.model import MultiTaskHead, ReviewModel


class FixedModel(nn.Module):
    def __init__(self, quality_logits):
        super().__init__()
        self.quality_logits = torch.tensor([quality_logits], dtype=torch.float32)

    def forward_vision(self, **kwargs):
        return torch.zeros(1, 4)

    def forward(self, embeddings):
        return {
            "risk": torch.tensor([[4., 0., 0., 0., 0.]]),
            "type": torch.tensor([[4., -4., -4., -4.]]),
            "injury": torch.tensor([[4., 0., 0., 0.]]),
            "quality": self.quality_logits,
        }


def make_engine(quality_logits):
    engine = ReviewEngine.__new__(ReviewEngine)
    engine.model = FixedModel(quality_logits)
    engine.processor = object()
    engine.trained = False
    engine.type_thresholds = dict.fromkeys(labels.TYPE_CLASSES, 0.5)
    engine.quality_thresholds = dict.fromkeys(labels.QUALITY_CLASSES, 0.5)
    engine.prepare_image = lambda image: {
        "pixel_values": torch.zeros(1),
        "pixel_attention_mask": torch.ones(1),
        "spatial_shapes": torch.ones(1),
    }
    return engine


class QualityInferenceTests(unittest.TestCase):
    def test_predict_and_workflow_evidence_preserve_multiple_labels(self):
        output = make_engine([-4., 4., -4., -4., 4.]).predict(b"in-memory fixture")
        expected = [labels.QUALITY_CLASSES[1], labels.QUALITY_CLASSES[4]]
        self.assertEqual(output.quality_labels, expected)
        self.assertEqual(set(output.quality), {"labels", "scores", "thresholds"})
        self.assertGreater(sum(output.quality["scores"].values()), 1)
        evidence = text_evidence(output)
        for label in expected:
            self.assertIn(label, evidence)
        self.assertIn("多标签", evidence)
        self.assertTrue(0 <= output.conf <= 1)

    def test_quality_does_not_force_argmax_when_no_class_reaches_threshold(self):
        output = make_engine([-4.] * 5).predict(b"fixture")
        self.assertEqual(output.quality_labels, [])

    def test_quality_supports_individual_thresholds(self):
        engine = make_engine([-4., 2., -4., -4., 2.])
        engine.quality_thresholds[labels.QUALITY_CLASSES[1]] = 0.9
        output = engine.predict(b"fixture")
        self.assertEqual(output.quality_labels, [labels.QUALITY_CLASSES[4]])

    def test_threshold_comparison_precedes_probability_rounding(self):
        logits = torch.logit(torch.tensor([[0.4999998]]))
        output = _multi_label(logits, ["label"], {"label": 0.5})
        self.assertEqual(output["scores"]["label"], 0.5)
        self.assertEqual(output["labels"], [])

    def test_mock_and_cli_use_multilabel_quality(self):
        self.assertEqual(MockReviewEngine().predict(b"fixture").quality_labels, ["清晰"])
        engine = make_engine([-4., 4., -4., -4., 4.])
        stdout = io.StringIO()
        with patch("sys.argv", ["cli", "--image", "unused.png"]), \
             patch("src.engine.ReviewEngine", return_value=engine), \
             patch("PIL.Image.open", return_value=Image.new("RGB", (2, 2))), \
             contextlib.redirect_stdout(stdout):
            cli.main()
        quality_line = next(line for line in stdout.getvalue().splitlines() if line.startswith("quality:"))
        self.assertIn(labels.QUALITY_CLASSES[1], quality_line)
        self.assertIn(labels.QUALITY_CLASSES[4], quality_line)

    def test_fp32_head_accepts_half_precision_embeddings_and_backpropagates(self):
        model = ReviewModel.__new__(ReviewModel)
        nn.Module.__init__(model)
        model.vision = nn.Linear(4, 4).half()
        model.head = MultiTaskHead(embed_dim=4, hidden=8).float()
        output = model(torch.ones(2, 4, dtype=torch.float16))
        self.assertEqual(output["quality"].dtype, torch.float32)
        loss = nn.functional.binary_cross_entropy_with_logits(output["quality"], torch.ones(2, 5))
        loss.backward()
        self.assertIsNotNone(model.head.quality_head.weight.grad)


class QualityCheckpointTests(unittest.TestCase):
    def make_loader(self):
        engine = ReviewEngine.__new__(ReviewEngine)
        engine.model = Mock()
        engine.lora = False
        engine.head_loaded = False
        engine.adapter_loaded = False
        engine.trained = False
        return engine

    def test_rejects_legacy_single_label_checkpoint_before_loading_head(self):
        legacy_schema = {name: values for name, values in labels.checkpoint_schema().items()
                         if name in ("risk", "type", "injury", "quality")}
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "head.pt").write_bytes(b"unused")
            (Path(directory) / "labels.json").write_text(json.dumps(legacy_schema), encoding="utf-8")
            engine = self.make_loader()
            with self.assertRaisesRegex(ValueError, "quality.*多标签"):
                engine._load_checkpoint(directory)
            engine.model.load_head.assert_not_called()

    def test_rejects_weights_without_task_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "head.pt").write_bytes(b"unused")
            engine = self.make_loader()
            with self.assertRaisesRegex(ValueError, "缺少 labels.json"):
                engine._load_checkpoint(directory)
            engine.model.load_head.assert_not_called()

    def test_multilabel_checkpoint_round_trip(self):
        model = ReviewModel.__new__(ReviewModel)
        nn.Module.__init__(model)
        model.vision = nn.Linear(4, 4)
        model.head = MultiTaskHead(embed_dim=4, hidden=8)
        expected = model.head.quality_head.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            model.save_trainable(directory)
            schema = json.loads((Path(directory) / "labels.json").read_text(encoding="utf-8"))
            self.assertEqual(schema["task_types"]["quality"], "multi_label")
            with torch.no_grad():
                model.head.quality_head.weight.zero_()
            engine = self.make_loader()
            engine.model = model
            engine._load_checkpoint(directory)
            self.assertTrue(engine.trained)
            torch.testing.assert_close(model.head.quality_head.weight, expected)


if __name__ == "__main__":
    unittest.main()
