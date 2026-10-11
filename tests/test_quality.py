# -*- coding: utf-8 -*-

import io
import json
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from PIL import Image

from src.dataset import parse_label_studio
from src.evaluate import evaluate_loader
from src.labels import QUALITY_CLASSES
from src.losses import MultiTaskLoss
from src.metrics import multilabel_metrics
from src.training_dataset import ReviewCollator


def quality_task(task_id, labels=None):
    result = [] if labels is None else [{
        "from_name": "quality", "value": {"taxonomy": [[label] for label in labels]},
    }]
    return {
        "id": task_id, "data": {"image": f"{task_id}.png"},
        "annotations": [{"result": result}],
    }


def parse_tasks(tasks):
    # 模拟导出文件，测试不创建图片、JSON 或其他临时文件。
    with patch("builtins.open", return_value=io.StringIO(json.dumps(tasks, ensure_ascii=False))):
        return parse_label_studio("annotations.json", report=False)


class QualityTests(unittest.TestCase):
    def test_multiselection_is_preserved_independent_of_order(self):
        samples, _ = parse_tasks([
            quality_task(1, ["故意遮挡", "轻微模糊"]),
            quality_task(2, ["轻微模糊", "故意遮挡", "轻微模糊"]),
        ])
        for sample in samples:
            self.assertEqual(sample.quality_set, [0, 1, 0, 0, 1])
            self.assertTrue(sample.quality_valid)

    def test_missing_and_unmapped_quality_are_masked(self):
        samples, report = parse_tasks([
            quality_task(1), quality_task(2, []), quality_task(3, ["未知质量"]),
            quality_task(4, ["轻微模糊", "未知质量"]), quality_task(5, ["清晰"]),
        ])
        self.assertEqual([s.quality_valid for s in samples], [False, False, False, False, True])
        self.assertEqual(samples[0].quality_set, [0, 0, 0, 0, 0])
        self.assertEqual(samples[3].quality_set, [0, 1, 0, 0, 0])
        self.assertEqual(samples[4].quality_set, [1, 0, 0, 0, 0])
        self.assertEqual(report["missing_count"]["quality"], 2)
        self.assertEqual(report["field_distribution"]["unmapped_quality"]["未知质量"], 2)

    def test_collator_emits_multihot_targets_and_validity(self):
        samples, _ = parse_tasks([quality_task(1, ["轻微模糊", "故意遮挡"]), quality_task(2)])

        def processor(**kwargs):
            return {"pixel_values": torch.zeros(len(kwargs["images"]), 1)}

        items = [
            {"image": Image.new("RGB", (16, 16)), "sample": sample, "sample_index": index}
            for index, sample in enumerate(samples)
        ]
        batch = ReviewCollator(processor)(items)
        self.assertEqual(batch["quality_target"].dtype, torch.float32)
        self.assertEqual(batch["quality_target"].tolist(), [[0, 1, 0, 0, 1], [0, 0, 0, 0, 0]])
        self.assertEqual(batch["quality_valid"].dtype, torch.bool)
        self.assertEqual(batch["quality_valid"].tolist(), [True, False])
        self.assertNotIn("quality_label", batch)

    @staticmethod
    def loss_inputs():
        logits = {
            "risk": torch.zeros(2, 5, requires_grad=True),
            "type": torch.zeros(2, 4, requires_grad=True),
            "injury": torch.zeros(2, 4, requires_grad=True),
            "quality": torch.zeros(2, 5, requires_grad=True),
        }
        batch = {
            "risk_label": torch.full((2,), -1, dtype=torch.long),
            "type_target": torch.zeros(2, 4),
            "type_valid": torch.zeros(2, dtype=torch.bool),
            "injury_label": torch.full((2,), -1, dtype=torch.long),
            "quality_target": torch.tensor([[0., 1., 0., 0., 1.], [0., 0., 0., 0., 0.]]),
            "quality_valid": torch.tensor([True, False]),
        }
        return logits, batch

    def test_quality_uses_weighted_bce_and_excludes_missing_gradients(self):
        logits, batch = self.loss_inputs()
        pos_weight = torch.tensor([1., 2., 3., 4., 5.])
        criterion = MultiTaskLoss(quality_pos_weight=pos_weight)
        parts = criterion(logits, batch, return_parts=True)
        expected = F.binary_cross_entropy_with_logits(
            logits["quality"][:1], batch["quality_target"][:1], pos_weight=pos_weight,
        )
        torch.testing.assert_close(parts["quality"], expected)
        parts["total"].backward()
        self.assertTrue(torch.equal(logits["quality"].grad[1], torch.zeros(5)))
        self.assertLess(logits["quality"].grad[0, 1].item(), 0)
        self.assertLess(logits["quality"].grad[0, 4].item(), 0)
        self.assertGreater(logits["quality"].grad[0, 0].item(), 0)

    def test_all_quality_missing_has_zero_loss_and_keeps_graph(self):
        logits, batch = self.loss_inputs()
        batch["quality_valid"].fill_(False)
        parts = MultiTaskLoss()(logits, batch, return_parts=True)
        self.assertEqual(parts["quality"].item(), 0.0)
        parts["total"].backward()
        self.assertIsNotNone(logits["quality"].grad)
        self.assertTrue(torch.equal(logits["quality"].grad, torch.zeros(2, 5)))

    def test_multilabel_quality_metrics_include_fifth_class_and_mask(self):
        metrics = multilabel_metrics(
            [[0, 1, 0, 0, 1], [0, 0, 0, 0, 0]],
            [[0, 1, 0, 0, 0], [1, 1, 1, 1, 1]], [True, False], QUALITY_CLASSES,
        )
        self.assertEqual(metrics["count"], 1)
        self.assertAlmostEqual(metrics["micro_f1"], 2 / 3)
        self.assertAlmostEqual(metrics["macro_f1"], 1 / 5)
        self.assertEqual(set(metrics["per_class"]), set(QUALITY_CLASSES))
        self.assertEqual(metrics["per_class"][QUALITY_CLASSES[4]]["recall"], 0.0)
        self.assertNotIn("accuracy", metrics)

    def test_no_quality_annotations_has_no_metric(self):
        metrics = multilabel_metrics([[0] * 5], [[1] * 5], [False], QUALITY_CLASSES)
        self.assertEqual(metrics["count"], 0)
        self.assertIsNone(metrics["micro_f1"])
        self.assertIsNone(metrics["macro_f1"])

    def test_evaluation_decodes_independent_labels_with_sigmoid(self):
        logits, batch = self.loss_inputs()
        logits["quality"] = torch.tensor([[-5., 5., -5., -5., 5.], [5., 5., 5., 5., 5.]])
        batch.update({
            "pixel_values": torch.zeros(2, 1),
            "pixel_attention_mask": torch.ones(2, 1),
            "spatial_shapes": torch.ones(2, 2),
        })

        class FakeModel:
            def eval(self):
                return self

            def forward_vision(self, *args):
                return args[0]

            def __call__(self, embeds):
                return logits

        metrics = evaluate_loader(FakeModel(), [batch], "cpu")
        self.assertEqual(metrics["quality"]["count"], 1)
        self.assertEqual(metrics["quality"]["micro_f1"], 1.0)
        self.assertEqual(metrics["quality"]["per_class"][QUALITY_CLASSES[4]]["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
