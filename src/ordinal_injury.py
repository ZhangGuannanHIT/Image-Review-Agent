# -*- coding: utf-8 -*-
"""Ordinal injury head, CORAL probabilities, and severity metrics."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .labels import INJURY_CLASSES


class OrdinalInjuryHead(nn.Module):
    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, len(INJURY_CLASSES) - 1),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.network(embeddings.float())


def cumulative_targets(labels: torch.Tensor) -> torch.Tensor:
    labels = labels.long().clamp(min=0, max=len(INJURY_CLASSES) - 1)
    thresholds = torch.arange(1, len(INJURY_CLASSES), device=labels.device)
    return (labels.unsqueeze(-1) >= thresholds).float()


def class_probabilities(logits: torch.Tensor) -> torch.Tensor:
    cumulative = torch.sigmoid(logits.float())
    cumulative = torch.cummin(cumulative, dim=-1).values
    probabilities = [
        1 - cumulative[..., 0],
        cumulative[..., 0] - cumulative[..., 1],
        cumulative[..., 1] - cumulative[..., 2],
        cumulative[..., 2],
    ]
    return torch.stack(probabilities, dim=-1).clamp(min=0, max=1)


def quadratic_weighted_kappa(expected, predicted, classes: int = 4) -> float:
    expected = np.asarray(expected, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    matrix = np.zeros((classes, classes), dtype=np.float64)
    for truth, guess in zip(expected, predicted):
        if 0 <= truth < classes and 0 <= guess < classes:
            matrix[truth, guess] += 1
    weights = np.zeros((classes, classes), dtype=np.float64)
    for i in range(classes):
        for j in range(classes):
            weights[i, j] = ((i - j) ** 2) / ((classes - 1) ** 2)
    expected_matrix = np.outer(matrix.sum(axis=1), matrix.sum(axis=0)) / max(matrix.sum(), 1)
    numerator = float((weights * matrix).sum())
    denominator = float((weights * expected_matrix).sum())
    return 1.0 - numerator / denominator if denominator else 0.0


def severity_metrics(expected, predicted) -> dict:
    expected = np.asarray(expected, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    if len(expected) != len(predicted) or len(expected) == 0:
        raise ValueError("expected 与 predicted 必须等长且非空")
    matrix = np.zeros((len(INJURY_CLASSES), len(INJURY_CLASSES)), dtype=int)
    for truth, guess in zip(expected, predicted):
        matrix[truth, guess] += 1
    per_class = {}
    f1_values = []
    for index, label in enumerate(INJURY_CLASSES):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_class[label] = {
            "precision": precision, "recall": recall, "f1": f1, "support": int(matrix[index].sum()),
        }
        f1_values.append(f1)
    return {
        "n": len(expected),
        "accuracy": float((expected == predicted).mean()),
        "adjacent_accuracy": float((np.abs(expected - predicted) <= 1).mean()),
        "macro_f1": float(np.mean(f1_values)),
        "qwk": quadratic_weighted_kappa(expected, predicted),
        "severe_recall": per_class[INJURY_CLASSES[3]]["recall"],
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
    }


def score_image(engine, head: OrdinalInjuryHead, image) -> dict:
    inputs = engine.prepare_image(image)
    with torch.inference_mode():
        embeddings = engine.model.forward_vision(
            inputs["pixel_values"],
            inputs["pixel_attention_mask"],
            inputs["spatial_shapes"],
        )
        logits = head(embeddings)
        probabilities = class_probabilities(logits)[0].cpu().tolist()
    index = int(np.argmax(probabilities))
    return {
        "label": INJURY_CLASSES[index],
        "index": index,
        "confidence": float(probabilities[index]),
        "probabilities": {
            label: float(probability)
            for label, probability in zip(INJURY_CLASSES, probabilities)
        },
    }
