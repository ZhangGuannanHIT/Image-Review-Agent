# -*- coding: utf-8 -*-
"""Frozen-embedding binary injury head and operating-point selection."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


class BinaryInjuryHead(nn.Module):
    """Small calibrated head over the frozen FG-CLIP2 visual embedding."""

    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.network(embeddings.float()).squeeze(-1)


def select_threshold(
    labels, scores, max_false_positive_rate: float = 0.10,
) -> dict:
    """Choose the most sensitive threshold under an FPR budget."""
    y = np.asarray(list(labels), dtype=bool)
    s = np.asarray(list(scores), dtype=np.float64)
    if len(y) != len(s) or len(y) == 0:
        raise ValueError("labels 与 scores 必须等长且非空")
    if not 0 <= max_false_positive_rate <= 1:
        raise ValueError("max_false_positive_rate 必须位于 [0, 1]")
    candidates = []
    for threshold in np.linspace(0.05, 0.95, 181):
        predicted = s >= threshold
        tp = int(np.logical_and(predicted, y).sum())
        fp = int(np.logical_and(predicted, ~y).sum())
        tn = int(np.logical_and(~predicted, ~y).sum())
        fn = int(np.logical_and(~predicted, y).sum())
        sensitivity = tp / max(tp + fn, 1)
        false_positive_rate = fp / max(fp + tn, 1)
        candidates.append(
            {
                "threshold": float(threshold),
                "sensitivity": sensitivity,
                "false_positive_rate": false_positive_rate,
                "specificity": 1 - false_positive_rate,
            }
        )
    feasible = [
        item for item in candidates
        if item["false_positive_rate"] <= max_false_positive_rate
    ]
    pool = feasible or candidates
    if feasible:
        best = max(pool, key=lambda item: (item["sensitivity"], item["threshold"]))
    else:
        best = min(
            pool,
            key=lambda item: (item["false_positive_rate"], -item["sensitivity"]),
        )
    return {**best, "fpr_budget": max_false_positive_rate, "feasible": bool(feasible)}


def score_image(engine, head: BinaryInjuryHead, image) -> float:
    """Run one image through the frozen visual tower and binary head."""
    inputs = engine.prepare_image(image)
    required = ("pixel_values", "pixel_attention_mask", "spatial_shapes")
    missing = [name for name in required if name not in inputs]
    if missing:
        raise RuntimeError(f"processor 缺少字段: {missing}")
    engine.model.eval()
    head.eval()
    with torch.inference_mode():
        embeddings = engine.model.forward_vision(
            inputs["pixel_values"],
            inputs["pixel_attention_mask"],
            inputs["spatial_shapes"],
        )
        score = torch.sigmoid(head(embeddings)).item()
    return float(score)
