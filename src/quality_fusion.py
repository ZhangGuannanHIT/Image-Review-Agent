# -*- coding: utf-8 -*-
"""Small multi-label head combining CLIP embeddings and quality features."""

from __future__ import annotations

import torch
from torch import nn


class QualityFusionHead(nn.Module):
    def __init__(self, embed_dim: int, feature_dim: int, hidden: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(embed_dim + feature_dim),
            nn.Linear(embed_dim + feature_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 5),
        )

    def forward(self, embeddings: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([embeddings.float(), features.float()], dim=-1))
