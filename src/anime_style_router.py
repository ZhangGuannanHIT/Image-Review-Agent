"""Lightweight anime/real style router over frozen FG-CLIP embeddings."""

from __future__ import annotations

from pathlib import Path
import json

import torch
from torch import nn


class AnimeStyleRouter(nn.Module):
    def __init__(self, embed_dim: int = 768, hidden: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.network(embeddings.float()).squeeze(-1)


def load_anime_style_router(
    directory: str | Path,
    embed_dim: int,
    device: str | torch.device,
) -> tuple[AnimeStyleRouter, dict]:
    directory = Path(directory)
    metadata = json.loads(
        (directory / "metadata.json").read_text(encoding="utf-8")
    )
    head = AnimeStyleRouter(
        embed_dim=int(metadata.get("embedding_dim", embed_dim)),
        hidden=int(metadata.get("hidden_dim", 128)),
    )
    state = torch.load(
        directory / "router.pt", map_location=device, weights_only=True
    )
    head.load_state_dict(state, strict=True)
    return head.to(device).eval(), metadata
