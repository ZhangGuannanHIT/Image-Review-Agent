# -*- coding: utf-8 -*-

import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.anime_style_router import AnimeStyleRouter, load_anime_style_router


class AnimeStyleRouterTests(unittest.TestCase):
    def test_router_shape_and_checkpoint_round_trip(self):
        head = AnimeStyleRouter(embed_dim=8, hidden=4)
        values = head(torch.randn(2, 8))
        self.assertEqual(tuple(values.shape), (2,))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            torch.save(head.state_dict(), path / "router.pt")
            (path / "metadata.json").write_text(
                json.dumps({
                    "embedding_dim": 8,
                    "hidden_dim": 4,
                    "threshold": 0.9,
                }),
                encoding="utf-8",
            )
            loaded, metadata = load_anime_style_router(
                path, embed_dim=8, device="cpu"
            )
            self.assertEqual(metadata["threshold"], 0.9)
            self.assertEqual(tuple(loaded(torch.randn(1, 8)).shape), (1,))


if __name__ == "__main__":
    unittest.main()
