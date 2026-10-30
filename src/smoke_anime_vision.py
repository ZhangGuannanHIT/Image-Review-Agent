# -*- coding: utf-8 -*-
"""Smoke-test the configured model's anime image-review capability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.anime_vision import AnimeVisualReviewer
from backend.llm import LLMClient


def main() -> None:
    parser = argparse.ArgumentParser(description="动漫视觉复核连通性测试")
    parser.add_argument("--image", required=True)
    parser.add_argument("--config", default="config/llm.yaml")
    parser.add_argument("--max-side", type=int, default=1280)
    args = parser.parse_args()
    model = LLMClient.from_config(args.config).build_model()
    result = AnimeVisualReviewer(model, max_side=args.max_side).review(
        Path(args.image).read_bytes()
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
