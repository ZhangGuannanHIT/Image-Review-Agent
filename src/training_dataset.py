# -*- coding: utf-8 -*-
"""基于冻结 Label Studio 标签契约的 PyTorch Dataset 与动态图片 Collator。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

from .dataset import Sample
from .engine import determine_max_patches
from .labels import IGNORE_INDEX


class ReviewDataset(Dataset):
    def __init__(self, samples: Sequence[Sample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        path = Path(sample.image_path)
        if not path.is_file():
            raise FileNotFoundError(f"训练图片不存在: {path}")
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        return {"image": image, "sample": sample, "sample_index": index}


class ReviewCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        images = [item["image"] for item in items]
        max_patches = max(determine_max_patches(image) for image in images)
        encoded = self.processor(images=images, max_num_patches=max_patches, return_tensors="pt")
        samples = [item["sample"] for item in items]
        encoded.update({
            "risk_label": torch.tensor([
                s.risk if s.risk is not None else IGNORE_INDEX for s in samples
            ], dtype=torch.long),
            "type_target": torch.tensor([s.type_set for s in samples], dtype=torch.float32),
            "type_valid": torch.tensor([s.type_valid for s in samples], dtype=torch.bool),
            "injury_label": torch.tensor([
                s.injury if s.injury is not None else IGNORE_INDEX for s in samples
            ], dtype=torch.long),
            "quality_target": torch.tensor([s.quality_set for s in samples], dtype=torch.float32),
            "quality_valid": torch.tensor([s.quality_valid for s in samples], dtype=torch.bool),
            "sample_indices": torch.tensor([item["sample_index"] for item in items], dtype=torch.long),
        })
        return encoded


def move_batch_to_device(batch: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
