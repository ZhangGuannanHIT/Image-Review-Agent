# -*- coding: utf-8 -*-
"""读取 checkpoint 并按当前四字段标签体系评估。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from .dataset import parse_label_studio
from .engine import ReviewEngine
from .metrics import review_metrics
from .training_dataset import ReviewCollator, ReviewDataset, move_batch_to_device


def evaluate_loader(model, loader, device, type_threshold: float = 0.5) -> dict:
    records = {key: [] for key in (
        "risk_true", "risk_pred", "type_true", "type_pred", "type_valid",
        "injury_true", "injury_pred", "quality_true", "quality_pred",
    )}
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            embeds = model.forward_vision(
                batch["pixel_values"], batch["pixel_attention_mask"], batch["spatial_shapes"]
            )
            logits = model(embeds)
            records["risk_true"].extend(batch["risk_label"].cpu().tolist())
            records["risk_pred"].extend(logits["risk"].argmax(-1).cpu().tolist())
            records["type_true"].extend(batch["type_target"].cpu().int().tolist())
            records["type_pred"].extend((torch.sigmoid(logits["type"]) >= type_threshold).cpu().int().tolist())
            records["type_valid"].extend(batch["type_valid"].cpu().tolist())
            records["injury_true"].extend(batch["injury_label"].cpu().tolist())
            records["injury_pred"].extend(logits["injury"].argmax(-1).cpu().tolist())
            records["quality_true"].extend(batch["quality_label"].cpu().tolist())
            records["quality_pred"].extend(logits["quality"].argmax(-1).cpu().tolist())
    return review_metrics(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="评估 FG-CLIP2 审核模型")
    parser.add_argument("--config", default="config/train.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    data = config["data"]
    if not data.get("json_path") or not data.get("image_root"):
        raise ValueError("请先在 train.yaml 配置 data.json_path 和 data.image_root")
    samples, _ = parse_label_studio(data["json_path"], data["image_root"], report=False)
    with open(data.get("split_path", "data/split.json"), "r", encoding="utf-8") as file:
        split = json.load(file)
    chosen = [samples[index] for index in split[args.split]]
    engine = ReviewEngine(
        config["model_path"], ckpt_dir=args.checkpoint,
        device=config.get("device", "cuda"), strict_load=True,
    )
    loader = DataLoader(
        ReviewDataset(chosen), batch_size=config["optim"].get("batch_size", 16),
        shuffle=False, collate_fn=ReviewCollator(engine.processor),
    )
    metrics = evaluate_loader(engine.model, loader, engine.device)
    output_dir = Path(config.get("eval_dir", "eval_results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.split}_metrics.json"
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
