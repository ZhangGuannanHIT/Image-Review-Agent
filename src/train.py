# -*- coding: utf-8 -*-
"""最小可复现训练入口；最终数据冻结前仅用于工程冒烟。"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor

from .dataset import Sample, parse_label_studio
from .evaluate import evaluate_loader
from .labels import INJURY_CLASSES, LOSS_WEIGHTS, QUALITY_CLASSES, RISK_CLASSES, TYPE_CLASSES
from .losses import MultiTaskLoss, median_freq_weights
from .model import ReviewModel
from .training_dataset import ReviewCollator, ReviewDataset, move_batch_to_device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_splits(samples: list[Sample], ratios: list[float], seed: int) -> dict[str, list[int]]:
    if len(ratios) != 3 or not np.isclose(sum(ratios), 1.0):
        raise ValueError("data.split 必须是和为 1 的 train/val/test 三项")
    rng = random.Random(seed)
    groups: dict[int, list[int]] = {}
    for index, sample in enumerate(samples):
        groups.setdefault(sample.risk if sample.risk is not None else -1, []).append(index)
    result = {"train": [], "val": [], "test": []}
    for indices in groups.values():
        rng.shuffle(indices)
        train_end = round(len(indices) * ratios[0])
        val_end = train_end + round(len(indices) * ratios[1])
        result["train"].extend(indices[:train_end])
        result["val"].extend(indices[train_end:val_end])
        result["test"].extend(indices[val_end:])
    for values in result.values():
        rng.shuffle(values)
    return result


def _weights(samples: list[Sample], field: str, count: int) -> torch.Tensor:
    values = [getattr(sample, field) for sample in samples if getattr(sample, field) is not None]
    counts = torch.bincount(torch.tensor(values, dtype=torch.long), minlength=count).float()
    return median_freq_weights(counts)


def _type_pos_weight(samples: list[Sample]) -> torch.Tensor:
    valid = torch.tensor([sample.type_set for sample in samples if sample.type_valid], dtype=torch.float32)
    if valid.numel() == 0:
        return torch.ones(len(TYPE_CLASSES))
    positive = valid.sum(0).clamp(min=1)
    return ((len(valid) - positive) / positive).clamp(min=0.1, max=10.0)


def _quality_pos_weight(samples: list[Sample]) -> torch.Tensor:
    valid = torch.tensor(
        [sample.quality_set for sample in samples if sample.quality_valid], dtype=torch.float32
    )
    if valid.numel() == 0:
        return torch.ones(len(QUALITY_CLASSES))
    positive = valid.sum(0)
    return ((len(valid) - positive) / positive.clamp(min=1)).clamp(min=0.1, max=10.0)


def load_training_config(
    config_path: str | None = None,
    smoke_steps: int = 0,
    *,
    project_root: Path | None = None,
) -> tuple[dict, int]:
    """选择训练配置，并在任何输出写入之前检查冒烟产物隔离。"""
    if smoke_steps < 0:
        raise ValueError("--smoke-steps 不能为负数")
    root = (project_root or Path.cwd()).resolve()
    selected = config_path or ("config/train.smoke.yaml" if smoke_steps else "config/train.yaml")
    with (root / selected).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("训练配置必须是 YAML 映射")
    is_smoke = smoke_steps > 0 or config.get("smoke", False)
    if is_smoke:
        paths = (
            ("data.split_path", config["data"].get("split_path", "data/split.json"), "data/smoke", True),
            ("output_dir", config.get("output_dir", "checkpoints"), "checkpoints/smoke", False),
            ("eval_dir", config.get("eval_dir", "eval_results"), "eval_results/smoke", False),
        )
        for name, value, dedicated_dir, is_file in paths:
            allowed = root / dedicated_dir
            actual = (root / value).resolve()
            if not actual.is_relative_to(allowed) or (is_file and actual == allowed):
                raise ValueError(
                    f"冒烟配置的 {name} 必须位于 {dedicated_dir}/ 内，实际为 {value!r}；"
                    "请使用 config/train.smoke.yaml，避免覆盖正式训练产物"
                )
        smoke_steps = smoke_steps or 1
    return config, smoke_steps


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 FG-CLIP2 四任务审核模型")
    parser.add_argument("--config", help="默认使用 train.yaml；指定 --smoke-steps 时默认使用 train.smoke.yaml")
    parser.add_argument("--smoke-steps", type=int, default=0, help="大于 0 时最多运行指定优化步数，且强制隔离输出")
    args = parser.parse_args()
    config, smoke_steps = load_training_config(args.config, args.smoke_steps)
    data_cfg, optim_cfg = config["data"], config["optim"]
    if not data_cfg.get("json_path") or not data_cfg.get("image_root"):
        raise ValueError("请先配置 data.json_path 和 data.image_root")
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    samples, _ = parse_label_studio(data_cfg["json_path"], data_cfg["image_root"], report=True)
    split_path = Path(data_cfg.get("split_path", "data/split.json"))
    if split_path.exists():
        split = json.loads(split_path.read_text(encoding="utf-8"))
    else:
        split = create_splits(samples, data_cfg.get("split", [0.85, 0.10, 0.05]), seed)
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    train_samples = [samples[index] for index in split["train"]]
    val_samples = [samples[index] for index in split["val"]]

    requested_device = config.get("device", "cuda")
    device = "cuda" if requested_device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = ReviewModel(
        config["model_path"], lora=True,
        lora_r=config["lora"]["r"], lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"], dtype=dtype, device=device,
    )
    # AMP 使用半精度计算，但被优化参数与梯度需保留 FP32，供 GradScaler 安全反缩放。
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    processor = AutoImageProcessor.from_pretrained(
        config["model_path"], trust_remote_code=True, local_files_only=True
    )
    collator = ReviewCollator(processor)
    train_loader = DataLoader(
        ReviewDataset(train_samples), batch_size=optim_cfg["batch_size"], shuffle=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        ReviewDataset(val_samples), batch_size=optim_cfg["batch_size"], shuffle=False,
        collate_fn=collator,
    )
    loss_fn = MultiTaskLoss(
        risk_weight=_weights(train_samples, "risk", len(RISK_CLASSES)),
        type_pos_weight=_type_pos_weight(train_samples),
        injury_weight=_weights(train_samples, "injury", len(INJURY_CLASSES)),
        quality_pos_weight=_quality_pos_weight(train_samples),
        label_smoothing=config["loss"].get("label_smoothing", 0.05),
        focal_gamma=config["loss"].get("focal_gamma", 0.0),
        task_weights=LOSS_WEIGHTS,
    ).to(device)
    lora_params = [p for p in model.vision.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": optim_cfg["lr_lora"]},
        {"params": model.head.parameters(), "lr": optim_cfg["lr_head"]},
    ], weight_decay=optim_cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda" and optim_cfg.get("amp", True))
    best_score, patience_used, global_step = -1.0, 0, 0
    output_dir = Path(config.get("output_dir", "checkpoints")) / "best"

    for epoch in range(int(optim_cfg["epochs"])):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = move_batch_to_device(batch, device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda" and optim_cfg.get("amp", True)):
                embeds = model.forward_vision(
                    batch["pixel_values"], batch["pixel_attention_mask"], batch["spatial_shapes"]
                )
                loss = loss_fn(model(embeds), batch) / int(optim_cfg.get("grad_accum", 1))
            scaler.scale(loss).backward()
            if batch_index % int(optim_cfg.get("grad_accum", 1)) == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if smoke_steps and global_step >= smoke_steps:
                    break
        metrics = evaluate_loader(model, val_loader, device)
        score = metrics["risk"]["macro_f1"] or 0.0
        print(f"epoch={epoch + 1} val_risk_macro_f1={score:.4f}")
        if score > best_score:
            best_score, patience_used = score, 0
            model.save_trainable(str(output_dir))
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        else:
            patience_used += 1
        if smoke_steps and global_step >= smoke_steps:
            break
        if patience_used >= int(optim_cfg.get("patience", 5)):
            break


if __name__ == "__main__":
    main()
