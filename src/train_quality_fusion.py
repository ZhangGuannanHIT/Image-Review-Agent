# -*- coding: utf-8 -*-
"""Train a quality fusion head without changing the FG-CLIP2 backbone."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from torch import nn

from .dataset import Sample, parse_label_studio
from .engine import ReviewEngine
from .labels import QUALITY_CLASSES
from .metrics import multilabel_metrics
from .quality_features import extract_quality_features
from .quality_fusion import QualityFusionHead
from .train_binary_injury import (
    cached_embeddings, checkpoint_cache_tag, sample_file, seed_everything,
)


def valid_quality_samples(samples: list[Sample]) -> list[Sample]:
    return [sample for sample in samples if sample.quality_valid]


def augment_image(image: Image.Image, kind: str, rng: random.Random) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    if kind == "mild_blur":
        result = cv2.GaussianBlur(rgb, (7, 7), 2.2)
    elif kind == "severe_blur":
        scale = 0.22
        small = cv2.resize(
            rgb, (max(16, int(width * scale)), max(16, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        restored = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
        result = cv2.GaussianBlur(restored, (17, 17), 5.5)
    elif kind in {"black_bar", "mosaic"}:
        result = rgb.copy()
        region_width = rng.randint(max(32, width // 5), max(40, width // 2))
        region_height = rng.randint(max(24, height // 10), max(32, height // 3))
        x = rng.randint(0, max(0, width - region_width))
        y = rng.randint(0, max(0, height - region_height))
        region = result[y:y + region_height, x:x + region_width]
        if kind == "black_bar":
            result[y:y + region_height, x:x + region_width] = 0
        else:
            block = max(8, rng.choice([12, 16, 24, 32]))
            small = cv2.resize(
                region,
                (max(1, region_width // block), max(1, region_height // block)),
                interpolation=cv2.INTER_AREA,
            )
            mosaic = cv2.resize(
                small, (region_width, region_height),
                interpolation=cv2.INTER_NEAREST,
            )
            result[y:y + region_height, x:x + region_width] = mosaic
    else:
        raise ValueError(f"未知增强: {kind}")
    return Image.fromarray(result, mode="RGB")


def extract_feature_cache(
    samples: list[Sample],
    path: str | Path,
) -> np.ndarray:
    cache = Path(path)
    if cache.is_file():
        return torch.load(cache, map_location="cpu", weights_only=False)["features"].numpy()
    features = []
    for index, sample in enumerate(samples, start=1):
        with Image.open(sample.image_path) as opened:
            features.append(extract_quality_features(opened.convert("RGB")))
        if index % 200 == 0:
            print(f"quality-features: {index}/{len(samples)}", flush=True)
    array = np.stack(features)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": torch.from_numpy(array)}, cache)
    return array


def synthetic_features(
    clear_samples: list[Sample],
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    for sample in clear_samples:
        with Image.open(sample.image_path) as opened:
            image = opened.convert("RGB")
        for kind in ("mild_blur", "severe_blur"):
            features.append(extract_quality_features(augment_image(image, kind, rng)))
            target = np.zeros(len(QUALITY_CLASSES), dtype=np.float32)
            target[QUALITY_CLASSES.index("轻微模糊" if kind == "mild_blur" else "严重模糊")] = 1
            labels.append(target)
        for kind in ("black_bar", "mosaic"):
            features.append(extract_quality_features(augment_image(image, kind, rng)))
            target = np.zeros(len(QUALITY_CLASSES), dtype=np.float32)
            target[QUALITY_CLASSES.index("故意遮挡（马赛克/黑条/贴纸）")] = 1
            labels.append(target)
    return np.stack(features), np.stack(labels)


def calibrate_thresholds(
    labels: np.ndarray,
    scores: np.ndarray,
) -> tuple[dict[str, float], dict]:
    thresholds = {}
    metrics = {}
    for index, label in enumerate(QUALITY_CLASSES):
        best = (0.5, -1.0)
        for threshold in np.linspace(0.05, 0.95, 181):
            predicted = (scores[:, index] >= threshold).astype(int)
            tp = int(np.logical_and(predicted, labels[:, index] == 1).sum())
            fp = int(np.logical_and(predicted, labels[:, index] == 0).sum())
            fn = int(np.logical_and(predicted == 0, labels[:, index] == 1).sum())
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-12)
            if f1 > best[1]:
                best = (float(threshold), f1)
        thresholds[label] = best[0]
        metrics[label] = {"threshold": best[0], "f1": best[1]}
    return thresholds, metrics


def predict(
    head: QualityFusionHead,
    embeddings: torch.Tensor,
    features: torch.Tensor,
    threshold_values: dict[str, float],
    device: str,
) -> np.ndarray:
    with torch.inference_mode():
        scores = torch.sigmoid(head(embeddings.to(device), features.to(device))).cpu().numpy()
    thresholds = np.asarray([threshold_values[label] for label in QUALITY_CLASSES])
    return (scores >= thresholds).astype(int), scores


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 FG-CLIP2 + 低层质量特征融合头")
    parser.add_argument("--config", default="config/train.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best")
    parser.add_argument("--output-dir", default="checkpoints/quality_fusion")
    parser.add_argument("--eval-out", default="eval_results/quality_fusion_metrics.json")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--cache-dir", default="data/cache/quality_fusion")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    rng = random.Random(seed)
    data = config["data"]
    all_samples, _ = parse_label_studio(data["json_path"], data["image_root"], report=False)
    split = json.loads(Path(data["split_path"]).read_text(encoding="utf-8"))
    train = valid_quality_samples([all_samples[index] for index in split["train"]])
    val = valid_quality_samples([all_samples[index] for index in split["val"]])
    test = valid_quality_samples([all_samples[index] for index in split["test"]])

    engine = ReviewEngine(
        config["model_path"],
        ckpt_dir=args.checkpoint,
        device=config.get("device", "cuda"),
        lora=bool(config.get("lora", True)),
        strict_load=True,
    )
    checkpoint_tag = checkpoint_cache_tag(args.checkpoint)
    cache_dir = Path(args.cache_dir) / checkpoint_tag
    train_x, _, _ = cached_embeddings(
        engine, train, split="train", cache_dir=cache_dir, batch_size=args.batch_size,
    )
    val_x, _, _ = cached_embeddings(
        engine, val, split="val", cache_dir=cache_dir, batch_size=args.batch_size,
    )
    test_x, _, _ = cached_embeddings(
        engine, test, split="test", cache_dir=cache_dir, batch_size=args.batch_size,
    )
    train_features = extract_feature_cache(
        train, Path(args.cache_dir) / "features_train.pt",
    )
    val_features = extract_feature_cache(
        val, Path(args.cache_dir) / "features_val.pt",
    )
    test_features = extract_feature_cache(
        test, Path(args.cache_dir) / "features_test.pt",
    )
    train_labels = np.asarray([sample.quality_set for sample in train], dtype=np.float32)
    val_labels = np.asarray([sample.quality_set for sample in val], dtype=np.float32)
    test_labels = np.asarray([sample.quality_set for sample in test], dtype=np.float32)

    clear_samples = [
        sample for sample in train if sample.quality_set == [1, 0, 0, 0, 0]
    ]
    extra_features, extra_labels = synthetic_features(clear_samples, rng)
    print(
        f"train={len(train)} synthetic={len(extra_features)} val={len(val)} test={len(test)}",
        flush=True,
    )
    train_features = np.concatenate([train_features, extra_features], axis=0)
    train_labels = np.concatenate([train_labels, extra_labels], axis=0)
    train_index_by_file = {
        sample_file(sample): index for index, sample in enumerate(train)
    }
    synthetic_embeddings = torch.stack([
        train_x[train_index_by_file[sample_file(sample)]]
        for sample in clear_samples
        for _ in range(4)
    ])
    train_x = torch.cat([train_x, synthetic_embeddings], dim=0)

    feature_mean = torch.from_numpy(train_features.mean(axis=0, keepdims=True))
    feature_std = torch.from_numpy(train_features.std(axis=0, keepdims=True))
    feature_std[feature_std < 1e-6] = 1.0
    train_features_t = (torch.from_numpy(train_features) - feature_mean) / feature_std
    val_features_t = (torch.from_numpy(val_features) - feature_mean) / feature_std
    test_features_t = (torch.from_numpy(test_features) - feature_mean) / feature_std
    train_labels_t = torch.from_numpy(train_labels)
    val_labels_t = torch.from_numpy(val_labels)

    positives = train_labels_t.sum(0).clamp(min=1)
    negatives = len(train_labels_t) - positives
    pos_weight = (negatives / positives).clamp(min=0.1, max=10.0)
    head = QualityFusionHead(train_x.shape[1], train_features.shape[1]).to(engine.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    generator = torch.Generator().manual_seed(seed)
    best_score = -1.0
    best_state = None
    best_thresholds = {}
    history = []
    for epoch in range(1, args.epochs + 1):
        permutation = torch.randperm(len(train_x), generator=generator)
        head.train()
        total_loss = 0.0
        for start in range(0, len(permutation), 128):
            indices = permutation[start:start + 128]
            logits = head(
                train_x[indices].to(engine.device),
                train_features_t[indices].to(engine.device),
            )
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits,
                train_labels_t[indices].to(engine.device),
                pos_weight=pos_weight.to(engine.device),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        with torch.inference_mode():
            val_scores = torch.sigmoid(head(
                val_x.to(engine.device), val_features_t.to(engine.device)
            )).cpu().numpy()
        thresholds, _ = calibrate_thresholds(val_labels, val_scores)
        val_pred, _ = predict(head, val_x, val_features_t, thresholds, engine.device)
        metrics = multilabel_metrics(
            val_labels.astype(int), val_pred,
            np.ones(len(val_pred), dtype=bool), QUALITY_CLASSES,
        )
        score = metrics["macro_f1"]
        history.append({
            "epoch": epoch,
            "loss": total_loss / len(train_x),
            "val_macro_f1": metrics["macro_f1"],
            "val_micro_f1": metrics["micro_f1"],
            "thresholds": thresholds,
        })
        print(
            f"quality-epoch={epoch} loss={history[-1]['loss']:.5f} "
            f"val_macro_f1={metrics['macro_f1']:.4f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_thresholds = thresholds
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in head.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("quality fusion 未产生 checkpoint")
    head.load_state_dict(best_state)
    head.eval()
    test_pred, test_scores = predict(
        head, test_x, test_features_t, best_thresholds, engine.device,
    )
    test_metrics = multilabel_metrics(
        test_labels.astype(int), test_pred,
        np.ones(len(test_pred), dtype=bool), QUALITY_CLASSES,
    )
    report = {
        "checkpoint": args.checkpoint,
        "embedding_dim": int(train_x.shape[1]),
        "feature_dim": int(train_features.shape[1]),
        "quality_classes": QUALITY_CLASSES,
        "thresholds": best_thresholds,
        "feature_mean": feature_mean.squeeze(0).tolist(),
        "feature_std": feature_std.squeeze(0).tolist(),
        "counts": {
            "train": len(train),
            "synthetic": len(extra_features),
            "val": len(val),
            "test": len(test),
        },
        "history": history,
        "test": test_metrics,
        "test_exact_match": float(np.all(test_labels.astype(int) == test_pred, axis=1).mean()),
        "test_rows": [
            {
                "label": test_labels[index].astype(int).tolist(),
                "predicted": test_pred[index].tolist(),
                "scores": test_scores[index].tolist(),
            }
            for index in range(len(test))
        ],
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), output_dir / "head.pt")
    (output_dir / "metadata.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    eval_out = Path(args.eval_out)
    eval_out.parent.mkdir(parents=True, exist_ok=True)
    eval_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "best_val_macro_f1": best_score,
        "thresholds": best_thresholds,
        "test_macro_f1": test_metrics["macro_f1"],
        "test_micro_f1": test_metrics["micro_f1"],
        "test_exact_match": report["test_exact_match"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
