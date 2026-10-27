# -*- coding: utf-8 -*-
"""Train a frozen-embedding anime style router with real-image negatives."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
import yaml
from PIL import Image, ImageOps

from src.anime_style_router import AnimeStyleRouter
from src.engine import ReviewEngine
from src.hard_probe import load_manifest


ROOT = Path(__file__).resolve().parents[1]


def _split(path: str) -> str:
    bucket = int(hashlib.sha256(path.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "val"
    return "test"


def _quality_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def collect_items(
    *,
    anime_root: Path,
    anime_annotations: Path,
    anime_manifest: Path,
    solid_manifest: Path,
    solid_root: Path,
    quality_manifest: Path,
    quality_root: Path,
) -> list[dict]:
    with anime_annotations.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        anime_rows = list(csv.DictReader(file))
    with anime_manifest.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        splits = {
            row["relative_path"]: row["split"] for row in csv.DictReader(file)
        }
    items = [
        {
            "path": anime_root / row["relative_path"],
            "label": 1,
            "split": {
                "dev": "train",
                "validation": "val",
                "test": "test",
            }.get(splits.get(row["relative_path"], "dev"), "train"),
            "source": "anime",
        }
        for row in anime_rows
    ]
    solid = load_manifest(
        solid_manifest,
        solid_root,
    )
    for row in solid:
        if row.domain != "real":
            continue
        items.append({
            "path": Path(row.path),
            "label": 0,
            "split": _split(f"real-solid/{row.file}"),
            "source": "real_solid",
        })
    for row in _quality_rows(quality_manifest):
        if row["origin"] != "real":
            continue
        items.append({
            "path": quality_root / row["file"],
            "label": 0,
            "split": _split(f"real-quality/{row['file']}"),
            "source": "real_quality",
        })
    return items


def extract_embeddings(engine, items: list[dict]) -> torch.Tensor:
    vectors = []
    for index, item in enumerate(items, start=1):
        with Image.open(item["path"]) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            inputs = engine.prepare_image(image)
            with torch.inference_mode():
                embedding = engine.model.forward_vision(
                    inputs["pixel_values"],
                    inputs["pixel_attention_mask"],
                    inputs["spatial_shapes"],
                )
        vectors.append(embedding.float().cpu())
        if index % 20 == 0:
            print(f"embeddings: {index}/{len(items)}", flush=True)
    return torch.cat(vectors, dim=0)


def metrics(labels, scores, threshold: float) -> dict:
    predicted = [score >= threshold for score in scores]
    tp = sum(pred and label for pred, label in zip(predicted, labels))
    tn = sum(not pred and not label for pred, label in zip(predicted, labels))
    fp = sum(pred and not label for pred, label in zip(predicted, labels))
    fn = sum(not pred and label for pred, label in zip(predicted, labels))
    return {
        "n": len(labels),
        "threshold": threshold,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "anime_recall": round(tp / max(tp + fn, 1), 6),
        "real_false_positive_rate": round(fp / max(fp + tn, 1), 6),
    }


def choose_threshold(labels, scores, max_real_fpr: float = 0.05) -> float:
    best = None
    for step in range(1, 100):
        threshold = step / 100
        item = metrics(labels, scores, threshold)
        if item["real_false_positive_rate"] > max_real_fpr:
            continue
        if best is None or item["anime_recall"] > best["anime_recall"]:
            best = item
    return float(best["threshold"]) if best else 0.5


def main() -> None:
    parser = argparse.ArgumentParser(description="训练动漫风格路由头")
    parser.add_argument("--config", default="config/inference.yaml")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument(
        "--output-dir", default="checkpoints/anime_style_router"
    )
    parser.add_argument(
        "--eval-out", default="eval_results/anime_style_router.json"
    )
    parser.add_argument(
        "--cache-out",
        default="data/cache/anime_style_router/embeddings.pt",
    )
    parser.add_argument("--anime-root", default="/mnt/d/动漫图片")
    parser.add_argument(
        "--anime-annotations",
        default="data/anime_safe/human_annotations_2026-10-27.csv",
    )
    parser.add_argument(
        "--anime-manifest", default="data/anime_safe/manifest.csv"
    )
    parser.add_argument(
        "--solid-manifest", default="data/hard_probe_solid/manifest.csv"
    )
    parser.add_argument("--solid-root", default="solid")
    parser.add_argument(
        "--quality-manifest", default="data/quality_probe/final.csv"
    )
    parser.add_argument("--quality-root", default="/mnt/d/quality")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    engine = ReviewEngine(
        model_path=config["model_path"],
        ckpt_dir=config["ckpt_dir"],
        device=config.get("device", "cuda"),
        lora=bool(config.get("lora", True)),
        strict_load=True,
        type_thresholds=config.get("type_thresholds"),
        quality_thresholds=config.get("quality_thresholds"),
        binary_injury_dir=config.get("binary_injury_dir", ""),
        quality_fusion_dir=config.get("quality_fusion_dir", ""),
    )
    items = collect_items(
        anime_root=Path(args.anime_root),
        anime_annotations=Path(args.anime_annotations),
        anime_manifest=Path(args.anime_manifest),
        solid_manifest=Path(args.solid_manifest),
        solid_root=Path(args.solid_root),
        quality_manifest=Path(args.quality_manifest),
        quality_root=Path(args.quality_root),
    )
    embeddings = extract_embeddings(engine, items)
    cache_path = Path(args.cache_out)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "embeddings": embeddings.half(),
        "labels": torch.tensor([item["label"] for item in items], dtype=torch.int64),
        "splits": [item["split"] for item in items],
        "paths": [str(item["path"]) for item in items],
        "sources": [item["source"] for item in items],
    }, cache_path)
    labels = torch.tensor([item["label"] for item in items], dtype=torch.float32)
    splits = [item["split"] for item in items]
    device = engine.device
    embeddings = embeddings.to(device)
    labels = labels.to(device)
    per_split = {
        name: torch.tensor([split == name for split in splits], device=device)
        for name in ("train", "val", "test")
    }
    train_idx = per_split["train"]
    val_idx = per_split["val"]
    test_idx = per_split["test"]
    if not train_idx.any() or not val_idx.any() or not test_idx.any():
        raise RuntimeError("train/val/test 至少各需要一张图片")

    torch.manual_seed(20261027)
    router = AnimeStyleRouter(hidden=128).to(device)
    optimizer = torch.optim.AdamW(
        router.parameters(), lr=1e-3, weight_decay=1e-3
    )
    positives = float((labels[train_idx] == 1).sum())
    negatives = float((labels[train_idx] == 0).sum())
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / max(positives, 1), device=device)
    )
    best_state = None
    best_val = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        router.train()
        optimizer.zero_grad(set_to_none=True)
        logits = router(embeddings[train_idx])
        loss = criterion(logits, labels[train_idx])
        loss.backward()
        optimizer.step()
        router.eval()
        with torch.inference_mode():
            val_loss = criterion(
                router(embeddings[val_idx]), labels[val_idx]
            ).item()
        history.append({"epoch": epoch, "train_loss": loss.item(), "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in router.state_dict().items()
            }
    router.load_state_dict(best_state or router.state_dict())
    router.eval()
    with torch.inference_mode():
        scores = torch.sigmoid(router(embeddings)).cpu().tolist()
    labels_cpu = labels.cpu().int().tolist()
    val_positions = val_idx.nonzero().flatten().cpu().tolist()
    threshold = choose_threshold(
        [labels_cpu[index] for index in val_positions],
        [scores[index] for index in val_positions],
    )
    result = {
        "items": len(items),
        "source_counts": dict(Counter(item["source"] for item in items)),
        "split_counts": dict(Counter(splits)),
        "threshold": threshold,
        "metrics": {},
    }
    result["threshold_metrics"] = {
        str(threshold): {},
        "0.5": {},
        "0.9": {},
        "0.99": {},
    }
    for name in ("train", "val", "test"):
        positions = per_split[name].nonzero().flatten().cpu().tolist()
        result["metrics"][name] = metrics(
            [labels_cpu[index] for index in positions],
            [scores[index] for index in positions],
            threshold,
        )
        for candidate in (threshold, 0.5, 0.9, 0.99):
            result["threshold_metrics"][str(candidate)][name] = metrics(
                [labels_cpu[index] for index in positions],
                [scores[index] for index in positions],
                candidate,
            )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(router.state_dict(), output_dir / "router.pt")
    metadata = {
        "embedding_dim": router.network[1].in_features,
        "hidden_dim": router.network[1].out_features,
        "threshold": threshold,
        "metrics": result["metrics"],
        "source_counts": result["source_counts"],
        "split_counts": result["split_counts"],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result["history_tail"] = history[-10:]
    eval_out = Path(args.eval_out)
    eval_out.parent.mkdir(parents=True, exist_ok=True)
    eval_out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
