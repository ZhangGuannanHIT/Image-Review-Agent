# -*- coding: utf-8 -*-
"""Train a binary injury head on frozen FG-CLIP2 embeddings.

The four-class injury labels are collapsed to blood/no-blood. The existing
LoRA vision tower stays frozen, so this is a low-risk experiment before any
architecture migration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from .binary_injury import BinaryInjuryHead, select_threshold
from .dataset import Sample, parse_label_studio
from .engine import ReviewEngine
from .hard_probe import binary_metrics
from .training_dataset import ReviewCollator, ReviewDataset, move_batch_to_device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def valid_binary_samples(samples: list[Sample]) -> list[Sample]:
    return [sample for sample in samples if sample.injury is not None]


def binary_labels(samples: list[Sample]) -> torch.Tensor:
    return torch.tensor(
        [int(sample.injury > 0) for sample in samples],
        dtype=torch.float32,
    )


def load_hard_negatives(path: str | Path | None) -> set[str]:
    if not path:
        return set()
    hard_negative_path = Path(path)
    if not hard_negative_path.is_file():
        return set()
    with hard_negative_path.open("r", encoding="utf-8-sig", newline="") as file:
        return {str(row["file"]).strip() for row in csv.DictReader(file)}


def sample_file(sample: Sample) -> str:
    return str((sample.raw or {}).get("file_upload", ""))


def extract_embeddings(
    engine: ReviewEngine,
    samples: list[Sample],
    *,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    loader = DataLoader(
        ReviewDataset(samples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ReviewCollator(engine.processor),
    )
    embeddings: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    files: list[str] = []
    engine.model.eval()
    with torch.inference_mode():
        for index, batch in enumerate(loader, start=1):
            batch = move_batch_to_device(batch, engine.device)
            outputs = engine.model.forward_vision(
                batch["pixel_values"],
                batch["pixel_attention_mask"],
                batch["spatial_shapes"],
            )
            embeddings.append(outputs.float().cpu())
            labels.append(batch["injury_label"].gt(0).float().cpu())
            files.extend(
                sample_file(samples[index])
                for index in batch["sample_indices"].tolist()
            )
            if index % 50 == 0:
                print(f"embedding: {index * batch_size}/{len(samples)}", flush=True)
    return torch.cat(embeddings), torch.cat(labels), files


def cached_embeddings(
    engine: ReviewEngine,
    samples: list[Sample],
    *,
    split: str,
    cache_dir: str | Path,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    cache_path = Path(cache_dir) / f"{split}.pt"
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if len(payload["files"]) == len(samples):
            print(f"embedding-cache: {cache_path}", flush=True)
            return payload["embeddings"], payload["labels"], payload["files"]
    embeddings, labels, files = extract_embeddings(
        engine, samples, batch_size=batch_size,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"embeddings": embeddings, "labels": labels, "files": files},
        cache_path,
    )
    return embeddings, labels, files


def checkpoint_cache_tag(checkpoint: str | Path) -> str:
    digest = hashlib.sha256()
    root = Path(checkpoint)
    for name in ("head.pt", "adapter_model.safetensors", "labels.json"):
        path = root / name
        if path.is_file():
            digest.update(name.encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def fit_head(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    device: str,
    hard_negatives: set[str],
    train_files: list[str],
    seed: int,
) -> BinaryInjuryHead:
    generator = torch.Generator().manual_seed(seed)
    positives = float(train_y.sum())
    negatives = float(len(train_y) - positives)
    pos_weight = torch.tensor([max(1.0, negatives / max(positives, 1.0))], device=device)
    head = BinaryInjuryHead(train_x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=1e-3)
    hard_weight = torch.tensor(
        [2.0 if file in hard_negatives else 1.0 for file in train_files],
        dtype=torch.float32,
        device=device,
    )
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(len(train_x), generator=generator)
        head.train()
        total_loss = 0.0
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start:start + batch_size]
            logits = head(train_x[indices].to(device))
            loss_each = nn.functional.binary_cross_entropy_with_logits(
                logits,
                train_y[indices].to(device),
                pos_weight=pos_weight,
                reduction="none",
            )
            loss = (
                loss_each * hard_weight.index_select(0, indices.to(device))
            ).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        print(f"binary-epoch={epoch} loss={total_loss / len(train_x):.5f}", flush=True)
    head.eval()
    return head


def head_scores(head: BinaryInjuryHead, embeddings: torch.Tensor, device: str) -> list[float]:
    with torch.inference_mode():
        return torch.sigmoid(head(embeddings.to(device))).cpu().tolist()


def metrics_with_threshold(
    labels: torch.Tensor, scores: list[float], threshold: float,
) -> dict:
    predicted = [score >= threshold for score in scores]
    y = labels.bool().tolist()
    tp = sum(expected and actual for expected, actual in zip(y, predicted))
    fp = sum((not expected) and actual for expected, actual in zip(y, predicted))
    tn = sum((not expected) and (not actual) for expected, actual in zip(y, predicted))
    fn = sum(expected and (not actual) for expected, actual in zip(y, predicted))
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "false_positive_rate": fp / max(fp + tn, 1),
        "precision": tp / max(tp + fp, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="训练冻结 embedding 的 binary injury head")
    parser.add_argument("--config", default="config/train.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best")
    parser.add_argument("--hard-negatives", default="data/train_hard_negatives.csv")
    parser.add_argument("--output-dir", default="checkpoints/injury_binary")
    parser.add_argument("--eval-out", default="eval_results/binary_injury_metrics.json")
    parser.add_argument("--fpr-budget", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cache-dir", default="data/cache/binary_injury")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    data = config["data"]
    samples, _ = parse_label_studio(
        data["json_path"], data["image_root"], report=False,
    )
    split = json.loads(Path(data["split_path"]).read_text(encoding="utf-8"))
    train = valid_binary_samples([samples[index] for index in split["train"]])
    val = valid_binary_samples([samples[index] for index in split["val"]])
    test = valid_binary_samples([samples[index] for index in split["test"]])

    engine = ReviewEngine(
        config["model_path"],
        ckpt_dir=args.checkpoint,
        device=config.get("device", "cuda"),
        lora=bool(config.get("lora", True)),
        strict_load=True,
    )
    cache_dir = Path(args.cache_dir) / checkpoint_cache_tag(args.checkpoint)
    train_x, train_y, train_files = cached_embeddings(
        engine, train, split="train", cache_dir=cache_dir,
        batch_size=args.batch_size,
    )
    val_x, val_y, _ = cached_embeddings(
        engine, val, split="val", cache_dir=cache_dir,
        batch_size=args.batch_size,
    )
    test_x, test_y, _ = cached_embeddings(
        engine, test, split="test", cache_dir=cache_dir,
        batch_size=args.batch_size,
    )
    hard_negatives = load_hard_negatives(args.hard_negatives)
    head = fit_head(
        train_x,
        train_y,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=128,
        device=engine.device,
        hard_negatives=hard_negatives,
        train_files=train_files,
        seed=seed,
    )
    val_scores = head_scores(head, val_x, engine.device)
    operating_point = select_threshold(
        val_y.bool().tolist(), val_scores, args.fpr_budget,
    )
    test_scores = head_scores(head, test_x, engine.device)
    test_metrics = metrics_with_threshold(
        test_y, test_scores, operating_point["threshold"],
    )
    report = {
        "checkpoint": args.checkpoint,
        "embedding_dim": int(train_x.shape[1]),
        "counts": {
            "train": len(train), "val": len(val), "test": len(test),
            "train_positive": int(train_y.sum()),
            "train_hard_negatives": len(hard_negatives),
        },
        "validation_operating_point": operating_point,
        "test": test_metrics,
        "test_ranking": binary_metrics(test_y.bool().tolist(), test_scores),
        "val_rows": [
            {"label": int(label), "score": score}
            for label, score in zip(val_y.tolist(), val_scores)
        ],
        "test_rows": [
            {"label": int(label), "score": score}
            for label, score in zip(test_y.tolist(), test_scores)
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
        "validation_operating_point": operating_point,
        "test": test_metrics,
        "test_auroc": report["test_ranking"]["auroc"],
        "test_auprc": report["test_ranking"]["auprc"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
