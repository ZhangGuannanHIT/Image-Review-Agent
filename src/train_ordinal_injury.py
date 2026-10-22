# -*- coding: utf-8 -*-
"""Train a CORAL ordinal injury head on frozen FG-CLIP2 embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .dataset import Sample, parse_label_studio
from .engine import ReviewEngine
from .hard_probe import binary_metrics
from .labels import INJURY_CLASSES
from .ordinal_injury import (
    OrdinalInjuryHead, class_probabilities, cumulative_targets, severity_metrics,
)
from .train_binary_injury import (
    cached_embeddings, checkpoint_cache_tag, load_hard_negatives,
    sample_file, seed_everything, valid_binary_samples,
)


def class_weights(labels: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(labels.long(), minlength=len(INJURY_CLASSES)).float()
    weights = 1.0 / torch.sqrt(counts.clamp(min=1))
    return weights / weights.mean()


def fit_ordinal_head(
    train_x: torch.Tensor,
    train_labels: torch.Tensor,
    val_x: torch.Tensor,
    val_labels: torch.Tensor,
    *,
    train_files: list[str],
    hard_negatives: set[str],
    device: str,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> tuple[OrdinalInjuryHead, dict, list[list[float]]]:
    generator = torch.Generator().manual_seed(seed)
    head = OrdinalInjuryHead(train_x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=1e-3)
    targets = cumulative_targets(train_labels.to(device))
    positives = targets.sum(0).clamp(min=1)
    negatives = len(train_labels) - positives
    pos_weight = (negatives / positives).clamp(min=0.1, max=10.0)
    weights_by_class = class_weights(train_labels)
    sample_weight = weights_by_class[train_labels.long()].clone()
    for index, file in enumerate(train_files):
        if train_labels[index].item() == 0 and file in hard_negatives:
            sample_weight[index] *= 2.0
    sample_weight = sample_weight.to(device)

    best_score = -1.0
    best_state = None
    best_epoch = 0
    history = []
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(len(train_x), generator=generator)
        head.train()
        total_loss = 0.0
        for start in range(0, len(permutation), 128):
            indices = permutation[start:start + 128]
            logits = head(train_x[indices].to(device))
            loss_each = nn.functional.binary_cross_entropy_with_logits(
                logits,
                targets[indices],
                pos_weight=pos_weight,
                reduction="none",
            ).mean(dim=-1)
            loss = (loss_each * sample_weight[indices.to(device)]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        with torch.inference_mode():
            val_logits = head(val_x.to(device))
            val_pred = class_probabilities(val_logits).argmax(-1).cpu().tolist()
        metrics = severity_metrics(val_labels.tolist(), val_pred)
        score = metrics["macro_f1"] + 0.25 * metrics["qwk"] + 0.25 * metrics["severe_recall"]
        history.append({
            "epoch": epoch,
            "loss": total_loss / max(len(train_x), 1),
            "val_macro_f1": metrics["macro_f1"],
            "val_qwk": metrics["qwk"],
            "val_severe_recall": metrics["severe_recall"],
        })
        print(
            f"ordinal-epoch={epoch} loss={history[-1]['loss']:.5f} "
            f"val_macro_f1={metrics['macro_f1']:.4f} qwk={metrics['qwk']:.4f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in head.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("ordinal head 未产生有效 checkpoint")
    head.load_state_dict(best_state)
    head.eval()
    return head, {
        "best_epoch": best_epoch,
        "selection_score": best_score,
    }, history


def predict(head: OrdinalInjuryHead, embeddings: torch.Tensor, device: str) -> list[dict]:
    with torch.inference_mode():
        probabilities = class_probabilities(head(embeddings.to(device))).cpu()
    output = []
    for row in probabilities:
        index = int(row.argmax().item())
        output.append({
            "index": index,
            "label": INJURY_CLASSES[index],
            "confidence": float(row[index].item()),
            "probabilities": {
                label: float(value)
                for label, value in zip(INJURY_CLASSES, row.tolist())
            },
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 CORAL ordinal injury head")
    parser.add_argument("--config", default="config/train.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best")
    parser.add_argument("--hard-negatives", default="data/train_hard_negatives.csv")
    parser.add_argument("--output-dir", default="checkpoints/injury_ordinal")
    parser.add_argument("--eval-out", default="eval_results/ordinal_injury_metrics.json")
    parser.add_argument("--cache-dir", default="data/cache/binary_injury")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    import yaml

    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    data = config["data"]
    samples, _ = parse_label_studio(data["json_path"], data["image_root"], report=False)
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
    train_x, _, train_files = cached_embeddings(
        engine, train, split="train", cache_dir=cache_dir, batch_size=args.batch_size,
    )
    val_x, _, _ = cached_embeddings(
        engine, val, split="val", cache_dir=cache_dir, batch_size=args.batch_size,
    )
    test_x, _, _ = cached_embeddings(
        engine, test, split="test", cache_dir=cache_dir, batch_size=args.batch_size,
    )
    train_labels = torch.tensor([sample.injury for sample in train], dtype=torch.long)
    val_labels = torch.tensor([sample.injury for sample in val], dtype=torch.long)
    test_labels = torch.tensor([sample.injury for sample in test], dtype=torch.long)
    head, selection, history = fit_ordinal_head(
        train_x,
        train_labels,
        val_x,
        val_labels,
        train_files=train_files,
        hard_negatives=load_hard_negatives(args.hard_negatives),
        device=engine.device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=seed,
    )
    val_predictions = predict(head, val_x, engine.device)
    test_predictions = predict(head, test_x, engine.device)
    val_metrics = severity_metrics(val_labels.tolist(), [row["index"] for row in val_predictions])
    test_metrics = severity_metrics(test_labels.tolist(), [row["index"] for row in test_predictions])
    binary_test = binary_metrics(
        (label > 0 for label in test_labels.tolist()),
        (1 - row["probabilities"][INJURY_CLASSES[0]] for row in test_predictions),
    )
    report = {
        "checkpoint": args.checkpoint,
        "embedding_dim": int(train_x.shape[1]),
        "counts": {
            "train": len(train), "val": len(val), "test": len(test),
        },
        "selection": selection,
        "history": history,
        "validation": val_metrics,
        "test": test_metrics,
        "test_binary_ranking": binary_test,
        "test_rows": [
            {
                "label": int(label),
                **prediction,
            }
            for label, prediction in zip(test_labels.tolist(), test_predictions)
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
        "selection": selection,
        "validation": {
            "macro_f1": val_metrics["macro_f1"],
            "qwk": val_metrics["qwk"],
        },
        "test": {
            "macro_f1": test_metrics["macro_f1"],
            "qwk": test_metrics["qwk"],
            "severe_recall": test_metrics["severe_recall"],
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
