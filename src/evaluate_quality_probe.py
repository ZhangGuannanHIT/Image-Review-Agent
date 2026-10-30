# -*- coding: utf-8 -*-
"""Evaluate the multi-label quality head on the frozen quality probe."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from backend.policy import DecisionPolicy, decide
from .engine import ReviewEngine
from .labels import QUALITY_CLASSES
from .metrics import multilabel_metrics
from .quality_features import extract_quality_features
from .quality_fusion import QualityFusionHead
from .split_quality_probe import QUALITY_TAG_MAP


def load_quality_manifest(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("quality manifest 不能为空")
    for row in rows:
        tags = json.loads(row["quality_tags"])
        unknown = set(tags) - set(QUALITY_TAG_MAP)
        if unknown:
            raise ValueError(f"{row['file']} 存在未知 tag: {sorted(unknown)}")
        row["quality_tags"] = tags
    return rows


def tag_vector(tags: list[str]) -> list[int]:
    mapped = {QUALITY_TAG_MAP[tag] for tag in tags}
    return [int(label in mapped) for label in QUALITY_CLASSES]


def fusion_prediction(engine, fusion, image: Image.Image) -> tuple[list[str], dict[str, float]]:
    inputs = engine.prepare_image(image)
    engine.model.eval()
    fusion["head"].eval()
    with torch.inference_mode():
        embeddings = engine.model.forward_vision(
            inputs["pixel_values"],
            inputs["pixel_attention_mask"],
            inputs["spatial_shapes"],
        )
        feature = torch.from_numpy(extract_quality_features(image)).unsqueeze(0)
        feature = (feature - fusion["mean"]) / fusion["std"]
        scores = torch.sigmoid(
            fusion["head"](embeddings, feature.to(engine.device))
        )[0].cpu().tolist()
    labels = [
        label for label, score in zip(QUALITY_CLASSES, scores)
        if score >= fusion["thresholds"][label]
    ]
    return labels, dict(zip(QUALITY_CLASSES, scores))


def evaluate(
    rows: list[dict],
    engine,
    image_root: str | Path,
    fusion: dict | None = None,
    policy: DecisionPolicy | None = None,
) -> tuple[list[dict], dict]:
    output_rows = []
    for index, row in enumerate(rows, start=1):
        path = Path(image_root) / row["file"]
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            output = engine.predict(image)
            if fusion:
                quality_labels, quality_scores = fusion_prediction(engine, fusion, image)
            else:
                quality_labels = output.quality_labels
                quality_scores = output.quality["scores"]
        expected = tag_vector(row["quality_tags"])
        predicted = [int(label in quality_labels) for label in QUALITY_CLASSES]
        action = None
        uncertain = None
        reason = None
        if policy is not None:
            evidence = output.to_dict()
            evidence["quality"] = {
                "labels": quality_labels,
                "scores": quality_scores,
                "thresholds": (
                    fusion["thresholds"] if fusion else engine.quality_thresholds
                ),
            }
            decision = decide(evidence, policy)
            action = decision["action"]
            uncertain = decision["uncertain"]
            reason = decision["reason"]
        output_rows.append(
            {
                "file": row["file"],
                "category": row["category"],
                "origin": row["origin"],
                "label": row["label"],
                "style": row["style"],
                "color": row["color"],
                "expected_tags": row["quality_tags"],
                "expected": expected,
                "predicted_labels": quality_labels,
                "predicted": predicted,
                "scores": quality_scores,
                "risk": output.risk_label,
                "action": action,
                "uncertain": uncertain,
                "policy_reason": reason,
            }
        )
        if index % 10 == 0:
            print(f"quality-probe: {index}/{len(rows)}", flush=True)

    true = np.asarray([row["expected"] for row in output_rows], dtype=int)
    pred = np.asarray([row["predicted"] for row in output_rows], dtype=int)
    valid = np.ones(len(output_rows), dtype=bool)
    metrics = {
        "all": multilabel_metrics(true, pred, valid, QUALITY_CLASSES),
        "exact_match": float(np.all(true == pred, axis=1).mean()),
        "by_category": {},
        "by_origin": {},
        "by_label": {},
        "actions_by_category": {},
        "actions_by_label": {},
    }
    for key in ("category", "origin", "label"):
        groups = defaultdict(list)
        for index, row in enumerate(output_rows):
            groups[row[key]].append(index)
        metrics[f"by_{key}"] = {
            name: {
                **multilabel_metrics(
                    true[indices], pred[indices],
                    np.ones(len(indices), dtype=bool), QUALITY_CLASSES,
                ),
                "exact_match": float(np.all(
                    true[indices] == pred[indices], axis=1
                ).mean()),
            }
            for name, indices in groups.items()
        }
    if policy is not None:
        for key in ("category", "label"):
            groups = defaultdict(list)
            for row in output_rows:
                groups[row[key]].append(row["action"])
            metrics[f"actions_by_{key}"] = {
                name: dict(sorted(Counter(actions).items()))
                for name, actions in groups.items()
            }
    return output_rows, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="质量退化 frozen probe 评估")
    parser.add_argument("--manifest", default="data/quality_probe/dev.csv")
    parser.add_argument("--image-root", default="../quality")
    parser.add_argument("--config", default="config/inference.yaml")
    parser.add_argument("--backend-config", default="backend/config.yaml")
    parser.add_argument("--fusion-head", default="", help="可选 quality fusion checkpoint 目录")
    parser.add_argument("--fusion-thresholds", default="", help="可选逐类阈值 JSON")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    with open(args.backend_config, "r", encoding="utf-8") as file:
        backend_config = yaml.safe_load(file)
    engine = ReviewEngine(
        model_path=config["model_path"],
        ckpt_dir=config["ckpt_dir"],
        device=config.get("device", "cuda"),
        lora=bool(config.get("lora", True)),
        strict_load=bool(config.get("strict_load", True)),
        type_thresholds=config.get("type_thresholds"),
        quality_thresholds=config.get("quality_thresholds"),
        binary_injury_dir=config.get("binary_injury_dir", ""),
    )
    fusion = None
    if args.fusion_head:
        head_dir = Path(args.fusion_head)
        metadata = json.loads((head_dir / "metadata.json").read_text(encoding="utf-8"))
        head = QualityFusionHead(
            int(metadata["embedding_dim"]), int(metadata["feature_dim"]),
        )
        head.load_state_dict(
            torch.load(head_dir / "head.pt", map_location=engine.device, weights_only=True)
        )
        head.to(engine.device)
        fusion = {
            "head": head,
            "thresholds": metadata["thresholds"],
            "mean": torch.tensor(metadata["feature_mean"], dtype=torch.float32),
            "std": torch.tensor(metadata["feature_std"], dtype=torch.float32),
        }
        if args.fusion_thresholds:
            threshold_data = json.loads(
                Path(args.fusion_thresholds).read_text(encoding="utf-8")
            )
            fusion["thresholds"] = threshold_data["thresholds"]
    rows = load_quality_manifest(args.manifest)
    policy = DecisionPolicy(**(backend_config.get("decision_policy") or {}))
    output_rows, metrics = evaluate(rows, engine, args.image_root, fusion, policy)
    report = {
        "manifest": args.manifest,
        "image_root": args.image_root,
        "checkpoint": config["ckpt_dir"],
        "quality_thresholds": fusion["thresholds"] if fusion else engine.quality_thresholds,
        "fusion_head": args.fusion_head or None,
        "metrics": metrics,
        "rows": output_rows,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "all": metrics["all"],
        "exact_match": metrics["exact_match"],
        "by_category": metrics["by_category"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
