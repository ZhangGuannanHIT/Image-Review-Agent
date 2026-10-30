# -*- coding: utf-8 -*-
"""Evaluate a checkpoint on a frozen hard-probe manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from PIL import Image

from .binary_injury import BinaryInjuryHead, score_image
from .engine import ReviewEngine
from .hard_probe import (
    evaluate_items, grouped_metrics, load_manifest, metrics_at_threshold,
)
from .labels import INJURY_CLASSES
from .ordinal_injury import OrdinalInjuryHead, score_image as ordinal_score_image
from .ordinal_injury import severity_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="冻结硬探针集评估")
    parser.add_argument("--manifest", default="data/hard_probe/manifest.csv")
    parser.add_argument("--image-root", default="pic/pic")
    parser.add_argument("--config", default="config/inference.yaml")
    parser.add_argument("--binary-head", default="", help="可选 binary injury checkpoint 目录")
    parser.add_argument("--ordinal-head", default="", help="可选 ordinal injury checkpoint 目录")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.binary_head and args.ordinal_head:
        raise ValueError("--binary-head 与 --ordinal-head 不能同时使用")

    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    manifest = Path(args.manifest)
    image_root = Path(args.image_root)
    items = load_manifest(manifest, image_root)
    engine = ReviewEngine(
        model_path=config["model_path"],
        ckpt_dir=config["ckpt_dir"],
        device=config.get("device", "cuda"),
        lora=bool(config.get("lora", True)),
        strict_load=bool(config.get("strict_load", True)),
        type_thresholds=config.get("type_thresholds"),
        quality_thresholds=config.get("quality_thresholds"),
    )
    if args.ordinal_head:
        head_dir = Path(args.ordinal_head)
        metadata = json.loads(
            (head_dir / "metadata.json").read_text(encoding="utf-8")
        )
        head = OrdinalInjuryHead(int(metadata["embedding_dim"]))
        head.load_state_dict(
            torch.load(head_dir / "head.pt", map_location=engine.device, weights_only=True)
        )
        head.to(engine.device)
        rows = []
        for index, item in enumerate(items, start=1):
            with Image.open(item.path) as opened:
                image = opened.convert("RGB")
            prediction = ordinal_score_image(engine, head, image)
            probabilities = prediction["probabilities"]
            rows.append(
                {
                    "file": item.file,
                    "folder": item.folder,
                    "domain": item.domain,
                    "red_level": item.red_level,
                    "has_blood": item.has_blood,
                    "severity_truth": item.severity_truth,
                    "p_blood": 1.0 - probabilities[INJURY_CLASSES[0]],
                    "severity_label": prediction["label"],
                    "severity_probabilities": probabilities,
                    "risk_label": "",
                    "injury_label": prediction["label"],
                    "type_labels": [],
                }
            )
            if index % 10 == 0:
                print(f"ordinal-hard-probe: {index}/{len(items)}", flush=True)
        score_key = "p_blood"
        head_metadata = {
            "path": str(head_dir),
            "selection": metadata["selection"],
            "test": metadata["test"],
        }
    elif args.binary_head:
        head_dir = Path(args.binary_head)
        metadata = json.loads(
            (head_dir / "metadata.json").read_text(encoding="utf-8")
        )
        threshold = float(metadata["validation_operating_point"]["threshold"])
        head = BinaryInjuryHead(int(metadata["embedding_dim"]))
        head.load_state_dict(
            torch.load(head_dir / "head.pt", map_location=engine.device, weights_only=True)
        )
        head.to(engine.device)
        rows = []
        for index, item in enumerate(items, start=1):
            with Image.open(item.path) as opened:
                image = opened.convert("RGB")
            score = score_image(engine, head, image)
            rows.append(
                {
                    "file": item.file,
                    "folder": item.folder,
                    "domain": item.domain,
                    "red_level": item.red_level,
                    "has_blood": item.has_blood,
                    "severity_truth": item.severity_truth,
                    "p_blood": score,
                    "predicted_blood": score >= threshold,
                    "risk_label": "",
                    "injury_label": "binary",
                    "type_labels": [],
                }
            )
            if index % 10 == 0:
                print(f"binary-hard-probe: {index}/{len(items)}", flush=True)
        score_key = "p_blood"
        head_metadata = {
            "path": str(head_dir),
            "threshold": threshold,
            "validation_operating_point": metadata["validation_operating_point"],
            "test": metadata["test"],
        }
    else:
        rows = evaluate_items(items, engine)
        score_key = "p_blood"
        head_metadata = None
    report = {
        "manifest": str(manifest),
        "image_root": str(image_root),
        "checkpoint": config["ckpt_dir"],
        "n": len(rows),
        "binary_head": head_metadata,
        "metrics": grouped_metrics(rows, score_key=score_key),
        "rows": rows,
    }
    if args.binary_head:
        report["selected_threshold_metrics"] = metrics_at_threshold(
            (row["has_blood"] for row in rows),
            (row["p_blood"] for row in rows),
            report["binary_head"]["threshold"],
        )
    if args.ordinal_head:
        expected = []
        predicted = []
        for row in rows:
            truth = row["severity_truth"] or (
                INJURY_CLASSES[0] if not row["has_blood"] else ""
            )
            if truth:
                expected.append(INJURY_CLASSES.index(truth))
                predicted.append(INJURY_CLASSES.index(row["severity_label"]))
        report["severity_metrics"] = severity_metrics(expected, predicted)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"binary": report["metrics"]["all"]}
    if "severity_metrics" in report:
        summary["severity"] = report["severity_metrics"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
