# -*- coding: utf-8 -*-
"""在验证集上标定 violence type / quality 的逐类阈值，并用标定结果评估测试集。

做法：
1. 用训练好的 checkpoint 在验证集上收集两类多标签头的概率；
2. 逐类扫描阈值 [0.05, 0.95]，取 F1 最高的阈值；正样本少于 min_pos 的类保持 0.5；
3. 用标定后的阈值重算验证集/测试集指标，写出 JSON 供人工复核。

产物只用于分析和后续推理配置，不改标签、不改 checkpoint。

用法（仓库根目录）：
    python -m src.calibrate_thresholds \
        --config config/train.yaml \
        --checkpoint checkpoints/best \
        --split val \
        --test-split test \
        --out eval_results/threshold_calibration.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from .dataset import parse_label_studio
from .engine import ReviewEngine
from .labels import QUALITY_CLASSES, TYPE_CLASSES
from .metrics import multilabel_metrics
from .training_dataset import ReviewCollator, ReviewDataset, move_batch_to_device


def collect_scores(engine: ReviewEngine, loader: DataLoader, device: str) -> dict[str, list]:
    """跑一遍数据，收集多标签任务的标签、概率与有效掩码。"""
    records: dict[str, list] = {
        "type_true": [], "type_scores": [], "type_valid": [],
        "quality_true": [], "quality_scores": [], "quality_valid": [],
    }
    engine.model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            embeds = engine.model.forward_vision(
                batch["pixel_values"], batch["pixel_attention_mask"], batch["spatial_shapes"]
            )
            logits = engine.model(embeds)
            records["type_true"].extend(batch["type_target"].cpu().int().tolist())
            records["type_scores"].extend(torch.sigmoid(logits["type"].float()).cpu().tolist())
            records["type_valid"].extend(batch["type_valid"].cpu().tolist())
            records["quality_true"].extend(batch["quality_target"].cpu().int().tolist())
            records["quality_scores"].extend(torch.sigmoid(logits["quality"].float()).cpu().tolist())
            records["quality_valid"].extend(batch["quality_valid"].cpu().tolist())
    return records


def _f1(tp: float, fp: float, fn: float) -> float:
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def sweep_class(y_true: list[int], scores: list[float],
                candidates: list[float]) -> tuple[float, float, list[dict]]:
    """返回该类别 F1 最高的阈值、F1 以及完整阈值曲线。"""
    best_threshold, best_f1 = 0.5, -1.0
    curve = []
    for threshold in candidates:
        tp = fp = fn = 0
        for truth, score in zip(y_true, scores):
            pred = 1 if score >= threshold else 0
            tp += int(truth == 1 and pred == 1)
            fp += int(truth == 0 and pred == 1)
            fn += int(truth == 1 and pred == 0)
        score_f1 = _f1(tp, fp, fn)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        curve.append({"threshold": threshold, "precision": round(precision, 4),
                      "recall": round(recall, 4), "f1": round(score_f1, 4)})
        if score_f1 > best_f1:
            best_threshold, best_f1 = threshold, score_f1
    return round(best_threshold, 2), round(best_f1, 4), curve


def calibrate(
    true_rows: list[list[int]], score_rows: list[list[float]], valid: list[bool],
    classes: list[str], candidates: list[float], min_pos: int,
) -> dict[str, Any]:
    """逐类标定；返回 applied 阈值、逐类 F1 与扫描细节。"""
    applied: dict[str, float] = {}
    details: dict[str, Any] = {}
    for index, label in enumerate(classes):
        y_true = [row[index] for row, ok in zip(true_rows, valid) if ok]
        probs = [row[index] for row, ok in zip(score_rows, valid) if ok]
        positives = sum(y_true)
        if positives < min_pos:
            applied[label] = 0.5
            details[label] = {
                "threshold": 0.5, "val_f1": None,
                "positives": positives, "note": f"正样本少于 {min_pos}，保持 0.5",
            }
            continue
        threshold, val_f1, curve = sweep_class(y_true, probs, candidates)
        applied[label] = threshold
        details[label] = {"threshold": threshold, "val_f1": val_f1,
                          "positives": positives, "curve": curve}
    return {"applied": applied, "per_class": details}


def apply_thresholds(
    true_rows: list[list[int]], score_rows: list[list[float]], valid: list[bool],
    thresholds: dict[str, float], classes: list[str],
) -> tuple[list[list[int]], list[list[int]]]:
    pred_rows = [
        [1 if score >= thresholds[label] else 0 for label, score in zip(classes, row)]
        for row in score_rows
    ]
    return true_rows, pred_rows


def metrics_for(
    true_rows, score_rows, valid, thresholds, classes, split_name,
) -> dict[str, Any]:
    true_rows, pred_rows = apply_thresholds(true_rows, score_rows, valid, thresholds, classes)
    result = multilabel_metrics(true_rows, pred_rows, valid, classes)
    result["split"] = split_name
    result["thresholds"] = thresholds
    return result


def _loader(engine: ReviewEngine, samples, batch_size: int) -> DataLoader:
    return DataLoader(
        ReviewDataset(samples), batch_size=batch_size, shuffle=False,
        collate_fn=ReviewCollator(engine.processor),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="标定多标签阈值并评估")
    parser.add_argument("--config", default="config/train.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--out", default="eval_results/threshold_calibration.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-pos", type=int, default=5,
                        help="验证集中正样本少于此数的类别不标定，保持 0.5")
    parser.add_argument("--override", action="append", default=[],
                        help="人工覆盖某个类别的阈值，格式 类别=阈值，可重复")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    data = config["data"]
    samples, _ = parse_label_studio(data["json_path"], data["image_root"], report=False)
    with open(data.get("split_path", "data/split.json"), "r", encoding="utf-8") as file:
        split = json.load(file)

    engine = ReviewEngine(
        config["model_path"], ckpt_dir=args.checkpoint,
        device=config.get("device", "cuda"), strict_load=True,
    )
    device = engine.device
    candidates = [round(0.05 + 0.05 * step, 2) for step in range(19)]  # 0.05 .. 0.95

    val_samples = [samples[index] for index in split[args.split]]
    val_records = collect_scores(engine, _loader(engine, val_samples, args.batch_size), device)
    type_cal = calibrate(
        val_records["type_true"], val_records["type_scores"], val_records["type_valid"],
        TYPE_CLASSES, candidates, args.min_pos,
    )
    quality_cal = calibrate(
        val_records["quality_true"], val_records["quality_scores"], val_records["quality_valid"],
        QUALITY_CLASSES, candidates, args.min_pos,
    )
    overrides = {}
    for item in args.override:
        label, _, value = item.partition("=")
        label, value = label.strip(), float(value)
        if label in type_cal["applied"]:
            type_cal["applied"][label] = value
        elif label in quality_cal["applied"]:
            quality_cal["applied"][label] = value
        else:
            raise ValueError(f"未知类别: {label}")
        overrides[label] = value

    result: dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "calibration_split": args.split,
        "min_pos": args.min_pos,
        "overrides": overrides,
        "type": type_cal,
        "quality": quality_cal,
        "default_thresholds": {
            "type": {label: 0.5 for label in TYPE_CLASSES},
            "quality": {label: 0.5 for label in QUALITY_CLASSES},
        },
        "val_at_default": {
            "type": metrics_for(val_records["type_true"], val_records["type_scores"],
                                val_records["type_valid"],
                                {label: 0.5 for label in TYPE_CLASSES}, TYPE_CLASSES, args.split),
            "quality": metrics_for(val_records["quality_true"], val_records["quality_scores"],
                                   val_records["quality_valid"],
                                   {label: 0.5 for label in QUALITY_CLASSES}, QUALITY_CLASSES, args.split),
        },
        "val_calibrated": {
            "type": metrics_for(val_records["type_true"], val_records["type_scores"],
                                val_records["type_valid"], type_cal["applied"], TYPE_CLASSES, args.split),
            "quality": metrics_for(val_records["quality_true"], val_records["quality_scores"],
                                   val_records["quality_valid"], quality_cal["applied"],
                                   QUALITY_CLASSES, args.split),
        },
    }

    if args.test_split and args.test_split in split:
        test_samples = [samples[index] for index in split[args.test_split]]
        test_records = collect_scores(engine, _loader(engine, test_samples, args.batch_size), device)
        result["test_calibrated"] = {
            "type": metrics_for(test_records["type_true"], test_records["type_scores"],
                                test_records["type_valid"], type_cal["applied"],
                                TYPE_CLASSES, args.test_split),
            "quality": metrics_for(test_records["quality_true"], test_records["quality_scores"],
                                   test_records["quality_valid"], quality_cal["applied"],
                                   QUALITY_CLASSES, args.test_split),
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "type_applied": type_cal["applied"],
        "quality_applied": quality_cal["applied"],
        "val_macro_f1_default": {
            "type": result["val_at_default"]["type"]["macro_f1"],
            "quality": result["val_at_default"]["quality"]["macro_f1"],
        },
        "val_macro_f1_calibrated": {
            "type": result["val_calibrated"]["type"]["macro_f1"],
            "quality": result["val_calibrated"]["quality"]["macro_f1"],
        },
        "test_macro_f1_calibrated": {
            "type": result.get("test_calibrated", {}).get("type", {}).get("macro_f1"),
            "quality": result.get("test_calibrated", {}).get("quality", {}).get("macro_f1"),
        },
        "out": str(out_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
