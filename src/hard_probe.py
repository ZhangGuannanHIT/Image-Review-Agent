# -*- coding: utf-8 -*-
"""Frozen hard-probe manifest loading and binary injury metrics."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .labels import INJURY_CLASSES


REQUIRED_COLUMNS = {
    "file", "folder", "domain", "red_level", "has_blood",
}
BOOLEAN_VALUES = {"0", "1", "false", "true", "no", "yes"}


@dataclass(frozen=True)
class HardProbeItem:
    file: str
    path: Path
    folder: str
    domain: str
    red_level: str
    has_blood: bool
    severity_truth: str = ""
    risk_truth: str = ""
    type_truth: str = ""
    context: str = ""
    notes: str = ""


def _parse_bool(value: str, row_number: int) -> bool:
    normalized = str(value).strip().casefold()
    if normalized not in BOOLEAN_VALUES:
        raise ValueError(f"manifest 第 {row_number} 行 has_blood 无效: {value!r}")
    return normalized in {"1", "true", "yes"}


def load_manifest(path: str | Path, image_root: str | Path) -> list[HardProbeItem]:
    """Load and validate the frozen probe; reject duplicate files."""
    manifest_path = Path(path)
    root = Path(image_root)
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"hard-probe manifest 缺少列: {sorted(missing)}")
        items: list[HardProbeItem] = []
        seen: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            relative = str(row["file"]).strip().replace("\\", "/")
            if not relative:
                raise ValueError(f"manifest 第 {row_number} 行缺少 file")
            if relative in seen:
                raise ValueError(f"manifest 存在重复 file: {relative}")
            seen.add(relative)
            image_path = root / relative
            if not image_path.is_file():
                raise FileNotFoundError(f"硬探针图片不存在: {image_path}")
            severity = str(row.get("severity_truth", "")).strip()
            if severity and severity not in INJURY_CLASSES[1:]:
                raise ValueError(f"manifest 第 {row_number} 行 severity_truth 无效: {severity}")
            has_blood = _parse_bool(row["has_blood"], row_number)
            folder = str(row["folder"]).strip()
            if "_no_blood" in folder and has_blood:
                raise ValueError(f"manifest 第 {row_number} 行 no_blood 目录却标为有血")
            if folder.endswith("_blood") and "_no_blood" not in folder and not has_blood:
                raise ValueError(f"manifest 第 {row_number} 行 blood 目录却标为无血")
            items.append(
                HardProbeItem(
                    file=relative,
                    path=image_path,
                    folder=folder,
                    domain=str(row["domain"]).strip(),
                    red_level=str(row["red_level"]).strip(),
                    has_blood=has_blood,
                    severity_truth=severity,
                    risk_truth=str(row.get("risk_truth", "")).strip(),
                    type_truth=str(row.get("type_truth", "")).strip(),
                    context=str(row.get("context", "")).strip(),
                    notes=str(row.get("notes", "")).strip(),
                )
            )
    if not items:
        raise ValueError("hard-probe manifest 不能为空")
    return items


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def binary_metrics(labels: Iterable[bool], scores: Iterable[float]) -> dict:
    y = np.asarray(list(labels), dtype=bool)
    s = np.asarray(list(scores), dtype=np.float64)
    if len(y) != len(s) or len(y) == 0:
        raise ValueError("labels 与 scores 必须等长且非空")
    if not np.isfinite(s).all():
        raise ValueError("scores 必须全部是有限值")
    positive = int(y.sum())
    negative = int((~y).sum())
    auroc = None
    if positive and negative:
        ranks = average_ranks(s)
        auroc = float(
            (ranks[y].sum() - positive * (positive + 1) / 2.0)
            / (positive * negative)
        )

    order = np.argsort(-s, kind="mergesort")
    sorted_labels = y[order]
    cumulative_tp = np.cumsum(sorted_labels)
    cumulative_fp = np.cumsum(~sorted_labels)
    precision_curve = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1)
    recall_curve = cumulative_tp / max(positive, 1)
    auprc = (
        float(np.sum(np.diff(np.r_[0.0, recall_curve]) * precision_curve))
        if positive else None
    )

    thresholds = {}
    for threshold in (0.50, 0.65, 0.75, 0.85):
        predicted = s >= threshold
        tp = int(np.logical_and(predicted, y).sum())
        fp = int(np.logical_and(predicted, ~y).sum())
        tn = int(np.logical_and(~predicted, ~y).sum())
        fn = int(np.logical_and(~predicted, y).sum())
        thresholds[f"{threshold:.2f}"] = {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "sensitivity": tp / max(tp + fn, 1),
            "specificity": tn / max(tn + fp, 1),
            "false_positive_rate": fp / max(fp + tn, 1),
            "precision": tp / max(tp + fp, 1),
        }
    return {
        "n": len(y),
        "positive": positive,
        "negative": negative,
        "mean_score": float(s.mean()),
        "auroc": auroc,
        "auprc": auprc,
        "thresholds": thresholds,
    }


def metrics_at_threshold(
    labels: Iterable[bool], scores: Iterable[float], threshold: float,
) -> dict:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold 必须位于 [0, 1]")
    y = np.asarray(list(labels), dtype=bool)
    s = np.asarray(list(scores), dtype=np.float64)
    if len(y) != len(s) or len(y) == 0:
        raise ValueError("labels 与 scores 必须等长且非空")
    predicted = s >= threshold
    tp = int(np.logical_and(predicted, y).sum())
    fp = int(np.logical_and(predicted, ~y).sum())
    tn = int(np.logical_and(~predicted, ~y).sum())
    fn = int(np.logical_and(~predicted, y).sum())
    return {
        "threshold": threshold,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "false_positive_rate": fp / max(fp + tn, 1),
        "precision": tp / max(tp + fp, 1),
    }


def grouped_metrics(rows: list[dict], score_key: str = "p_blood") -> dict:
    groups: dict[str, list[dict]] = {"all": rows}
    for key in ("domain", "red_level", "folder"):
        for row in rows:
            groups.setdefault(f"{key}:{row[key]}", []).append(row)
    return {
        name: binary_metrics(
            (row["has_blood"] for row in values),
            (row[score_key] for row in values),
        )
        for name, values in groups.items()
    }


def injury_probability(output) -> float:
    scores = output.injury.get("scores", {}) or {}
    if set(scores) != set(INJURY_CLASSES):
        raise ValueError("injury 输出缺少完整类别分数")
    value = sum(float(scores[label]) for label in INJURY_CLASSES[1:])
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"injury 血腥概率无效: {value!r}")
    return value


def evaluate_items(items: list[HardProbeItem], engine) -> list[dict]:
    rows = []
    for index, item in enumerate(items, start=1):
        output = engine.predict(str(item.path))
        rows.append(
            {
                "file": item.file,
                "folder": item.folder,
                "domain": item.domain,
                "red_level": item.red_level,
                "has_blood": item.has_blood,
                "severity_truth": item.severity_truth,
                "p_blood": injury_probability(output),
                "injury_label": output.injury.get("label", ""),
                "risk_label": output.risk_label,
                "type_labels": list(output.type_labels),
            }
        )
        if index % 10 == 0:
            print(f"hard-probe: {index}/{len(items)}", flush=True)
    return rows
