# -*- coding: utf-8 -*-
"""Calibrate quality-fusion thresholds on the development probe only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .labels import QUALITY_CLASSES


def calibrate(rows: list[dict]) -> tuple[dict[str, float], dict]:
    expected = np.asarray([row["expected"] for row in rows], dtype=int)
    scores = np.asarray(
        [[row["scores"][label] for label in QUALITY_CLASSES] for row in rows],
        dtype=np.float64,
    )
    thresholds = {}
    details = {}
    for index, label in enumerate(QUALITY_CLASSES):
        best = {"threshold": 0.5, "f1": -1.0}
        for threshold in np.linspace(0.05, 0.95, 181):
            predicted = scores[:, index] >= threshold
            truth = expected[:, index] == 1
            tp = int(np.logical_and(predicted, truth).sum())
            fp = int(np.logical_and(predicted, ~truth).sum())
            fn = int(np.logical_and(~predicted, truth).sum())
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-12)
            if f1 > best["f1"]:
                best = {
                    "threshold": float(threshold),
                    "f1": f1,
                    "precision": precision,
                    "recall": recall,
                }
        thresholds[label] = best["threshold"]
        details[label] = best
    return thresholds, details


def main() -> None:
    parser = argparse.ArgumentParser(description="在 quality dev 上标定融合头阈值")
    parser.add_argument("--dev-report", default="eval_results/quality_dev_fusion.json")
    parser.add_argument("--out", default="eval_results/quality_fusion_dev_thresholds.json")
    args = parser.parse_args()
    report = json.loads(Path(args.dev_report).read_text(encoding="utf-8"))
    thresholds, details = calibrate(report["rows"])
    output = {"thresholds": thresholds, "details": details}
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
