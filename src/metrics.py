# -*- coding: utf-8 -*-
"""只依赖 NumPy 的四任务基础评估指标。"""

from __future__ import annotations

from typing import Any

import numpy as np

from .labels import IGNORE_INDEX, INJURY_CLASSES, QUALITY_CLASSES, RISK_CLASSES, TYPE_CLASSES


def _prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def single_label_metrics(y_true, y_pred, classes: list[str]) -> dict[str, Any]:
    true, pred = np.asarray(y_true), np.asarray(y_pred)
    valid = true != IGNORE_INDEX
    if not valid.any():
        return {"count": 0, "accuracy": None, "macro_f1": None, "confusion_matrix": []}
    true, pred = true[valid], pred[valid]
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for expected, actual in zip(true, pred):
        if 0 <= expected < len(classes) and 0 <= actual < len(classes):
            matrix[int(expected), int(actual)] += 1
    f1_values = []
    for index in range(len(classes)):
        tp = matrix[index, index]
        fp = matrix[:, index].sum() - tp
        fn = matrix[index, :].sum() - tp
        f1_values.append(_prf(float(tp), float(fp), float(fn))[2])
    return {
        "count": int(valid.sum()),
        "accuracy": float((true == pred).mean()),
        "macro_f1": float(np.mean(f1_values)),
        "confusion_matrix": matrix.tolist(),
    }


def multilabel_metrics(y_true, y_pred, valid_mask, classes: list[str] = TYPE_CLASSES) -> dict[str, Any]:
    true, pred = np.asarray(y_true), np.asarray(y_pred)
    valid = np.asarray(valid_mask, dtype=bool)
    if not valid.any():
        return {"count": 0, "micro_f1": None, "macro_f1": None, "per_class": {}}
    true, pred = true[valid], pred[valid]
    per_class, f1_values = {}, []
    total_tp = total_fp = total_fn = 0.0
    for index, label in enumerate(classes):
        tp = float(np.logical_and(true[:, index] == 1, pred[:, index] == 1).sum())
        fp = float(np.logical_and(true[:, index] == 0, pred[:, index] == 1).sum())
        fn = float(np.logical_and(true[:, index] == 1, pred[:, index] == 0).sum())
        precision, recall, f1 = _prf(tp, fp, fn)
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
        f1_values.append(f1)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    return {
        "count": int(valid.sum()),
        "micro_f1": _prf(total_tp, total_fp, total_fn)[2],
        "macro_f1": float(np.mean(f1_values)),
        "per_class": per_class,
    }


def review_metrics(records: dict[str, list]) -> dict[str, Any]:
    result = {
        "risk": single_label_metrics(records["risk_true"], records["risk_pred"], RISK_CLASSES),
        "type": multilabel_metrics(records["type_true"], records["type_pred"], records["type_valid"], TYPE_CLASSES),
        "injury": single_label_metrics(records["injury_true"], records["injury_pred"], INJURY_CLASSES),
        "quality": multilabel_metrics(
            records["quality_true"], records["quality_pred"], records["quality_valid"], QUALITY_CLASSES
        ),
    }
    risk_true, risk_pred = np.asarray(records["risk_true"]), np.asarray(records["risk_pred"])
    l3 = risk_true == RISK_CLASSES.index("L3_严重违规")
    result["risk"]["l3_recall"] = float((risk_pred[l3] == risk_true[l3]).mean()) if l3.any() else None
    return result
