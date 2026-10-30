# -*- coding: utf-8 -*-
"""Evaluate the lightweight OCR gate on a frozen OCR probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.text_detection import LightweightTextDetector


def evaluate(rows: list[dict], image_root: str | Path, detector=None) -> dict:
    detector = detector or LightweightTextDetector()
    true_positive = false_positive = true_negative = false_negative = 0
    errors = []
    output_rows = []
    for row in rows:
        path = Path(image_root) / row["file"]
        expected = bool(str(row.get("text", "") or "").strip())
        try:
            result = detector.detect(path.read_bytes())
            predicted = bool(result.get("has_text"))
            error = None
        except Exception as exc:  # noqa: BLE001
            # A detector failure is deliberately treated as "send to OCR" by
            # the workflow, not as proof that no text exists.
            result = {"status": "failed", "has_text": False}
            predicted = True
            error = type(exc).__name__
            errors.append({"file": row["file"], "error": error})
        if expected and predicted:
            true_positive += 1
        elif expected:
            false_negative += 1
        elif predicted:
            false_positive += 1
        else:
            true_negative += 1
        output_rows.append({
            "file": row["file"],
            "expected_text": expected,
            "detected": predicted,
            "status": result.get("status"),
            "box_count": result.get("box_count", 0),
            "confidence": result.get("confidence"),
            "error": error,
        })
    positives = true_positive + false_negative
    negatives = true_negative + false_positive
    return {
        "n": len(rows),
        "metrics": {
            "recall": round(true_positive / max(positives, 1), 6),
            "false_trigger_rate": round(false_positive / max(negatives, 1), 6),
            "precision": round(
                true_positive / max(true_positive + false_positive, 1), 6
            ),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "true_negative": true_negative,
            "false_negative": false_negative,
        },
        "errors": errors,
        "rows": output_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="轻量文字检测器冻结评估")
    parser.add_argument("--probe", default="eval_results/quality_ocr_probe.json")
    parser.add_argument("--image-root", default="../quality")
    parser.add_argument("--scan-root", default="")
    parser.add_argument("--min-layout-score", type=float, default=0.48)
    parser.add_argument("--min-region-score", type=float, default=0.52)
    parser.add_argument("--min-aligned-glyphs", type=int, default=2)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    detector = LightweightTextDetector(
        min_layout_score=args.min_layout_score,
        min_region_score=args.min_region_score,
        min_aligned_glyphs=args.min_aligned_glyphs,
    )
    payload = json.loads(Path(args.probe).read_text(encoding="utf-8"))
    rows = payload.get("rows", payload) if isinstance(payload, dict) else payload
    report = evaluate(rows, args.image_root, detector)
    scan = None
    if args.scan_root:
        image_paths = [
            path for path in Path(args.scan_root).rglob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ]
        detected = 0
        errors = []
        for path in image_paths:
            try:
                detected += int(bool(detector.detect(path.read_bytes())["has_text"]))
            except Exception as exc:  # noqa: BLE001
                detected += 1
                errors.append({"file": str(path), "error": type(exc).__name__})
        scan = {
            "root": args.scan_root,
            "n": len(image_paths),
            "detected": detected,
            "trigger_rate": round(detected / max(len(image_paths), 1), 6),
            "errors": errors,
        }
    report.update({
        "probe": args.probe,
        "image_root": args.image_root,
        "detector": detector.method,
        "parameters": {
            "min_layout_score": args.min_layout_score,
            "min_region_score": args.min_region_score,
            "min_aligned_glyphs": args.min_aligned_glyphs,
        },
        "scan": scan,
    })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "metrics": report["metrics"],
        "scan": report["scan"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
