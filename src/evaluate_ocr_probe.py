# -*- coding: utf-8 -*-
"""Evaluate OCR readability on the quality development probe.

Results are cached after every successful API call so reruns do not consume
additional OCR quota.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from statistics import mean

from backend.ocr import OCRClient


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    return "".join(char for char in normalized if char.isalnum())


def character_recall(expected: str, actual: str) -> float | None:
    expected_chars = set(normalize_text(expected))
    if not expected_chars:
        return None
    actual_chars = set(normalize_text(actual))
    return len(expected_chars & actual_chars) / len(expected_chars)


def load_rows(manifest: str | Path, metadata: str | Path) -> list[dict]:
    with Path(metadata).open("r", encoding="utf-8-sig", newline="") as file:
        metadata_by_path = {row["path"]: row for row in csv.DictReader(file)}
    with Path(manifest).open("r", encoding="utf-8-sig", newline="") as file:
        rows = [
            row for row in csv.DictReader(file)
            if row["category"] in {"quality_clear", "quality_text_distortion"}
        ]
    for row in rows:
        row["original_text"] = metadata_by_path.get(row["file"], {}).get(
            "original_text", ""
        )
    return rows


def aggregate(rows: list[dict]) -> dict:
    output = {}
    for category in sorted({row["category"] for row in rows}):
        for label in sorted({row["label"] for row in rows}):
            group = [
                row for row in rows
                if row["category"] == category and row["label"] == label
            ]
            if not group:
                continue
            successful = [row for row in group if row["success"]]
            confidences = [
                row["mean_confidence"] for row in successful
                if row["mean_confidence"] is not None
            ]
            recalls = [
                row["character_recall"] for row in successful
                if row["character_recall"] is not None
            ]
            elapsed = [
                row["elapsed_ms"] for row in group
                if row.get("elapsed_ms") is not None
            ]
            output[f"{category}:{label}"] = {
                "count": len(group),
                "success": len(successful),
                "nonempty": sum(bool(row["text"]) for row in successful),
                "readable": sum(bool(row["readable"]) for row in successful),
                "mean_confidence": float(mean(confidences)) if confidences else None,
                "mean_character_recall": float(mean(recalls)) if recalls else None,
                "mean_elapsed_ms": float(mean(elapsed)) if elapsed else None,
            }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="质量开发集 OCR 小样本评估")
    parser.add_argument("--manifest", default="data/quality_probe/dev.csv")
    parser.add_argument("--metadata", default="../quality/metadata.csv")
    parser.add_argument("--image-root", default="../quality")
    parser.add_argument("--ocr-config", default="config/ocr.yaml")
    parser.add_argument("--out", default="eval_results/quality_ocr_probe.json")
    parser.add_argument("--max-calls", type=int, default=20)
    args = parser.parse_args()

    output_path = Path(args.out)
    cached = {}
    if output_path.is_file():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        cached = {row["file"]: row for row in existing.get("rows", [])}

    rows = load_rows(args.manifest, args.metadata)
    client = OCRClient.from_config(args.ocr_config)
    if not client.available():
        raise RuntimeError("OCR 未配置")

    calls = 0
    for row in rows:
        if row["file"] in cached:
            continue
        if calls >= args.max_calls:
            break
        image_path = Path(args.image_root) / row["file"]
        started = time.monotonic()
        result = client.recognize(image_path.read_bytes())
        elapsed_ms = round((time.monotonic() - started) * 1000)
        text = " ".join(result.get("texts", []) or [])
        confidence = result.get("mean_confidence")
        char_count = len(normalize_text(text))
        cached[row["file"]] = {
            "file": row["file"],
            "category": row["category"],
            "origin": row["origin"],
            "label": row["label"],
            "style": row["style"],
            "original_text": row["original_text"],
            "success": bool(result.get("success")),
            "available": bool(result.get("available")),
            "error": result.get("error"),
            "text": text,
            "char_count": char_count,
            "mean_confidence": confidence,
            "line_confidences": result.get("line_confidences", []),
            "box_count": len(result.get("boxes", []) or []),
            "readable": bool(
                result.get("success")
                and char_count >= 2
                and isinstance(confidence, (int, float))
                and confidence >= 0.75
            ),
            "character_recall": character_recall(row["original_text"], text),
            "request_id": result.get("request_id", ""),
            "elapsed_ms": elapsed_ms,
        }
        calls += 1
        print(
            f"ocr {calls}/{args.max_calls}: {row['category']} "
            f"{row['label']} success={result.get('success')} "
            f"conf={confidence} chars={char_count}",
            flush=True,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "calls_this_run": calls,
                    "rows": list(cached.values()),
                    "metrics": aggregate(list(cached.values())),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        if not result.get("success"):
            break

    all_rows = list(cached.values())
    report = {
        "calls_this_run": calls,
        "rows": all_rows,
        "metrics": aggregate(all_rows),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
