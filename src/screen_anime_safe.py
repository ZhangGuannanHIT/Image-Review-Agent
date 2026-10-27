# -*- coding: utf-8 -*-
"""Build an auditable manifest for an external safe-anime image directory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def _dhash(image: Image.Image, size: int = 8) -> str:
    gray = ImageOps.grayscale(image).resize((size + 1, size), Image.Resampling.LANCZOS)
    values = np.asarray(gray, dtype=np.int16)
    bits = values[:, 1:] > values[:, :-1]
    return f"{int(''.join('1' if value else '0' for value in bits.flat), 2):0{size * size}x}"


def _hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _split_for(relative_path: str) -> str:
    bucket = int(hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 60:
        return "dev"
    if bucket < 80:
        return "validation"
    return "test"


def build_manifest(
    root: str | Path,
    output: str | Path,
    estimated_violation_rate: float = 0.0075,
) -> dict:
    root = Path(root)
    output = Path(output)
    rows = []
    errors = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative = path.relative_to(root).as_posix()
        try:
            payload = path.read_bytes()
            with Image.open(path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                width, height = image.size
                dhash = _dhash(image)
            rows.append({
                "relative_path": relative,
                "category": path.parent.name,
                "presumed_safe": 1,
                "human_verified": 0,
                "label_status": "unreviewed",
                "width": width,
                "height": height,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "dhash": dhash,
                "duplicate_of": "",
                "near_duplicate_of": "",
                "selected": 1,
                "split": _split_for(relative),
            })
        except Exception as exc:  # noqa: BLE001
            errors.append({
                "relative_path": relative,
                "error": type(exc).__name__,
            })

    by_sha = {}
    for row in rows:
        previous = by_sha.get(row["sha256"])
        if previous:
            row["duplicate_of"] = previous["relative_path"]
            row["selected"] = 0
            row["split"] = previous["split"]
        else:
            by_sha[row["sha256"]] = row

    selected = [row for row in rows if row["selected"]]
    for index, row in enumerate(selected):
        for other in selected[index + 1:]:
            if row["category"] != other["category"]:
                continue
            if _hamming(row["dhash"], other["dhash"]) <= 3:
                other["near_duplicate_of"] = row["relative_path"]
                other["selected"] = 0
                other["split"] = row["split"]

    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else [
        "relative_path", "category", "presumed_safe", "human_verified",
        "label_status", "width", "height", "bytes",
        "sha256", "dhash", "duplicate_of", "near_duplicate_of", "selected", "split",
    ]
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "root": str(root),
        "manifest": str(output),
        "n_found": len(rows) + len(errors),
        "n_valid": len(rows),
        "n_selected": sum(row["selected"] for row in rows),
        "n_exact_duplicate": sum(bool(row["duplicate_of"]) for row in rows),
        "n_near_duplicate": sum(bool(row["near_duplicate_of"]) for row in rows),
        "categories": dict(Counter(row["category"] for row in rows)),
        "splits": dict(Counter(row["split"] for row in rows)),
        "selected_splits": dict(
            Counter(row["split"] for row in rows if row["selected"])
        ),
        "errors": errors,
        "estimated_violation_rate": estimated_violation_rate,
        "label_note": (
            f"人工抽检提示约 {estimated_violation_rate:.1%} 为 L1 轻度违规；"
            "该清单是假定安全候选，不能直接作为零误差负样本。"
        ),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="构建安全二次元清单")
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--manifest",
        default="data/anime_safe/manifest.csv",
    )
    parser.add_argument(
        "--report",
        default="data/anime_safe/screen_report.json",
    )
    parser.add_argument(
        "--estimated-violation-rate",
        type=float,
        default=0.0075,
    )
    args = parser.parse_args()
    if not 0 <= args.estimated_violation_rate <= 1:
        parser.error("--estimated-violation-rate 必须位于 [0, 1]")
    summary = build_manifest(
        args.root,
        args.manifest,
        estimated_violation_rate=args.estimated_violation_rate,
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
