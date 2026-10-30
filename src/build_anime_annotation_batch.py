# -*- coding: utf-8 -*-
"""Create a deterministic Label Studio batch for anime policy calibration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def _rank(prefix: str, path: str) -> str:
    return hashlib.sha256(f"{prefix}|{path}".encode("utf-8")).hexdigest()


def _batch_name(index: int, row: dict) -> str:
    source = Path(row["relative_path"])
    stem = re.sub(r"[^0-9A-Za-z_.-]+", "_", source.stem).strip("_")
    return f"{index:03d}_{row['category']}_{stem}{source.suffix.lower()}"


def main() -> None:
    parser = argparse.ArgumentParser(description="生成二次元 Label Studio 标注批次")
    parser.add_argument(
        "--evaluation",
        default="data/anime_safe/evaluation_v5_rescored.json",
    )
    parser.add_argument("--image-root", default="/mnt/d/动漫图片")
    parser.add_argument(
        "--output", default="/mnt/d/动漫图片_待标注_2026-10-27"
    )
    parser.add_argument("--random-count", type=int, default=60)
    parser.add_argument("--hard-count", type=int, default=60)
    args = parser.parse_args()

    payload = json.loads(Path(args.evaluation).read_text(encoding="utf-8"))
    unverified = [
        row for row in payload["rows"]
        if str(row.get("human_verified", "0")) != "1"
    ]
    random_pool = sorted(
        unverified,
        key=lambda row: _rank("random", row["relative_path"]),
    )
    random_rows = random_pool[:args.random_count]
    selected = {row["relative_path"] for row in random_rows}
    hard_pool = sorted(
        (
            row for row in unverified
            if row["risk"] in {"X_无法判定", "L3_严重违规"}
            and row["relative_path"] not in selected
        ),
        key=lambda row: _rank("hard", row["relative_path"]),
    )
    hard_rows = hard_pool[:args.hard_count]
    if len(hard_rows) < args.hard_count:
        remaining = sorted(
            (
                row for row in unverified
                if row["relative_path"] not in selected
                and row not in hard_rows
            ),
            key=lambda row: _rank("fill", row["relative_path"]),
        )
        hard_rows.extend(remaining[:args.hard_count - len(hard_rows)])

    output = Path(args.output)
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    source_root = Path(args.image_root)
    manifest = []
    for index, (row, reason) in enumerate(
        [(row, "random_rate_sample") for row in random_rows]
        + [(row, "hard_x_l3_sample") for row in hard_rows],
        start=1,
    ):
        batch_filename = _batch_name(index, row)
        source = source_root / row["relative_path"]
        destination = image_dir / batch_filename
        shutil.copy2(source, destination)
        if hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(
            destination.read_bytes()
        ).digest():
            raise RuntimeError(f"图片复制校验失败: {row['relative_path']}")
        with Image.open(destination) as copied:
            copied.verify()
        manifest.append({
            "batch_id": index,
            "batch_filename": batch_filename,
            "original_relative_path": row["relative_path"],
            "source_category": row["category"],
            "split": row["split"],
            "selection_reason": reason,
            "model_risk": row["risk"],
            "risk_confidence": row.get("risk_confidence"),
            "binary_score": row.get("binary_score"),
            "domain_matched": row.get("domain_matched"),
            "model_action": row["action"],
            "model_effective_risk": row["effective_risk"],
            "human_label": "",
            "human_note": "",
            "human_verified": 0,
        })

    with (output / "annotation_manifest.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)

    tasks = [
        {
            "data": {"image": f"images/{row['batch_filename']}"},
            "meta": {
                "batch_id": row["batch_id"],
                "original_relative_path": row["original_relative_path"],
                "selection_reason": row["selection_reason"],
            },
        }
        for row in manifest
    ]
    (output / "labelstudio_tasks.json").write_text(
        json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    shutil.copy2(
        ROOT / "config" / "anime_annotation_labelstudio.xml",
        output / "labelstudio_config.xml",
    )
    shutil.copy2(
        ROOT / "config" / "anime_annotation_guide.md",
        output / "README_标注说明.md",
    )
    summary = {
        "output": str(output),
        "n": len(manifest),
        "selection": dict(Counter(row["selection_reason"] for row in manifest)),
        "source_category": dict(Counter(row["source_category"] for row in manifest)),
        "model_risk": dict(Counter(row["model_risk"] for row in manifest)),
        "model_action": dict(Counter(row["model_action"] for row in manifest)),
        "verified_images": len(manifest),
    }
    (output / "batch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
