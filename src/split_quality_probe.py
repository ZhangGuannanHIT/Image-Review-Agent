# -*- coding: utf-8 -*-
"""Build a quality-probe manifest and deterministic dev/final split."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


QUALITY_TAG_MAP = {
    "clear": "清晰",
    "mild_blur": "轻微模糊",
    "severe_blur": "严重模糊",
    "text_distortion": "文字扭曲形变",
    "occlusion": "故意遮挡（马赛克/黑条/贴纸）",
}
DEV_CELLS = {
    ("real", "safe"): 3,
    ("real", "violation"): 2,
    ("ai", "safe"): 2,
    ("ai", "violation"): 3,
}


def load_quality_rows(metadata_csv: str | Path) -> list[dict]:
    with Path(metadata_csv).open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    required = {"path", "category", "origin", "label", "style", "color", "quality_tags"}
    missing = required - set(rows[0] if rows else {})
    if missing:
        raise ValueError(f"quality metadata 缺少字段: {sorted(missing)}")
    output = []
    for row in rows:
        tags = json.loads(row["quality_tags"])
        unknown = set(tags) - set(QUALITY_TAG_MAP)
        if unknown:
            raise ValueError(f"{row['path']} 存在未知 quality tag: {sorted(unknown)}")
        output.append(
            {
                "file": row["path"],
                "category": row["category"],
                "origin": row["origin"],
                "label": row["label"],
                "style": row["style"],
                "color": row["color"],
                "quality_tags": tags,
                "reviewer": row.get("reviewer", ""),
                "source_url": row.get("source_url", ""),
                "license": row.get("license", ""),
            }
        )
    return output


def make_split(rows: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
    dev: list[dict] = []
    final: list[dict] = []
    for category in sorted(by_category):
        selected: list[dict] = []
        category_rows = by_category[category]
        for origin, label in sorted(DEV_CELLS):
            count = DEV_CELLS[(origin, label)]
            pool = [
                row for row in category_rows
                if row["origin"] == origin and row["label"] == label
            ]
            rng.shuffle(pool)
            if len(pool) < count:
                raise ValueError(f"{category}/{origin}/{label} 样本不足: {len(pool)} < {count}")
            selected.extend(pool[:count])
        selected_files = {row["file"] for row in selected}
        dev.extend(sorted(selected, key=lambda row: row["file"]))
        final.extend(sorted(
            (row for row in category_rows if row["file"] not in selected_files),
            key=lambda row: row["file"],
        ))
    return dev, final


def write_manifest(path: str | Path, rows: list[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "file", "category", "origin", "label", "style", "color",
        "quality_tags", "reviewer", "source_url", "license",
    ]
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            item = dict(row)
            item["quality_tags"] = json.dumps(item["quality_tags"], ensure_ascii=False)
            writer.writerow(item)


def summarize(rows: list[dict]) -> dict:
    return {
        "count": len(rows),
        "category": dict(Counter(row["category"] for row in rows)),
        "origin": dict(Counter(row["origin"] for row in rows)),
        "label": dict(Counter(row["label"] for row in rows)),
        "color": dict(Counter(row["color"] for row in rows)),
        "style": dict(Counter(row["style"] for row in rows)),
        "quality_tags": dict(Counter(tag for row in rows for tag in row["quality_tags"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="生成质量探针 manifest 与固定划分")
    parser.add_argument("--metadata", default="../quality/metadata.csv")
    parser.add_argument("--out-dir", default="data/quality_probe")
    parser.add_argument("--seed", type=int, default=20261022)
    args = parser.parse_args()

    rows = load_quality_rows(args.metadata)
    dev, final = make_split(rows, args.seed)
    output = Path(args.out_dir)
    write_manifest(output / "manifest.csv", rows)
    write_manifest(output / "dev.csv", dev)
    write_manifest(output / "final.csv", final)
    report = {
        "seed": args.seed,
        "dev_cell_targets": {
            f"{origin}:{label}": count
            for (origin, label), count in sorted(DEV_CELLS.items())
        },
        "all": summarize(rows),
        "dev": summarize(dev),
        "final": summarize(final),
    }
    (output / "split.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
