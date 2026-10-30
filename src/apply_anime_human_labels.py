# -*- coding: utf-8 -*-
"""Merge confirmed human labels into the generated anime manifest."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="合并二次元人工真值")
    parser.add_argument("--manifest", default="data/anime_safe/manifest.csv")
    parser.add_argument("--labels", default="data/anime_safe/human_labels.csv")
    parser.add_argument(
        "--report", default="data/anime_safe/label_merge_report.json"
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
        fieldnames = list(rows[0]) if rows else []
    with Path(args.labels).open("r", encoding="utf-8-sig", newline="") as file:
        labels = list(csv.DictReader(file))
    by_path = {row["relative_path"]: row for row in rows}
    missing = []
    for label in labels:
        row = by_path.get(label["relative_path"])
        if row is None:
            missing.append(label["relative_path"])
            continue
        if label.get("label_status") == "label_conflict_note":
            continue
        for field in (
            "human_label", "violence_type", "injury_detail", "quality",
            "style_domain", "scene_context", "human_note", "reviewer",
            "reviewed_at", "label_status",
        ):
            if field not in fieldnames:
                fieldnames.append(field)
            row[field] = label.get(field, "")
        row["human_verified"] = 1
        row["label_status"] = "confirmed"

    with manifest_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "manifest": str(manifest_path),
        "labels": str(args.labels),
        "confirmed_rows": sum(
            str(row.get("human_verified", "0")) == "1" for row in rows
        ),
        "matched_labels": len(labels) - len(missing),
        "missing_labels": missing,
        "label_counts": dict(
            Counter(
                row.get("human_label", "")
                for row in rows
                if str(row.get("human_verified", "0")) == "1"
            )
        ),
    }
    report_path = Path(args.report)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
