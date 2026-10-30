# -*- coding: utf-8 -*-
"""Create a human-review queue from the anime safe-candidate evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def priority(row: dict) -> int:
    risk = row.get("risk")
    if risk == "L1_轻微违规":
        return 1
    if risk == "L0_安全" and float(row.get("binary_score") or 0) >= 0.5:
        return 2
    if risk in {"L3_严重违规", "X_无法判定"}:
        return 3
    if risk == "L2_中度违规":
        return 4
    return 5


def reason(row: dict) -> str:
    item_priority = priority(row)
    return {
        1: "模型 risk=L1，优先确认人工抽检发现的轻度违规",
        2: "模型 risk=L0，但 binary 较高",
        3: "模型 risk=L3/X，需确认是风格假阳性还是边界违规",
        4: "模型 risk=L2，可能是动漫域假阳性",
        5: "常规优先级",
    }[item_priority]


def main() -> None:
    parser = argparse.ArgumentParser(description="构建二次元人工复核队列")
    parser.add_argument(
        "--evaluation", default="data/anime_safe/evaluation_v5_rescored.json"
    )
    parser.add_argument("--image-root", default=r"D:\动漫图片")
    parser.add_argument(
        "--human-labels", default="data/anime_safe/human_labels.csv"
    )
    parser.add_argument("--out", default="data/anime_safe/review_queue.csv")
    args = parser.parse_args()
    payload = json.loads(Path(args.evaluation).read_text(encoding="utf-8"))
    labels = {}
    labels_path = Path(args.human_labels)
    if labels_path.is_file():
        with labels_path.open("r", encoding="utf-8-sig", newline="") as file:
            labels = {
                row["relative_path"]: row for row in csv.DictReader(file)
            }
    rows = sorted(
        payload["rows"],
        key=lambda row: (
            row["relative_path"] in labels,
            priority(row),
            -float(row.get("risk_confidence") or 0),
            -float(row.get("binary_score") or 0),
            row["relative_path"],
        ),
    )
    output_rows = []
    for row in rows:
        human = labels.get(row["relative_path"], {})
        output_rows.append({
            "priority": priority(row),
            "priority_reason": reason(row),
            "relative_path": row["relative_path"],
            "image_path": str(Path(args.image_root) / row["relative_path"]),
            "category": row["category"],
            "split": row["split"],
            "model_risk": row["risk"],
            "risk_confidence": row["risk_confidence"],
            "binary_score": row["binary_score"],
            "binary_state": row["binary_state"],
            "quality_labels": "|".join(row["quality_labels"]),
            "domain_matched": row["domain_matched"],
            "model_action": row["action"],
            "model_effective_risk": row["effective_risk"],
            "human_label": human.get("human_label", ""),
            "human_note": human.get("human_note", ""),
            "human_verified": 1 if human else 0,
        })
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(json.dumps({
        "out": str(output),
        "n": len(output_rows),
        "priority_counts": {
            str(level): sum(row["priority"] == level for row in output_rows)
            for level in range(1, 6)
        },
        "next_unverified_files": [
            row["relative_path"] for row in output_rows
            if not row["human_verified"]
        ][:10],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
