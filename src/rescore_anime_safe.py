# -*- coding: utf-8 -*-
"""Rescore cached anime evidence after policy or human-label changes."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import yaml

from backend.policy import DecisionPolicy, decide
from src.evaluate_anime_safe import _aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description="重算安全二次元策略动作")
    parser.add_argument(
        "--evaluation", default="data/anime_safe/evaluation_v5_final.json"
    )
    parser.add_argument("--manifest", default="data/anime_safe/manifest.csv")
    parser.add_argument("--backend-config", default="backend/config.yaml")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.evaluation).read_text(encoding="utf-8"))
    with Path(args.manifest).open("r", encoding="utf-8-sig", newline="") as file:
        manifest = {
            row["relative_path"]: row for row in csv.DictReader(file)
        }
    backend = yaml.safe_load(
        Path(args.backend_config).read_text(encoding="utf-8")
    ) or {}
    policy = DecisionPolicy(**(backend.get("decision_policy") or {}))
    rows = []
    for old in payload["rows"]:
        current = manifest.get(old["relative_path"], {})
        evidence = old["evidence"]
        decision = decide(evidence, policy)
        human_label = current.get("human_label", "")
        rows.append({
            **old,
            "human_label": human_label,
            "human_verified": current.get("human_verified", "0"),
            "confirmed_violation": bool(
                current.get("human_verified") == "1"
                and human_label in {
                    "L1_轻微违规", "L2_中度违规", "L3_严重违规"
                }
            ),
            "effective_risk": decision["effective_risk"],
            "action": decision["action"],
            "conflict_type": decision["conflict_type"],
            "adjustments": decision["adjustments"],
            "reason": decision["reason"],
        })
    report = {
        **{key: value for key, value in payload.items() if key != "rows"},
        "metrics": {
            "all": {
                "n": len(rows),
                "actions": dict(Counter(row["action"] for row in rows)),
                "review_rate": round(
                    sum(row["action"] == "review" for row in rows)
                    / max(len(rows), 1),
                    6,
                ),
                "confirmed_violations": sum(
                    row["confirmed_violation"] for row in rows
                ),
                "confirmed_violations_auto_allowed": sum(
                    row["confirmed_violation"] and row["action"] == "allow"
                    for row in rows
                ),
            },
            "by_split": _aggregate(rows, "split"),
            "by_category": _aggregate(rows, "category"),
            "by_risk": _aggregate(rows, "risk"),
        },
        "rows": rows,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
