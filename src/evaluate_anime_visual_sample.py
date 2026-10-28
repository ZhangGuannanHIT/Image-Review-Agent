# -*- coding: utf-8 -*-
"""Small live sample for anime visual-review trigger and reconciliation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml

from backend.anime_vision import AnimeVisualReviewer
from backend.anime_vision_policy import anime_visual_gate, reconcile_anime_visual
from backend.llm import LLMClient
from backend.policy import DecisionPolicy, decide


def rank(path: str) -> str:
    return hashlib.sha256(path.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="动漫视觉复核小样本实测")
    parser.add_argument(
        "--anime-evaluation",
        default="data/anime_safe/evaluation_router_v1.json",
    )
    parser.add_argument(
        "--solid-evaluation",
        default="eval_results/solid_effective_v5_visual_full.json",
    )
    parser.add_argument("--anime-root", default="/mnt/d/动漫图片")
    parser.add_argument("--solid-root", default="solid")
    parser.add_argument("--llm-config", default="config/llm.yaml")
    parser.add_argument("--backend-config", default="backend/config.yaml")
    parser.add_argument("--manifest", default="data/anime_safe/manifest.csv")
    parser.add_argument("--safe-count", type=int, default=10)
    parser.add_argument("--positive-count", type=int, default=3)
    parser.add_argument(
        "--reuse-results",
        default="",
        help="reuse previously returned visual reviews without new API calls",
    )
    parser.add_argument("--out", default="data/anime_safe/visual_sample.json")
    args = parser.parse_args()

    backend = yaml.safe_load(
        Path(args.backend_config).read_text(encoding="utf-8")
    ) or {}
    policy = DecisionPolicy(**(backend.get("decision_policy") or {}))
    min_style = float(
        (backend.get("anime_vision_review") or {}).get(
            "min_style_score", 0.90
        )
    )
    allow_safety_review = bool(
        (backend.get("anime_vision_review") or {}).get(
            "allow_safety_review", True
        )
    )
    anime = json.loads(
        Path(args.anime_evaluation).read_text(encoding="utf-8")
    )["rows"]
    solid = json.loads(
        Path(args.solid_evaluation).read_text(encoding="utf-8")
    )["rows"]
    with Path(args.manifest).open("r", encoding="utf-8-sig", newline="") as file:
        labels = {
            item["relative_path"]: item for item in csv.DictReader(file)
        }
    safe = sorted(
        (
            row for row in anime
            if labels[row["relative_path"]].get("human_verified") == "1"
            and labels[row["relative_path"]].get("human_label") == "L0_安全"
            and row.get("action") != "allow"
        ),
        key=lambda row: rank(row["relative_path"]),
    )[:args.safe_count]
    confirmed_l1 = [
        row for row in anime
        if labels[row["relative_path"]].get("human_verified") == "1"
        and labels[row["relative_path"]].get("human_label") == "L1_轻微违规"
    ]
    positives = sorted(
        (
            row for row in solid
            if row.get("domain") == "ai" and row.get("is_violation")
        ),
        key=lambda row: rank(row["file"]),
    )[:args.positive_count]
    items = (
        [(row, "safe", Path(args.anime_root) / row["relative_path"]) for row in safe]
        + [(row, "L1", Path(args.anime_root) / row["relative_path"]) for row in confirmed_l1]
        + [(row, "blood_positive", Path(args.solid_root) / row["file"]) for row in positives]
    )
    model = LLMClient.from_config(args.llm_config).build_model()
    reviewer = AnimeVisualReviewer(model)
    reused = {}
    if args.reuse_results:
        previous = json.loads(
            Path(args.reuse_results).read_text(encoding="utf-8")
        )
        reused = {
            row["path"]: row.get("visual_review")
            for row in previous.get("rows", [])
        }
    output = []
    for index, (row, truth, path) in enumerate(items, start=1):
        evidence = row["evidence"]
        decision = decide(evidence, policy)
        gate = anime_visual_gate(
            decision,
            evidence,
            min_style_score=min_style,
            allow_safety_review=allow_safety_review,
        )
        if not gate["eligible"]:
            review = None
        elif str(path) in reused:
            review = reused[str(path)]
        else:
            review = reviewer.review(path.read_bytes())
        final = (
            reconcile_anime_visual(decision, evidence, review)
            if review is not None else decision
        )
        output.append({
            "truth": truth,
            "path": str(path),
            "gate": gate,
            "initial_action": decision["action"],
            "visual_review": review,
            "final_action": final["action"],
            "final_effective_risk": final["effective_risk"],
            "visual_effect": final.get("anime_visual_effect", "not_called"),
        })
        print(
            f"visual-sample: {index}/{len(items)} truth={truth} "
            f"initial={decision['action']} final={final['action']}",
            flush=True,
        )
    summary = {
        "n": len(output),
        "truth_counts": dict(Counter(item["truth"] for item in output)),
        "final_actions": dict(Counter(item["final_action"] for item in output)),
        "by_truth": {
            truth: dict(
                Counter(
                    item["final_action"]
                    for item in output if item["truth"] == truth
                )
            )
            for truth in ("safe", "L1", "blood_positive")
        },
        "review_failures": sum(
            bool(item["visual_review"])
            and item["visual_review"].get("status") != "ok"
            for item in output
        ),
    }
    result = {"summary": summary, "rows": output}
    Path(args.out).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
