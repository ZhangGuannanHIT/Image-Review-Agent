# -*- coding: utf-8 -*-
"""Descriptive calibration analysis for anime-safe and AI-blood probes."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def load(path: str) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["rows"]


def quantiles(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    values = sorted(values)
    return {
        "n": len(values),
        "min": round(values[0], 6),
        "p10": round(values[int((len(values) - 1) * 0.1)], 6),
        "median": round(statistics.median(values), 6),
        "p90": round(values[int((len(values) - 1) * 0.9)], 6),
        "max": round(values[-1], 6),
    }


def features(row: dict) -> dict:
    evidence = row.get("evidence", {})
    binary = evidence.get("blood_binary") or {}
    injury = evidence.get("injury") or {}
    type_scores = (evidence.get("type") or {}).get("scores", {})
    concepts = row.get("scene_scores", {})
    return {
        "domain_score": float(
            (concepts.get("anime_illustration") or {}).get("score", -1)
        ),
        "real_score": float(
            (concepts.get("real_photo") or {}).get("score", -1)
        ),
        "binary": float(binary.get("p_blood", 0)),
        "injury_label": injury.get("label", ""),
        "injury_max": max(
            [float(value) for value in injury.get("scores", {}).values()] or [0]
        ),
        "gore": float((concepts.get("gore_injury") or {}).get("score", -1)),
        "direct": float(type_scores.get("直接暴力（攻击进行中）", 0)),
        "threat": float(type_scores.get("暗示/威胁（武器指向/捆绑）", 0)),
        "aftermath": float(type_scores.get("后果展示（尸体/伤口）", 0)),
    }


def summarize(label: str, rows: list[dict]) -> dict:
    values = [features(row) for row in rows]
    return {
        "label": label,
        "n": len(rows),
        "risk": dict(Counter(row.get("risk") or row.get("risk_raw") for row in rows)),
        "actions": dict(Counter(row["action"] for row in rows)),
        "domain": quantiles([item["domain_score"] for item in values]),
        "binary": quantiles([item["binary"] for item in values]),
        "gore": quantiles([item["gore"] for item in values]),
        "direct": quantiles([item["direct"] for item in values]),
        "threat": quantiles([item["threat"] for item in values]),
        "injury": dict(Counter(item["injury_label"] for item in values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="动漫策略联合校准分析")
    parser.add_argument(
        "--safe", default="data/anime_safe/evaluation_v5_initial.json"
    )
    parser.add_argument(
        "--solid",
        default="eval_results/solid_effective_v5_visual_full.json",
    )
    parser.add_argument("--out", default="data/anime_safe/calibration_analysis.json")
    args = parser.parse_args()
    safe = [row for row in load(args.safe) if row["split"] == "dev"]
    solid = load(args.solid)
    ai_positive = [
        row for row in solid
        if row["domain"] == "ai" and row["is_violation"]
    ]
    ai_negative = [
        row for row in solid
        if row["domain"] == "ai" and not row["is_violation"]
    ]
    report = {
        "presumed_safe_anime_dev": summarize(
            "presumed_safe_anime_dev", safe
        ),
        "solid_ai_blood": summarize("solid_ai_blood", ai_positive),
        "solid_ai_no_blood": summarize("solid_ai_no_blood", ai_negative),
        "safe_by_action": {
            action: summarize(action, [row for row in safe if row["action"] == action])
            for action in ("allow", "limit", "review")
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
