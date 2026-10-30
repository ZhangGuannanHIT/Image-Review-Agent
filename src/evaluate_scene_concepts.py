# -*- coding: utf-8 -*-
"""Evaluate the frozen open-vocabulary scene concept branch."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image

from backend.scene_concepts import SceneConceptScorer
from .engine import ReviewEngine
from .hard_probe import binary_metrics


def load_rows(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def truth_value(row: dict) -> bool:
    if row.get("label"):
        return row["label"] == "violation"
    if row.get("has_blood"):
        return str(row["has_blood"]).strip().lower() in {"1", "true", "yes"}
    raise ValueError(f"manifest 行缺少 label/has_blood: {row.get('file')}")


def summarize(rows: list[dict]) -> dict:
    labels = [row["truth"] for row in rows]
    return {
        "count": len(rows),
        "positive": sum(labels),
        "negative": len(labels) - sum(labels),
        "risk_margin": binary_metrics(
            labels, [row["risk_margin"] for row in rows]
        ),
        "risk_max": binary_metrics(labels, [row["risk_max"] for row in rows]),
        "gore_injury": binary_metrics(
            labels, [row["scores"]["gore_injury"]["score"] for row in rows]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="场景概念分支冻结评估")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    engine = ReviewEngine(
        "fg_clip",
        ckpt_dir="checkpoints/best",
        device=args.device,
        lora=True,
        strict_load=True,
    )
    scorer = SceneConceptScorer(engine)
    manifest = load_rows(args.manifest)
    rows = []
    for index, item in enumerate(manifest, start=1):
        with Image.open(Path(args.image_root) / item["file"]) as opened:
            result = scorer.score(opened.convert("RGB"))
        scores = result["concepts"]
        risk_max = max(
            value["score"] for value in scores.values() if value["kind"] == "risk"
        )
        safe_max = max(
            value["score"] for value in scores.values()
            if value["kind"] == "safe_context"
        )
        confounder_max = max(
            value["score"] for value in scores.values()
            if value["kind"] == "confounder"
        )
        rows.append({
            **item,
            "truth": truth_value(item),
            "scores": scores,
            "top": result["top"],
            "risk_max": risk_max,
            "safe_max": safe_max,
            "confounder_max": confounder_max,
            "risk_margin": risk_max - max(safe_max, confounder_max),
        })
        if index % 20 == 0:
            print(f"scene-concepts: {index}/{len(manifest)}", flush=True)
    report = {
        "manifest": args.manifest,
        "image_root": args.image_root,
        "n": len(rows),
        "metrics": summarize(rows),
        "rows": rows,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
