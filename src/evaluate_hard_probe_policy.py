# -*- coding: utf-8 -*-
"""Evaluate the deployed evidence and policy on a frozen hard probe."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image

from backend.policy import DecisionPolicy, decide, render
from .engine import ReviewEngine
from .hard_probe import load_manifest


def aggregate(rows: list[dict], key: str | None = None) -> dict:
    groups = {"all": rows}
    if key:
        for row in rows:
            groups.setdefault(str(row[key]), []).append(row)
    output = {}
    for name, values in groups.items():
        output[name] = {
            "n": len(values),
            "has_blood": sum(row["has_blood"] for row in values),
            "actions": dict(Counter(row["action"] for row in values)),
            "uncertain": sum(row["uncertain"] for row in values),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="冻结硬探针完整策略评估")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--config", default="config/inference.yaml")
    parser.add_argument("--backend-config", default="backend/config.yaml")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    with open(args.backend_config, "r", encoding="utf-8") as file:
        backend_config = yaml.safe_load(file)
    engine = ReviewEngine(
        model_path=config["model_path"],
        ckpt_dir=config["ckpt_dir"],
        device=config.get("device", "cuda"),
        lora=bool(config.get("lora", True)),
        strict_load=bool(config.get("strict_load", True)),
        type_thresholds=config.get("type_thresholds"),
        quality_thresholds=config.get("quality_thresholds"),
        binary_injury_dir=config.get("binary_injury_dir", ""),
        quality_fusion_dir=config.get("quality_fusion_dir", ""),
    )
    policy = DecisionPolicy(**(backend_config.get("decision_policy") or {}))
    rows = []
    items = load_manifest(args.manifest, args.image_root)
    for index, item in enumerate(items, start=1):
        with Image.open(item.path) as opened:
            output = engine.predict(opened.convert("RGB"))
        evidence = output.to_dict()
        decision = decide(evidence, policy)
        rows.append(
            {
                "file": item.file,
                "folder": item.folder,
                "domain": item.domain,
                "has_blood": item.has_blood,
                "severity_truth": item.severity_truth,
                "risk": output.risk_label,
                "risk_confidence": float(output.risk.get("confidence", 0.0)),
                "injury": output.injury["label"],
                "blood_binary": output.blood_binary,
                "quality": output.quality,
                "action": decision["action"],
                "uncertain": decision["uncertain"],
                "reason": decision["reason"],
                "suggestion": render(decision),
            }
        )
        if index % 20 == 0:
            print(f"policy-probe: {index}/{len(items)}", flush=True)
    report = {
        "manifest": args.manifest,
        "image_root": args.image_root,
        "n": len(rows),
        "metrics": {
            "all": aggregate(rows)["all"],
            "by_domain": aggregate(rows, "domain"),
            "by_severity": aggregate(rows, "severity_truth"),
            "by_blood": aggregate(rows, "has_blood"),
        },
        "rows": rows,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
