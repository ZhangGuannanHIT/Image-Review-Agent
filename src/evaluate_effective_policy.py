# -*- coding: utf-8 -*-
"""Evaluate the effective-risk policy on frozen real/AI visual probes."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image

from backend.policy import DecisionPolicy, decide
from backend.scene_concepts import build_scene_evidence
from backend.text_detection import LightweightTextDetector
from src.engine import ReviewEngine
from src.hard_probe import load_manifest


def _load_scene_scores(path: str | None) -> dict[str, dict]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        row["file"]: row.get("scores", {})
        for row in payload.get("rows", [])
    }


def _quality_rows(path: str) -> list[dict]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _aggregate(rows: list[dict], key: str) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, ""))].append(row)
    return {
        name: {
            "n": len(values),
            "actions": dict(Counter(row["action"] for row in values)),
            "violations_auto_allowed": sum(
                row["is_violation"] and row["action"] == "allow"
                for row in values
            ),
        }
        for name, values in groups.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="effective risk 冻结策略评估")
    parser.add_argument("--kind", choices=["solid", "quality"], required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--scene-probe", default="")
    parser.add_argument("--config", default="config/inference.yaml")
    parser.add_argument("--backend-config", default="backend/config.yaml")
    parser.add_argument(
        "--ignore-text",
        action="store_true",
        help="visual-policy ablation: do not let unimplemented OCR block a decision",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    backend = yaml.safe_load(
        Path(args.backend_config).read_text(encoding="utf-8")
    ) or {}
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
        anime_style_router_dir=(
            backend.get("model", {}).get("anime_style_router_dir", "")
        ),
        anime_style_min_score=float(
            backend.get("model", {}).get("anime_style_min_score", 0.90)
        ),
    )
    policy_config = backend.get("decision_policy") or {}
    policy = DecisionPolicy(**policy_config)
    scene_options = backend.get("scene_concepts") or {}
    scene_options_path = Path(
        scene_options.get("config_path", "config/scene_concepts.yaml")
    )
    scene_config = yaml.safe_load(
        scene_options_path.read_text(encoding="utf-8")
    ) or {}
    scene_scores = _load_scene_scores(args.scene_probe)
    detector = LightweightTextDetector()

    if args.kind == "solid":
        manifest_rows = [
            {
                "file": item.file,
                "domain": item.domain,
                "is_violation": item.has_blood,
                "truth": str(item.has_blood).lower(),
                "style": item.domain,
            }
            for item in load_manifest(args.manifest, args.image_root)
        ]
    else:
        manifest_rows = []
        for item in _quality_rows(args.manifest):
            manifest_rows.append({
                "file": item["file"],
                "domain": item["origin"],
                "is_violation": item["label"] == "violation",
                "truth": item["label"],
                "style": item["style"],
            })
    if args.limit > 0:
        manifest_rows = manifest_rows[:args.limit]

    rows = []
    image_root = Path(args.image_root)
    for index, item in enumerate(manifest_rows, start=1):
        image_path = image_root / item["file"]
        image_bytes = image_path.read_bytes()
        with Image.open(image_path) as opened:
            output = engine.predict(opened.convert("RGB"))
        evidence = output.to_dict()
        scores = scene_scores.get(item["file"])
        if scores:
            evidence["scene_concepts"] = build_scene_evidence(
                {"concepts": scores},
                risk_margin_threshold=float(
                    scene_config.get("risk_margin_threshold", -0.001031)
                ),
                safe_context_threshold=float(
                    scene_config.get("safe_context_threshold", 0.006)
                ),
                confounder_threshold=float(
                    scene_config.get("confounder_threshold", 0.006)
                ),
                anime_score_threshold=float(
                    scene_config.get("anime_score_threshold", 0.015)
                ),
                domain_margin_threshold=float(
                    scene_config.get("domain_margin_threshold", 0.01)
                ),
                top_k=int(scene_config.get("top_k", 3)),
            )
        evidence["text_detection"] = (
            {"status": "empty", "has_text": False}
            if args.ignore_text
            else detector.detect(image_bytes)
        )
        evidence["text_evidence"] = {
            "status": "not_available",
            "high_risk_rule_ids": [],
            "safe_context_rule_ids": [],
        }
        decision = decide(evidence, policy)
        rows.append({
            **item,
            "risk": output.risk_label,
            "injury": output.injury["label"],
            "risk_raw": decision["risk_raw"],
            "effective_risk": decision["effective_risk"],
            "binary_state": decision["binary_state"],
            "binary_score": (output.blood_binary or {}).get("p_blood"),
            "action": decision["action"],
            "conflict_type": decision["conflict_type"],
            "reason": decision["reason"],
            "domain_prompt": (
                decision.get("effective_risk_candidate", {}).get("anime_matched")
            ),
            "text_detected": evidence["text_detection"].get("has_text"),
            "scene_scores": scores,
            "evidence": evidence,
        })
        if index % 10 == 0:
            print(f"effective-policy: {index}/{len(manifest_rows)}", flush=True)

    report = {
        "kind": args.kind,
        "manifest": args.manifest,
        "scene_probe": args.scene_probe,
        "n": len(rows),
        "metrics": {
            "all": {
                "n": len(rows),
                "actions": dict(Counter(row["action"] for row in rows)),
                "violations_auto_allowed": sum(
                    row["is_violation"] and row["action"] == "allow"
                    for row in rows
                ),
            },
            "by_domain": _aggregate(rows, "domain"),
            "by_style": _aggregate(rows, "style"),
            "by_truth": _aggregate(rows, "truth"),
        },
        "rows": rows,
    }
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
