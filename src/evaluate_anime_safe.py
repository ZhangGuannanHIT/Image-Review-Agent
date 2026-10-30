# -*- coding: utf-8 -*-
"""Evaluate the effective-risk policy on an external safe-anime set."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image, ImageOps

from backend.policy import DecisionPolicy, decide
from backend.scene_concepts import SceneConceptScorer, build_scene_evidence
from backend.text_detection import LightweightTextDetector
from backend.anime_vision_policy import anime_visual_gate
from src.engine import ReviewEngine


def _load_rows(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        return [
            row for row in csv.DictReader(file)
            if str(row.get("selected", "1")) == "1"
        ]


def _aggregate(rows: list[dict], key: str) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, ""))].append(row)
    return {
        name: {
            "n": len(values),
            "actions": dict(Counter(row["action"] for row in values)),
            "review_rate": round(
                sum(row["action"] == "review" for row in values) / len(values),
                6,
            ),
            "unsafe_action_rate": round(
                sum(row["action"] in {"limit", "block"} for row in values)
                / len(values),
                6,
            ),
            "confirmed_violations": sum(
                row["confirmed_violation"] for row in values
            ),
            "confirmed_violations_auto_allowed": sum(
                row["confirmed_violation"] and row["action"] == "allow"
                for row in values
            ),
            "domain_matched": sum(
                bool(row["domain_matched"]) for row in values
            ),
            "text_detected": sum(
                bool(row["text_detected"]) for row in values
            ),
            "anime_visual_triggers": sum(
                bool(row.get("anime_visual_trigger")) for row in values
            ),
        }
        for name, values in groups.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="安全二次元 effective-risk 评估")
    parser.add_argument(
        "--manifest", default="data/anime_safe/manifest.csv"
    )
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--config", default="config/inference.yaml")
    parser.add_argument("--backend-config", default="backend/config.yaml")
    parser.add_argument("--scene-config", default="config/scene_concepts.yaml")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--use-text-gate",
        action="store_true",
        help="let detected text without OCR force review",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    backend = yaml.safe_load(
        Path(args.backend_config).read_text(encoding="utf-8")
    ) or {}
    scene_config = yaml.safe_load(
        Path(args.scene_config).read_text(encoding="utf-8")
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
    scene_scorer = SceneConceptScorer(
        engine, scene_config["prompts_path"]
    )
    policy = DecisionPolicy(**(backend.get("decision_policy") or {}))
    anime_visual_cfg = backend.get("anime_vision_review") or {}
    detector = LightweightTextDetector()
    rows = _load_rows(args.manifest)
    if args.limit > 0:
        rows = rows[:args.limit]

    output_rows = []
    image_root = Path(args.image_root)
    for index, item in enumerate(rows, start=1):
        path = image_root / item["relative_path"]
        payload = path.read_bytes()
        detection = detector.detect(payload)
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            model_output = engine.predict(image)
            scene_raw = scene_scorer.score(image)
        evidence = model_output.to_dict()
        evidence["scene_concepts"] = build_scene_evidence(
            scene_raw,
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
            detection if args.use_text_gate
            else {"status": "empty", "has_text": False}
        )
        evidence["text_evidence"] = {
            "status": "not_available",
            "high_risk_rule_ids": [],
            "safe_context_rule_ids": [],
        }
        decision = decide(evidence, policy)
        visual_gate = anime_visual_gate(
            decision,
            evidence,
            min_style_score=float(
                anime_visual_cfg.get("min_style_score", 0.90)
            ),
            allow_safety_review=bool(
                anime_visual_cfg.get("allow_safety_review", True)
            ),
        )
        scene = evidence["scene_concepts"]
        output_rows.append({
            **item,
            "human_label": item.get("human_label", ""),
            "human_verified": str(item.get("human_verified", "0")) == "1",
            "confirmed_violation": bool(
                str(item.get("human_verified", "0")) == "1"
                and item.get("human_label") in {
                    "L1_轻微违规", "L2_中度违规", "L3_严重违规"
                }
            ),
            "risk": model_output.risk_label,
            "risk_confidence": model_output.risk.get("confidence"),
            "binary_state": decision["binary_state"],
            "binary_score": (model_output.blood_binary or {}).get("p_blood"),
            "quality_labels": model_output.quality_labels,
            "effective_risk": decision["effective_risk"],
            "action": decision["action"],
            "conflict_type": decision["conflict_type"],
            "adjustments": decision["adjustments"],
            "reason": decision["reason"],
            "domain_matched": bool(
                decision.get("effective_risk_candidate", {}).get(
                    "anime_matched"
                )
            ),
            "domain_source": decision.get(
                "effective_risk_candidate", {}
            ).get("anime_source"),
            "style_router_score": (
                (evidence.get("anime_style") or {}).get("score")
            ),
            "domain_score": scene.get("domain", {}).get("anime_score"),
            "domain_margin": scene.get("domain", {}).get("anime_margin"),
            "fiction_score": (
                scene_raw.get("concepts", {})
                .get("fiction_context", {})
                .get("score")
            ),
            "text_detected": bool(detection.get("has_text")),
            "anime_visual_trigger": bool(visual_gate["eligible"]),
            "anime_visual_gate": visual_gate,
            "text_detection": detection,
            "scene_scores": scene_raw.get("concepts", {}),
            "evidence": evidence,
        })
        if index % 10 == 0:
            print(f"anime-safe: {index}/{len(rows)}", flush=True)

    report = {
        "manifest": args.manifest,
        "image_root": args.image_root,
        "n": len(output_rows),
        "use_text_gate": args.use_text_gate,
        "metrics": {
            "all": {
                "n": len(output_rows),
                "actions": dict(Counter(row["action"] for row in output_rows)),
                "review_rate": round(
                    sum(row["action"] == "review" for row in output_rows)
                    / max(len(output_rows), 1),
                    6,
                ),
                "unsafe_action_rate": round(
                    sum(row["action"] in {"limit", "block"} for row in output_rows)
                    / max(len(output_rows), 1),
                    6,
                ),
                "domain_matched": sum(
                    bool(row["domain_matched"]) for row in output_rows
                ),
                "text_detected": sum(
                    bool(row["text_detected"]) for row in output_rows
                ),
                "anime_visual_triggers": sum(
                    bool(row["anime_visual_trigger"]) for row in output_rows
                ),
                "confirmed_violations": sum(
                    row["confirmed_violation"] for row in output_rows
                ),
                "confirmed_violations_auto_allowed": sum(
                    row["confirmed_violation"] and row["action"] == "allow"
                    for row in output_rows
                ),
            },
            "by_split": _aggregate(output_rows, "split"),
            "by_category": _aggregate(output_rows, "category"),
            "by_risk": _aggregate(output_rows, "risk"),
        },
        "rows": output_rows,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
