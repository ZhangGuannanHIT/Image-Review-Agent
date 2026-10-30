# -*- coding: utf-8 -*-
"""Compare visual-review image sizes without changing the prompt or schema."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from backend.anime_vision import AnimeVisualReviewer
from backend.llm import LLMClient


FIELDS = (
    "injury",
    "attack",
    "threat",
    "weapon_visible",
    "context",
    "appearance",
    "threat_basis",
)


def _model(config_path: str, backend_path: str):
    import yaml

    backend = yaml.safe_load(Path(backend_path).read_text(encoding="utf-8")) or {}
    cfg = backend.get("anime_vision_review") or {}
    return LLMClient.from_config(config_path).build_model(
        max_retries=int(cfg.get("max_retries", 1)),
        max_tokens=int(cfg.get("max_tokens", 512)),
        response_format={"type": "json_object"} if cfg.get("json_response", True) else None,
        disable_thinking=True,
    )


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    both_ok = [
        row for row in rows
        if row["sizes"]["1280"]["status"] == "ok"
        and row["sizes"]["960"]["status"] == "ok"
    ]
    field_agreement = {
        field: sum(
            row["sizes"]["1280"].get(field) == row["sizes"]["960"].get(field)
            for row in both_ok
        )
        for field in FIELDS
    }
    statuses = Counter(
        values["status"]
        for row in rows
        for values in row["sizes"].values()
    )
    elapsed = {
        size: round(
            sum(row["sizes"][size]["elapsed_ms"] for row in rows)
            / max(1, len(rows)),
            1,
        )
        for size in ("1280", "960")
    }
    return {
        "n": len(rows),
        "both_ok": len(both_ok),
        "status_counts": dict(statuses),
        "field_agreement": field_agreement,
        "mean_elapsed_ms": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="对比虚拟视觉输入分辨率")
    parser.add_argument(
        "--samples",
        default="data/anime_safe/visual_sample_rescored.json",
    )
    parser.add_argument("--llm-config", default="config/llm.yaml")
    parser.add_argument("--backend-config", default="backend/config.yaml")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument(
        "--out",
        default="eval_results/anime_vision_param_compare.json",
    )
    args = parser.parse_args()

    sample = json.loads(Path(args.samples).read_text(encoding="utf-8"))
    rows = sample.get("rows", [])[:args.limit]
    reviewer = AnimeVisualReviewer(_model(args.llm_config, args.backend_config))
    output = []
    for index, item in enumerate(rows, start=1):
        path = Path(item["path"])
        payload = path.read_bytes()
        sizes = {}
        for size in (1280, 960):
            reviewer.max_side = size
            result = reviewer.review(payload)
            sizes[str(size)] = {
                key: result.get(key)
                for key in ("status", *FIELDS, "elapsed_ms", "error")
            }
        output.append({
            "path": item["path"],
            "truth": item.get("truth"),
            "sizes": sizes,
        })
        print(f"visual-params: {index}/{len(rows)}", flush=True)

    report = {"summary": _summary(output), "rows": output}
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
