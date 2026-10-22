# -*- coding: utf-8 -*-
"""Merge Label Studio injury severity annotations into a hard-probe manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from urllib.parse import unquote

from .labels import INJURY_CLASSES


def annotation_choices(task: dict) -> list[str]:
    annotation = next(
        (item for item in task.get("annotations", []) if not item.get("was_cancelled")),
        None,
    )
    if not annotation:
        return []
    choices = []
    for result in annotation.get("result", []):
        if result.get("from_name") != "injury_detail":
            continue
        values = (result.get("value") or {}).get("choices") or []
        choices.extend(str(value).strip() for value in values)
    return list(dict.fromkeys(choices))


def merge_manifest(
    manifest_path: str | Path,
    export_path: str | Path,
    output_path: str | Path,
) -> dict:
    manifest = Path(manifest_path)
    with manifest.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    with Path(export_path).open("r", encoding="utf-8") as file:
        tasks = json.load(file)

    known_names = {
        Path(str(row["file"]).replace("\\", "/")).name for row in rows
    }
    truth_by_name: dict[str, str] = {}
    for task in tasks:
        candidates = [
            str(task.get("file_upload") or "").strip(),
            str((task.get("data") or {}).get("image") or "").strip(),
        ]
        matched_name = ""
        for candidate in candidates:
            candidate = unquote(candidate).replace("\\", "/")
            basename = Path(candidate).name
            if basename in known_names:
                matched_name = basename
                break
            matched_name = next(
                (name for name in known_names if name in candidate),
                "",
            )
            if matched_name:
                break
        choices = annotation_choices(task)
        if not matched_name or not choices:
            continue
        if len(choices) != 1 or choices[0] not in INJURY_CLASSES[1:]:
            raise ValueError(f"{matched_name} 的 injury_detail 必须是单个有效三级标签: {choices}")
        truth_by_name[matched_name] = choices[0]

    updated = 0
    missing = []
    for row in rows:
        if str(row.get("has_blood", "")).strip().lower() not in {"1", "true", "yes"}:
            continue
        name = Path(str(row["file"]).replace("\\", "/")).name
        severity = truth_by_name.get(name)
        if not severity:
            missing.append(name)
            continue
        row["severity_truth"] = severity
        updated += 1

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {"updated": updated, "missing": missing, "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description="合并 Label Studio severity 标注")
    parser.add_argument("--manifest", default="data/hard_probe/manifest.csv")
    parser.add_argument("--export", required=True)
    parser.add_argument("--out", default="data/hard_probe/manifest.csv")
    args = parser.parse_args()
    result = merge_manifest(args.manifest, args.export, args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
