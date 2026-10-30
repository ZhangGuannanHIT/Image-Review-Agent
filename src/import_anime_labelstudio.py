# -*- coding: utf-8 -*-
"""Import Label Studio anime annotations back to original image paths."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def flatten_taxonomy(value) -> list[str]:
    output: list[str] = []
    if isinstance(value, list):
        for item in value:
            output.extend(flatten_taxonomy(item))
    elif isinstance(value, str) and value.strip():
        output.append(value.strip())
    return output


def field_values(result: list[dict], name: str) -> list[str]:
    for item in result:
        if item.get("from_name") != name:
            continue
        value = item.get("value", {})
        return (
            flatten_taxonomy(value.get("taxonomy", []))
            or list(value.get("choices", []) or [])
            or [str(text) for text in value.get("text", [])]
        )
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="导入二次元 Label Studio 标注")
    parser.add_argument(
        "--export",
        default=(
            "D:/动漫图片_待标注_2026-10-27/"
            "project-8-at-2026-10-27-11-52-f839b93e.json"
        ),
    )
    parser.add_argument(
        "--batch-manifest",
        default="D:/动漫图片_待标注_2026-10-27/annotation_manifest.csv",
    )
    parser.add_argument(
        "--existing-labels", default="data/anime_safe/human_labels.csv"
    )
    parser.add_argument(
        "--corrections", default="data/anime_safe/human_label_corrections.csv"
    )
    parser.add_argument(
        "--annotations-out",
        default="data/anime_safe/human_annotations_2026-10-27.csv",
    )
    parser.add_argument("--report", default="data/anime_safe/import_report.json")
    args = parser.parse_args()

    export = json.loads(Path(args.export).read_text(encoding="utf-8"))
    with Path(args.batch_manifest).open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        batch = {
            row["batch_filename"]: row for row in csv.DictReader(file)
        }
    with Path(args.existing_labels).open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        existing = {
            row["relative_path"]: row for row in csv.DictReader(file)
        }
    corrections_path = Path(args.corrections)
    corrections = {}
    if corrections_path.is_file():
        with corrections_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as file:
            corrections = {
                row["relative_path"]: row for row in csv.DictReader(file)
            }

    rows = []
    conflicts = []
    errors = []
    for task in export:
        upload = str(task.get("file_upload", ""))
        matched = next(
            (name for name in batch if upload.endswith(name)),
            "",
        )
        if not matched:
            errors.append({"task": task.get("id"), "error": "batch_filename_not_found"})
            continue
        annotations = task.get("annotations", [])
        if len(annotations) != 1 or annotations[0].get("was_cancelled"):
            errors.append({"task": task.get("id"), "error": "annotation_missing_or_cancelled"})
            continue
        annotation = annotations[0]
        result = annotation.get("result", [])
        risk = field_values(result, "risk_level")
        injury = field_values(result, "injury_detail")
        if len(risk) != 1 or len(injury) != 1:
            errors.append({"task": task.get("id"), "error": "single_label_invalid"})
            continue
        batch_row = batch[matched]
        note = " ".join(field_values(result, "note")).strip()
        safe_note = any(
            phrase in note
            for phrase in ("无违规场景", "安全", "可接受")
        )
        label_status = (
            "label_conflict_note"
            if risk[0] != "L0_安全" and safe_note
            else "confirmed"
        )
        row = {
            "relative_path": batch_row["original_relative_path"],
            "batch_filename": matched,
            "human_label": risk[0],
            "violence_type": json.dumps(
                field_values(result, "violence_type"), ensure_ascii=False
            ),
            "injury_detail": injury[0],
            "quality": json.dumps(
                field_values(result, "quality"), ensure_ascii=False
            ),
            "style_domain": "|".join(field_values(result, "style_domain")),
            "scene_context": json.dumps(
                field_values(result, "scene_context"), ensure_ascii=False
            ),
            "human_note": note,
            "reviewer": "label_studio",
            "reviewed_at": annotation.get("updated_at", ""),
            "label_status": label_status,
        }
        correction = corrections.get(row["relative_path"])
        if correction:
            row["human_label"] = correction["corrected_label"]
            row["label_status"] = "confirmed"
            row["human_note"] = (
                row["human_note"] + "；" + correction.get("correction_note", "")
            ).strip("；")
            conflicts = [
                item for item in conflicts
                if item["relative_path"] != row["relative_path"]
            ]
        rows.append(row)
        if row["label_status"] == "label_conflict_note":
            conflicts.append({
                "relative_path": row["relative_path"],
                "selected_risk": row["human_label"],
                "note": note,
                "suggested_by_note": "L0_安全",
            })

    if errors:
        raise RuntimeError(f"标注导入失败: {errors[:5]}")
    if len(rows) != len(batch):
        raise RuntimeError(f"映射数量不一致: annotations={len(rows)} batch={len(batch)}")

    fieldnames = list(rows[0])
    annotations_out = Path(args.annotations_out)
    annotations_out.parent.mkdir(parents=True, exist_ok=True)
    with annotations_out.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    merged = dict(existing)
    for row in rows:
        merged[row["relative_path"]] = row
    merged_rows = list(merged.values())
    merged_fields = []
    for row in merged_rows:
        for key in row:
            if key not in merged_fields:
                merged_fields.append(key)
    with Path(args.existing_labels).open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=merged_fields)
        writer.writeheader()
        writer.writerows(merged_rows)

    confusion = Counter(
        (
            next(
                row["model_risk"]
                for row in batch.values()
                if row["original_relative_path"] == annotation["relative_path"]
            ),
            annotation["human_label"],
        )
        for annotation in rows
    )
    report = {
        "export": args.export,
        "n_tasks": len(export),
        "n_imported": len(rows),
        "errors": errors,
        "risk_counts": dict(Counter(row["human_label"] for row in rows)),
        "style_counts": dict(Counter(row["style_domain"] for row in rows)),
        "model_risk_to_human_risk": {
            f"{model}->{human}": count
            for (model, human), count in sorted(confusion.items())
        },
        "merged_human_labels": len(merged_rows),
        "label_conflicts": conflicts,
        "confirmed_rows_for_training": sum(
            row["label_status"] == "confirmed" for row in rows
        ),
    }
    report_path = Path(args.report)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
