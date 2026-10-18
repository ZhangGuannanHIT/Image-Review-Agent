# -*- coding: utf-8 -*-
"""数据集构建：Label Studio 导出 JSON → 可直接训练的干净数据集 + 清洗报告。

三步走，规则全部是确定性的，改标签的只有第一步的配对和第三步的矛盾剔除：

1. 配对：按 file_upload / data.image 的文件名与图片目录一一对应，缺图任务剔除；
2. 标签头对齐：只保留 risk_level / violence_type / injury_detail / quality 四个模型输出头，
   取值归一化为 src/labels.py 的规范常量；原 target（敏感对象）与 note（备注）不进入训练文件，
   另存 <out>/review_extras.json 备查；
3. 矛盾剔除：违规等级与暴力/血腥表现互相矛盾的样本直接删除，其余语义张力只进复核清单。

注意：quality 是多标签（标注里存在"严重模糊+文字扭曲形变"这类组合），不要用单标签口径取第一个。

用法（从仓库根目录执行）：
    python -m src.build_dataset \
        --json /path/to/project-6-export.json \
        --image-root /path/to/images \
        --out data/clean
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from .dataset import _encode_multihot, _encode_single, _parse_annotation
from .labels import INJURY_CLASSES, QUALITY_CLASSES, RISK_CLASSES, TYPE_CLASSES

HEAD_FIELDS = ("risk_level", "violence_type", "injury_detail", "quality")
BASE_FIELDS = ("target", "note")

NO_VIOLENCE = "无暴力"
DIRECT_VIOLENCE = "直接暴力（攻击进行中）"
AFTERMATH = "后果展示（尸体/伤口）"
THREAT = "暗示/威胁（武器指向/捆绑）"
NO_BLOOD = "无血腥"
UNCLEAR = "X_无法判定"
READABLE = "清晰"
HARD_TO_READ = ("严重模糊", "文字扭曲形变")

# 确定性删除规则：违规等级与表现字段互相排斥才算"明显错误"。
# 字典顺序即优先级：一条样本命中多条时，按第一个命中的规则归因。
DELETION_RULES = {
    "C1_type_self_conflict": "violence_type 同时选了「无暴力」和「直接暴力（攻击进行中）」",
    "C2_safe_but_violence": "风险等级 L0_安全，却标注直接暴力/后果展示",
    "C3_safe_but_blood": "风险等级 L0_安全，却标注存在血腥",
    "C5_safe_but_threat": "风险等级 L0_安全，却标注暗示/威胁（武器指向/捆绑）",
    "C4_risk_without_evidence": "风险等级 L1/L2/L3，却标注无暴力且无血腥",
    "C6_risk_without_evidence_field": "风险等级 L1/L2/L3，却缺失暴力类型或血腥程度标注",
}


def _canon_single(values: list[str], classes: list[str]) -> tuple[Optional[str], str]:
    index, unmapped = _encode_single(values, classes)
    return (classes[index] if index is not None else None), unmapped


def _canon_multi(values: list[str], classes: list[str]) -> tuple[list[str], str]:
    mask, unmapped = _encode_multihot(values, classes)
    return [label for label, flag in zip(classes, mask) if flag], unmapped


def _pick_annotation(task: dict) -> Optional[dict]:
    """取第一条未取消的标注（与 src/dataset.py 的训练口径一致）。"""
    annotations = task.get("annotations") or []
    return next((a for a in annotations if not a.get("was_cancelled")), None)


def audit(risk: Optional[str], types: set[str], injury: Optional[str],
          quality: set[str]) -> tuple[list[str], list[str]]:
    """返回 (命中的确定性删除规则, 人工复核原因)，删除规则按 DELETION_RULES 优先级排列。"""
    matched: set[str] = set()
    review: list[str] = []

    if NO_VIOLENCE in types and DIRECT_VIOLENCE in types:
        matched.add("C1_type_self_conflict")
    if risk and risk.startswith("L0") and (types & {DIRECT_VIOLENCE, AFTERMATH}):
        matched.add("C2_safe_but_violence")
    if risk and risk.startswith("L0") and injury and injury != NO_BLOOD:
        matched.add("C3_safe_but_blood")
    if risk and risk.startswith("L0") and THREAT in types:
        matched.add("C5_safe_but_threat")
    if (risk and risk.startswith(("L1", "L2", "L3"))
            and types == {NO_VIOLENCE} and injury == NO_BLOOD):
        matched.add("C4_risk_without_evidence")
    if risk and risk.startswith(("L1", "L2", "L3")) and (not types or injury is None):
        matched.add("C6_risk_without_evidence_field")

    # 以下只进复核清单，不动标签。
    if risk and risk.startswith("L0") and (quality & set(HARD_TO_READ)):
        review.append("R3_safe_but_unreadable")
    if risk == UNCLEAR and READABLE in quality:
        review.append("R4_unclear_but_sharp")
    deleted = [rule for rule in DELETION_RULES if rule in matched]
    return deleted, review


def _to_result(from_name: str, values: list[str], field_type: str) -> dict[str, Any]:
    """按原 Label Studio 结构回写一个控件结果。"""
    if field_type == "choices":
        value: dict[str, Any] = {"choices": values}
    else:
        value = {"taxonomy": [[v] for v in values]}
    return {"from_name": from_name, "to_name": "img", "type": field_type, "value": value}


def build(json_path: str, image_root: str, out_dir: str, copy_images: bool = True) -> dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as file:
        tasks = json.load(file)

    image_root = str(image_root)
    image_dir = Path(out_dir) / "images"
    if copy_images:
        image_dir.mkdir(parents=True, exist_ok=True)

    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    extras: dict[str, Any] = {}
    unmapped: Counter = Counter()
    dist_before: dict[str, Counter] = {field: Counter() for field in HEAD_FIELDS}
    dist_after: dict[str, Counter] = {field: Counter() for field in HEAD_FIELDS}
    missing_fields: Counter = Counter()
    quality_multi = 0

    for task in tasks:
        task_id = task.get("id")
        file_upload = task.get("file_upload")
        image_uri = (task.get("data") or {}).get("image", "")
        basename = os.path.basename(file_upload or image_uri or "")
        source = os.path.join(image_root, basename)
        annotation = _pick_annotation(task)
        if annotation is None:
            removed.append({"task_id": task_id, "image": basename,
                            "reasons": ["C0_no_valid_annotation"],
                            "note": "标注全部取消或缺失，无法用于训练"})
            continue
        if not os.path.isfile(source):
            removed.append({"task_id": task_id, "image": basename,
                            "reasons": ["P0_image_missing"],
                            "note": f"图片不存在: {source}"})
            continue

        parsed = _parse_annotation(annotation)
        risk, risk_bad = _canon_single(parsed.get("risk_level", []), RISK_CLASSES)
        types_list, type_bad = _canon_multi(parsed.get("violence_type", []), TYPE_CLASSES)
        injury, injury_bad = _canon_single(parsed.get("injury_detail", []), INJURY_CLASSES)
        quality_list, quality_bad = _canon_multi(parsed.get("quality", []), QUALITY_CLASSES)
        for field, raw, bad in (("risk_level", parsed.get("risk_level", []), risk_bad),
                                ("violence_type", parsed.get("violence_type", []), type_bad),
                                ("injury_detail", parsed.get("injury_detail", []), injury_bad),
                                ("quality", parsed.get("quality", []), quality_bad)):
            if raw and bad:
                unmapped[f"{field}:{bad}"] += 1

        for field, values in (("risk_level", [risk]), ("violence_type", types_list),
                              ("injury_detail", [injury]), ("quality", quality_list)):
            for value in values:
                if value:
                    dist_before[field][value] += 1

        deleted, needs_review = audit(risk, set(types_list), injury, set(quality_list))
        record = {
            "task_id": task_id,
            "image": basename,
            "risk_level": risk,
            "violence_type": types_list,
            "injury_detail": injury,
            "quality": quality_list,
            "target": parsed.get("target", []),
            "note": parsed.get("note", []),
            "deleted_rules": deleted,
            "review_rules": needs_review,
        }
        if deleted:
            removed.append(record)
            continue
        if needs_review:
            review.append(record)

        for field, raw in (("risk_level", parsed.get("risk_level", [])),
                           ("violence_type", parsed.get("violence_type", [])),
                           ("injury_detail", parsed.get("injury_detail", [])),
                           ("quality", parsed.get("quality", []))):
            if not raw:
                missing_fields[field] += 1

        for field, values in (("risk_level", [risk]), ("violence_type", types_list),
                              ("injury_detail", [injury]), ("quality", quality_list)):
            for value in values:
                if value:
                    dist_after[field][value] += 1
        if len(quality_list) > 1:
            quality_multi += 1

        extras[str(task_id)] = {"target": parsed.get("target", []), "note": parsed.get("note", [])}
        kept.append({
            "id": task_id,
            "file_upload": basename,
            "data": {"image": basename},
            "annotations": [{
                "was_cancelled": False,
                "result": [
                    _to_result("risk_level", [risk] if risk else [], "choices"),
                    _to_result("violence_type", types_list, "taxonomy"),
                    _to_result("injury_detail", [injury] if injury else [], "choices"),
                    _to_result("quality", quality_list, "taxonomy"),
                ],
            }],
        })
        if copy_images:
            destination = image_dir / basename
            if not destination.exists():
                shutil.copy2(source, destination)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    pruned = 0
    if copy_images and image_dir.is_dir():
        # 只清理本脚本自己产出的 images/ 目录，避免上一次构建遗留已被剔除样本的图片。
        kept_names = {task["file_upload"] for task in kept}
        for item in image_dir.iterdir():
            if item.is_file() and item.name not in kept_names:
                item.unlink()
                pruned += 1
    with open(out_path / "labels.json", "w", encoding="utf-8") as file:
        json.dump(kept, file, ensure_ascii=False, indent=2)
    with open(out_path / "review_extras.json", "w", encoding="utf-8") as file:
        json.dump(extras, file, ensure_ascii=False, indent=2)
    with open(out_path / "removed_samples.jsonl", "w", encoding="utf-8") as file:
        for record in removed:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    deleted_by_rule = Counter()
    deleted_rule_hits = Counter()
    for record in removed:
        rules = record.get("reasons") or record.get("deleted_rules") or []
        if not rules:
            continue
        deleted_by_rule[rules[0]] += 1
        for rule in rules:
            deleted_rule_hits[rule] += 1
    review_by_rule = Counter()
    for record in review:
        for rule in record["review_rules"]:
            review_by_rule[rule] += 1

    report = {
        "source_json": str(json_path),
        "source_image_root": image_root,
        "out_dir": str(out_path),
        "total_tasks": len(tasks),
        "paired_samples": len(tasks) - deleted_by_rule["P0_image_missing"],
        "kept_samples": len(kept),
        "removed_samples": len(removed),
        "pass_rate": round(len(kept) / len(tasks), 4) if tasks else 0.0,
        "task3_removed_samples": sum(1 for record in removed if record.get("deleted_rules")),
        "deleted_by_rule": dict(deleted_by_rule),
        "deleted_rule_hits": dict(deleted_rule_hits),
        "removed_ids": [record["task_id"] for record in removed],
        "removed_ids_by_rule": {
            rule: [record["task_id"] for record in removed
                   if (record.get("reasons") or record.get("deleted_rules") or [None])[0] == rule]
            for rule in list(DELETION_RULES) + ["C0_no_valid_annotation", "P0_image_missing"]
        },
        "review_by_rule": dict(review_by_rule),
        "review_samples": len(review),
        "quality_multi_label_samples": quality_multi,
        "pruned_stale_images": pruned,
        "missing_field_counts_kept": dict(missing_fields),
        "unmapped_values": dict(unmapped),
        "distribution_before": {k: dict(v) for k, v in dist_before.items()},
        "distribution_after": {k: dict(v) for k, v in dist_after.items()},
        "deletion_rule_notes": DELETION_RULES,
    }
    with open(out_path / "report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    with open(out_path / "report.md", "w", encoding="utf-8") as file:
        file.write(render_report(report))
    return report


def render_report(report: dict[str, Any]) -> str:
    lines = ["# 数据集体检与清洗报告", ""]
    lines.append(f"- 源文件：`{report['source_json']}`")
    lines.append(f"- 图片根目录：`{report['source_image_root']}`")
    lines.append(f"- 输出目录：`{report['out_dir']}`")
    lines.append("")
    lines.append("## 通过率")
    lines.append("")
    lines.append(f"- 原始任务数：{report['total_tasks']}")
    lines.append(f"- 成对样本（有图 + 有标注）：{report['paired_samples']}")
    lines.append(f"- 清洗后保留：{report['kept_samples']}")
    lines.append(f"- 删除合计：{report['removed_samples']}")
    lines.append(f"- 总体通过率：{report['pass_rate'] * 100:.2f}%")
    lines.append(f"- 其中任务三（确定性矛盾）删除：{report['task3_removed_samples']}")
    lines.append("")
    lines.append("## 删除明细")
    lines.append("")
    lines.append("| 规则 | 含义 | 删除数 |")
    lines.append("|---|---|---:|")
    for rule, note in DELETION_RULES.items():
        lines.append(f"| `{rule}` | {note} | {report['deleted_by_rule'].get(rule, 0)} |")
    extra = {k: v for k, v in report["deleted_by_rule"].items() if k not in DELETION_RULES}
    for rule, count in extra.items():
        lines.append(f"| `{rule}` | 配对/标注有效性剔除 | {count} |")
    lines.append("")
    lines.append("删除数按主因归类（一条样本命中多条规则时计入优先级最高的一条）；"
                 f"规则命中次数合计 {sum(report['deleted_rule_hits'].values())}。")
    lines.append("")
    lines.append("## 逐条删除样本")
    lines.append("")
    out_dir = report["out_dir"]
    lines.append(f"- 完整清单见 `{out_dir}/removed_samples.jsonl`"
                 "（每行含原标签、删除规则、复核规则）。")
    for rule, ids in report["removed_ids_by_rule"].items():
        if not ids:
            continue
        shown = ", ".join(str(i) for i in ids[:30])
        more = f" …（共 {len(ids)} 条）" if len(ids) > 30 else ""
        lines.append(f"- 主因为 `{rule}`：{shown}{more}")
    lines.append("")
    lines.append("一条样本可能同时命中多条规则；完整命中情况见 `removed_samples.jsonl` 的 `deleted_rules`。")
    lines.append("")
    lines.append("## 复核清单（未改动标签）")
    lines.append("")
    if report["review_by_rule"]:
        lines.append("| 复核规则 | 样本数 |")
        lines.append("|---|---:|")
        for rule, count in report["review_by_rule"].items():
            lines.append(f"| `{rule}` | {count} |")
    else:
        lines.append("- 无")
    lines.append("")
    lines.append("## 字段分布变化")
    lines.append("")
    for field in HEAD_FIELDS:
        lines.append(f"### {field}")
        lines.append("")
        lines.append("| 取值 | 清洗前 | 清洗后 |")
        lines.append("|---|---:|---:|")
        before = report["distribution_before"].get(field, {})
        after = report["distribution_after"].get(field, {})
        for value in sorted(set(before) | set(after)):
            lines.append(f"| {value} | {before.get(value, 0)} | {after.get(value, 0)} |")
        lines.append("")
    lines.append(f"- quality 多标签样本数：{report['quality_multi_label_samples']}")
    lines.append(f"- 保留样本中字段缺失数：{report['missing_field_counts_kept'] or '无'}")
    lines.append(f"- 未能映射的取值：{report['unmapped_values'] or '无'}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="构建干净可训练数据集并输出清洗报告")
    parser.add_argument("--json", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--out", default="data/clean")
    parser.add_argument("--no-copy-images", action="store_true",
                        help="只写标签与报告，不复制图片")
    args = parser.parse_args()
    report = build(args.json, args.image_root, args.out, copy_images=not args.no_copy_images)
    print(f"原始任务 {report['total_tasks']} / 保留 {report['kept_samples']} / "
          f"删除 {report['removed_samples']}（任务三 {report['task3_removed_samples']}）/ "
          f"通过率 {report['pass_rate'] * 100:.2f}%")
    print(f"删除明细: {report['deleted_by_rule']}")
    print(f"复核清单: {report['review_by_rule']}")
    print(f"输出目录: {report['out_dir']}")


if __name__ == "__main__":
    main()
