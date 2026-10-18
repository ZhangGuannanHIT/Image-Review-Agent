# -*- coding: utf-8 -*-
"""数据管道：Label Studio 标注 JSON → 结构化的 Sample 列表 + 数据体检报告。

输入：标注 JSON 路径 + 图片根目录。JSON 与图片按 file_upload 字段（文件名）join。
输出：list[Sample]，字段与 src/labels.py 常量对齐；并输出体检报告（分布/缺失/矛盾/清洗）。
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from .labels import (
    RISK_CLASSES, TYPE_CLASSES, INJURY_CLASSES, QUALITY_CLASSES,
)

# ---- 各控件（from_name）-> 类别常量名 ----
_RISK = "risk_level"
_TYPE = "violence_type"
_INJURY = "injury_detail"
_QUALITY = "quality"
_TARGET = "target"  # 不训练，但需要捕获以用于体检/备查


@dataclass
class Sample:
    """一条样本：单标签缺失用 None，多标签缺失用全零目标和 valid=False。"""
    image_path: str
    risk: Optional[int]          # RISK_CLASSES 索引，缺失 -> None/IGNORE_INDEX
    type_set: list[int]          # TYPE_CLASSES 多标签 0/1（长度=4）
    type_valid: bool             # type 字段是否有效（缺失为 False）
    injury: Optional[int]        # INJURY_CLASSES 索引
    quality_set: list[int]       # QUALITY_CLASSES 多标签 0/1（长度=5）
    quality_valid: bool          # 缺失或无法完整映射时为 False，不作为负标签训练
    raw: dict[str, Any] = field(default_factory=dict)  # 原始标签（备查/复用）

    @property
    def risk_valid(self) -> bool:
        return self.risk is not None

    @property
    def injury_valid(self) -> bool:
        return self.injury is not None


# ---- 从 Label Studio taxonomy / choices 值提取叶子集合 ----
def _flatten(paths: Any, out: list[str]) -> list[str]:
    """递归展平 taxonomy 值（可能是嵌套 list，如 [['轻微模糊']] 或 [['A','B']]）。"""
    if isinstance(paths, list):
        for x in paths:
            _flatten(x, out)
    elif isinstance(paths, str):
        out.append(paths)
    return out


def _parse_annotation(annotation: dict) -> dict[str, list[str]]:
    """把一条 annotation 的 result[] 按 from_name 分组，返回 {from_name: [叶子字符串]}。"""
    res: dict[str, list[str]] = defaultdict(list)
    for r in annotation.get("result", []):
        fn = r.get("from_name")
        v = r.get("value", {}) or {}
        leaf: list[str] = []
        if "taxonomy" in v:
            _flatten(v["taxonomy"], leaf)
        elif "choices" in v:
            leaf = list(v["choices"])
        elif "text" in v:
            # Label Studio 的 textarea 存 list[str]（备注/note），标量形式也兼容。
            raw_text = v["text"]
            leaf = [str(x) for x in raw_text] if isinstance(raw_text, list) else [str(raw_text)]
        if leaf:
            res[fn].extend(leaf)
    return {k: list(dict.fromkeys(v)) for k, v in res.items()}  # 去序（集合）


def _encode_single(values: list[str], classes: list[str]) -> tuple[Optional[int], str]:
    """单选字段：返回 (索引, 未能映射的取值)。找不到返回 None。

    匹配策略（由宽松到严格），用于容忍 Label Studio 存值带括号注释：
      - 精确等于；
      - 值以类别开头（"L2_中度违规（明显暴力…）" → L2_中度违规）；
      - 类别以值开头（"暗示/威胁" 与 "暗示/威胁（武器指向/捆绑）" 之类）；
      - 类别关键片段出现在值中。
    """
    for v in values:
        clean = v.strip()
        for i, c in enumerate(classes):
            cc = c.strip()
            if clean == cc:
                return i, ""
            if cc.startswith(clean) or clean.startswith(cc):
                return i, ""
            if cc.replace("（", "_").replace("）", "_") in clean or clean in cc:
                return i, ""
    return None, (";".join(values) if values else "空")


def _encode_multihot(values: list[str], classes: list[str]) -> tuple[list[int], str]:
    """多选字段：返回 0/1 列表（长度 len(classes)），及未能映射的取值。"""
    mask = [0] * len(classes)
    unmapped: list[str] = []
    for v in values:
        clean = v.strip()
        found = False
        for i, c in enumerate(classes):
            cc = c.strip()
            if clean == cc or cc.startswith(clean) or clean.startswith(cc):
                mask[i] = 1
                found = True
                break
        if not found:
            unmapped.append(v)
    return mask, (";".join(unmapped) if unmapped else "")


# ---- 清洗规则：violence_type 含"无暴力"+其他类 → 整条剔除 ----
def _canonical_type(value: str) -> Optional[str]:
    """把 Label Studio 的完整说明值映射到冻结的类型标签。"""
    clean = value.strip()
    for label in TYPE_CLASSES:
        if clean == label or clean.startswith(label) or label.startswith(clean):
            return label
    return None


def _is_conflict(values: list[str]) -> bool:
    normalized = {_canonical_type(v) for v in values}
    normalized.discard(None)
    return "无暴力" in normalized and len(normalized) > 1


def _resolve_image_path(image_uri: str, file_upload: Optional[str], image_root: str) -> tuple[str, bool]:
    """返回解析路径及存在状态；配置 root 后不再静默回退远程 URI。"""
    if image_root:
        fname = os.path.basename(file_upload or image_uri or "")
        candidate = os.path.join(image_root, fname)
        return candidate, os.path.isfile(candidate)
    return image_uri, bool(image_uri)


def parse_label_studio(
    json_path: str, image_root: str = "", report: bool = True
) -> tuple[list[Sample], dict[str, Any]]:
    """解析 Label Studio 导出 JSON。

    返回 (samples, report)。
    report 含：每个字段分布/缺失/矛盾组合/清洗记录；供标注负责人核对。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    samples: list[Sample] = []
    cleaned: list[dict] = []
    removed_conflict: list[int] = []
    missing_images: list[dict[str, Any]] = []

    # 体检统计
    dist: dict[str, Counter] = defaultdict(Counter)
    missing: dict[str, int] = defaultdict(int)
    conflicts: list[tuple] = []

    for task in tasks:
        task_id = task.get("id")
        anns = task.get("annotations") or []
        first = next((a for a in anns if not a.get("was_cancelled")), None)
        if not first:
            missing["no_annotation"] += 1
            continue

        parsed = _parse_annotation(first)
        file_upload = task.get("file_upload")
        image_uri = (task.get("data") or {}).get("image", "")

        # --- 编码各字段 ---
        risk, risk_unmapped = _encode_single(parsed.get(_RISK, []), RISK_CLASSES)
        type_mask, type_unmapped = _encode_multihot(parsed.get(_TYPE, []), TYPE_CLASSES)
        injury, inj_unmapped = _encode_single(parsed.get(_INJURY, []), INJURY_CLASSES)
        quality_mask, qual_unmapped = _encode_multihot(parsed.get(_QUALITY, []), QUALITY_CLASSES)

        # --- 记录分布/缺失/未映射 ---
        if parsed.get(_RISK):
            for v in parsed[_RISK]:
                dist["risk"][v] += 1
        else:
            missing["risk_level"] += 1
        if parsed.get(_TYPE):
            for v in parsed[_TYPE]:
                dist["violence_type"][v] += 1
        else:
            missing["violence_type"] += 1
        if parsed.get(_INJURY):
            for v in parsed[_INJURY]:
                dist["injury_detail"][v] += 1
        else:
            missing["injury_detail"] += 1
        if parsed.get(_QUALITY):
            for v in parsed[_QUALITY]:
                dist["quality"][v] += 1
        else:
            missing["quality"] += 1
        for unm, nm in [(risk_unmapped, "risk"), (type_unmapped, "type"),
                        (inj_unmapped, "injury"), (qual_unmapped, "quality")]:
            if unm:
                dist["unmapped_" + nm][unm] += 1

        # --- 矛盾组合（跨字段语义张力，供复核，不自动清洗）---
        risk_label = parsed.get(_RISK, [None])[0]
        type_vals = parsed.get(_TYPE, [])
        normalized_types = {_canonical_type(v) for v in type_vals}
        if risk_label and risk_label.startswith("L0") and "直接暴力（攻击进行中）" in normalized_types:
            conflicts.append((task_id, "L0_安全 却标 直接暴力", parsed))
        if risk_label and risk_label.startswith("L3") and "无暴力" in type_vals:
            conflicts.append((task_id, "L3_严重 却标 无暴力", parsed))

        # --- 清洗：violence_type 含"无暴力"+其他类 → 整条剔除 ---
        if _is_conflict(type_vals):
            removed_conflict.append(task_id)
            continue

        image_path, image_exists = _resolve_image_path(image_uri, file_upload, image_root)
        if image_root and not image_exists:
            missing_images.append({"task_id": task_id, "path": image_path})
        samples.append(Sample(
            image_path=image_path,
            risk=risk,
            type_set=type_mask,
            type_valid=bool(parsed.get(_TYPE)),
            injury=injury,
            quality_set=quality_mask,
            quality_valid=bool(any(quality_mask)) and not qual_unmapped,
            raw={"task_id": task_id, "file_upload": file_upload,
                 "parsed": parsed, "image_uri": image_uri},
        ))

    report_data = _build_report(
        len(tasks), len(samples), dist, missing, conflicts, removed_conflict, missing_images
    )

    if report:
        _print_report(report_data)
    return samples, report_data


def _build_report(total, kept, dist, missing, conflicts, removed, missing_images) -> dict:
    return {
        "total_tasks": total,
        "kept_samples": kept,
        "cleaned_count": len(removed),
        "removed_ids": removed,
        "field_distribution": {k: dict(cc) for k, cc in dist.items()},
        "missing_count": dict(missing),
        "conflicts": conflicts,
        "missing_images": missing_images,
    }


def _print_report(report: dict) -> None:
    print("=" * 60)
    print(f"[数据体检] 任务数 {report['total_tasks']} / 保留样本 {report['kept_samples']} / 清洗 {report['cleaned_count']}")
    if report["removed_ids"]:
        print(f"  剔除冲突样本 id: {report['removed_ids']}")
    print("-- 字段分布 --")
    for k, d in report["field_distribution"].items():
        print(f"  {k}: {d}")
    print("-- 缺失统计 --")
    print(f"  {report['missing_count']}")
    if report["missing_images"]:
        print(f"-- 缺失图片: {len(report['missing_images'])} --")
    print("-- 矛盾组合（供复核，未清洗）--")
    for task_id, why, _ in report["conflicts"]:
        print(f"  task {task_id}: {why}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Label Studio 导出 JSON 路径")
    ap.add_argument("--image-root", default="", help="图片本地根目录")
    args = ap.parse_args()
    samples, report = parse_label_studio(args.json, args.image_root, report=True)
    print(f"\n解析完成：{len(samples)} 条样本")
