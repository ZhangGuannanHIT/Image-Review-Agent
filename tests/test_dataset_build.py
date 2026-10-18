# -*- coding: utf-8 -*-
"""数据集构建测试：配对、标签头收敛、多标签 quality、确定性删除规则。

本模块不依赖 torch，CPU 环境也能运行。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.build_dataset import build
from src.dataset import parse_label_studio


def _result(from_name: str, field_type: str, values: list[str]) -> dict:
    value = {"choices": values} if field_type == "choices" else {"taxonomy": [[v] for v in values]}
    return {"from_name": from_name, "to_name": "img", "type": field_type, "value": value}


def _task(task_id: int, image: str, results: list[dict], cancelled: bool = False) -> dict:
    return {
        "id": task_id,
        "file_upload": image,
        "data": {"image": f"/data/upload/6/{image}"},
        "annotations": [{"was_cancelled": cancelled, "result": results}],
    }


class BuildDatasetTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.image_root = root / "raw_images"
        self.image_root.mkdir()
        self.out_dir = root / "clean"

        # 1) 正常样本：quality 多标签（严重模糊 + 文字扭曲形变），带 target/note
        keep_multi = _task(1, "keep_multi.jpg", [
            _result("risk_level", "choices", ["L2_中度违规（明显暴力/开放性伤口）"]),
            _result("violence_type", "taxonomy", ["直接暴力（攻击进行中）"]),
            _result("injury_detail", "choices", ["中度（明显开放性伤口/大量血泊）"]),
            _result("quality", "taxonomy", ["严重模糊", "文字扭曲形变"]),
            _result("target", "taxonomy", ["人类"]),
            {"from_name": "note", "to_name": "img", "type": "textarea",
             "value": {"text": ["文字扭曲的血腥暴力画面"]}},
        ])
        # 2) 正常样本：L0 安全、清晰
        keep_safe = _task(2, "keep_safe.jpg", [
            _result("risk_level", "choices", ["L0_安全（无违规内容）"]),
            _result("violence_type", "taxonomy", ["无暴力"]),
            _result("injury_detail", "choices", ["无血腥"]),
            _result("quality", "taxonomy", ["清晰"]),
        ])
        # 3) 任务三：L0 安全却标直接暴力 → 删除
        drop_contradiction = _task(3, "drop_contradiction.jpg", [
            _result("risk_level", "choices", ["L0_安全（无违规内容）"]),
            _result("violence_type", "taxonomy", ["直接暴力（攻击进行中）"]),
            _result("injury_detail", "choices", ["无血腥"]),
            _result("quality", "taxonomy", ["轻微模糊"]),
        ])
        # 4) 标注全部取消 → 删除
        drop_cancelled = _task(4, "drop_cancelled.jpg", [], cancelled=True)
        # 5) 图片不存在 → 删除
        drop_missing = _task(5, "not_on_disk.jpg", [
            _result("risk_level", "choices", ["L1_轻微违规（轻度暴力/少量血迹）"]),
        ])
        # 6) 任务三：L0 安全却标"暗示/威胁"（无血腥）→ 删除
        drop_threat = _task(6, "drop_threat.jpg", [
            _result("risk_level", "choices", ["L0_安全（无违规内容）"]),
            _result("violence_type", "taxonomy", ["暗示/威胁（武器指向/捆绑）"]),
            _result("injury_detail", "choices", ["无血腥"]),
            _result("quality", "taxonomy", ["轻微模糊"]),
        ])
        # 7) 任务三：L2 却缺失暴力类型与血腥程度 → 删除
        drop_missing_field = _task(7, "drop_missing_field.jpg", [
            _result("risk_level", "choices", ["L2_中度违规（明显暴力/开放性伤口）"]),
            _result("quality", "taxonomy", ["轻微模糊"]),
        ])

        self.json_path = self.image_root / "export.json"
        self.json_path.write_text(
            json.dumps([keep_multi, keep_safe, drop_contradiction, drop_cancelled, drop_missing,
                        drop_threat, drop_missing_field], ensure_ascii=False),
            encoding="utf-8",
        )
        for name in ("keep_multi.jpg", "keep_safe.jpg", "drop_contradiction.jpg",
                     "drop_threat.jpg", "drop_missing_field.jpg", "unlabelled.jpg"):
            (self.image_root / name).write_bytes(b"fake-image")

    def tearDown(self):
        self._tmp.cleanup()

    def test_build_pairs_and_cleans(self):
        report = build(str(self.json_path), str(self.image_root), str(self.out_dir))
        self.assertEqual(report["total_tasks"], 7)
        self.assertEqual(report["kept_samples"], 2)
        self.assertEqual(report["removed_samples"], 5)
        self.assertEqual(report["task3_removed_samples"], 3)
        self.assertEqual(report["deleted_by_rule"]["C2_safe_but_violence"], 1)
        self.assertEqual(report["deleted_by_rule"]["C5_safe_but_threat"], 1)
        self.assertEqual(report["deleted_by_rule"]["C6_risk_without_evidence_field"], 1)
        self.assertEqual(report["deleted_by_rule"]["C0_no_valid_annotation"], 1)
        self.assertEqual(report["deleted_by_rule"]["P0_image_missing"], 1)

        kept = json.loads((self.out_dir / "labels.json").read_text(encoding="utf-8"))
        self.assertEqual([t["id"] for t in kept], [1, 2])
        for task in kept:
            names = {r["from_name"] for r in task["annotations"][0]["result"]}
            self.assertEqual(names, {"risk_level", "violence_type", "injury_detail", "quality"})

        # 图片按配对复制，无标签图片不进入数据集
        copied = sorted(p.name for p in (self.out_dir / "images").iterdir())
        self.assertEqual(copied, ["keep_multi.jpg", "keep_safe.jpg"])

        # target / note 不进入训练文件，改存复核副产物
        extras = json.loads((self.out_dir / "review_extras.json").read_text(encoding="utf-8"))
        self.assertEqual(extras["1"]["target"], ["人类"])
        self.assertEqual(extras["1"]["note"], ["文字扭曲的血腥暴力画面"])
        self.assertEqual(extras["2"]["target"], [])

    def test_second_build_prunes_stale_images(self):
        build(str(self.json_path), str(self.image_root), str(self.out_dir))
        stale = self.out_dir / "images" / "keep_safe.jpg"
        stale.write_bytes(b"stale")
        report = build(str(self.json_path), str(self.image_root), str(self.out_dir))
        self.assertEqual(report["pruned_stale_images"], 0)
        self.assertTrue(stale.is_file())
        (self.out_dir / "images" / "removed_sample.jpg").write_bytes(b"stale")
        report = build(str(self.json_path), str(self.image_root), str(self.out_dir))
        self.assertEqual(report["pruned_stale_images"], 1)
        self.assertFalse((self.out_dir / "images" / "removed_sample.jpg").exists())

    def test_built_json_is_trainable_and_keeps_multilabel_quality(self):
        build(str(self.json_path), str(self.image_root), str(self.out_dir))
        samples, report = parse_label_studio(
            str(self.out_dir / "labels.json"), str(self.out_dir / "images"), report=False
        )
        self.assertEqual(len(samples), 2)
        self.assertEqual(report["missing_images"], [])
        self.assertEqual(report["cleaned_count"], 0)
        first = next(s for s in samples if s.raw["task_id"] == 1)
        self.assertEqual(first.risk, 2)
        self.assertEqual([i for i, flag in enumerate(first.quality_set) if flag], [2, 3])
        self.assertTrue(first.quality_valid)
        self.assertEqual(first.injury, 2)


if __name__ == "__main__":
    unittest.main()
