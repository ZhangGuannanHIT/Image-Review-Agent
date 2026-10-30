# -*- coding: utf-8 -*-

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.merge_hard_probe_severity import merge_manifest


class HardProbeMergeTests(unittest.TestCase):
    def test_merges_export_by_basename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            manifest.write_text(
                "file,folder,domain,red_level,has_blood,severity_truth\n"
                "blood/a.jpg,f,real,high,1,\n"
                "no_blood/b.jpg,f,real,high,0,\n",
                encoding="utf-8",
            )
            export = root / "export.json"
            export.write_text(json.dumps([{
                "file_upload": "upload/12/uuid-a.jpg",
                "annotations": [{
                    "was_cancelled": False,
                    "result": [{
                        "from_name": "injury_detail",
                        "value": {"choices": ["中度（明显开放性伤口/大量血泊）"]},
                    }],
                }],
            }], ensure_ascii=False), encoding="utf-8")
            output = root / "merged.csv"
            result = merge_manifest(manifest, export, output)
            self.assertEqual(result["updated"], 1)
            with output.open(encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(rows[0]["severity_truth"], "中度（明显开放性伤口/大量血泊）")
            self.assertEqual(rows[1]["severity_truth"], "")


if __name__ == "__main__":
    unittest.main()
