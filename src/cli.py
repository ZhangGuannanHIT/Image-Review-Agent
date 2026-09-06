# -*- coding: utf-8 -*-
"""命令行入口：单图审核调试 / 基模测用。

用法：
  python -m src.cli --image sample_images/knife_man.jpg
  python -m src.cli --image x.jpg --json          # 输出结构化 JSON
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main():
    ap = argparse.ArgumentParser(description="FG-CLIP2 违规审核 CLI")
    ap.add_argument("--image", required=True, help="图片路径")
    ap.add_argument("--ckpt-dir", default="", help="训练权重目录（head.pt 等）")
    ap.add_argument("--device", default="cuda", help="运行设备")
    ap.add_argument("--json", action="store_true", help="输出结构化 JSON")
    args = ap.parse_args()

    from src.engine import ReviewEngine

    model_path = os.path.join(_REPO_ROOT, "fg_clip")
    engine = ReviewEngine(model_path, ckpt_dir=args.ckpt_dir, device=args.device)

    from PIL import Image
    img = Image.open(args.image).convert("RGB")

    result = engine.predict(img).to_dict()

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"risk: {result['risk']['label']}")
        print(f"violence_type: {result['type']['labels']}")
        print(f"injury: {result['injury']['label']}")
        print(f"quality: {result['quality']['label']}")
        print(f"confidence: {result['conf']:.4f}")


if __name__ == "__main__":
    main()
