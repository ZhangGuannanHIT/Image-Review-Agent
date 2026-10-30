# -*- coding: utf-8 -*-
"""命令行入口：单图审核调试 / 基模测用。

默认读取 config/inference.yaml 的模型路径、checkpoint、设备与多标签阈值；
命令行参数可以逐项覆盖配置。

用法：
  python -m src.cli --image sample_images/knife_man.jpg
  python -m src.cli --image x.jpg --json                # 输出结构化 JSON
  python -m src.cli --image x.jpg --ckpt-dir checkpoints/best
  python -m src.cli --image x.jpg --config config/inference.yaml --no-lora
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

DEFAULT_CONFIG_PATH = os.path.join(_REPO_ROOT, "config", "inference.yaml")


def load_inference_config(path: str = DEFAULT_CONFIG_PATH) -> dict:
    """读取单图推理配置；文件不存在时返回空配置（全部走默认值）。"""
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"推理配置必须是 YAML 映射: {path}")
    return config


def _resolve(path: str) -> str:
    """相对路径按仓库根目录解析。"""
    if not path or os.path.isabs(path):
        return path
    return os.path.join(_REPO_ROOT, path)


def main():
    ap = argparse.ArgumentParser(description="FG-CLIP2 违规审核 CLI")
    ap.add_argument("--image", required=True, help="图片路径")
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="推理配置路径")
    ap.add_argument("--model-path", default=None, help="覆盖配置中的基础模型目录")
    ap.add_argument("--ckpt-dir", default=None, help="训练权重目录（head.pt 等）")
    ap.add_argument("--device", default=None, help="运行设备")
    ap.add_argument("--lora", action=argparse.BooleanOptionalAction, default=None,
                    help="是否加载视觉塔 LoRA（未安装 peft 时用 --no-lora）")
    ap.add_argument("--json", action="store_true", help="输出结构化 JSON")
    args = ap.parse_args()

    from src.engine import ReviewEngine

    config = load_inference_config(args.config)
    model_path = _resolve(args.model_path or config.get("model_path") or "fg_clip")
    ckpt_dir = _resolve(args.ckpt_dir if args.ckpt_dir is not None else config.get("ckpt_dir", ""))
    engine = ReviewEngine(
        model_path,
        ckpt_dir=ckpt_dir,
        device=args.device or config.get("device", "cuda"),
        lora=args.lora if args.lora is not None else config.get("lora", True),
        type_thresholds=config.get("type_thresholds"),
        quality_thresholds=config.get("quality_thresholds"),
        binary_injury_dir=_resolve(config.get("binary_injury_dir", "")),
        quality_fusion_dir=_resolve(config.get("quality_fusion_dir", "")),
        anime_style_router_dir=_resolve(
            config.get("anime_style_router_dir", "")
        ),
        anime_style_min_score=float(
            config.get("anime_style_min_score", 0.90)
        ),
        strict_load=config.get("strict_load", False),
    )

    from PIL import Image
    img = Image.open(args.image).convert("RGB")

    result = engine.predict(img).to_dict()

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"risk: {result['risk']['label']}")
        print(f"violence_type: {result['type']['labels']}")
        print(f"injury: {result['injury']['label']}")
        if result.get("blood_binary"):
            print(f"blood_binary: {result['blood_binary']['label']} ({result['blood_binary']['p_blood']:.3f})")
        print(f"quality: {result['quality']['labels']}")
        print(f"confidence: {result['conf']:.4f}")


if __name__ == "__main__":
    main()
