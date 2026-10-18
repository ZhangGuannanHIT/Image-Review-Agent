# -*- coding: utf-8 -*-
"""后端启动脚本：加载模型配置，初始化编排器，启动 FastAPI。

用法：
  python run_backend.py            # 默认读 backend/config.yaml
  $env:LLM_API_KEY="xxx"; python run_backend.py
"""

from __future__ import annotations

import os
import sys

import yaml

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def load_config(path="backend/config.yaml"):
    if not os.path.isabs(path):
        path = os.path.join(_REPO_ROOT, path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if os.getenv("BACKEND_PORT"):
        cfg.setdefault("server", {})["port"] = int(os.environ["BACKEND_PORT"])
    return cfg


def main():
    os.chdir(_REPO_ROOT)
    cfg = load_config()
    mode = cfg.get("runtime", {}).get("mode", "real")
    vision_client = None
    from backend.service import ReviewService
    from backend.api import init_service

    if mode == "mock":
        from backend.mock import MockOCRClient, MockReviewEngine, RuleBasedMockChatModel
        engine = MockReviewEngine()
        ocr_client = MockOCRClient()
        llm_model = RuleBasedMockChatModel()
    elif mode == "real":
        from src.engine import ReviewEngine
        from backend.ocr import OCRClient
        from backend.llm import LLMClient
        inference_cfg = {}
        inference_path = cfg.get("inference_config_path")
        if inference_path:
            if not os.path.isabs(inference_path):
                inference_path = os.path.join(_REPO_ROOT, inference_path)
            with open(inference_path, "r", encoding="utf-8") as file:
                inference_cfg = yaml.safe_load(file) or {}
        model_path = os.path.join(_REPO_ROOT, cfg["model"]["path"])
        engine = ReviewEngine(
            model_path,
            ckpt_dir=os.path.join(_REPO_ROOT, cfg["model"]["ckpt_dir"]),
            device=cfg["model"]["device"],
            lora=cfg["model"].get("lora", True),
            strict_load=True,
            type_thresholds=inference_cfg.get("type_thresholds"),
            quality_thresholds=inference_cfg.get("quality_thresholds"),
        )
        if cfg["model"].get("require_checkpoint", True) and not engine.trained:
            raise RuntimeError("真实模式需要已训练的 checkpoint（LoRA adapter + head.pt）")
        ocr_config_path = cfg.get("ocr_config_path", "config/ocr.yaml")
        if not os.path.isabs(ocr_config_path):
            ocr_config_path = os.path.join(_REPO_ROOT, ocr_config_path)
        ocr_client = OCRClient.from_config(ocr_config_path)
        llm_config_path = cfg.get("llm_config_path", "config/llm.yaml")
        if not os.path.isabs(llm_config_path):
            llm_config_path = os.path.join(_REPO_ROOT, llm_config_path)
        llm_client = LLMClient.from_config(llm_config_path)
        llm_model = llm_client.build_model()
        if cfg.get("vision_review", {}).get("enabled", False):
            from backend.vision import DeepSeekBloodReviewer
            vision_client = DeepSeekBloodReviewer(llm_client.config)
    else:
        raise ValueError(f"不支持的 runtime.mode: {mode}")

    features = cfg.get("features", {})
    service = ReviewService(
        llm_model=llm_model, engine=engine, ocr_client=ocr_client,
        log_dir=os.path.join(_REPO_ROOT, cfg["log_dir"]),
        max_rounds=cfg["agent"].get("max_rounds", 5),
        on_tool_fail=cfg["agent"].get("on_tool_fail", "skip"),
        mock_mode=mode == "mock",
        ocr_enabled=features.get("ocr_enabled", True),
        lowlevel_enabled=features.get("lowlevel_enabled", True),
        rag_enabled=features.get("rag_enabled", True),
        decision_policy=cfg.get("decision_policy"),
        vision_client=vision_client,
        vision_gate=cfg.get("vision_review", {}).get("trigger"),
        save_images=cfg.get("audit", {}).get("save_images", True))

    init_service(service, max_image_mb=cfg["server"].get("max_image_mb", 10))

    import uvicorn
    uvicorn.run("backend.api:app", host=cfg["server"]["host"], port=cfg["server"]["port"])


if __name__ == "__main__":
    main()
