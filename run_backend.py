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
    if os.getenv("BACKEND_MODE"):
        cfg.setdefault("runtime", {})["mode"] = os.environ["BACKEND_MODE"]
    return cfg


def main():
    os.chdir(_REPO_ROOT)
    cfg = load_config()
    mode = cfg.get("runtime", {}).get("mode", "real")
    vision_client = None
    anime_visual_reviewer = None
    scene_scorer = None
    scene_options = None
    features = cfg.get("features", {})
    from backend.service import ReviewService
    from backend.review_store import ReviewStore
    from backend.api import init_service
    from backend.text_detection import (
        LightweightTextDetector, RapidOCRTextDetector,
    )

    text_cfg = cfg.get("text_detection", {})
    if text_cfg.get("provider", "opencv_light") == "rapidocr":
        text_detector = RapidOCRTextDetector(
            model_root_dir=text_cfg.get("rapidocr_model_dir", "")
        )
    else:
        text_detector = LightweightTextDetector(
            max_side=int(text_cfg.get("max_side", 1600)),
            min_line_height=int(text_cfg.get("min_line_height", 7)),
            min_layout_score=float(text_cfg.get("min_layout_score", 0.48)),
            min_region_score=float(text_cfg.get("min_region_score", 0.52)),
            min_aligned_glyphs=int(text_cfg.get("min_aligned_glyphs", 2)),
        )

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
            binary_injury_dir=os.path.join(
                _REPO_ROOT,
                cfg["model"].get("binary_injury_dir", "checkpoints/best/injury_binary"),
            ),
            quality_fusion_dir=os.path.join(
                _REPO_ROOT,
                cfg["model"].get("quality_fusion_dir", "checkpoints/best/quality_fusion"),
            ),
            anime_style_router_dir=(
                os.path.join(
                    _REPO_ROOT,
                    cfg["model"]["anime_style_router_dir"],
                )
                if cfg["model"].get("anime_style_router_dir")
                else ""
            ),
            anime_style_min_score=float(
                cfg["model"].get("anime_style_min_score", 0.90)
            ),
        )
        if cfg["model"].get("require_checkpoint", True) and not engine.trained:
            raise RuntimeError("真实模式需要已训练的 checkpoint（LoRA adapter + head.pt）")
        ocr_client = None
        if features.get("ocr_enabled", True):
            ocr_config_path = cfg.get("ocr_config_path", "config/ocr.yaml")
            if not os.path.isabs(ocr_config_path):
                ocr_config_path = os.path.join(_REPO_ROOT, ocr_config_path)
            ocr_client = OCRClient.from_config(ocr_config_path)
        llm_config_path = cfg.get("llm_config_path", "config/llm.yaml")
        if not os.path.isabs(llm_config_path):
            llm_config_path = os.path.join(_REPO_ROOT, llm_config_path)
        llm_client = LLMClient.from_config(llm_config_path)
        llm_model = llm_client.build_model()
        anime_visual_cfg = cfg.get("anime_vision_review", {})
        if anime_visual_cfg.get("enabled", False):
            from backend.anime_vision import AnimeVisualReviewer
            anime_visual_model = llm_client.build_model(
                max_retries=int(anime_visual_cfg.get("max_retries", 1)),
                max_tokens=int(anime_visual_cfg.get("max_tokens", 512)),
                response_format=(
                    {"type": "json_object"}
                    if anime_visual_cfg.get("json_response", True)
                    else None
                ),
                disable_thinking=True,
            )
            anime_visual_reviewer = AnimeVisualReviewer(
                anime_visual_model,
                max_side=int(anime_visual_cfg.get("max_side", 1280)),
            )
        if features.get("scene_concepts_enabled", False):
            from backend.scene_concepts import SceneConceptScorer
            scene_cfg = cfg.get("scene_concepts", {})
            scene_config_path = scene_cfg.get(
                "config_path", "config/scene_concepts.yaml"
            )
            if not os.path.isabs(scene_config_path):
                scene_config_path = os.path.join(_REPO_ROOT, scene_config_path)
            with open(scene_config_path, "r", encoding="utf-8") as file:
                scene_options = yaml.safe_load(file) or {}
            prompt_path = scene_options.get(
                "prompts_path", "knowledge/scene_concepts.json"
            )
            if not os.path.isabs(prompt_path):
                prompt_path = os.path.join(_REPO_ROOT, prompt_path)
            scene_scorer = SceneConceptScorer(engine, prompt_path)
    else:
        raise ValueError(f"不支持的 runtime.mode: {mode}")

    service = ReviewService(
        llm_model=llm_model, engine=engine, ocr_client=ocr_client,
        text_detector=text_detector,
        log_dir=os.path.join(_REPO_ROOT, cfg["log_dir"]),
        mock_mode=mode == "mock",
        ocr_enabled=features.get("ocr_enabled", True),
        lowlevel_enabled=features.get("lowlevel_enabled", True),
        rag_enabled=features.get("rag_enabled", True),
        decision_policy=cfg.get("decision_policy"),
        vision_client=vision_client,
        scene_scorer=scene_scorer,
        scene_concepts_enabled=(
            mode == "real" and features.get("scene_concepts_enabled", False)
        ),
        scene_options=scene_options,
        explanation_enabled=features.get("llm_explanation_enabled", True),
        explanation_mode=features.get("llm_explanation_mode", "selective"),
        explanation_llm_actions=features.get("llm_explanation_actions"),
        anime_visual_reviewer=anime_visual_reviewer,
        anime_visual_enabled=(
            mode == "real"
            and cfg.get("anime_vision_review", {}).get("enabled", False)
        ),
        anime_visual_min_style_score=float(
            cfg.get("anime_vision_review", {}).get(
                "min_style_score", 0.90
            )
        ),
        anime_visual_allow_safety_review=bool(
            cfg.get("anime_vision_review", {}).get(
                "allow_safety_review", True
            )
        ),
        record_store=ReviewStore(
            os.path.join(
                _REPO_ROOT,
                cfg.get("history_db", "data/reviews.sqlite3"),
            )
        ),
        save_images=cfg.get("audit", {}).get("save_images", True))

    init_service(service, max_image_mb=cfg["server"].get("max_image_mb", 10))

    import uvicorn
    uvicorn.run("backend.api:app", host=cfg["server"]["host"], port=cfg["server"]["port"])


if __name__ == "__main__":
    main()
