# -*- coding: utf-8 -*-
"""审核服务：建立请求上下文、运行固定证据流程并记录审计轨迹。"""

from __future__ import annotations

import json
import re
import os
import uuid
import hashlib
import time
import io
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image
from typing import Any

from . import tools as tools_mod
from .explanation import compose_final_suggestion, generate_final_explanation
from .schemas import ReviewResponse
from .policy import DecisionPolicy, decide
from . import rag as rag_mod
from .text_evidence import (
    assess_text_evidence, build_decision_evidence, collect_knowledge_hits,
)
from .text_detection import LightweightTextDetector
from .text_semantic import TextSemanticReviewer
from .anime_vision_policy import anime_visual_gate, reconcile_anime_visual
from .vision import VisionGate
from .workflow import FixedReviewWorkflow


class ReviewService:
    def __init__(
        self,
        llm_model: Any,
        engine: Any,
        ocr_client: Any = None,
        text_detector: Any = None,
        retriever: Any = None,
        log_dir: str = "logs",
        mock_mode: bool = False,
        ocr_enabled: bool = True,
        lowlevel_enabled: bool = True,
        rag_enabled: bool = True,
        decision_policy: dict | None = None,
        save_images: bool = True,
        vision_client: Any = None,
        vision_gate: dict | None = None,
        scene_scorer: Any = None,
        scene_concepts_enabled: bool = False,
        scene_options: dict | None = None,
        explanation_enabled: bool = True,
        text_semantic_reviewer: Any = None,
        anime_visual_reviewer: Any = None,
        anime_visual_enabled: bool = False,
        anime_visual_min_style_score: float = 0.90,
    ):
        self.llm_model = llm_model
        self.engine = engine
        self.ocr_client = ocr_client
        self.text_detector = text_detector or LightweightTextDetector()
        self.policy = DecisionPolicy(**(decision_policy or {}))
        self.rag_enabled = rag_enabled
        self.retriever = retriever or rag_mod.retrieve
        self.text_semantic_reviewer = (
            text_semantic_reviewer or TextSemanticReviewer(llm_model)
        )
        self.anime_visual_reviewer = anime_visual_reviewer
        self.anime_visual_enabled = bool(anime_visual_enabled)
        self.anime_visual_min_style_score = float(anime_visual_min_style_score)
        self.tools = tools_mod.build_tools(
            engine,
            ocr_client,
            retriever,
            text_detector=self.text_detector,
            enable_ocr=ocr_enabled,
            enable_lowlevel=lowlevel_enabled,
            enable_rag=rag_enabled,
            vision_client=vision_client,
            vision_gate=VisionGate(**(vision_gate or {})),
            scene_scorer=scene_scorer,
            scene_options=scene_options,
        )
        self.workflow = FixedReviewWorkflow(
            self.tools,
            ocr_enabled=ocr_enabled,
            scene_concepts_enabled=scene_concepts_enabled,
            rag_enabled=rag_enabled,
            retriever=self.retriever,
            text_semantic_reviewer=self.text_semantic_reviewer,
        )
        self.log_dir = log_dir
        self.mock_mode = mock_mode
        self.explanation_enabled = explanation_enabled
        self.save_images = save_images
        self.model_identity = {"checkpoint": str(getattr(engine, "ckpt_dir", "")),
                               "trained": bool(getattr(engine, "trained", False)),
                               "anime_style_router_loaded": bool(
                                   getattr(engine, "anime_style_router", None)
                                   is not None
                               ),
                               "type_thresholds": getattr(engine, "type_thresholds", {}),
                               "quality_thresholds": getattr(engine, "quality_thresholds", {})}
        checkpoint = self.model_identity["checkpoint"]
        if checkpoint:
            self.model_identity["sha256"] = {
                name: hashlib.sha256((Path(checkpoint) / name).read_bytes()).hexdigest()
                for name in ("head.pt", "adapter_model.safetensors", "labels.json")
                if (Path(checkpoint) / name).is_file()
            }
        style_router_dir = getattr(engine, "anime_style_router_dir", "")
        if isinstance(style_router_dir, (str, os.PathLike)) and style_router_dir:
            self.model_identity["anime_style_router_sha256"] = {
                name: hashlib.sha256(
                    (Path(style_router_dir) / name).read_bytes()
                ).hexdigest()
                for name in ("router.pt", "metadata.json")
                if (Path(style_router_dir) / name).is_file()
            }
        os.makedirs(log_dir, exist_ok=True)

    @property
    def ready(self) -> bool:
        return self.workflow is not None and self.engine is not None

    @staticmethod
    def _redact_text(text: str) -> str:
        redacted = re.sub(r"(?<!\d)1[3-9]\d{9}(?!\d)", "1**********", str(text))
        return re.sub(r"(?<!\d)\d{17}[\dXx](?!\d)", "[身份证号已脱敏]", redacted)

    @staticmethod
    def _file_log_fields(filename: str) -> dict[str, str]:
        if not filename:
            return {"filename_hash": "", "file_extension": ""}
        digest = hashlib.sha256(filename.encode("utf-8")).hexdigest()[:16]
        return {"filename_hash": digest, "file_extension": os.path.splitext(filename)[1].lower()}

    @staticmethod
    def _sanitize_tool_results(results: dict[str, Any]) -> dict[str, Any]:
        """保留审核审计信息，不把 OCR 原文或完整知识正文写入日志。"""
        sanitized = dict(results)
        ocr = sanitized.get("ocr_image")
        if isinstance(ocr, dict):
            sanitized["ocr_image"] = {
                "success": bool(ocr.get("success", False)),
                "available": bool(ocr.get("available", True)),
                "provider": ocr.get("provider", ""),
                "model": ocr.get("model", ""),
                "text_count": len(ocr.get("texts", []) or []),
                "mean_confidence": ocr.get("mean_confidence"),
                "line_confidences": ocr.get("line_confidences", []),
                "request_id": ocr.get("request_id", ""),
                "error": ocr.get("error"),
            }
        detection = sanitized.get("detect_text")
        if isinstance(detection, dict):
            sanitized["detect_text"] = {
                "status": detection.get("status"),
                "has_text": bool(detection.get("has_text")),
                "box_count": int(detection.get("box_count", 0) or 0),
                "confidence": detection.get("confidence"),
                "method": detection.get("method", ""),
                "error": detection.get("error"),
            }

        searches = sanitized.get("search_knowledge")
        if isinstance(searches, dict):
            safe_searches: dict[str, list[dict[str, Any]]] = {}
            for cache_key, matches in searches.items():
                safe_searches[cache_key] = [
                    {
                        "id": item.get("id", ""),
                        "type": item.get("type", item.get("profile", "")),
                        "score": item.get("score"),
                        "source_status": item.get("source", {}).get("locator_status", "")
                        if isinstance(item.get("source"), dict) else "",
                    }
                    for item in (matches or []) if isinstance(item, dict)
                ]
            sanitized["search_knowledge"] = safe_searches
        return sanitized

    def _log(self, record: dict[str, Any]) -> None:
        record = dict(record)
        if "filename" in record:
            record.update(self._file_log_fields(record.pop("filename")))
        for field in ("suggestion",):
            if field in record:
                record[field] = self._redact_text(record[field])
        final_explanation = record.get("final_explanation")
        if isinstance(final_explanation, dict):
            final_explanation = dict(final_explanation)
            if isinstance(final_explanation.get("explanation"), str):
                final_explanation["explanation"] = self._redact_text(
                    final_explanation["explanation"]
                )
            record["final_explanation"] = final_explanation
        detection = record.get("text_detection")
        if isinstance(detection, dict):
            record["text_detection"] = {
                "status": detection.get("status"),
                "has_text": bool(detection.get("has_text")),
                "box_count": int(detection.get("box_count", 0) or 0),
                "confidence": detection.get("confidence"),
                "method": detection.get("method", ""),
                "error": detection.get("error"),
            }
        if "tool_results" in record:
            record["tool_results"] = self._sanitize_tool_results(record["tool_results"])
        # Raw messages and tool arguments can repeat OCR text and search queries.
        record.pop("messages", None)
        record.pop("tool_cache", None)
        record["steps"] = [
            {key: value for key, value in step.items() if key not in {"args", "output"}}
            for step in record.get("steps", [])
        ]
        path = os.path.join(self.log_dir, "review.log.jsonl")
        with open(path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _finalize_policy_decision(
        self,
        context: tools_mod.RequestToolContext,
        log: dict[str, Any],
    ) -> str | None:
        """完成结构化证据包、程序裁决和最终 LLM 解释。"""
        if "analyze_image" not in context.cache:
            return None
        evidence = dict(context.cache["analyze_image"])
        evidence["text_detection"] = context.cache.get(
            "text_detection", {"status": "not_available", "has_text": False}
        )
        lowlevel = context.cache.get("analyze_lowlevel")
        if isinstance(lowlevel, dict):
            evidence["lowlevel"] = {
                "status": "ok",
                "source": context.cache.get(
                    "analyze_lowlevel_source", "workflow_tool"
                ),
                "features": lowlevel,
                "text_probe_ok": bool(
                    context.cache.get("text_probe_ok", True)
                ),
                "text_suspected": bool(
                    context.cache.get("text_suspected", False)
                ),
            }
        else:
            evidence["lowlevel"] = {
                "status": "failed" if context.cache.get("text_probe_error")
                else "not_called",
                "source": "automatic_text_probe",
                "error": context.cache.get("text_probe_error"),
                "text_probe_ok": bool(
                    context.cache.get("text_probe_ok", False)
                ),
                "text_suspected": bool(
                    context.cache.get("text_suspected", False)
                ),
            }
        text_assessment = context.cache.get("text_evidence")
        if not isinstance(text_assessment, dict):
            text_assessment = assess_text_evidence(
                context.cache.get("ocr_image"),
                self.retriever if self.rag_enabled else None,
            )
        evidence["text_evidence"] = text_assessment
        knowledge_hits = collect_knowledge_hits(context.cache)
        evidence["search_knowledge"] = knowledge_hits
        if "match_scene_concepts" in context.cache:
            evidence["scene_concepts"] = context.cache["match_scene_concepts"]
            log["scene_concepts"] = context.cache["match_scene_concepts"]

        decision_evidence = build_decision_evidence(evidence)
        decision = decide(evidence, self.policy, context.cache.get("review_blood"))
        visual_gate = anime_visual_gate(
            decision,
            evidence,
            min_style_score=self.anime_visual_min_style_score,
        )
        log["anime_visual_gate"] = visual_gate
        if (
            self.anime_visual_enabled
            and self.anime_visual_reviewer is not None
            and visual_gate["eligible"]
        ):
            review = self.anime_visual_reviewer.review(context.image)
            decision = reconcile_anime_visual(decision, evidence, review)
            log["anime_visual_review"] = review
        if self.explanation_enabled:
            explanation_result = generate_final_explanation(
                self.llm_model, decision_evidence, decision
            )
        else:
            explanation_result = {
                "status": "disabled",
                "reason": "feature_disabled",
            }
        suggestion = compose_final_suggestion(decision, explanation_result)

        log["decision_evidence"] = decision_evidence
        log["decision"] = decision
        log["text_detection"] = evidence["text_detection"]
        log["text_evidence"] = text_assessment
        log["search_knowledge_hits"] = knowledge_hits
        log["final_explanation"] = explanation_result
        log["suggestion_source"] = (
            "llm_explanation"
            if explanation_result.get("status") == "ok"
            else "deterministic_fallback"
        )
        return suggestion

    def review(self, image_bytes: bytes, filename: str = "") -> ReviewResponse:
        started = time.monotonic()
        request_id = uuid.uuid4().hex
        token = tools_mod.set_current_image(image_bytes)
        log: dict[str, Any] = {
            "request_id": request_id,
            "filename": filename,
            "mock": self.mock_mode,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "model": self.model_identity,
            "decision_policy": asdict(self.policy),
        }
        try:
            if self.save_images:
                image_dir = Path(self.log_dir) / "images"
                image_dir.mkdir(parents=True, exist_ok=True)
                # 用服务生成的编号命名，不把上传文件名当路径。
                with Image.open(io.BytesIO(image_bytes)) as original:
                    extension = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp",
                                 "GIF": ".gif", "BMP": ".bmp"}.get(original.format, ".image")
                image_path = image_dir / f"{request_id}{extension}"
                image_path.write_bytes(image_bytes)
                log["image_path"] = str(image_path.resolve())
            result = self.workflow.run()
            context = tools_mod.get_request_context()
            if result.get("clip_succeeded"):
                policy_suggestion = self._finalize_policy_decision(context, log)
                suggestion = policy_suggestion or "未能生成审核建议，请稍后重试。"
            else:
                suggestion = "未能取得 CLIP 视觉证据，无法完成图片审核，请稍后重试。"
            log.update({
                "workflow_version": result.get("workflow_version"),
                "clip_called": result.get("clip_called", False),
                "clip_succeeded": result.get("clip_succeeded", False),
                "finalized": result.get("finalized", False),
                "degraded": result.get("degraded", False),
                "tool_results": result.get("tool_results", {}),
                "tool_cache_keys": sorted(context.cache.keys()),
                "steps": result.get("steps", []),
                "suggestion": suggestion,
                "elapsed_ms": round((time.monotonic() - started) * 1000),
            })
            self._log(log)
            return ReviewResponse(id=request_id, suggestion=suggestion)
        except Exception as exc:  # noqa: BLE001
            suggestion = "审核流程暂不可用，无法完成本次审核，请稍后重试。"
            context = tools_mod.get_request_context()
            policy_suggestion = self._finalize_policy_decision(context, log)
            if policy_suggestion:
                suggestion = policy_suggestion
            log.update({
                "error": f"{type(exc).__name__}: {exc}",
                "degraded": True,
                "suggestion": suggestion,
                "tool_cache": context.cache,
                "elapsed_ms": round((time.monotonic() - started) * 1000),
            })
            self._log(log)
            return ReviewResponse(id=request_id, suggestion=suggestion)
        finally:
            tools_mod.reset_current_image(token)
