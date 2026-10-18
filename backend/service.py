# -*- coding: utf-8 -*-
"""审核服务：建立请求上下文、运行带调用守卫的 Agent、记录轨迹。"""

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

from langchain_core.messages import HumanMessage

from . import tools as tools_mod
from .agent import build_agent
from .parser import extract_suggestion
from .prompts import build_system_prompt
from .schemas import ReviewResponse
from .policy import DecisionPolicy, decide, render
from .vision import VisionGate


class ReviewService:
    def __init__(
        self,
        llm_model: Any,
        engine: Any,
        ocr_client: Any = None,
        retriever: Any = None,
        log_dir: str = "logs",
        max_rounds: int = 5,
        on_tool_fail: str = "skip",
        mock_mode: bool = False,
        ocr_enabled: bool = True,
        lowlevel_enabled: bool = True,
        rag_enabled: bool = True,
        decision_policy: dict | None = None,
        save_images: bool = True,
        vision_client: Any = None,
        vision_gate: dict | None = None,
    ):
        self.llm_model = llm_model
        self.engine = engine
        self.ocr_client = ocr_client
        self.tools = tools_mod.build_tools(
            engine,
            ocr_client,
            retriever,
            enable_ocr=ocr_enabled,
            enable_lowlevel=lowlevel_enabled,
            enable_rag=rag_enabled,
            vision_client=vision_client,
            vision_gate=VisionGate(**(vision_gate or {})),
        )
        self.agent = build_agent(
            llm_model,
            self.tools,
            system_prompt=build_system_prompt(rag_enabled=rag_enabled),
            max_rounds=max_rounds,
            on_tool_fail=on_tool_fail,
        )
        self.log_dir = log_dir
        self.max_rounds = max_rounds
        self.mock_mode = mock_mode
        self.rag_enabled = rag_enabled
        self.policy = DecisionPolicy(**(decision_policy or {}))
        self.save_images = save_images
        self.model_identity = {"checkpoint": str(getattr(engine, "ckpt_dir", "")),
                               "trained": bool(getattr(engine, "trained", False)),
                               "type_thresholds": getattr(engine, "type_thresholds", {}),
                               "quality_thresholds": getattr(engine, "quality_thresholds", {})}
        checkpoint = self.model_identity["checkpoint"]
        if checkpoint:
            self.model_identity["sha256"] = {
                name: hashlib.sha256((Path(checkpoint) / name).read_bytes()).hexdigest()
                for name in ("head.pt", "adapter_model.safetensors", "labels.json")
                if (Path(checkpoint) / name).is_file()
            }
        os.makedirs(log_dir, exist_ok=True)

    @property
    def ready(self) -> bool:
        return self.agent is not None and self.engine is not None

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
                "request_id": ocr.get("request_id", ""),
                "error": ocr.get("error"),
            }

        final_suggestion = sanitized.get("finalize_review")
        if isinstance(final_suggestion, str):
            sanitized["finalize_review"] = ReviewService._redact_text(final_suggestion)

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
        for field in ("suggestion", "agent_suggestion"):
            if field in record:
                record[field] = self._redact_text(record[field])
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
            result = self.agent.invoke(
                {
                    "messages": [HumanMessage(content="请审核当前图片。")],
                    "remaining_steps": self.max_rounds * 2 + 2,
                    "filename": filename,
                    "tool_results": {},
                    "steps": [],
                    "suggestion": "",
                    "rounds": 0,
                    "clip_called": False,
                    "clip_succeeded": False,
                    "finalized": False,
                    "degraded": False,
                    "error": "",
                },
                config={"recursion_limit": self.max_rounds * 4 + 10},
            )
            suggestion = extract_suggestion(result)
            context = tools_mod.get_request_context()
            log["agent_suggestion"] = suggestion
            if result.get("clip_succeeded") and "analyze_image" in context.cache:
                decision = decide(context.cache["analyze_image"], self.policy, context.cache.get("review_blood"))
                suggestion = render(decision)
                log["decision"] = decision
            log["evidence_text"] = context.cache.get("analyze_image_text", "")
            log["messages"] = [
                {"type": message.type, "name": getattr(message, "name", None),
                 "content": message.content,
                 "tool_calls": getattr(message, "tool_calls", [])}
                for message in result.get("messages", [])
            ]
            log.update({
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
            if "analyze_image" in context.cache:
                decision = decide(context.cache["analyze_image"], self.policy, context.cache.get("review_blood"))
                suggestion = render(decision)
                log["decision"] = decision
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
