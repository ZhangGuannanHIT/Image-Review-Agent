# -*- coding: utf-8 -*-
"""审核服务：建立请求上下文、运行带调用守卫的 Agent、记录轨迹。"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

from langchain_core.messages import HumanMessage

from . import tools as tools_mod
from .agent import build_agent
from .parser import extract_suggestion
from .prompts import build_system_prompt
from .schemas import ReviewResponse


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
        os.makedirs(log_dir, exist_ok=True)

    @property
    def ready(self) -> bool:
        return self.agent is not None and self.engine is not None

    def _log(self, record: dict[str, Any]) -> None:
        path = os.path.join(self.log_dir, "review.log.jsonl")
        with open(path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def review(self, image_bytes: bytes, filename: str = "") -> ReviewResponse:
        request_id = uuid.uuid4().hex
        token = tools_mod.set_current_image(image_bytes)
        log: dict[str, Any] = {
            "request_id": request_id,
            "filename": filename,
            "mock": self.mock_mode,
        }
        try:
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
            log.update({
                "clip_called": result.get("clip_called", False),
                "clip_succeeded": result.get("clip_succeeded", False),
                "finalized": result.get("finalized", False),
                "degraded": result.get("degraded", False),
                "tool_results": result.get("tool_results", {}),
                "tool_cache_keys": sorted(context.cache.keys()),
                "steps": result.get("steps", []),
                "suggestion": suggestion,
            })
            self._log(log)
            return ReviewResponse(id=request_id, suggestion=suggestion)
        except Exception as exc:  # noqa: BLE001
            suggestion = "审核流程暂不可用，无法完成本次审核，请稍后重试。"
            log.update({
                "error": f"{type(exc).__name__}: {exc}",
                "degraded": True,
                "suggestion": suggestion,
            })
            self._log(log)
            return ReviewResponse(id=request_id, suggestion=suggestion)
        finally:
            tools_mod.reset_current_image(token)
