# -*- coding: utf-8 -*-
"""审核 Agent 工具。

图片与工具缓存保存在请求级 ContextVar 中，LLM 只看到工具返回的文本证据。
analyze_image 必须成功执行；其余证据工具由 ReAct Agent 按需选择。
"""

from __future__ import annotations

import contextvars
import json
from dataclasses import dataclass, field
from typing import Any, Optional

from langchain_core.tools import BaseTool, tool

from . import lowlevel as lowlevel_mod
from . import rag as rag_mod


@dataclass
class RequestToolContext:
    image: bytes
    cache: dict[str, Any] = field(default_factory=dict)


_current_request: contextvars.ContextVar[Optional[RequestToolContext]] = contextvars.ContextVar(
    "review_tool_context", default=None
)


def set_current_image(image: bytes) -> contextvars.Token:
    return _current_request.set(RequestToolContext(image=image))


def reset_current_image(token: contextvars.Token) -> None:
    _current_request.reset(token)


def get_request_context() -> RequestToolContext:
    context = _current_request.get()
    if context is None or not context.image:
        raise RuntimeError("当前审核请求没有可用图片")
    return context


def _as_dict(output: Any) -> dict[str, Any]:
    if hasattr(output, "to_dict"):
        return output.to_dict()
    if isinstance(output, dict):
        return output
    return {
        "risk": getattr(output, "risk", {}),
        "type": getattr(output, "type", {}),
        "injury": getattr(output, "injury", {}),
        "quality": getattr(output, "quality", {}),
        "conf": getattr(output, "conf", 0.0),
        "trained": getattr(output, "trained", False),
    }


def text_evidence(output: Any) -> str:
    """把冻结标签体系下的结构化视觉输出转成 LLM 证据。"""
    data = _as_dict(output)
    risk = data.get("risk", {})
    violence = data.get("type", {})
    injury = data.get("injury", {})
    quality = data.get("quality", {})
    return (
        f"违规等级: {risk.get('label', '')} (置信度 {float(risk.get('confidence', 0)):.2f}) | "
        f"暴力类型: {violence.get('labels', [])} | "
        f"血腥程度: {injury.get('label', '')} (置信度 {float(injury.get('confidence', 0)):.2f}) | "
        f"图像质量（多标签）: {quality.get('labels', [])} (各类概率 {quality.get('scores', {})}) | "
        f"审核头已训练: {bool(data.get('trained', False))}"
    )


def build_tools(
    engine: Any,
    ocr_client: Any = None,
    retriever: Any = None,
    *,
    enable_ocr: bool = True,
    enable_lowlevel: bool = True,
    enable_rag: bool = True,
) -> list[BaseTool]:
    """构造请求工具；依赖通过闭包捕获，不使用模块级共享服务。"""
    retrieve = retriever or rag_mod.retrieve

    @tool
    def analyze_image() -> str:
        """必须调用一次。使用 FG-CLIP2 分析当前图片，返回违规等级、暴力类型、血腥程度和图像质量。"""
        context = get_request_context()
        if "analyze_image_text" not in context.cache:
            output = engine.predict(context.image)
            context.cache["analyze_image"] = _as_dict(output)
            context.cache["analyze_image_text"] = text_evidence(output)
        return context.cache["analyze_image_text"]

    @tool
    def ocr_image() -> str:
        """可选。使用 GLM-OCR 提取广告、标语、字幕、水印等文字；只返回文字证据，不分析图片语义。"""
        context = get_request_context()
        if "ocr_image" not in context.cache:
            result = ocr_client.recognize(context.image) if ocr_client else {
                "success": False, "available": False, "texts": [], "boxes": [],
                "error": "not_configured",
            }
            context.cache["ocr_image"] = result
        result = context.cache["ocr_image"]
        if not result.get("success", False):
            if not result.get("available", True):
                return "GLM-OCR 未配置，本次没有获得文字证据"
            return f"GLM-OCR 调用失败({result.get('error', 'unknown')})，本次没有获得文字证据"
        texts = result.get("texts", []) or []
        return "未检出文字" if not texts else "图中文字: " + "；".join(map(str, texts))

    @tool
    def analyze_lowlevel() -> str:
        """可选。CLIP 低置信、结果矛盾或怀疑红色、模糊、打码、密集纹理时调用；仅提供像素提示，不做裁决。"""
        context = get_request_context()
        if "analyze_lowlevel" not in context.cache:
            context.cache["analyze_lowlevel"] = lowlevel_mod.extract(context.image)
        return lowlevel_mod.to_text(context.cache["analyze_lowlevel"])

    @tool
    def search_knowledge(query: str) -> str:
        """可选。需要审核规则或相似判例佐证时调用；知识库为空时返回无相似判例。"""
        context = get_request_context()
        cache_key = f"search_knowledge:{query}"
        if cache_key not in context.cache:
            context.cache[cache_key] = retrieve(query, k=5)
        cases = context.cache[cache_key]
        return "无相似判例" if not cases else json.dumps(cases, ensure_ascii=False)

    @tool
    def finalize_review(suggestion: str) -> str:
        """提交面向运营的一句话最终审核建议。调用守卫会拒绝任何缺少成功 CLIP 分析的提交。"""
        return suggestion.strip()

    tools: list[BaseTool] = [analyze_image]
    if enable_ocr:
        tools.append(ocr_image)
    if enable_lowlevel:
        tools.append(analyze_lowlevel)
    if enable_rag:
        tools.append(search_knowledge)
    tools.append(finalize_review)
    return tools


def tools_by_name(tools: list[BaseTool]) -> dict[str, BaseTool]:
    return {item.name: item for item in tools}
