# -*- coding: utf-8 -*-
"""审核 Agent 工具。

图片与工具缓存保存在请求级 ContextVar 中，LLM 只看到工具返回的文本证据。
analyze_image 必须成功执行；其余证据工具由 ReAct Agent 按需选择。
"""

from __future__ import annotations

import contextvars
import hashlib
import io
import json
from dataclasses import dataclass, field
from typing import Any, Optional

from langchain_core.tools import BaseTool, tool
from PIL import Image

from . import lowlevel as lowlevel_mod
from . import rag as rag_mod
from .scene_concepts import build_scene_evidence
from .vision import VisionGate, trigger


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


def knowledge_cache_key(query: str) -> str:
    """用查询摘要作为缓存键，避免 OCR 原文进入日志。"""
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
    return f"search_knowledge:{digest}"


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


def is_l0_without_conflict(evidence: dict[str, Any]) -> bool:
    """True only for a high-confidence L0 result with no contradictory evidence."""
    if evidence.get("risk", {}).get("label") != "L0_安全":
        return False
    confidence = evidence.get("risk", {}).get("confidence")
    if not isinstance(confidence, (int, float)) or confidence < 0.75:
        return False
    quality_labels = set(evidence.get("quality", {}).get("labels", []))
    if quality_labels & {"严重模糊", "文字扭曲形变", "故意遮挡（马赛克/黑条/贴纸）"}:
        return False
    type_labels = set(evidence.get("type", {}).get("labels", []))
    if type_labels - {"无暴力"}:
        return False
    injury_label = evidence.get("injury", {}).get("label", "")
    if injury_label and injury_label != "无血腥":
        return False
    binary = evidence.get("blood_binary")
    if isinstance(binary, dict):
        score = binary.get("p_blood")
        threshold = binary.get("threshold")
        if (
            isinstance(score, (int, float))
            and isinstance(threshold, (int, float))
            and score >= threshold * 0.5
        ):
            return False
    return True


def text_evidence(output: Any) -> str:
    """把冻结标签体系下的结构化视觉输出转成 LLM 证据。"""
    data = _as_dict(output)
    return (
        "以下是模型预测，类别括号内是标签定义，不是观察到的具体事实。分数未经概率校准。"
        "type（暴力类型）和 quality（图像质量）为多标签，risk 和 injury 为单标签。"
        "blood_binary 是独立校准的血腥存在性证据，其阈值来自验证集，不能无视。"
        "quality 标签可能表示严重模糊、文字扭曲或故意遮挡，程序会据此提高人工复核优先级。"
        "无暴力仅指无正在进行的暴力行为，可以与血腥、后果展示和暗示威胁共存。"
        "保留各类分数与阈值供比较，不能只凭最高类别断言存在伤口或血泊："
        + json.dumps(data, ensure_ascii=False)
    )



def build_tools(
    engine: Any,
    ocr_client: Any = None,
    retriever: Any = None,
    *,
    enable_ocr: bool = True,
    enable_lowlevel: bool = True,
    enable_rag: bool = True,
    vision_client: Any = None,
    vision_gate: VisionGate | None = None,
    scene_scorer: Any = None,
    scene_options: dict | None = None,
) -> list[BaseTool]:
    """构造请求工具；依赖通过闭包捕获，不使用模块级共享服务。"""
    retrieve = retriever or rag_mod.retrieve
    gate = vision_gate or VisionGate()
    scene_options = scene_options or {}

    @tool
    def review_blood() -> str:
        """仅复核低可信血腥候选。必须先完成 CLIP；程序检查分数/分差，不满足则禁止发送图片。"""
        context = get_request_context()
        condition = trigger(context.cache.get("analyze_image", {}), gate)
        if not condition["eligible"]:
            return json.dumps({"status": "skipped", "trigger": condition}, ensure_ascii=False)
        if "review_blood" not in context.cache:
            try:
                result = vision_client.review(context.image) if vision_client else {
                    "status": "failed", "verdict": "uncertain", "error": "not_configured"}
            except Exception as exc:
                result = {"status": "failed", "verdict": "uncertain", "error": type(exc).__name__}
            context.cache["review_blood"] = {**result, "trigger": condition}
        return json.dumps(context.cache["review_blood"], ensure_ascii=False)

    @tool
    def analyze_image() -> str:
        """必须调用一次。使用 FG-CLIP2 分析当前图片，返回违规等级、暴力类型、血腥程度和图像质量。"""
        context = get_request_context()
        if "analyze_image_text" not in context.cache:
            output = engine.predict(context.image)
            context.cache["analyze_image"] = _as_dict(output)
            context.cache["analyze_image_text"] = text_evidence(output)
            text_probe_ok = True
            text_suspected = False
            try:
                lowlevel = lowlevel_mod.extract(context.image)
                text_suspected = lowlevel_mod.has_much_text(lowlevel)
                context.cache["analyze_lowlevel"] = lowlevel
                context.cache["analyze_lowlevel_source"] = "automatic_text_probe"
            except Exception as exc:  # noqa: BLE001
                # 文字探测失败不能等同于“没有文字”，保守地禁止 L0 短路。
                text_probe_ok = False
                context.cache["text_probe_error"] = type(exc).__name__
            context.cache["text_probe_ok"] = text_probe_ok
            context.cache["text_suspected"] = text_suspected
            context.cache["l0_short_circuit"] = bool(
                is_l0_without_conflict(context.cache["analyze_image"])
                and text_probe_ok
                and not text_suspected
            )
        content = context.cache["analyze_image_text"]
        if vision_client and trigger(context.cache["analyze_image"], gate)["eligible"]:
            content += "\n程序按触发条件调用 review_blood 的独立复核结果：" + review_blood.invoke({})
        quality_labels = context.cache["analyze_image"].get("quality", {}).get("labels", [])
        short_circuit = bool(context.cache.get("l0_short_circuit", False))
        text_suspected = bool(context.cache.get("text_suspected", False))
        text_probe_ok = bool(context.cache.get("text_probe_ok", False))
        if (
            not short_circuit
            and enable_ocr
            and ocr_client
            and (
                "文字扭曲形变" in quality_labels
                or text_suspected
                or not text_probe_ok
            )
        ):
            reason = (
                "quality 命中文字扭曲"
                if "文字扭曲形变" in quality_labels
                else "低层文字探测失败" if not text_probe_ok else "检测到疑似文字"
            )
            content += f"\n程序因{reason}而自动调用 OCR：" + ocr_image.invoke({})
        risk_label = context.cache["analyze_image"].get("risk", {}).get("label")
        if not short_circuit and scene_scorer is not None and risk_label != "L0_安全":
            try:
                content += "\n程序自动调用场景概念匹配：" + match_scene_concepts.invoke({})
            except Exception as exc:  # noqa: BLE001
                content += (
                    "\n场景概念工具失败："
                    f"{type(exc).__name__}；基础 CLIP 证据仍然有效"
                )
        return content

    @tool
    def ocr_image() -> str:
        """可选。使用 OCR 提取广告、标语、字幕、水印等文字及置信度；只返回文字证据，不分析图片语义。"""
        context = get_request_context()
        if (
            "analyze_image" in context.cache
            and context.cache.get("l0_short_circuit")
        ):
            return "CLIP 已判定 L0 且无内容冲突，程序跳过 OCR。"
        if "ocr_image" not in context.cache:
            result = ocr_client.recognize(context.image) if ocr_client else {
                "success": False, "available": False, "texts": [], "boxes": [],
                "error": "not_configured",
            }
            context.cache["ocr_image"] = result
        result = context.cache["ocr_image"]
        if not result.get("success", False):
            if not result.get("available", True):
                return "OCR 未配置，本次没有获得文字证据"
            return f"OCR 调用失败({result.get('error', 'unknown')})，本次没有获得文字证据"
        texts = result.get("texts", []) or []
        if not texts:
            return "未检出文字"
        confidence = result.get("mean_confidence")
        confidence_text = (
            f"；平均置信度 {confidence:.3f}"
            if isinstance(confidence, (int, float)) else ""
        )
        return "图中文字: " + "；".join(map(str, texts)) + confidence_text

    @tool
    def match_scene_concepts() -> str:
        """仅在 CLIP 风险非 L0 时调用。返回候选违规场景、安全语境、RAG 命中和图文冲突提示。"""
        context = get_request_context()
        evidence = context.cache.get("analyze_image")
        if not isinstance(evidence, dict):
            raise RuntimeError("必须先成功调用 analyze_image")
        risk_label = evidence.get("risk", {}).get("label", "")
        if risk_label == "L0_安全":
            return json.dumps({
                "status": "skipped",
                "reason": "risk_l0",
                "action_effect": "none",
            }, ensure_ascii=False)
        if scene_scorer is None:
            return json.dumps({
                "status": "skipped",
                "reason": "not_configured",
                "action_effect": "none",
            }, ensure_ascii=False)
        if "match_scene_concepts" not in context.cache:
            with Image.open(io.BytesIO(context.image)) as opened:
                raw = scene_scorer.score(opened.convert("RGB"))
            context.cache["match_scene_concepts"] = build_scene_evidence(
                raw,
                risk_margin_threshold=float(
                    scene_options.get("risk_margin_threshold", -0.001031)
                ),
                safe_context_threshold=float(
                    scene_options.get("safe_context_threshold", 0.006)
                ),
                top_k=int(scene_options.get("top_k", 3)),
                retrieve=retrieve,
            )
        return json.dumps(context.cache["match_scene_concepts"], ensure_ascii=False)

    @tool
    def analyze_lowlevel() -> str:
        """可选。需要颜色、边缘、方差等数值时调用；不识别血腥、遮挡或违规。"""
        context = get_request_context()
        if (
            "analyze_image" in context.cache
            and context.cache.get("l0_short_circuit")
        ):
            return "CLIP 已判定 L0 且无内容冲突，程序跳过低层分析。"
        if "analyze_lowlevel" not in context.cache:
            context.cache["analyze_lowlevel"] = lowlevel_mod.extract(context.image)
        return lowlevel_mod.to_text(context.cache["analyze_lowlevel"])

    @tool
    def search_knowledge(query: str) -> str:
        """可选。在取得 CLIP 证据后，按原样标签、置信度和 OCR 文字检索规则或判例；不得添加未观察到的图片事实。"""
        context = get_request_context()
        if "analyze_image_text" not in context.cache:
            raise RuntimeError("必须先成功调用 analyze_image，才能检索规则或判例")
        if context.cache.get("l0_short_circuit"):
            return "CLIP 已判定 L0 且无内容冲突，程序跳过 RAG 检索。"
        cache_key = knowledge_cache_key(query)
        if cache_key not in context.cache:
            context.cache[cache_key] = retrieve(query, k=5)
        cases = context.cache[cache_key]
        return "未检索到相关规则或判例" if not cases else json.dumps(cases, ensure_ascii=False)

    @tool
    def finalize_review(suggestion: str) -> str:
        """提交面向运营的一句话最终审核建议。调用守卫会拒绝任何缺少成功 CLIP 分析的提交。"""
        return suggestion.strip()

    tools: list[BaseTool] = [analyze_image]
    if vision_client:
        tools.append(review_blood)
    if enable_ocr:
        tools.append(ocr_image)
    if enable_lowlevel:
        tools.append(analyze_lowlevel)
    if enable_rag:
        tools.append(search_knowledge)
    if scene_scorer is not None:
        tools.append(match_scene_concepts)
    tools.append(finalize_review)
    return tools


def tools_by_name(tools: list[BaseTool]) -> dict[str, BaseTool]:
    return {item.name: item for item in tools}
