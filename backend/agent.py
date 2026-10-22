# -*- coding: utf-8 -*-
"""带 CLIP 调用守卫的受控 ReAct 状态图。"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from .prompts import SYSTEM_PROMPT
from .state import ReviewState
from .tools import (
    get_request_context, knowledge_cache_key, tools_by_name,
)


def build_agent(
    llm_model: Any,
    tools: list,
    system_prompt: str = SYSTEM_PROMPT,
    max_rounds: int = 5,
    on_tool_fail: str = "skip",
):
    """构建受控 ReAct 图：CLIP 必须成功，其他工具由模型按需调用。"""
    tool_map = tools_by_name(tools)
    model_with_tools = llm_model.bind_tools(tools)

    def call_model(state: ReviewState) -> dict:
        response = model_with_tools.invoke(
            [SystemMessage(content=system_prompt), *state.get("messages", [])]
        )
        return {"messages": [response], "rounds": state.get("rounds", 0) + 1}

    def execute_tools(state: ReviewState) -> dict:
        message = state["messages"][-1]
        tool_results = dict(state.get("tool_results", {}))
        steps = list(state.get("steps", []))
        clip_called = state.get("clip_called", False)
        clip_succeeded = state.get("clip_succeeded", False)
        finalized = state.get("finalized", False)
        degraded = state.get("degraded", False)
        suggestion = state.get("suggestion", "")
        output_messages: list[ToolMessage] = []

        for call in getattr(message, "tool_calls", []) or []:
            name = call.get("name", "")
            args = call.get("args", {}) or {}
            call_id = call.get("id", name or "tool")

            if name == "finalize_review" and not clip_succeeded:
                content = "调用守卫拒绝提交：必须先成功调用 analyze_image 获取 CLIP 视觉证据。"
                output_messages.append(ToolMessage(content=content, tool_call_id=call_id, name=name))
                steps.append({"type": "guard_reject", "name": name, "reason": "clip_required"})
                continue

            selected = tool_map.get(name)
            if selected is None:
                content = f"未知工具: {name}"
                output_messages.append(ToolMessage(content=content, tool_call_id=call_id, name=name))
                steps.append({"type": "tool_error", "name": name, "error": content})
                continue

            cached_before = False
            try:
                context = get_request_context()
                cache_name = name if name != "search_knowledge" else knowledge_cache_key(str(args.get("query", "")))
                cached_before = cache_name in context.cache or (
                    name == "analyze_image" and "analyze_image_text" in context.cache
                )
                content = str(selected.invoke(args))
                if name == "analyze_image":
                    clip_called = True
                    clip_succeeded = True
                    tool_results[name] = context.cache.get("analyze_image", content)
                elif name == "finalize_review":
                    suggestion = content.strip()
                    finalized = True
                    tool_results[name] = suggestion
                elif name == "search_knowledge":
                    searches = dict(tool_results.get(name, {}))
                    searches[cache_name] = context.cache.get(cache_name, content)
                    tool_results[name] = searches
                else:
                    tool_results[name] = context.cache.get(cache_name, content)
                steps.append({"type": "tool_call", "name": name, "cached": cached_before, "ok": True, "args": args, "output": content})
                if name == "analyze_image" and "review_blood" in context.cache:
                    review = context.cache["review_blood"]
                    tool_results["review_blood"] = review
                    steps.append({"type": "tool_call", "name": "review_blood", "automatic": True,
                                  "cached": cached_before, "ok": review.get("status") == "ok",
                                  "args": {}, "output": review})
                output_messages.append(ToolMessage(content=content, tool_call_id=call_id, name=name))
                if name == "analyze_image" and context.cache.get("l0_short_circuit"):
                    finalized = True
                    suggestion = "CLIP 风险头判定为 L0_安全，未发现需要继续取证的内容冲突。"
                    break
                if finalized:
                    break
            except Exception as exc:  # noqa: BLE001
                degraded = True
                content = f"工具 {name} 调用失败: {type(exc).__name__}: {exc}"
                if name == "analyze_image":
                    clip_called = True
                    clip_succeeded = False
                    finalized = True
                    suggestion = "视觉模型暂不可用，无法完成图片审核，请稍后重试。"
                elif on_tool_fail == "abort":
                    finalized = True
                    suggestion = "审核工具暂不可用，无法完成本次审核，请稍后重试。"
                output_messages.append(ToolMessage(content=content, tool_call_id=call_id, name=name))
                steps.append({"type": "tool_error", "name": name, "ok": False, "error": content})

        return {
            "messages": output_messages,
            "tool_results": tool_results,
            "steps": steps,
            "clip_called": clip_called,
            "clip_succeeded": clip_succeeded,
            "finalized": finalized,
            "degraded": degraded,
            "suggestion": suggestion,
        }

    def guard_retry(_: ReviewState) -> dict:
        return {
            "messages": [HumanMessage(content="你尚未取得 CLIP 视觉证据。请立即调用 analyze_image，不要直接作答。")],
            "steps": [{"type": "guard_reject", "reason": "clip_required"}],
        }

    def implicit_finalize(state: ReviewState) -> dict:
        last = state.get("messages", [])[-1]
        content = str(getattr(last, "content", "") or "").strip()
        return {"suggestion": content or "视觉证据已取得，由程序处置规则生成结论。", "finalized": True}

    def limit_finalize(state: ReviewState) -> dict:
        if state.get("clip_succeeded"):
            evidence = state.get("tool_results", {}).get("analyze_image", {})
            risk = evidence.get("risk", {}).get("label", "未知等级") if isinstance(evidence, dict) else "未知等级"
            suggestion = f"视觉模型判定为{risk}；Agent 已达到最大轮次，由程序处置规则生成结论。"
        else:
            suggestion = "未能取得 CLIP 视觉证据，无法完成图片审核，请稍后重试。"
        return {"suggestion": suggestion, "finalized": True, "degraded": True}

    def route_after_agent(state: ReviewState) -> str:
        last = state.get("messages", [])[-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        if state.get("clip_succeeded"):
            return "implicit_finalize"
        if state.get("rounds", 0) >= max_rounds:
            return "limit_finalize"
        return "guard_retry"

    def route_after_tools(state: ReviewState) -> str:
        if state.get("finalized"):
            return END
        if state.get("rounds", 0) >= max_rounds:
            return "limit_finalize"
        return "agent"

    graph = StateGraph(ReviewState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", execute_tools)
    graph.add_node("guard_retry", guard_retry)
    graph.add_node("implicit_finalize", implicit_finalize)
    graph.add_node("limit_finalize", limit_finalize)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent)
    graph.add_conditional_edges("tools", route_after_tools)
    graph.add_edge("guard_retry", "agent")
    graph.add_edge("implicit_finalize", END)
    graph.add_edge("limit_finalize", END)
    return graph.compile()
