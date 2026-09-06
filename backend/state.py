# -*- coding: utf-8 -*-
"""受控 ReAct 图的请求状态。"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class ReviewState(TypedDict, total=False):
    messages: Annotated[list[Any], add_messages]
    remaining_steps: int
    filename: str
    tool_results: dict[str, Any]
    steps: list[dict[str, Any]]
    suggestion: str
    rounds: int
    clip_called: bool
    clip_succeeded: bool
    finalized: bool
    degraded: bool
    error: str
