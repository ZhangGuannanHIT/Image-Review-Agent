# -*- coding: utf-8 -*-
"""后端请求/响应 pydantic 模型。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ReviewTool(BaseModel):
    id: str
    label: str
    status: str
    status_label: str
    outcome: str
    outcome_label: str
    summary: str
    duration_ms: int | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ReviewGroup(BaseModel):
    id: str
    label: str
    description: str
    tools: list[ReviewTool] = Field(default_factory=list)


class ReviewResult(BaseModel):
    action: str
    action_label: str
    tone: str
    effective_risk: str
    risk_raw: str | None = None
    domain: str
    domain_label: str
    domain_score: float | None = None
    domain_source: str
    confidence: float | None = None
    reason: str
    primary_reasons: list[str] = Field(default_factory=list)
    adjustments: list[str] = Field(default_factory=list)
    conflict_type: str
    uncertain: bool
    policy_version: str | None = None
    suggestion_source: str
    degraded: bool
    recommended_action: str
    recommended_action_label: str
    review_hint: bool
    review_reason: str


class TimingStage(BaseModel):
    id: str
    label: str
    category: str
    duration_ms: int


class ReviewTiming(BaseModel):
    total_ms: int
    stages: list[TimingStage] = Field(default_factory=list)


class ReviewResponse(BaseModel):
    """Backward-compatible response with structured review evidence."""

    id: str
    suggestion: str
    schema_version: str | None = None
    result: ReviewResult | None = None
    groups: list[ReviewGroup] = Field(default_factory=list)
    timing: ReviewTiming | None = None


class FeedbackRequest(BaseModel):
    resolution: Literal["confirm", "adjust", "reject"]
    final_action: Literal["allow", "limit", "review", "block"] = "review"
    adjustment_direction: Literal["downgrade", "upgrade", ""] = ""
    reason: str = Field(default="", max_length=1000)


class ErrorResponse(BaseModel):
    """统一错误响应（非 200）。"""
    code: int
    message: str
    id: str = ""


class HealthResponse(BaseModel):
    status: str
    service_ready: bool
    model_loaded: bool
    mock: bool = False
