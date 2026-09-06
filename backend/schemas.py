# -*- coding: utf-8 -*-
"""后端请求/响应 pydantic 模型。

冻结契约：API 响应仅含 id + suggestion 两个字段。
"""

from __future__ import annotations

from pydantic import BaseModel


class ReviewResponse(BaseModel):
    """审核响应（前端唯一消费的两个字段）。"""
    id: str                     # 本次审核唯一编号
    suggestion: str             # 审核建议正文


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
