# -*- coding: utf-8 -*-
"""FastAPI 接口：上传图片只返回 id 与 suggestion。"""

from __future__ import annotations

import io
import uuid
from typing import Optional

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from .schemas import ErrorResponse, HealthResponse, ReviewResponse
from .service import ReviewService

app = FastAPI(title="违规内容审核系统", version="0.4")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(?:localhost|127\.0\.0\.1)(?::[0-9]+)?",
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
_service: Optional[ReviewService] = None
_max_image_bytes = 10 * 1024 * 1024
_allowed_formats = {"JPEG", "PNG", "WEBP"}


class APIError(Exception):
    def __init__(self, status_code: int, code: int, message: str, request_id: str = ""):
        self.status_code = status_code
        self.payload = ErrorResponse(code=code, message=message, id=request_id)


@app.exception_handler(APIError)
def handle_api_error(_: Request, exc: APIError):
    return JSONResponse(status_code=exc.status_code, content=exc.payload.model_dump())


@app.exception_handler(RequestValidationError)
def handle_validation_error(_: Request, __: RequestValidationError):
    payload = ErrorResponse(code=40002, message="缺少必填字段 image")
    return JSONResponse(status_code=400, content=payload.model_dump())


def init_service(service: ReviewService, max_image_mb: int = 10) -> ReviewService:
    global _service, _max_image_bytes
    _service = service
    _max_image_bytes = max_image_mb * 1024 * 1024
    return service


def _validate_image(data: bytes, request_id: str) -> None:
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in _allowed_formats:
                raise APIError(400, 40001, "仅支持 jpg、png、webp 图片", request_id)
            image.verify()
    except APIError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise APIError(400, 40001, "无法解析图片文件", request_id) from exc


@app.get("/health", response_model=HealthResponse)
def health():
    ready = bool(_service and _service.ready)
    engine = getattr(_service, "engine", None) if _service else None
    model_loaded = bool(getattr(engine, "loaded", False) or getattr(engine, "is_mock", False))
    return HealthResponse(
        status="ok" if ready else "not_ready",
        service_ready=ready,
        model_loaded=model_loaded,
        mock=bool(getattr(_service, "mock_mode", False)),
    )


@app.post("/api/review", response_model=ReviewResponse)
def review(image: UploadFile = File(...)):
    request_id = uuid.uuid4().hex
    if _service is None:
        raise APIError(503, 50300, "审核服务尚未初始化", request_id)
    if not image.filename:
        raise APIError(400, 40002, "缺少必填字段 image", request_id)
    data = image.file.read(_max_image_bytes + 1)
    if not data:
        raise APIError(400, 40001, "图片为空", request_id)
    if len(data) > _max_image_bytes:
        raise APIError(400, 40001, f"图片大小不能超过 {_max_image_bytes // 1024 // 1024}MB", request_id)
    _validate_image(data, request_id)
    try:
        return _service.review(data, filename=image.filename)
    except Exception as exc:  # noqa: BLE001
        raise APIError(500, 50000, "审核服务内部错误", request_id) from exc
