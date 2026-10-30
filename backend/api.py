# -*- coding: utf-8 -*-
"""FastAPI 接口：上传图片返回兼容建议和结构化审核证据。"""

from __future__ import annotations

import io
import re
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    Response,
)
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from PIL.Image import DecompressionBombError

from .schemas import (
    ErrorResponse,
    FeedbackRequest,
    HealthResponse,
    ReviewResponse,
)
from .service import ReviewService

app = FastAPI(title="违规内容审核系统", version="0.5")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(?:localhost|127\.0\.0\.1)(?::[0-9]+)?",
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
_service: Optional[ReviewService] = None
_max_image_bytes = 10 * 1024 * 1024
_allowed_formats = {"JPEG", "PNG", "WEBP"}
_FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
_NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, max-age=0",
    "Pragma": "no-cache",
}


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


def _record_store():
    store = getattr(_service, "record_store", None) if _service else None
    if store is None:
        raise APIError(503, 50300, "审核历史存储尚未初始化")
    return store


def _validated_workspace_id(value: str, request_id: str = "") -> str:
    workspace_id = (value or "default").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", workspace_id):
        raise APIError(400, 40002, "workspace_id 格式无效", request_id)
    return workspace_id


def _serve_static(filename: str, media_type: str) -> Response:
    """只提供前端目录下固定的静态文件，避免暴露其他路径。"""
    allowed = {"index.html", "styles.css", "landing.css", "config.js", "app.js"}
    if filename not in allowed:
        raise APIError(404, 50000, "资源不存在")
    path = _FRONTEND_DIR / filename
    if not path.is_file():
        raise APIError(404, 50000, "资源不存在")
    return Response(
        content=path.read_bytes(),
        media_type=media_type,
        headers=_NO_CACHE_HEADERS,
    )


@app.get("/", include_in_schema=False)
def index() -> HTMLResponse:
    path = _FRONTEND_DIR / "index.html"
    if not path.is_file():
        raise APIError(404, 50000, "前端页面不存在")
    return HTMLResponse(
        content=path.read_text(encoding="utf-8"),
        headers=_NO_CACHE_HEADERS,
    )


@app.get("/styles.css", include_in_schema=False)
def styles() -> Response:
    return _serve_static("styles.css", "text/css")


@app.get("/landing.css", include_in_schema=False)
def landing_styles() -> Response:
    return _serve_static("landing.css", "text/css")


@app.get("/app.js", include_in_schema=False)
def app_js() -> Response:
    return _serve_static("app.js", "text/javascript")


@app.get("/config.js", include_in_schema=False)
def frontend_config() -> Response:
    return _serve_static("config.js", "text/javascript")


def _validate_image(data: bytes, request_id: str) -> None:
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in _allowed_formats:
                raise APIError(400, 40001, "仅支持 jpg、png、webp 图片", request_id)
            if Image.MAX_IMAGE_PIXELS and image.width * image.height > Image.MAX_IMAGE_PIXELS:
                raise APIError(400, 40001, "图片像素尺寸过大", request_id)
            image.verify()
    except APIError:
        raise
    except DecompressionBombError as exc:
        raise APIError(400, 40001, "图片像素尺寸过大", request_id) from exc
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
def review(
    image: UploadFile = File(...),
    workspace_id: str = Form(default="default"),
):
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
    workspace_id = _validated_workspace_id(workspace_id, request_id)
    try:
        return _service.review(
            data,
            filename=image.filename,
            workspace_id=workspace_id,
        )
    except Exception as exc:  # noqa: BLE001
        raise APIError(500, 50000, "审核服务内部错误", request_id) from exc


@app.get("/api/records")
def records(
    domain: str = Query(default=""),
    action: str = Query(default=""),
    risk: str = Query(default=""),
    workspace_id: str = Query(default=""),
    limit: int = Query(default=200, ge=1, le=500),
):
    store = _record_store()
    return {
        "items": store.list_records(
            domain=domain,
            action=action,
            risk=risk,
            workspace_id=workspace_id,
            limit=limit,
        ),
        "total": store.count_records(workspace_id=workspace_id),
    }


@app.get("/api/records/export.csv", include_in_schema=False)
def export_records(workspace_id: str = Query(default="")):
    content = _record_store().export_csv(
        workspace_id=_validated_workspace_id(workspace_id) if workspace_id else ""
    )
    return Response(
        content=content,
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": 'attachment; filename="review-records.csv"',
            "Cache-Control": "no-store",
        },
    )


@app.post("/api/records/clear")
def clear_records(workspace_id: str = Query(default="default")):
    store = _record_store()
    workspace_id = _validated_workspace_id(workspace_id)
    cleared, image_paths = store.clear_records(workspace_id=workspace_id)
    image_root = (Path(_service.log_dir) / "images").resolve()
    for path in image_paths:
        resolved = path.resolve()
        if image_root not in resolved.parents:
            continue
        try:
            resolved.unlink(missing_ok=True)
        except OSError:
            continue
    return {"workspace_id": workspace_id, "cleared": cleared}


@app.get("/api/records/{record_id}/image", include_in_schema=False)
def record_image(record_id: str):
    store = _record_store()
    path = store.image_path(record_id)
    if path is None:
        raise APIError(404, 40401, "审核原图不存在", record_id)
    image_root = (Path(_service.log_dir) / "images").resolve()
    resolved = path.resolve()
    if image_root not in resolved.parents:
        raise APIError(404, 40401, "审核原图不存在", record_id)
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    return FileResponse(
        resolved,
        media_type=media_types.get(resolved.suffix.lower(), "application/octet-stream"),
        headers=_NO_CACHE_HEADERS,
    )


@app.get("/api/records/{record_id}")
def record_detail(record_id: str):
    record = _record_store().get_record(record_id)
    if record is None:
        raise APIError(404, 40401, "审核记录不存在", record_id)
    return record


@app.post("/api/records/{record_id}/feedback")
def record_feedback(record_id: str, feedback: FeedbackRequest):
    try:
        record = _record_store().record_feedback(
            record_id,
            resolution=feedback.resolution,
            final_action=feedback.final_action,
            adjustment_direction=feedback.adjustment_direction,
            reason=feedback.reason,
        )
    except ValueError as exc:
        raise APIError(400, 40003, str(exc), record_id) from exc
    if record is None:
        raise APIError(404, 40401, "审核记录不存在", record_id)
    return record
