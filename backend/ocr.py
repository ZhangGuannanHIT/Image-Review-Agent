# -*- coding: utf-8 -*-
"""智谱 GLM-OCR 适配器。

只负责字符与版面提取，不要求模型描述图片语义。官方接口：
POST /api/paas/v4/layout_parsing，JSON 字段 file 支持 URL 或 Base64。
"""

from __future__ import annotations

import base64
import io
import os
import re
from dataclasses import dataclass
from typing import Any, Optional

import requests
import yaml
from PIL import Image, UnidentifiedImageError


@dataclass(frozen=True)
class GLMOCRConfig:
    provider: str = "glm_ocr"
    model: str = "glm-ocr"
    endpoint: str = "https://open.bigmodel.cn/api/paas/v4/layout_parsing"
    api_key: str = ""
    timeout: int = 20
    max_image_mb: int = 10
    return_crop_images: bool = False
    need_layout_visualization: bool = False

    @classmethod
    def from_yaml(cls, path: str) -> "GLMOCRConfig":
        with open(path, "r", encoding="utf-8") as file:
            raw = yaml.safe_load(file) or {}
        section = raw.get("glm_ocr", raw)
        configured = cls(**section)
        api_key = os.getenv("OCR_API_KEY") or configured.api_key
        return cls(**{**configured.__dict__, "api_key": api_key})


def _failure(config: GLMOCRConfig, error: str, available: bool = True) -> dict[str, Any]:
    return {
        "success": False,
        "available": available,
        "provider": config.provider,
        "model": config.model,
        "texts": [],
        "boxes": [],
        "box_format": "normalized_xyxy",
        "request_id": "",
        "error": error,
    }


def _prepare_image(image_bytes: bytes, max_bytes: int) -> tuple[bytes, str]:
    """GLM-OCR 官方支持 JPG/PNG；其他可解码格式统一转换为 PNG。"""
    if not image_bytes:
        raise ValueError("empty_image")
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image_format = (image.format or "").upper()
            if image_format in {"JPEG", "PNG"}:
                prepared = image_bytes
                mime_type = "image/jpeg" if image_format == "JPEG" else "image/png"
            else:
                output = io.BytesIO()
                image.convert("RGB").save(output, format="PNG")
                prepared = output.getvalue()
                mime_type = "image/png"
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("invalid_image") from exc
    if len(prepared) > max_bytes:
        raise ValueError("image_too_large")
    return prepared, mime_type


def _iter_layout_items(layout_details: Any):
    if isinstance(layout_details, list):
        for item in layout_details:
            if isinstance(item, list):
                yield from _iter_layout_items(item)
            elif isinstance(item, dict):
                yield item


def _parse_response(payload: dict[str, Any], config: GLMOCRConfig) -> dict[str, Any]:
    texts: list[str] = []
    boxes: list[list[float]] = []
    for item in _iter_layout_items(payload.get("layout_details", [])):
        if item.get("label") == "image":
            continue
        content = str(item.get("content", "") or "").strip()
        if not content:
            continue
        texts.append(content)
        box = item.get("bbox_2d")
        boxes.append(list(box) if isinstance(box, list) and len(box) == 4 else [])

    markdown = str(payload.get("md_results", "") or "").strip()
    markdown = re.sub(r"!\[[^\]]*\]\([^\n]*?\)", "", markdown)
    markdown = re.sub(r"<img\b[^>]*>", "", markdown, flags=re.IGNORECASE).strip()
    if not texts and markdown:
        texts = [markdown]

    return {
        "success": True,
        "available": True,
        "provider": config.provider,
        "model": str(payload.get("model", config.model)),
        "texts": texts,
        "boxes": boxes,
        "box_format": "normalized_xyxy",
        "request_id": str(payload.get("request_id", "") or ""),
        "error": None,
    }


class OCRClient:
    """只支持 GLM-OCR 的同步客户端，返回统一结构供 Agent Tool 使用。"""

    def __init__(self, config: Optional[GLMOCRConfig] = None, **legacy: Any):
        if config is None:
            config = GLMOCRConfig(
                endpoint=legacy.get("endpoint", GLMOCRConfig.endpoint),
                api_key=legacy.get("api_key", ""),
                timeout=int(legacy.get("timeout", GLMOCRConfig.timeout)),
            )
        if config.provider != "glm_ocr" or config.model != "glm-ocr":
            raise ValueError("本项目 OCR Provider 已冻结为 glm_ocr / glm-ocr")
        self.config = config

    @classmethod
    def from_config(cls, path: str) -> "OCRClient":
        return cls(GLMOCRConfig.from_yaml(path))

    def available(self) -> bool:
        return bool(self.config.endpoint and self.config.api_key and self.config.model)

    def recognize(self, image_bytes: bytes) -> dict[str, Any]:
        if not self.available():
            return _failure(self.config, "not_configured", available=False)
        try:
            prepared, mime_type = _prepare_image(
                image_bytes, max_bytes=self.config.max_image_mb * 1024 * 1024
            )
        except ValueError as exc:
            return _failure(self.config, str(exc))

        request_body = {
            "model": self.config.model,
            "file": f"data:{mime_type};base64,{base64.b64encode(prepared).decode('ascii')}",
            "return_crop_images": self.config.return_crop_images,
            "need_layout_visualization": self.config.need_layout_visualization,
        }
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = requests.post(
                self.config.endpoint,
                headers=headers,
                json=request_body,
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                return _failure(self.config, "invalid_response")
            return _parse_response(payload, self.config)
        except requests.Timeout:
            return _failure(self.config, "timeout")
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", "unknown")
            return _failure(self.config, f"http_{status}")
        except (requests.RequestException, ValueError):
            return _failure(self.config, "request_failed")


def ocr(image_bytes: bytes, client: Optional[OCRClient] = None) -> dict[str, Any]:
    return (client or OCRClient()).recognize(image_bytes)
