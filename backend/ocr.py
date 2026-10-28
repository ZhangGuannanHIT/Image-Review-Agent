# -*- coding: utf-8 -*-
"""OCR provider adapters.

Supports Aliyun Marketplace AppCode OCR and the legacy GLM-OCR endpoint.
The provider only extracts characters and layout; it does not classify content.
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


@dataclass(frozen=True)
class AliyunOCRConfig:
    provider: str = "aliyun_market_ocr"
    model: str = "advanced"
    endpoint: str = "https://gjbsb.market.alicloudapi.com/ocrservice/advanced"
    appcode: str = ""
    timeout: int = 60
    max_image_mb: int = 25
    prob: bool = True
    char_info: bool = False
    rotate: bool = True
    table: bool = False
    sort_page: bool = False
    no_stamp: bool = False
    figure: bool = False
    row: bool = True
    paragraph: bool = False
    oricoord: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "AliyunOCRConfig":
        with open(path, "r", encoding="utf-8") as file:
            raw = yaml.safe_load(file) or {}
        section = raw.get("aliyun_market_ocr", raw)
        aliases = {
            "charInfo": "char_info",
            "sortPage": "sort_page",
            "noStamp": "no_stamp",
            "oricoord": "oricoord",
        }
        normalized = {aliases.get(key, key): value for key, value in section.items()}
        configured = cls(**normalized)
        appcode = os.getenv("ALIYUN_OCR_APPCODE") or configured.appcode
        return cls(**{**configured.__dict__, "appcode": appcode})


@dataclass(frozen=True)
class LocalGLMOCRConfig:
    provider: str = "local_glm_ocr"
    model: str = "glm-ocr"
    endpoint: str = "http://127.0.0.1:8080/v1/chat/completions"
    timeout: int = 180
    max_image_mb: int = 15
    max_side: int = 1280
    api_mode: str = "direct_vllm"
    prompt: str = "Text Recognition:"
    max_tokens: int = 2048

    @classmethod
    def from_yaml(cls, path: str) -> "LocalGLMOCRConfig":
        with open(path, "r", encoding="utf-8") as file:
            raw = yaml.safe_load(file) or {}
        section = raw.get("local_glm_ocr", raw)
        return cls(**section)


def _failure(config: Any, error: str, available: bool = True) -> dict[str, Any]:
    return {
        "success": False,
        "available": available,
        "provider": config.provider,
        "model": config.model,
        "texts": [],
        "boxes": [],
        "box_format": "normalized_xyxy",
        "request_id": "",
        "mean_confidence": None,
        "line_confidences": [],
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


def _prepare_local_image(
    image_bytes: bytes,
    *,
    max_bytes: int,
    max_side: int,
) -> tuple[bytes, str]:
    if not image_bytes:
        raise ValueError("empty_image")
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = source.convert("RGB")
            image.thumbnail((max_side, max_side))
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=92)
            prepared = output.getvalue()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("invalid_image") from exc
    if len(prepared) > max_bytes:
        raise ValueError("image_too_large")
    return prepared, "image/jpeg"


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
        "request_id": str(
            payload.get("request_id", "")
            or payload.get("id", "")
            or ""
        ),
        "mean_confidence": None,
        "line_confidences": [],
        "error": None,
    }


def _normalize_polygon(points: Any, width: int, height: int) -> list[float]:
    if not isinstance(points, list) or not points:
        return []
    xs = [float(point.get("x", 0)) for point in points if isinstance(point, dict)]
    ys = [float(point.get("y", 0)) for point in points if isinstance(point, dict)]
    if not xs or not ys or width <= 0 or height <= 0:
        return []
    return [
        max(0.0, min(1.0, min(xs) / width)),
        max(0.0, min(1.0, min(ys) / height)),
        max(0.0, min(1.0, max(xs) / width)),
        max(0.0, min(1.0, max(ys) / height)),
    ]


def _parse_aliyun_response(payload: dict[str, Any], config: AliyunOCRConfig) -> dict[str, Any]:
    width = int(payload.get("width", 0) or 0)
    height = int(payload.get("height", 0) or 0)
    words = payload.get("prism_wordsInfo") or []
    texts = [
        str(row.get("word", "")).strip()
        for row in payload.get("prism_rowsInfo") or []
        if str(row.get("word", "")).strip()
    ]
    if not texts:
        content = str(payload.get("content", "") or "").strip()
        texts = [content] if content else []
    boxes = []
    confidences = []
    for word in words if isinstance(words, list) else []:
        if not isinstance(word, dict):
            continue
        box = _normalize_polygon(word.get("pos"), width, height)
        if box:
            boxes.append(box)
        if word.get("prob") is not None:
            try:
                confidences.append(max(0.0, min(1.0, float(word["prob"]) / 100.0)))
            except (TypeError, ValueError):
                pass
    return {
        "success": True,
        "available": True,
        "provider": config.provider,
        "model": config.model,
        "texts": texts,
        "boxes": boxes,
        "box_format": "normalized_xyxy",
        "request_id": str(payload.get("sid", "") or ""),
        "mean_confidence": (
            round(float(sum(confidences) / len(confidences)), 6)
            if confidences else None
        ),
        "line_confidences": [round(value, 6) for value in confidences],
        "error": None,
    }


class OCRClient:
    """Synchronous OCR client with a provider-neutral response."""

    def __init__(self, config: Optional[GLMOCRConfig] = None, **legacy: Any):
        if config is None:
            config = AliyunOCRConfig(
                endpoint=legacy.get("endpoint", AliyunOCRConfig.endpoint),
                appcode=legacy.get("appcode", ""),
                timeout=int(legacy.get("timeout", AliyunOCRConfig.timeout)),
            )
        if config.provider not in {
            "aliyun_market_ocr",
            "glm_ocr",
            "local_glm_ocr",
        }:
            raise ValueError(f"不支持的 OCR Provider: {config.provider}")
        self.config = config

    @classmethod
    def from_config(cls, path: str) -> "OCRClient":
        with open(path, "r", encoding="utf-8") as file:
            raw = yaml.safe_load(file) or {}
        if "local_glm_ocr" in raw:
            return cls(LocalGLMOCRConfig.from_yaml(path))
        if "aliyun_market_ocr" in raw:
            return cls(AliyunOCRConfig.from_yaml(path))
        return cls(GLMOCRConfig.from_yaml(path))

    def available(self) -> bool:
        if self.config.provider == "local_glm_ocr":
            return bool(self.config.endpoint and self.config.model)
        credential = self.config.appcode if (
            self.config.provider == "aliyun_market_ocr"
        ) else self.config.api_key
        return bool(self.config.endpoint and credential and self.config.model)

    def recognize(self, image_bytes: bytes) -> dict[str, Any]:
        if not self.available():
            return _failure(self.config, "not_configured", available=False)
        try:
            if self.config.provider == "local_glm_ocr":
                prepared, mime_type = _prepare_local_image(
                    image_bytes,
                    max_bytes=self.config.max_image_mb * 1024 * 1024,
                    max_side=self.config.max_side,
                )
            else:
                prepared, mime_type = _prepare_image(
                    image_bytes,
                    max_bytes=self.config.max_image_mb * 1024 * 1024,
                )
        except ValueError as exc:
            return _failure(self.config, str(exc))

        encoded = base64.b64encode(prepared).decode("ascii")
        if self.config.provider == "local_glm_ocr":
            image_data = f"data:{mime_type};base64,{encoded}"
            if self.config.api_mode == "direct_vllm":
                request_body = {
                    "model": self.config.model,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.config.prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_data},
                            },
                        ],
                    }],
                    "temperature": 0.0,
                    "max_tokens": self.config.max_tokens,
                }
            else:
                request_body = {
                    "images": [image_data],
                    "model": self.config.model,
                }
            headers = {"Content-Type": "application/json"}
        elif self.config.provider == "aliyun_market_ocr":
            request_body = {
                "img": encoded,
                "prob": self.config.prob,
                "charInfo": self.config.char_info,
                "rotate": self.config.rotate,
                "table": self.config.table,
                "sortPage": self.config.sort_page,
                "noStamp": self.config.no_stamp,
                "figure": self.config.figure,
                "row": self.config.row,
                "paragraph": self.config.paragraph,
                "oricoord": self.config.oricoord,
            }
            headers = {
                "Authorization": f"APPCODE {self.config.appcode}",
                "Content-Type": "application/json; charset=UTF-8",
            }
        else:
            request_body = {
                "model": self.config.model,
                "file": f"data:{mime_type};base64,{encoded}",
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
            if (
                self.config.provider == "local_glm_ocr"
                and self.config.api_mode == "direct_vllm"
            ):
                choices = payload.get("choices") or []
                content = ""
                if choices and isinstance(choices[0], dict):
                    message = choices[0].get("message") or {}
                    content = str(message.get("content", "") or "").strip()
                if not content:
                    return _failure(self.config, "empty_response")
                return {
                    "success": True,
                    "available": True,
                    "provider": self.config.provider,
                    "model": self.config.model,
                    "texts": [content],
                    "boxes": [],
                    "box_format": "normalized_xyxy",
                    "request_id": str(payload.get("id", "") or ""),
                    "mean_confidence": None,
                    "line_confidences": [],
                    "error": None,
                }
            if self.config.provider == "aliyun_market_ocr":
                return _parse_aliyun_response(payload, self.config)
            return _parse_response(payload, self.config)
        except requests.Timeout:
            return _failure(self.config, "timeout")
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", "unknown")
            return _failure(self.config, f"http_{status}")
        except requests.ConnectionError:
            return _failure(self.config, "service_unavailable")
        except (requests.RequestException, ValueError):
            return _failure(self.config, "request_failed")


def ocr(image_bytes: bytes, client: Optional[OCRClient] = None) -> dict[str, Any]:
    return (client or OCRClient()).recognize(image_bytes)
