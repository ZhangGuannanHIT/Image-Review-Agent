# -*- coding: utf-8 -*-
"""FG-CLIP2 统一推理引擎。

图片经本地 processor 和视觉塔得到表示，再由四任务头产生与当前 Label
Studio 标注契约一致的结构化证据。证据只在后端内部使用。
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Union

import torch
from PIL import Image

from . import labels
from .model import ReviewModel

ImageInput = Union[bytes, bytearray, str, os.PathLike, Image.Image]


@dataclass
class ModelOutput:
    risk: dict[str, Any]
    type: dict[str, Any]
    injury: dict[str, Any]
    quality: dict[str, Any]
    conf: float
    trained: bool = False

    @property
    def risk_label(self) -> str:
        return self.risk.get("label", "")

    @property
    def type_labels(self) -> list[str]:
        return self.type.get("labels", [])

    @property
    def quality_label(self) -> str:
        return self.quality.get("label", "")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def determine_max_patches(image: Image.Image) -> int:
    """沿用 FG-CLIP2 官方示例的动态 patch 档位。"""
    width, height = image.size
    patch_count = (width // 16) * (height // 16)
    if patch_count > 784:
        return 1024
    if patch_count > 576:
        return 784
    if patch_count > 256:
        return 576
    if patch_count > 128:
        return 256
    return 128


def decode_image(image: ImageInput) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (bytes, bytearray)):
        with Image.open(io.BytesIO(bytes(image))) as opened:
            return opened.convert("RGB")
    if isinstance(image, (str, os.PathLike)):
        with Image.open(image) as opened:
            return opened.convert("RGB")
    raise TypeError(f"不支持的图片输入类型: {type(image)!r}")


def _single_label(logits: torch.Tensor, classes: list[str]) -> dict[str, Any]:
    probs = torch.softmax(logits.float(), dim=-1)[0]
    index = int(probs.argmax().item())
    return {
        "label": classes[index],
        "index": index,
        "confidence": round(float(probs[index].item()), 6),
        "scores": {label: round(float(score), 6) for label, score in zip(classes, probs.tolist())},
    }


class ReviewEngine:
    """加载 FG-CLIP2、LoRA/Head，并提供单图结构化推理。"""

    def __init__(
        self,
        model_path: str,
        ckpt_dir: str = "",
        lora: bool = True,
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
        type_thresholds: Optional[dict[str, float]] = None,
        strict_load: bool = False,
    ):
        self.model_path = str(model_path)
        self.ckpt_dir = str(ckpt_dir or "")
        self.lora = lora
        self.device = self._resolve_device(device)
        self.dtype = dtype or (torch.float32 if self.device == "cpu" else torch.float16)
        self.type_thresholds = type_thresholds or {label: 0.5 for label in labels.TYPE_CLASSES}
        self.strict_load = strict_load
        self.model: Optional[ReviewModel] = None
        self.processor = None
        self.trained = False
        self.head_loaded = False
        self.adapter_loaded = False
        self._load_error = ""
        self._load()

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return device

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    def _load(self) -> None:
        try:
            from transformers import AutoImageProcessor

            self.processor = AutoImageProcessor.from_pretrained(
                self.model_path, trust_remote_code=True, local_files_only=True
            )
            self.model = ReviewModel(
                self.model_path, lora=self.lora, dtype=self.dtype, device=self.device
            )
            if self.ckpt_dir:
                self._load_checkpoint(self.ckpt_dir)
            self.model.eval()
        except Exception as exc:  # noqa: BLE001
            self.model = None
            self.processor = None
            self._load_error = f"{type(exc).__name__}: {exc}"
            if self.strict_load:
                raise RuntimeError(f"FG-CLIP2 加载失败: {self._load_error}") from exc

    def _load_checkpoint(self, ckpt_dir: str) -> None:
        assert self.model is not None
        labels_path = os.path.join(ckpt_dir, "labels.json")
        if os.path.isfile(labels_path):
            with open(labels_path, "r", encoding="utf-8") as file:
                saved_schema = json.load(file)
            current_schema = {
                "risk": labels.RISK_CLASSES,
                "type": labels.TYPE_CLASSES,
                "injury": labels.INJURY_CLASSES,
                "quality": labels.QUALITY_CLASSES,
            }
            if saved_schema != current_schema:
                raise ValueError("checkpoint 标签顺序与当前 Label Studio 标签契约不一致")
        head_path = os.path.join(ckpt_dir, "head.pt")
        if os.path.isfile(head_path):
            self.model.load_head(head_path)
            self.head_loaded = True

        adapter_path = os.path.join(ckpt_dir, "adapter_model.safetensors")
        adapter_bin = os.path.join(ckpt_dir, "adapter_model.bin")
        if os.path.isfile(adapter_path) or os.path.isfile(adapter_bin):
            from peft import load_peft_weights, set_peft_model_state_dict

            adapter_state = load_peft_weights(ckpt_dir, device=self.device)
            set_peft_model_state_dict(self.model.vision, adapter_state)
            self.adapter_loaded = True
        self.trained = self.head_loaded and (not self.lora or self.adapter_loaded)

    def save_checkpoint(self, output_dir: str, metrics: Optional[dict] = None) -> None:
        if self.model is None:
            raise RuntimeError("模型未加载，无法保存 checkpoint")
        self.model.save_trainable(output_dir)
        schema = {
            "risk": labels.RISK_CLASSES,
            "type": labels.TYPE_CLASSES,
            "injury": labels.INJURY_CLASSES,
            "quality": labels.QUALITY_CLASSES,
        }
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(output_dir, "labels.json"), "w", encoding="utf-8") as file:
            json.dump(schema, file, ensure_ascii=False, indent=2)
        if metrics is not None:
            with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as file:
                json.dump(metrics, file, ensure_ascii=False, indent=2)

    def prepare_image(self, image: ImageInput) -> dict[str, torch.Tensor]:
        if self.processor is None:
            raise RuntimeError(f"图片处理器未加载: {self._load_error}")
        pil_image = decode_image(image)
        inputs = self.processor(
            images=pil_image,
            max_num_patches=determine_max_patches(pil_image),
            return_tensors="pt",
        )
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }

    def predict(self, image: ImageInput) -> ModelOutput:
        if not self.loaded or self.model is None:
            raise RuntimeError(f"模型未加载成功，无法推理。请检查权重与环境。{self._load_error}")

        inputs = self.prepare_image(image)
        required = ("pixel_values", "pixel_attention_mask", "spatial_shapes")
        missing = [name for name in required if name not in inputs]
        if missing:
            raise RuntimeError(f"FG-CLIP2 processor 缺少输出字段: {missing}")

        self.model.eval()
        with torch.inference_mode():
            image_embeds = self.model.forward_vision(
                pixel_values=inputs["pixel_values"],
                pixel_attention_mask=inputs["pixel_attention_mask"],
                spatial_shapes=inputs["spatial_shapes"],
            )
            logits = self.model(image_embeds)

        risk = _single_label(logits["risk"], labels.RISK_CLASSES)
        injury = _single_label(logits["injury"], labels.INJURY_CLASSES)
        quality = _single_label(logits["quality"], labels.QUALITY_CLASSES)
        type_probs = torch.sigmoid(logits["type"].float())[0]
        type_scores = {
            label: round(float(score), 6)
            for label, score in zip(labels.TYPE_CLASSES, type_probs.tolist())
        }
        selected = [
            label for label, score in type_scores.items()
            if score >= float(self.type_thresholds.get(label, 0.5))
        ]
        type_result = {"labels": selected, "scores": type_scores, "thresholds": self.type_thresholds}
        confidence = min(risk["confidence"], injury["confidence"], quality["confidence"])
        return ModelOutput(
            risk=risk, type=type_result, injury=injury, quality=quality,
            conf=round(float(confidence), 6), trained=self.trained,
        )

    def review(self, image: ImageInput, opts: Optional[dict] = None) -> dict[str, Any]:
        """兼容旧 CLI：这里只返回视觉证据，不代替 Agent 编排。"""
        return {"model_output": self.predict(image).to_dict()}
