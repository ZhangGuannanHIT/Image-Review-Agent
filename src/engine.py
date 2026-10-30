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
    blood_binary: Optional[dict[str, Any]] = None
    anime_style: Optional[dict[str, Any]] = None

    @property
    def risk_label(self) -> str:
        return self.risk.get("label", "")

    @property
    def type_labels(self) -> list[str]:
        return self.type.get("labels", [])

    @property
    def quality_labels(self) -> list[str]:
        return self.quality.get("labels", [])

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


def _multi_label(
    logits: torch.Tensor, classes: list[str], thresholds: dict[str, float],
) -> dict[str, Any]:
    probs = torch.sigmoid(logits.float())[0].tolist()
    # 在原始概率上判断阈值，避免显示时的四舍五入改变标签。
    return {
        "labels": [label for label, score in zip(classes, probs) if score >= thresholds[label]],
        "scores": {label: round(float(score), 6) for label, score in zip(classes, probs)},
        "thresholds": dict(thresholds),
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
        quality_thresholds: Optional[dict[str, float]] = None,
        binary_injury_dir: str = "",
        quality_fusion_dir: str = "",
        anime_style_router_dir: str = "",
        anime_style_min_score: float = 0.90,
    ):
        self.model_path = str(model_path)
        self.ckpt_dir = str(ckpt_dir or "")
        self.lora = lora
        self.device = self._resolve_device(device)
        self.dtype = dtype or (torch.float32 if self.device == "cpu" else torch.float16)
        self.type_thresholds = {label: float((type_thresholds or {}).get(label, 0.5))
                                for label in labels.TYPE_CLASSES}
        self.quality_thresholds = {label: float((quality_thresholds or {}).get(label, 0.5))
                                   for label in labels.QUALITY_CLASSES}
        if any(not 0 <= value <= 1 for value in (*self.type_thresholds.values(), *self.quality_thresholds.values())):
            raise ValueError("多标签阈值必须位于 [0, 1]")
        self.strict_load = strict_load
        self.binary_injury_dir = str(binary_injury_dir or "")
        self.quality_fusion_dir = str(quality_fusion_dir or "")
        self.anime_style_router_dir = str(anime_style_router_dir or "")
        self.anime_style_min_score = float(anime_style_min_score)
        self.model: Optional[ReviewModel] = None
        self.processor = None
        self.binary_head = None
        self.binary_threshold = 0.5
        self.quality_fusion_head = None
        self.quality_fusion_mean = None
        self.quality_fusion_std = None
        self.anime_style_router = None
        self.anime_style_threshold = 0.5
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
            if self.binary_injury_dir:
                self._load_binary_injury(self.binary_injury_dir)
            if self.quality_fusion_dir:
                self._load_quality_fusion(self.quality_fusion_dir)
            if (
                self.anime_style_router_dir
                and Path(self.anime_style_router_dir).is_dir()
            ):
                self._load_anime_style_router(self.anime_style_router_dir)
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
        head_path = os.path.join(ckpt_dir, "head.pt")
        adapter_path = os.path.join(ckpt_dir, "adapter_model.safetensors")
        adapter_bin = os.path.join(ckpt_dir, "adapter_model.bin")
        has_weights = any(os.path.isfile(path) for path in (head_path, adapter_path, adapter_bin))
        if has_weights and not os.path.isfile(labels_path):
            raise ValueError("checkpoint 缺少 labels.json，无法确认 quality 多标签契约；请重新训练")
        if os.path.isfile(labels_path):
            with open(labels_path, "r", encoding="utf-8") as file:
                saved_schema = json.load(file)
            current_schema = labels.checkpoint_schema()
            if saved_schema != current_schema:
                raise ValueError("checkpoint 标签或任务类型不匹配：quality 已改为多标签，旧单标签权重需重新训练")
        if os.path.isfile(head_path):
            self.model.load_head(head_path)
            self.head_loaded = True

        if os.path.isfile(adapter_path) or os.path.isfile(adapter_bin):
            from peft import load_peft_weights, set_peft_model_state_dict

            adapter_state = load_peft_weights(ckpt_dir, device=self.device)
            set_peft_model_state_dict(self.model.vision, adapter_state)
            self.adapter_loaded = True
        self.trained = self.head_loaded and (not self.lora or self.adapter_loaded)

    def _load_binary_injury(self, head_dir: str) -> None:
        assert self.model is not None
        from .binary_injury import BinaryInjuryHead

        metadata_path = Path(head_dir) / "metadata.json"
        weights_path = Path(head_dir) / "head.pt"
        if not metadata_path.is_file() or not weights_path.is_file():
            raise ValueError("binary injury checkpoint 缺少 metadata.json 或 head.pt")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        embedding_dim = int(metadata.get("embedding_dim", self.model.embed_dim))
        if embedding_dim != self.model.embed_dim:
            raise ValueError(
                f"binary injury embedding 维度不匹配: {embedding_dim} != {self.model.embed_dim}"
            )
        threshold = float(metadata["validation_operating_point"]["threshold"])
        if not 0 <= threshold <= 1:
            raise ValueError("binary injury 阈值必须位于 [0, 1]")
        state = torch.load(weights_path, map_location=self.device, weights_only=True)
        head = BinaryInjuryHead(embedding_dim)
        head.load_state_dict(state, strict=True)
        self.binary_head = head.to(self.device).eval()
        self.binary_threshold = threshold

    def _load_quality_fusion(self, head_dir: str) -> None:
        assert self.model is not None
        from .quality_fusion import QualityFusionHead

        metadata_path = Path(head_dir) / "metadata.json"
        weights_path = Path(head_dir) / "head.pt"
        if not metadata_path.is_file() or not weights_path.is_file():
            raise ValueError("quality fusion checkpoint 缺少 metadata.json 或 head.pt")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        head = QualityFusionHead(
            int(metadata["embedding_dim"]), int(metadata["feature_dim"])
        )
        head.load_state_dict(
            torch.load(weights_path, map_location=self.device, weights_only=True)
        )
        self.quality_fusion_head = head.to(self.device).eval()
        self.quality_fusion_mean = torch.tensor(
            metadata["feature_mean"], dtype=torch.float32
        )
        self.quality_fusion_std = torch.tensor(
            metadata["feature_std"], dtype=torch.float32
        )
        thresholds = metadata["thresholds"]
        if set(thresholds) != set(labels.QUALITY_CLASSES):
            raise ValueError("quality fusion 阈值未覆盖全部 quality 类别")
        self.quality_thresholds = {
            label: float(thresholds[label]) for label in labels.QUALITY_CLASSES
        }

    def _load_anime_style_router(self, head_dir: str) -> None:
        assert self.model is not None
        from .anime_style_router import load_anime_style_router

        self.anime_style_router, metadata = load_anime_style_router(
            head_dir, self.model.embed_dim, self.device
        )
        self.anime_style_threshold = max(
            float(metadata["threshold"]), self.anime_style_min_score
        )

    def save_checkpoint(self, output_dir: str, metrics: Optional[dict] = None) -> None:
        if self.model is None:
            raise RuntimeError("模型未加载，无法保存 checkpoint")
        self.model.save_trainable(output_dir)
        schema = labels.checkpoint_schema()
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

        pil_image = decode_image(image) if getattr(
            self, "quality_fusion_head", None
        ) is not None else None
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
            blood_binary = None
            if getattr(self, "binary_head", None) is not None:
                blood_score = float(
                    torch.sigmoid(self.binary_head(image_embeds)).item()
                )
                blood_binary = {
                    "p_blood": round(blood_score, 6),
                    "threshold": self.binary_threshold,
                    "label": "有血腥" if blood_score >= self.binary_threshold else "无血腥",
                    "trained": True,
                }
            anime_style = None
            if getattr(self, "anime_style_router", None) is not None:
                anime_score = float(
                    torch.sigmoid(
                        self.anime_style_router(image_embeds)
                    ).item()
                )
                anime_style = {
                    "score": round(anime_score, 6),
                    "threshold": self.anime_style_threshold,
                    "is_anime": anime_score >= self.anime_style_threshold,
                    "source": "frozen_clip_embedding_router",
                }

        risk = _single_label(logits["risk"], labels.RISK_CLASSES)
        injury = _single_label(logits["injury"], labels.INJURY_CLASSES)
        quality = _multi_label(logits["quality"], labels.QUALITY_CLASSES, self.quality_thresholds)
        if getattr(self, "quality_fusion_head", None) is not None:
            from .quality_features import extract_quality_features

            assert pil_image is not None
            feature = torch.from_numpy(
                extract_quality_features(pil_image)
            ).unsqueeze(0)
            feature = (
                feature - self.quality_fusion_mean
            ) / self.quality_fusion_std
            with torch.inference_mode():
                quality_scores = torch.sigmoid(
                    self.quality_fusion_head(
                        image_embeds, feature.to(self.device)
                    )
                )[0].cpu().tolist()
            quality = {
                "labels": [
                    label for label, score in zip(labels.QUALITY_CLASSES, quality_scores)
                    if score >= self.quality_thresholds[label]
                ],
                "scores": {
                    label: round(float(score), 6)
                    for label, score in zip(labels.QUALITY_CLASSES, quality_scores)
                },
                "thresholds": dict(self.quality_thresholds),
                "source": "clip_feature_fusion",
            }
        type_result = _multi_label(logits["type"], labels.TYPE_CLASSES, self.type_thresholds)
        # quality 是五个独立的二元判断；取最不确定判断的概率作为保守提示。
        quality_confidence = min(
            score if label in quality["labels"] else 1 - score
            for label, score in quality["scores"].items()
        )
        confidence = min(risk["confidence"], injury["confidence"], quality_confidence)
        return ModelOutput(
            risk=risk, type=type_result, injury=injury, quality=quality,
            conf=round(float(confidence), 6), trained=self.trained,
            blood_binary=blood_binary, anime_style=anime_style,
        )

    def review(self, image: ImageInput, opts: Optional[dict] = None) -> dict[str, Any]:
        """兼容旧 CLI：这里只返回视觉证据，不代替 Agent 编排。"""
        return {"model_output": self.predict(image).to_dict()}
