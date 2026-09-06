# -*- coding: utf-8 -*-
"""模型结构：FG-CLIP2 视觉塔 LoRA + 多任务分类头（方案 A 判别式）。

结构代码不依赖已有权重文件即可导入（权重在运行时由 engine/train 加载）。
"""

from __future__ import annotations

import os
import sys
import json
from typing import Optional

import torch
import torch.nn as nn

# 定位 fg_clip 目录（当前仓库结构：<repo>/fg_clip/）
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "fg_clip")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from .labels import RISK_CLASSES, TYPE_CLASSES, INJURY_CLASSES, QUALITY_CLASSES


def _get_fgclip_model_cls():
    """延迟导入 Fgclip2Model（避免模型代码在无 transformers 时被 import 报错）。"""
    from transformers import AutoModel  # noqa: F401
    from fg_clip.modeling_fgclip2 import Fgclip2Model
    return Fgclip2Model


def load_base_model(model_path: str, dtype=torch.float16, device="cuda") -> nn.Module:
    """加载原始 FG-CLIP2 权重（不动头部）。

    model_path 指向含 config.json 的目录（如 <repo>/fg_clip）。
    用 trust_remote_code 加载本地 modeling_fgclip2.py。
    """
    Fgclip2Model = _get_fgclip_model_cls()
    kwargs = dict(trust_remote_code=True)
    if dtype:
        kwargs["dtype"] = dtype
    model = Fgclip2Model.from_pretrained(model_path, **kwargs)
    if device:
        model = model.to(device)
    model.eval()
    return model


class MultiTaskHead(nn.Module):
    """共享投影 + 4 个任务头。

    从视觉塔 pooler_output(768d) → 共享投影(512d) → 各任务头。
    各头输出 logits；type 为多标签时输出 4 个 sigmoid-logits（BCE 用 logits 即可）。
    """

    def __init__(self, embed_dim: int = 768, hidden: int = 512):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.GELU(), nn.Dropout(0.1)
        )
        self.risk_head = nn.Linear(hidden, len(RISK_CLASSES))
        self.type_head = nn.Linear(hidden, len(TYPE_CLASSES))
        self.injury_head = nn.Linear(hidden, len(INJURY_CLASSES))
        self.quality_head = nn.Linear(hidden, len(QUALITY_CLASSES))

    def forward(self, image_embeds: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.shared(image_embeds)
        return {
            "risk": self.risk_head(h),          # [B, 5]
            "type": self.type_head(h),          # [B, 4]
            "injury": self.injury_head(h),      # [B, 4]
            "quality": self.quality_head(h),    # [B, 5]
        }


class ReviewModel(nn.Module):
    """视觉塔(LoRA 可选) + 多任务头 的组合封装。

    wrap_lora=True：对视觉塔注入 LoRA（q/k/v/out_proj），文本塔冻结。
    冻结策略：文本塔、双塔 embedding、logit_scale 等一律冻结且不参与反向。
    """

    def __init__(
        self,
        model_path: str,
        lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        dtype=torch.float16,
        device="cuda",
    ):
        super().__init__()
        self.base = load_base_model(model_path, dtype=dtype, device=device)
        self.device_name = str(device)
        self.embed_dim = self.base.config.text_config.hidden_size  # 768

        self.vision = self.base.vision_model
        self.freeze_non_target()

        if lora:
            self.apply_lora(r=lora_r, alpha=lora_alpha, dropout=lora_dropout)

        self.head = MultiTaskHead(embed_dim=self.embed_dim)
        vision_param = next(self.vision.parameters())
        self.head.to(device=vision_param.device, dtype=vision_param.dtype)

    @property
    def model_device(self) -> torch.device:
        return next(self.vision.parameters()).device

    @property
    def model_dtype(self) -> torch.dtype:
        return next(self.vision.parameters()).dtype

    def freeze_non_target(self):
        """冻结除视觉塔外的全部参数（文本塔/embedding/logit_scale 等）。"""
        for name, p in self.base.named_parameters():
            # 视觉塔参数保留可训练（LoRA 只作用于其 q/k/v/out_proj）
            if name.startswith("vision_model"):
                continue
            p.requires_grad = False

    def apply_lora(self, r: int, alpha: int, dropout: float):
        """仅对视觉塔注入 LoRA。用 peft 包住 self.base.vision_model。"""
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as e:
            raise ImportError("peft 未安装，LoRA 不可用：pip install peft") from e
        cfg = LoraConfig(
            r=r, lora_alpha=alpha, lora_dropout=dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        self.vision = get_peft_model(self.base.vision_model, cfg)
        # 文本塔保持冻结（peft 默认会给所有匹配模块加 LoRA，这里只包 vision 已规避）

    def forward_vision(self, pixel_values, pixel_attention_mask, spatial_shapes) -> torch.Tensor:
        """视觉塔一次 forward，返回 pooler_output（未归一化）。"""
        out = self.vision(
            pixel_values=pixel_values,
            attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
        )
        return out.pooler_output  # [B, 768]

    def forward(self, image_embeds: torch.Tensor) -> dict[str, torch.Tensor]:
        """由 image_embeds 推断所有任务 logits。"""
        return self.head(image_embeds.to(device=self.model_device, dtype=self.model_dtype))

    def save_trainable(self, output_dir: str) -> None:
        """保存最小推理产物：视觉 LoRA adapter 与四任务分类头。"""
        os.makedirs(output_dir, exist_ok=True)
        if hasattr(self.vision, "save_pretrained"):
            self.vision.save_pretrained(output_dir, safe_serialization=True)
        torch.save(self.head.state_dict(), os.path.join(output_dir, "head.pt"))
        schema = {
            "risk": RISK_CLASSES,
            "type": TYPE_CLASSES,
            "injury": INJURY_CLASSES,
            "quality": QUALITY_CLASSES,
        }
        with open(os.path.join(output_dir, "labels.json"), "w", encoding="utf-8") as file:
            json.dump(schema, file, ensure_ascii=False, indent=2)

    def load_head(self, path: str) -> None:
        state = torch.load(path, map_location=self.model_device, weights_only=True)
        self.head.load_state_dict(state, strict=True)
        self.head.to(device=self.model_device, dtype=self.model_dtype)


if __name__ == "__main__":
    # 冒烟：仅验证结构能实例化（不加载真实大权重，用随机 Mock 验证 head seq）
    import torch
    head = MultiTaskHead()
    x = torch.randn(2, 768)
    out = head(x)
    for k, v in out.items():
        print(k, tuple(v.shape))
    # 验证 labels 数量与头输出对齐
    assert out["risk"].shape[1] == len(RISK_CLASSES)
    assert out["type"].shape[1] == len(TYPE_CLASSES)
    assert out["injury"].shape[1] == len(INJURY_CLASSES)
    assert out["quality"].shape[1] == len(QUALITY_CLASSES)
    print("MultiTaskHead 结构自检通过")
