# -*- coding: utf-8 -*-
"""损失函数：多任务加权 + 类不平衡处理 + 缺失掩码。

支持：加权 CE（单标签）、加权 BCE（多标签）、Focal Loss（可选）、ignore_index 掩码。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .labels import IGNORE_INDEX


def median_freq_weights(freq_counts: torch.Tensor, max_clip: float = 10.0) -> torch.Tensor:
    """中位数频率平衡权重：w_c = clip(median(freq)/freq_c, max)。用于类不平衡。"""
    median = freq_counts.median().clamp(min=1.0)
    w = median / freq_counts.clamp(min=1.0)
    return w.clamp(max=max_clip)


def inverse_freq_weights(freq_counts: torch.Tensor, max_clip: float = 10.0) -> torch.Tensor:
    """直接反比权重，用于多标签各类的正样本频率平衡。"""
    return (1.0 / freq_counts.clamp(min=1.0)).clamp(max=max_clip)


class WeightedCE(nn.Module):
    """加权交叉熵（单标签），支持标签平滑与 ignore_index 掩码。

    - class_weight: 每类权重，可以是 Tensor 或 None（不加权）。
    - label_smoothing: 0.0 表示关闭。
    - ignore_index: 缺失标签的取值，该位置 loss 置 0。
    """

    def __init__(
        self, class_weight=None, label_smoothing: float = 0.0,
        ignore_index: int = IGNORE_INDEX, focal_gamma: float = 0.0,
    ):
        super().__init__()
        if class_weight is not None:
            self.register_buffer("class_weight", torch.as_tensor(class_weight, dtype=torch.float32))
        else:
            self.class_weight = None
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.focal_gamma = focal_gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # logits: [B, C]; labels: [B] (含 ignore_index)
        ce = F.cross_entropy(
            logits, labels, weight=self.class_weight,
            label_smoothing=self.label_smoothing, ignore_index=self.ignore_index,
            reduction="none",
        )
        if self.focal_gamma > 0:
            probs = F.softmax(logits, dim=-1)
            target = labels.clamp(min=0)
            p_t = probs.gather(1, target.unsqueeze(1)).squeeze(1)
            ce = ((1 - p_t) ** self.focal_gamma) * ce
        valid = labels != self.ignore_index
        return ce[valid].mean() if valid.any() else logits.sum() * 0.0


class WeightedBCE(nn.Module):
    """加权二元交叉熵（多标签），支持每类权重与正样本频率平衡。

    logits: [B, C]（未过 sigmoid）；targets: [B, C] 0/1。
    """

    def __init__(self, pos_weight=None, focal_gamma: float = 0.0):
        super().__init__()
        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.as_tensor(pos_weight, dtype=torch.float32))
        else:
            self.pos_weight = None
        self.focal_gamma = focal_gamma

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        if self.focal_gamma > 0:
            probs = torch.sigmoid(logits)
            p_t = probs * targets + (1 - probs) * (1 - targets)
            loss = ((1 - p_t) ** self.focal_gamma) * loss
        loss = loss.mean(dim=-1)
        if valid_mask is not None:
            valid_mask = valid_mask.to(device=logits.device, dtype=torch.bool)
            return loss[valid_mask].mean() if valid_mask.any() else logits.sum() * 0.0
        return loss.mean()


class MultiTaskLoss(nn.Module):
    """多任务加权组合损失。对应 L = 1.0·risk + 0.5·type + 0.5·injury + 0.5·quality。

    - 各子头使用加权 CE 或 BCE；缺失字段被对应头忽略（若样本曾捕获为缺失）。
    - weights 可由 labels.LOSS_WEIGHTS 传入。
    """

    def __init__(
        self,
        risk_weight=None, type_pos_weight=None,
        injury_weight=None, quality_pos_weight=None,
        label_smoothing: float = 0.05, focal_gamma: float = 0.0,
        task_weights: dict | None = None,
    ):
        super().__init__()
        self.risk_loss = WeightedCE(risk_weight, label_smoothing, focal_gamma=focal_gamma)
        self.type_loss = WeightedBCE(type_pos_weight, focal_gamma=focal_gamma)
        self.injury_loss = WeightedCE(injury_weight, label_smoothing, focal_gamma=focal_gamma)
        self.quality_loss = WeightedBCE(quality_pos_weight, focal_gamma=focal_gamma)
        self.w = task_weights or {"risk": 1.0, "type": 0.5, "injury": 0.5, "quality": 0.5}

    def forward(
        self, logits: dict[str, torch.Tensor], batch: dict[str, torch.Tensor],
        return_parts: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """计算多任务损失；缺失的 type / quality 不会成为伪负标签。"""
        parts = {
            "risk": self.risk_loss(logits["risk"], batch["risk_label"]),
            "type": self.type_loss(logits["type"], batch["type_target"], batch.get("type_valid")),
            "injury": self.injury_loss(logits["injury"], batch["injury_label"]),
            "quality": self.quality_loss(logits["quality"], batch["quality_target"], batch["quality_valid"]),
        }
        parts["total"] = sum(self.w[name] * parts[name] for name in ("risk", "type", "injury", "quality"))
        return parts if return_parts else parts["total"]


if __name__ == "__main__":
    torch.manual_seed(0)
    # 冒烟：验证各 loss 形状与掩码
    risk_w = median_freq_weights(torch.tensor([1093.0, 369, 267, 88, 130]))
    print("risk 权重:", risk_w.tolist())

    ce = WeightedCE(risk_w, label_smoothing=0.05)
    logits = torch.randn(4, 5)
    labels = torch.tensor([0, 1, IGNORE_INDEX, 3])  # 第 3 项缺失
    print("WeightedCE 带掩码:", ce(logits, labels).item())

    bce = WeightedBCE()
    print("WeightedBCE:", bce(torch.randn(4, 4), torch.randint(0, 2, (4, 4)).float()).item())

    mt = MultiTaskLoss(risk_weight=risk_w)
    d = {
        "risk": torch.randn(4, 5), "type": torch.randn(4, 4),
        "injury": torch.randn(4, 4), "quality": torch.randn(4, 5),
    }
    b = {
        "risk_label": torch.tensor([0, 1, 2, 3]),
        "type_target": torch.randint(0, 2, (4, 4)).float(),
        "injury_label": torch.tensor([0, 1, 2, 3]),
        "quality_target": torch.randint(0, 2, (4, 5)).float(),
        "quality_valid": torch.tensor([True, True, False, True]),
    }
    print("MultiTaskLoss:", mt(d, b).item())
    print("losses 冒烟通过")
