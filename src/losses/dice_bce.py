#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统一损失：Dice + BCEWithLogitsLoss（Phase 1）。

四个 baseline 使用**完全相同**的损失函数：

.. math::

    L = L_{dice} + L_{bce}

其中 Dice 使用 soft（可微）形式并带 epsilon 防止除零。

明确不包含：Focal / Tversky / Boundary / Hausdorff / 辅助损失。
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DiceLoss", "DiceBCELoss", "dice_coefficient_from_logits"]


class DiceLoss(nn.Module):
    """Soft Dice 损失（对 logits 先做 sigmoid）。

    Args:
        eps: 分子分母的平滑项，防止除零。
        reduction: ``"mean"`` / ``"sum"`` / ``"none"``。
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"未知 reduction: {reduction}")
        self.eps = float(eps)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """计算 Dice 损失。

        Args:
            logits: ``[B, 1, H, W]`` 未激活 logits。
            targets: ``[B, 1, H, W]``，取值 {0,1}。

        Returns:
        标量损失（``reduction="none"`` 时为 ``[B]``）。
        """
        if logits.shape != targets.shape:
            raise ValueError(f"shape 不匹配: logits {tuple(logits.shape)} vs "
                             f"targets {tuple(targets.shape)}")
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        intersection = (probs * targets).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        loss = 1.0 - dice

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class DiceBCELoss(nn.Module):
    """``Dice + BCEWithLogitsLoss`` 组合损失。

    Args:
        dice_weight: Dice 项权重。
        bce_weight: BCE 项权重。
        eps: Dice 平滑项。
        bce_pos_weight: 传给 ``BCEWithLogitsLoss`` 的 ``pos_weight``；``None`` 为不加权。

    Note:
        四个 baseline 必须使用同一套超参（configs 中保持一致）。
    """

    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0,
                 eps: float = 1e-6,
                 bce_pos_weight: Optional[float] = None) -> None:
        super().__init__()
        self.dice = DiceLoss(eps=eps)
        pos_weight = (torch.tensor([bce_pos_weight])
                      if bce_pos_weight is not None else None)
        # pos_weight 由 forward 时搬到对应 device（若提供）
        self.register_buffer("_pos_weight", pos_weight if pos_weight is not None else torch.tensor([]))
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=self._pos_weight if self._pos_weight.numel() > 0 else None
        )
        self.dice_weight = float(dice_weight)
        self.bce_weight = float(bce_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """计算组合损失。

        Args:
            logits: ``[B, 1, H, W]`` 未激活 logits。
            targets: ``[B, 1, H, W]``，取值 {0,1}。

        Returns:
        标量损失。
        """
        if targets.dtype != logits.dtype:
            targets = targets.to(logits.dtype)
        dice_loss = self.dice(logits, targets)
        bce_loss = self.bce(logits, targets)
        total = self.dice_weight * dice_loss + self.bce_weight * bce_loss
        if not torch.isfinite(total):
            raise FloatingPointError(
                f"损失出现非有限值: total={total.item()}, "
                f"dice={dice_loss.item()}, bce={bce_loss.item()}"
            )
        return total

    def components(self, logits: torch.Tensor,
                   targets: torch.Tensor) -> Dict[str, float]:
        """返回分项损失的 Python 数值（供日志使用）。"""
        with torch.no_grad():
            d = float(self.dice(logits, targets).item())
            b = float(self.bce(logits, targets).item())
        return {"dice_loss": d, "bce_loss": b, "total": self.dice_weight * d + self.bce_weight * b}


def dice_coefficient_from_logits(logits: torch.Tensor, targets: torch.Tensor,
                                 threshold: float = 0.5,
                                 eps: float = 1e-6) -> torch.Tensor:
    """按 batch 维度返回 Dice 系数（逐样本）。

    Args:
        logits: ``[B, 1, H, W]``。
        targets: ``[B, 1, H, W]``，取值 {0,1}。
        threshold: 二值化阈值。
        eps: 平滑项。

    Returns:
        ``[B]`` 的 Dice 系数张量。

    Note:
        当 GT 与预测同时为空时，本函数返回 1.0。是否计入平均值由调用方
        （见 :mod:`src.metrics.segmentation`）决定，不要直接混入 lesion Dice。
    """
    probs = torch.sigmoid(logits)
    pred = (probs > threshold).float()
    dims = (1, 2, 3)
    inter = (pred * targets).sum(dim=dims)
    card = pred.sum(dim=dims) + targets.sum(dim=dims)
    return (2.0 * inter + eps) / (card + eps)
