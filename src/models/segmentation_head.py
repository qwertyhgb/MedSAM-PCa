#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统一的二值分割输出头（Phase 1）。

所有 baseline（E0–E3）使用同一个输出头，保证公平比较：
``3x3 Conv -> GroupNorm -> GELU -> 1x1 Conv -> 上采样到输入分辨率``。

设计上**不含**任何注意力 / 池化金字塔 / 空洞卷积。
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SegmentationHead", "TwoStageSegHead", "LinearHead"]


class LinearHead(nn.Module):
    """E0: 最简线性输出头。

    只做 ``1x1 Conv(in -> 1)``，然后双线性上采样到 ``out_size``。
    **不包含**任何额外卷积、归一化、激活或金字塔结构。

    Args:
        in_channels: 输入通道数（MedSAM neck = 256）。
        out_size: 输出边长（1024）。
    """

    def __init__(self, in_channels: int = 256, out_size: int = 1024) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_size = int(out_size)
        self.conv = nn.Conv2d(self.in_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向。

        Args:
            x: ``[B, in_channels, H, W]``。

        Returns:
            ``[B, 1, out_size, out_size]`` logits。
        """
        if x.dim() != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"期望 [B, {self.in_channels}, H, W]，收到 {tuple(x.shape)}"
            )
        y = self.conv(x)
        return F.interpolate(y, size=(self.out_size, self.out_size),
                             mode="bilinear", align_corners=False)

    def extra_repr(self) -> str:
        return f"in_channels={self.in_channels}, out_size={self.out_size}"


class TwoStageSegHead(nn.Module):
    """E1 / E2 / E3 **共用**的两级输出头（保证 baseline 间公平比较）。

    结构（与 E1 规格一致）::

        3x3 Conv c->mid -> GN -> GELU
        -> x4 bilinear 上采样
        -> 3x3 Conv mid->fine -> GN -> GELU
        -> 1x1 Conv fine->1

    E3 与 E1 的唯一差别在于**金字塔特征如何构造**，输出头完全相同。

    Args:
        in_channels: 输入通道数（128）。
        mid_channels: 第一级通道（64）。
        fine_channels: 第二级通道（32）。
        out_size: 输出边长（1024）。
        upsample_factor: 中间上采样倍率（4）。
        norm_groups: GroupNorm 分组数。
    """

    def __init__(self, in_channels: int, mid_channels: int = 64,
                 fine_channels: int = 32, out_size: int = 1024,
                 upsample_factor: int = 4, norm_groups: int = 32) -> None:
        super().__init__()
        for name, ch in (("mid_channels", mid_channels), ("fine_channels", fine_channels)):
            if ch % norm_groups != 0:
                raise ValueError(f"{name}={ch} 必须能被 norm_groups={norm_groups} 整除")
        self.in_channels = int(in_channels)
        self.out_size = int(out_size)
        self.upsample_factor = int(upsample_factor)

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, int(mid_channels), kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(norm_groups, int(mid_channels)),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(int(mid_channels), int(fine_channels), kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(norm_groups, int(fine_channels)),
            nn.GELU(),
        )
        self.out_conv = nn.Conv2d(int(fine_channels), 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向。

        Args:
            x: ``[B, in_channels, H, W]`` 融合后的最高分辨率特征。

        Returns:
            ``[B, 1, out_size, out_size]`` logits。
        """
        if x.dim() != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"期望 [B, {self.in_channels}, H, W]，收到 {tuple(x.shape)}"
            )
        y = self.conv1(x)
        y = F.interpolate(y, scale_factor=self.upsample_factor,
                          mode="bilinear", align_corners=False)
        y = self.conv2(y)
        y = self.out_conv(y)
        if y.shape[-2:] != (self.out_size, self.out_size):
            y = F.interpolate(y, size=(self.out_size, self.out_size),
                              mode="bilinear", align_corners=False)
        return y

    def extra_repr(self) -> str:
        return (f"in_channels={self.in_channels}, out_size={self.out_size}, "
                f"upsample_factor={self.upsample_factor}")


class SegmentationHead(nn.Module):
    """多尺度特征 -> ``[B, 1, out_size, out_size]`` logits。

    Args:
        in_channels: 输入特征通道数（由调用方给出）。
        mid_channels: 中间层通道数。
        out_size: 输出空间边长，Phase 1 为 1024。
        num_convs: 中间 3x3 卷积层数（>=1）。
        norm_groups: GroupNorm 分组数。
    """

    def __init__(self, in_channels: int, mid_channels: int = 64,
                 out_size: int = 1024, num_convs: int = 1,
                 norm_groups: int = 32) -> None:
        super().__init__()
        if num_convs < 1:
            raise ValueError("num_convs 必须 >= 1")
        if mid_channels % norm_groups != 0:
            raise ValueError(
                f"mid_channels={mid_channels} 必须能被 norm_groups={norm_groups} 整除"
            )
        self.in_channels = int(in_channels)
        self.mid_channels = int(mid_channels)
        self.out_size = int(out_size)
        self.num_convs = int(num_convs)

        layers: List[nn.Module] = []
        c_in = self.in_channels
        for _ in range(self.num_convs):
            layers += [
                nn.Conv2d(c_in, self.mid_channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(norm_groups, self.mid_channels),
                nn.GELU(),
            ]
            c_in = self.mid_channels
        self.body = nn.Sequential(*layers)
        self.out_conv = nn.Conv2d(self.mid_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向。

        Args:
            x: ``[B, in_channels, H, W]`` 特征图。

        Returns:
            ``[B, 1, out_size, out_size]`` logits。
        """
        if x.dim() != 4:
            raise ValueError(f"期望 4D 输入，收到 {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"通道数不符: 期望 {self.in_channels}, 收到 {x.shape[1]}"
            )
        y = self.body(x)
        y = self.out_conv(y)
        if y.shape[-2:] != (self.out_size, self.out_size):
            y = F.interpolate(y, size=(self.out_size, self.out_size),
                              mode="bilinear", align_corners=False)
        return y

    def extra_repr(self) -> str:
        return (f"in_channels={self.in_channels}, mid_channels={self.mid_channels}, "
                f"out_size={self.out_size}, num_convs={self.num_convs}")
