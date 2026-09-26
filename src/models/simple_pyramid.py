#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E1：Final Feature Simple Pyramid（Phase 1 baseline）。

参考 ViTDet 的 Simple Feature Pyramid 思路：**只使用 MedSAM 的最终特征**
（``neck``，``[B, 256, 64, 64]``），从同一张特征图上通过纯卷积/转置卷积
构造 4 级金字塔，再做标准 FPN top-down 融合。

特征层级（输入 1024x1024，patch 16 → grid 64）：

===========  =========  ==========  ================
层级          空间尺寸    相对 P4     构造方式
===========  =========  ==========  ================
P2           256x256    x4          ConvTranspose x2 两次
P3           128x128    x2          ConvTranspose x2 一次
P4            64x64     x1          identity + 3x3 conv
P5            32x32     x0.5        3x3 stride=2
===========  =========  ==========  ================

融合：``D5=P5``; ``D4=Smooth(P4+Up(D5))``; ``D3=Smooth(P3+Up(D4))``;
``D2=Smooth(P2+Up(D3))``。

明确不含：注意力、PPM、ASPP、SE、CBAM。
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .segmentation_head import TwoStageSegHead

__all__ = ["SimplePyramid"]


def _gn_gelu(channels: int, groups: int) -> nn.Sequential:
    """GroupNorm + GELU 组合。"""
    return nn.Sequential(nn.GroupNorm(groups, channels), nn.GELU())


class _SmoothBlock(nn.Module):
    """FPN 的 3x3 Conv + GN + GELU 平滑层。"""

    def __init__(self, channels: int, groups: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.GroupNorm(groups, channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class _UpBlock(nn.Module):
    """x2 转置卷积上采样 + GN + GELU。"""

    def __init__(self, channels: int, groups: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2)
        self.norm = nn.GroupNorm(groups, channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.up(x)))


class SimplePyramid(nn.Module):
    """E1: 单尺度 final feature 构造简单金字塔 + FPN 融合。

    Args:
        in_channels: 输入特征通道数（MedSAM neck 输出 = 256）。
        pyramid_channels: 金字塔统一通道数（128）。
        mid_channels: 输出头中间通道数（64）。
        fine_channels: 输出头第二级通道数（32）。
        out_size: 输出分辨率（1024）。
        norm_groups: GroupNorm 分组数。
    """

    def __init__(self, in_channels: int = 256, pyramid_channels: int = 128,
                 mid_channels: int = 64, fine_channels: int = 32,
                 out_size: int = 1024, norm_groups: int = 32) -> None:
        super().__init__()
        c = int(pyramid_channels)
        g = int(norm_groups)
        for name, ch in (("pyramid_channels", c), ("mid_channels", mid_channels),
                         ("fine_channels", fine_channels)):
            if ch % g != 0:
                raise ValueError(f"{name}={ch} 必须能被 norm_groups={g} 整除")

        self.in_channels = int(in_channels)
        self.pyramid_channels = c
        self.out_size = int(out_size)

        # 统一投影：256 -> 128
        self.stem = nn.Conv2d(self.in_channels, c, kernel_size=1)

        # P2: 64 -> 128 -> 256
        self.p2_up1 = _UpBlock(c, g)
        self.p2_up2 = _UpBlock(c, g)
        # P3: 64 -> 128
        self.p3_up = _UpBlock(c, g)
        # P4: 3x3 conv
        self.p4_conv = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            _gn_gelu(c, g),
        )
        # P5: 3x3 stride=2
        self.p5_down = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, stride=2, padding=1, bias=False),
            _gn_gelu(c, g),
        )

        # top-down 平滑
        self.smooth4 = _SmoothBlock(c, g)
        self.smooth3 = _SmoothBlock(c, g)
        self.smooth2 = _SmoothBlock(c, g)

        # 输出头：与 E2/E3 **完全相同**（TwoStageSegHead，128 -> 64 -> x4 -> 32 -> 1）
        self.head = TwoStageSegHead(
            in_channels=c, mid_channels=int(mid_channels),
            fine_channels=int(fine_channels), out_size=self.out_size,
            upsample_factor=4, norm_groups=g,
        )

    def forward(self, neck: torch.Tensor
                ) -> torch.Tensor:
        """前向。

        Args:
            neck: MedSAM neck 特征 ``[B, 256, 64, 64]``。

        Returns:
            ``[B, 1, out_size, out_size]`` logits。
        """
        if neck.dim() != 4 or neck.shape[1] != self.in_channels:
            raise ValueError(
                f"期望 [B, {self.in_channels}, H, W]，收到 {tuple(neck.shape)}"
            )

        x = self.stem(neck)                    # [B,128,64,64]
        p4 = self.p4_conv(x)                   # [B,128,64,64]
        p5 = self.p5_down(x)                   # [B,128,32,32]
        p3 = self.p3_up(x)                     # [B,128,128,128]
        p2 = self.p2_up2(self.p2_up1(x))       # [B,128,256,256]

        d5 = p5
        d4 = self.smooth4(p4 + F.interpolate(d5, size=p4.shape[-2:], mode="nearest"))
        d3 = self.smooth3(p3 + F.interpolate(d4, size=p3.shape[-2:], mode="nearest"))
        d2 = self.smooth2(p2 + F.interpolate(d3, size=p2.shape[-2:], mode="nearest"))

        return self.head(d2)                   # [B,1,1024,1024]

    def intermediate_shapes(self, height: int = 64, width: int = 64) -> Dict[str, List[int]]:
        """返回各级中间特征 shape（供测试与文档使用）。

        Args:
            height: 输入特征高（patch grid）。
            width: 输入特征宽。

        Returns:
            层级名 -> ``[C, H, W]``。
        """
        return {
            "stem": [self.pyramid_channels, height, width],
            "P5": [self.pyramid_channels, height // 2, width // 2],
            "P4": [self.pyramid_channels, height, width],
            "P3": [self.pyramid_channels, height * 2, width * 2],
            "P2": [self.pyramid_channels, height * 4, width * 4],
        }
