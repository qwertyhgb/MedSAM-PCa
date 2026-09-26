#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E3：Multi-Level FPN（Phase 1 最重要的 baseline）。

与 E1（只用最终特征）不同，E3 同时利用 **多个 transformer 层** 与 MedSAM
预训练 neck，构造四级金字塔后做标准 FPN top-down 融合。

特征层级（输入 1024x1024，patch 16 → grid 64）：

===========  ==========  ==========  ==============================
层级         空间尺寸     来源         构造
===========  ==========  ==========  ==============================
P2           256x256     f3  (浅)     768->128 投影, 64->128->256
P3           128x128     f6           768->128 投影, 64->128
P4            64x64      f9           768->128 投影, 保持 64
P5            32x32      neck (预训练) 256->128 投影, 64->32
===========  ==========  ==========  ==============================

融合::

    D5 = P5
    D4 = Smooth(P4 + Up(D5))
    D3 = Smooth(P3 + Up(D4))
    D2 = Smooth(P2 + Up(D3))

输出头与 E1 **完全相同**（``TwoStageSegHead``，128->64 -> x4 -> 32 -> 1），
因此 E1 与 E3 的唯一差别在于"金字塔特征如何构造"。

明确不含：任何 lesion-aware gate / attention / PPM / ASPP。
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .segmentation_head import TwoStageSegHead

__all__ = ["MultiLevelFPN"]


class _UpBlock(nn.Module):
    """x2 转置卷积上采样 + GN + GELU。"""

    def __init__(self, channels: int, groups: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2)
        self.norm = nn.GroupNorm(groups, channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.up(x)))


class _SmoothBlock(nn.Module):
    """FPN 的 3x3 Conv + GN + GELU 平滑层。"""

    def __init__(self, channels: int, groups: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.GroupNorm(groups, channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class MultiLevelFPN(nn.Module):
    """E3: 多层级 FPN decoder。

    Args:
        in_channels: transformer 层通道数（768）。
        neck_channels: MedSAM neck 通道数（256）。
        pyramid_channels: 金字塔统一通道（128）。
        out_size: 输出边长（1024）。
        norm_groups: GroupNorm 分组数。
        feature_keys: 三级 transformer 特征键，顺序 ``(浅, 中, 深)``。
        neck_key: neck 特征键。
    """

    def __init__(self, in_channels: int = 768, neck_channels: int = 256,
                 pyramid_channels: int = 128, out_size: int = 1024,
                 norm_groups: int = 32,
                 feature_keys: Sequence[str] = ("f3", "f6", "f9"),
                 neck_key: str = "neck") -> None:
        super().__init__()
        c = int(pyramid_channels)
        g = int(norm_groups)
        if c % g != 0:
            raise ValueError(f"pyramid_channels={c} 必须能被 norm_groups={g} 整除")
        if len(feature_keys) != 3:
            raise ValueError("feature_keys 必须恰好 3 个（浅/中/深 transformer 层）")

        self.in_channels = int(in_channels)
        self.neck_channels = int(neck_channels)
        self.pyramid_channels = c
        self.out_size = int(out_size)
        self.feature_keys = tuple(feature_keys)
        self.neck_key = str(neck_key)

        # 四个分支的 1x1 投影
        self.proj_shallow = nn.Conv2d(self.in_channels, c, kernel_size=1)   # -> P2
        self.proj_mid = nn.Conv2d(self.in_channels, c, kernel_size=1)       # -> P3
        self.proj_deep = nn.Conv2d(self.in_channels, c, kernel_size=1)      # -> P4
        self.proj_neck = nn.Conv2d(self.neck_channels, c, kernel_size=1)    # -> P5

        # P2: 64 -> 128 -> 256
        self.p2_up1 = _UpBlock(c, g)
        self.p2_up2 = _UpBlock(c, g)
        # P3: 64 -> 128
        self.p3_up = _UpBlock(c, g)
        # P5: 64 -> 32
        self.p5_down = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(g, c),
            nn.GELU(),
        )

        # top-down 平滑
        self.smooth4 = _SmoothBlock(c, g)
        self.smooth3 = _SmoothBlock(c, g)
        self.smooth2 = _SmoothBlock(c, g)

        # 与 E1 完全相同的输出头
        self.head = TwoStageSegHead(in_channels=c, mid_channels=64,
                                    fine_channels=32, out_size=self.out_size,
                                    upsample_factor=4, norm_groups=g)

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向。

        Args:
            features: encoder 输出字典，需含 ``f3`` / ``f6`` / ``f9`` / ``neck``。

        Returns:
            ``[B, 1, out_size, out_size]`` logits。
        """
        required = list(self.feature_keys) + [self.neck_key]
        missing = [k for k in required if k not in features]
        if missing:
            raise KeyError(f"features 缺少键: {missing}；实际 keys={sorted(features)}")

        k_shallow, k_mid, k_deep = self.feature_keys
        f_shallow = features[k_shallow]
        f_mid = features[k_mid]
        f_deep = features[k_deep]

        x_shallow = self.proj_shallow(f_shallow)      # [B,128,64,64]
        p2 = self.p2_up2(self.p2_up1(x_shallow))      # [B,128,256,256]
        p3 = self.p3_up(self.proj_mid(f_mid))         # [B,128,128,128]
        p4 = self.proj_deep(f_deep)                   # [B,128,64,64]
        p5 = self.p5_down(self.proj_neck(features[self.neck_key]))  # [B,128,32,32]

        d5 = p5
        d4 = self.smooth4(p4 + F.interpolate(d5, size=p4.shape[-2:], mode="nearest"))
        d3 = self.smooth3(p3 + F.interpolate(d4, size=p3.shape[-2:], mode="nearest"))
        d2 = self.smooth2(p2 + F.interpolate(d3, size=p2.shape[-2:], mode="nearest"))

        return self.head(d2)

    def intermediate_shapes(self, height: int = 64, width: int = 64) -> Dict[str, List[int]]:
        """返回各级中间特征 shape（供测试与文档使用）。

        Args:
            height: 输入 patch grid 高。
            width: 输入 patch grid 宽。

        Returns:
            层级名 -> ``[C, H, W]``。
        """
        c = self.pyramid_channels
        return {
            "P2": [c, height * 4, width * 4],
            "P3": [c, height * 2, width * 2],
            "P4": [c, height, width],
            "P5": [c, height // 2, width // 2],
            "D2": [c, height * 4, width * 4],
            "output": [1, self.out_size, self.out_size],
        }
