#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E2：same-resolution multi-level transformer feature fusion decoder。

**架构命名修正（Phase 2A）**：本 decoder 在历史配置/checkpoint 中名为
``e2_unetr``（为保持 checkpoint 兼容性不改名），但**它不是**严格意义上的
spatial hierarchical UNETR：MedSAM ViT-B 的 ``f3`` / ``f6`` / ``f9`` /
``f12`` 全部位于同一个 patch grid（64x64），因此 ``_up_to(x, ref)`` 的三次
调用都发生在**相同空间尺寸**上——所谓"多级"是 **transformer 深度方向**的
多级，而不是分辨率方向的多级金字塔。

真实前向路径（输入 ``[B, 1, 1024, 1024]``，patch 16 → grid 64）：

===================  ================  ==========================================
阶段                  空间尺寸            操作
===================  ================  ==========================================
proj_deep            64x64             1x1 投影 ``f12``（最深）
fuse_mid2            64x64             concat(prev, proj(f9)) -> Conv3x3 (同分辨率)
fuse_mid1            64x64             concat(prev, proj(f6)) -> Conv3x3 (同分辨率)
fuse_shallow         64x64             concat(prev, proj(f3)) -> Conv3x3 (同分辨率)
refine              128x128           双线性 x2 -> Conv3x3
head                128x128 -> 1024   ``TwoStageSegHead(upsample_factor=1)``
===================  ================  ==========================================

也就是说：**唯一的一次空间上采样发生在 fuse 之后**（64 -> 128），最终由输出头
内部的双线性插值放大到 1024。历史文档曾错误地写成 ``64 -> 128 -> 256 -> 512
-> 1024`` 的逐级上采样，那是与实现不符的描述。

明确不含：attention gate、SE、CBAM、PPM、ASPP、deep supervision。
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .segmentation_head import TwoStageSegHead

__all__ = ["UNETRStyleDecoder"]

#: 融合结束后唯一的空间上采样倍率（64 -> 128），随后由输出头放大到 out_size。
REFINE_UPSCALE = 2


class _FuseBlock(nn.Module):
    """concat 后的 3x3 融合：``Conv3x3(2c -> c) + GN + GELU``。"""

    def __init__(self, in_channels: int, out_channels: int, groups: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class _RefineBlock(nn.Module):
    """``Conv3x3(c -> c) + GN + GELU``。"""

    def __init__(self, channels: int, groups: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.GroupNorm(groups, channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class UNETRStyleDecoder(nn.Module):
    """E2: same-resolution multi-level transformer feature fusion decoder。

    四个 ViT 层（``f3`` / ``f6`` / ``f9`` / ``f12``）在 MedSAM ViT-B 中具有
    **相同的空间分辨率**（64x64），因此本 decoder 的融合全部发生在 64x64，
    只在融合结束后做一次 x2 上采样（64 -> 128），最后交给输出头放大到 1024。
    详见模块 docstring 的架构命名修正说明。

    Args:
        in_channels: 每个 transformer 层的通道数（768）。
        decoder_channels: decoder 统一宽度（128）。
        out_size: 输出边长（1024）。
        norm_groups: GroupNorm 分组数。
        feature_keys: 从 encoder 输出字典中取用的键，顺序为
            ``(最浅, 次浅, 次深, 最深)``。
    """

    def __init__(self, in_channels: int = 768, decoder_channels: int = 128,
                 out_size: int = 1024, norm_groups: int = 32,
                 feature_keys: Sequence[str] = ("f3", "f6", "f9", "f12")) -> None:
        super().__init__()
        c = int(decoder_channels)
        g = int(norm_groups)
        if c % g != 0:
            raise ValueError(f"decoder_channels={c} 必须能被 norm_groups={g} 整除")
        if len(feature_keys) != 4:
            raise ValueError("feature_keys 必须恰好 4 个（f3/f6/f9/f12）")

        self.in_channels = int(in_channels)
        self.decoder_channels = c
        self.out_size = int(out_size)
        self.feature_keys = tuple(feature_keys)

        # 四个 transformer 层的 1x1 投影
        self.proj_shallow = nn.Conv2d(self.in_channels, c, kernel_size=1)   # f3
        self.proj_mid1 = nn.Conv2d(self.in_channels, c, kernel_size=1)      # f6
        self.proj_mid2 = nn.Conv2d(self.in_channels, c, kernel_size=1)      # f9
        self.proj_deep = nn.Conv2d(self.in_channels, c, kernel_size=1)      # f12

        self.fuse_mid2 = _FuseBlock(2 * c, c, g)   # 128x128
        self.fuse_mid1 = _FuseBlock(2 * c, c, g)   # 256x256
        self.fuse_shallow = _FuseBlock(2 * c, c, g)  # 512x512
        self.refine = _RefineBlock(c, g)             # 1024x1024

        # 与 E1/E3 相同结构的输出头（输入已是 1024，故 factor=1）
        self.head = TwoStageSegHead(in_channels=c, mid_channels=64,
                                    fine_channels=32, out_size=self.out_size,
                                    upsample_factor=1, norm_groups=g)

    @staticmethod
    def _up_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """把 ``x`` 双线性上采样到 ``ref`` 的空间尺寸。"""
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向。

        Args:
            features: encoder 输出字典，至少含 ``f3`` / ``f6`` / ``f9`` / ``f12``。

        Returns:
            ``[B, 1, out_size, out_size]`` logits。
        """
        missing = [k for k in self.feature_keys if k not in features]
        if missing:
            raise KeyError(f"features 缺少键: {missing}；实际 keys={sorted(features)}")

        k_shallow, k_mid1, k_mid2, k_deep = self.feature_keys
        p_shallow = self.proj_shallow(features[k_shallow])   # 64x64
        p_mid1 = self.proj_mid1(features[k_mid1])            # 64x64
        p_mid2 = self.proj_mid2(features[k_mid2])            # 64x64
        x = self.proj_deep(features[k_deep])                 # 64x64 (deepest)

        x = self.fuse_mid2(torch.cat([self._up_to(x, p_mid2), p_mid2], dim=1))       # 128
        x = self.fuse_mid1(torch.cat([self._up_to(x, p_mid1), p_mid1], dim=1))       # 256
        x = self.fuse_shallow(torch.cat([self._up_to(x, p_shallow), p_shallow], dim=1))  # 512
        x = self.refine(F.interpolate(x, scale_factor=2, mode="bilinear",
                                      align_corners=False))                          # 1024
        return self.head(x)

    def intermediate_shapes(self, height: int = 64, width: int = 64) -> Dict[str, List[int]]:
        """返回各级中间特征的**真实**空间尺寸（供测试与文档使用）。

        Phase 2A 修正：四次融合全部发生在**相同分辨率**（等于输入 patch
        grid 尺寸，MedSAM ViT-B 下为 64x64），只有 ``refine`` 位于 x2 之后。
        旧版本曾错误地声明为 ``64 -> 128 -> 256 -> 512 -> 1024`` 的逐级上采样。

        Args:
            height: 输入 patch grid 高（ViT-B 为 64）。
            width: 输入 patch grid 宽（ViT-B 为 64）。

        Returns:
            阶段名 -> ``[C, H, W]``；与真实 ``forward`` 的中间张量一致，
            由 ``tests/test_decoder_shapes.py`` 通过 forward hook 校验。
        """
        c = self.decoder_channels
        return {
            "proj_deep": [c, height, width],
            "fuse_mid2": [c, height, width],
            "fuse_mid1": [c, height, width],
            "fuse_shallow": [c, height, width],
            "refine": [c, height * REFINE_UPSCALE, width * REFINE_UPSCALE],
            "output": [1, self.out_size, self.out_size],
        }
