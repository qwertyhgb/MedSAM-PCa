#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MedSAM-PCa：冻结 MedSAM encoder + 自定义 decoder 的统一组装（Phase 1）。

统一 model factory：:

    build_model("e3_multilevel_fpn", checkpoint_path="weights/medsam_vit_b.pth")

四个 baseline：

===========  ====================================================
model_name   说明
===========  ====================================================
e0_linear    neck -> 1x1 Conv -> bilinear 上采样（最弱基线）
e1_simple_pyramid  neck -> 单尺度 Simple Feature Pyramid -> FPN
e2_unetr     f3/f6/f9/f12 -> UNETR-style 逐级上采样 + skip
e3_multilevel_fpn  f3/f6/f9/neck -> 多层级金字塔 -> FPN
===========  ====================================================

统一输出 ``{"logits": [B, 1, 1024, 1024]}``，便于将来加 aux logits。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .medsam_encoder import (
    DEFAULT_CHECKPOINT,
    DEFAULT_MEDSAM_REPO,
    VIT_B_BLOCK_INDICES,
    MedSAMMultiLevelEncoder,
    build_medsam_multilevel_encoder,
)
from .multilevel_fpn import MultiLevelFPN
from .segmentation_head import LinearHead
from .simple_pyramid import SimplePyramid
from .unetr_decoder import UNETRStyleDecoder

LOGGER = logging.getLogger(__name__)

__all__ = ["MedSAMPCA", "build_model", "SUPPORTED_MODELS", "MODEL_FEATURE_KEYS"]

SUPPORTED_MODELS: Tuple[str, ...] = (
    "e0_linear",
    "e1_simple_pyramid",
    "e2_unetr",
    "e3_multilevel_fpn",
)

#: 各 decoder 实际使用的 encoder 特征键
MODEL_FEATURE_KEYS: Dict[str, Tuple[str, ...]] = {
    "e0_linear": ("neck",),
    "e1_simple_pyramid": ("neck",),
    "e2_unetr": ("f3", "f6", "f9", "f12"),
    "e3_multilevel_fpn": ("f3", "f6", "f9", "neck"),
}


class SingleFeatureAdapter(nn.Module):
    """把 ``decoder(features: dict)`` 的统一接口适配到只吃单个特征的 decoder。

    E0 / E1 只使用 MedSAM neck，但为了保持所有 decoder 的**调用接口一致**
    （都接收 encoder 特征字典），用本适配器显式取键，避免在
    :class:`MedSAMPCA` 内部做隐式分派。

    Args:
        decoder: 接收单个张量的 decoder（如 ``LinearHead`` / ``SimplePyramid``）。
        key: 从特征字典中取用的键。

    Raises:
        KeyError: 前向时字典缺少 ``key``。
    """

    def __init__(self, decoder: nn.Module, key: str) -> None:
        super().__init__()
        self.decoder = decoder
        self.key = str(key)

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向。

        Args:
            features: encoder 输出的特征字典。

        Returns:
            decoder 输出 logits。
        """
        if self.key not in features:
            raise KeyError(
                f"特征字典缺少 {self.key!r}；实际 keys={sorted(features)}"
            )
        return self.decoder(features[self.key])

    def extra_repr(self) -> str:
        return f"key={self.key}"


class MedSAMPCA(nn.Module):
    """MedSAM encoder 与 decoder 的统一封装。

    Args:
        encoder: 已加载权重的 :class:`MedSAMMultiLevelEncoder`。
        decoder: decoder 模块（接收 encoder 特征字典，返回 logits）。
        model_name: 模型标识，如 ``"e3_multilevel_fpn"``。
        feature_keys: 该 decoder 使用的特征键（文档与校验用途）。
    """

    def __init__(self, encoder: MedSAMMultiLevelEncoder, decoder: nn.Module,
                 model_name: str, feature_keys: Sequence[str]) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.model_name = str(model_name)
        self.feature_keys = tuple(feature_keys)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向。

        Args:
            x: ``[B, 3, 1024, 1024]``（T2W 复制 3 通道）。

        Returns:
            ``{"logits": [B, 1, 1024, 1024]}``。
        """
        features = self.encoder(x)
        logits = self.decoder(features)
        return {"logits": logits}

    # ------------------------------------------------------------------ #
    def parameter_stats(self) -> Dict[str, int]:
        """返回 encoder / decoder 的参数量统计。"""
        enc_total = sum(p.numel() for p in self.encoder.parameters())
        enc_train = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        dec_total = sum(p.numel() for p in self.decoder.parameters())
        dec_train = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        return {
            "encoder_total": enc_total,
            "encoder_trainable": enc_train,
            "decoder_total": dec_total,
            "decoder_trainable": dec_train,
            "total": enc_total + dec_total,
            "trainable": enc_train + dec_train,
        }

    def trainable_parameters(self, named: bool = False
                             ) -> List[torch.nn.Parameter] | List[Tuple[str, torch.nn.Parameter]]:
        """只返回 ``requires_grad=True`` 的参数（**必须**用于 optimizer）。

        Args:
            named: 为 ``True`` 时返回 ``(name, param)`` 列表。

        Returns:
            可训练参数列表。
        """
        pairs = [(n, p) for n, p in self.named_parameters() if p.requires_grad]
        return pairs if named else [p for _, p in pairs]

    def frozen_parameters(self) -> List[Tuple[str, torch.nn.Parameter]]:
        """返回被冻结的参数（用于检查 encoder 未进入 optimizer）。"""
        return [(n, p) for n, p in self.named_parameters() if not p.requires_grad]

    def describe(self) -> Dict[str, Any]:
        """返回模型摘要（供日志/报告使用）。"""
        stats = self.parameter_stats()
        return {
            "model_name": self.model_name,
            "feature_keys": list(self.feature_keys),
            **stats,
        }


def build_model(
    model_name: str,
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT,
    freeze_encoder: bool = True,
    device: Optional[object] = None,
    medsam_repo: Path | str = DEFAULT_MEDSAM_REPO,
    out_size: int = 1024,
) -> MedSAMPCA:
    """统一构造 MedSAM-PCa 模型。

    Args:
        model_name: ``e0_linear`` / ``e1_simple_pyramid`` / ``e2_unetr`` /
            ``e3_multilevel_fpn``。
        checkpoint_path: MedSAM ViT-B 权重路径。
        freeze_encoder: 是否冻结 encoder（Phase 1 固定 ``True``）。
        device: 目标设备；``None`` 表示 CPU。
        medsam_repo: 官方 MedSAM 仓库目录。
        out_size: 输出分辨率。

    Returns:
        组装好的 :class:`MedSAMPCA`。

    Raises:
        ValueError: ``model_name`` 不在支持列表中。
    """
    name = str(model_name).strip().lower()
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"不支持的 model_name={model_name!r}，可选: {list(SUPPORTED_MODELS)}")

    encoder = build_medsam_multilevel_encoder(
        checkpoint=Path(checkpoint_path),
        medsam_repo=Path(medsam_repo),
        out_indices=VIT_B_BLOCK_INDICES,
        freeze=freeze_encoder,
        device=device,
    )

    if name == "e0_linear":
        decoder: nn.Module = SingleFeatureAdapter(
            LinearHead(in_channels=256, out_size=out_size), key="neck")
    elif name == "e1_simple_pyramid":
        decoder = SingleFeatureAdapter(
            SimplePyramid(in_channels=256, out_size=out_size), key="neck")
    elif name == "e2_unetr":
        decoder = UNETRStyleDecoder(in_channels=768, out_size=out_size)
    else:  # e3_multilevel_fpn
        decoder = MultiLevelFPN(in_channels=768, neck_channels=256, out_size=out_size)

    if device is not None:
        decoder = decoder.to(device=device)

    model = MedSAMPCA(encoder=encoder, decoder=decoder, model_name=name,
                      feature_keys=MODEL_FEATURE_KEYS[name])
    LOGGER.info("已构建模型 %s: %s", name, model.describe())
    return model
