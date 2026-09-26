#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MedSAM ViT-B 多层特征编码器（Phase 0 / Task D）。

在 **不修改** ``external/MedSAM`` 官方源码的前提下，包装官方
``ImageEncoderViT``，提取 Transformer 多层特征::

    f3   (block idx 2,  0-based)  -> [B, 768, 64, 64]
    f6   (block idx 5,  0-based)  -> [B, 768, 64, 64]
    f9   (block idx 8,  0-based)  -> [B, 768, 64, 64]
    f12  (block idx 11, 0-based)  -> [B, 768, 64, 64]
    neck                          -> [B, 256, 64, 64]

实现要点：

* 显式顺序执行 ``patch_embed -> pos_embed -> blocks -> neck``，
  不使用全局变量、也不使用 forward hook 暂存 feature。
* Transformer block 内部张量布局为 ``[B, H, W, C]``，保存 feature 时
  统一 ``permute(0, 3, 1, 2).contiguous()`` 转为 ``[B, C, H, W]``。
* ``freeze=True`` 时冻结全部 encoder 参数（``requires_grad=False``）
  且 forward 不建图；``freeze=False`` 时开启梯度，为未来 LoRA /
  partial fine-tuning / full fine-tuning 留出空间。

用法::

    from src.models.medsam_encoder import build_medsam_multilevel_encoder

    encoder = build_medsam_multilevel_encoder(freeze=True, device="cuda")
    feats = encoder(torch.randn(1, 3, 1024, 1024, device="cuda"))
    feats["f3"].shape   # torch.Size([1, 768, 64, 64])
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

LOGGER = logging.getLogger(__name__)

#: 项目根目录 ``MedSAM-PCa/``
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
#: 官方 MedSAM 仓库（只读引用）
DEFAULT_MEDSAM_REPO: Path = PROJECT_ROOT / "external" / "MedSAM"
#: MedSAM ViT-B 预训练权重
DEFAULT_CHECKPOINT: Path = PROJECT_ROOT / "weights" / "medsam_vit_b.pth"

#: ViT-B 使用全局注意力的 block 索引（0-based）
VIT_B_BLOCK_INDICES: Tuple[int, ...] = (2, 5, 8, 11)
VIT_B_EMBED_DIM: int = 768
VIT_B_DEPTH: int = 12
VIT_B_PATCH_SIZE: int = 16
VIT_B_NECK_OUT_CHANNELS: int = 256
VIT_B_IMG_SIZE: int = 1024

__all__ = [
    "MedSAMMultiLevelEncoder",
    "add_medsam_to_sys_path",
    "build_medsam_multilevel_encoder",
    "load_official_sam",
    "VIT_B_BLOCK_INDICES",
    "DEFAULT_MEDSAM_REPO",
    "DEFAULT_CHECKPOINT",
]


# --------------------------------------------------------------------------- #
# 官方代码的安全导入
# --------------------------------------------------------------------------- #
def add_medsam_to_sys_path(
    repo: Path = DEFAULT_MEDSAM_REPO, verify: bool = True
) -> Path:
    """把官方 MedSAM 仓库加入 ``sys.path`` 并校验 ``segment_anything`` 来源。

    本函数只操作 ``sys.path`` / ``sys.modules``，**不会复制或修改**官方源码。

    Args:
        repo: 官方 MedSAM 仓库根目录（其下应有 ``segment_anything/``）。
        verify: 是否校验已导入的 ``segment_anything`` 确实来自 ``repo``。

    Returns:
        解析后的仓库绝对路径。

    Raises:
        FileNotFoundError: 仓库或 ``segment_anything`` 包不存在。
        RuntimeError: 已导入的 ``segment_anything`` 来自其他位置（例如 pip 安装版）。
    """
    repo = Path(repo).resolve()
    pkg_dir = repo / "segment_anything"
    if not (pkg_dir / "__init__.py").is_file():
        raise FileNotFoundError(
            f"在 {repo} 下找不到 segment_anything 包，请确认官方 MedSAM 仓库路径。"
        )

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    # 若此前已从别处导入过 segment_anything，先卸载，避免拿到 pip 版本
    existing = sys.modules.get("segment_anything")
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        same = existing_file is not None and Path(existing_file).resolve().parent == pkg_dir
        if not same:
            for name in [
                n for n in list(sys.modules)
                if n == "segment_anything" or n.startswith("segment_anything.")
            ]:
                del sys.modules[name]
            LOGGER.warning("已卸载来源不明的 segment_anything: %s", existing_file)

    if verify:
        import segment_anything  # noqa: WPS433 - 延迟导入是刻意设计

        origin = Path(segment_anything.__file__).resolve().parent
        if origin != pkg_dir:
            raise RuntimeError(
                f"segment_anything 实际来源为 {origin}，与官方仓库 {pkg_dir} 不一致。"
            )
    return repo


def load_official_sam(
    checkpoint: Path = DEFAULT_CHECKPOINT,
    medsam_repo: Path = DEFAULT_MEDSAM_REPO,
    model_type: str = "vit_b",
) -> nn.Module:
    """通过官方 ``sam_model_registry`` 构建并加载 MedSAM。

    Args:
        checkpoint: ``medsam_vit_b.pth`` 路径。
        medsam_repo: 官方 MedSAM 仓库根目录。
        model_type: 注册表键名，Phase 0 固定使用 ``"vit_b"``。

    Returns:
        官方 ``Sam`` 模块（eval 模式，已载入权重）。

    Raises:
        FileNotFoundError: checkpoint 或仓库不存在。
        KeyError: ``model_type`` 不在官方注册表中。
    """
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint}")

    add_medsam_to_sys_path(medsam_repo)
    from segment_anything import sam_model_registry  # noqa: WPS433

    if model_type not in sam_model_registry:
        raise KeyError(
            f"未知模型类型 {model_type!r}，可选: {sorted(sam_model_registry)}"
        )

    LOGGER.info("通过官方 sam_model_registry[%r] 加载: %s", model_type, checkpoint)
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    LOGGER.info("MedSAM ViT-B successfully loaded. (%s)", checkpoint.name)
    return sam


# --------------------------------------------------------------------------- #
# 多层特征编码器
# --------------------------------------------------------------------------- #
class MedSAMMultiLevelEncoder(nn.Module):
    """从官方 MedSAM ``ImageEncoderViT`` 提取多层特征。

    该模块持有官方 encoder 的**引用**（不复制、不修改其源码），只重写
    forward 顺序，把中间 block 的输出按 ``[B, C, H, W]`` 形式收集起来。

    Args:
        image_encoder: 官方 ``ImageEncoderViT`` 实例。
        out_indices: 需要抽取的 block 索引（0-based）。ViT-B 建议 ``(2, 5, 8, 11)``。
        freeze: 是否冻结全部 encoder 参数。冻结时 forward 在
            ``torch.no_grad()`` 语义下运行（通过 ``torch.set_grad_enabled``），
            但**不是**永久写死，可通过 :meth:`set_freeze` 动态切换。
        expected_input_size: 期望的输入边长，用于 shape 校验；``None`` 表示不校验。

    Attributes:
        feature_keys: 输出字典的键顺序，形如 ``["f3", "f6", "f9", "f12", "neck"]``。
    """

    def __init__(
        self,
        image_encoder: nn.Module,
        out_indices: Sequence[int] = VIT_B_BLOCK_INDICES,
        freeze: bool = True,
        expected_input_size: Optional[int] = VIT_B_IMG_SIZE,
    ) -> None:
        super().__init__()
        self._validate_encoder(image_encoder)

        self.image_encoder = image_encoder
        self.out_indices: Tuple[int, ...] = tuple(int(i) for i in out_indices)
        self.expected_input_size = expected_input_size
        self.freeze = bool(freeze)

        depth = len(self.image_encoder.blocks)
        if len(set(self.out_indices)) != len(self.out_indices):
            raise ValueError(f"out_indices 存在重复: {self.out_indices}")
        for idx in self.out_indices:
            if not 0 <= idx < depth:
                raise ValueError(
                    f"out_indices 越界: {idx}，encoder depth={depth}"
                )

        self.set_freeze(self.freeze)

    # ------------------------------------------------------------------ #
    # 校验与配置
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_encoder(image_encoder: nn.Module) -> None:
        """确认传入对象具备官方 ``ImageEncoderViT`` 的必要子模块。

        Args:
            image_encoder: 待校验模块。

        Raises:
            TypeError: 不是 ``nn.Module``。
            AttributeError: 缺少 ``patch_embed`` / ``blocks`` / ``neck``。
        """
        if not isinstance(image_encoder, nn.Module):
            raise TypeError(
                f"image_encoder 必须是 nn.Module，收到 {type(image_encoder).__name__}"
            )
        for attr in ("patch_embed", "blocks", "neck"):
            if not hasattr(image_encoder, attr):
                raise AttributeError(
                    f"image_encoder 缺少必要子模块 {attr!r}，"
                    "请确认传入的是官方 ImageEncoderViT。"
                )
        if len(image_encoder.blocks) == 0:
            raise ValueError("image_encoder.blocks 为空。")

    @property
    def feature_keys(self) -> List[str]:
        """返回输出特征字典的键（按 block 顺序 + neck）。"""
        return [f"f{i + 1}" for i in self.out_indices] + ["neck"]

    def set_freeze(self, freeze: bool) -> None:
        """设置 encoder 的冻结状态。

        Args:
            freeze: ``True`` 冻结全部 encoder 参数并将其置于 eval 模式；
                ``False`` 解冻全部参数（full fine-tuning 场景）。

        Note:
            未来若引入 LoRA / partial fine-tuning，可先 ``set_freeze(False)``
            再用 :meth:`unfreeze_by_name` 之外的逻辑自行管理 ``requires_grad``；
            若仅需训练 LoRA 适配器，应保持模块属性 ``freeze=False``
            （以便 forward 建图）但把 base 参数置为 ``requires_grad=False``。
        """
        self.freeze = bool(freeze)
        for param in self.image_encoder.parameters():
            param.requires_grad = not self.freeze
        if self.freeze:
            self.image_encoder.eval()
        LOGGER.debug("encoder freeze=%s", self.freeze)

    def unfreeze_by_name(self, pattern: str) -> List[str]:
        """按名称子串解冻参数（为 partial fine-tuning 预留）。

        Args:
            pattern: 参数名中需包含的子串，例如 ``"blocks.11"``。

        Returns:
            实际被解冻的参数名列表。
        """
        matched: List[str] = []
        for name, param in self.image_encoder.named_parameters():
            if pattern in name:
                param.requires_grad = True
                matched.append(name)
        if matched:
            self.freeze = False
        return matched

    def parameter_stats(self) -> Dict[str, int]:
        """统计 encoder 参数量。

        Returns:
            含 ``total`` / ``trainable`` / ``frozen`` 的字典。
        """
        total = sum(p.numel() for p in self.image_encoder.parameters())
        trainable = sum(
            p.numel() for p in self.image_encoder.parameters() if p.requires_grad
        )
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向，返回多层特征。

        Args:
            x: 输入张量 ``[B, 3, S, S]``，``S`` 建议为 1024（patch grid 64x64）。

        Returns:
            字典，含 ``f{i+1}``（每个 ``[B, 768, S/16, S/16]``）与
            ``neck``（``[B, 256, S/16, S/16]``）。

        Raises:
            ValueError: 输入维度或通道数不合法，或与 ``expected_input_size`` 不符。
        """
        if x.dim() != 4:
            raise ValueError(f"输入必须为 4D [B, 3, H, W]，收到 {tuple(x.shape)}")
        if x.shape[1] != 3:
            raise ValueError(
                f"输入通道数必须为 3（MedSAM 约定），收到 {x.shape[1]}"
            )
        if self.expected_input_size is not None:
            h, w = x.shape[2], x.shape[3]
            if h != self.expected_input_size or w != self.expected_input_size:
                raise ValueError(
                    f"输入空间尺寸必须为 {self.expected_input_size}x"
                    f"{self.expected_input_size}，收到 {h}x{w}"
                )

        # freeze=True 时不建图；freeze=False（含未来 LoRA / 微调）时正常建图
        with torch.set_grad_enabled(not self.freeze):
            return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """显式执行 patch_embed -> pos_embed -> blocks -> neck。

        Args:
            x: 输入张量 ``[B, 3, H, W]``。

        Returns:
            多层特征字典。
        """
        encoder = self.image_encoder
        features: Dict[str, torch.Tensor] = {}

        # PatchEmbed 输出布局为 [B, H, W, C]
        x = encoder.patch_embed(x)
        pos_embed = getattr(encoder, "pos_embed", None)
        if pos_embed is not None:
            x = x + pos_embed

        for idx, block in enumerate(encoder.blocks):
            x = block(x)
            if idx in self.out_indices:
                # [B, H, W, C] -> [B, C, H, W]
                features[f"f{idx + 1}"] = x.permute(0, 3, 1, 2).contiguous()

        # 与官方 forward 一致：neck 接收 permute 后的张量
        features["neck"] = encoder.neck(x.permute(0, 3, 1, 2))
        return features

    def describe(self) -> Dict[str, object]:
        """返回模块配置摘要（便于日志与报告）。"""
        stats = self.parameter_stats()
        return {
            "num_blocks": len(self.image_encoder.blocks),
            "out_indices": list(self.out_indices),
            "feature_keys": self.feature_keys,
            "freeze": self.freeze,
            "total_params": stats["total"],
            "trainable_params": stats["trainable"],
        }

    def extra_repr(self) -> str:
        return (
            f"out_indices={self.out_indices}, freeze={self.freeze}, "
            f"expected_input_size={self.expected_input_size}"
        )


# --------------------------------------------------------------------------- #
# 便捷构建器
# --------------------------------------------------------------------------- #
def build_medsam_multilevel_encoder(
    checkpoint: Path = DEFAULT_CHECKPOINT,
    medsam_repo: Path = DEFAULT_MEDSAM_REPO,
    model_type: str = "vit_b",
    out_indices: Sequence[int] = VIT_B_BLOCK_INDICES,
    freeze: bool = True,
    device: Optional[object] = "cuda",
    dtype: torch.dtype = torch.float32,
) -> MedSAMMultiLevelEncoder:
    """一步构建「已加载预训练权重」的 MedSAM 多层特征编码器。

    Args:
        checkpoint: ``medsam_vit_b.pth`` 路径。
        medsam_repo: 官方 MedSAM 仓库根目录。
        model_type: 官方注册表键名。
        out_indices: 抽取的 block 索引（0-based）。
        freeze: 是否冻结 encoder 参数。
        device: 目标设备；``None`` 则保持 CPU。若为 ``"cuda"`` 但不可用，
            会给出警告并回退到 CPU。
        dtype: 参数与计算 dtype。

    Returns:
        加载完毕并已 ``to(device, dtype)`` 的编码器。

    Raises:
        FileNotFoundError: checkpoint 不存在。
        RuntimeError: 官方代码来源校验失败。
    """
    sam = load_official_sam(checkpoint=checkpoint, medsam_repo=medsam_repo,
                            model_type=model_type)

    encoder = MedSAMMultiLevelEncoder(
        image_encoder=sam.image_encoder,
        out_indices=out_indices,
        freeze=freeze,
    )

    if device is not None:
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            LOGGER.warning("CUDA 不可用，回退到 CPU。")
            device = "cpu"
        encoder = encoder.to(device=device, dtype=dtype)

    LOGGER.info("MedSAMMultiLevelEncoder 就绪: %s", encoder.describe())
    return encoder
