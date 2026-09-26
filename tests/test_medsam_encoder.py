#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MedSAM 多层特征编码器 —— 单元测试 / Smoke Test（Phase 0 / Task E、F、G、H）。

覆盖内容：

1. checkpoint 加载
2. 随机输入 forward 与 shape 校验
3. 冻结参数验证（``freeze=True`` 时 trainable == 0）
4. GPU 显存峰值测量
5. 真实 PI-CAI T2W 单 slice smoke test（含 NaN/Inf 检查）

运行方式::

    # 直接运行（推荐，会输出完整摘要）
    /root/anaconda3/envs/lm/bin/python tests/test_medsam_encoder.py

    # 或 pytest
    /root/anaconda3/envs/lm/bin/python -m pytest tests/test_medsam_encoder.py -v -s

注意：
    本文件中的 MRI 预处理（percentile clip / normalize / resize）仅为
    smoke test 用途，**不是**最终训练用的预处理流程。

    测试函数会 ``return`` 一个 dict（供直接运行时汇总打印），因此 pytest
    会给出 ``PytestReturnNotNoneWarning``。这是刻意设计，断言本身在函数内，
    警告无害。
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.medsam_encoder import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_MEDSAM_REPO,
    VIT_B_BLOCK_INDICES,
    MedSAMMultiLevelEncoder,
    build_medsam_multilevel_encoder,
)

LOGGER = logging.getLogger("test_medsam_encoder")

#: 固定随机种子（约束要求）
SEED: int = 20260924
#: 输入边长
INPUT_SIZE: int = 1024
#: 期望输出 shape（ViT-B, patch 16 -> grid 64）
EXPECTED_SHAPES: Dict[str, Tuple[int, int, int, int]] = {
    "f3": (1, 768, 64, 64),
    "f6": (1, 768, 64, 64),
    "f9": (1, 768, 64, 64),
    "f12": (1, 768, 64, 64),
    "neck": (1, 256, 64, 64),
}

DEFAULT_DATA_ROOT = os.environ.get(
    "PICAI_DATA_ROOT", "/opt/data/private/lm/data/Prostate/PI-CAI"
)

#: 测试目标设备
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

_ENCODER_CACHE: Dict[str, MedSAMMultiLevelEncoder] = {}


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #
def get_encoder(freeze: bool = True, device: Optional[str] = None) -> MedSAMMultiLevelEncoder:
    """构建（并缓存）编码器，避免在每个测试中重复加载 358MB 权重。

    Args:
        freeze: 是否冻结 encoder。
        device: 目标设备，``None`` 表示使用全局 ``DEVICE``。

    Returns:
        已加载权重的 :class:`MedSAMMultiLevelEncoder`。
    """
    dev = device or DEVICE
    key = f"{dev}|freeze={freeze}"
    if key not in _ENCODER_CACHE:
        _ENCODER_CACHE[key] = build_medsam_multilevel_encoder(
            checkpoint=DEFAULT_CHECKPOINT,
            medsam_repo=DEFAULT_MEDSAM_REPO,
            out_indices=VIT_B_BLOCK_INDICES,
            freeze=freeze,
            device=dev,
        )
    return _ENCODER_CACHE[key]


def bytes_to_mib(n: int) -> float:
    """字节转 MiB。"""
    return n / (1024 ** 2)


# --------------------------------------------------------------------------- #
# Test 1 —— checkpoint 加载
# --------------------------------------------------------------------------- #
def test_checkpoint_load() -> Dict[str, Any]:
    """确认 MedSAM ViT-B checkpoint 可以被官方代码成功加载。"""
    assert DEFAULT_CHECKPOINT.is_file(), f"checkpoint 不存在: {DEFAULT_CHECKPOINT}"

    encoder = get_encoder(freeze=True)
    assert isinstance(encoder, nn.Module), "encoder 必须是 nn.Module"

    stats = encoder.parameter_stats()
    assert stats["total"] > 0, "encoder 参数量为 0，权重未正确加载"

    info = {
        "checkpoint": str(DEFAULT_CHECKPOINT),
        "size_mib": round(DEFAULT_CHECKPOINT.stat().st_size / (1024 ** 2), 2),
        "num_blocks": len(encoder.image_encoder.blocks),
        "out_indices": list(encoder.out_indices),
        "feature_keys": encoder.feature_keys,
        "total_params": stats["total"],
        "message": "MedSAM ViT-B checkpoint loads successfully",
    }
    LOGGER.info("[Test 1] checkpoint 加载成功: %s", info)
    return info


# --------------------------------------------------------------------------- #
# Test 2 —— 随机输入 forward
# --------------------------------------------------------------------------- #
def test_random_forward() -> Dict[str, Any]:
    """随机张量 forward，打印并校验真实输出 shape。"""
    torch.manual_seed(SEED)
    encoder = get_encoder(freeze=True)

    x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device=DEVICE)
    LOGGER.info("[Test 2] 输入: %s (%s)", tuple(x.shape), DEVICE)

    features = encoder(x)

    actual = {k: tuple(v.shape) for k, v in features.items()}
    LOGGER.info("[Test 2] 真实输出 shape:")
    for k, shape in actual.items():
        LOGGER.info("          %-5s %s", k, shape)

    # 逐个对比，不一致时打印真实 shape 并解释可能原因（不强行改输出）
    for key, expected in EXPECTED_SHAPES.items():
        assert key in actual, f"缺少输出 {key}，实际 keys={sorted(actual)}"
        if actual[key] != expected:
            raise AssertionError(
                f"{key} shape 不符: 期望 {expected}, 实际 {actual[key]}。\n"
                f"  可能原因: (a) checkpoint 不是 ViT-B; (b) patch_size 不为 16; "
                f"(c) 输入边长不是 1024; (d) out_indices 与预期块号不同。\n"
                f"  encoder 实际配置: blocks={len(encoder.image_encoder.blocks)}, "
                f"out_indices={list(encoder.out_indices)}, "
                f"img_size={getattr(encoder.image_encoder, 'img_size', None)}"
            )

    assert set(actual) == set(EXPECTED_SHAPES), (
        f"输出键不一致: 实际 {sorted(actual)} vs 期望 {sorted(EXPECTED_SHAPES)}"
    )

    finite = {k: bool(torch.isfinite(v).all()) for k, v in features.items()}
    assert all(finite.values()), f"随机 forward 出现非有限值: {finite}"

    return {"input_shape": list(x.shape), "output_shapes": {k: list(v) for k, v in actual.items()},
            "all_finite": finite}


# --------------------------------------------------------------------------- #
# Test 3 —— 冻结验证（Task F）
# --------------------------------------------------------------------------- #
def test_freeze() -> Dict[str, Any]:
    """验证 ``freeze=True`` 时 trainable encoder 参数为 0。"""
    frozen = get_encoder(freeze=True)
    total = sum(p.numel() for p in frozen.parameters())
    trainable = sum(p.numel() for p in frozen.parameters() if p.requires_grad)

    LOGGER.info("[Test 3] freeze=True : total=%d trainable=%d", total, trainable)
    assert trainable == 0, f"freeze=True 时仍有 {trainable} 个可训练参数"

    # 解冻路径同样验证，确认未来 fine-tuning 有出口（不改变缓存实例的常态）
    unfrozen = get_encoder(freeze=False)
    total_u = sum(p.numel() for p in unfrozen.parameters())
    trainable_u = sum(p.numel() for p in unfrozen.parameters() if p.requires_grad)
    LOGGER.info("[Test 3] freeze=False: total=%d trainable=%d", total_u, trainable_u)
    assert trainable_u == total_u, "freeze=False 时应全部可训练"

    # 恢复冻结，避免影响其它测试
    unfrozen.set_freeze(True)
    assert sum(p.numel() for p in unfrozen.parameters() if p.requires_grad) == 0

    return {
        "freeze_true": {"total": total, "trainable": trainable},
        "freeze_false": {"total": total_u, "trainable": trainable_u},
    }


# --------------------------------------------------------------------------- #
# Test 4 —— 显存监测（Task H）
# --------------------------------------------------------------------------- #
def measure_forward_memory(encoder: MedSAMMultiLevelEncoder,
                           x: torch.Tensor) -> Optional[Dict[str, float]]:
    """测量单次 forward 的 CUDA 显存峰值。

    Args:
        encoder: 编码器。
        x: 输入张量（应已在 GPU 上）。

    Returns:
        峰值显存字典；非 CUDA 环境返回 ``None``。
    """
    if not torch.cuda.is_available() or not x.is_cuda:
        return None

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = encoder(x)
    torch.cuda.synchronize()

    return {
        "peak_allocated_mib": round(bytes_to_mib(torch.cuda.max_memory_allocated()), 2),
        "peak_reserved_mib": round(bytes_to_mib(torch.cuda.max_memory_reserved()), 2),
        "current_allocated_mib": round(bytes_to_mib(torch.cuda.memory_allocated()), 2),
        "device": torch.cuda.get_device_name(0),
    }


def test_gpu_memory() -> Dict[str, Any]:
    """记录随机张量 forward 的显存峰值。"""
    if not torch.cuda.is_available():
        LOGGER.warning("[Test 4] CUDA 不可用，跳过显存测试")
        return {"skipped": True, "reason": "CUDA not available"}

    encoder = get_encoder(freeze=True)
    torch.manual_seed(SEED)
    x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device="cuda")
    mem = measure_forward_memory(encoder, x)
    assert mem is not None
    LOGGER.info("[Test 4] 随机张量显存峰值: allocated=%.2f MiB reserved=%.2f MiB",
                mem["peak_allocated_mib"], mem["peak_reserved_mib"])
    return {"random_tensor": mem}


# --------------------------------------------------------------------------- #
# Test 5 —— 真实 MRI 单 slice（Task G）
# --------------------------------------------------------------------------- #
def find_t2w_case(data_root: Path, case_key: Optional[str] = None) -> Tuple[str, Path]:
    """定位一个含 T2W 的 PI-CAI case。

    Args:
        data_root: PI-CAI 根目录。
        case_key: 指定 ``{patient_id}_{study_id}``；``None`` 则取字典序第一个。

    Returns:
        ``(case_key, t2w_path)``。

    Raises:
        FileNotFoundError: 未找到符合条件的 T2W 文件。
    """
    images_root = data_root / "images"
    assert images_root.is_dir(), f"images 目录不存在: {images_root}"

    if case_key is not None:
        pid = case_key.split("_")[0]
        cands = sorted((images_root / pid).glob(f"{case_key}_*t2w.mha"))
        if not cands:
            raise FileNotFoundError(f"未找到 T2W: {images_root / pid}/{case_key}_*t2w.mha")
        return case_key, cands[0]

    for case_dir in sorted(p for p in images_root.iterdir() if p.is_dir()):
        cands = sorted(case_dir.glob("*_t2w.mha"))
        if cands:
            return cands[0].name[: -len("_t2w.mha")], cands[0]
    raise FileNotFoundError(f"在 {images_root} 下未找到任何 *_t2w.mha")


def preprocess_slice_for_smoke(slice_2d: np.ndarray,
                               size: int = INPUT_SIZE) -> torch.Tensor:
    """把一个 2D slice 处理成 MedSAM 输入（仅 smoke test）。

    步骤：0.5%–99.5% percentile clip -> 归一化到 [0,1] -> 双线性 resize
    -> 复制为 3 通道。

    Args:
        slice_2d: 单通道 2D 数组。
        size: 目标边长。

    Returns:
        ``[1, 3, size, size]`` 的 float32 张量。

    Raises:
        ValueError: slice 为常数（无法归一化）。
    """
    arr = slice_2d.astype(np.float32)
    lo, hi = np.percentile(arr, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError(
            f"slice 动态范围无效 (lo={lo}, hi={hi})，可能是空白层，请换 slice。"
        )
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo)

    t = torch.from_numpy(arr).float()[None, None]  # [1,1,H,W]
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    t = t.repeat(1, 3, 1, 1).contiguous()          # [1,3,size,size]
    return t


def test_real_mri_slice(data_root: Optional[Path] = None,
                        case_key: Optional[str] = None,
                        slice_index: Optional[int] = None) -> Dict[str, Any]:
    """真实 PI-CAI T2W 单 slice 送入 encoder 的 smoke test。

    所有参数均带默认值，以便 pytest 直接收集（避免被当作 fixture 请求）。

    Args:
        data_root: PI-CAI 根目录；``None`` 使用 ``DEFAULT_DATA_ROOT``。
        case_key: 指定 case；``None`` 自动选取。
        slice_index: axial slice 索引；``None`` 取中间层。

    Returns:
        含几何、统计、shape 与 finite 检查的结果字典。
    """
    import SimpleITK as sitk

    root = Path(data_root) if data_root is not None else Path(DEFAULT_DATA_ROOT)
    key, t2w_path = find_t2w_case(root, case_key)
    LOGGER.info("[Test 5] 使用 case=%s  file=%s", key, t2w_path.name)

    itk_img = sitk.ReadImage(str(t2w_path))
    volume = sitk.GetArrayFromImage(itk_img)  # [Z, Y, X]
    assert volume.ndim == 3, f"T2W 应为 3D，实际 {volume.shape}"

    z = slice_index if slice_index is not None else volume.shape[0] // 2
    assert 0 <= z < volume.shape[0], f"slice_index {z} 越界 (Z={volume.shape[0]})"
    slice_2d = volume[z]

    meta = {
        "case_key": key,
        "file": t2w_path.name,
        "volume_shape_zyx": list(volume.shape),
        "spacing_xyz": [round(float(s), 4) for s in itk_img.GetSpacing()],
        "origin_xyz": [round(float(o), 4) for o in itk_img.GetOrigin()],
        "direction": [round(float(d), 4) for d in itk_img.GetDirection()],
        "dtype": str(volume.dtype),
        "slice_index": int(z),
        "slice_stats": {
            "min": float(slice_2d.min()),
            "max": float(slice_2d.max()),
            "mean": float(slice_2d.mean()),
            "std": float(slice_2d.std()),
        },
    }
    LOGGER.info("[Test 5] 原始 T2W: shape=%s spacing=%s dtype=%s",
                meta["volume_shape_zyx"], meta["spacing_xyz"], meta["dtype"])
    LOGGER.info("[Test 5] slice[%d] 统计: %s", z, meta["slice_stats"])

    x = preprocess_slice_for_smoke(slice_2d).to(DEVICE)
    meta["input_shape"] = list(x.shape)
    LOGGER.info("[Test 5] 预处理后输入: %s (仅 smoke test 用)", tuple(x.shape))

    encoder = get_encoder(freeze=True)
    with torch.no_grad():
        features = encoder(x)

    finite = {k: bool(torch.isfinite(v).all()) for k, v in features.items()}
    shapes = {k: list(v.shape) for k, v in features.items()}

    LOGGER.info("[Test 5] 输出 shape: %s", shapes)
    LOGGER.info("[Test 5] 全部有限: %s", finite)

    for k, ok in finite.items():
        assert ok, f"真实 MRI slice 的 {k} 出现 NaN/Inf"
    for k, shape in shapes.items():
        assert tuple(shape) == EXPECTED_SHAPES[k], (
            f"{k} shape 不符: 期望 {EXPECTED_SHAPES[k]}, 实际 {tuple(shape)}"
        )

    meta["output_shapes"] = shapes
    meta["all_features_finite"] = finite

    mem = measure_forward_memory(encoder, x)
    if mem is not None:
        meta["memory"] = mem
        LOGGER.info("[Test 5] 真实 slice 显存峰值: allocated=%.2f MiB reserved=%.2f MiB",
                    mem["peak_allocated_mib"], mem["peak_reserved_mib"])

    # 同时打印整卷统计（约束要求打印 volume 级 min/max/mean/std）
    vol_stats = {
        "min": float(volume.min()), "max": float(volume.max()),
        "mean": float(volume.mean()), "std": float(volume.std()),
    }
    meta["volume_stats"] = vol_stats
    LOGGER.info("[Test 5] 整卷统计: %s", vol_stats)

    return meta


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    """顺序执行全部 smoke test 并打印摘要。"""
    parser = argparse.ArgumentParser(
        description="MedSAM 多层特征编码器 smoke test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT),
                        help="PI-CAI 根目录（含 images/）")
    parser.add_argument("--case-key", default=None,
                        help="指定 case，如 10000_1000000；默认自动选取")
    parser.add_argument("--slice-index", type=int, default=None,
                        help="axial slice 索引；默认取中间层")
    parser.add_argument("--skip-real-mri", action="store_true",
                        help="跳过真实 MRI smoke test")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    LOGGER.info("=" * 70)
    LOGGER.info("MedSAM ViT-B 多层特征 Smoke Test (device=%s, seed=%d)", DEVICE, SEED)
    LOGGER.info("=" * 70)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    summary: Dict[str, Any] = {"device": DEVICE, "seed": SEED}

    summary["checkpoint_load"] = test_checkpoint_load()
    summary["random_forward"] = test_random_forward()
    summary["freeze"] = test_freeze()
    summary["memory"] = test_gpu_memory()

    if args.skip_real_mri:
        LOGGER.warning("已跳过真实 MRI smoke test")
        summary["real_mri"] = {"skipped": True}
    else:
        summary["real_mri"] = test_real_mri_slice(
            data_root=args.data_root.resolve(),
            case_key=args.case_key,
            slice_index=args.slice_index,
        )

    # -------- 汇总 -------- #
    L: List[str] = []
    add = L.append
    add("")
    add("=" * 70)
    add("SUMMARY")
    add("=" * 70)
    add(f"device                : {summary['device']}")
    add(f"checkpoint            : {summary['checkpoint_load']['checkpoint']}")
    add(f"total encoder params  : {summary['checkpoint_load']['total_params']:,}")
    add(f"trainable (frozen)    : {summary['freeze']['freeze_true']['trainable']}")
    add("")
    add("random tensor forward:")
    for k, v in summary["random_forward"]["output_shapes"].items():
        add(f"  {k:<5}: {tuple(v)}")
    add(f"  all finite          : {all(summary['random_forward']['all_finite'].values())}")
    if not summary["memory"].get("skipped"):
        m = summary["memory"]["random_tensor"]
        add(f"  peak allocated      : {m['peak_allocated_mib']} MiB")
        add(f"  peak reserved       : {m['peak_reserved_mib']} MiB")
    add("")
    rm = summary["real_mri"]
    if rm.get("skipped"):
        add("real MRI slice        : SKIPPED")
    else:
        add("real MRI slice:")
        add(f"  case                : {rm['case_key']}  ({rm['file']})")
        add(f"  original volume     : {rm['volume_shape_zyx']} ({rm['dtype']})")
        add(f"  spacing (x,y,z)     : {rm['spacing_xyz']}")
        add(f"  origin              : {rm['origin_xyz']}")
        add(f"  direction           : {rm['direction']}")
        add(f"  volume min/max/mean/std: {rm['volume_stats']}")
        add(f"  slice index         : {rm['slice_index']}")
        add(f"  slice min/max/mean/std : {rm['slice_stats']}")
        add(f"  input shape         : {rm['input_shape']}")
        add(f"  output shapes       : {rm['output_shapes']}")
        add(f"  all features finite : {rm['all_features_finite']}")
        if "memory" in rm:
            add(f"  peak allocated      : {rm['memory']['peak_allocated_mib']} MiB")
            add(f"  peak reserved       : {rm['memory']['peak_reserved_mib']} MiB")
    add("")
    add("ALL TESTS PASSED")
    add("=" * 70)

    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
