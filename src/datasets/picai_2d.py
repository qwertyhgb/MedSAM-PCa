#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PI-CAI 2D axial slice 数据集（Phase 1）。

设计要点
--------
* **只使用 T2W**，单通道复制为 3 通道 ``[T2, T2, T2]``。
* 不预导出 PNG/JPG；按需从 ``.mha`` 中读取**单个 slice**
  （``ImageFileReader.SetExtractIndex``），避免整卷常驻内存。
* lesion mask 一律 ``mask > 0`` 二值化（标签值可能是 1 / 2/3/4/5）。
* mask 与 T2W 几何不一致时，以 **T2W 为 reference** 做
  **nearest-neighbor** 重采样（绝不修改原始标注文件）。
* 预处理：非零区域 percentile clip (0.5/99.5) → [0,1] → resize 1024
  （image bilinear，mask nearest）→ 3 通道复制。
* 训练 augmentation 仅：水平翻转、小角度旋转、小尺度缩放（image/mask 同步）。

约束见 ``PROJECT_CONSTRAINTS.md``。
"""
from __future__ import annotations

import functools
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as TF

LOGGER = logging.getLogger(__name__)

GEOM_TOLERANCE = 1e-4
CLIP_LOW_PCT = 0.5
CLIP_HIGH_PCT = 99.5

__all__ = [
    "PICAI2DDataset",
    "BalancedSliceSampler",
    "read_geometry",
    "geometries_match",
    "scan_case_slices",
    "clip_normalize",
    "resize_image_tensor",
    "resize_mask_tensor",
    "augment_pair",
]


# --------------------------------------------------------------------------- #
# 几何与读取
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=512)
def read_geometry(path: str) -> Dict[str, Any]:
    """读取 image header 的几何信息（带 LRU 缓存）。

    Args:
        path: 影像文件路径。

    Returns:
        含 ``size`` / ``spacing`` / ``origin`` / ``direction`` / ``pixel_id``。
    """
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    reader.ReadImageInformation()
    return {
        "size": tuple(int(v) for v in reader.GetSize()),
        "spacing": tuple(float(v) for v in reader.GetSpacing()),
        "origin": tuple(float(v) for v in reader.GetOrigin()),
        "direction": tuple(float(v) for v in reader.GetDirection()),
        "pixel_id": reader.GetPixelID(),
    }


def geometries_match(a: Dict[str, Any], b: Dict[str, Any],
                     tol: float = GEOM_TOLERANCE) -> bool:
    """判断两组几何是否一致。

    Args:
        a: 几何字典。
        b: 几何字典。
        tol: 数值容差。

    Returns:
        一致返回 ``True``。
    """
    if tuple(a["size"]) != tuple(b["size"]):
        return False
    for field in ("spacing", "origin", "direction"):
        va, vb = tuple(a[field]), tuple(b[field])
        if len(va) != len(vb) or any(abs(x - y) > tol for x, y in zip(va, vb)):
            return False
    return True


def read_slice(path: str, z: int, geometry: Dict[str, Any]) -> np.ndarray:
    """从 volume 中只读取第 ``z`` 层 axial slice。

    Args:
        path: 影像文件路径。
        z: axial 索引。
        geometry: :func:`read_geometry` 的结果。

    Returns:
        ``[H, W]`` 的 2D 数组。
    """
    w, h, depth = geometry["size"]
    if not 0 <= z < depth:
        raise IndexError(f"slice {z} 越界 (Z={depth}) 于 {path}")
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    reader.ReadImageInformation()
    reader.SetExtractIndex([0, 0, z])
    reader.SetExtractSize([w, h, 1])
    arr = sitk.GetArrayFromImage(reader.Execute())  # [1, H, W]
    return arr[0]


def _resample_mask_to_reference(mask_path: str, ref_path: str) -> sitk.Image:
    """把 mask 以 nearest-neighbor 重采样到 reference 的几何。

    Args:
        mask_path: mask 文件路径。
        ref_path: reference（T2W）文件路径。

    Returns:
        重采样后的 ``sitk.Image``（标签值不变，仅几何对齐）。
    """
    mask_img = sitk.ReadImage(mask_path)
    ref_img = sitk.ReadImage(ref_path)
    return sitk.Resample(
        mask_img, ref_img, sitk.Transform(),
        sitk.sitkNearestNeighbor, 0, mask_img.GetPixelID(),
    )


def aligned_cache_path(mask_path: str, t2w_path: str,
                       cache_dir: Path | str) -> Path:
    """计算「对齐后 mask」的缓存路径（**不触发任何写入**）。

    使用 ``md5(mask_path|t2w_path)`` 作为键，因此同一 mask 在不同 T2W 参考下
    会得到不同缓存，不会互相覆盖。

    Args:
        mask_path: 原始 mask 路径。
        t2w_path: T2W（reference）路径。
        cache_dir: 缓存目录。

    Returns:
        缓存文件路径。
    """
    key = hashlib.md5(f"{mask_path}|{t2w_path}".encode("utf-8")).hexdigest()[:16]
    stem = Path(mask_path).name.replace(".nii.gz", "").replace(".nii", "")
    return Path(cache_dir) / f"{stem}__{key}.nii.gz"


def _ensure_aligned_mask(mask_path: str, t2w_path: str, cache_dir: Path) -> str:
    """确保存在一份与 T2W 几何一致的 mask，必要时生成磁盘缓存。

    **不会修改原始标注文件**；结果写入 ``cache_dir``（项目 ``data/cache/`` 下，
    已被 ``.gitignore`` 排除）。使用 md5(mask|t2w) 作为缓存键，原子写入。

    Args:
        mask_path: 原始 mask 路径。
        t2w_path: T2W（reference）路径。
        cache_dir: 缓存目录。

    Returns:
        对齐后 mask 的文件路径。
    """
    out = aligned_cache_path(mask_path, t2w_path, cache_dir)
    if out.is_file():
        return str(out)

    aligned = _resample_mask_to_reference(mask_path, t2w_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp.nii.gz")
    sitk.WriteImage(aligned, str(tmp), True)
    tmp.replace(out)  # 原子替换，避免多 worker 竞争产生半成品
    LOGGER.info("已缓存对齐 mask: %s", out)
    return str(out)


def read_mask_slice(mask_path: str, t2w_path: str, z: int,
                    t2w_geometry: Dict[str, Any],
                    cache_dir: Optional[Path] = None) -> np.ndarray:
    """读取与 T2W 第 ``z`` 层空间对应的 mask slice。

    几何一致时直接抽取单层（快）；不一致时以 T2W 为 reference 做
    nearest-neighbor 对齐（结果磁盘缓存，避免每次重复重采样），再取层。

    Args:
        mask_path: lesion mask 路径。
        t2w_path: T2W 路径（reference）。
        z: axial 索引。
        t2w_geometry: T2W 几何。
        cache_dir: 重采样缓存目录；``None`` 时退回"每次现场重采样"。

    Returns:
        ``[H, W]`` 的 2D 数组。
    """
    mask_geom = read_geometry(mask_path)
    if geometries_match(mask_geom, t2w_geometry):
        return read_slice(mask_path, z, mask_geom)

    if cache_dir is not None:
        aligned_path = _ensure_aligned_mask(mask_path, t2w_path, Path(cache_dir))
        return read_slice(aligned_path, z, read_geometry(aligned_path))

    resampled = _resample_mask_to_reference(mask_path, t2w_path)
    arr = sitk.GetArrayFromImage(resampled)
    if not 0 <= z < arr.shape[0]:
        raise IndexError(
            f"重采样后 mask 层数 {arr.shape[0]} 与请求 slice {z} 不匹配于 {mask_path}"
        )
    return arr[z]


def scan_case_slices(case_id: str, patient_id: str, t2w_path: str,
                     mask_path: Optional[str]) -> Dict[str, Any]:
    """扫描单个 case，返回每层的病灶像素数（用于构建 slice manifest）。

    该函数运行在子进程中，必须是模块级可 pickle 函数。

    Args:
        case_id: ``{patient_id}_{study_id}``。
        patient_id: 患者 ID。
        t2w_path: T2W 绝对路径。
        mask_path: lesion mask 绝对路径；``None`` 表示无标注。

    Returns:
        含 ``num_slices`` 与 ``per_slice_lesion_pixels`` 的字典。
    """
    t2w_geom = read_geometry(t2w_path)
    num_slices = t2w_geom["size"][2]
    counts = [0] * num_slices
    status = "no_mask"

    if mask_path is not None:
        mask_geom = read_geometry(mask_path)
        if geometries_match(mask_geom, t2w_geom):
            arr = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
            status = "aligned"
        else:
            arr = sitk.GetArrayFromImage(_resample_mask_to_reference(mask_path, t2w_path))
            status = "resampled"

    if mask_path is not None:
        fg = arr > 0
        n = min(num_slices, fg.shape[0])
        counts = [int(fg[z].sum()) for z in range(n)]
        if n < num_slices:
            LOGGER.warning("%s: mask 层数 %d < T2W 层数 %d，尾部补 0",
                           case_id, fg.shape[0], num_slices)
        del arr, fg

    return {
        "case_id": case_id,
        "patient_id": patient_id,
        "num_slices": int(num_slices),
        "per_slice_lesion_pixels": counts,
        "mask_status": status,
    }


# --------------------------------------------------------------------------- #
# 预处理
# --------------------------------------------------------------------------- #
def clip_normalize(arr: np.ndarray, low_pct: float = CLIP_LOW_PCT,
                   high_pct: float = CLIP_HIGH_PCT,
                   eps: float = 1e-6) -> np.ndarray:
    """非零区域 percentile clip 后归一化到 [0, 1]。

    Fallback 策略（显式、可预期）：
        1. 非零像素为空 → 退化为整幅 percentile；
        2. 动态范围退化（``hi - lo < eps``）或非有限 → 返回全零。

    Args:
        arr: 2D 单通道数组。
        low_pct: 低百分位。
        high_pct: 高百分位。
        eps: 动态范围下限。

    Returns:
        ``float32`` 的 ``[H, W]`` 数组，取值在 [0, 1]。
    """
    a = arr.astype(np.float32, copy=False)
    nonzero = a[a > 0]
    if nonzero.size > 0:
        lo, hi = np.percentile(nonzero, [low_pct, high_pct])
    else:
        lo, hi = float(a.min()), float(a.max())

    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < eps:
        return np.zeros_like(a, dtype=np.float32)

    out = np.clip(a, lo, hi)
    out = (out - lo) / (hi - lo)
    return out.astype(np.float32, copy=False)


def resize_image_tensor(img: torch.Tensor, size: int) -> torch.Tensor:
    """双线性 resize 单通道图像张量。

    Args:
        img: ``[1, H, W]`` 或 ``[H, W]``。
        size: 目标边长。

    Returns:
        与输入同维度的张量。
    """
    squeeze = img.dim() == 2
    t = img[None, None] if squeeze else img[None]
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t[0, 0] if squeeze else t[0]


def resize_mask_tensor(mask: torch.Tensor, size: int) -> torch.Tensor:
    """最近邻 resize 单通道 mask 张量，并断言取值仍为 {0, 1}。

    Args:
        mask: ``[1, H, W]`` 或 ``[H, W]``，取值 {0, 1}。
        size: 目标边长。

    Returns:
        同维度张量。

    Raises:
        AssertionError: resize 后出现 {0,1} 之外的值。
    """
    squeeze = mask.dim() == 2
    t = mask[None, None] if squeeze else mask[None]
    t = F.interpolate(t, size=(size, size), mode="nearest")
    t = (t > 0.5).to(torch.float32)

    uniq = torch.unique(t)
    assert torch.all((uniq == 0) | (uniq == 1)), (
        f"mask resize 后出现非 0/1 取值: {uniq.tolist()}"
    )
    return t[0, 0] if squeeze else t[0]


def augment_pair(image: torch.Tensor, mask: torch.Tensor,
                 rng: np.random.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    """同步的轻度几何增强（水平翻转 + 小角度旋转 + 小尺度缩放）。

    明确不包含：垂直翻转、elastic、强度/颜色增强、CutMix、MixUp。

    Args:
        image: ``[3, H, W]``。
        mask: ``[1, H, W]``。
        rng: numpy 随机数生成器。

    Returns:
        ``(image, mask)``，两者变换完全一致；mask 保持 {0,1}。
    """
    if rng.random() < 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)

    angle = float(rng.uniform(-15.0, 15.0))
    scale = float(rng.uniform(0.9, 1.1))

    image = TF.affine(image, angle=angle, translate=[0, 0], scale=scale, shear=[0.0],
                      interpolation=InterpolationMode.BILINEAR, fill=0)
    mask = TF.affine(mask, angle=angle, translate=[0, 0], scale=scale, shear=[0.0],
                     interpolation=InterpolationMode.NEAREST, fill=0)
    mask = (mask > 0.5).to(torch.float32)

    uniq = torch.unique(mask)
    assert torch.all((uniq == 0) | (uniq == 1)), (
        f"augmentation 后 mask 出现非 0/1 取值: {uniq.tolist()}"
    )
    return image, mask


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class PICAI2DDataset(Dataset):
    """PI-CAI T2W axial slice 数据集。

    Args:
        records: slice 级记录（``DataFrame`` 或 ``list[dict]``），需包含
            ``case_id`` / ``patient_id`` / ``slice_idx`` / ``t2w_path`` /
            ``mask_path`` / ``is_positive``。路径为**相对 ``data_root``** 的路径。
        data_root: PI-CAI 根目录。
        image_size: 输出边长，Phase 1 固定 1024。
        augment: 是否启用训练增强（validation 必须为 ``False``）。
        seed: 增强随机种子基准。

    Returns:
        ``__getitem__`` 返回 ``{"image": [3,S,S], "mask": [1,S,S], ...}``。
    """

    def __init__(
        self,
        records: Sequence[Dict[str, Any]] | pd.DataFrame,
        data_root: Path | str,
        image_size: int = 1024,
        augment: bool = False,
        seed: int = 0,
        mask_cache_dir: Optional[Path | str] = None,
    ) -> None:
        self.records: pd.DataFrame = (
            records.copy() if isinstance(records, pd.DataFrame)
            else pd.DataFrame(list(records))
        )
        if len(self.records) == 0:
            raise ValueError("records 为空，无法构建数据集")

        required = {"case_id", "patient_id", "slice_idx", "t2w_path", "is_positive"}
        missing = required - set(self.records.columns)
        if missing:
            raise KeyError(f"records 缺少必要列: {sorted(missing)}")

        self.data_root = Path(data_root)
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.seed = int(seed)
        # 需要重采样对齐的 mask 会缓存到这里（不修改原始标注）
        self.mask_cache_dir = Path(mask_cache_dir) if mask_cache_dir else None
        self._epoch = 0

        if "mask_path" not in self.records.columns:
            self.records["mask_path"] = None

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        """设置 epoch（用于增强随机性可复现）。"""
        self._epoch = int(epoch)

    @property
    def labels(self) -> np.ndarray:
        """返回每条记录的 ``is_positive``（int64），供 sampler 使用。"""
        return self.records["is_positive"].astype(bool).to_numpy().astype(np.int64)

    def _abs(self, rel: Optional[str]) -> Optional[str]:
        """把 manifest 中的相对路径转成绝对路径字符串。"""
        if rel is None or (isinstance(rel, float) and np.isnan(rel)) or str(rel) == "":
            return None
        p = Path(str(rel))
        return str(p if p.is_absolute() else self.data_root / p)

    # ------------------------------------------------------------------ #
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """取一个 slice，返回预处理后的 image/mask 及元信息。"""
        row = self.records.iloc[index]
        t2w_path = self._abs(row["t2w_path"])
        mask_path = self._abs(row.get("mask_path"))
        z = int(row["slice_idx"])
        if t2w_path is None:
            raise FileNotFoundError(f"记录 {index} 的 t2w_path 为空")

        t2w_geom = read_geometry(t2w_path)
        raw = read_slice(t2w_path, z, t2w_geom)                     # [H, W]
        img = clip_normalize(raw)

        if mask_path is not None:
            m = read_mask_slice(mask_path, t2w_path, z, t2w_geom,
                                cache_dir=self.mask_cache_dir)       # [H, W]
            mask = (m > 0).astype(np.float32)                       # 关键: > 0
        else:
            mask = np.zeros_like(raw, dtype=np.float32)

        img_t = torch.from_numpy(img)[None]                          # [1,H,W]
        mask_t = torch.from_numpy(mask)[None]                        # [1,H,W]

        img_t = resize_image_tensor(img_t, self.image_size)
        mask_t = resize_mask_tensor(mask_t, self.image_size)

        if self.augment:
            rng = np.random.default_rng(self.seed + self._epoch * 1_000_003 + index)
            img_t, mask_t = augment_pair(img_t, mask_t, rng)

        image = img_t.repeat(3, 1, 1).contiguous()                   # [3,S,S] = [T2,T2,T2]

        return {
            "image": image,
            "mask": mask_t,
            "case_id": str(row["case_id"]),
            "patient_id": str(row["patient_id"]),
            "slice_idx": z,
            "is_positive": bool(mask_t.any().item()),
        }


# --------------------------------------------------------------------------- #
# Sampler
# --------------------------------------------------------------------------- #
class BalancedSliceSampler(Sampler[int]):
    """正负 slice 平衡采样器（Phase 1: positive : negative ≈ 1:1）。

    不删除任何 negative slice，只是改变采样频率；validation 不使用本采样器。

    Args:
        labels: 每条记录的 0/1 标签。
        positive_ratio: ``pos / (pos + neg)`` 目标比例，0.5 即 1:1。
        num_samples: 每个 epoch 的样本数；``None`` 表示用 ``2 * n_pos``。
        seed: 随机种子。
        shuffle: 是否打乱输出顺序。
    """

    def __init__(self, labels: np.ndarray, positive_ratio: float = 0.5,
                 num_samples: Optional[int] = None, seed: int = 0,
                 shuffle: bool = True) -> None:
        labels = np.asarray(labels).astype(np.int64)
        if not ((labels == 0) | (labels == 1)).all():
            raise ValueError("labels 必须只含 0/1")
        self.pos_idx = np.flatnonzero(labels == 1).tolist()
        self.neg_idx = np.flatnonzero(labels == 0).tolist()
        if not self.pos_idx:
            raise ValueError("训练集里没有任何 positive slice，无法平衡采样")
        if not self.neg_idx:
            raise ValueError("训练集里没有任何 negative slice")

        if not 0.0 < positive_ratio < 1.0:
            raise ValueError(f"positive_ratio 必须在 (0,1) 内，收到 {positive_ratio}")
        self.positive_ratio = float(positive_ratio)
        self.num_samples = int(num_samples) if num_samples else int(
            round(len(self.pos_idx) / self.positive_ratio)
        )
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """设置 epoch。"""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        n_pos = int(round(self.num_samples * self.positive_ratio))
        n_neg = self.num_samples - n_pos

        pos = rng.choice(self.pos_idx, size=n_pos, replace=n_pos > len(self.pos_idx))
        neg = rng.choice(self.neg_idx, size=n_neg, replace=n_neg > len(self.neg_idx))

        idx = np.concatenate([pos, neg]).tolist()
        if self.shuffle:
            rng.shuffle(idx)
        return iter(idx)
