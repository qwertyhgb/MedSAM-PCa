#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""多阈值切片级指标累计（Phase 2A）。

一次 model forward 得到的概率图可以同时服务任意多个阈值：若对每个阈值都
重新做一遍 ``probs >= t`` + TP/FP/FN/TN 统计，代价随阈值数量线性增长。
本模块改用**概率直方图**：对每张切片只需一次 ``bincount``，即可在
``O(n_bins)`` 时间内得到任意阈值的混淆统计。

实现要点
--------
* 概率被量化到 ``bin_width``（默认 5e-4，共 2000 个 bin），
  ``idx = floor(p / bin_width)``；
* 阈值 ``t`` 对应的起始 bin 由 :func:`src.metrics.segmentation.threshold_to_bin`
  计算，当 ``t`` 是 ``bin_width`` 的整数倍时，``p >= t`` 与 ``idx >= k`` **严格等价**；
* 每个阈值的累计复用 :class:`~src.metrics.segmentation.SegmentationMetrics`
  的统计逻辑（``update_from_counts``），因此与精确布尔实现口径完全一致。

默认阈值网格（0.05, 0.10, ..., 0.95）均为 5e-4 的整数倍，因此结果精确。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .segmentation import SegmentationMetrics, threshold_to_bin

__all__ = ["ThresholdScanMetrics", "DEFAULT_THRESHOLDS"]

#: 默认阈值网格。
DEFAULT_THRESHOLDS: tuple = tuple(round(0.05 * i, 2) for i in range(1, 20))

#: 默认直方图 bin 宽度（0.0005 → 2000 个 bin）。
DEFAULT_BIN_WIDTH = 5e-4


class ThresholdScanMetrics:
    """在**一次遍历**中同时累计多个阈值的切片级指标。

    用法::

        scan = ThresholdScanMetrics([0.1, 0.3, 0.5])
        for batch:
            scan.update_probabilities(probs, gt, case_ids=[...])
        rows = scan.rows()          # 每个阈值一行指标

    Args:
        thresholds: 概率阈值序列，需落在 ``(0, 1]``。
        bin_width: 概率直方图 bin 宽度。
        eps: 指标平滑项。

    Raises:
        ValueError: 阈值为空、越界或 ``bin_width`` 非正。
    """

    def __init__(self, thresholds: Sequence[float],
                 bin_width: float = DEFAULT_BIN_WIDTH,
                 eps: float = 1e-8) -> None:
        ths = [float(t) for t in thresholds]
        if not ths:
            raise ValueError("thresholds 不能为空")
        if any(not 0.0 < t <= 1.0 for t in ths):
            raise ValueError(f"thresholds 必须落在 (0, 1]，收到 {ths}")
        if bin_width <= 0:
            raise ValueError(f"bin_width 必须为正，收到 {bin_width}")

        self.thresholds = ths
        self.bin_width = float(bin_width)
        self.n_bins = int(math.ceil(1.0 / self.bin_width)) + 1
        self._bin_index = [threshold_to_bin(t, self.bin_width) for t in ths]
        self.metrics: List[SegmentationMetrics] = [
            SegmentationMetrics(threshold=t, eps=eps) for t in ths
        ]
        self._n_updates = 0

    # ------------------------------------------------------------------ #
    def update_probabilities(self, probabilities: Any, gt: Any,
                             case_ids: Optional[List[str]] = None) -> None:
        """累计一个 batch 的概率图与 GT（同时更新全部阈值）。

        Args:
            probabilities: ``[B, 1, H, W]`` / ``[B, H, W]`` / ``[H, W]`` 概率，
                取值需落在 ``[0, 1]``。
            gt: 与 ``probabilities`` 同 shape 的 GT（内部按 ``!= 0`` 二值化）。
            case_ids: 长度 B 的 case 标识。

        Raises:
            ValueError: shape 不一致、维度不合法、取值越界或含非有限值。
        """
        probs = _to_numpy(probabilities)
        if probs.dtype == bool:
            probs = probs.astype(np.float32)
        if probs.size:
            if not np.isfinite(probs).all():
                raise ValueError("probabilities 含 NaN/Inf")
            lo, hi = float(probs.min()), float(probs.max())
            if lo < 0.0 or hi > 1.0:
                raise ValueError(
                    f"probabilities 取值 [{lo:.4f}, {hi:.4f}] 超出 [0,1]；"
                    "若输入是 logits 请先做 sigmoid。"
                )

        p = SegmentationMetrics._normalize_to_bhw(np.asarray(probs, dtype=np.float32))
        g = SegmentationMetrics._normalize_to_bhw(_to_numpy(gt) != 0)
        if p.shape != g.shape:
            raise ValueError(f"shape 不一致: probs {p.shape} vs gt {g.shape}")

        idx = (p * (1.0 / self.bin_width)).astype(np.int32)
        np.clip(idx, 0, self.n_bins - 1, out=idx)
        n_bins = self.n_bins

        for i in range(p.shape[0]):
            flat_idx = idx[i].ravel()
            flat_gt = g[i].ravel()
            n_total = int(flat_idx.size)
            n_gt = int(flat_gt.sum())

            hist_all = np.bincount(flat_idx, minlength=n_bins)
            if n_gt:
                hist_gt = np.bincount(flat_idx[flat_gt], minlength=n_bins)
                suf_gt = np.cumsum(hist_gt[::-1])[::-1]
            else:
                suf_gt = None
            suf_all = np.cumsum(hist_all[::-1])[::-1]

            cid = None
            if case_ids is not None and i < len(case_ids):
                cid = str(case_ids[i])

            for m, k in zip(self.metrics, self._bin_index):
                if k >= n_bins:
                    tp = pred_pos = 0
                else:
                    pred_pos = int(suf_all[k])
                    tp = int(suf_gt[k]) if suf_gt is not None else 0
                fp = pred_pos - tp
                fn = n_gt - tp
                tn = n_total - n_gt - fp
                m.update_from_counts(tp, fp, fn, tn, cid)

        self._n_updates += 1

    # ------------------------------------------------------------------ #
    def rows(self) -> List[Dict[str, Any]]:
        """返回每个阈值一行的指标表。

        Returns:
            行列表，键包含 ``threshold`` / ``positive_slice_dice`` /
            ``positive_slice_iou`` / ``all_slice_dice`` / ``micro_dice`` /
            ``micro_precision`` / ``micro_recall`` / ``micro_specificity`` /
            ``fp_slice_rate`` / ``fp_slices`` 等。
        """
        out: List[Dict[str, Any]] = []
        for m in self.metrics:
            s = m.summary()
            out.append({
                "threshold": m.threshold,
                "positive_slice_dice": s["positive_slice_dice"],
                "positive_slice_iou": s["positive_slice_iou"],
                "all_slice_dice": s["all_slice_dice"],
                "micro_dice": s["micro_dice"],
                "micro_iou": s["micro_iou"],
                "micro_precision": s["micro_precision"],
                "micro_recall": s["micro_recall"],
                "micro_specificity": s["micro_specificity"],
                "fp_slice_rate": s["fp_slice_rate"],
                "fp_slices": s["fp_slices"],
                "num_slices": s["num_slices"],
                "num_positive_gt_slices": s["num_positive_gt_slices"],
                "num_empty_gt_slices": s["num_empty_gt_slices"],
            })
        return out

    def best_by(self, key: str = "positive_slice_dice") -> Dict[str, Any]:
        """按给定指标挑选最优阈值。

        Args:
            key: 排序依据的指标名。

        Returns:
            该指标最高的一行（``None`` 值被跳过）。

        Raises:
            KeyError: 指标名不存在。
        """
        rows = [r for r in self.rows() if r.get(key) is not None]
        if not rows:
            raise KeyError(f"没有可用的 {key} 指标")
        return max(rows, key=lambda r: float(r[key]))

    def reset(self) -> None:
        """重置全部阈值的累计状态。"""
        for m in self.metrics:
            m.reset()
        self._n_updates = 0


def _to_numpy(x: Any) -> np.ndarray:
    """torch.Tensor / ndarray / 序列 -> numpy 数组（保留 dtype）。"""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)
