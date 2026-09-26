#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""分割评价指标（Phase 1）。

实现：Dice、IoU、Precision、Recall/Sensitivity、Specificity。

**空 GT 的处理是刻意的**：

自动病灶分割里，绝大部分 slice 的 GT 为空。若把 "GT 空 & 预测空" 的
Dice 记为 1.0 后直接混进平均，会严重高估性能。因此本模块同时报告：

* ``positive_slice_dice`` —— **只在 GT 非空的 slice 上**平均（论文主指标）
* ``all_slice_dice`` —— 所有 slice 平均，且**单独统计**其中有多少来自空 GT
* ``fp_slices`` —— GT 为空但预测非空的 slice 数（FP/case 的基础）

像素级 Precision/Recall/Specificity 采用全局微平均（先累计 TP/FP/FN/TN）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

__all__ = [
    "SliceConfusion",
    "compute_confusion",
    "metrics_from_confusion",
    "SegmentationMetrics",
]


@dataclass
class SliceConfusion:
    """单个 slice（或像素集合）的混淆统计。"""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def gt_positive(self) -> bool:
        """GT 是否含前景。"""
        return (self.tp + self.fn) > 0

    @property
    def pred_positive(self) -> bool:
        """预测是否含前景。"""
        return (self.tp + self.fp) > 0


def compute_confusion(pred: np.ndarray, gt: np.ndarray) -> SliceConfusion:
    """计算二值预测与二值 GT 的 TP/FP/FN/TN。

    Args:
        pred: 二值数组（``bool`` 或 0/1）。
        gt: 二值数组，shape 必须一致。

    Returns:
        :class:`SliceConfusion`。

    Raises:
        ValueError: shape 不一致。
    """
    p = np.asarray(pred).astype(bool, copy=False)
    g = np.asarray(gt).astype(bool, copy=False)
    if p.shape != g.shape:
        raise ValueError(f"shape 不一致: pred {p.shape} vs gt {g.shape}")
    return SliceConfusion(
        tp=int(np.logical_and(p, g).sum()),
        fp=int(np.logical_and(p, ~g).sum()),
        fn=int(np.logical_and(~p, g).sum()),
        tn=int(np.logical_and(~p, ~g).sum()),
    )


def metrics_from_confusion(c: SliceConfusion, eps: float = 1e-8) -> Dict[str, float]:
    """由混淆统计计算 Dice / IoU / Precision / Recall / Specificity。

    Args:
        c: 混淆统计。
        eps: 平滑项。

    Returns:
        指标字典。空 GT 且空预测时 Dice=IoU=1.0（调用方需自行决定是否计入）。
    """
    tp, fp, fn, tn = float(c.tp), float(c.fp), float(c.fn), float(c.tn)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
    }


@dataclass
class SegmentationMetrics:
    """跨 batch / 跨 slice 的累计评估器。

    用法::

        m = SegmentationMetrics()
        for batch:
            m.update(logits_or_prob, gt_mask, case_ids=[...])
        print(m.summary())
    """

    threshold: float = 0.5
    eps: float = 1e-8

    _tp: int = field(default=0, init=False)
    _fp: int = field(default=0, init=False)
    _fn: int = field(default=0, init=False)
    _tn: int = field(default=0, init=False)

    _n_slices: int = field(default=0, init=False)
    _n_empty_gt: int = field(default=0, init=False)
    _n_positive_gt: int = field(default=0, init=False)
    _fp_slices: int = field(default=0, init=False)
    _empty_gt_empty_pred: int = field(default=0, init=False)

    _dice_pos: List[float] = field(default_factory=list, init=False)
    _dice_all: List[float] = field(default_factory=list, init=False)
    _iou_pos: List[float] = field(default_factory=list, init=False)

    _per_case_tp: Dict[str, int] = field(default_factory=dict, init=False)
    _per_case_fp: Dict[str, int] = field(default_factory=dict, init=False)
    _per_case_fn: Dict[str, int] = field(default_factory=dict, init=False)
    _per_case_gt_slices: Dict[str, int] = field(default_factory=dict, init=False)

    # ------------------------------------------------------------------ #
    def update(self, pred: np.ndarray | "torch.Tensor",  # type: ignore[name-defined]
               gt: np.ndarray | "torch.Tensor",  # type: ignore[name-defined]
               case_ids: Optional[List[str]] = None) -> None:
        """累计一个 batch。

        Args:
            pred: ``[B, 1, H, W]`` 概率或 logits，或二值 mask。
            gt: ``[B, 1, H, W]`` 二值 mask。
            case_ids: 长度 B 的 case 标识，用于 per-case 统计。

        Raises:
            ValueError: 维度不合法。
        """
        p = self._to_numpy_binary(pred)
        g = self._to_numpy_binary(gt, is_probability=False)
        if p.shape != g.shape:
            raise ValueError(f"shape 不一致: pred {p.shape} vs gt {g.shape}")

        p = self._normalize_to_bhw(p)
        g = self._normalize_to_bhw(g)

        for i in range(p.shape[0]):
            pi, gi = p[i], g[i]
            c = compute_confusion(pi, gi)
            self._tp += c.tp
            self._fp += c.fp
            self._fn += c.fn
            self._tn += c.tn
            self._n_slices += 1

            m = metrics_from_confusion(c, eps=self.eps)
            self._dice_all.append(m["dice"])
            if c.gt_positive:
                self._n_positive_gt += 1
                self._dice_pos.append(m["dice"])
                self._iou_pos.append(m["iou"])
            else:
                self._n_empty_gt += 1
                if c.pred_positive:
                    self._fp_slices += 1
                else:
                    self._empty_gt_empty_pred += 1

            if case_ids is not None and i < len(case_ids):
                cid = str(case_ids[i])
                self._per_case_tp[cid] = self._per_case_tp.get(cid, 0) + c.tp
                self._per_case_fp[cid] = self._per_case_fp.get(cid, 0) + c.fp
                self._per_case_fn[cid] = self._per_case_fn.get(cid, 0) + c.fn
                if c.gt_positive:
                    self._per_case_gt_slices[cid] = self._per_case_gt_slices.get(cid, 0) + 1

    @staticmethod
    def _normalize_to_bhw(x: np.ndarray) -> np.ndarray:
        """把 ``[B,1,H,W]`` / ``[B,H,W]`` / ``[H,W]`` 统一成 ``[B,H,W]``。

        Args:
            x: 输入数组。

        Returns:
            ``[B, H, W]`` 数组。

        Raises:
            ValueError: 维度或通道数不受支持。
        """
        if x.ndim == 4:
            if x.shape[1] != 1:
                raise ValueError(f"通道维必须为 1，收到 {x.shape}")
            return x[:, 0]
        if x.ndim == 3:
            return x
        if x.ndim == 2:
            return x[None]
        raise ValueError(f"期望 [B,1,H,W] / [B,H,W] / [H,W]，收到 {x.shape}")

    def _to_numpy_binary(self, x, is_probability: bool = True) -> np.ndarray:
        """把 tensor/ndarray 转成二值 numpy mask。"""
        if hasattr(x, "detach"):  # torch.Tensor
            arr = x.detach().float().cpu().numpy()
        else:
            arr = np.asarray(x)
        if is_probability:
            if arr.dtype == bool:
                return arr
            if arr.min() < 0 or arr.max() > 1:  # logits
                arr = 1.0 / (1.0 + np.exp(-arr))
            return arr >= self.threshold
        return arr.astype(bool, copy=False)

    # ------------------------------------------------------------------ #
    def summary(self) -> Dict[str, Any]:
        """返回汇总指标。

        Returns:
            含 ``positive_slice_dice`` / ``all_slice_dice`` / 像素级微平均指标 /
            空 GT 统计 / per-case FP 的字典。
        """
        micro = metrics_from_confusion(
            SliceConfusion(self._tp, self._fp, self._fn, self._tn), eps=self.eps
        )
        out: Dict[str, Any] = {
            "num_slices": self._n_slices,
            "num_positive_gt_slices": self._n_positive_gt,
            "num_empty_gt_slices": self._n_empty_gt,
            "positive_slice_dice": float(np.mean(self._dice_pos)) if self._dice_pos else None,
            "positive_slice_iou": float(np.mean(self._iou_pos)) if self._iou_pos else None,
            "all_slice_dice": float(np.mean(self._dice_all)) if self._dice_all else None,
            "all_slice_dice_including_empty": float(np.mean(self._dice_all)) if self._dice_all else None,
            "empty_gt_empty_pred_slices": self._empty_gt_empty_pred,
            "fp_slices": self._fp_slices,
            "fp_slice_rate": (self._fp_slices / self._n_empty_gt) if self._n_empty_gt else None,
            "tp": self._tp, "fp": self._fp, "fn": self._fn, "tn": self._tn,
        }
        out["micro_dice"] = micro["dice"]
        out["micro_iou"] = micro["iou"]
        out["micro_precision"] = micro["precision"]
        out["micro_recall"] = micro["recall"]
        out["micro_specificity"] = micro["specificity"]
        return out

    def per_case_summary(self) -> List[Dict[str, Any]]:
        """返回 per-case 汇总（供后续 lesion-wise / FP-per-case 分析）。

        Returns:
            每个 case 一条记录。
        """
        rows: List[Dict[str, Any]] = []
        for cid in sorted(self._per_case_tp):
            tp = self._per_case_tp.get(cid, 0)
            fp = self._per_case_fp.get(cid, 0)
            fn = self._per_case_fn.get(cid, 0)
            rows.append({
                "case_id": cid,
                "tp": tp, "fp": fp, "fn": fn,
                "gt_positive_slices": self._per_case_gt_slices.get(cid, 0),
                "dice": (2 * tp + self.eps) / (2 * tp + fp + fn + self.eps),
            })
        return rows

    def reset(self) -> None:
        """重置全部累计状态。"""
        self._tp = self._fp = self._fn = self._tn = 0
        self._n_slices = self._n_empty_gt = self._n_positive_gt = 0
        self._fp_slices = self._empty_gt_empty_pred = 0
        self._dice_pos.clear()
        self._dice_all.clear()
        self._iou_pos.clear()
        self._per_case_tp.clear()
        self._per_case_fp.clear()
        self._per_case_fn.clear()
        self._per_case_gt_slices.clear()
