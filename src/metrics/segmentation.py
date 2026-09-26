#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""分割评价指标（Phase 1 / Phase 2A）。

实现：Dice、IoU、Precision、Recall/Sensitivity、Specificity。

**空 GT 的处理是刻意的**：

自动病灶分割里，绝大部分 slice 的 GT 为空。若把 "GT 空 & 预测空" 的
Dice 记为 1.0 后直接混进平均，会严重高估性能。因此本模块同时报告：

* ``positive_slice_dice`` —— **只在 GT 非空的 slice 上**平均（论文主指标）
* ``all_slice_dice`` —— 所有 slice 平均，且**单独统计**其中有多少来自空 GT
* ``fp_slices`` —— GT 为空但预测非空的 slice 数（FP/case 的基础）

像素级 Precision/Recall/Specificity 采用全局微平均（先累计 TP/FP/FN/TN）。

**输入类型必须由调用方显式声明（Phase 2A 修正）**：早期版本用
``arr.min() < 0 or arr.max() > 1`` 猜测输入是 logits 还是概率；当 "logits
恰好都落在 [0,1]" 或 "概率被误当 logits" 时会静默产生错误结果。现在提供
三个显式入口：

* :meth:`SegmentationMetrics.update_logits` —— 输入 logits，内部 sigmoid 后按
  ``threshold`` 二值化；
* :meth:`SegmentationMetrics.update_probabilities` —— 输入概率，内部只做
  ``>= threshold``（越界直接报错，防止误传 logits）；
* :meth:`SegmentationMetrics.update_binary` —— 输入已是二值 mask。

旧的 ``update()`` 仍保留，但**必须显式传入 ``kind``**（``"logits"`` /
``"probabilities"`` / ``"binary"``），否则抛 ``TypeError``。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

__all__ = [
    "SliceConfusion",
    "compute_confusion",
    "metrics_from_confusion",
    "SegmentationMetrics",
    "UPDATABLE_KINDS",
    "threshold_to_bin",
]

UPDATABLE_KINDS = ("logits", "probabilities", "binary")


# --------------------------------------------------------------------------- #
# 数组工具（无任何"猜测输入类型"的逻辑）
# --------------------------------------------------------------------------- #
def _to_numpy(x: Any) -> np.ndarray:
    """把 torch.Tensor / ndarray / 序列统一转成 numpy 数组（保留 dtype）。"""
    if hasattr(x, "detach"):  # torch.Tensor
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _sigmoid(arr: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid。"""
    a = arr.astype(np.float64, copy=False)
    out = np.empty_like(a)
    pos = a >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-a[pos]))
    exp_a = np.exp(a[~pos])
    out[~pos] = exp_a / (1.0 + exp_a)
    return out.astype(np.float32, copy=False)


def _require_finite(arr: np.ndarray, name: str) -> None:
    """检查数组全部为有限值。

    Args:
        arr: 待检查数组。
        name: 报错时显示的名称。

    Raises:
        ValueError: 存在 NaN / Inf。
    """
    if arr.size and not np.isfinite(arr).all():
        n_bad = int((~np.isfinite(arr)).sum())
        raise ValueError(f"{name} 含 {n_bad} 个非有限值（NaN/Inf）")


def _as_binary(arr: np.ndarray, name: str) -> np.ndarray:
    """把数组转成 bool，并校验取值只能是 bool 或 {0, 1}。

    Args:
        arr: 输入数组。
        name: 报错时显示的名称。

    Returns:
        ``bool`` 数组。

    Raises:
        ValueError: 取值不是 bool 且不是 {0, 1} 子集。
    """
    if arr.dtype == bool:
        return arr
    uniq = np.unique(arr)
    if uniq.size and not np.all(np.isin(uniq, (0, 1))):
        raise ValueError(
            f"{name} 必须是 bool 或 {{0,1}}，实际取值示例={uniq[:5].tolist()}；"
            "若输入是 logits/概率请改用 update_logits()/update_probabilities()。"
        )
    return arr.astype(bool, copy=False)


# --------------------------------------------------------------------------- #
# 单切片混淆统计
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# 累计评估器
# --------------------------------------------------------------------------- #
@dataclass
class SegmentationMetrics:
    """跨 batch / 跨 slice 的累计评估器。

    输入类型必须显式声明（见模块 docstring）::

        m = SegmentationMetrics(threshold=0.5)
        for batch:
            m.update_logits(logits, gt_mask, case_ids=[...])       # logits
            # 或 m.update_probabilities(probs, gt_mask, ...)
            # 或 m.update_binary(pred_bool, gt_mask, ...)
        print(m.summary())

    Args:
        threshold: 概率二值化阈值（仅对 logits / probabilities 输入生效）。
        eps: 平滑项。
        range_tol: 概率范围检查容差。
    """

    threshold: float = 0.5
    eps: float = 1e-8
    range_tol: float = 1e-4

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
    # 显式输入类型 API（Phase 2A）
    # ------------------------------------------------------------------ #
    def update_logits(self, logits: Any, gt: Any,
                      case_ids: Optional[List[str]] = None) -> None:
        """用 logits 累计一个 batch（内部 ``sigmoid`` 后按阈值二值化）。

        Args:
            logits: ``[B, 1, H, W]`` / ``[B, H, W]`` / ``[H, W]`` logits。
            gt: 与 ``logits`` 同 shape 的 GT（内部按 ``!= 0`` 二值化）。
            case_ids: 长度 B 的 case 标识，用于 per-case 统计。

        Raises:
            ValueError: shape 不一致、维度不合法或含非有限值。
        """
        arr = _to_numpy(logits)
        _require_finite(arr, "logits")
        self._accumulate(_sigmoid(arr) >= float(self.threshold), gt, case_ids)

    def update_probabilities(self, probabilities: Any, gt: Any,
                             case_ids: Optional[List[str]] = None) -> None:
        """用概率累计一个 batch（不再做 sigmoid）。

        Args:
            probabilities: ``[B, 1, H, W]`` 概率，取值需落在 ``[0, 1]``。
            gt: 与输入同 shape 的 GT。
            case_ids: 长度 B 的 case 标识。

        Raises:
            ValueError: 概率越界（提示误传 logits）或含非有限值。
        """
        arr = _to_numpy(probabilities)
        _require_finite(arr, "probabilities")
        if arr.dtype != bool and arr.size:
            lo, hi = float(arr.min()), float(arr.max())
            if lo < -self.range_tol or hi > 1.0 + self.range_tol:
                raise ValueError(
                    f"update_probabilities() 收到取值范围 [{lo:.4f}, {hi:.4f}]，"
                    "超出 [0,1]；若输入是 logits 请改用 update_logits()。"
                )
        self._accumulate(arr >= float(self.threshold), gt, case_ids)

    def update_binary(self, binary: Any, gt: Any,
                      case_ids: Optional[List[str]] = None) -> None:
        """用已二值化的预测累计一个 batch。

        Args:
            binary: bool 或 ``{0,1}`` 数组。
            gt: 与输入同 shape 的 GT。
            case_ids: 长度 B 的 case 标识。

        Raises:
            ValueError: 取值不是 bool 也不是 ``{0,1}``。
        """
        self._accumulate(_as_binary(_to_numpy(binary), "binary"), gt, case_ids)

    def update(self, x: Any, gt: Any, case_ids: Optional[List[str]] = None,
               kind: Optional[str] = None) -> None:
        """兼容入口：**必须显式指定 ``kind``**，禁止按数值范围猜测类型。

        Args:
            x: logits / 概率 / 二值预测。
            gt: GT mask。
            case_ids: 长度 B 的 case 标识。
            kind: ``"logits"``、``"probabilities"`` 或 ``"binary"``。

        Raises:
            TypeError: 未提供 ``kind`` 或取值非法。
        """
        if kind is None:
            raise TypeError(
                "SegmentationMetrics.update() 必须显式指定 kind="
                "'logits' | 'probabilities' | 'binary'；"
                "推荐直接调用 update_logits() / update_probabilities() / update_binary()。"
            )
        if kind not in UPDATABLE_KINDS:
            raise TypeError(f"kind 必须是 {UPDATABLE_KINDS} 之一，收到 {kind!r}")
        if kind == "logits":
            self.update_logits(x, gt, case_ids)
        elif kind == "probabilities":
            self.update_probabilities(x, gt, case_ids)
        else:
            self.update_binary(x, gt, case_ids)

    # ------------------------------------------------------------------ #
    # 由混淆计数直接累计（供 threshold scan 的直方图实现复用）
    # ------------------------------------------------------------------ #
    def update_from_counts(self, tp: int, fp: int, fn: int, tn: int,
                           case_id: Optional[str] = None) -> None:
        """直接累计单个 slice 的 TP/FP/FN/TN。

        Args:
            tp: 真阳性像素数。
            fp: 假阳性像素数。
            fn: 假阴性像素数。
            tn: 真阴性像素数。
            case_id: 该 slice 所属 case；``None`` 时不更新 per-case 统计。
        """
        if min(tp, fp, fn, tn) < 0:
            raise ValueError(f"混淆计数不能为负: tp={tp} fp={fp} fn={fn} tn={tn}")
        self._consume(SliceConfusion(int(tp), int(fp), int(fn), int(tn)), case_id)

    # ------------------------------------------------------------------ #
    def _accumulate(self, pred_binary: np.ndarray, gt: Any,
                    case_ids: Optional[List[str]] = None) -> None:
        """把二值预测与 GT 展开成逐 slice 累计。

        Args:
            pred_binary: 已是 bool 的预测数组。
            gt: GT（任意数值，内部按 ``!= 0`` 二值化）。
            case_ids: 长度 B 的 case 标识。

        Raises:
            ValueError: shape 不一致或维度不合法。
        """
        p = self._normalize_to_bhw(np.asarray(pred_binary))
        g_arr = _to_numpy(gt)
        _require_finite(g_arr, "gt")
        g = self._normalize_to_bhw(g_arr != 0)
        if p.shape != g.shape:
            raise ValueError(f"shape 不一致: pred {p.shape} vs gt {g.shape}")

        for i in range(p.shape[0]):
            c = compute_confusion(p[i], g[i])
            cid = None
            if case_ids is not None and i < len(case_ids):
                cid = str(case_ids[i])
            self._consume(c, cid)

    def _consume(self, c: SliceConfusion, case_id: Optional[str]) -> None:
        """把单个 slice 的混淆统计并入累计状态。"""
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

        if case_id is not None:
            self._per_case_tp[case_id] = self._per_case_tp.get(case_id, 0) + c.tp
            self._per_case_fp[case_id] = self._per_case_fp.get(case_id, 0) + c.fp
            self._per_case_fn[case_id] = self._per_case_fn.get(case_id, 0) + c.fn
            if c.gt_positive:
                self._per_case_gt_slices[case_id] = \
                    self._per_case_gt_slices.get(case_id, 0) + 1

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
            "empty_gt_empty_pred_slices": self._empty_gt_empty_pred,
            "fp_slices": self._fp_slices,
            "fp_slice_rate": (self._fp_slices / self._n_empty_gt) if self._n_empty_gt else None,
            "tp": self._tp, "fp": self._fp, "fn": self._fn, "tn": self._tn,
            "threshold": float(self.threshold),
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


def threshold_to_bin(threshold: float, bin_width: float) -> int:
    """把概率阈值转换成直方图 bin 下标（``p >= threshold`` 等价于 ``idx >= 返回值``）。

    要求 ``bin_width > 0``；当 ``threshold`` 恰为 ``bin_width`` 的整数倍时结果精确。

    Args:
        threshold: 概率阈值，需落在 ``[0, 1]``。
        bin_width: 直方图 bin 宽度。

    Returns:
        起始 bin 下标。
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold 必须在 [0,1]，收到 {threshold}")
    ratio = float(threshold) / float(bin_width)
    k = int(math.floor(ratio + 1e-9))
    if k * bin_width < threshold - 1e-12:
        k += 1
    return k
