#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Lesion-wise（病灶级）评价指标（Phase 2A）。

设计要点
--------
* 连通域使用 **3D connected components**（默认 26-连通，SimpleITK
  ``ConnectedComponent(fullyConnected=True)``）；
* GT 的每个连通域视为一个 lesion，预测的每个连通域视为一个 predicted lesion；
* **支持 multi-lesion case**：不做 "只保留最大连通域" 这类会丢标注的操作；
* 匹配规则**可配置且必须显式报告**（见 :data:`MATCHING_CRITERIA`）：

  - ``"any_overlap"``：预测连通域与 GT 连通域有任意体素重叠即算命中；
  - ``"dice"``：按体素 Dice 判定（``>= criterion_threshold``）；
  - ``"iou"``：按体素 IoU 判定；
  - ``"overlap_fraction"``：``overlap / gt_voxels >= criterion_threshold``
    （即 GT 被覆盖的比例）。

* 一个 GT 可能被多个预测连通域命中：**取重叠体素最多者作为主匹配**，
  其余预测连通域计为 **FP component**（一个病灶只允许一次检出），
  同时在 ``extra_matched_preds`` 中保留明细，避免隐藏匹配过程；
* 存在 ``min_pred_volume_mm3`` 用于评估时的微小连通域过滤分析，
  该过滤**只作用于预测**，绝不作用于 GT。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk

__all__ = [
    "MATCHING_CRITERIA",
    "ComponentStats",
    "label_components",
    "match_lesions",
    "analyze_case",
    "GT_VOLUME_BINS_CC",
    "stratify_by_volume",
]

#: 支持的匹配规则。
MATCHING_CRITERIA = ("any_overlap", "dice", "iou", "overlap_fraction")

#: 默认的病灶体积分层（单位 cc = cm³）；仅为探索性分析分层，不是临床标准。
GT_VOLUME_BINS_CC: Tuple[Tuple[str, float, float], ...] = (
    ("<0.5cc", 0.0, 0.5),
    ("0.5-1.0cc", 0.5, 1.0),
    (">1.0cc", 1.0, float("inf")),
)


@dataclass
class ComponentStats:
    """单个 3D 连通域的统计。"""

    label: int
    voxels: int
    volume_mm3: float
    bbox: Optional[Tuple[int, int, int, int, int, int]] = None

    @property
    def volume_cc(self) -> float:
        """体积（cm³，1 cc = 1000 mm³）。"""
        return self.volume_mm3 / 1000.0


def label_components(mask: np.ndarray, spacing: Sequence[float],
                     fully_connected: bool = True) -> Tuple[np.ndarray, List[ComponentStats]]:
    """对 3D 二值 mask 做连通域标记。

    Args:
        mask: ``[Z, H, W]`` 二值数组（非零视为前景）。
        spacing: 体素物理尺寸 ``(sx, sy, sz)``（mm）。
        fully_connected: ``True`` 使用 26-连通，``False`` 使用 6-连通。

    Returns:
        ``(labels, components)``；``labels`` 为 ``int32`` 数组（0 为背景），
        ``components`` 按标签顺序排列。

    Raises:
        ValueError: mask 不是 3D 或 spacing 长度不为 3。
    """
    arr = np.asarray(mask)
    if arr.ndim != 3:
        raise ValueError(f"mask 必须是 3D，收到 {arr.shape}")
    if len(spacing) != 3:
        raise ValueError(f"spacing 必须是 3 个元素，收到 {spacing}")

    binary = (arr != 0).astype(np.uint8)
    if binary.max() == 0:
        return np.zeros(binary.shape, dtype=np.int32), []

    img = sitk.GetImageFromArray(binary)
    img.SetSpacing(tuple(float(s) for s in spacing))
    cc = sitk.ConnectedComponent(img, bool(fully_connected))
    labels = sitk.GetArrayFromImage(cc).astype(np.int32)

    voxel_volume = float(spacing[0]) * float(spacing[1]) * float(spacing[2])
    components: List[ComponentStats] = []
    for label in range(1, int(labels.max()) + 1):
        idx = np.argwhere(labels == label)
        components.append(ComponentStats(
            label=label,
            voxels=int(idx.shape[0]),
            volume_mm3=float(idx.shape[0]) * voxel_volume,
            bbox=(int(idx[:, 0].min()), int(idx[:, 0].max()),
                  int(idx[:, 1].min()), int(idx[:, 1].max()),
                  int(idx[:, 2].min()), int(idx[:, 2].max())),
        ))
    return labels, components


def _pair_overlaps(gt_labels: np.ndarray, pred_labels: np.ndarray,
                   n_pred: int) -> Dict[Tuple[int, int], int]:
    """统计每对 (gt_label, pred_label) 的重叠体素数量。

    Args:
        gt_labels: GT 连通域标记。
        pred_labels: 预测连通域标记。
        n_pred: 预测连通域总数。

    Returns:
        ``{(gt_label, pred_label): overlap_voxels}``，只包含 overlap > 0 的对。
    """
    sel = (gt_labels > 0) & (pred_labels > 0)
    if not sel.any():
        return {}
    g = gt_labels[sel].astype(np.int64)
    p = pred_labels[sel].astype(np.int64)
    keys = g * (int(n_pred) + 1) + p
    uniq, counts = np.unique(keys, return_counts=True)
    out: Dict[Tuple[int, int], int] = {}
    for k, c in zip(uniq.tolist(), counts.tolist()):
        out[(k // (n_pred + 1), k % (n_pred + 1))] = int(c)
    return out


def _satisfies(overlap: int, gt_voxels: int, pred_voxels: int,
               criterion: str, threshold: float) -> bool:
    """判断一对连通域是否满足给定匹配规则。"""
    if criterion == "any_overlap":
        return overlap > 0
    if criterion == "dice":
        denom = gt_voxels + pred_voxels
        return denom > 0 and (2.0 * overlap / denom) >= threshold
    if criterion == "iou":
        union = gt_voxels + pred_voxels - overlap
        return union > 0 and (overlap / union) >= threshold
    if criterion == "overlap_fraction":
        return gt_voxels > 0 and (overlap / gt_voxels) >= threshold
    raise ValueError(f"未知匹配规则 {criterion!r}，可选 {MATCHING_CRITERIA}")


def match_lesions(gt_labels: np.ndarray, gt_components: List[ComponentStats],
                  pred_labels: np.ndarray, pred_components: List[ComponentStats],
                  criterion: str = "any_overlap",
                  criterion_threshold: float = 0.1) -> Dict[str, Any]:
    """按给定规则匹配 GT 与预测连通域。

    Args:
        gt_labels: GT 连通域标记数组。
        gt_components: GT 连通域统计（长度需与标签数一致）。
        pred_labels: 预测连通域标记数组。
        pred_components: 预测连通域统计。
        criterion: 匹配规则，见 :data:`MATCHING_CRITERIA`。
        criterion_threshold: 规则阈值（``any_overlap`` 忽略）。

    Returns:
        含 ``matches``（GT label -> 主匹配信息）、``fp_pred_labels``、
        ``extra_matched_preds`` 的字典。

    Raises:
        ValueError: 匹配规则非法。
    """
    if criterion not in MATCHING_CRITERIA:
        raise ValueError(f"未知匹配规则 {criterion!r}，可选 {MATCHING_CRITERIA}")

    gt_by_label = {c.label: c for c in gt_components}
    pred_by_label = {c.label: c for c in pred_components}
    overlaps = _pair_overlaps(gt_labels, pred_labels, len(pred_components))

    # GT label -> [(pred_label, overlap, dice, iou, frac), ...]（已按规则筛选）
    candidates: Dict[int, List[Tuple[int, int, float, float, float]]] = {}
    for (g_lab, p_lab), ov in overlaps.items():
        g_stat = gt_by_label.get(g_lab)
        p_stat = pred_by_label.get(p_lab)
        if g_stat is None or p_stat is None:
            continue
        if not _satisfies(ov, g_stat.voxels, p_stat.voxels, criterion, criterion_threshold):
            continue
        dice = 2.0 * ov / (g_stat.voxels + p_stat.voxels)
        union = g_stat.voxels + p_stat.voxels - ov
        iou = ov / union if union else 0.0
        frac = ov / g_stat.voxels if g_stat.voxels else 0.0
        candidates.setdefault(g_lab, []).append((p_lab, ov, dice, iou, frac))

    matches: Dict[int, Dict[str, Any]] = {}
    matched_pred_labels: set = set()
    extra_matched: List[Dict[str, Any]] = []
    for g_lab, cand in candidates.items():
        # 主匹配：重叠体素最多；并列时取 dice 更大者
        cand_sorted = sorted(cand, key=lambda t: (t[1], t[2]), reverse=True)
        p_lab, ov, dice, iou, frac = cand_sorted[0]
        matches[g_lab] = {
            "gt_label": g_lab,
            "pred_label": p_lab,
            "overlap_voxels": ov,
            "gt_voxels": gt_by_label[g_lab].voxels,
            "pred_voxels": pred_by_label[p_lab].voxels,
            "dice": dice,
            "iou": iou,
            "overlap_fraction_of_gt": frac,
        }
        matched_pred_labels.add(p_lab)
        # 同一个 GT 上的其余预测连通域：计为 FP component（一个病灶只允许一次检出），
        # 但保留明细以便报告透明。
        for p_extra, ov_e, dice_e, _iou_e, _frac_e in cand_sorted[1:]:
            extra_matched.append({
                "gt_label": g_lab,
                "pred_label": p_extra,
                "overlap_voxels": ov_e,
                "dice": dice_e,
                "note": "extra_pred_hitting_same_gt_counted_as_fp",
            })

    fp_pred_labels = [c.label for c in pred_components if c.label not in matched_pred_labels]
    return {
        "matches": matches,
        "fp_pred_labels": fp_pred_labels,
        "extra_matched_preds": extra_matched,
        "criterion": criterion,
        "criterion_threshold": float(criterion_threshold),
    }


def analyze_case(gt: np.ndarray, pred: np.ndarray, spacing: Sequence[float], *,
                 connectivity: int = 26,
                 criterion: str = "any_overlap",
                 criterion_threshold: float = 0.1,
                 min_pred_volume_mm3: float = 0.0,
                 return_components: bool = False) -> Dict[str, Any]:
    """对单个 case 做 lesion-wise 分析。

    Args:
        gt: ``[Z, H, W]`` GT 二值 mask。
        pred: ``[Z, H, W]`` 预测二值 mask。
        spacing: 体素物理尺寸 ``(sx, sy, sz)``（mm）。
        connectivity: 26 或 6。
        criterion: 匹配规则。
        criterion_threshold: 规则阈值。
        min_pred_volume_mm3: 预测连通域的最小体积过滤（仅作用于预测）。
        return_components: 是否在返回值中附带预测连通域明细。

    Returns:
        单 case 的 lesion 级指标与（可选）明细。

    Raises:
        ValueError: shape 不一致或 connectivity 非法。
    """
    g = np.asarray(gt) != 0
    p = np.asarray(pred) != 0
    if g.shape != p.shape:
        raise ValueError(f"gt/pred shape 不一致: {g.shape} vs {p.shape}")
    if connectivity not in (6, 26):
        raise ValueError(f"connectivity 只能是 6 或 26，收到 {connectivity}")

    fully = connectivity == 26
    gt_labels, gt_components = label_components(g, spacing, fully)
    pred_labels_all, pred_components_all = label_components(p, spacing, fully)

    voxel_volume = float(spacing[0]) * float(spacing[1]) * float(spacing[2])
    if min_pred_volume_mm3 > 0:
        keep = {c.label for c in pred_components_all if c.volume_mm3 >= min_pred_volume_mm3}
        if len(keep) != len(pred_components_all):
            mask = np.isin(pred_labels_all, list(keep)) if keep else np.zeros_like(pred_labels_all, bool)
            pred_labels, pred_components = label_components(mask, spacing, fully)
        else:
            pred_labels, pred_components = pred_labels_all, pred_components_all
    else:
        pred_labels, pred_components = pred_labels_all, pred_components_all

    matched = match_lesions(gt_labels, gt_components, pred_labels, pred_components,
                            criterion=criterion, criterion_threshold=criterion_threshold)

    n_gt = len(gt_components)
    n_pred = len(pred_components)
    n_detected = len(matched["matches"])
    dice_list = [m["dice"] for m in matched["matches"].values()]
    gt_dice_list = [matched["matches"][c.label]["dice"] if c.label in matched["matches"] else 0.0
                    for c in gt_components]
    gt_volumes_cc = [c.volume_cc for c in gt_components]

    result: Dict[str, Any] = {
        "num_gt_lesions": n_gt,
        "num_pred_lesions": n_pred,
        "num_pred_lesions_raw": len(pred_components_all),
        "num_detected_gt_lesions": n_detected,
        "lesion_sensitivity": (n_detected / n_gt) if n_gt else None,
        "false_positive_lesions": int(len(matched["fp_pred_labels"])),
        "matched_lesion_dice": (float(np.mean(dice_list)) if dice_list else None),
        "gt_lesion_dice": gt_dice_list,
        "gt_lesion_volumes_cc": gt_volumes_cc,
        "extra_preds_hitting_same_gt": len(matched["extra_matched_preds"]),
        "criterion": criterion,
        "criterion_threshold": float(criterion_threshold),
        "connectivity": connectivity,
        "min_pred_volume_mm3": float(min_pred_volume_mm3),
        "voxel_volume_mm3": voxel_volume,
        "has_gt_lesion": n_gt > 0,
    }
    if return_components:
        result["pred_components"] = [c.__dict__ for c in pred_components]
        result["gt_components"] = [c.__dict__ for c in gt_components]
    return result


def stratify_by_volume(gt_lesion_volumes_cc: Sequence[float],
                       detected: Sequence[bool],
                       dices: Sequence[float],
                       bins: Sequence[Tuple[str, float, float]] = GT_VOLUME_BINS_CC,
                       ) -> List[Dict[str, Any]]:
    """按 GT 病灶体积分层汇总检测率与 Dice。

    Args:
        gt_lesion_volumes_cc: 每个 GT 病灶的体积（cc）。
        detected: 每个 GT 病灶是否被检出（与上者等长）。
        dices: 每个 GT 病灶的匹配 Dice（未检出为 0）。
        bins: ``(名称, 下界 cc, 上界 cc)`` 列表，上界为 ``inf`` 表示无上限。

    Returns:
        每层一行的汇总。

    Raises:
        ValueError: 输入长度不一致。
    """
    if not (len(gt_lesion_volumes_cc) == len(detected) == len(dices)):
        raise ValueError("体积、检出标记与 Dice 列表长度必须一致")

    rows: List[Dict[str, Any]] = []
    for name, lo, hi in bins:
        idx = [i for i, v in enumerate(gt_lesion_volumes_cc) if lo <= v < hi]
        n = len(idx)
        n_det = sum(1 for i in idx if detected[i])
        mean_dice = float(np.mean([dices[i] for i in idx])) if n else None
        detected_dice = [dices[i] for i in idx if detected[i]]
        rows.append({
            "bin": name,
            "min_cc": lo,
            "max_cc": hi,
            "num_lesions": n,
            "num_detected": n_det,
            "detection_sensitivity": (n_det / n) if n else None,
            "mean_dice_all_lesions": mean_dice,
            "mean_dice_detected_only": (float(np.mean(detected_dice))
                                        if detected_dice else None),
        })
    return rows
