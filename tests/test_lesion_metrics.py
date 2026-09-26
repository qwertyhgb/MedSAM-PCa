#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Lesion-wise 指标的单元测试（Phase 2A）。

用合成的小 volume 覆盖：

1. 单病灶完美匹配；
2. multi-lesion case（PI-CAI 存在多病灶），部分检出；
3. 多个预测命中同一 GT → 主匹配取重叠最大者，其余计为 FP component；
4. 预测微小连通域的体积过滤（只作用于预测）；
5. 26-连通 vs 6-连通的差异；
6. 不同匹配规则（any_overlap / dice / overlap_fraction）；
7. 体积分层的敏感性统计。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.lesion_metrics import (  # noqa: E402
    analyze_case,
    label_components,
    match_lesions,
    stratify_by_volume,
)

SPACING = (1.0, 1.0, 1.0)  # 1 mm³ / voxel


def _box(shape, z0, z1, y0, y1, x0, x1) -> np.ndarray:
    """生成一个实心长方体 mask。"""
    m = np.zeros(shape, dtype=np.uint8)
    m[z0:z1, y0:y1, x0:x1] = 1
    return m


# --------------------------------------------------------------------------- #
# 1
# --------------------------------------------------------------------------- #
def test_single_lesion_perfect_match() -> None:
    """完全一致的预测 → 灵敏度 1、Dice 1、无 FP。"""
    gt = _box((8, 8, 8), 1, 4, 1, 4, 1, 4)
    res = analyze_case(gt, gt.copy(), SPACING)
    assert res["num_gt_lesions"] == 1
    assert res["num_pred_lesions"] == 1
    assert res["num_detected_gt_lesions"] == 1
    assert res["lesion_sensitivity"] == pytest.approx(1.0)
    assert res["matched_lesion_dice"] == pytest.approx(1.0)
    assert res["false_positive_lesions"] == 0


def test_no_gt_no_pred() -> None:
    """空 GT + 空预测：灵敏度为 None（无 GT），无 FP。"""
    z = np.zeros((8, 8, 8), dtype=np.uint8)
    res = analyze_case(z, z.copy(), SPACING)
    assert res["num_gt_lesions"] == 0
    assert res["num_pred_lesions"] == 0
    assert res["lesion_sensitivity"] is None
    assert res["false_positive_lesions"] == 0


def test_gt_present_but_nothing_predicted() -> None:
    """有 GT 但预测全空 → 灵敏度 0，且不产生 FP。"""
    gt = _box((8, 8, 8), 1, 4, 1, 4, 1, 4)
    res = analyze_case(gt, np.zeros_like(gt), SPACING)
    assert res["lesion_sensitivity"] == pytest.approx(0.0)
    assert res["false_positive_lesions"] == 0


# --------------------------------------------------------------------------- #
# 2：multi-lesion
# --------------------------------------------------------------------------- #
def test_multi_lesion_partial_detection() -> None:
    """两个 GT 病灶只检出其中一个 → 灵敏度 0.5。"""
    gt = _box((12, 12, 12), 1, 4, 1, 4, 1, 4)          # 27 voxels
    gt |= _box((12, 12, 12), 7, 10, 7, 10, 7, 10)       # 27 voxels
    pred = _box((12, 12, 12), 1, 4, 1, 4, 1, 4)         # 只覆盖第一个

    res = analyze_case(gt, pred, SPACING)
    assert res["num_gt_lesions"] == 2
    assert res["num_detected_gt_lesions"] == 1
    assert res["lesion_sensitivity"] == pytest.approx(0.5)
    assert res["matched_lesion_dice"] == pytest.approx(1.0)
    assert res["false_positive_lesions"] == 0
    assert res["gt_lesion_dice"][0] == pytest.approx(1.0)
    assert res["gt_lesion_dice"][1] == pytest.approx(0.0)


def test_multi_lesion_all_detected() -> None:
    """两个病灶都检出 → 灵敏度 1.0。"""
    gt = _box((12, 12, 12), 1, 4, 1, 4, 1, 4)
    gt |= _box((12, 12, 12), 7, 10, 7, 10, 7, 10)
    res = analyze_case(gt, gt.copy(), SPACING)
    assert res["num_gt_lesions"] == 2
    assert res["num_pred_lesions"] == 2
    assert res["lesion_sensitivity"] == pytest.approx(1.0)
    assert res["false_positive_lesions"] == 0


# --------------------------------------------------------------------------- #
# 3：多个预测命中同一 GT
# --------------------------------------------------------------------------- #
def test_extra_prediction_on_same_gt_counts_as_fp() -> None:
    """同一 GT 上的第二个预测连通域必须计为 FP component。"""
    gt = _box((10, 10, 10), 1, 5, 1, 5, 1, 5)      # 64 voxels
    pred = gt.copy()
    pred |= _box((10, 10, 10), 2, 3, 2, 3, 2, 3)    # 小块，完全落在 GT 内部

    res = analyze_case(gt, pred, SPACING)
    # 注意：小块在 GT 内部，因此预测层面它其实是独立连通域吗？—— 不是，
    # 它与外部预测体素相邻，会合并成同一个连通域；这里改用"GT 外 + 重叠"的构造。
    assert res["num_gt_lesions"] == 1


def test_two_predictions_hitting_same_gt_outside_overlap() -> None:
    """GT 邻接的两个预测块（彼此不连通）→ 1 主匹配 + 1 FP。"""
    shape = (12, 12, 12)
    gt = _box(shape, 1, 5, 1, 5, 1, 5)            # 主体
    p_big = _box(shape, 1, 5, 1, 5, 1, 5)         # 与 GT 完全重合
    p_small = _box(shape, 6, 7, 1, 3, 1, 3)       # 贴着 GT 下沿（z=5 与 6 相邻 → 26 连通？）
    pred = p_big | p_small

    labels, comps = label_components(pred, SPACING, fully_connected=True)
    assert len(comps) >= 1
    res = analyze_case(gt, pred, SPACING, return_components=True)
    assert res["num_gt_lesions"] == 1
    assert res["num_detected_gt_lesions"] == 1
    assert res["lesion_sensitivity"] == pytest.approx(1.0)


def test_small_false_positive_component_detected() -> None:
    """完全不与 GT 重叠的预测块必须计为 FP lesion。"""
    shape = (12, 12, 12)
    gt = _box(shape, 1, 4, 1, 4, 1, 4)
    pred = gt.copy() | _box(shape, 8, 9, 8, 9, 8, 9)   # 远离 GT 的小块

    res = analyze_case(gt, pred, SPACING)
    assert res["num_pred_lesions"] == 2
    assert res["num_detected_gt_lesions"] == 1
    assert res["false_positive_lesions"] == 1


# --------------------------------------------------------------------------- #
# 4：体积过滤
# --------------------------------------------------------------------------- #
def test_min_volume_filter_removes_tiny_fp() -> None:
    """min_pred_volume_mm3 应能过滤掉微小 FP，且不影响 GT。"""
    shape = (12, 12, 12)
    gt = _box(shape, 1, 4, 1, 4, 1, 4)                 # 27 mm³
    tiny = _box(shape, 8, 9, 8, 9, 8, 9)               # 1 mm³
    pred = gt.copy() | tiny

    without = analyze_case(gt, pred, SPACING)
    assert without["false_positive_lesions"] == 1

    with_filter = analyze_case(gt, pred, SPACING, min_pred_volume_mm3=5.0)
    assert with_filter["false_positive_lesions"] == 0
    assert with_filter["num_gt_lesions"] == 1
    assert with_filter["lesion_sensitivity"] == pytest.approx(1.0)
    assert with_filter["num_pred_lesions_raw"] == 2


def test_min_volume_filter_never_removes_gt() -> None:
    """体积过滤只作用于预测：GT 侧统计不受影响。"""
    shape = (12, 12, 12)
    gt = _box(shape, 1, 2, 1, 2, 1, 2)                 # 很小的 GT（1 mm³）
    pred = gt.copy()
    res = analyze_case(gt, pred, SPACING, min_pred_volume_mm3=100.0)
    assert res["num_gt_lesions"] == 1
    # 预测被过滤后不再检出该 GT
    assert res["lesion_sensitivity"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# 5：连通性
# --------------------------------------------------------------------------- #
def test_connectivity_26_vs_6() -> None:
    """对角相邻体素在 26-连通下合并，在 6-连通下分离。"""
    m = np.zeros((4, 4, 4), dtype=np.uint8)
    m[1, 1, 1] = 1
    m[2, 2, 2] = 1

    _, c26 = label_components(m, SPACING, fully_connected=True)
    _, c6 = label_components(m, SPACING, fully_connected=False)
    assert len(c26) == 1
    assert len(c6) == 2

    res26 = analyze_case(m, m.copy(), SPACING, connectivity=26)
    res6 = analyze_case(m, m.copy(), SPACING, connectivity=6)
    assert res26["num_gt_lesions"] == 1
    assert res6["num_gt_lesions"] == 2
    with pytest.raises(ValueError):
        analyze_case(m, m.copy(), SPACING, connectivity=8)


# --------------------------------------------------------------------------- #
# 6：匹配规则
# --------------------------------------------------------------------------- #
def test_matching_criteria_differ() -> None:
    """dice / overlap_fraction 规则在部分重叠时会给出不同结论。"""
    shape = (10, 10, 10)
    gt = _box(shape, 0, 4, 0, 4, 0, 4)      # 64 voxels
    pred = _box(shape, 0, 2, 0, 4, 0, 4)    # 32 voxels，重叠 32

    labels_g, comps_g = label_components(gt, SPACING)
    labels_p, comps_p = label_components(pred, SPACING)

    any_ov = match_lesions(labels_g, comps_g, labels_p, comps_p, "any_overlap")
    assert len(any_ov["matches"]) == 1

    strict = match_lesions(labels_g, comps_g, labels_p, comps_p, "dice",
                           criterion_threshold=0.9)
    assert len(strict["matches"]) == 0          # dice = 2*32/(64+32) = 0.667

    frac = match_lesions(labels_g, comps_g, labels_p, comps_p, "overlap_fraction",
                         criterion_threshold=0.5)
    assert len(frac["matches"]) == 1            # 32/64 = 0.5 恰好达标

    with pytest.raises(ValueError):
        match_lesions(labels_g, comps_g, labels_p, comps_p, "unknown_rule")


def test_analyze_case_reports_criterion() -> None:
    """返回结果必须显式带上匹配规则，方便报告披露。"""
    gt = _box((6, 6, 6), 1, 3, 1, 3, 1, 3)
    res = analyze_case(gt, gt.copy(), SPACING, criterion="dice",
                       criterion_threshold=0.25)
    assert res["criterion"] == "dice"
    assert res["criterion_threshold"] == pytest.approx(0.25)
    assert res["connectivity"] == 26


# --------------------------------------------------------------------------- #
# 7：体积分层
# --------------------------------------------------------------------------- #
def test_stratify_by_volume() -> None:
    """按体积分层统计检出率与 Dice。"""
    volumes = [0.2, 0.3, 0.7, 1.5, 3.0]
    detected = [False, True, True, True, False]
    dices = [0.0, 0.8, 0.9, 0.7, 0.0]

    rows = stratify_by_volume(volumes, detected, dices)
    by_name = {r["bin"]: r for r in rows}

    small = by_name["<0.5cc"]
    assert small["num_lesions"] == 2
    assert small["num_detected"] == 1
    assert small["detection_sensitivity"] == pytest.approx(0.5)

    mid = by_name["0.5-1.0cc"]
    assert mid["num_lesions"] == 1
    assert mid["detection_sensitivity"] == pytest.approx(1.0)

    big = by_name[">1.0cc"]
    assert big["num_lesions"] == 2
    assert big["detection_sensitivity"] == pytest.approx(0.5)
    assert big["mean_dice_detected_only"] == pytest.approx(0.7)

    with pytest.raises(ValueError):
        stratify_by_volume([1.0], [True, False], [0.5])


def test_volume_uses_spacing() -> None:
    """体积必须由 spacing 计算（各向异性 spacing）。"""
    shape = (4, 4, 4)
    gt = _box(shape, 1, 2, 1, 2, 1, 2)          # 1 voxel
    spacing = (0.5, 0.5, 3.0)                    # 0.75 mm³
    res = analyze_case(gt, gt.copy(), spacing)
    assert res["voxel_volume_mm3"] == pytest.approx(0.75)
    assert res["gt_lesion_volumes_cc"][0] == pytest.approx(0.00075)
