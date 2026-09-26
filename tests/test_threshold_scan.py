#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ThresholdScanMetrics（概率直方图多阈值累计）与精确实现的一致性测试。

核心断言：对同一批概率与 GT，直方图法给出的每个阈值指标必须与
"逐阈值重新做布尔比较" 的 :class:`SegmentationMetrics` **完全一致**
（阈值取 5e-4 的整数倍时，``p >= t`` 与 ``idx >= k`` 严格等价）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.segmentation import SegmentationMetrics, threshold_to_bin  # noqa: E402
from src.metrics.threshold_scan import DEFAULT_THRESHOLDS, ThresholdScanMetrics  # noqa: E402


def _make_data(seed: int = 0, batch: int = 3, size: int = 32):
    """构造带少量前景的随机概率与 GT。"""
    rng = np.random.default_rng(seed)
    probs = rng.random((batch, 1, size, size), dtype=np.float32)
    gt = (rng.random((batch, 1, size, size)) > 0.85).astype(np.uint8)
    # 让部分概率贴近阈值，制造边界情形
    probs[0, 0, :4, :4] = 0.5
    probs[1, 0, :4, :4] = 0.0
    probs[2, 0, :4, :4] = 1.0
    gt[0, 0, :2, :2] = 1
    gt[1, 0, :8, :8] = 1
    case_ids = [f"case_{i}" for i in range(batch)]
    return probs, gt, case_ids


def _exact_rows(thresholds, probs, gt, case_ids):
    """逐阈值精确实现，返回与 ThresholdScanMetrics.rows() 同构的表。"""
    rows = []
    for t in thresholds:
        m = SegmentationMetrics(threshold=float(t))
        m.update_probabilities(probs, gt, case_ids=list(case_ids))
        s = m.summary()
        rows.append({
            "threshold": float(t),
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
    return rows


def test_histogram_matches_exact_implementation() -> None:
    """默认阈值网格下，直方图法与精确法必须逐字段一致。"""
    probs, gt, case_ids = _make_data()
    scan = ThresholdScanMetrics(DEFAULT_THRESHOLDS)
    scan.update_probabilities(probs, gt, case_ids=list(case_ids))

    got = scan.rows()
    want = _exact_rows(DEFAULT_THRESHOLDS, probs, gt, case_ids)

    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert g["threshold"] == pytest.approx(w["threshold"])
        for key in ("positive_slice_dice", "positive_slice_iou", "all_slice_dice",
                    "micro_dice", "micro_iou", "micro_precision", "micro_recall",
                    "micro_specificity", "fp_slice_rate", "fp_slices",
                    "num_slices", "num_positive_gt_slices", "num_empty_gt_slices"):
            assert g[key] == pytest.approx(w[key], abs=0.0), \
                f"阈值 {g['threshold']} 的 {key} 不一致: {g[key]} vs {w[key]}"


def test_custom_thresholds_match_exact() -> None:
    """任意（bin 整数倍）阈值集合也必须一致。"""
    thresholds = [0.05, 0.175, 0.3, 0.3335, 0.5, 0.75, 0.9995]
    probs, gt, case_ids = _make_data(seed=7)
    scan = ThresholdScanMetrics(thresholds)
    scan.update_probabilities(probs, gt, case_ids=list(case_ids))

    for g, w in zip(scan.rows(), _exact_rows(thresholds, probs, gt, case_ids)):
        assert g["positive_slice_dice"] == pytest.approx(w["positive_slice_dice"], abs=0.0)
        assert g["micro_dice"] == pytest.approx(w["micro_dice"], abs=0.0)


def test_threshold_to_bin_boundaries() -> None:
    """阈值 -> bin 下标换算的边界行为。"""
    bw = 5e-4
    assert threshold_to_bin(0.05, bw) == 100
    assert threshold_to_bin(0.5, bw) == 1000
    assert threshold_to_bin(0.95, bw) == 1900
    # 非整倍：必须向上取整（保证 p >= t 的语义）
    assert threshold_to_bin(0.0501, bw) == 101
    assert threshold_to_bin(0.0001, bw) == 1
    with pytest.raises(ValueError):
        threshold_to_bin(1.5, bw)


def test_best_by_returns_argmax_threshold() -> None:
    """best_by 必须返回指定指标最大的阈值行。"""
    probs, gt, case_ids = _make_data(seed=3)
    scan = ThresholdScanMetrics([0.2, 0.4, 0.6])
    scan.update_probabilities(probs, gt, case_ids=list(case_ids))
    rows = scan.rows()
    best = scan.best_by("micro_precision")
    assert best["threshold"] == max(rows, key=lambda r: r["micro_precision"])["threshold"]


def test_rejects_out_of_range_probabilities() -> None:
    """概率越界（误传 logits）必须报错。"""
    scan = ThresholdScanMetrics([0.5])
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        scan.update_probabilities(np.array([[[[1.7]]]], dtype=np.float32),
                                  np.ones((1, 1, 1, 1), dtype=np.uint8))
    with pytest.raises(ValueError, match="NaN"):
        scan.update_probabilities(np.array([[[[np.nan]]]], dtype=np.float32),
                                  np.ones((1, 1, 1, 1), dtype=np.uint8))


def test_empty_gt_slice_counted_as_fp_when_predicted() -> None:
    """全负 GT + 全高概率预测 → fp_slice_rate=1，positive_slice_dice 为 None。"""
    scan = ThresholdScanMetrics([0.5])
    scan.update_probabilities(np.ones((1, 1, 4, 4), dtype=np.float32),
                              np.zeros((1, 1, 4, 4), dtype=np.uint8))
    row = scan.rows()[0]
    assert row["fp_slice_rate"] == pytest.approx(1.0)
    assert row["positive_slice_dice"] is None
    assert row["all_slice_dice"] == pytest.approx(0.0, abs=1e-6)


def test_multi_batch_accumulation() -> None:
    """分多个 batch 累计的结果必须与一次累计一致。"""
    probs, gt, case_ids = _make_data(seed=11)
    once = ThresholdScanMetrics([0.3, 0.6])
    once.update_probabilities(probs, gt, case_ids=list(case_ids))

    split = ThresholdScanMetrics([0.3, 0.6])
    for i in range(probs.shape[0]):
        split.update_probabilities(probs[i:i + 1], gt[i:i + 1],
                                   case_ids=[case_ids[i]])

    for a, b in zip(once.rows(), split.rows()):
        assert a["micro_dice"] == pytest.approx(b["micro_dice"], abs=0.0)
        assert a["fp_slices"] == b["fp_slices"]
