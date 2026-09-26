#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SegmentationMetrics 显式输入类型 API 的单元测试（Phase 2A）。

覆盖点：

1. ``logits=0.2`` → ``sigmoid=0.5498`` → threshold 0.5 应判 positive；
2. ``logits=-0.2`` → ``sigmoid=0.4502`` → 应判 negative；
3. ``probability=0.2`` → 应判 negative；
4. 概率接口不得接受 logits（越界必须报错），logits 接口不得被误用为概率；
5. 禁止按数值范围猜类型：取值落在 ``[0,1]`` 的 logits 也必须走 sigmoid；
6. ``update()`` 不显式传 ``kind`` 必须抛 ``TypeError``；
7. 非有限值与非法二值输入必须报错。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.segmentation import SegmentationMetrics  # noqa: E402


def _gt_ones(shape=(1, 1, 4, 4)) -> np.ndarray:
    """全前景 GT。"""
    return np.ones(shape, dtype=np.uint8)


def _dice(m: SegmentationMetrics) -> float:
    """取当前累计的 positive-slice Dice。"""
    value = m.summary()["positive_slice_dice"]
    assert value is not None
    return float(value)


# --------------------------------------------------------------------------- #
# 1 & 2：logits 语义
# --------------------------------------------------------------------------- #
def test_logits_0_2_predicts_positive_at_threshold_0_5() -> None:
    """logits=0.2 的 sigmoid 为 0.5498 > 0.5，应完全命中全前景 GT。"""
    assert math.isclose(1.0 / (1.0 + math.exp(-0.2)), 0.549834, rel_tol=1e-6)

    m = SegmentationMetrics(threshold=0.5)
    m.update_logits(torch.full((1, 1, 4, 4), 0.2), _gt_ones())
    assert _dice(m) == pytest.approx(1.0, abs=1e-6)
    assert m.summary()["tp"] == 16


def test_logits_minus_0_2_predicts_negative_at_threshold_0_5() -> None:
    """logits=-0.2 的 sigmoid 为 0.4502 < 0.5，全前景 GT 下 Dice 应为 0。"""
    assert math.isclose(1.0 / (1.0 + math.exp(0.2)), 0.450166, rel_tol=1e-6)

    m = SegmentationMetrics(threshold=0.5)
    m.update_logits(torch.full((1, 1, 4, 4), -0.2), _gt_ones())
    assert _dice(m) == pytest.approx(0.0, abs=1e-6)
    assert m.summary()["tp"] == 0
    assert m.summary()["fn"] == 16


# --------------------------------------------------------------------------- #
# 3：概率语义
# --------------------------------------------------------------------------- #
def test_probability_0_2_predicts_negative() -> None:
    """概率 0.2 < 0.5，应判 negative。"""
    m = SegmentationMetrics(threshold=0.5)
    m.update_probabilities(np.full((1, 1, 4, 4), 0.2, dtype=np.float32), _gt_ones())
    assert _dice(m) == pytest.approx(0.0, abs=1e-6)


def test_probability_0_8_predicts_positive() -> None:
    """概率 0.8 > 0.5，应判 positive。"""
    m = SegmentationMetrics(threshold=0.5)
    m.update_probabilities(np.full((1, 1, 4, 4), 0.8, dtype=np.float32), _gt_ones())
    assert _dice(m) == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# 5：禁止按数值范围猜类型
# --------------------------------------------------------------------------- #
def test_in_range_logits_are_not_treated_as_probabilities() -> None:
    """logits 全部落在 (0,1) 时也必须走 sigmoid，而非直接阈值化。

    旧实现用 ``arr.min() < 0 or arr.max() > 1`` 判断 logits，此处 0.2 会被
    当成概率 → 预测 negative。新实现必须把它当 logits → sigmoid(0.2)=0.5498
    → positive。
    """
    logits = np.full((1, 1, 4, 4), 0.2, dtype=np.float32)
    m = SegmentationMetrics(threshold=0.5)
    m.update_logits(logits, _gt_ones())
    assert _dice(m) == pytest.approx(1.0, abs=1e-6)


def test_logits_zero_maps_to_probability_half() -> None:
    """logits=0 → 概率 0.5，在 ``>= 0.5`` 口径下应判 positive。"""
    m = SegmentationMetrics(threshold=0.5)
    m.update_logits(np.zeros((1, 1, 2, 2), dtype=np.float32), _gt_ones((1, 1, 2, 2)))
    assert _dice(m) == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# 4：接口混用必须报错
# --------------------------------------------------------------------------- #
def test_probability_update_rejects_logits() -> None:
    """把 logits 传给 update_probabilities 必须报错，而不是静默出错。"""
    m = SegmentationMetrics(threshold=0.5)
    with pytest.raises(ValueError, match="update_logits"):
        m.update_probabilities(np.array([[[[-3.0]]]], dtype=np.float32), _gt_ones())
    with pytest.raises(ValueError, match="update_logits"):
        m.update_probabilities(np.array([[[[2.5]]]], dtype=np.float32), _gt_ones())


def test_probability_update_rejects_non_finite() -> None:
    """含 NaN 的概率必须报错。"""
    m = SegmentationMetrics(threshold=0.5)
    bad = np.array([[[[np.nan]]]], dtype=np.float32)
    with pytest.raises(ValueError, match="非有限"):
        m.update_probabilities(bad, _gt_ones())


def test_logits_update_rejects_non_finite() -> None:
    """含 Inf 的 logits 必须报错。"""
    m = SegmentationMetrics(threshold=0.5)
    bad = torch.tensor([[[[float("inf")]]]])
    with pytest.raises(ValueError, match="非有限"):
        m.update_logits(bad, _gt_ones())


def test_binary_update_rejects_probabilities() -> None:
    """把概率/整型多值传给 update_binary 必须报错。"""
    m = SegmentationMetrics(threshold=0.5)
    with pytest.raises(ValueError, match="update_logits"):
        m.update_binary(np.array([[[[2]]]], dtype=np.int32), _gt_ones())


def test_binary_update_accepts_bool_and_01() -> None:
    """bool 与 0/1 两种二值输入都应被接受且结果一致。"""
    gt = _gt_ones((1, 1, 2, 2))
    m_bool = SegmentationMetrics(threshold=0.5)
    m_bool.update_binary(np.ones((1, 1, 2, 2), dtype=bool), gt)
    m_int = SegmentationMetrics(threshold=0.5)
    m_int.update_binary(np.ones((1, 1, 2, 2), dtype=np.uint8), gt)
    assert _dice(m_bool) == pytest.approx(_dice(m_int), abs=1e-9)
    assert _dice(m_bool) == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# 6：兼容入口必须显式传 kind
# --------------------------------------------------------------------------- #
def test_update_requires_explicit_kind() -> None:
    """不传 kind 必须 TypeError（禁止猜测）。"""
    m = SegmentationMetrics(threshold=0.5)
    with pytest.raises(TypeError, match="kind"):
        m.update(np.full((1, 1, 2, 2), 0.2, dtype=np.float32), _gt_ones((1, 1, 2, 2)))
    with pytest.raises(TypeError, match="kind"):
        m.update(np.zeros((1, 1, 2, 2), dtype=np.float32), _gt_ones((1, 1, 2, 2)),
                 kind="probability")


def test_update_with_explicit_kind_matches_dedicated_api() -> None:
    """显式 kind 的结果必须与专用方法一致。"""
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(2, 1, 8, 8)).astype(np.float32)
    gt = (rng.random((2, 1, 8, 8)) > 0.7).astype(np.uint8)

    m_direct = SegmentationMetrics(threshold=0.5)
    m_direct.update_logits(logits, gt)

    m_kind = SegmentationMetrics(threshold=0.5)
    m_kind.update(logits, gt, kind="logits")

    assert m_direct.summary() == m_kind.summary()


# --------------------------------------------------------------------------- #
# 阈值行为
# --------------------------------------------------------------------------- #
def test_threshold_changes_decision_for_probability_input() -> None:
    """同一概率在不同阈值下应给出不同判定。"""
    probs = np.full((1, 1, 2, 2), 0.3, dtype=np.float32)
    gt = _gt_ones((1, 1, 2, 2))

    low = SegmentationMetrics(threshold=0.25)
    low.update_probabilities(probs, gt)
    assert _dice(low) == pytest.approx(1.0, abs=1e-6)

    high = SegmentationMetrics(threshold=0.5)
    high.update_probabilities(probs, gt)
    assert _dice(high) == pytest.approx(0.0, abs=1e-6)


def test_shape_mismatch_raises() -> None:
    """pred 与 gt shape 不一致必须报错。"""
    m = SegmentationMetrics(threshold=0.5)
    with pytest.raises(ValueError, match="shape"):
        m.update(np.ones((1, 1, 4, 4), dtype=bool),
                 np.ones((1, 1, 5, 5), dtype=bool), kind="binary")
