#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Decoder 中间特征真实空间尺寸的单元测试（Phase 2A）。

背景：E2 的 ``intermediate_shapes()`` 曾错误地声明为
``64 -> 128 -> 256 -> 512 -> 1024`` 的逐级上采样，而真实实现里四次融合都
发生在 64x64（因为 ``f3/f6/f9/f12`` 在 ViT-B 中分辨率相同）。

本测试用 ``forward hook`` 抓取真实中间张量，逐层与 ``intermediate_shapes()``
比对，确保**文档/声明与实现一致**，并锁定 E2 的"同分辨率融合"事实。

不涉及任何训练：全部在 ``torch.no_grad()`` 下做单次前向。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.multilevel_fpn import MultiLevelFPN  # noqa: E402
from src.models.simple_pyramid import SimplePyramid  # noqa: E402
from src.models.unetr_decoder import UNETRStyleDecoder  # noqa: E402

#: ``intermediate_shapes()`` 的键 -> 真实 module 属性名
SHAPE_KEY_TO_MODULE = {
    "e1": {"stem": "stem", "P5": "p5_down", "P4": "p4_conv",
           "P3": "p3_up", "P2": "p2_up2"},
    "e2": {"proj_deep": "proj_deep", "fuse_mid2": "fuse_mid2",
           "fuse_mid1": "fuse_mid1", "fuse_shallow": "fuse_shallow",
           "refine": "refine"},
    "e3": {"P2": "p2_up2", "P3": "p3_up", "P4": "proj_deep", "P5": "p5_down",
           "D2": "smooth2"},
}

#: 这些键描述的不是"某个模块的输出张量"，不参与 hook 比对。
NON_MODULE_KEYS = ("output",)


def _capture_shapes(model: torch.nn.Module, key_to_module: Dict[str, str],
                    forward_fn) -> Dict[str, List[int]]:
    """注册 forward hook，运行一次前向，返回每层输出的 shape（去掉 batch 维）。

    Args:
        model: decoder。
        key_to_module: shape 键 -> module 属性名。
        forward_fn: 调用模型前向的可调用对象。

    Returns:
        键 -> ``[C, H, W]``（已去掉 batch 维）。

    Raises:
        AttributeError: 指定的 module 属性不存在。
    """
    records: Dict[str, List[int]] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module, _inp, out):
            records[name] = list(out.shape[1:])
        return hook

    for key, attr in key_to_module.items():
        handles.append(getattr(model, attr).register_forward_hook(make_hook(key)))
    try:
        with torch.no_grad():
            logits = forward_fn()
    finally:
        for h in handles:
            h.remove()
    records["__logits__"] = list(logits.shape)
    return records


@pytest.mark.parametrize("height,width", [(64, 64), (32, 48)])
def test_e1_shapes_match_declared(height: int, width: int) -> None:
    """E1：声明的金字塔 shape 必须与真实前向一致。"""
    model = SimplePyramid(out_size=256).eval()
    neck = torch.zeros(1, 256, height, width)
    records = _capture_shapes(model, SHAPE_KEY_TO_MODULE["e1"], lambda: model(neck))

    declared = model.intermediate_shapes(height, width)
    for key, shape in declared.items():
        if key in NON_MODULE_KEYS:
            continue
        assert records[key] == shape, f"E1 {key}: 声明 {shape}，实际 {records[key]}"
    assert records["__logits__"] == [1, 1, 256, 256]


@pytest.mark.parametrize("height,width", [(64, 64), (32, 48)])
def test_e2_shapes_match_declared(height: int, width: int) -> None:
    """E2：声明的 shape 必须与真实前向一致（修正后的同分辨率版本）。"""
    model = UNETRStyleDecoder(out_size=256).eval()
    feats = {k: torch.zeros(1, 768, height, width) for k in ("f3", "f6", "f9", "f12")}
    records = _capture_shapes(model, SHAPE_KEY_TO_MODULE["e2"], lambda: model(feats))

    declared = model.intermediate_shapes(height, width)
    for key, shape in declared.items():
        if key in NON_MODULE_KEYS:
            continue
        assert records[key] == shape, f"E2 {key}: 声明 {shape}，实际 {records[key]}"
    assert records["__logits__"] == [1, 1, 256, 256]


def test_e2_fusion_is_same_resolution() -> None:
    """E2 的三次融合必须发生在**输入分辨率**上，而不是逐级上采样。"""
    model = UNETRStyleDecoder(out_size=256).eval()
    feats = {k: torch.zeros(1, 768, 64, 64) for k in ("f3", "f6", "f9", "f12")}
    records = _capture_shapes(model, SHAPE_KEY_TO_MODULE["e2"], lambda: model(feats))

    for key in ("proj_deep", "fuse_mid2", "fuse_mid1", "fuse_shallow"):
        assert records[key][1:] == [64, 64], f"{key} 应为 64x64，实际 {records[key]}"
    assert records["refine"][1:] == [128, 128], "refine 应在 x2 之后（128x128）"

    declared = model.intermediate_shapes(64, 64)
    assert declared["fuse_mid2"] == [128, 64, 64]
    assert declared["fuse_mid1"] == [128, 64, 64]
    assert declared["fuse_shallow"] == [128, 64, 64]
    assert declared["refine"] == [128, 128, 128]


@pytest.mark.parametrize("height,width", [(64, 64), (32, 48)])
def test_e3_shapes_match_declared(height: int, width: int) -> None:
    """E3：P2=4x、P3=2x、P4=1x、P5=1/2x 必须与真实前向一致。"""
    model = MultiLevelFPN(out_size=256).eval()
    feats = {k: torch.zeros(1, 768, height, width) for k in ("f3", "f6", "f9")}
    feats["neck"] = torch.zeros(1, 256, height, width)
    records = _capture_shapes(model, SHAPE_KEY_TO_MODULE["e3"], lambda: model(feats))

    declared = model.intermediate_shapes(height, width)
    for key, shape in declared.items():
        if key in NON_MODULE_KEYS:
            continue
        assert records[key] == shape, f"E3 {key}: 声明 {shape}，实际 {records[key]}"
    assert records["__logits__"] == [1, 1, 256, 256]


def test_e3_p2_comes_from_shallowest_feature() -> None:
    """E3 的 P2 必须来自 f3（最浅）且分辨率最高，P3 来自 f6。"""
    model = MultiLevelFPN(out_size=256).eval()
    feats = {k: torch.zeros(1, 768, 64, 64) for k in ("f3", "f6", "f9")}
    feats["neck"] = torch.zeros(1, 256, 64, 64)

    captured = {}

    def make_probe(name):
        def hook(_m, inp, _out):
            captured[name] = list(inp[0].shape[1:])
        return hook

    h1 = model.proj_shallow.register_forward_hook(make_probe("p2_src"))
    h2 = model.proj_mid.register_forward_hook(make_probe("p3_src"))
    h3 = model.proj_deep.register_forward_hook(make_probe("p4_src"))
    h4 = model.proj_neck.register_forward_hook(make_probe("p5_src"))
    try:
        with torch.no_grad():
            model(feats)
    finally:
        for h in (h1, h2, h3, h4):
            h.remove()

    assert captured["p2_src"] == [768, 64, 64] and captured["p3_src"] == [768, 64, 64]
    assert captured["p5_src"] == [256, 64, 64]

    declared = model.intermediate_shapes(64, 64)
    assert declared["P2"][1:] == [256, 256]
    assert declared["P3"][1:] == [128, 128]
    assert declared["P4"][1:] == [64, 64]
    assert declared["P5"][1:] == [32, 32]


def test_declared_output_matches_forward_for_all_decoders() -> None:
    """三个 decoder 的 ``output`` 声明都必须与实际 logits 空间尺寸一致。"""
    size = 128
    e1 = SimplePyramid(out_size=size).eval()
    e2 = UNETRStyleDecoder(out_size=size).eval()
    e3 = MultiLevelFPN(out_size=size).eval()

    with torch.no_grad():
        out1 = e1(torch.zeros(1, 256, 64, 64))
        out2 = e2({k: torch.zeros(1, 768, 64, 64) for k in ("f3", "f6", "f9", "f12")})
        feats = {k: torch.zeros(1, 768, 64, 64) for k in ("f3", "f6", "f9")}
        feats["neck"] = torch.zeros(1, 256, 64, 64)
        out3 = e3(feats)

    assert list(out1.shape) == [1, 1, size, size]
    assert list(out2.shape) == [1, 1, size, size]
    assert list(out3.shape) == [1, 1, size, size]
    assert e3.intermediate_shapes(64, 64)["output"] == [1, size, size]
