#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""四个 decoder baseline 的强制 shape / 参数 / 梯度测试（Phase 1 / 第二十六、二十七节）。

对每个模型验证：

1. 输入 ``[1, 3, 1024, 1024]`` → 输出 ``[1, 1, 1024, 1024]``
2. 记录关键中间 shape
3. ``total / frozen encoder / trainable decoder`` 参数量
4. encoder 冻结（trainable encoder == 0、梯度不存在）
5. decoder 梯度非零（反向传播可达）

运行::

    /root/anaconda3/envs/lm/bin/python tests/test_decoders.py
    /root/anaconda3/envs/lm/bin/python -m pytest tests/test_decoders.py -v -s
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.medsam_pca import SUPPORTED_MODELS, MedSAMPCA, build_model  # noqa: E402

LOGGER = logging.getLogger("test_decoders")

SEED = 20260924
INPUT_SHAPE = (1, 3, 1024, 1024)
EXPECTED_OUTPUT = (1, 1, 1024, 1024)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_CACHE: Dict[str, MedSAMPCA] = {}


def get_model(name: str, device: Optional[str] = None) -> MedSAMPCA:
    """构建并缓存模型（避免重复加载 358MB 权重）。"""
    dev = device or DEVICE
    key = f"{name}|{dev}"
    if key not in _CACHE:
        _CACHE[key] = build_model(name, freeze_encoder=True, device=dev)
    return _CACHE[key]


def _shapes(model: MedSAMPCA, name: str) -> Dict[str, Any]:
    """跑一次前向，收集 shape、参数量与输出有限性。"""
    torch.manual_seed(SEED)
    model.eval()
    x = torch.randn(*INPUT_SHAPE, device=DEVICE)

    with torch.no_grad():
        out = model(x)
    logits = out["logits"]

    assert set(out.keys()) == {"logits"}, f"输出键应只有 logits，实际 {sorted(out)}"
    actual = tuple(logits.shape)
    assert actual == EXPECTED_OUTPUT, (
        f"{name} 输出 shape 不符: 期望 {EXPECTED_OUTPUT}, 实际 {actual}"
    )
    assert torch.isfinite(logits).all(), f"{name} 输出含 NaN/Inf"

    stats = model.parameter_stats()
    return {
        "model_name": name,
        "input_shape": list(INPUT_SHAPE),
        "output_shape": list(actual),
        "feature_keys": list(model.feature_keys),
        "encoder_total": stats["encoder_total"],
        "encoder_trainable": stats["encoder_trainable"],
        "decoder_total": stats["decoder_total"],
        "decoder_trainable": stats["decoder_trainable"],
        "total": stats["total"],
        "trainable": stats["trainable"],
        "all_finite": bool(torch.isfinite(logits).all()),
    }


def _gradient_check(model: MedSAMPCA, name: str) -> Dict[str, Any]:
    """检查 decoder 梯度可达、encoder 无梯度。"""
    model.train()
    x = torch.randn(*INPUT_SHAPE, device=DEVICE)

    x.requires_grad_(False)
    out = model(x)
    loss = out["logits"].mean()
    loss.backward()

    dec_grads = [(n, p) for n, p in model.decoder.named_parameters()
                 if p.grad is not None]
    dec_nonzero = [n for n, p in dec_grads if float(p.grad.abs().sum()) > 0]
    enc_with_grad = [n for n, p in model.encoder.named_parameters()
                     if p.grad is not None and float(p.grad.abs().sum()) > 0]

    assert enc_with_grad == [], (
        f"{name} 的 encoder 出现梯度（应完全冻结）: {enc_with_grad[:5]}"
    )
    assert len(dec_nonzero) > 0, f"{name} 的 decoder 没有任何非零梯度"

    nonzero_ratio = len(dec_nonzero) / max(len(dec_grads), 1)
    model.zero_grad(set_to_none=True)
    return {
        "decoder_params_with_grad": len(dec_grads),
        "decoder_params_nonzero_grad": len(dec_nonzero),
        "decoder_nonzero_ratio": round(nonzero_ratio, 4),
        "encoder_params_with_grad": len(enc_with_grad),
    }


def test_e0_linear() -> Dict[str, Any]:
    """E0: Final Feature Linear Head。"""
    m = get_model("e0_linear")
    info = _shapes(m, "e0_linear")
    info.update(_gradient_check(m, "e0_linear"))
    LOGGER.info("[E0] %s", info)
    return info


def test_e1_simple_pyramid() -> Dict[str, Any]:
    """E1: Final Feature Simple Pyramid。"""
    m = get_model("e1_simple_pyramid")
    info = _shapes(m, "e1_simple_pyramid")
    info["intermediate_shapes"] = m.decoder.decoder.intermediate_shapes()
    info.update(_gradient_check(m, "e1_simple_pyramid"))
    LOGGER.info("[E1] %s", info)
    return info


def test_e2_unetr() -> Dict[str, Any]:
    """E2: UNETR-style decoder。"""
    m = get_model("e2_unetr")
    info = _shapes(m, "e2_unetr")
    info["intermediate_shapes"] = m.decoder.intermediate_shapes()
    info.update(_gradient_check(m, "e2_unetr"))
    LOGGER.info("[E2] %s", info)
    return info


def test_e3_multilevel_fpn() -> Dict[str, Any]:
    """E3: Multi-Level FPN。"""
    m = get_model("e3_multilevel_fpn")
    info = _shapes(m, "e3_multilevel_fpn")
    info["intermediate_shapes"] = m.decoder.intermediate_shapes()
    info.update(_gradient_check(m, "e3_multilevel_fpn"))
    LOGGER.info("[E3] %s", info)
    return info


TESTS = {
    "e0_linear": test_e0_linear,
    "e1_simple_pyramid": test_e1_simple_pyramid,
    "e2_unetr": test_e2_unetr,
    "e3_multilevel_fpn": test_e3_multilevel_fpn,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """顺序运行四个模型的测试并打印汇总表。"""
    parser = argparse.ArgumentParser(description="decoder shape / 参数 / 梯度测试")
    parser.add_argument("--models", nargs="*", default=list(TESTS),
                        choices=list(TESTS), help="要测试的模型")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)

    LOGGER.info("=" * 78)
    LOGGER.info("Decoder Shape Test (device=%s)", DEVICE)
    LOGGER.info("=" * 78)

    results: List[Dict[str, Any]] = []
    for name in args.models:
        results.append(TESTS[name]())

    L: List[str] = []
    add = L.append
    add("")
    add("=" * 100)
    add("%-20s %-20s %-14s %-14s %-14s %s" %
        ("Model", "Output", "Enc frozen", "Dec trainable", "Dec total", "Grad ok"))
    add("-" * 100)
    for r in results:
        add("%-20s %-20s %-14s %-14s %-14s %s" % (
            r["model_name"], tuple(r["output_shape"]),
            f"{r['encoder_trainable']}", f"{r['decoder_trainable']:,}",
            f"{r['decoder_total']:,}",
            f"{r['decoder_params_nonzero_grad']}/{r['decoder_params_with_grad']}"))
    add("-" * 100)
    add("ALL SHAPE TESTS PASSED")
    add("=" * 100)
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
