#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""四个 decoder 的显存与延迟基准（Phase 1 / 第二十三、二十八节）。

对每个模型、每个候选 batch，测量：

* batch=1 及以上的 forward 延迟（warmup 后取均值）
* forward + backward 的延迟
* peak allocated / peak reserved 显存
* 是否 OOM

产出::

    outputs/benchmarks/decoder_benchmark.json

用法::

    /root/anaconda3/envs/lm/bin/python scripts/benchmark_decoders.py --batches 1 2 4 8
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.medsam_pca import SUPPORTED_MODELS, build_model  # noqa: E402

LOGGER = logging.getLogger("benchmark")


def _sync() -> None:
    """同步 CUDA。"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_call(fn, warmup: int, repeats: int) -> float:
    """测量函数调用平均耗时（秒）。

    Args:
        fn: 无参可调用对象。
        warmup: 预热次数。
        repeats: 正式测量次数。

    Returns:
        平均秒数。
    """
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    _sync()
    return (time.perf_counter() - t0) / repeats


def benchmark_one(model, batch_size: int, warmup: int, repeats: int,
                  image_size: int = 1024) -> Dict[str, Any]:
    """测量单个模型在指定 batch 下的延迟与显存。

    Args:
        model: :class:`MedSAMPCA`。
        batch_size: 物理 batch。
        warmup: 预热次数。
        repeats: 测量次数。
        image_size: 输入边长。

    Returns:
        结果字典（OOM 时 ``oom=True``）。
    """
    result: Dict[str, Any] = {"batch_size": batch_size, "oom": False}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        x = torch.randn(batch_size, 3, image_size, image_size, device="cuda")
        model.train()
        model.encoder.eval()
        trainable = model.trainable_parameters()

        # ---- forward only ----
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            fwd = time_call(lambda: model(x), warmup, repeats)
        _sync()
        result["forward_peak_allocated_mib"] = round(
            torch.cuda.max_memory_allocated() / 1024 ** 2, 2)
        result["forward_peak_reserved_mib"] = round(
            torch.cuda.max_memory_reserved() / 1024 ** 2, 2)

        # ---- forward + backward ----
        def step():
            out = model(x)
            loss = out["logits"].mean()
            loss.backward()
            model.zero_grad(set_to_none=True)

        torch.cuda.reset_peak_memory_stats()
        fb = time_call(step, warmup, repeats)
        _sync()
        result["forward_backward_peak_allocated_mib"] = round(
            torch.cuda.max_memory_allocated() / 1024 ** 2, 2)
        result["forward_backward_peak_reserved_mib"] = round(
            torch.cuda.max_memory_reserved() / 1024 ** 2, 2)

        result["forward_seconds"] = round(fwd, 4)
        result["forward_backward_seconds"] = round(fb, 4)
        result["images_per_second"] = round(batch_size / fb, 3)
        del x
    except torch.cuda.OutOfMemoryError:
        result["oom"] = True
        _sync()
        torch.cuda.empty_cache()
        LOGGER.warning("batch=%d 发生 OOM", batch_size)
    finally:
        torch.cuda.empty_cache()
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    """入口。"""
    parser = argparse.ArgumentParser(description="decoder 显存/延迟基准",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--models", nargs="*", default=list(SUPPORTED_MODELS),
                        choices=list(SUPPORTED_MODELS))
    parser.add_argument("--batches", nargs="*", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "outputs" / "benchmarks" / "decoder_benchmark.json")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)

    if not torch.cuda.is_available():
        LOGGER.error("CUDA 不可用，无法进行显存基准测试")
        return 1

    report: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": torch.cuda.get_device_name(0),
        "total_memory_mib": round(
            torch.cuda.get_device_properties(0).total_memory / 1024 ** 2, 2),
        "warmup": args.warmup, "repeats": args.repeats,
        "models": {},
    }

    for name in args.models:
        LOGGER.info("=" * 70)
        model = build_model(name, freeze_encoder=True, device="cuda")
        stats = model.parameter_stats()
        LOGGER.info("[%s] decoder trainable=%d total=%d", name,
                    stats["decoder_trainable"], stats["decoder_total"])

        entry: Dict[str, Any] = {"parameter_stats": stats, "batches": []}
        for bs in args.batches:
            r = benchmark_one(model, bs, args.warmup, args.repeats)
            entry["batches"].append(r)
            if r["oom"]:
                LOGGER.info("[%s] batch=%d -> OOM", name, bs)
            else:
                LOGGER.info("[%s] batch=%d fwd=%.3fs fwd+bwd=%.3fs "
                            "peak_alloc=%.0fMiB peak_reserved=%.0fMiB",
                            name, bs, r["forward_seconds"], r["forward_backward_seconds"],
                            r["forward_backward_peak_allocated_mib"],
                            r["forward_backward_peak_reserved_mib"])
            if r["oom"]:
                break  # 更大 batch 必然也 OOM

        report["models"][name] = entry
        del model
        torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    LOGGER.info("已写出 %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
