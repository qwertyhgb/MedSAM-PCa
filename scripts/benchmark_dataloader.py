#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DataLoader 吞吐基准（Phase 1.5 / 第十二节）。

**纯 CPU 任务，不使用 GPU，不进行任何训练。**

对若干 ``num_workers`` 配置读取固定数量的 batch，测量吞吐并选择正式训练默认值。

产出::

    outputs/benchmarks/dataloader_benchmark.json

用法::

    /root/anaconda3/envs/lm/bin/python scripts/benchmark_dataloader.py --workers 2 4 8 --batches 200
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

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.picai_2d import PICAI2DDataset  # noqa: E402

LOGGER = logging.getLogger("benchmark_dataloader")

DEFAULT_DATA_ROOT = "/opt/data/private/lm/data/Prostate/PI-CAI"


def benchmark_workers(records: pd.DataFrame, data_root: Path, workers: int,
                      batch_size: int, num_batches: int, prefetch_factor: int,
                      persistent: bool) -> Dict[str, Any]:
    """测量指定 num_workers 下的 DataLoader 吞吐。

    Args:
        records: slice 级 manifest。
        data_root: PI-CAI 根目录。
        workers: ``num_workers``。
        batch_size: batch 大小。
        num_batches: 读取的 batch 数。
        prefetch_factor: ``prefetch_factor``。
        persistent: 是否启用 ``persistent_workers``。

    Returns:
        吞吐统计字典。
    """
    ds = PICAI2DDataset(records=records, data_root=data_root,
                        image_size=1024, augment=True, seed=42,
                        mask_cache_dir=PROJECT_ROOT / "data" / "cache" / "masks_aligned")
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=workers, pin_memory=False,          # 纯 CPU 测量
        prefetch_factor=prefetch_factor if workers > 0 else None,
        persistent_workers=persistent if workers > 0 else False,
    )

    result: Dict[str, Any] = {"num_workers": workers, "batch_size": batch_size,
                              "prefetch_factor": prefetch_factor,
                              "persistent_workers": persistent}
    t0 = time.perf_counter()
    n_img = 0
    iterator = iter(loader)
    bar = tqdm(range(num_batches), desc=f"workers={workers}", unit="batch",
               file=sys.stdout, ncols=100)
    for _ in bar:
        try:
            batch = next(iterator)
        except StopIteration:
            break
        n_img += int(batch["image"].shape[0])
        elapsed = time.perf_counter() - t0
        bar.set_postfix({"img/s": f"{n_img / max(elapsed, 1e-6):.2f}"})
    elapsed = time.perf_counter() - t0
    bar.close()
    del loader, ds

    result.update({
        "seconds": round(elapsed, 3),
        "batches": min(num_batches, max(n_img // batch_size, 0)),
        "images": n_img,
        "batches_per_second": round(num_batches / max(elapsed, 1e-6), 4),
        "images_per_second": round(n_img / max(elapsed, 1e-6), 4),
        "seconds_per_batch": round(elapsed / max(num_batches, 1), 4),
    })
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    """入口。"""
    parser = argparse.ArgumentParser(description="DataLoader 吞吐基准（CPU only）",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=PROJECT_ROOT / "data" / "manifests" / "picai_slices_fold0_train.csv")
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT))
    parser.add_argument("--workers", nargs="*", type=int, default=[2, 4, 8])
    parser.add_argument("--batches", type=int, default=200, help="每个配置读取的 batch 数")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "outputs" / "benchmarks" / "dataloader_benchmark.json")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)

    if not args.manifest.is_file():
        raise FileNotFoundError(f"manifest 不存在: {args.manifest}")
    records = pd.read_csv(args.manifest)
    LOGGER.info("manifest: %d slices | batch_size=%d | batches=%d",
                len(records), args.batch_size, args.batches)

    report: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "manifest": str(args.manifest),
        "num_slices": int(len(records)),
        "batch_size": args.batch_size,
        "num_batches": args.batches,
        "note": "纯 CPU 基准，不使用 GPU；pin_memory=False",
        "results": [],
    }

    for w in args.workers:
        LOGGER.info("测试 num_workers=%d ...", w)
        r = benchmark_workers(records, args.data_root, w, args.batch_size,
                              args.batches, args.prefetch_factor, persistent=True)
        report["results"].append(r)
        LOGGER.info("  workers=%d -> %.3f s | %.3f batch/s | %.2f img/s",
                    w, r["seconds"], r["batches_per_second"], r["images_per_second"])

    if report["results"]:
        best = max(report["results"], key=lambda r: r["images_per_second"])
        report["recommended_num_workers"] = best["num_workers"]
        LOGGER.info("推荐 num_workers = %d (%.2f img/s)",
                    best["num_workers"], best["images_per_second"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    LOGGER.info("已写出 %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
