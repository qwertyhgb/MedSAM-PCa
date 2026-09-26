#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""预构建 mask 几何对齐缓存（Phase 1.5 / 第十三节）。

背景：部分 human-expert lesion mask 与对应 T2W 的几何不一致
（Pooch25 8/205、resampled 32/1295）。dataset 在首次访问时会现场重采样，
若训练中途才首次命中会造成**突然阻塞**。本脚本一次性把所有需要的对齐结果
预生成到 ``data/cache/masks_aligned/``。

规则：
    1. 以 **T2W 作为 reference**；
    2. mask 使用 **nearest-neighbor**；
    3. **原始数据只读**，绝不修改；
    4. 已存在且**几何校验通过**的缓存默认跳过；
    5. ``--overwrite`` 强制重建；
    6. 全程 ``tqdm`` 进度条，结束时打印 成功/跳过/失败 数量。

用法::

    /root/anaconda3/envs/lm/bin/python scripts/build_mask_cache.py
    /root/anaconda3/envs/lm/bin/python scripts/build_mask_cache.py --overwrite
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.picai_2d import (  # noqa: E402
    aligned_cache_path,
    geometries_match,
    read_geometry,
    _resample_mask_to_reference,
)

LOGGER = logging.getLogger("build_mask_cache")

DEFAULT_DATA_ROOT = os.environ.get(
    "PICAI_DATA_ROOT", "/opt/data/private/lm/data/Prostate/PI-CAI"
)


def plan_cache(manifest: Path, data_root: Path, cache_dir: Path,
               overwrite: bool) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """扫描 manifest，决定每个 case 需要「新建」还是「跳过」。

    Args:
        manifest: case-level manifest（``picai_cases.csv``）。
        data_root: PI-CAI 根目录。
        cache_dir: 缓存目录。
        overwrite: 是否强制重建。

    Returns:
        ``(to_build, to_skip)``，元素为含 ``case_id`` / ``mask`` / ``t2w`` /
        ``cache`` / ``reason`` 的字典。
    """
    df = pd.read_csv(manifest)
    to_build: List[Dict[str, Any]] = []
    to_skip: List[Dict[str, Any]] = []

    rows = list(df.itertuples(index=False))
    for row in tqdm(rows, desc="扫描 geometry", unit="case", ncols=100, file=sys.stdout):
        if pd.isna(row.lesion_mask_path) or pd.isna(row.t2w_path):
            to_skip.append({"case_id": row.case_id, "reason": "missing path"})
            continue
        mask_abs = str(data_root / str(row.lesion_mask_path))
        t2w_abs = str(data_root / str(row.t2w_path))
        if not (Path(mask_abs).is_file() and Path(t2w_abs).is_file()):
            to_skip.append({"case_id": row.case_id, "reason": "file not found"})
            continue

        if geometries_match(read_geometry(mask_abs), read_geometry(t2w_abs)):
            to_skip.append({"case_id": row.case_id, "reason": "already aligned"})
            continue

        cache = aligned_cache_path(mask_abs, t2w_abs, cache_dir)
        if cache.is_file() and not overwrite:
            # 校验缓存几何是否与 T2W 一致
            if geometries_match(read_geometry(str(cache)), read_geometry(t2w_abs)):
                to_skip.append({"case_id": row.case_id, "reason": "cache valid",
                                "cache": str(cache)})
                continue
            LOGGER.warning("缓存几何校验失败，将重建: %s", cache)
        to_build.append({"case_id": row.case_id, "mask": mask_abs, "t2w": t2w_abs,
                         "cache": str(cache), "reason": "geometry mismatch"})
    return to_build, to_skip


def build_one(task: Dict[str, Any], overwrite: bool) -> Dict[str, Any]:
    """构建（或跳过）单个对齐缓存。

    使用线程而非进程：SimpleITK 的读写会释放 GIL，且避免重复导入开销。

    Args:
        task: :func:`plan_cache` 输出的任务字典。
        overwrite: 是否强制重建。

    Returns:
        含 ``status`` (``created`` / ``skipped`` / ``failed``) 的结果字典。
    """
    cache = Path(task["cache"])
    try:
        if cache.is_file() and not overwrite:
            # plan 阶段已确认需要重建，这里只在并发下再校一次
            pass
        aligned = _resample_mask_to_reference(task["mask"], task["t2w"])
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_name(cache.name + ".tmp.nii.gz")
        import SimpleITK as sitk
        sitk.WriteImage(aligned, str(tmp), True)
        tmp.replace(cache)
        # 写入后校验
        if not geometries_match(read_geometry(str(cache)), read_geometry(task["t2w"])):
            return {**task, "status": "failed", "error": "post-write geometry mismatch"}
        return {**task, "status": "created"}
    except Exception as exc:  # noqa: BLE001 - 显式记录，不静默吞掉
        return {**task, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """入口。"""
    parser = argparse.ArgumentParser(
        description="预构建 mask 几何对齐缓存",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=PROJECT_ROOT / "data" / "manifests" / "picai_cases.csv")
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT))
    parser.add_argument("--cache-dir", type=Path,
                        default=PROJECT_ROOT / "data" / "cache" / "masks_aligned")
    parser.add_argument("--workers", type=int, default=4, help="并发线程数")
    parser.add_argument("--overwrite", action="store_true", help="强制重建已有缓存")
    parser.add_argument("--report", type=Path,
                        default=PROJECT_ROOT / "outputs" / "data_audit" / "mask_cache_report.json")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)

    if not args.manifest.is_file():
        raise FileNotFoundError(f"manifest 不存在: {args.manifest}")

    to_build, to_skip = plan_cache(args.manifest, args.data_root, args.cache_dir,
                                   args.overwrite)
    LOGGER.info("需要对齐缓存: %d 个 case；无需处理: %d 个", len(to_build), len(to_skip))

    created, failed = [], []
    if to_build:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futures = [ex.submit(build_one, t, args.overwrite) for t in to_build]
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc="构建对齐缓存", unit="case", ncols=100, file=sys.stdout):
                r = fut.result()
                if r["status"] == "created":
                    created.append(r)
                elif r["status"] == "failed":
                    failed.append(r)
    else:
        LOGGER.info("无需新建任何缓存。")

    skip_reasons: Dict[str, int] = {}
    for s in to_skip:
        skip_reasons[s["reason"]] = skip_reasons.get(s["reason"], 0) + 1

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "manifest": str(args.manifest),
        "cache_dir": str(args.cache_dir),
        "overwrite": args.overwrite,
        "cases_requiring_alignment": len(to_build),
        "cache_created": len(created),
        "cache_skipped": len(to_skip),
        "cache_failed": len(failed),
        "skip_reasons": skip_reasons,
        "created_cases": [c["case_id"] for c in created],
        "failed": [{"case_id": f["case_id"], "error": f.get("error")} for f in failed],
        "existing_cache_files": len(list(args.cache_dir.glob("*.nii.gz")))
        if args.cache_dir.is_dir() else 0,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    L: List[str] = []
    add = L.append
    add("")
    add("=" * 70)
    add("Mask Alignment Cache 构建结果")
    add("=" * 70)
    add(f"需要对齐的 case 数 : {len(to_build)}")
    add(f"新建成功           : {len(created)}")
    add(f"跳过               : {len(to_skip)}  {skip_reasons}")
    add(f"失败               : {len(failed)}")
    for f in failed:
        add(f"   !! {f['case_id']}: {f.get('error')}")
    add(f"缓存目录内文件总数 : {summary['existing_cache_files']}")
    add(f"报告               : {args.report}")
    add("=" * 70)
    print("\n".join(L))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
