#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""多阈值切片级评估（Phase 2A）。

**只做前向推理**：不训练、不 backward、不改动任何 checkpoint。

特点
----
* 一次 model forward 同时累计**多个阈值**的切片级指标（概率直方图法，见
  :mod:`src.metrics.threshold_scan`），避免"每个阈值重跑一遍 encoder"；
* 默认覆盖 8 个 checkpoint：E0/E1/E2/E3 的 ``best`` 与 ``latest``；
* 可选把每个 case 的**原始分辨率概率体积**缓存到磁盘（``--cache-probs``），
  供 :mod:`scripts.evaluate_volumes` 复用，避免重复 forward；
* 所有耗时段（forward、下采样、写盘）均带 tqdm 进度条。

用法::

    python scripts/evaluate_thresholds.py \
        --runs e0_linear:best e1_simple_pyramid:best \
        --thresholds 0.05 0.10 0.15 0.30 0.50 0.70 0.90 0.95

    # 为 E2/E3 的 4 个 checkpoint 额外缓存概率体积
    python scripts/evaluate_thresholds.py --cache-probs --cache-runs e2_unetr e3_multilevel_fpn

注意：阈值扫描属于 **validation 分析**，不是最终 test 性能。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.picai_2d import PICAI2DDataset, read_geometry  # noqa: E402
from src.metrics.threshold_scan import DEFAULT_THRESHOLDS, ThresholdScanMetrics  # noqa: E402
from src.models.medsam_pca import build_model  # noqa: E402
from train import load_config, resolve_path, set_seed  # noqa: E402

LOGGER = logging.getLogger("evaluate_thresholds")

DEFAULT_RUNS: Tuple[str, ...] = (
    "e0_linear:best", "e0_linear:latest",
    "e1_simple_pyramid:best", "e1_simple_pyramid:latest",
    "e2_unetr:best", "e2_unetr:latest",
    "e3_multilevel_fpn:best", "e3_multilevel_fpn:latest",
)

CSV_FIELDS = [
    "threshold", "positive_slice_dice", "positive_slice_iou", "all_slice_dice",
    "micro_dice", "micro_iou", "micro_precision", "micro_recall",
    "micro_specificity", "fp_slice_rate", "fp_slices", "num_slices",
    "num_positive_gt_slices", "num_empty_gt_slices",
]


# --------------------------------------------------------------------------- #
def _parse_run_spec(spec: str) -> Tuple[str, str]:
    """解析 ``run:tag`` 形式。

    Args:
        spec: 形如 ``e3_multilevel_fpn:best``。

    Returns:
        ``(run_name, tag)``。

    Raises:
        ValueError: 格式不正确。
    """
    if ":" not in spec:
        raise ValueError(f"--runs 需要 ``run:tag`` 形式，收到 {spec!r}")
    run, tag = spec.split(":", 1)
    run, tag = run.strip(), tag.strip()
    if tag not in ("best", "latest"):
        raise ValueError(f"tag 只能是 best/latest，收到 {tag!r}")
    return run, tag


def _safe_best(scan: ThresholdScanMetrics, key: str) -> Optional[Dict[str, Any]]:
    """按指标挑最优阈值行；若该指标全部不可用（如子集无阳性切片）返回 ``None``。

    Args:
        scan: 多阈值累计器。
        key: 指标名。

    Returns:
        最优行或 ``None``。
    """
    try:
        return scan.best_by(key)
    except KeyError:
        LOGGER.warning("没有可用的 %s 指标（当前子集可能不含阳性切片）", key)
        return None


def _build_loader(cfg: Dict[str, Any], batch_size: Optional[int],
                  num_workers: Optional[int],
                  limit_slices: Optional[int]) -> Tuple[DataLoader, pd.DataFrame]:
    """构建验证集 DataLoader。

    Args:
        cfg: 配置。
        batch_size: 覆盖配置中的 batch size。
        num_workers: 覆盖配置中的 worker 数。
        limit_slices: 仅用前 N 个切片（冒烟测试）。

    Returns:
        ``(loader, records)``。
    """
    data_cfg = cfg["data"]
    records = pd.read_csv(resolve_path(data_cfg["val_manifest"]))
    if limit_slices is not None and limit_slices > 0:
        records = records.iloc[:limit_slices].reset_index(drop=True)
        LOGGER.info("冒烟模式：仅使用前 %d 个切片", len(records))

    mask_cache = data_cfg.get("mask_cache_dir")
    ds = PICAI2DDataset(
        records=records,
        data_root=resolve_path(data_cfg["dataset_root"]),
        image_size=int(data_cfg.get("image_size", 1024)),
        augment=False,
        mask_cache_dir=resolve_path(mask_cache) if mask_cache else None,
    )
    loader = DataLoader(
        ds,
        batch_size=int(batch_size or cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(num_workers if num_workers is not None else data_cfg.get("num_workers", 8)),
        pin_memory=True,
        persistent_workers=False,
    )
    return loader, records


def _load_model(cfg: Dict[str, Any], checkpoint: Path, device: str):
    """构建模型并载入可训练参数。

    Args:
        cfg: 配置。
        checkpoint: checkpoint 路径。
        device: ``cuda`` / ``cpu``。

    Returns:
        ``(model, ckpt_meta)``，其中 ``ckpt_meta`` 含 ``epoch`` 与 ``best_metric``。

    Raises:
        RuntimeError: checkpoint 出现意外键。
    """
    model = build_model(cfg["model_name"],
                        checkpoint_path=resolve_path(cfg["checkpoint"]),
                        freeze_encoder=True, device=device)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["trainable_state_dict"], strict=False)
    if unexpected:
        raise RuntimeError(f"checkpoint 出现意外键: {unexpected[:5]}")
    if missing:
        LOGGER.warning("checkpoint 缺少键: %s", missing[:5])
    model.eval()
    return model, {"epoch": ckpt.get("epoch"), "best_metric": ckpt.get("best_metric"),
                   "seed": ckpt.get("seed")}


# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_checkpoint(run: str, tag: str, thresholds: Sequence[float],
                        out_dir: Path, cache_root: Optional[Path],
                        cache_this: bool, args: argparse.Namespace) -> Dict[str, Any]:
    """对单个 checkpoint 做多阈值评估（必要时缓存概率体积）。

    Args:
        run: 运行名。
        tag: ``best`` 或 ``latest``。
        thresholds: 阈值序列。
        out_dir: 结果输出目录。
        cache_root: 概率缓存根目录（``None`` 表示不缓存）。
        cache_this: 是否为该 checkpoint 写缓存。
        args: CLI 参数。

    Returns:
        结果字典。
    """
    cfg_path = PROJECT_ROOT / "configs" / f"{run}.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"缺少配置文件 {cfg_path}")
    cfg = load_config(cfg_path)
    checkpoint = resolve_path(Path("outputs/runs") / run / f"checkpoint_{tag}.pth")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"缺少 checkpoint {checkpoint}")

    device = args.device
    loader, records = _build_loader(cfg, args.batch_size, args.num_workers, args.limit_slices)
    model, ckpt_meta = _load_model(cfg, checkpoint, device)

    scan = ThresholdScanMetrics(thresholds)
    n_slices = 0
    t0 = time.time()

    # ---- 概率体积缓存（原始分辨率） ---- #
    buffers: Dict[str, Dict[int, np.ndarray]] = defaultdict(dict)
    geom_cache: Dict[str, Dict[str, Any]] = {}
    t2w_cache: Dict[str, str] = {}
    data_root = resolve_path(cfg["data"]["dataset_root"])
    record_by_case = records.drop_duplicates("case_id").set_index("case_id")["t2w_path"].to_dict()

    bar = tqdm(loader, desc=f"{run}:{tag} forward", unit="batch", ncols=120, file=sys.stdout)
    for batch in bar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16,
                                enabled=(device == "cuda")):
            logits = model(images)["logits"]
        probs = torch.sigmoid(logits.float())

        case_ids = [str(c) for c in batch["case_id"]]
        scan.update_probabilities(probs, masks, case_ids=case_ids)
        n_slices += int(probs.shape[0])

        if cache_root is not None and cache_this:
            for i, cid in enumerate(case_ids):
                if cid not in geom_cache:
                    rel = str(record_by_case[cid])
                    abs_p = str(data_root / rel)
                    t2w_cache[cid] = abs_p
                    geom_cache[cid] = read_geometry(abs_p)
                geom = geom_cache[cid]
                w, h = int(geom["size"][0]), int(geom["size"][1])
                small = F.interpolate(probs[i:i + 1], size=(h, w),
                                      mode="bilinear", align_corners=False)
                buffers[cid][int(batch["slice_idx"][i])] = \
                    small[0, 0].to(torch.float16).cpu().numpy()

        bar.set_postfix({"slices": n_slices})

    # ---- 写阈值指标 ---- #
    rows = scan.rows()
    run_dir = out_dir / f"{run}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)[CSV_FIELDS]
    df.to_csv(run_dir / "threshold_metrics.csv", index=False)

    best_pos = _safe_best(scan, "positive_slice_dice")
    best_micro = _safe_best(scan, "micro_dice")
    payload: Dict[str, Any] = {
        "run": run,
        "tag": tag,
        "model_name": cfg["model_name"],
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": ckpt_meta["epoch"],
        "checkpoint_best_metric": ckpt_meta["best_metric"],
        "seed": ckpt_meta["seed"],
        "num_slices": n_slices,
        "num_cases": int(records["case_id"].nunique()),
        "thresholds": list(map(float, thresholds)),
        "best_by_positive_slice_dice": best_pos,
        "best_by_micro_dice": best_micro,
        "rows": rows,
        "elapsed_seconds": round(time.time() - t0, 1),
        "note": "validation-split threshold analysis; not a test-set performance claim",
    }
    with (run_dir / "threshold_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    # ---- 写概率缓存 ---- #
    cached_files = 0
    if cache_root is not None and cache_this and buffers:
        cache_dir = cache_root / f"{run}_{tag}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        meta: Dict[str, Any] = {"run": run, "tag": tag, "checkpoint": str(checkpoint),
                                "cases": {}}
        for cid, slices in tqdm(buffers.items(), desc=f"{run}:{tag} 缓存概率",
                                unit="case", ncols=120, file=sys.stdout):
            geom = geom_cache[cid]
            z_max = max(slices)
            vol = np.zeros((z_max + 1, int(geom["size"][1]), int(geom["size"][0])),
                           dtype=np.float16)
            for z, arr in slices.items():
                if z < vol.shape[0]:
                    vol[z] = arr
            np.save(cache_dir / f"{cid}.npy", vol)
            meta["cases"][cid] = {
                "spacing": list(geom["spacing"]),
                "origin": list(geom["origin"]),
                "direction": list(geom["direction"]),
                "size": list(geom["size"]),
                "t2w_path": t2w_cache[cid],
                "num_slices_written": int(vol.shape[0]),
                "max_manifest_slice": int(z_max),
            }
            cached_files += 1
        with (cache_dir / "_meta.json").open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
        LOGGER.info("已缓存 %d 个 case 的概率体积到 %s", cached_files, cache_dir)

    payload["cached_cases"] = cached_files
    LOGGER.info("%s:%s 完成 | pos-Dice@best=%.4f (t=%.2f) | micro-Dice@best=%.4f (t=%.2f) | %.1fs",
                run, tag, best_pos["positive_slice_dice"], best_pos["threshold"],
                best_micro["micro_dice"], best_micro["threshold"], payload["elapsed_seconds"])
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    """入口。"""
    parser = argparse.ArgumentParser(
        description="多阈值切片级评估（仅前向推理）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS),
                        help="``run:tag`` 列表，tag ∈ {best, latest}")
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=list(DEFAULT_THRESHOLDS), help="概率阈值")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("outputs/evaluation/threshold_scan"))
    parser.add_argument("--cache-probs", action="store_true",
                        help="缓存原始分辨率概率体积供 volume 评估复用")
    parser.add_argument("--cache-root", type=Path,
                        default=Path("outputs/evaluation/prob_cache"))
    parser.add_argument("--cache-runs", nargs="*", default=None,
                        help="只为这些 run 写缓存（默认：--cache-probs 时全部）")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-slices", type=int, default=None,
                        help="仅用前 N 个切片（冒烟测试）")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)

    specs = [_parse_run_spec(s) for s in args.runs]
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_root = resolve_path(args.cache_root) if args.cache_probs else None

    all_rows: List[Dict[str, Any]] = []
    summary: List[Dict[str, Any]] = []
    for run, tag in tqdm(specs, desc="Checkpoints", unit="ckpt", ncols=120, file=sys.stdout):
        cache_this = cache_root is not None and (
            args.cache_runs is None or run in set(args.cache_runs))
        payload = evaluate_checkpoint(run, tag, args.thresholds, out_dir,
                                      cache_root, cache_this, args)
        summary.append({
            "run": run, "tag": tag,
            "checkpoint_epoch": payload["checkpoint_epoch"],
            "best_by_pos_dice_threshold": payload["best_by_positive_slice_dice"]["threshold"],
            "best_by_pos_dice": payload["best_by_positive_slice_dice"]["positive_slice_dice"],
            "best_by_pos_dice_precision": payload["best_by_positive_slice_dice"]["micro_precision"],
            "best_by_pos_dice_recall": payload["best_by_positive_slice_dice"]["micro_recall"],
            "best_by_pos_dice_fp_rate": payload["best_by_positive_slice_dice"]["fp_slice_rate"],
            "best_by_micro_dice_threshold": payload["best_by_micro_dice"]["threshold"],
            "best_by_micro_dice": payload["best_by_micro_dice"]["micro_dice"],
            "elapsed_seconds": payload["elapsed_seconds"],
        })
        for row in payload["rows"]:
            row = dict(row)
            row["run"] = run
            row["tag"] = tag
            all_rows.append(row)

    if summary:
        pd.DataFrame(summary).to_csv(out_dir / "summary_best_per_checkpoint.csv", index=False)
    if all_rows:
        cols = ["run", "tag"] + CSV_FIELDS
        pd.DataFrame(all_rows)[cols].to_csv(out_dir / "all_thresholds.csv", index=False)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump({"checkpoints": summary,
                   "thresholds": list(map(float, args.thresholds)),
                   "note": "validation-split threshold analysis"}, fh,
                  ensure_ascii=False, indent=2)

    LOGGER.info("全部完成：%d 个 checkpoint，结果目录 %s", len(specs), out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
