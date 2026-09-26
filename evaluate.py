#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase 1 评估脚本。

功能：

1. 在 validation split 的**全部** slice 上评估（可按 case 限制）。
2. 报告 positive-slice Dice / all-slice Dice / IoU / Precision / Recall /
   Specificity / FP slices / 空 GT 统计。
3. **按 case 重建并保存完整 volume 预测**（nii.gz，使用 T2W 几何），
   为 Phase 2 的 lesion-wise sensitivity、FP/case、3D 连通域分析做准备。
   Phase 1 不计算 lesion-wise 指标。

用法::

    /root/anaconda3/envs/lm/bin/python evaluate.py \\
        --config configs/e3_multilevel_fpn.yaml \\
        --checkpoint outputs/runs/e3_multilevel_fpn/checkpoint_best.pth \\
        --max-cases 40
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.picai_2d import PICAI2DDataset, read_geometry  # noqa: E402
from src.metrics.segmentation import SegmentationMetrics  # noqa: E402
from src.models.medsam_pca import build_model  # noqa: E402
from train import load_config, resolve_path, set_seed  # noqa: E402

LOGGER = logging.getLogger("evaluate")


@torch.no_grad()
def evaluate(cfg: Dict[str, Any], checkpoint: Path, out_dir: Path,
             threshold: float = 0.5, max_cases: Optional[int] = None,
             save_predictions: bool = True) -> Dict[str, Any]:
    """执行评估。

    Args:
        cfg: 配置。
        checkpoint: ``checkpoint_best.pth`` 路径。
        out_dir: 输出目录。
        threshold: 概率二值化阈值。
        max_cases: 仅评估前 N 个 case（``None`` 为全部）。
        save_predictions: 是否保存 volume 预测。

    Returns:
        指标字典。
    """
    data_cfg = cfg["data"]
    records = pd.read_csv(resolve_path(data_cfg["val_manifest"]))
    if max_cases is not None:
        keep = sorted(records["case_id"].unique())[:max_cases]
        records = records[records["case_id"].isin(keep)].reset_index(drop=True)
        LOGGER.info("限制评估前 %d 个 case（%d 个 slice）", len(keep), len(records))

    mask_cache = data_cfg.get("mask_cache_dir")
    ds = PICAI2DDataset(
        records=records,
        data_root=resolve_path(data_cfg["dataset_root"]),
        image_size=int(data_cfg.get("image_size", 1024)),
        augment=False,
        mask_cache_dir=resolve_path(mask_cache) if mask_cache else None,
    )
    loader = DataLoader(ds, batch_size=int(cfg["train"]["batch_size"]),
                        shuffle=False, num_workers=int(data_cfg.get("num_workers", 8)),
                        pin_memory=True)

    model = build_model(cfg["model_name"],
                        checkpoint_path=resolve_path(cfg["checkpoint"]),
                        freeze_encoder=True,
                        device="cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["trainable_state_dict"], strict=False)
    if unexpected:
        raise RuntimeError(f"checkpoint 出现意外键: {unexpected[:5]}")
    LOGGER.info("已载入 %s (epoch=%s, best=%s)", checkpoint, ckpt.get("epoch"),
                ckpt.get("best_metric"))
    model.eval()

    metrics = SegmentationMetrics(threshold=threshold)
    # case_id -> {slice_idx: 原始尺寸二值 mask(uint8)}
    volumes: Dict[str, Dict[int, np.ndarray]] = defaultdict(dict)
    geom_cache: Dict[str, Dict[str, Any]] = {}
    t2w_cache: Dict[str, str] = {}

    bar = tqdm(loader, desc="Evaluating (forward only)", unit="batch",
               ncols=120, dynamic_ncols=True, file=sys.stdout)
    for batch in bar:
        images = batch["image"].cuda(non_blocking=True)
        masks = batch["mask"].cuda(non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16,
                                enabled=torch.cuda.is_available()):
            logits = model(images)["logits"].float()
        metrics.update_logits(logits, masks, case_ids=list(batch["case_id"]))

        if save_predictions:
            pred = (torch.sigmoid(logits) > threshold).float()  # [B,1,1024,1024]
            for i, cid in enumerate(batch["case_id"]):
                cid = str(cid)
                z = int(batch["slice_idx"][i])
                if cid not in geom_cache:
                    rel = str(records.loc[records["case_id"] == cid, "t2w_path"].iloc[0])
                    abs_p = str(resolve_path(data_cfg["dataset_root"]) / rel)
                    t2w_cache[cid] = abs_p
                    geom_cache[cid] = read_geometry(abs_p)
                h, w = geom_cache[cid]["size"][1], geom_cache[cid]["size"][0]
                # 回到原始分辨率：nearest，保持 0/1
                small = F.interpolate(pred[i: i + 1], size=(h, w), mode="nearest")
                volumes[cid][z] = (small[0, 0].cpu().numpy() > 0.5).astype(np.uint8)

    summary = metrics.summary()
    summary["per_case"] = metrics.per_case_summary()
    summary["checkpoint"] = str(checkpoint)
    summary["model_name"] = cfg["model_name"]
    summary["threshold"] = threshold
    summary["num_cases"] = len(records["case_id"].unique())

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "eval_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    # -------- 保存 volume 预测 -------- #
    if save_predictions and volumes:
        pred_dir = out_dir / "volume_predictions"
        pred_dir.mkdir(exist_ok=True)
        for cid, slices in tqdm(volumes.items(), desc="Writing volume predictions",
                                unit="case", ncols=120, file=sys.stdout):
            geom = geom_cache[cid]
            z_max = max(slices)
            vol = np.zeros((z_max + 1, geom["size"][1], geom["size"][0]), dtype=np.uint8)
            for z, m in slices.items():
                if z < vol.shape[0]:
                    vol[z] = m
            img = sitk.GetImageFromArray(vol)
            img.SetSpacing(tuple(geom["spacing"]))
            img.SetOrigin(tuple(geom["origin"]))
            img.SetDirection(tuple(geom["direction"]))
            sitk.WriteImage(img, str(pred_dir / f"{cid}_pred.nii.gz"), True)
        LOGGER.info("已保存 %d 个 case 的 volume 预测到 %s", len(volumes), pred_dir)

    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    """入口。"""
    parser = argparse.ArgumentParser(description="Phase 1 评估",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--no-save-predictions", action="store_true")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    out_dir = args.out_dir or (resolve_path(cfg["output"]["root"]) /
                               (cfg.get("run_name") or cfg["model_name"]) / "eval")
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)

    set_seed(int(cfg["train"]["seed"]))
    summary = evaluate(cfg, args.checkpoint, out_dir, threshold=args.threshold,
                       max_cases=args.max_cases,
                       save_predictions=not args.no_save_predictions)

    LOGGER.info("positive_slice_dice = %s", summary["positive_slice_dice"])
    LOGGER.info("all_slice_dice      = %s", summary["all_slice_dice"])
    LOGGER.info("micro precision/recall/specificity = %.4f / %.4f / %.4f",
                summary["micro_precision"], summary["micro_recall"],
                summary["micro_specificity"])
    LOGGER.info("FP slices = %d / %d 空 GT slice",
                summary["fp_slices"], summary["num_empty_gt_slices"])
    LOGGER.info("已写出 %s", out_dir / "eval_metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
