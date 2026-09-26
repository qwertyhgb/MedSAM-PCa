#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Volume-level / lesion-level 评估（Phase 2A）。

**只做评估**：不训练、不 backward、不改动 checkpoint。数据来源是
:mod:`scripts.evaluate_thresholds` 缓存的**原始分辨率概率体积**，因此
不会重复跑 encoder 前向。

产出
----
* ``volume_metrics.csv`` —— 每个 case × 每个阈值的 3D Dice / IoU / 像素混淆量；
* ``volume_summary.csv/json`` —— 按 positive / negative / all cases 分类汇总，
  主指标为 ``positive_case_volume_dice``；negative cases 单独报告
  ``false_positive_case_rate`` 与 FP 体素；
* ``lesion_metrics.csv`` / ``lesion_summary.json`` —— 3D 连通域（26-连通）级的
  病灶灵敏度、FP lesions/case、匹配病灶 Dice（匹配规则显式记录）；
* ``small_lesion_stratification.csv`` —— 按 GT 病灶物理体积分层（<0.5 / 0.5–1.0 / >1.0 cc）；
* ``component_filtering.csv`` —— 评估期最小预测连通域体积过滤扫描；
* ``volume_predictions/<run>_<tag>/<case>_pred.nii.gz`` —— 选中阈值的二值体积预测
  （与 T2W 的 spacing/origin/direction 完全一致）。

阈值化位置说明：概率先以 **bilinear** 从 1024 下采样到原始分辨率，再在原始分辨率
上做阈值化得到二值 mask（二值 mask 全程不做插值）。

用法::

    python scripts/evaluate_volumes.py \
        --cache-root outputs/evaluation/prob_cache \
        --runs e2_unetr:best e2_unetr:latest e3_multilevel_fpn:best e3_multilevel_fpn:latest \
        --thresholds 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 \
        --primary-thresholds 0.5 --min-volumes-mm3 0 10 25 50 100 250
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.picai_2d import read_mask_volume  # noqa: E402
from src.metrics.lesion_metrics import label_components, match_lesions  # noqa: E402
from train import load_config, resolve_path  # noqa: E402

LOGGER = logging.getLogger("evaluate_volumes")

DEFAULT_RUNS: Tuple[str, ...] = (
    "e2_unetr:best", "e2_unetr:latest",
    "e3_multilevel_fpn:best", "e3_multilevel_fpn:latest",
)
DEFAULT_THRESHOLDS: Tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(1, 20))
DEFAULT_MIN_VOLUMES_MM3: Tuple[float, ...] = (0.0, 10.0, 25.0, 50.0, 100.0, 250.0)


# --------------------------------------------------------------------------- #
def _parse_run_spec(spec: str) -> Tuple[str, str]:
    """解析 ``run:tag``。"""
    if ":" not in spec:
        raise ValueError(f"--runs 需要 ``run:tag`` 形式，收到 {spec!r}")
    run, tag = spec.split(":", 1)
    run, tag = run.strip(), tag.strip()
    if tag not in ("best", "latest"):
        raise ValueError(f"tag 只能是 best/latest，收到 {tag!r}")
    return run, tag


def _case_records(run: str) -> pd.DataFrame:
    """读取 run 的验证集 manifest，返回每个 case 一行。

    Args:
        run: 运行名。

    Returns:
        含 ``case_id`` / ``t2w_path`` / ``mask_path`` 的 DataFrame。
    """
    cfg = load_config(PROJECT_ROOT / "configs" / f"{run}.yaml")
    data_cfg = cfg["data"]
    records = pd.read_csv(resolve_path(data_cfg["val_manifest"]))
    case_df = records.drop_duplicates("case_id")[
        ["case_id", "t2w_path", "mask_path", "is_positive"]].copy()
    case_df["dataset_root"] = str(resolve_path(data_cfg["dataset_root"]))
    mask_cache = data_cfg.get("mask_cache_dir")
    case_df["mask_cache_dir"] = str(resolve_path(mask_cache)) if mask_cache else ""
    return case_df.reset_index(drop=True)


def _pad_to(prob: np.ndarray, shape: Tuple[int, int, int]) -> np.ndarray:
    """把概率体积补齐/裁剪到 GT 的 ``[Z, H, W]``。"""
    if prob.shape == shape:
        return prob
    out = np.zeros(shape, dtype=prob.dtype)
    z = min(prob.shape[0], shape[0])
    y = min(prob.shape[1], shape[1])
    x = min(prob.shape[2], shape[2])
    out[:z, :y, :x] = prob[:z, :y, :x]
    return out


def _volume_confusion(pred: np.ndarray, gt: np.ndarray) -> Dict[str, int]:
    """3D 体素级混淆统计。"""
    p = pred.astype(bool, copy=False)
    g = gt.astype(bool, copy=False)
    tp = int(np.logical_and(p, g).sum())
    fp = int(np.logical_and(p, ~g).sum())
    fn = int(np.logical_and(~p, g).sum())
    tn = int(p.size - tp - fp - fn)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _dice_iou(tp: int, fp: int, fn: int, eps: float = 1e-8) -> Tuple[float, float]:
    """由体素混淆量计算 Dice 与 IoU。"""
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    return float(dice), float(iou)


# --------------------------------------------------------------------------- #
def evaluate_run(run: str, tag: str, cache_root: Path, thresholds: Sequence[float],
                 primary_thresholds: Sequence[float], min_volumes: Sequence[float],
                 criterion: str, criterion_threshold: float, out_dir: Path,
                 save_predictions: bool, limit_cases: Optional[int]) -> Dict[str, Any]:
    """对单个 checkpoint 的概率缓存做 volume / lesion 级评估。

    Args:
        run: 运行名。
        tag: ``best`` / ``latest``。
        cache_root: 概率缓存根目录。
        thresholds: 需要计算 3D Dice 的阈值网格。
        primary_thresholds: 需要做连通域分析的阈值。
        min_volumes: 预测连通域最小体积过滤扫描（mm³）。
        criterion: lesion 匹配规则。
        criterion_threshold: 匹配规则阈值。
        out_dir: 结果根目录。
        save_predictions: 是否写二值 volume 预测。
        limit_cases: 仅处理前 N 个 case（冒烟）。

    Returns:
        该 checkpoint 的结果字典。

    Raises:
        FileNotFoundError: 概率缓存缺失。
    """
    cache_dir = cache_root / f"{run}_{tag}"
    meta_path = cache_dir / "_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"缺少概率缓存 {cache_dir}；请先运行：\n"
            f"  python scripts/evaluate_thresholds.py --cache-probs "
            f"--cache-runs {run} --runs {run}:{tag}"
        )
    with meta_path.open(encoding="utf-8") as fh:
        meta = json.load(fh)

    case_df = _case_records(run)
    case_ids = [c for c in case_df["case_id"].astype(str).tolist() if c in meta["cases"]]
    if limit_cases:
        case_ids = case_ids[:limit_cases]
    LOGGER.info("%s:%s 缓存 %d 个 case，本次处理 %d 个", run, tag,
                len(meta["cases"]), len(case_ids))

    rows_per_case: List[Dict[str, Any]] = []
    lesion_rows: List[Dict[str, Any]] = []
    lesion_detail: List[Dict[str, Any]] = []
    filter_rows: List[Dict[str, Any]] = []
    pred_writer: List[Tuple[str, np.ndarray, Dict[str, Any]]] = []
    label_suffix = f"{run}_{tag}"
    out_sub = out_dir / label_suffix
    out_sub.mkdir(parents=True, exist_ok=True)

    info_by_case = {str(k): v for k, v in meta["cases"].items()}
    for cid in tqdm(case_ids, desc=f"{run}:{tag} volume", unit="case", ncols=120,
                    file=sys.stdout):
        info = info_by_case[cid]
        spacing = tuple(float(s) for s in info["spacing"])
        voxel_mm3 = spacing[0] * spacing[1] * spacing[2]

        rec = case_df[case_df["case_id"].astype(str) == cid].iloc[0]
        t2w_abs = str(Path(rec["dataset_root"]) / str(rec["t2w_path"]))
        mask_cache = rec["mask_cache_dir"] or None
        gt_vol = (read_mask_volume(str(rec["mask_path"]), t2w_abs,
                                   Path(mask_cache) if mask_cache else None) != 0)

        prob = np.load(cache_dir / f"{cid}.npy").astype(np.float32)
        prob = _pad_to(prob, gt_vol.shape)
        if prob.shape != gt_vol.shape:
            raise ValueError(f"{cid}: 概率体积 {prob.shape} 与 GT {gt_vol.shape} 不一致")

        gt_voxels = int(gt_vol.sum())
        has_gt = gt_voxels > 0

        # ---- 多阈值 3D Dice / IoU ---- #
        for t in thresholds:
            pred = prob >= float(t)
            c = _volume_confusion(pred, gt_vol)
            dice, iou = _dice_iou(c["tp"], c["fp"], c["fn"])
            rows_per_case.append({
                "run": run, "tag": tag, "case_id": cid, "threshold": float(t),
                "has_gt": has_gt, "gt_voxels": gt_voxels,
                "pred_voxels": int(pred.sum()),
                "tp": c["tp"], "fp": c["fp"], "fn": c["fn"], "tn": c["tn"],
                "dice3d": dice, "iou3d": iou,
            })

        # ---- 连通域级分析（仅 primary thresholds） ---- #
        gt_labels, gt_comps = label_components(gt_vol, spacing, fully_connected=True)
        gt_volumes_cc = [comp.volume_cc for comp in gt_comps]

        for t in primary_thresholds:
            pred = prob >= float(t)
            pred_labels, pred_comps = label_components(pred, spacing, fully_connected=True)

            for min_vol in min_volumes:
                comps_used = ([c for c in pred_comps if c.volume_mm3 >= float(min_vol)]
                              if min_vol > 0 else pred_comps)
                matched = match_lesions(gt_labels, gt_comps, pred_labels, comps_used,
                                        criterion=criterion,
                                        criterion_threshold=criterion_threshold)
                n_gt = len(gt_comps)
                n_det = len(matched["matches"])
                dice_vals = [m["dice"] for m in matched["matches"].values()]
                filter_rows.append({
                    "run": run, "tag": tag, "case_id": cid, "threshold": float(t),
                    "min_pred_volume_mm3": float(min_vol),
                    "has_gt": has_gt,
                    "num_gt_lesions": n_gt,
                    "num_pred_lesions": len(comps_used),
                    "num_detected_gt_lesions": n_det,
                    "lesion_sensitivity": (n_det / n_gt) if n_gt else None,
                    "false_positive_lesions": int(len(matched["fp_pred_labels"])),
                    "matched_lesion_dice": (float(np.mean(dice_vals)) if dice_vals else None),
                })
                if min_vol == 0.0:
                    lesion_rows.append({
                        "run": run, "tag": tag, "case_id": cid, "threshold": float(t),
                        "has_gt": has_gt, "num_gt_lesions": n_gt,
                        "num_pred_lesions": len(comps_used),
                        "num_detected_gt_lesions": n_det,
                        "lesion_sensitivity": (n_det / n_gt) if n_gt else None,
                        "false_positive_lesions": int(len(matched["fp_pred_labels"])),
                        "matched_lesion_dice": (float(np.mean(dice_vals))
                                                if dice_vals else None),
                        "extra_preds_hitting_same_gt": len(matched["extra_matched_preds"]),
                        "criterion": criterion,
                        "criterion_threshold": float(criterion_threshold),
                    })
                    for comp in gt_comps:
                        m = matched["matches"].get(comp.label)
                        lesion_detail.append({
                            "run": run, "tag": tag, "case_id": cid,
                            "threshold": float(t),
                            "gt_label": comp.label,
                            "gt_voxels": comp.voxels,
                            "gt_volume_mm3": comp.volume_mm3,
                            "gt_volume_cc": comp.volume_cc,
                            "detected": m is not None,
                            "matched_dice": (m["dice"] if m else 0.0),
                            "matched_overlap_voxels": (m["overlap_voxels"] if m else 0),
                        })

        # ---- 保存二值预测（第一个 primary threshold） ---- #
        if save_predictions and primary_thresholds:
            t_main = float(primary_thresholds[0])
            pred_main = (prob >= t_main)
            pred_writer.append((cid, pred_main.astype(np.uint8), info))

    # ---- 写 volume 预测 ---- #
    if pred_writer:
        pred_dir = out_sub / "volume_predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        for cid, arr, info in tqdm(pred_writer, desc=f"{run}:{tag} 写 volume 预测",
                                   unit="case", ncols=120, file=sys.stdout):
            img = sitk.GetImageFromArray(arr)
            img.SetSpacing(tuple(float(s) for s in info["spacing"]))
            img.SetOrigin(tuple(float(s) for s in info["origin"]))
            img.SetDirection(tuple(float(s) for s in info["direction"]))
            sitk.WriteImage(img, str(pred_dir / f"{cid}_pred.nii.gz"), True)
        LOGGER.info("已写出 %d 个二值体积预测到 %s", len(pred_writer), pred_dir)

    # ---- 落盘 ---- #
    df_vol = pd.DataFrame(rows_per_case)
    df_vol.to_csv(out_sub / "volume_metrics.csv", index=False)
    df_lesion = pd.DataFrame(lesion_rows)
    if not df_lesion.empty:
        df_lesion.to_csv(out_sub / "lesion_metrics.csv", index=False)
    df_detail = pd.DataFrame(lesion_detail)
    if not df_detail.empty:
        df_detail.to_csv(out_sub / "lesion_detail.csv", index=False)
    df_filter = pd.DataFrame(filter_rows)
    if not df_filter.empty:
        df_filter.to_csv(out_sub / "component_filtering_raw.csv", index=False)

    volume_summary = _summarize_volume(df_vol)
    lesion_summary = _summarize_lesions(df_lesion, df_detail, criterion, criterion_threshold)
    strat = _stratify(df_detail)
    filt = _summarize_filtering(df_filter)

    if volume_summary:
        pd.DataFrame(volume_summary).to_csv(out_sub / "volume_summary.csv", index=False)
    if strat:
        pd.DataFrame(strat).to_csv(out_sub / "small_lesion_stratification.csv", index=False)
    if filt:
        pd.DataFrame(filt).to_csv(out_sub / "component_filtering.csv", index=False)

    payload = {
        "run": run, "tag": tag,
        "checkpoint": meta.get("checkpoint"),
        "num_cases": len(case_ids),
        "thresholds": list(map(float, thresholds)),
        "primary_thresholds": list(map(float, primary_thresholds)),
        "min_volumes_mm3": list(map(float, min_volumes)),
        "matching_rule": {"criterion": criterion,
                          "criterion_threshold": float(criterion_threshold),
                          "connectivity": 26},
        "volume_summary": volume_summary,
        "lesion_summary": lesion_summary,
        "small_lesion_stratification": strat,
        "component_filtering": filt,
        "note": "validation-split volume/lesion analysis; probability resized bilinear, "
                "binarization performed at native resolution",
    }
    with (out_sub / "volume_lesion_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    LOGGER.info("%s:%s volume 评估完成：%d 个 case，%d 个阈值",
                run, tag, len(case_ids), len(thresholds))
    return payload


# --------------------------------------------------------------------------- #
def _summarize_volume(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """按阈值汇总 volume dice（positive / negative / all cases 分组）。"""
    rows: List[Dict[str, Any]] = []
    if df.empty:
        return rows
    for t, group in df.groupby("threshold"):
        pos = group[group["has_gt"]]
        neg = group[~group["has_gt"]]
        allc = group
        tp, fp, fn = int(group["tp"].sum()), int(group["fp"].sum()), int(group["fn"].sum())
        micro_dice, micro_iou = _dice_iou(tp, fp, fn)
        n_neg = len(neg)
        fp_cases = int((neg["pred_voxels"] > 0).sum()) if n_neg else 0
        rows.append({
            "threshold": float(t),
            "num_cases": int(len(allc)),
            "num_positive_cases": int(len(pos)),
            "num_negative_cases": n_neg,
            "positive_case_volume_dice": float(pos["dice3d"].mean()) if len(pos) else None,
            "positive_case_volume_iou": float(pos["iou3d"].mean()) if len(pos) else None,
            "positive_case_median_dice": float(pos["dice3d"].median()) if len(pos) else None,
            "all_case_volume_dice_including_empty": float(allc["dice3d"].mean()),
            "all_case_volume_iou_including_empty": float(allc["iou3d"].mean()),
            "micro_volume_dice": micro_dice,
            "micro_volume_iou": micro_iou,
            "micro_volume_precision": (tp / (tp + fp)) if (tp + fp) else None,
            "micro_volume_recall": (tp / (tp + fn)) if (tp + fn) else None,
            "false_positive_case_rate": (fp_cases / n_neg) if n_neg else None,
            "fp_cases": fp_cases,
            "mean_fp_voxels_per_negative_case": (float(neg["pred_voxels"].mean())
                                                 if n_neg else None),
        })
    return rows


def _summarize_lesions(df_lesion: pd.DataFrame, df_detail: pd.DataFrame,
                       criterion: str, criterion_threshold: float) -> Dict[str, Any]:
    """汇总 lesion 级指标（必须显式记录匹配规则）。"""
    if df_lesion.empty:
        return {}
    per_t: List[Dict[str, Any]] = []
    for t, group in df_lesion.groupby("threshold"):
        gt_total = int(group["num_gt_lesions"].sum())
        det_total = int(group["num_detected_gt_lesions"].sum())
        pred_total = int(group["num_pred_lesions"].sum())
        fp_total = int(group["false_positive_lesions"].sum())
        dice_vals = group["matched_lesion_dice"].dropna().tolist()
        det_rates = group.loc[group["has_gt"], "lesion_sensitivity"].dropna().tolist()
        per_t.append({
            "threshold": float(t),
            "num_cases": int(len(group)),
            "num_cases_with_gt": int(group["has_gt"].sum()),
            "num_gt_lesions": gt_total,
            "num_pred_lesions": pred_total,
            "num_detected_gt_lesions": det_total,
            "lesion_sensitivity_pooled": (det_total / gt_total) if gt_total else None,
            "lesion_sensitivity_per_case_mean": (float(np.mean(det_rates))
                                                 if det_rates else None),
            "false_positive_lesions": fp_total,
            "fp_lesions_per_case": fp_total / len(group) if len(group) else None,
            "matched_lesion_dice": (float(np.mean(dice_vals)) if dice_vals else None),
            "extra_preds_hitting_same_gt": int(group["extra_preds_hitting_same_gt"].sum()),
        })
    return {
        "matching_rule": {"criterion": criterion,
                          "criterion_threshold": float(criterion_threshold),
                          "connectivity": 26,
                          "note": "一个 GT 只允许一次检出；同一 GT 上的其余预测连通域计为 FP"},
        "per_threshold": per_t,
    }


def _stratify(df_detail: pd.DataFrame) -> List[Dict[str, Any]]:
    """按 GT 病灶体积分层（<0.5 / 0.5–1.0 / >1.0 cc）。"""
    if df_detail.empty:
        return []
    bins = (("<0.5cc", 0.0, 0.5), ("0.5-1.0cc", 0.5, 1.0), (">1.0cc", 1.0, float("inf")))
    rows: List[Dict[str, Any]] = []
    for t, group in df_detail.groupby("threshold"):
        for name, lo, hi in bins:
            sel = group[(group["gt_volume_cc"] >= lo) & (group["gt_volume_cc"] < hi)]
            n = len(sel)
            n_det = int(sel["detected"].sum()) if n else 0
            det_dice = sel.loc[sel["detected"], "matched_dice"]
            rows.append({
                "threshold": float(t),
                "bin": name, "min_cc": lo, "max_cc": hi,
                "num_lesions": n,
                "num_detected": n_det,
                "detection_sensitivity": (n_det / n) if n else None,
                "mean_dice_all_lesions": (float(sel["matched_dice"].mean()) if n else None),
                "mean_dice_detected_only": (float(det_dice.mean())
                                            if len(det_dice) else None),
            })
    return rows


def _summarize_filtering(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """汇总连通域体积过滤扫描结果。"""
    if df.empty:
        return []
    rows: List[Dict[str, Any]] = []
    gb = df.groupby(["threshold", "min_pred_volume_mm3"])
    for (t, min_vol), group in gb:
        gt_total = int(group["num_gt_lesions"].sum())
        det_total = int(group["num_detected_gt_lesions"].sum())
        pred_total = int(group["num_pred_lesions"].sum())
        fp_total = int(group["false_positive_lesions"].sum())
        dice_vals = group["matched_lesion_dice"].dropna().tolist()
        rows.append({
            "threshold": float(t),
            "min_pred_volume_mm3": float(min_vol),
            "num_cases": int(len(group)),
            "num_gt_lesions": gt_total,
            "num_pred_lesions": pred_total,
            "num_pred_lesions_removed": None,
            "lesion_sensitivity_pooled": (det_total / gt_total) if gt_total else None,
            "false_positive_lesions": fp_total,
            "fp_lesions_per_case": fp_total / len(group) if len(group) else None,
            "matched_lesion_dice": (float(np.mean(dice_vals)) if dice_vals else None),
        })
    rows.sort(key=lambda r: (r["threshold"], r["min_pred_volume_mm3"]))
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    """入口。"""
    parser = argparse.ArgumentParser(
        description="Volume / lesion 级评估（读取 threshold scan 的概率缓存，只做评估）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS))
    parser.add_argument("--cache-root", type=Path,
                        default=Path("outputs/evaluation/prob_cache"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/evaluation/volume"))
    parser.add_argument("--thresholds", nargs="+", type=float, default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--primary-thresholds", nargs="+", type=float, default=[0.5],
                        help="做连通域分析与体积预测的阈值")
    parser.add_argument("--min-volumes-mm3", nargs="+", type=float,
                        default=list(DEFAULT_MIN_VOLUMES_MM3))
    parser.add_argument("--criterion", default="any_overlap",
                        choices=["any_overlap", "dice", "iou", "overlap_fraction"])
    parser.add_argument("--criterion-threshold", type=float, default=0.1)
    parser.add_argument("--no-save-predictions", action="store_true")
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)

    cache_root = resolve_path(args.cache_root)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = [_parse_run_spec(s) for s in args.runs]
    summary: List[Dict[str, Any]] = []
    for run, tag in specs:
        payload = evaluate_run(run, tag, cache_root, args.thresholds,
                               args.primary_thresholds, args.min_volumes_mm3,
                               args.criterion, args.criterion_threshold,
                               out_dir, not args.no_save_predictions, args.limit_cases)
        # 取第一个 primary threshold 的关键数字用于全局汇总
        t_main = float(args.primary_thresholds[0])
        vol_row = next((r for r in payload["volume_summary"]
                        if r["threshold"] == t_main), {})
        les_row = next((r for r in payload["lesion_summary"].get("per_threshold", [])
                        if r["threshold"] == t_main), {})
        summary.append({
            "run": run, "tag": tag,
            "primary_threshold": t_main,
            "positive_case_volume_dice": vol_row.get("positive_case_volume_dice"),
            "micro_volume_dice": vol_row.get("micro_volume_dice"),
            "false_positive_case_rate": vol_row.get("false_positive_case_rate"),
            "lesion_sensitivity_pooled": les_row.get("lesion_sensitivity_pooled"),
            "fp_lesions_per_case": les_row.get("fp_lesions_per_case"),
            "matched_lesion_dice": les_row.get("matched_lesion_dice"),
        })

    if summary:
        pd.DataFrame(summary).to_csv(out_dir / "summary_primary_threshold.csv", index=False)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump({"primary_threshold_summary": summary,
                   "thresholds": list(map(float, args.thresholds)),
                   "primary_thresholds": list(map(float, args.primary_thresholds)),
                   "matching_rule": {"criterion": args.criterion,
                                     "criterion_threshold": args.criterion_threshold},
                   "note": "validation-split volume/lesion analysis"},
                  fh, ensure_ascii=False, indent=2)
    LOGGER.info("全部完成：%d 个 checkpoint，结果目录 %s", len(specs), out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
