#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 Phase 2A 的评估产物汇总成 Markdown 表格（供报告引用）。

输入（由 `scripts/evaluate_thresholds.py` 与 `scripts/evaluate_volumes.py` 生成）：

* ``outputs/evaluation/threshold_scan/summary.json`` 与 ``all_thresholds.csv``
* ``outputs/evaluation/volume/summary.json``
* ``outputs/evaluation/volume/<run>_<tag>/volume_summary.csv``
* ``outputs/evaluation/volume/<run>_<tag>/volume_lesion_summary.json``

输出：``docs/experiments/tables/phase2a_*.md``（纯数据表，不含结论文字）。

用法::

    python scripts/summarize_phase2a.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RUN_ORDER = ["e0_linear", "e1_simple_pyramid", "e2_unetr", "e3_multilevel_fpn"]
TAG_ORDER = ["best", "latest"]

VOLUME_COLS = [
    ("threshold", "阈值"),
    ("num_positive_cases", "阳性 case"),
    ("positive_case_volume_dice", "阳性 case 3D Dice"),
    ("positive_case_median_dice", "阳性 case 中位 Dice"),
    ("micro_volume_dice", "微平均 3D Dice"),
    ("micro_volume_precision", "微平均 precision"),
    ("micro_volume_recall", "微平均 recall"),
    ("false_positive_case_rate", "FP case 率"),
]


def _fmt(value: Any, digits: int = 4) -> str:
    """格式化数值用于 Markdown 表格。"""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if value != value:  # NaN
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """生成 Markdown 表格。"""
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out) + "\n"


def _key(run: str, tag: str) -> str:
    """按期望顺序排序用的键。"""
    return f"{RUN_ORDER.index(run) if run in RUN_ORDER else 99}_{TAG_ORDER.index(tag)}"


# --------------------------------------------------------------------------- #
def build_threshold_tables(scan_dir: Path, out_dir: Path) -> List[Path]:
    """生成阈值扫描相关表格。

    Args:
        scan_dir: ``outputs/evaluation/threshold_scan``。
        out_dir: 表格输出目录。

    Returns:
        写出的文件列表。
    """
    written: List[Path] = []
    summary_path = scan_dir / "summary.json"
    all_csv = scan_dir / "all_thresholds.csv"
    if not summary_path.is_file() or not all_csv.is_file():
        print(f"[skip] 缺少 {summary_path} 或 {all_csv}")
        return written

    with summary_path.open(encoding="utf-8") as fh:
        summary = json.load(fh)
    df = pd.read_csv(all_csv)
    df["__k"] = [_key(r, t) for r, t in zip(df["run"], df["tag"])]

    # 每个 checkpoint 的最优 operating point
    rows = []
    for ck in sorted(summary["checkpoints"],
                     key=lambda c: _key(c["run"], c["tag"])):
        rows.append([
            f"`{ck['run']}`", ck["tag"], _fmt(ck.get("checkpoint_epoch"), 0),
            _fmt(ck.get("best_by_pos_dice_threshold"), 2),
            _fmt(ck.get("best_by_pos_dice")),
            _fmt(ck.get("best_by_pos_dice_precision")),
            _fmt(ck.get("best_by_pos_dice_recall")),
            _fmt(ck.get("best_by_pos_dice_fp_rate")),
            _fmt(ck.get("best_by_micro_dice_threshold"), 2),
            _fmt(ck.get("best_by_micro_dice")),
        ])
    text = "# Phase 2A：阈值扫描 —— 每个 checkpoint 的最优 operating point\n\n"
    text += ("> 数据源：`outputs/evaluation/threshold_scan/summary.json`；"
             "阈值在 fold0 验证集扫描（**validation 分析，不是 test 性能**）。\n"
             "> `best-by-pos-Dice` = 在扫描网格上使阳性切片 Dice 最大的阈值；"
             "同一行的 precision/recall/FP 率取自该阈值。\n\n")
    text += _table(
        ["checkpoint", "tag", "epoch", "argmax 阈值", "阳性切片 Dice", "precision",
         "recall", "FP-slice rate", "micro-Dice 最优阈值", "micro-Dice"], rows)
    p = out_dir / "phase2a_threshold_best_per_checkpoint.md"
    p.write_text(text, encoding="utf-8")
    written.append(p)

    # 全阈值网格：按 checkpoint 分块
    parts = ["# Phase 2A：全阈值网格切片级指标\n",
             "> 数据源：`outputs/evaluation/threshold_scan/all_thresholds.csv`。\n",
             "> `num_positive_gt_slices` 为 GT 非空切片数；`fp_slice_rate` 为阴性切片被误报比例。\n"]
    for (run, tag), group in df.groupby(["run", "tag"], sort=False):
        group = group.sort_values("threshold")
        parts.append(f"\n## {run} ({tag})\n\n")
        rows = [[_fmt(r["threshold"], 2), _fmt(r["positive_slice_dice"]),
                 _fmt(r["positive_slice_iou"]), _fmt(r["all_slice_dice"]),
                 _fmt(r["micro_dice"]), _fmt(r["micro_precision"]),
                 _fmt(r["micro_recall"]), _fmt(r["fp_slice_rate"]),
                 _fmt(r["fp_slices"], 0)] for _, r in group.iterrows()]
        parts.append(_table(
            ["阈值", "阳性切片 Dice", "阳性切片 IoU", "all-slice Dice", "微平均 Dice",
             "precision", "recall", "FP-slice rate", "FP 切片数"], rows))
    p = out_dir / "phase2a_threshold_full_grid.md"
    p.write_text("\n".join(parts), encoding="utf-8")
    written.append(p)
    return written


def build_volume_tables(volume_dir: Path, out_dir: Path) -> List[Path]:
    """生成 volume / lesion 相关表格。

    Args:
        volume_dir: ``outputs/evaluation/volume``。
        out_dir: 表格输出目录。

    Returns:
        写出的文件列表。
    """
    written: List[Path] = []
    summary_path = volume_dir / "summary.json"
    if not summary_path.is_file():
        print(f"[skip] 缺少 {summary_path}")
        return written

    with summary_path.open(encoding="utf-8") as fh:
        summary = json.load(fh)

    rows = []
    for item in sorted(summary["primary_threshold_summary"],
                       key=lambda c: _key(c["run"], c["tag"])):
        rows.append([
            f"`{item['run']}`", item["tag"], _fmt(item["primary_threshold"], 2),
            _fmt(item.get("positive_case_volume_dice")),
            _fmt(item.get("micro_volume_dice")),
            _fmt(item.get("false_positive_case_rate")),
            _fmt(item.get("lesion_sensitivity_pooled")),
            _fmt(item.get("fp_lesions_per_case"), 2),
            _fmt(item.get("matched_lesion_dice")),
        ])
    text = ("# Phase 2A：volume / lesion 级结果（主阈值）\n\n"
            "> 数据源：`outputs/evaluation/volume/summary.json`。\n"
            "> 3D Dice 在 T2W **原始分辨率**上计算（概率 bilinear 下采样、二值化在原始分辨率执行）。\n"
            "> `FP case 率` = 阴性 case 中被预测出至少一个前景体素的比例。\n"
            "> lesion 指标基于 3D 26-连通域、匹配规则 `any_overlap`（详见各 checkpoint 的 JSON）。\n\n")
    text += _table(
        ["checkpoint", "tag", "主阈值", "阳性 case 3D Dice", "微平均 3D Dice",
         "FP case 率", "病灶灵敏度(pooled)", "FP lesions/case", "匹配病灶 Dice"], rows)

    details = []
    for sub in sorted(volume_dir.glob("*_*"), key=lambda p: _key(*(p.name.split("_", 1)[0], "best"))):
        json_path = sub / "volume_lesion_summary.json"
        if not json_path.is_file():
            continue
        with json_path.open(encoding="utf-8") as fh:
            payload = json.load(fh)
        details.append((sub.name, payload))

    for name, payload in details:
        vol_rows = payload.get("volume_summary") or []
        if vol_rows:
            rows = [[_fmt(r["threshold"], 2), _fmt(r["num_positive_cases"], 0),
                     _fmt(r["positive_case_volume_dice"]), _fmt(r["positive_case_median_dice"]),
                     _fmt(r["micro_volume_dice"]), _fmt(r["micro_volume_precision"]),
                     _fmt(r["micro_volume_recall"]), _fmt(r["false_positive_case_rate"]),
                     _fmt(r["mean_fp_voxels_per_negative_case"], 0)]
                    for r in vol_rows]
            text += f"\n## {name}：全阈值 volume 指标\n\n"
            text += _table(
                ["阈值", "阳性 case", "阳性 case 3D Dice", "中位 Dice", "微平均 Dice",
                 "precision", "recall", "FP case 率", "阴性 case 平均 FP 体素"], rows)

    p = out_dir / "phase2a_volume.md"
    p.write_text(text, encoding="utf-8")
    written.append(p)

    # lesion 表
    parts = ["# Phase 2A：lesion-wise（病灶级）结果\n",
             "> 3D 连通域（26-连通），GT 每个连通域计一个 lesion，支持 multi-lesion case。\n",
             "> 匹配规则：一个 GT 只允许一次检出；同一 GT 上重叠较小的其余预测连通域计为 FP component。\n"]
    for name, payload in details:
        heading = f"\n## {name}\n"
        parts.append(heading)
        rule = payload.get("matching_rule", {})
        parts.append(f"\n匹配规则：`{rule.get('criterion')}`（阈值 {rule.get('criterion_threshold')}，"
                     f"{rule.get('connectivity')}-连通）\n\n")
        les = (payload.get("lesion_summary") or {}).get("per_threshold") or []
        if les:
            rows = [[_fmt(r["threshold"], 2), _fmt(r["num_gt_lesions"], 0),
                     _fmt(r["num_pred_lesions"], 0), _fmt(r["num_detected_gt_lesions"], 0),
                     _fmt(r["lesion_sensitivity_pooled"]), _fmt(r["lesion_sensitivity_per_case_mean"]),
                     _fmt(r["false_positive_lesions"], 0), _fmt(r["fp_lesions_per_case"], 2),
                     _fmt(r["matched_lesion_dice"])] for r in les]
            parts.append(_table(
                ["阈值", "GT 病灶", "预测连通域", "检出 GT", "灵敏度(pooled)",
                 "灵敏度(case 均值)", "FP 连通域", "FP/case", "匹配病灶 Dice"], rows))

        strat = payload.get("small_lesion_stratification") or []
        if strat:
            parts.append("\n按 GT 病灶体积分层（探索性分组，非临床标准）：\n\n")
            rows = [[_fmt(r["threshold"], 2), r["bin"], _fmt(r["num_lesions"], 0),
                     _fmt(r["num_detected"], 0), _fmt(r["detection_sensitivity"]),
                     _fmt(r["mean_dice_all_lesions"]), _fmt(r["mean_dice_detected_only"])]
                    for r in strat]
            parts.append(_table(
                ["阈值", "体积分层", "病灶数", "检出数", "检出灵敏度",
                 "匹配 Dice(全部)", "匹配 Dice(仅检出)"], rows))

        filt = payload.get("component_filtering") or []
        if filt:
            parts.append("\n预测连通域最小体积过滤扫描：\n\n")
            rows = [[_fmt(r["threshold"], 2), _fmt(r["min_pred_volume_mm3"], 1),
                     _fmt(r["num_pred_lesions"], 0), _fmt(r["lesion_sensitivity_pooled"]),
                     _fmt(r["false_positive_lesions"], 0), _fmt(r["fp_lesions_per_case"], 2),
                     _fmt(r["matched_lesion_dice"])] for r in filt]
            parts.append(_table(
                ["阈值", "最小体积(mm³)", "剩余预测连通域", "灵敏度(pooled)",
                 "FP 连通域", "FP/case", "匹配病灶 Dice"], rows))

    p = out_dir / "phase2a_lesion.md"
    p.write_text("\n".join(parts), encoding="utf-8")
    written.append(p)
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    """入口。"""
    parser = argparse.ArgumentParser(description="汇总 Phase 2A 结果为 Markdown 表格")
    parser.add_argument("--scan-dir", type=Path,
                        default=Path("outputs/evaluation/threshold_scan"))
    parser.add_argument("--volume-dir", type=Path, default=Path("outputs/evaluation/volume"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/experiments/tables"))
    args = parser.parse_args(argv)

    scan_dir = PROJECT_ROOT / args.scan_dir if not args.scan_dir.is_absolute() else args.scan_dir
    volume_dir = PROJECT_ROOT / args.volume_dir if not args.volume_dir.is_absolute() else args.volume_dir
    out_dir = PROJECT_ROOT / args.out_dir if not args.out_dir.is_absolute() else args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    for step, fn in tqdm([("threshold", lambda: build_threshold_tables(scan_dir, out_dir)),
                          ("volume", lambda: build_volume_tables(volume_dir, out_dir))],
                         desc="汇总", unit="step", ncols=100, file=sys.stdout):
        written.extend(fn())
    for p in written:
        print(f"已写出 {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
