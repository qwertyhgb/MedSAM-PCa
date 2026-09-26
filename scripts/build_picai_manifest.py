#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""构建 PI-CAI case-level manifest，并校验 lesion mask 与 T2W 的几何对齐（Phase 1）。

规则
----
1. lesion mask 来源优先级：
   ``labels/csPCa_lesion_delineations/human_expert/resampled`` (1295)
   → ``labels/csPCa_lesion_delineations/human_expert/Pooch25``   (205)
2. 所有 mask 一律按 ``mask > 0`` 二值化；**禁止** ``mask == 1``
   （resampled/original 使用 2/3/4/5 表示 ISUP grade）。
3. 只读，不修改任何原始标注。

用法::

    /root/anaconda3/envs/lm/bin/python scripts/build_picai_manifest.py

产出::

    data/manifests/picai_cases.csv
    outputs/data_audit/pooch25_alignment.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

LOGGER = logging.getLogger("build_manifest")
PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_FILE_RE = re.compile(
    r"^(?P<pid>\d+)_(?P<sid>\d+)_(?P<mod>[A-Za-z0-9]+)\.(?P<ext>mha|nii\.gz|nii)$",
    re.IGNORECASE,
)

REL_RESAMPLED = Path("csPCa_lesion_delineations/human_expert/resampled")
REL_POOCH25 = Path("csPCa_lesion_delineations/human_expert/Pooch25")

LABEL_SOURCE_RESAMPLED = "human_expert_resampled"
LABEL_SOURCE_POOCH25 = "human_expert_pooch25"
LABEL_SOURCE_MISSING = "missing"

MODALITIES = ("t2w", "adc", "hbv")
GEOM_TOLERANCE = 1e-4

DEFAULT_DATA_ROOT = os.environ.get(
    "PICAI_DATA_ROOT", "/opt/data/private/lm/data/Prostate/PI-CAI"
)


# --------------------------------------------------------------------------- #
# 扫描
# --------------------------------------------------------------------------- #
def scan_cases(images_root: Path) -> Dict[str, Dict[str, Any]]:
    """扫描 images 目录，按 ``{patient_id}_{study_id}`` 聚合模态路径。

    Args:
        images_root: PI-CAI ``images`` 目录。

    Returns:
        ``{case_id: {"patient_id", "study_id", "modalities": {mod: rel_path}}}``。
    """
    cases: Dict[str, Dict[str, Any]] = {}
    for case_dir in sorted(p for p in images_root.iterdir() if p.is_dir()):
        for f in sorted(case_dir.iterdir()):
            if not f.is_file():
                continue
            m = IMAGE_FILE_RE.match(f.name)
            if m is None:
                continue
            case_id = f"{m.group('pid')}_{m.group('sid')}"
            entry = cases.setdefault(
                case_id,
                {"patient_id": m.group("pid"), "study_id": m.group("sid"),
                 "modalities": {}},
            )
            entry["modalities"][m.group("mod").lower()] = str(
                f.relative_to(images_root.parent)
            )
    return cases


def resolve_mask(case_id: str, labels_root: Path) -> Tuple[Optional[Path], str]:
    """按优先级为 case 解析 human-expert lesion mask。

    Args:
        case_id: ``{patient_id}_{study_id}``。
        labels_root: PI-CAI ``labels`` 目录。

    Returns:
        ``(mask_path 或 None, label_source)``。
    """
    resampled = labels_root / REL_RESAMPLED / f"{case_id}.nii.gz"
    if resampled.is_file():
        return resampled, LABEL_SOURCE_RESAMPLED
    pooch = labels_root / REL_POOCH25 / f"{case_id}.nii.gz"
    if pooch.is_file():
        return pooch, LABEL_SOURCE_POOCH25
    return None, LABEL_SOURCE_MISSING


# --------------------------------------------------------------------------- #
# mask 分析（子进程）
# --------------------------------------------------------------------------- #
def _mask_worker(task: Tuple[str, str, str]) -> Dict[str, Any]:
    """读取单个 mask：几何信息 + 是否非空 + 连通域（病灶）数量。

    Args:
        task: ``(case_id, mask_path, label_source)``。

    Returns:
        统计字典；``ok=False`` 时携带 ``error``。
    """
    import numpy as np
    import SimpleITK as sitk
    from scipy import ndimage

    case_id, mask_path, label_source = task
    out: Dict[str, Any] = {"case_id": case_id, "mask_path": mask_path,
                           "label_source": label_source, "ok": False, "error": None}
    try:
        img = sitk.ReadImage(mask_path)
        arr = sitk.GetArrayFromImage(img)
        fg = arr > 0  # 关键：绝不能写成 arr == 1
        out.update(
            {
                "ok": True,
                "has_lesion": bool(fg.any()),
                "num_lesions": int(ndimage.label(fg)[1]) if fg.any() else 0,
                "lesion_voxels": int(fg.sum()),
                "label_values": sorted(int(v) for v in np.unique(arr)),
                "geometry": {
                    "size": [int(v) for v in img.GetSize()],
                    "spacing": [float(v) for v in img.GetSpacing()],
                    "origin": [float(v) for v in img.GetOrigin()],
                    "direction": [float(v) for v in img.GetDirection()],
                },
            }
        )
        del arr, fg
    except Exception as exc:  # noqa: BLE001 - 显式记录
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def _geometry_worker(case_id: str, path: str, role: str) -> Dict[str, Any]:
    """只读 header 获取几何信息。"""
    import SimpleITK as sitk

    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    reader.ReadImageInformation()
    return {
        "case_id": case_id,
        "role": role,
        "geometry": {
            "size": [int(v) for v in reader.GetSize()],
            "spacing": [float(v) for v in reader.GetSpacing()],
            "origin": [float(v) for v in reader.GetOrigin()],
            "direction": [float(v) for v in reader.GetDirection()],
        },
    }


def geometry_diff(a: Dict[str, Any], b: Dict[str, Any],
                  tol: float = GEOM_TOLERANCE) -> List[str]:
    """比较两组几何参数，返回不一致的字段名列表。

    Args:
        a: 第一个几何字典。
        b: 第二个几何字典。
        tol: 数值容差。

    Returns:
        不一致字段名（``size`` / ``spacing`` / ``origin`` / ``direction``）。
    """
    diffs: List[str] = []
    if list(a["size"]) != list(b["size"]):
        diffs.append("size")
    for field in ("spacing", "origin", "direction"):
        va, vb = list(a[field]), list(b[field])
        if len(va) != len(vb) or any(abs(x - y) > tol for x, y in zip(va, vb)):
            diffs.append(field)
    return diffs


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def build(
    data_root: Path,
    out_csv: Path,
    out_alignment: Path,
    workers: int,
) -> Dict[str, Any]:
    """构建 manifest 并做几何对齐校验。"""
    images_root = data_root / "images"
    labels_root = data_root / "labels"
    assert images_root.is_dir(), f"images 目录不存在: {images_root}"
    assert labels_root.is_dir(), f"labels 目录不存在: {labels_root}"

    LOGGER.info("扫描影像 ...")
    cases = scan_cases(images_root)
    LOGGER.info("  -> %d 个 case", len(cases))

    marksheet = labels_root / "clinical_information" / "marksheet.csv"
    assert marksheet.is_file(), f"marksheet 不存在: {marksheet}"
    df_clin = pd.read_csv(marksheet)
    cspca_map: Dict[str, str] = {}
    for _, row in df_clin.iterrows():
        key = f"{row['patient_id']}_{row['study_id']}"
        cspca_map[key] = str(row["case_csPCa"]).strip().upper()

    # -------- mask 解析 -------- #
    tasks: List[Tuple[str, str, str]] = []
    mask_source: Dict[str, str] = {}
    missing: List[str] = []
    for case_id in sorted(cases):
        path, source = resolve_mask(case_id, labels_root)
        mask_source[case_id] = source
        if path is None:
            missing.append(case_id)
        else:
            tasks.append((case_id, str(path), source))

    LOGGER.info("分析 %d 个 lesion mask (workers=%d) ...", len(tasks), workers)
    mask_info: Dict[str, Dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_mask_worker, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            mask_info[r["case_id"]] = r
            if i % 400 == 0:
                LOGGER.info("  mask 进度 %d/%d", i, len(tasks))

    failed = [r for r in mask_info.values() if not r["ok"]]
    for r in failed:
        LOGGER.error("mask 读取失败 %s: %s", r["case_id"], r["error"])

    # -------- T2W header -------- #
    LOGGER.info("读取 T2W header ...")
    t2w_tasks = [(cid, str(data_root / cases[cid]["modalities"]["t2w"]), "t2w")
                 for cid in sorted(cases) if "t2w" in cases[cid]["modalities"]]
    t2w_geom: Dict[str, Dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_geometry_worker, *t) for t in t2w_tasks]
        for fut in as_completed(futures):
            r = fut.result()
            t2w_geom[r["case_id"]] = r["geometry"]

    # -------- 组装 CSV -------- #
    rows: List[Dict[str, Any]] = []
    for case_id in sorted(cases):
        info = cases[case_id]
        mods = info["modalities"]
        mi = mask_info.get(case_id, {})
        rows.append(
            {
                "patient_id": info["patient_id"],
                "study_id": info["study_id"],
                "case_id": case_id,
                "t2w_path": mods.get("t2w"),
                "adc_path": mods.get("adc"),
                "hbv_path": mods.get("hbv"),
                "lesion_mask_path": mi.get("mask_path"),
                "label_source": mask_source[case_id],
                "case_cspca": cspca_map.get(case_id, "UNKNOWN"),
                "has_lesion": bool(mi.get("has_lesion", False)),
                "num_lesions": int(mi.get("num_lesions", 0)),
            }
        )

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    LOGGER.info("已写出 manifest: %s (%d 行)", out_csv, len(df))

    # -------- 几何对齐校验 -------- #
    alignment: Dict[str, Any] = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "tolerance": GEOM_TOLERANCE,
        "fields": ["size", "spacing", "origin", "direction"],
        "reference": "T2W",
    }
    for source in (LABEL_SOURCE_POOCH25, LABEL_SOURCE_RESAMPLED):
        sub = [cid for cid in sorted(mask_info)
               if mask_info[cid]["label_source"] == source and mask_info[cid]["ok"]]
        aligned, not_aligned, no_t2w = [], [], []
        details: List[Dict[str, Any]] = []
        for cid in sub:
            if cid not in t2w_geom:
                no_t2w.append(cid)
                continue
            diffs = geometry_diff(mask_info[cid]["geometry"], t2w_geom[cid])
            if diffs:
                not_aligned.append(cid)
                details.append({
                    "case_id": cid,
                    "differing_fields": diffs,
                    "mask_geometry": mask_info[cid]["geometry"],
                    "t2w_geometry": t2w_geom[cid],
                })
            else:
                aligned.append(cid)
        alignment[source] = {
            "num_masks": len(sub),
            "exactly_aligned": len(aligned),
            "not_aligned": len(not_aligned),
            "missing_t2w": no_t2w,
            "resampling_required": len(not_aligned) > 0,
            "not_aligned_details": details[:50],
        }

    out_alignment.parent.mkdir(parents=True, exist_ok=True)
    with out_alignment.open("w", encoding="utf-8") as fh:
        json.dump(alignment, fh, ensure_ascii=False, indent=2)
    LOGGER.info("已写出对齐校验: %s", out_alignment)

    summary = {
        "num_cases": len(df),
        "num_patients": int(df["patient_id"].nunique()),
        "label_source_counts": dict(df["label_source"].value_counts()),
        "case_cspca_counts": dict(df["case_cspca"].value_counts()),
        "has_lesion_counts": {str(k): int(v) for k, v in df["has_lesion"].value_counts().items()},
        "num_multi_lesion_cases": int((df["num_lesions"] > 1).sum()),
        "missing_mask_cases": missing,
        "masks_failed": [r["case_id"] for r in failed],
    }
    LOGGER.info("摘要: %s", json.dumps(summary, ensure_ascii=False, default=str))
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    """脚本入口。"""
    parser = argparse.ArgumentParser(
        description="构建 PI-CAI case manifest 并校验几何对齐",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT))
    parser.add_argument("--out-csv", type=Path,
                        default=PROJECT_ROOT / "data" / "manifests" / "picai_cases.csv")
    parser.add_argument("--out-alignment", type=Path,
                        default=PROJECT_ROOT / "outputs" / "data_audit" / "pooch25_alignment.json")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)

    build(args.data_root.resolve(), args.out_csv, args.out_alignment,
          max(1, args.workers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
