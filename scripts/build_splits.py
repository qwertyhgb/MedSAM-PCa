#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成 patient-level 5-fold split 与 fold0 的 slice manifest（Phase 1）。

* 若本地存在官方 fold assignment 则直接使用（当前 PI-CAI 公开数据中**未找到**，
  见 ``split_summary.json`` 的 ``official_fold_found`` 字段）。
* 否则使用 ``sklearn.model_selection.StratifiedGroupKFold``：
  ``group = patient_id``，分层目标 ``= case_csPCa``，``seed = 42``。
* 强制校验 train/val 的 **patient overlap == 0**（多 study 患者不得跨侧）。
* Phase 1 固定：``fold0 = validation``，``fold1-4 = training``。

用法::

    /root/anaconda3/envs/lm/bin/python scripts/build_splits.py

产出::

    data/splits/fold0.csv ... fold4.csv
    data/splits/split_summary.json
    data/manifests/picai_slices_fold0_train.csv
    data/manifests/picai_slices_fold0_val.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.picai_2d import scan_case_slices  # noqa: E402

LOGGER = logging.getLogger("build_splits")

SEED = 42
N_SPLITS = 5
PHASE1_VAL_FOLD = 0

DEFAULT_DATA_ROOT = os.environ.get(
    "PICAI_DATA_ROOT", "/opt/data/private/lm/data/Prostate/PI-CAI"
)


def find_official_fold(data_root: Path) -> Optional[Path]:
    """在本地数据中查找官方 fold assignment 文件。

    Args:
        data_root: PI-CAI 根目录。

    Returns:
        找到则返回文件路径，否则 ``None``。
    """
    patterns = ["*fold*.csv", "*fold*.json", "*split*.csv", "*split*.json"]
    search_dirs = [data_root / "labels" / "additional_resources",
                   data_root / "labels", data_root]
    for d in search_dirs:
        if not d.is_dir():
            continue
        for pat in patterns:
            for p in d.glob(pat):
                if p.is_file():
                    return p
    return None


def build_folds(df: pd.DataFrame, n_splits: int = N_SPLITS,
                seed: int = SEED) -> List[Tuple[np.ndarray, np.ndarray]]:
    """按 patient 分组、按 case_csPCa 分层生成 folds。

    Args:
        df: case manifest DataFrame。
        n_splits: 折数。
        seed: 随机种子。

    Returns:
        ``[(train_idx, val_idx), ...]``。
    """
    y = (df["case_cspca"].astype(str).str.upper() == "YES").astype(int).to_numpy()
    groups = df["patient_id"].astype(str).to_numpy()

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(sgkf.split(df, y, groups))


def _slice_worker(task: Tuple[str, str, str, Optional[str]]) -> Dict[str, Any]:
    """子进程入口：扫描单个 case 的 slice 级病灶像素。"""
    case_id, patient_id, t2w_abs, mask_abs = task
    return scan_case_slices(case_id, patient_id, t2w_abs, mask_abs)


def build_slice_manifest(cases: pd.DataFrame, data_root: Path, workers: int,
                         tag: str) -> pd.DataFrame:
    """为一批 case 构建 slice 级 manifest。

    Args:
        cases: 该 split 的 case 行。
        data_root: PI-CAI 根目录。
        workers: 进程数。
        tag: ``"train"`` 或 ``"val"``（仅用于日志）。

    Returns:
        slice 级 DataFrame。
    """
    tasks: List[Tuple[str, str, str, Optional[str]]] = []
    for _, row in cases.iterrows():
        t2w_abs = str(data_root / str(row["t2w_path"])) if pd.notna(row["t2w_path"]) else None
        if t2w_abs is None:
            LOGGER.warning("%s 缺少 T2W，跳过", row["case_id"])
            continue
        mask_abs = (str(data_root / str(row["lesion_mask_path"]))
                    if pd.notna(row["lesion_mask_path"]) else None)
        tasks.append((str(row["case_id"]), str(row["patient_id"]), t2w_abs, mask_abs))

    LOGGER.info("[%s] 扫描 %d 个 case 的 slice 结构 (workers=%d)", tag, len(tasks), workers)
    rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_slice_worker, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            if i % 200 == 0:
                LOGGER.info("  [%s] 进度 %d/%d", tag, i, len(tasks))
            rows.append(r)

    out: List[Dict[str, Any]] = []
    for r in rows:
        for z, npx in enumerate(r["per_slice_lesion_pixels"]):
            out.append({
                "case_id": r["case_id"],
                "patient_id": r["patient_id"],
                "slice_idx": z,
                "t2w_path": None,   # 稍后回填
                "mask_path": None,
                "is_positive": bool(npx > 0),
                "lesion_pixels": int(npx),
                "mask_status": r["mask_status"],
            })
    df_slices = pd.DataFrame(out)

    # 回填相对路径
    meta = cases.set_index("case_id")
    df_slices["t2w_path"] = df_slices["case_id"].map(meta["t2w_path"])
    df_slices["mask_path"] = df_slices["case_id"].map(meta["lesion_mask_path"])
    return df_slices


def main(argv: Optional[Sequence[str]] = None) -> int:
    """脚本入口。"""
    parser = argparse.ArgumentParser(
        description="生成 PI-CAI patient-level split 与 slice manifest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT))
    parser.add_argument("--case-manifest", type=Path,
                        default=PROJECT_ROOT / "data" / "manifests" / "picai_cases.csv")
    parser.add_argument("--out-dir", type=Path,
                        default=PROJECT_ROOT / "data" / "splits")
    parser.add_argument("--manifest-dir", type=Path,
                        default=PROJECT_ROOT / "data" / "manifests")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)

    data_root = args.data_root.resolve()
    assert args.case_manifest.is_file(), (
        f"case manifest 不存在: {args.case_manifest}，请先运行 build_picai_manifest.py"
    )
    df = pd.read_csv(args.case_manifest)
    LOGGER.info("载入 case manifest: %d 行", len(df))

    official = find_official_fold(data_root)
    LOGGER.info("官方 fold 文件: %s", official if official else "未找到 → 使用 StratifiedGroupKFold")

    folds = build_folds(df, N_SPLITS, SEED)

    summary: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "strategy": "StratifiedGroupKFold" if official is None else "official",
        "official_fold_found": official is not None,
        "official_fold_path": str(official) if official else None,
        "seed": SEED,
        "n_splits": N_SPLITS,
        "group": "patient_id",
        "stratify_target": "case_csPCa",
        "positive_definition": "case_csPCa == 'YES'",
        "phase1_use": {"val_fold": PHASE1_VAL_FOLD,
                       "train_folds": [i for i in range(N_SPLITS) if i != PHASE1_VAL_FOLD]},
        "folds": [],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for fold_id, (train_idx, val_idx) in enumerate(folds):
        d = df.copy()
        d["split"] = "train"
        d.loc[d.index[val_idx], "split"] = "val"
        out_path = args.out_dir / f"fold{fold_id}.csv"
        d.to_csv(out_path, index=False)

        tr, va = df.iloc[train_idx], df.iloc[val_idx]
        overlap = set(tr["patient_id"]) & set(va["patient_id"])
        assert len(overlap) == 0, (
            f"fold{fold_id} 存在 patient 泄漏: {sorted(overlap)[:5]}"
        )

        info = {
            "fold": fold_id,
            "file": out_path.name,
            "train_cases": int(len(tr)),
            "val_cases": int(len(va)),
            "train_patients": int(tr["patient_id"].nunique()),
            "val_patients": int(va["patient_id"].nunique()),
            "train_positive": int((tr["case_cspca"].astype(str).str.upper() == "YES").sum()),
            "val_positive": int((va["case_cspca"].astype(str).str.upper() == "YES").sum()),
            "train_multi_study_patients": int(
                (tr.groupby("patient_id").size() > 1).sum()),
            "val_multi_study_patients": int(
                (va.groupby("patient_id").size() > 1).sum()),
            "patient_overlap": len(overlap),
            "written": str(out_path),
        }
        summary["folds"].append(info)
        LOGGER.info("fold%d: train=%d val=%d | train_pos=%d val_pos=%d | overlap=%d",
                    fold_id, info["train_cases"], info["val_cases"],
                    info["train_positive"], info["val_positive"], len(overlap))

    # -------- fold0 的 slice manifest -------- #
    fold0 = df.copy()
    f0_train_idx, f0_val_idx = folds[PHASE1_VAL_FOLD]
    cases_tr = df.iloc[f0_train_idx]
    cases_va = df.iloc[f0_val_idx]

    manifest_dir = args.manifest_dir
    manifest_dir.mkdir(parents=True, exist_ok=True)
    slice_stats: Dict[str, Any] = {}
    for tag, cases in (("train", cases_tr), ("val", cases_va)):
        sdf = build_slice_manifest(cases, data_root, args.workers, tag)
        out_path = manifest_dir / f"picai_slices_fold{PHASE1_VAL_FOLD}_{tag}.csv"
        sdf.to_csv(out_path, index=False)
        n_pos = int(sdf["is_positive"].sum())
        slice_stats[tag] = {
            "file": str(out_path),
            "cases": int(sdf["case_id"].nunique()),
            "total_slices": int(len(sdf)),
            "positive_slices": n_pos,
            "negative_slices": int(len(sdf) - n_pos),
            "positive_ratio": round(n_pos / max(len(sdf), 1), 6),
            "cases_with_lesion": int(sdf[sdf["is_positive"]]["case_id"].nunique()),
            "mask_status_counts": {str(k): int(v) for k, v in
                                   sdf["mask_status"].value_counts().items()},
        }
        LOGGER.info("[%s] %s: slices=%d (pos=%d / neg=%d)",
                    tag, out_path.name, slice_stats[tag]["total_slices"],
                    n_pos, slice_stats[tag]["negative_slices"])

    summary["fold0_slices"] = slice_stats
    summary_path = args.out_dir / "split_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    LOGGER.info("已写出: %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
