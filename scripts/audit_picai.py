#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PI-CAI 数据集只读审计脚本（Phase 0 / Task C）。

用法::

    /root/anaconda3/envs/lm/bin/python scripts/audit_picai.py \\
        --data-root /opt/data/private/lm/data/Prostate/PI-CAI \\
        --out-dir outputs/data_audit \\
        --mask-workers 6

产出::

    outputs/data_audit/picai_audit.json
    outputs/data_audit/picai_audit.txt

设计约束（见 PROJECT_CONSTRAINTS.md）:
    * 只读，绝不修改原始数据；
    * 不把全部 volume 载入内存 —— 逐个文件读取并立即释放；
    * 优先通过文件名 / 目录结构统计，而非全量读像素；
    * 不做 registration / resampling / OCR；
    * 不硬编码假设：目录布局通过实际遍历得到。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

LOGGER = logging.getLogger("audit_picai")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: 影像文件名形如 ``10000_1000000_t2w.mha``
IMAGE_FILE_RE = re.compile(
    r"^(?P<pid>\d+)_(?P<sid>\d+)_(?P<mod>[A-Za-z0-9]+)\.(?P<ext>mha|nii\.gz|nii|nrrd|npz|npy)$",
    re.IGNORECASE,
)
#: 标注文件名形如 ``10000_1000000.nii.gz``
LABEL_FILE_RE = re.compile(
    r"^(?P<pid>\d+)_(?P<sid>\d+)\.(?P<ext>mha|nii\.gz|nii|nrrd|npz|npy)$",
    re.IGNORECASE,
)

#: 关注的 MRI 模态（其余如 cor/sag 属于附属扫描，仅统计）
PRIMARY_MODALITIES: Tuple[str, ...] = ("t2w", "adc", "hbv")
#: hbv 即 DWI 的高 b 值图（PI-CAI 命名）
MODALITY_ALIASES: Dict[str, Tuple[str, ...]] = {"hbv": ("hbv", "dwi")}

DEFAULT_DATA_ROOT = os.environ.get(
    "PICAI_DATA_ROOT", "/opt/data/private/lm/data/Prostate/PI-CAI"
)


# --------------------------------------------------------------------------- #
# 影像树扫描
# --------------------------------------------------------------------------- #
def scan_image_tree(images_root: Path) -> Dict[str, Any]:
    """扫描影像目录，按 ``{patient_id}_{study_id}`` 聚合模态信息。

    Args:
        images_root: PI-CAI ``images`` 目录。

    Returns:
        含 cases / 统计 / 未解析文件 的字典。
    """
    if not images_root.is_dir():
        raise FileNotFoundError(f"images 目录不存在: {images_root}")

    cases: Dict[str, Dict[str, Any]] = {}
    unparsed: List[str] = []
    ext_counter: Counter = Counter()
    modality_counter: Counter = Counter()
    patient_dirs = 0
    non_dir_entries: List[str] = []

    for case_dir in sorted(images_root.iterdir()):
        if not case_dir.is_dir():
            non_dir_entries.append(case_dir.name)
            continue
        patient_dirs += 1
        for f in sorted(case_dir.iterdir()):
            if not f.is_file():
                continue
            m = IMAGE_FILE_RE.match(f.name)
            if m is None:
                unparsed.append(str(f.relative_to(images_root)))
                continue

            key = f"{m.group('pid')}_{m.group('sid')}"
            mod = m.group("mod").lower()
            ext_counter[m.group("ext").lower()] += 1
            modality_counter[mod] += 1

            entry = cases.setdefault(
                key,
                {
                    "case_key": key,
                    "patient_id": m.group("pid"),
                    "study_id": m.group("sid"),
                    "parent_dir": case_dir.name,
                    "modalities": [],
                    "files": [],
                },
            )
            entry["modalities"].append(mod)
            entry["files"].append(f.name)

    # 一个 patient 目录下出现多个 study 的情况需要显式记录
    studies_per_patient: Counter = Counter()
    for c in cases.values():
        studies_per_patient[c["patient_id"]] += 1
    multi_study_patients = sorted(
        {pid for pid, n in studies_per_patient.items() if n > 1}
    )

    return {
        "root": str(images_root),
        "patient_dir_count": patient_dirs,
        "non_dir_entries": non_dir_entries,
        "num_cases": len(cases),
        "num_patients": len(studies_per_patient),
        "multi_study_patients": multi_study_patients,
        "modality_file_counts": dict(sorted(modality_counter.items())),
        "file_extensions": dict(sorted(ext_counter.items())),
        "unparsed_files": unparsed,
        "cases": cases,
    }


# --------------------------------------------------------------------------- #
# 标注树扫描
# --------------------------------------------------------------------------- #
def scan_label_tree(labels_root: Path) -> Dict[str, Any]:
    """扫描标注目录，按相对子目录分组统计（结构由实际遍历决定）。

    Args:
        labels_root: PI-CAI ``labels`` 目录。

    Returns:
        分组统计字典。
    """
    if not labels_root.is_dir():
        raise FileNotFoundError(f"labels 目录不存在: {labels_root}")

    groups: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"num_files": 0, "extensions": Counter(), "case_keys": set(),
                 "unparsed": []}
    )
    skipped_dirs = {".git"}

    for dirpath, dirnames, filenames in os.walk(labels_root):
        dirnames[:] = [d for d in dirnames if d not in skipped_dirs]
        if not filenames:
            continue
        rel = Path(dirpath).relative_to(labels_root)
        group_key = str(rel) if str(rel) != "." else "<root>"

        for name in sorted(filenames):
            if name.startswith("."):
                continue
            g = groups[group_key]
            m = LABEL_FILE_RE.match(name)
            if m is None:
                g["unparsed"].append(name)
                continue
            g["num_files"] += 1
            g["extensions"][Path(name).suffix.lower()] += 1
            g["case_keys"].add(f"{m.group('pid')}_{m.group('sid')}")

    out: Dict[str, Any] = {}
    for key, g in sorted(groups.items()):
        out[key] = {
            "num_files": g["num_files"],
            "extensions": dict(sorted(g["extensions"].items())),
            "unique_case_keys": len(g["case_keys"]),
            "unparsed_files": g["unparsed"][:10],
        }
    return {"root": str(labels_root), "groups": out,
            "case_keys_by_group": {k: sorted(v["case_keys"]) for k, v in groups.items()}}


# --------------------------------------------------------------------------- #
# mask 深度分析（子进程 worker）
# --------------------------------------------------------------------------- #
def _analyze_mask_worker(path_str: str) -> Dict[str, Any]:
    """分析单个 mask：前景体素数、label 取值、连通域数量。

    该函数运行在子进程中，必须是模块级可 pickle 函数。
    读取后立即释放数组，单个 worker 内存占用约为单个体积大小。

    Args:
        path_str: mask 文件绝对路径。

    Returns:
        统计结果字典；失败时 ``ok=False`` 并携带 ``error``。
    """
    # 子进程中导入，避免主进程无关开销
    import numpy as np
    import SimpleITK as sitk
    from scipy import ndimage

    res: Dict[str, Any] = {"path": path_str, "ok": False, "error": None}
    try:
        img = sitk.ReadImage(path_str)
        arr = sitk.GetArrayFromImage(img)
        fg = arr > 0
        n_fg = int(fg.sum())

        spacing = tuple(round(float(s), 4) for s in img.GetSpacing())
        res.update(
            {
                "ok": True,
                "shape": [int(v) for v in arr.shape],
                "spacing": list(spacing),
                "n_foreground_voxels": n_fg,
                "label_values": [int(v) for v in np.unique(arr)],
                "n_components_binary": int(ndimage.label(fg)[1]) if n_fg else 0,
                "n_components_by_value": {},
            }
        )
        if n_fg:
            by_value: Dict[str, int] = {}
            for v in np.unique(arr[fg]):
                by_value[str(int(v))] = int(ndimage.label(arr == v)[1])
            res["n_components_by_value"] = by_value
        del arr, fg
    except Exception as exc:  # noqa: BLE001 - 显式记录，不静默吞掉
        res["error"] = f"{type(exc).__name__}: {exc}"
    return res


def analyze_masks(
    paths: Sequence[Path], workers: int, log_every: int = 200
) -> Dict[str, Any]:
    """并行分析一批 mask 文件。

    Args:
        paths: mask 路径列表。
        workers: 进程数。
        log_every: 每处理多少个文件打印一次进度。

    Returns:
        汇总统计。
    """
    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    if not paths:
        return {"num_masks": 0, "results": [], "errors": []}

    if workers <= 1:
        iterator = (_analyze_mask_worker(str(p)) for p in paths)
        for i, r in enumerate(iterator, 1):
            results.append(r)
            if i % log_every == 0:
                LOGGER.info("  mask 分析进度 %d/%d", i, len(paths))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_analyze_mask_worker, str(p)): p for p in paths}
            for i, fut in enumerate(as_completed(futures), 1):
                results.append(fut.result())
                if i % log_every == 0:
                    LOGGER.info("  mask 分析进度 %d/%d", i, len(paths))

    for r in results:
        if not r["ok"]:
            errors.append({"path": r["path"], "error": str(r["error"])})

    ok = [r for r in results if r["ok"]]
    non_empty = [r for r in ok if r["n_foreground_voxels"] > 0]

    spacing_counter: Counter = Counter(tuple(r["spacing"]) for r in ok)
    shape_counter: Counter = Counter(tuple(r["shape"]) for r in ok)
    label_value_counter: Counter = Counter()
    for r in ok:
        for v in r["label_values"]:
            label_value_counter[str(v)] += 1

    comp_hist: Counter = Counter(r["n_components_binary"] for r in non_empty)

    return {
        "num_masks": len(paths),
        "num_ok": len(ok),
        "num_failed": len(errors),
        "num_non_empty": len(non_empty),
        "num_empty": len(ok) - len(non_empty),
        "label_value_distribution": dict(sorted(label_value_counter.items(),
                                                 key=lambda kv: int(kv[0]))),
        "spacing_distribution": {"x".join(map(str, k)): v
                                 for k, v in spacing_counter.most_common(20)},
        "shape_distribution": {"x".join(map(str, k)): v
                               for k, v in shape_counter.most_common(20)},
        "component_histogram": dict(sorted(comp_hist.items())),
        "multi_lesion_cases": sorted(
            Path(r["path"]).name for r in non_empty if r["n_components_binary"] > 1
        ),
        "empty_cases": sorted(Path(r["path"]).name for r in ok
                              if r["n_foreground_voxels"] == 0),
        "errors": errors,
    }


# --------------------------------------------------------------------------- #
# header 抽样
# --------------------------------------------------------------------------- #
def sample_headers(
    images_root: Path, case_keys: Sequence[str], modalities: Sequence[str],
    per_modality: int,
) -> List[Dict[str, Any]]:
    """对少量 case 读取 header（不读像素），记录几何信息。

    Args:
        images_root: images 根目录。
        case_keys: 候选 case key。
        modalities: 关心的模态。
        per_modality: 每个模态抽取数量。

    Returns:
        每个抽样文件一条记录。
    """
    import SimpleITK as sitk

    reader = sitk.ImageFileReader()
    out: List[Dict[str, Any]] = []
    counts: Counter = Counter()

    for key in case_keys:
        if all(counts[m] >= per_modality for m in modalities):
            break
        pid = key.split("_")[0]
        case_dir = images_root / pid
        if not case_dir.is_dir():
            continue
        for mod in modalities:
            if counts[mod] >= per_modality:
                continue
            matches = sorted(case_dir.glob(f"{key}_*"))
            target = [p for p in matches
                      if p.name.rsplit(".", 1)[0].lower().endswith(f"_{mod}")]
            if not target:
                continue
            reader.SetFileName(str(target[0]))
            reader.ReadImageInformation()
            out.append(
                {
                    "case_key": key,
                    "modality": mod,
                    "file": target[0].name,
                    "size": [int(v) for v in reader.GetSize()],
                    "spacing": [round(float(v), 4) for v in reader.GetSpacing()],
                    "origin": [round(float(v), 4) for v in reader.GetOrigin()],
                    "direction": [round(float(v), 4) for v in reader.GetDirection()],
                    "pixel_type": sitk.GetPixelIDValueAsString(reader.GetPixelID()),
                }
            )
            counts[mod] += 1
    return out


# --------------------------------------------------------------------------- #
# 临床信息
# --------------------------------------------------------------------------- #
def read_clinical(marksheet: Path) -> Dict[str, Any]:
    """读取 marksheet.csv（仅汇总，不复制原始数据到项目目录）。

    Args:
        marksheet: marksheet.csv 路径。

    Returns:
        列名、行数与关键字段分布。
    """
    import pandas as pd

    if not marksheet.is_file():
        return {"exists": False, "path": str(marksheet)}

    df = pd.read_csv(marksheet)
    info: Dict[str, Any] = {
        "exists": True,
        "path": str(marksheet),
        "num_rows": int(len(df)),
        "columns": list(df.columns),
    }
    for col in ("case_csPCa", "center", "histopath_type", "lesion_ISUP", "case_ISUP"):
        if col in df.columns:
            info[f"{col}_distribution"] = {
                str(k): int(v) for k, v in df[col].value_counts(dropna=False).items()
            }
    if {"patient_id", "study_id"}.issubset(df.columns):
        keys = (df["patient_id"].astype(str) + "_" + df["study_id"].astype(str))
        info["num_unique_case_keys"] = int(keys.nunique())
    return info


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def build_report(
    data_root: Path,
    mask_workers: int,
    header_samples: int,
    skip_mask_analysis: bool,
) -> Dict[str, Any]:
    """执行完整审计并返回结构化报告。"""
    images_root = data_root / "images"
    labels_root = data_root / "labels"

    LOGGER.info("扫描影像目录: %s", images_root)
    img_info = scan_image_tree(images_root)
    LOGGER.info("  -> %d 个 case（%d 个 patient 目录）", img_info["num_cases"],
                img_info["patient_dir_count"])

    LOGGER.info("扫描标注目录: %s", labels_root)
    lbl_info = scan_label_tree(labels_root)
    for k, v in lbl_info["groups"].items():
        LOGGER.info("  -> %-60s %d files", k, v["num_files"])

    image_keys = set(img_info["cases"].keys())

    # -------- 模态完整性 -------- #
    missing: Dict[str, List[str]] = {}
    for mod in PRIMARY_MODALITIES:
        aliases = MODALITY_ALIASES.get(mod, (mod,))
        miss = sorted(k for k, v in img_info["cases"].items()
                      if not any(a in v["modalities"] for a in aliases))
        missing[mod] = miss

    has_all_primary = sorted(
        k for k, v in img_info["cases"].items()
        if all(any(a in v["modalities"] for a in MODALITY_ALIASES.get(m, (m,)))
               for m in PRIMARY_MODALITIES)
    )

    # -------- 标注与影像的 ID 匹配 -------- #
    match: Dict[str, Any] = {}
    for group_key, keys in lbl_info["case_keys_by_group"].items():
        if not keys or "<root>" == group_key:
            continue
        s = set(keys)
        match[group_key] = {
            "label_keys": len(s),
            "intersect_with_images": len(s & image_keys),
            "labels_without_image": sorted(s - image_keys)[:20],
            "image_without_label": len(image_keys - s),
        }

    # -------- lesion mask 深度分析 -------- #
    lesion_groups = [g for g in lbl_info["groups"] if "lesion" in g.lower()]
    mask_analysis: Dict[str, Any] = {}
    if not skip_mask_analysis:
        for g in lesion_groups:
            group_dir = labels_root / g
            paths = sorted(p for p in group_dir.rglob("*")
                           if p.is_file() and p.suffix.lower() in (".gz", ".nii", ".mha"))
            LOGGER.info("分析 lesion mask 组 [%s]: %d 个文件 (workers=%d)",
                        g, len(paths), mask_workers)
            mask_analysis[g] = analyze_masks(paths, mask_workers)
            ma = mask_analysis[g]
            LOGGER.info("  -> 非空 %d / 空 %d，多病灶 %d 个",
                        ma["num_non_empty"], ma["num_empty"],
                        len(ma["multi_lesion_cases"]))
    else:
        LOGGER.info("已跳过 mask 深度分析 (--skip-mask-analysis)")

    # -------- header 抽样 -------- #
    LOGGER.info("抽样读取 header（每模态 %d 个）", header_samples)
    headers = sample_headers(
        images_root,
        sorted(has_all_primary)[:header_samples * 3] or sorted(image_keys)[:30],
        PRIMARY_MODALITIES,
        header_samples,
    )

    # -------- 临床信息 -------- #
    clinical = read_clinical(labels_root / "clinical_information" / "marksheet.csv")

    # -------- example cases -------- #
    positive_examples: List[Dict[str, Any]] = []
    for g, ma in mask_analysis.items():
        for name in ma["multi_lesion_cases"][:3]:
            key = name.replace(".nii.gz", "").replace(".nii", "")
            positive_examples.append({"case_key": key, "lesion_group": g,
                                      "n_components": ">1"})
    if not positive_examples:
        positive_examples = [{"case_key": k, "note": "no multi-lesion case found"}
                             for k in sorted(image_keys)[:3]]

    report: Dict[str, Any] = {
        "dataset_root": str(data_root),
        "scan_time": datetime.now().isoformat(timespec="seconds"),
        "num_cases": img_info["num_cases"],
        "num_patients": img_info["num_patients"],
        "num_patient_dirs": img_info["patient_dir_count"],
        "multi_study_patients": img_info["multi_study_patients"],
        "modalities": img_info["modality_file_counts"],
        "image_formats": img_info["file_extensions"],
        "num_cases_with_all_primary_modalities": len(has_all_primary),
        "missing_modalities": {k: {"count": len(v), "cases": v[:20]}
                               for k, v in missing.items()},
        "lesion_masks": {
            g: {"num_files": lbl_info["groups"][g]["num_files"],
                "unique_case_keys": lbl_info["groups"][g]["unique_case_keys"]}
            for g in lesion_groups
        },
        "all_label_groups": lbl_info["groups"],
        "label_image_id_match": match,
        "mask_analysis": mask_analysis,
        "header_samples": headers,
        "clinical_information": clinical,
        "example_cases": positive_examples[:10],
        "unparsed_image_files": img_info["unparsed_files"][:20],
        "notes": [
            "hbv 即 PI-CAI 命名下的 DWI 高 b 值图",
            "cor/sag 为冠状/矢状 T2W 附属扫描，仅统计未做模态完整性判定",
        ],
    }
    return report


def write_txt_report(report: Dict[str, Any], path: Path) -> None:
    """把结构化报告写成人类可读文本。

    Args:
        report: ``build_report`` 的返回值。
        path: 输出 txt 路径。
    """
    L: List[str] = []
    add = L.append

    add("=" * 78)
    add("PI-CAI 数据审计报告 (只读)")
    add("=" * 78)
    add(f"数据根目录 : {report['dataset_root']}")
    add(f"扫描时间   : {report['scan_time']}")
    add("")
    add("-" * 78)
    add("[1] 规模")
    add("-" * 78)
    add(f"case 总数        : {report['num_cases']}")
    add(f"patient 总数     : {report['num_patients']}")
    add(f"patient 目录数   : {report['num_patient_dirs']}")
    add(f"多 study patient : {len(report['multi_study_patients'])} "
        f"{report['multi_study_patients'][:10]}")
    add("")
    add("-" * 78)
    add("[2] 模态文件数")
    add("-" * 78)
    for k, v in sorted(report["modalities"].items()):
        add(f"  {k:<6}: {v}")
    add(f"图像格式        : {report['image_formats']}")
    add(f"三大模态齐全    : {report['num_cases_with_all_primary_modalities']} / "
        f"{report['num_cases']}")
    add("")
    add("-" * 78)
    add("[3] 缺失模态")
    add("-" * 78)
    for mod, info in report["missing_modalities"].items():
        add(f"  缺 {mod:<4}: {info['count']} 例  e.g. {info['cases'][:5]}")
    add("")
    add("-" * 78)
    add("[4] 标注分组（labels/ 下实际遍历结果）")
    add("-" * 78)
    for k, v in report["all_label_groups"].items():
        add(f"  {k:<62} {v['num_files']:>6} files")
    add("")
    add("-" * 78)
    add("[5] 标注 ↔ 影像 ID 匹配（按 {patient_id}_{study_id}）")
    add("-" * 78)
    for k, v in report["label_image_id_match"].items():
        add(f"  {k}")
        add(f"      标注数={v['label_keys']}  与影像交集={v['intersect_with_images']}  "
            f"影像无标注={v['image_without_label']}  标注无影像={len(v['labels_without_image'])}")
    add("")
    add("-" * 78)
    add("[6] lesion mask 深度分析")
    add("-" * 78)
    for g, ma in report["mask_analysis"].items():
        add(f"  组: {g}")
        add(f"      文件数={ma['num_masks']}  可读={ma['num_ok']}  失败={ma['num_failed']}")
        add(f"      非空={ma['num_non_empty']}  空={ma['num_empty']}")
        add(f"      label 取值分布 = {ma['label_value_distribution']}")
        add(f"      连通域直方图 (非空) = {ma['component_histogram']}")
        add(f"      多病灶病例数 = {len(ma['multi_lesion_cases'])} e.g. "
            f"{ma['multi_lesion_cases'][:5]}")
        add(f"      spacing 分布 (top5) = "
            f"{list(ma['spacing_distribution'].items())[:5]}")
        if ma["errors"]:
            add(f"      !! 错误 {len(ma['errors'])}: {ma['errors'][:3]}")
        add("")
    add("-" * 78)
    add("[7] header 抽样")
    add("-" * 78)
    for h in report["header_samples"]:
        add(f"  {h['case_key']:<18} {h['modality']:<4} size={h['size']} "
            f"spacing={h['spacing']} type={h['pixel_type']}")
    add("")
    add("-" * 78)
    add("[8] 临床信息 marksheet.csv")
    add("-" * 78)
    ci = report["clinical_information"]
    if ci.get("exists"):
        add(f"  行数: {ci['num_rows']}   列: {ci['columns']}")
        for k in ("case_csPCa_distribution", "case_ISUP_distribution",
                  "center_distribution"):
            if k in ci:
                add(f"  {k}: {ci[k]}")
    else:
        add(f"  未找到: {ci.get('path')}")
    add("")
    add("-" * 78)
    add("[9] 示例 case")
    add("-" * 78)
    for e in report["example_cases"]:
        add(f"  {e}")
    add("")
    add("=" * 78)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """脚本入口。"""
    parser = argparse.ArgumentParser(
        description="PI-CAI 数据集只读审计",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT),
                        help="PI-CAI 数据根目录（含 images/ 与 labels/）")
    parser.add_argument("--out-dir", type=Path,
                        default=PROJECT_ROOT / "outputs" / "data_audit",
                        help="输出目录")
    parser.add_argument("--mask-workers", type=int, default=6,
                        help="mask 分析进程数")
    parser.add_argument("--header-samples", type=int, default=3,
                        help="每个模态抽样读取 header 的例数")
    parser.add_argument("--skip-mask-analysis", action="store_true",
                        help="跳过 mask 像素级分析（只做文件名统计）")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    data_root: Path = args.data_root.resolve()
    assert data_root.is_dir(), f"数据根目录不存在: {data_root}"

    report = build_report(
        data_root=data_root,
        mask_workers=max(1, args.mask_workers),
        header_samples=args.header_samples,
        skip_mask_analysis=args.skip_mask_analysis,
    )

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "picai_audit.json"
    txt_path = out_dir / "picai_audit.txt"

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, default=str)
    write_txt_report(report, txt_path)

    LOGGER.info("已写出: %s", json_path)
    LOGGER.info("已写出: %s", txt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
