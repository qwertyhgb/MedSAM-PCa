#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""正式训练前 preflight 检查（Phase 1.5 / 第十四节）。

**本脚本不训练**：只做 forward，**绝不执行** ``loss.backward()`` 或
``optimizer.step()``。

检查项：

1. checkpoint / manifest / split 文件存在
2. dataset 路径有效、可取到 slice
3. mask cache 状态（需要对齐的 case 是否都已缓存）
4. CUDA 可用性、GPU 名称、空闲显存
5. 模型可构建
6. 一个 batch 的 DataLoader 可用
7. 一次 forward：输出 shape / loss 有限 / encoder 冻结

用法::

    /root/anaconda3/envs/lm/bin/python scripts/preflight_training.py --config configs/e0_linear.yaml
    # GPU 被占用时可用 CPU 做纯逻辑校验（不触碰 GPU）：
    /root/anaconda3/envs/lm/bin/python scripts/preflight_training.py --config configs/e0_linear.yaml --device cpu
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.picai_2d import (  # noqa: E402
    PICAI2DDataset,
    aligned_cache_path,
    geometries_match,
    read_geometry,
)
from src.losses.dice_bce import DiceBCELoss  # noqa: E402
from src.models.medsam_pca import build_model  # noqa: E402
from train import load_config, resolve_path, set_seed  # noqa: E402

LOGGER = logging.getLogger("preflight")

#: 运行正式训练所需的最低空闲显存（MiB），来自 Phase 1 benchmark 的 batch=4 峰值
MIN_FREE_VRAM_MIB = 12000


class Preflight:
    """收集检查结果（只读，不做任何训练动作）。"""

    def __init__(self) -> None:
        self.checks: List[Dict[str, Any]] = []

    def check(self, name: str, ok: bool, detail: Any = None) -> bool:
        """记录一项检查结果。

        Args:
            name: 检查名。
            ok: 是否通过。
            detail: 附加信息。

        Returns:
            ``ok`` 原值，便于短路。
        """
        self.checks.append({"name": name, "ok": bool(ok), "detail": detail})
        mark = "PASS" if ok else "FAIL"
        LOGGER.info("[%s] %s%s", mark, name, f" -> {detail}" if detail is not None else "")
        return bool(ok)

    def fail(self, name: str, detail: Any = None) -> None:
        """记录一项失败（用于异常分支）。"""
        self.check(name, False, detail)

    @property
    def all_ok(self) -> bool:
        """全部检查是否通过。"""
        return all(c["ok"] for c in self.checks)


def run_preflight(cfg: Dict[str, Any], device: Optional[str], batch_size: int) -> Preflight:
    """执行全部 preflight 检查。

    Args:
        cfg: 训练配置。
        device: 目标设备；``None`` 表示自动选择。
        batch_size: 试跑用的 batch 大小。

    Returns:
        :class:`Preflight` 结果对象。
    """
    pf = Preflight()
    data_cfg = cfg["data"]
    data_root = resolve_path(data_cfg["dataset_root"])

    # -------- 1. checkpoint -------- #
    ckpt = resolve_path(cfg["checkpoint"])
    pf.check("checkpoint 存在", ckpt.is_file(),
             f"{ckpt} ({ckpt.stat().st_size / 1024**2:.1f} MiB)" if ckpt.is_file() else ckpt)

    # -------- 2/3. manifest 与 split -------- #
    train_manifest = resolve_path(data_cfg["train_manifest"])
    val_manifest = resolve_path(data_cfg["val_manifest"])
    for tag, mp in (("train", train_manifest), ("val", val_manifest)):
        pf.check(f"{tag} manifest 存在", mp.is_file(),
                 f"{mp.name} ({sum(1 for _ in mp.open()) - 1 if mp.is_file() else 0} rows)")

    split_dir = PROJECT_ROOT / "data" / "splits"
    fold = int(data_cfg.get("fold", 0))
    fold_file = split_dir / f"fold{fold}.csv"
    pf.check(f"split fold{fold} 存在", fold_file.is_file(), str(fold_file))
    if fold_file.is_file():
        sdf = pd.read_csv(fold_file)
        tr_pat = set(sdf.loc[sdf["split"] == "train", "patient_id"])
        va_pat = set(sdf.loc[sdf["split"] == "val", "patient_id"])
        pf.check("patient overlap == 0", len(tr_pat & va_pat) == 0,
                 f"train_patients={len(tr_pat)}, val_patients={len(va_pat)}, overlap={len(tr_pat & va_pat)}")

    if not (train_manifest.is_file() and val_manifest.is_file()):
        pf.fail("后续检查中止", "manifest 缺失")
        return pf

    train_df = pd.read_csv(train_manifest)
    val_df = pd.read_csv(val_manifest)
    pf.check("val slice 数", len(val_df) > 0, f"val={len(val_df)}, train={len(train_df)}")
    pf.check("train 含 positive slice",
             bool(train_df["is_positive"].any()),
             f"pos={int(train_df['is_positive'].sum())}, "
             f"neg={int((~train_df['is_positive'].astype(bool)).sum())}")

    # -------- 4. dataset 路径有效性（抽样） -------- #
    sample = train_df.head(5)
    bad = [r.case_id for r in sample.itertuples() if not (data_root / str(r.t2w_path)).is_file()]
    pf.check("dataset 路径抽样有效", not bad, f"检查 {len(sample)} 条, 缺失 {bad}")

    # -------- 5. mask cache 状态 -------- #
    mask_cache = data_cfg.get("mask_cache_dir")
    cache_dir = resolve_path(mask_cache) if mask_cache else None
    need_alignment, cached, missing_cache = 0, 0, []
    for row in train_df.drop_duplicates("case_id").itertuples(index=False):
        if pd.isna(row.mask_path):
            continue
        mask_abs = str(data_root / str(row.mask_path))
        t2w_abs = str(data_root / str(row.t2w_path))
        try:
            if geometries_match(read_geometry(mask_abs), read_geometry(t2w_abs)):
                continue
        except Exception as exc:  # noqa: BLE001
            pf.fail(f"读取几何失败 {row.case_id}", str(exc))
            continue
        need_alignment += 1
        if cache_dir is not None:
            cp = aligned_cache_path(mask_abs, t2w_abs, cache_dir)
            if cp.is_file():
                cached += 1
            else:
                missing_cache.append(str(row.case_id))
    pf.check("mask 对齐缓存就绪", not missing_cache,
             f"需对齐 {need_alignment} 个 case, 已缓存 {cached}, 缺失 {missing_cache[:5]}")

    # -------- 6. CUDA -------- #
    cuda_ok = torch.cuda.is_available()
    # 显式 --device cpu 时（GPU 被其他任务占用、只做纯逻辑校验），CUDA 检查不阻塞
    cuda_required = (device is None) or str(device).startswith("cuda")
    pf.check("CUDA 可用" + ("" if cuda_required else " [CPU 模式，不阻塞]"),
             cuda_ok or not cuda_required,
             f"torch.cuda.is_available()={cuda_ok}, runtime={torch.version.cuda}")
    if cuda_ok:
        free, total = torch.cuda.mem_get_info(0)
        free_mib, total_mib = free / 1024 ** 2, total / 1024 ** 2
        pf.check("GPU", True, torch.cuda.get_device_name(0))
        pf.check("GPU 空闲显存充足", free_mib >= MIN_FREE_VRAM_MIB,
                 f"free={free_mib:.0f} MiB / total={total_mib:.0f} MiB "
                 f"(需要 >= {MIN_FREE_VRAM_MIB} MiB)")

    target_device = device or ("cuda" if cuda_ok else "cpu")
    pf.check("目标设备", True, target_device)

    # -------- 7. 模型构建 -------- #
    try:
        model = build_model(cfg["model_name"], checkpoint_path=ckpt,
                            freeze_encoder=bool(cfg.get("encoder_frozen", True)),
                            device=target_device)
        stats = model.parameter_stats()
        pf.check("模型构建成功", True, f"{cfg['model_name']}: {stats}")
        pf.check("encoder 完全冻结", stats["encoder_trainable"] == 0,
                 f"trainable encoder params = {stats['encoder_trainable']}")
        pf.check("decoder 有可训练参数", stats["decoder_trainable"] > 0,
                 f"decoder trainable = {stats['decoder_trainable']:,}")
    except Exception as exc:  # noqa: BLE001
        pf.fail("模型构建失败", f"{type(exc).__name__}: {exc}")
        return pf

    # -------- 8. DataLoader 一个 batch -------- #
    try:
        ds = PICAI2DDataset(records=train_df.head(max(batch_size * 4, 16)),
                            data_root=data_root,
                            image_size=int(data_cfg.get("image_size", 1024)),
                            augment=False,
                            mask_cache_dir=cache_dir)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        pf.check("DataLoader 可用", True,
                 f"image={tuple(batch['image'].shape)} mask={tuple(batch['mask'].shape)}")
    except Exception as exc:  # noqa: BLE001
        pf.fail("DataLoader 失败", f"{type(exc).__name__}: {exc}")
        return pf

    # -------- 9/10/11. 一次 forward（无 backward） -------- #
    try:
        images = batch["image"].to(target_device)
        masks = batch["mask"].to(target_device)
        model.eval()
        with torch.no_grad():
            out = model(images)
        logits = out["logits"]
        pf.check("输出 shape", tuple(logits.shape) == (batch_size, 1, 1024, 1024),
                 tuple(logits.shape))
        pf.check("输出有限", bool(torch.isfinite(logits).all()), "无 NaN/Inf")

        criterion = DiceBCELoss().to(target_device)
        with torch.no_grad():
            loss = criterion(logits, masks)
        pf.check("loss 有限", bool(torch.isfinite(loss)), f"loss={float(loss):.4f}")

        probs = torch.sigmoid(logits)
        pf.check("概率范围合法", bool((probs >= 0).all() and (probs <= 1).all()),
                 f"min={float(probs.min()):.4f} max={float(probs.max()):.4f}")
    except Exception as exc:  # noqa: BLE001
        pf.fail("forward 失败", f"{type(exc).__name__}: {exc}")
        return pf

    # -------- 12. 无梯度残留（确认未发生反向） -------- #
    grads = [n for n, p in model.named_parameters() if p.grad is not None]
    pf.check("未产生任何梯度（确认只做 forward）", not grads, f"有梯度的张量数={len(grads)}")

    return pf


def main(argv: Optional[Sequence[str]] = None) -> int:
    """入口。"""
    parser = argparse.ArgumentParser(
        description="正式训练前 preflight（只做 forward，不训练）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default=None,
                        help="cuda / cpu；默认自动。GPU 被占用时可用 cpu 做纯逻辑校验")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--json-out", type=Path, default=None,
                        help="把检查结果写成 JSON（可选）")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)

    cfg = load_config(args.config)
    set_seed(int(cfg["train"]["seed"]))

    LOGGER.info("=" * 78)
    LOGGER.info("PREFLIGHT: model=%s  config=%s  device=%s",
                cfg["model_name"], args.config, args.device or "auto")
    LOGGER.info("=" * 78)

    pf = run_preflight(cfg, args.device, args.batch_size)

    LOGGER.info("-" * 78)
    if pf.all_ok:
        LOGGER.info("READY FOR TRAINING")
    else:
        failed = [c["name"] for c in pf.checks if not c["ok"]]
        LOGGER.error("NOT READY: %d 项检查失败 -> %s", len(failed), failed)
    LOGGER.info("-" * 78)

    if args.json_out is not None:
        payload = {"config": str(args.config), "model_name": cfg["model_name"],
                   "device": args.device or "auto", "ready": pf.all_ok,
                   "checks": pf.checks}
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)

    return 0 if pf.all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
