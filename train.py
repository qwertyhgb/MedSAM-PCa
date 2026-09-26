#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase 1 正式训练脚本（原生 PyTorch，不使用 Lightning / Hydra）。

支持：AMP(fp16)、GradScaler、gradient accumulation、checkpoint / resume
（优化器 + 调度器 + scaler + epoch + best metric）、TensorBoard（缺失时降级为
仅 CSV）、CSV 指标、seed、num_workers、batch_size、learning_rate、
weight_decay、epochs、**train/val 双 tqdm 进度条**。

**encoder 全程冻结**，optimizer 只接收 ``requires_grad=True`` 的参数，
启动时断言 optimizer 中不含任何冻结的 encoder 参数。

⚠️ Rule 1：正式训练必须由**用户本人**启动，AI 不得执行本脚本。

用法::

    CUDA_VISIBLE_DEVICES=0 python train.py --config configs/e0_linear.yaml
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.picai_2d import BalancedSliceSampler, PICAI2DDataset  # noqa: E402
from src.losses.dice_bce import DiceBCELoss  # noqa: E402
from src.metrics.segmentation import SegmentationMetrics  # noqa: E402
from src.models.medsam_pca import build_model  # noqa: E402

LOGGER = logging.getLogger("train")

try:  # TensorBoard 可选
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
    _HAS_TB = True
except Exception:  # noqa: BLE001
    SummaryWriter = None  # type: ignore
    _HAS_TB = False

#: metrics.csv 字段（未执行 validation 的 epoch，val_* 留空）
CSV_FIELDS: Tuple[str, ...] = (
    "epoch", "train_loss", "learning_rate",
    "val_loss", "val_positive_dice", "val_positive_iou",
    "val_all_slice_dice", "val_precision", "val_recall", "val_specificity",
    "val_fp_slice_rate", "val_fp_slices", "val_slices",
    "train_seconds", "val_seconds",
)


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """固定全部随机源。

    Args:
        seed: 随机种子。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config(path: Path) -> Dict[str, Any]:
    """读取 YAML 配置。

    Args:
        path: YAML 路径。

    Returns:
        配置字典。
    """
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"配置文件格式错误: {path}")
    cfg["_config_path"] = str(path)
    return cfg


def resolve_path(p: str | Path) -> Path:
    """相对路径按项目根解析。"""
    path = Path(p)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def write_environment(out_dir: Path) -> None:
    """写出运行环境信息。"""
    lines = [
        f"time: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"python: {sys.version.split()[0]} ({sys.executable})",
        f"torch: {torch.__version__}",
        f"cuda runtime: {torch.version.cuda}",
        f"cuda available: {torch.cuda.is_available()}",
        f"tensorboard: {'yes' if _HAS_TB else 'no'}",
    ]
    if torch.cuda.is_available():
        lines.append(f"gpu: {torch.cuda.get_device_name(0)}")
        prop = torch.cuda.get_device_properties(0)
        lines.append(f"gpu total mem: {prop.total_memory / 1024**3:.2f} GiB")
        lines.append(f"gpu capability: sm_{prop.major}{prop.minor}")
    (out_dir / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def gpu_mem_gib() -> float:
    """当前进程已分配的 GPU 显存（GiB）；非 CUDA 返回 0。"""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024 ** 3


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #
def build_dataset(cfg: Dict[str, Any], which: str, augment: bool) -> PICAI2DDataset:
    """构建 slice 级数据集。

    Args:
        cfg: 全局配置。
        which: ``"train"`` 或 ``"val"``。
        augment: 是否启用增强。

    Returns:
        :class:`PICAI2DDataset`。
    """
    data_cfg = cfg["data"]
    manifest = resolve_path(data_cfg[f"{which}_manifest"])
    if not manifest.is_file():
        raise FileNotFoundError(
            f"{which} manifest 不存在: {manifest}，请先运行 scripts/build_splits.py"
        )
    records = pd.read_csv(manifest)
    if data_cfg.get("input_mode") != "t2_repeat":
        raise ValueError(
            f"Phase 1 仅支持 input_mode=t2_repeat，收到 {data_cfg.get('input_mode')}"
        )

    mask_cache = data_cfg.get("mask_cache_dir")
    return PICAI2DDataset(
        records=records,
        data_root=resolve_path(data_cfg["dataset_root"]),
        image_size=int(data_cfg.get("image_size", 1024)),
        augment=augment,
        seed=int(cfg["train"]["seed"]) + (0 if which == "train" else 99991),
        mask_cache_dir=resolve_path(mask_cache) if mask_cache else None,
    )


def build_loaders(cfg: Dict[str, Any]
                  ) -> Tuple[DataLoader, DataLoader, BalancedSliceSampler, PICAI2DDataset]:
    """构建 train / val DataLoader。

    Returns:
        ``(train_loader, val_loader, sampler, train_ds)``。
    """
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    seed = int(train_cfg["seed"])
    nw = int(data_cfg.get("num_workers", 4))
    pin = bool(data_cfg.get("pin_memory", True))
    prefetch = data_cfg.get("prefetch_factor", 2)
    persistent = bool(data_cfg.get("persistent_workers", True)) and nw > 0

    train_ds = build_dataset(cfg, "train", augment=True)
    val_ds = build_dataset(cfg, "val", augment=False)  # validation 不做随机增强

    s_cfg = data_cfg.get("sampler", {}) or {}
    sampler = BalancedSliceSampler(
        labels=train_ds.labels,
        positive_ratio=float(s_cfg.get("positive_ratio", 0.5)),
        num_samples=s_cfg.get("num_samples"),
        seed=seed,
    )

    train_loader = DataLoader(
        train_ds, batch_size=int(train_cfg["batch_size"]), sampler=sampler,
        num_workers=nw, pin_memory=pin, drop_last=True,
        prefetch_factor=prefetch if nw > 0 else None,
        persistent_workers=persistent,
    )
    # validation 使用全部 slice，不使用平衡采样
    val_loader = DataLoader(
        val_ds, batch_size=int(train_cfg["batch_size"]), shuffle=False,
        num_workers=nw, pin_memory=pin, drop_last=False,
        prefetch_factor=prefetch if nw > 0 else None,
        persistent_workers=persistent,
    )
    LOGGER.info("train slices=%d (sampler 每 epoch %d 个) | val slices=%d | "
                "num_workers=%d pin_memory=%s prefetch=%s persistent=%s",
                len(train_ds), len(sampler), len(val_ds), nw, pin, prefetch, persistent)
    return train_loader, val_loader, sampler, train_ds


# --------------------------------------------------------------------------- #
# 优化器 / 调度器
# --------------------------------------------------------------------------- #
def build_optimizer(model, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    """构建 AdamW，**只接入可训练参数**。

    Args:
        model: :class:`MedSAMPCA`。
        cfg: 配置。

    Returns:
        optimizer。

    Raises:
        AssertionError: 若发现冻结的 encoder 参数被接入 optimizer。
    """
    o_cfg = cfg["optimizer"]
    params = model.trainable_parameters()
    assert len(params) > 0, "没有任何可训练参数"

    frozen_ids = {id(p) for p in model.encoder.parameters() if not p.requires_grad}
    leaked = [id(p) for p in params if id(p) in frozen_ids]
    assert not leaked, f"冻结的 encoder 参数泄漏进 optimizer（{len(leaked)} 个）"

    opt = torch.optim.AdamW(
        params,
        lr=float(o_cfg.get("lr", 1e-4)),
        weight_decay=float(o_cfg.get("weight_decay", 1e-4)),
    )
    n_param = sum(p.numel() for p in params)
    LOGGER.info("AdamW: %d 个张量 / %d 参数, lr=%g, wd=%g",
                len(params), n_param, o_cfg.get("lr"), o_cfg.get("weight_decay"))
    return opt


def build_scheduler(opt: torch.optim.Optimizer, cfg: Dict[str, Any]):
    """构建 CosineAnnealingLR（``min_lr`` = eta_min）。"""
    s_cfg = cfg.get("scheduler", {}) or {}
    name = str(s_cfg.get("name", "cosine")).lower()
    if name not in ("cosine", "none"):
        raise ValueError(f"Phase 1 仅支持 cosine 调度器，收到 {name!r}")
    if name == "none":
        return None
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=int(cfg["train"]["epochs"]),
        eta_min=float(s_cfg.get("min_lr", 1e-6)),
    )


# --------------------------------------------------------------------------- #
# 训练 / 验证
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, criterion, optimizer, scaler, cfg,
                    epoch: int, total_epochs: int, writer,
                    max_iterations: Optional[int] = None) -> Dict[str, float]:
    """训练一个 epoch（带 tqdm 进度条）。

    Returns:
        含 loss / 耗时 / 迭代数的字典。
    """
    t_cfg = cfg["train"]
    amp = bool(t_cfg.get("amp", True)) and torch.cuda.is_available()
    amp_dtype = (torch.float16 if str(t_cfg.get("amp_dtype", "float16")) == "float16"
                 else torch.bfloat16)
    accum = max(1, int(t_cfg.get("gradient_accumulation", 1)))
    log_every = int(t_cfg.get("log_every", 20))

    model.train()
    model.encoder.eval()  # 冻结的 encoder 保持 eval
    total_loss, n_steps, n_skipped = 0.0, 0, 0
    t0 = time.time()

    bar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [train]", unit="batch",
               ncols=120, dynamic_ncols=True, file=sys.stderr, leave=True)
    for it, batch in enumerate(bar, 1):
        images = batch["image"].cuda(non_blocking=True)
        masks = batch["mask"].cuda(non_blocking=True)

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp):
            out = model(images)
            loss = criterion(out["logits"], masks)

        if not torch.isfinite(loss):
            raise FloatingPointError(f"epoch {epoch} iter {it}: loss 非有限值 {loss.item()}")

        scaler.scale(loss / accum).backward()

        if it % accum == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.trainable_parameters(), max_norm=1.0)
            if not torch.isfinite(grad_norm):
                # AMP 下偶发溢出：跳过该步，由 GradScaler 自动降低 scale
                n_skipped += 1
                LOGGER.warning("epoch %d iter %d: 梯度非有限，跳过该步", epoch, it)
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.item())
        n_steps += 1

        bar.set_postfix({
            "loss": f"{total_loss / max(n_steps, 1):.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            "gpu": f"{gpu_mem_gib():.1f}G",
        })
        if log_every > 0 and it % log_every == 0:
            bar.write(f"  epoch {epoch} iter {it}/{len(loader)} "
                      f"loss={total_loss / max(n_steps, 1):.4f}")
        if max_iterations is not None and it >= max_iterations:
            bar.write(f"  达到 max_iterations={max_iterations}，提前结束本 epoch")
            break
    bar.close()

    mean_loss = total_loss / max(n_steps, 1)
    return {"loss": mean_loss, "steps": n_steps, "skipped_steps": n_skipped,
            "seconds": time.time() - t0}


@torch.no_grad()
def validate(model, loader, criterion, cfg, epoch: int, total_epochs: int,
             collect_per_case: bool = False,
             max_batches: Optional[int] = None) -> Dict[str, Any]:
    """在**全部** validation slice 上评估（带 tqdm 进度条）。

    Args:
        model: 模型。
        loader: validation DataLoader（应包含全部 slice，不使用平衡采样）。
        criterion: 损失函数。
        cfg: 配置。
        epoch: 当前 epoch（进度条显示用）。
        total_epochs: 总 epoch 数。
        collect_per_case: 是否累计 per-case 统计。
        max_batches: 仅评估前 N 个 batch（smoke test 用，``None`` 为全部）。

    Returns:
        指标字典（含 positive-slice Dice / all-slice Dice / fp_slice_rate 等）。
    """
    t_cfg = cfg["train"]
    amp = bool(t_cfg.get("amp", True)) and torch.cuda.is_available()
    amp_dtype = (torch.float16 if str(t_cfg.get("amp_dtype", "float16")) == "float16"
                 else torch.bfloat16)

    model.eval()
    metrics = SegmentationMetrics()
    total_loss, n = 0.0, 0
    t0 = time.time()

    bar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [val]", unit="batch",
               ncols=120, dynamic_ncols=True, file=sys.stderr, leave=True)
    for i, batch in enumerate(bar):
        if max_batches is not None and i >= max_batches:
            bar.write(f"  达到 max_val_batches={max_batches}，提前结束 validation")
            break
        images = batch["image"].cuda(non_blocking=True)
        masks = batch["mask"].cuda(non_blocking=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp):
            out = model(images)
            loss = criterion(out["logits"], masks)
        total_loss += float(loss.item())
        n += 1
        metrics.update(out["logits"].float(), masks,
                       case_ids=list(batch["case_id"]) if collect_per_case else None)
        bar.set_postfix({"loss": f"{total_loss / max(n, 1):.4f}",
                         "gpu": f"{gpu_mem_gib():.1f}G"})
    bar.close()

    summary = metrics.summary()
    summary["loss"] = total_loss / max(n, 1)
    summary["seconds"] = time.time() - t0
    return summary


# --------------------------------------------------------------------------- #
# checkpoint
# --------------------------------------------------------------------------- #
def save_checkpoint(path: Path, model, optimizer, scaler, scheduler,
                    epoch: int, best: float, cfg: Dict[str, Any]) -> None:
    """保存 checkpoint（encoder 冻结，不入盘）。

    Args:
        path: 输出路径。
        model: 模型。
        optimizer: 优化器。
        scaler: GradScaler。
        scheduler: 调度器或 ``None``。
        epoch: 当前 epoch。
        best: 最佳指标。
        cfg: 配置。
    """
    trainable_state = {
        k: v for k, v in model.state_dict().items() if not k.startswith("encoder.")
    }
    payload = {
        "epoch": epoch,
        "model_name": model.model_name,
        "model": trainable_state,                      # decoder 权重
        "trainable_state_dict": trainable_state,       # 兼容旧键名
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "best_metric": best,
        "config": cfg,
        "seed": int(cfg["train"]["seed"]),
        "note": "encoder 参数已冻结，未包含在本文件中；请由 MedSAM checkpoint 恢复。",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    """训练入口。

    ⚠️ Rule 1：本函数只能由**用户本人**调用，AI 不得执行。
    """
    parser = argparse.ArgumentParser(
        description="Phase 1 MedSAM-PCa 正式训练（需由用户本人启动）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-iterations", type=int, default=None,
                        help="每个 epoch 最多迭代数（smoke test 用）")
    parser.add_argument("--max-val-batches", type=int, default=None,
                        help="validation 最多评估的 batch 数（smoke test 用）")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖配置中的 epochs")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    if args.seed is not None:
        cfg["train"]["seed"] = args.seed
    run_name = args.run_name or cfg.get("run_name") or cfg["model_name"]

    out_dir = resolve_path(cfg["output"]["root"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tensorboard").mkdir(exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(out_dir / "train.log", mode="a")],
    )

    with (out_dir / "config.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)
    write_environment(out_dir)

    seed = int(cfg["train"]["seed"])
    set_seed(seed)
    LOGGER.info("=" * 78)
    LOGGER.info("模型=%s  run=%s  seed=%d  out=%s",
                cfg["model_name"], run_name, seed, out_dir)
    LOGGER.info("=" * 78)

    train_loader, val_loader, sampler, train_ds = build_loaders(cfg)

    model = build_model(
        cfg["model_name"],
        checkpoint_path=resolve_path(cfg["checkpoint"]),
        freeze_encoder=bool(cfg.get("encoder_frozen", True)),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    stats = model.parameter_stats()
    LOGGER.info("参数量: encoder=%d(frozen %d) decoder=%d(trainable %d)",
                stats["encoder_total"], stats["encoder_total"] - stats["encoder_trainable"],
                stats["decoder_total"], stats["decoder_trainable"])
    assert stats["encoder_trainable"] == 0, "Phase 1 要求 encoder 完全冻结"

    criterion = DiceBCELoss(
        dice_weight=float(cfg["loss"].get("dice_weight", 1.0)),
        bce_weight=float(cfg["loss"].get("bce_weight", 1.0)),
    ).cuda()

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["train"].get("amp", True)))

    best_metric = -1.0
    start_epoch = 1
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        state = ckpt.get("trainable_state_dict") or ckpt.get("model")
        if state is None:
            raise KeyError(f"checkpoint 缺少 model 权重: {args.resume}")
        _, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(f"resume 出现意外键: {unexpected[:5]}")
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        if scheduler is not None and ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_metric = float(ckpt.get("best_metric", -1.0))
        LOGGER.info("已从 %s 恢复，从 epoch %d 继续", args.resume, start_epoch)

    writer = SummaryWriter(str(out_dir / "tensorboard")) if _HAS_TB else None
    if not _HAS_TB:
        LOGGER.warning("TensorBoard 不可用（未安装 tensorboard），仅写 CSV 指标")

    metrics_csv = out_dir / "metrics.csv"
    if not metrics_csv.exists():
        with metrics_csv.open("w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=list(CSV_FIELDS)).writeheader()

    epochs = int(cfg["train"]["epochs"])
    val_interval = max(1, int(cfg["train"].get("val_interval", 2)))

    for epoch in range(start_epoch, epochs + 1):
        sampler.set_epoch(epoch)
        train_ds.set_epoch(epoch)

        tr = train_one_epoch(model, train_loader, criterion, optimizer, scaler,
                             cfg, epoch, epochs, writer,
                             max_iterations=args.max_iterations)
        lr_now = optimizer.param_groups[0]["lr"]

        # epoch 1 额外验证一次；之后每 val_interval 个 epoch 验证；最后一个 epoch 必验证
        do_val = (epoch == 1) or (epoch % val_interval == 0) or (epoch == epochs)

        va: Dict[str, Any] = {}
        if do_val:
            va = validate(model, val_loader, criterion, cfg, epoch, epochs,
                          max_batches=args.max_val_batches)
        else:
            LOGGER.info("epoch %d/%d 跳过 validation（val_interval=%d）",
                        epoch, epochs, val_interval)

        if scheduler is not None:
            scheduler.step()

        row: Dict[str, Any] = {
            "epoch": epoch,
            "train_loss": round(tr["loss"], 6),
            "learning_rate": lr_now,
            "train_seconds": round(tr["seconds"], 2),
            "val_seconds": round(va["seconds"], 2) if va else "",
        }
        if va:
            row.update({
                "val_loss": round(va["loss"], 6),
                "val_positive_dice": va["positive_slice_dice"],
                "val_positive_iou": va["positive_slice_iou"],
                "val_all_slice_dice": va["all_slice_dice"],
                "val_precision": va["micro_precision"],
                "val_recall": va["micro_recall"],
                "val_specificity": va["micro_specificity"],
                "val_fp_slice_rate": va["fp_slice_rate"],
                "val_fp_slices": va["fp_slices"],
                "val_slices": va["num_slices"],
            })
        else:
            for k in ("val_loss", "val_positive_dice", "val_positive_iou",
                      "val_all_slice_dice", "val_precision", "val_recall",
                      "val_specificity", "val_fp_slice_rate", "val_fp_slices",
                      "val_slices"):
                row[k] = ""

        with metrics_csv.open("a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=list(CSV_FIELDS)).writerow(row)

        if va:
            LOGGER.info(
                "epoch %d/%d | train_loss=%.4f | val_loss=%.4f | pos-Dice=%s | "
                "all-Dice=%s | FP_slice_rate=%s | %.1fs+%.1fs",
                epoch, epochs, tr["loss"], va["loss"],
                f"{va['positive_slice_dice']:.4f}" if va["positive_slice_dice"] is not None else "n/a",
                f"{va['all_slice_dice']:.4f}" if va["all_slice_dice"] is not None else "n/a",
                f"{va['fp_slice_rate']:.4f}" if va["fp_slice_rate"] is not None else "n/a",
                tr["seconds"], va["seconds"])
        else:
            LOGGER.info("epoch %d/%d | train_loss=%.4f | (no validation) | %.1fs",
                        epoch, epochs, tr["loss"], tr["seconds"])

        if writer is not None:
            for k, v in row.items():
                if k != "epoch" and isinstance(v, (int, float)):
                    writer.add_scalar(k, v, epoch)

        save_checkpoint(out_dir / "checkpoint_latest.pth", model, optimizer, scaler,
                        scheduler, epoch, best_metric, cfg)
        if va and va["positive_slice_dice"] is not None:
            cur = float(va["positive_slice_dice"])
            if cur > best_metric:
                best_metric = cur
                save_checkpoint(out_dir / "checkpoint_best.pth", model, optimizer,
                                scaler, scheduler, epoch, best_metric, cfg)
                LOGGER.info("  新的最佳 positive-slice Dice = %.4f（已保存 checkpoint_best.pth）",
                            best_metric)

    if writer is not None:
        writer.close()
    LOGGER.info("训练结束，best positive-slice Dice=%.4f，产物目录: %s", best_metric, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
