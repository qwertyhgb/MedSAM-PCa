# Phase 1 实验协议（四组实验共用）

本文档记录 E0–E3 四组实验共享的数据、训练与评估协议，保证报告中的数字可在同一口径下比较。

## 1. 数据源

| 项目 | 值 |
| --- | --- |
| 数据集 | PI-CAI（Prostate Imaging: Cancer AI） |
| 数据根目录 | `/opt/data/private/lm/data/Prostate/PI-CAI`（本仓库外部，未纳入 Git） |
| case 总数 | 1500 |
| patient 总数 | 1476（23 个 patient 含多次 study） |
| 模态 | T2W 1500 / ADC 1500 / HBV 1500（三大模态齐全 1500/1500），另含 cor 1497 / sag 1498 |
| 图像格式 | `.mha`，共 7495 个文件 |
| 本阶段使用模态 | **仅 T2W**（`input_mode: t2_repeat`，单通道复制为 3 通道，适配 ViT 的三通道输入） |
| 标注来源 | `labels/csPCa_lesion_delineations/human_expert/resampled`（软组织对齐重采样版本） |

审计明细见 `outputs/data_audit/picai_audit.{txt,json}`（1500 例标注分组：Bosma22a 1500 / Pooch25 205 / original 1295）。

## 2. 数据划分

- 策略：`StratifiedGroupKFold`，`group = patient_id`（同 patient 多次 study 不跨折）、`stratify = case_csPCa`（以「case 级 csPCa 阳性」为分层目标）。
- 种子：42，5 折。
- **Phase 1 固定协议：fold0 = 验证集，fold1–4 = 训练集**（仓库未找到 PI-CAI 官方 fold 文件，`official_fold_found: false`，故使用自建分层划分）。

| 折 | 训练 case | 验证 case | 训练 patient | 验证 patient | patient 重叠 | 验证阳性 case |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0（本阶段） | 1200 | 300 | 1181 | 295 | **0** | 84 |
| 1 | 1201 | 299 | 1181 | 295 | 0 | 85 |
| 2 | 1202 | 298 | 1181 | 295 | 0 | 88 |
| 3 | 1195 | 305 | 1180 | 296 | 0 | 84 |
| 4 | 1202 | 298 | 1181 | 295 | 0 | 84 |

划分文件：`data/splits/fold{0..4}.csv`，汇总：`data/splits/split_summary.json`。

## 3. 切片级数据集（2D）

T2W 体数据按轴位逐层展开为 2D 样本，mask 与 T2W 几何对齐后同步取层。

| 项目 | train（fold1–4） | val（fold0） |
| --- | ---: | ---: |
| 切片总数 | 27,090 | 6,614 |
| 阳性切片（含病灶） | 1,844（6.81%） | 491（7.42%） |
| 阴性切片 | 25,246 | 6,123 |
| 含病灶 case | 341 | 84 |
| mask 状态 | aligned 26,264 / resampled 826 | aligned 6,539 / resampled 75 |

**mask 对齐**：当 mask 与 T2W 几何不一致时（容差 1e-4），以 T2W 为参考做最近邻重采样，结果落盘缓存（`data/cache/masks_aligned`，命名 `md5(mask_path|t2w_path)`，原子写入），不修改原始标注。缓存报告：需对齐 32 个 case，全部成功，失败 0（`outputs/data_audit/mask_cache_report.json`）。

**读取方式**：SimpleITK 按需只读单层（`SetExtractIndex([0,0,z])` + `SetExtractSize([w,h,1])`），不整卷常驻内存；几何信息带 LRU 缓存。

**mask 二值化**：一律 `mask > 0`（兼容标签值 1/2/3/4/5）。

**采样**：`BalancedSliceSampler`，`positive_ratio = 0.5`，即每个 epoch 采样 **1:1** 的正负切片；`num_samples = round(n_pos / 0.5) = 3688`（batch_size=4 → 922 step/epoch）。验证集全量、不打乱、不使用平衡采样。

## 4. 预处理与数据增强

**预处理（train/val 相同）**

1. 取单层 T2W → 复制为 3 通道；
2. `clip_normalize`：对**非零区域**做 0.5 / 99.5 百分位裁剪后线性映射到 [0,1]（非零区为空时退化为全图百分位；动态范围退化或出现非有限值时返回全零）；
3. resize 到 1024×1024：图像双线性、mask 最近邻（并断言 mask 取值仍为 {0,1}）。

> **归一化口径说明（Phase 2A 修正）**：本流程输出 **[0,1]** 区间，这与官方 MedSAM 的训练/推理口径**一致**——官方 `train_one_gpu.py` 明确要求把图像归一化到 [0,1] 后直接送入 `image_encoder`，`MedSAM_Inference.py` 同样以 min-max 归一化到 [0,1] 后直接前向。本项目的唯一区别是**先对非零区域做 0.5/99.5 百分位裁剪，再做 min-max**；官方示例以纯 min-max 为主。
>
> 因此**不应**把输入切换成 SAM/ImageNet 的 `pixel_mean = [123.675, 116.28, 103.53]` / `pixel_std = [58.395, 57.12, 57.375]`。若未来做归一化消融，应比较：(a) plain min-max；(b) percentile clipping + min-max。

**数据增强（仅训练集）**

| 增强 | 参数 |
| --- | --- |
| 水平翻转 | p = 0.5 |
| 旋转 | ±15° |
| 缩放 | 0.9–1.1 |

图像用双线性、mask 用最近邻（mask 断言仍为 {0,1}）。**未使用**：垂直翻转、弹性形变、强度/伽马扰动、CutMix/MixUp、随机裁剪。每样本随机数由 `seed + epoch × 1000003 + index` 派生，随 epoch 变化。

## 5. 模型与优化配置

| 项目 | 配置 |
| --- | --- |
| 编码器 | 官方 MedSAM ViT-B（`weights/medsam_vit_b.pth`，357.7 MiB），`img_size=1024`、`patch=16`、`embed_dim=768`、`depth=12` |
| 特征输出 | `f3/f6/f9/f12` = 第 3/6/9/12 个 block 输出 [B,768,64,64]；`neck` = [B,256,64,64] |
| 冻结策略 | encoder **全部冻结**（`requires_grad=False` 且 `eval()`，前向在 `torch.no_grad` 下不建图）；训练期断言 encoder 可训练参数为 0 |
| 输入尺寸 | 固定 1024×1024（encoder 内对输入尺寸做硬校验，**未实现 pos_embed 插值**，低分辨率输入会直接报错） |
| 损失 | `DiceBCELoss`：`L = 1.0 × (1 − softDice) + 1.0 × BCEWithLogitsLoss`，`eps=1e-6`，batch 内平均，**无 pos_weight 类别加权** |
| 优化器 | AdamW，仅接可训练参数；`lr=1e-4`、`weight_decay=1e-4` |
| 调度器 | CosineAnnealingLR，`T_max=40`、`eta_min=1e-6`，每 epoch step |
| batch size | 4（梯度累积 1） |
| epoch 数 | 40 |
| AMP | `torch.amp.autocast('cuda', dtype=float16)` + `GradScaler` |
| 梯度裁剪 | `clip_grad_norm_(trainable_params, max_norm=1.0)`（在 `scaler.unscale_` 之后）；梯度非有限则跳过该 step 并告警 |
| 随机种子 | 42（`random/numpy/torch/cuda/PYTHONHASHSEED` 全固定；验证数据集 seed 偏移 +99991） |
| 验证频率 | `val_interval = 2`，另加 epoch 1 与 epoch 40 必验证（共 21 个验证点） |
| 模型选择 | `best = max(val_positive_slice_dice)`，仅更优时覆盖 `checkpoint_best.pth`；每 epoch 保存 `checkpoint_latest.pth` |
| 早停 | **无** |
| checkpoint 内容 | `epoch / model_name / trainable_state_dict / optimizer / scaler / scheduler / best_metric / config / seed / note`（encoder 权重不重复保存，需从 `weights/medsam_vit_b.pth` 恢复） |

## 6. 评估指标

评估在 fold0 验证集全量切片上进行（训练期指标由 `train.py` 内联计算；`evaluate.py` 提供同口径的独立评估入口）。

| 指标 | 定义 | 聚合方式 |
| --- | --- | --- |
| `positive_slice_dice`（pos-Dice，主指标） | `(2TP+eps)/(2TP+FP+FN+eps)`，逐切片计算 | **仅 GT 非空切片**求平均 |
| `all_slice_dice` | 同上 | 全部切片求平均；**GT 空且预测空时记为 1.0** |
| `positive_slice_iou` | `(TP+eps)/(TP+FP+FN+eps)` | 仅 GT 非空切片 |
| precision / recall / specificity | `TP/(TP+FP)`、`TP/(TP+FN)`、`TN/(TN+FP)` | **像素级全局微平均**（先累加 TP/FP/FN/TN） |
| `fp_slice_rate` | 阴性切片中被预测出至少 1 个阳性像素的比例 | `FP 切片数 / GT 空切片数` |

- 阈值默认 **0.5**；Phase 2A 增加了多阈值扫描（见 6.1、6.2 节）。
- `eps = 1e-8`。
- 空 mask 规则：空 GT + 空预测 → Dice 1.0；空 GT + 非空预测 → 计入 FP 切片。
- Phase 1 未实现的部分已在 **Phase 2A** 补齐：volume 级 3D Dice/IoU、lesion-wise 指标
  （3D 26-连通域）、per-case 汇总、FP lesions/case、小病灶体积分层、评估期连通域体积过滤。

### 6.1 指标输入类型必须显式声明（Phase 2A 修正）

旧实现用 `arr.min() < 0 or arr.max() > 1` 猜测输入是 logits 还是概率：当 logits 恰好都落在
`[0, 1]` 时会被误判为概率，静默产生错误指标。现在改为**显式 API**：

| 入口 | 语义 |
| --- | --- |
| `metrics.update_logits(x, gt, case_ids)` | 内部 `sigmoid(x) >= threshold` |
| `metrics.update_probabilities(p, gt, case_ids)` | 内部 `p >= threshold`；越界（如误传 logits）直接报错 |
| `metrics.update_binary(b, gt, case_ids)` | `b` 必须是 bool 或 `{0,1}`，否则报错 |
| `metrics.update(x, gt, kind=...)` | 兼容入口，**不传 `kind` 直接 `TypeError`** |

`train.py` 的验证循环与 `evaluate.py` 均改用 `update_logits`；新增测试
`tests/test_metrics_api.py` 锁定行为（含 `logits=0.2 → 0.5498 → positive`、
`logits=-0.2 → negative`、`probability=0.2 → negative` 等用例）。

### 6.2 Phase 2A 评估流程与产物

**只做前向推理，不训练、不改动 checkpoint。** 两步流程：

```bash
# 步骤 1：8 个 checkpoint 的多阈值切片级扫描（一次 forward 同时累计 19 个阈值）
#         其中 E2/E3 的 4 个 checkpoint 额外缓存原始分辨率概率体积（供步骤 2 复用）
python scripts/evaluate_thresholds.py \
    --cache-probs --cache-runs e2_unetr e3_multilevel_fpn \
    --out-dir outputs/evaluation/threshold_scan \
    --cache-root outputs/evaluation/prob_cache

# 步骤 2：读取概率缓存做 volume / lesion 级评估（不重复 forward）
python scripts/evaluate_volumes.py \
    --cache-root outputs/evaluation/prob_cache \
    --out-dir outputs/evaluation/volume \
    --thresholds 0.05 ... 0.95 --primary-thresholds 0.5 \
    --min-volumes-mm3 0 10 25 50 100 250
```

多阈值累计使用**概率直方图法**（`src/metrics/threshold_scan.py`，2000 bins，bin 宽 5e-4）：
阈值网格是 bin 宽的整数倍时，`p >= t` 与「bin 下标 `>= k`」严格等价，因此与逐阈值布尔
实现的结果**完全一致**（由 `tests/test_threshold_scan.py` 断言）。

> **关于 `prob_cache/`**：它是可重建的中间产物（每个 checkpoint 约 4.5 GB，4 个 checkpoint
> 合计约 18 GB），不长期保留在工作区。需要重新做 volume / lesion 分析时，按步骤 1 的命令
> 重新生成即可（约 8 分钟/checkpoint，需要 GPU）；已有的分析结果不会因此丢失
> （`threshold_scan/` 与 `volume/` 下的 CSV / JSON 仍在）。

**分辨率与插入方式**：概率图在 1024×1024 上产生，以 **bilinear** 下采样回 T2W 原始分辨率，
**二值化在原始分辨率上执行**（二值 mask 全程不做插值）；GT 直接取自与 T2W 几何对齐的
原始标注。因此切片级指标（1024 网格）与 volume/lesion 级指标（原始网格）口径略有差异，
报告中必须分别标注。

## 7. 硬件与环境

| 项目 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3090（23.69 GiB，sm_86） |
| Python | 3.10.6（`/root/anaconda3/envs/lm/bin/python`） |
| PyTorch | 2.10.0+cu128 |
| CUDA runtime | 12.8 |
| TensorBoard | 可用 |

**实测显存/吞吐**（`outputs/benchmarks/decoder_benchmark.json`，warmup 5 / repeats 10）：

| 模型 | batch=4 前向 | batch=4 前向+反向 | peak reserved | images/s | batch=8 |
| --- | ---: | ---: | ---: | ---: | --- |
| E0 | 0.821 s | 0.825 s | 11.2 GiB | 4.85 | OOM |
| E1 | 0.760 s | 1.010 s | 11.2 GiB | 3.96 | OOM |
| E2 | 0.823 s | 0.854 s | 11.2 GiB | 4.69 | OOM |
| E3 | 0.865 s | 0.952 s | 11.2 GiB | 4.20 | OOM |

显存峰值由 1024×1024 输入下的 ViT-B 前向主导（E0 仅 257 参数时 fwd 与 fwd+bwd 峰值完全相同），**冻结编码器不节省显存**，只节省其梯度与优化器状态。

DataLoader 基准（纯 CPU，batch=4）：num_workers=2 → 7.87 img/s，4 → 14.80 img/s，8 → **26.30 img/s**（推荐值 8，为当前配置）。

## 8. 复现命令

```bash
# 环境
source /root/anaconda3/envs/lm/bin/activate    # 或直接用绝对路径 python

# 0) 数据准备（一次性；原始数据在仓库外）
python scripts/audit_picai.py
python scripts/build_picai_manifest.py
python scripts/build_splits.py
python scripts/build_mask_cache.py

# 1) 训练前自检（CPU 可跑）
python scripts/preflight_training.py --config configs/e3_multilevel_fpn.yaml

# 2) 单实验训练
python train.py --config configs/e3_multilevel_fpn.yaml

# 3) 串行跑多组（脚本按传入顺序执行，前一个失败即中止）
bash scripts/run_training_sequence.sh e0_linear e1_simple_pyramid e2_unetr e3_multilevel_fpn

# 4) 独立评估（本阶段尚未执行）
python evaluate.py --config configs/e3_multilevel_fpn.yaml \
  --checkpoint outputs/runs/e3_multilevel_fpn/checkpoint_best.pth \
  --threshold 0.5
```

`train.py` 受控参数：`--max-iterations`（冒烟测试）、`--max-val-batches`、`--epochs`、`--batch-size`、`--num-workers`、`--run-name`、`--resume`、`--seed`、`--log-level`。

`evaluate.py` 输出 `<out_dir>/eval_metrics.json`（含 per-case 汇总）与 `volume_predictions/<case_id>_pred.nii.gz`（sigmoid > threshold 后按原 T2W 几何还原成立体预测）。

## 9. 训练执行记录

四组实验在单卡 RTX 3090 上串行完成，命令由项目所有者手动启动：

| 实验 | 开始（本地时间） | 结束 | 墙钟时长 | 启动方式 |
| --- | --- | --- | ---: | --- |
| E0 linear | 2026-09-25 00:39:19 | 2026-09-25 06:03:53 | 5 h 24 min | 手动 |
| E1 simple pyramid | 2026-09-25 06:44:18 | 2026-09-25 12:48:48 | 6 h 05 min | 手动 |
| E2 unetr | 2026-09-25 14:13:22 | 2026-09-25 19:33:28 | 5 h 20 min | `run_training_sequence.sh` |
| E3 multilevel FPN | 2026-09-25 19:33:34 | 2026-09-26 01:43:59 | 6 h 10 min | `run_training_sequence.sh` |

冒烟测试（1–2 epoch、`--max-iterations` 限制）产物位于 `outputs/runs/_smoke_phase1/`，未纳入本报告结论。
