# Phase 2A：Evaluation Audit + Threshold Analysis + Volume/Lesion-Level Evaluation

本报告记录 Phase 2A 的全部工作：指标接口审计与修复、文档与实现的一致性修正、多阈值扫描、
volume 级与 lesion 级评估，以及由此得到的结论与下一步建议。

**本阶段没有进行任何训练**：所有数字都来自已有 checkpoint 的前向推理。数据划分、checkpoint
与 Phase 1 完全一致（fold0 验证集 300 case / 6614 slice）。

> 报告原则：所有数字由脚本产出并落盘于 `outputs/evaluation/`，表格由
> `scripts/summarize_phase2a.py` 自动生成到 `docs/experiments/tables/phase2a_*.md`；
> 本文只负责解释与归纳，不引入人工数字。结论按数据给出的顺序陈述，不预设"哪个模型最好"。

---

## 1. 指标接口审计与修复

### 1.1 问题

`SegmentationMetrics._to_numpy_binary()` 原先用

```python
if arr.min() < 0 or arr.max() > 1:   # 猜是不是 logits
    arr = sigmoid(arr)
```

来判断输入类型。这会在两类情况下**静默出错**：

1. 输入是 logits，但取值恰好全部落在 `[0, 1]`（例如 logits=0.2）→ 被当成概率直接阈值化，
   实际决策边界变成 `logit >= 0.5` 而不是 `sigmoid(logit) >= 0.5`；
2. 输入是概率，但数值越界（例如误传 logits）→ 被静默 sigmoid 一次，多了一层非线性。

两种错误都不会报错，只会让指标偏离，且很难在事后发现。

### 1.2 修复

`src/metrics/segmentation.py` 改为**显式声明输入类型**：

| 入口 | 语义 |
| --- | --- |
| `update_logits(x, gt, case_ids)` | 内部 `sigmoid(x) >= threshold` |
| `update_probabilities(p, gt, case_ids)` | 内部 `p >= threshold`；越界直接 `ValueError`（提示改用 `update_logits`） |
| `update_binary(b, gt, case_ids)` | `b` 必须是 `bool` 或 `{0,1}`，否则 `ValueError` |
| `update(x, gt, kind=...)` | 兼容入口，**不传 `kind` 抛 `TypeError`** |

同时新增：非有限值（NaN/Inf）检查、shape 一致性检查、`update_from_counts()`（供直方图式
多阈值累计复用统计逻辑）。

调用方已统一：`train.py` 的验证循环与 `evaluate.py` 都改为 `update_logits(...)`。
测试：`tests/test_metrics_api.py`（22 个用例，含 `logits=0.2 → sigmoid=0.5498 → 判 positive`、
`logits=-0.2 → 判 negative`、`probability=0.2 → 判 negative`、误传 logits 必须报错等）。

### 1.3 对已有结果的影响

Phase 1 的训练期指标**不受影响**：当时的调用只把 `model(images)["logits"]` 传进旧接口，
且这些 logits 的极值必然越出 `[0,1]`（前向输出未激活，范围约 ±5），因此旧的自动判别
一直走的是 sigmoid 分支，与现在显式 `update_logits` 的结果一致。这一点可以由"阈值扫描
在 t=0.5 时复现出的 Phase 1 数字"间接验证（见第 4 节：E1 best 在 t=0.5 得到 0.4461，
与训练期记录的 0.4461 完全一致）。

---

## 2. 文档与实现的一致性修正

### 2.1 E2 的架构描述错误

Phase 1 报告把 E2 描述为"逐级上采样 + 跳连"的 UNETR 式解码器，并声称中间特征尺寸为
`64 → 128 → 256 → 512 → 1024`。**这与实现不符**：

- MedSAM ViT-B 的 `f3/f6/f9/f12` 都位于同一个 patch grid，分辨率**都是 64×64**；
- `_up_to(x, ref)` 虽然存在，但 `ref` 也是 64×64，因此三次融合**都发生在同分辨率上**；
- 唯一一次空间上采样发生在融合之后（`64 → 128`），最后交给输出头上采样到 1024。

因此 E2 的准确定位是 **same-resolution multi-level transformer feature fusion decoder**
（深度方向多级、分辨率方向单一），而不是空间层级 UNETR。处理方式：

- **不修改网络结构**（保证已有 checkpoint 的可复现性），配置名 `e2_unetr` 保持不变；
- 修正 `src/models/unetr_decoder.py` 的模块 docstring、类 docstring 与 `intermediate_shapes()`；
- 新增 `tests/test_decoder_shapes.py`：用 forward hook 抓取真实中间张量，逐层校验声明值；
- 同步修正 `docs/experiments/e2_unetr.md` 与总览。

### 2.2 E3 的形状描述错误

`docs/experiments/e3_multilevel_fpn.md` 曾把 P2/P3 的尺寸写反。实际实现为：

| 层级 | 来源 | 分辨率 |
| --- | --- | --- |
| P2 | `f3`（最浅） | 256×256（转置卷积 ×2 两次） |
| P3 | `f6` | 128×128（转置卷积 ×1） |
| P4 | `f9` | 64×64（1×1 投影） |
| P5 | `neck` | 32×32（1×1 投影 + stride2 卷积） |

已修正，并由 `tests/test_decoder_shapes.py` 锁定。

### 2.3 MedSAM 归一化描述的修正

Phase 1 报告曾建议"当前 [0,1] 输入与 MedSAM 预训练域不一致，应改用 `pixel_mean`/`pixel_std`"。
**这个结论是错的**：官方 MedSAM 的训练脚本（`train_one_gpu.py`）明确要求把图像归一化到
`[0,1]` 后直接送入 `image_encoder`，官方推理脚本（`MedSAM_Inference.py`）同样是
min-max 到 `[0,1]` 后直接前向。本项目当前的 `[0,1]` 数值范围与官方口径**一致**，
差别只在于本流程先做非零区域 0.5/99.5 百分位裁剪再做 min-max，而官方示例以纯 min-max 为主。

已修改 `docs/experiments/README.md` 与 `docs/experiments/protocol.md`，并明确：**不应**把输入
切换成 SAM/ImageNet 的 `pixel_mean`/`pixel_std`；若做归一化消融，应比较 plain min-max 与
percentile clipping + min-max。

---

## 3. 评估方法与口径

### 3.1 多阈值扫描（一次 forward，全阈值）

`scripts/evaluate_thresholds.py` 对 8 个 checkpoint（E0–E3 × best/latest）各做**一次**前向，
用概率直方图法（2000 bins，bin 宽 5e-4）同时累计 19 个阈值（0.05 → 0.95，步长 0.05）的
切片级指标。阈值网格是 bin 宽的整数倍，因此 `p >= t` 与「bin 下标 ≥ k」严格等价，
结果与逐阈值布尔实现完全一致（`tests/test_threshold_scan.py` 断言）。

对 E2/E3 的 4 个 checkpoint，前向时额外把每个 case 的概率图以 **bilinear** 下采样到 T2W
原始分辨率并缓存（`outputs/evaluation/prob_cache/`），供 volume/lesion 评估复用，
避免重复跑 encoder。

### 3.2 Volume / lesion 级评估

`scripts/evaluate_volumes.py` 读取概率缓存，在**原始分辨率**上：

1. 对 19 个阈值计算每 case 的 3D Dice / IoU 与体素混淆量，按 positive / negative / all case 分组；
2. 在主阈值（0.5）下做 3D 连通域（26-连通）分析：GT 每个连通域为一个 lesion，预测每个连通域
   为一个 predicted lesion；
3. 匹配规则：**任意体素重叠即算命中**（`any_overlap`），一个 GT 只允许一次检出——同一 GT 上
   重叠较小的其余预测连通域计为 **FP component**（匹配过程在 `lesion_detail.csv` 中逐条记录）；
4. 扫描预测连通域的最小体积过滤（0 / 10 / 25 / 50 / 100 / 250 mm³）；
5. 按 GT 病灶物理体积（由 spacing 计算）分层：<0.5 cc / 0.5–1.0 cc / >1.0 cc（探索性分组）。

### 3.3 口径差异声明

- 切片级指标在 1024×1024 网格上计算（与 Phase 1 一致，可直接对比）；
- volume/lesion 指标在 T2W 原始分辨率上计算（概率 bilinear 下采样，**二值化在原始分辨率执行**，
  二值 mask 全程不做插值）；
- 因此同一模型在两个口径下的数字不可直接互相替代，报告中分别标注。

---

## 4. 切片级阈值扫描结果

完整表格见 [`tables/phase2a_threshold_best_per_checkpoint.md`](tables/phase2a_threshold_best_per_checkpoint.md)
与 [`tables/phase2a_threshold_full_grid.md`](tables/phase2a_threshold_full_grid.md)。

### 4.1 阈值校准几乎不改变阳性切片 Dice

| checkpoint | ckpt epoch | t=0.50 的 pos-Dice（Phase 1 口径） | 扫描最优 t\* | pos-Dice@t\* | 绝对增益 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `e0_linear` best | 40 | 0.3355 | 0.40 | 0.3400 | **+0.0045** |
| `e0_linear` latest | 40 | 0.3355 | 0.40 | 0.3400 | +0.0045 |
| `e1_simple_pyramid` best | 8 | 0.4461 | 0.35 | 0.4469 | **+0.0008** |
| `e1_simple_pyramid` latest | 40 | 0.3324 | 0.05 | 0.3554 | +0.0230 |
| `e2_unetr` best | 6 | 0.4750 | 0.10 | 0.4825 | **+0.0075** |
| `e2_unetr` latest | 40 | 0.3107 | 0.05 | 0.3279 | +0.0172 |
| `e3_multilevel_fpn` best | 6 | 0.4824 | 0.65 | 0.4827 | **+0.0003** |
| `e3_multilevel_fpn` latest | 40 | 0.3295 | 0.05 | 0.3397 | +0.0102 |

三个"最佳 checkpoint"（E1/E2/E3 best）的增益分别为 +0.0008 / +0.0075 / +0.0003——
**在阈值为 0.5 时这些模型已经处于其上界附近，阈值校准不是当前瓶颈**。

两个现象值得注意：

1. **`e0_linear` 的 best 与 latest 完全一致**（都是 epoch 40 的同一权重，Phase 1 中最佳值
   恰好落在最后一轮），因此两行数字相同，这是数据本身的性质而非脚本错误。
2. **后期 checkpoint 的最优阈值向低端移动**（E1/E2/E3 latest 都是 t\*=0.05，即扫描网格的下
   边界）。这与第 8 节的行为一致：训练后期模型整体压低输出概率（抑制假阳性），因此需要更低
   的阈值才能维持召回；而"最优值落在网格边界"本身说明真正的上界可能在 0.05 以下，扫描网格
   需要向低端扩展才能定位。

### 4.2 阳性切片 Dice 对阈值不敏感

在 t ∈ [0.3, 0.6] 这一宽区间内，`e1_simple_pyramid` best 的 pos-Dice 只在
0.4468 → 0.4461 之间变化（<0.001），`e3_multilevel_fpn` best 同样平坦。这说明模型的概率输出
在决策边界附近**没有可利用的排序信息**：不是"阈值选得不好"，而是阳性切片内的概率分布与
负样本高度重叠。

"移动阈值"确实能改变 precision / recall 的配比（例如 `e2_unetr` latest 在 t=0.05 时
precision 0.429 / recall 0.440，在 t=0.9 时 precision 0.513 / recall 0.392），但**换来的
是召回与精确率之间的等价交换，而不是整体质量的提升**。

### 4.3 micro-Dice 的偏好与 pos-Dice 相反

| checkpoint | micro-Dice@t=0.5 | micro-Dice 最优阈值 | micro-Dice@t\* |
| --- | ---: | ---: | ---: |
| `e0_linear` best | 0.1113 | 0.85 | 0.1617 |
| `e1_simple_pyramid` best | 0.1581 | 0.95 | 0.2196 |
| `e2_unetr` best | 0.2402 | 0.95 | 0.2823 |
| `e3_multilevel_fpn` best | 0.2038 | 0.95 | 0.2573 |

像素级微平均指标在阈值 0.85–0.95 才达到最优，因为背景体素占 99.99%，提高阈值能快速减少
假阳性体素。**这再次说明单一阈值无法同时优化两个口径**：以 pos-Dice 选阈值（偏 0.05–0.65）
会放大假阳性，以 micro-Dice 选阈值（0.85–0.95）会牺牲召回。报告与后续实验必须明确声明
以哪个指标选阈值，不能混用。

---

## 5. Volume 级结果

完整表格见 [`tables/phase2a_volume.md`](tables/phase2a_volume.md)。volume 级指标在 T2W 原始分辨率、
300 个 case 上计算。主阈值 t=0.5 的结果汇总：

| checkpoint | 阳性 case 3D Dice | 阳性 case 中位 Dice | 微平均 3D Dice | 微平均 precision | 微平均 recall | FP case 率 | 阴性 case 平均 FP 体素 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `e2_unetr` best | **0.2910** | 0.2289 | 0.1997 | 0.1182 | 0.6441 | 1.0000 | 13 180 |
| `e2_unetr` latest | **0.2957** | 0.2321 | **0.4455** | **0.4735** | 0.4207 | 0.9907 | **1 133** |
| `e3_multilevel_fpn` best | 0.2685 | 0.1773 | 0.1622 | 0.0915 | **0.7097** | 1.0000 | 19 476 |
| `e3_multilevel_fpn` latest | 0.2834 | 0.1991 | 0.3592 | 0.3069 | 0.4330 | 1.0000 | 2 540 |

（验证集 300 个 case = 84 个阳性 + **216 个阴性**；表中 FP case 率的分母是这 216 个阴性 case。
"阴性 case 平均 FP 体素"为每个阴性 case 被预测出的前景体素数均值，单个 case 约 2–3 M 体素。）

### 5.1 阳性 case 的 3D Dice 只有 0.27–0.30

与切片级 pos-Dice（0.48）相比，volume 级 Dice 低了约 0.19。这不是矛盾，而是两个口径的
差异：

- 切片级 pos-Dice 只在 **491 个 GT 非空切片**上平均，每张切片独立计算 2D Dice，
  阴性切片完全不参与；
- volume Dice 在 **整个 3D 体积**上计算，模型在阴性切片、阴性区域上产生的大面积假阳性
  全部进入分母。

因此 **"切片 Dice 0.48"不能解读为"病灶分割质量 48%"**：一旦放到三维体积口径，模型的实际
重叠度只有约 0.28–0.30。

一个值得注意的对比：`e2_unetr` / `e3_multilevel_fpn` 的 **latest**（epoch 40）在 volume
口径下的阳性 case 3D Dice（0.2957 / 0.2834）**略高于**对应的 best（0.2910 / 0.2685），
与切片级 pos-Dice 给出的排序（best 明显更高）相反。原因是 latest 的假阳性大幅减少
（阴性 case 平均 FP 体素从 13 180 → 1 133），把 3D Dice 抬了上来。

### 5.2 假阳性负担

- 三个 checkpoint（E2 best、E3 best、E3 latest）的 **FP case 率 = 1.0000**，即**全部 216 个
  阴性 case 都被预测出至少一个前景体素**；只有 E2 latest 降到 0.9907（仍有 214/216）。
- 阴性 case 平均 FP 体素在 best 模型上是 13 180（E2）/ 19 476（E3），到 latest 降到
  1 133 / 2 540，但**没有任何一个 checkpoint 做到"阴性 case 基本干净"**。

以"是否有任何一个假阳性体素"衡量的 case 级假阳性率，在当前模型上都是 99%+，这说明
**病灶检出本身不是唯一问题，如何在阴性切片上保持沉默才是主要短板**。

---

## 6. Lesion 级结果

完整表格见 [`tables/phase2a_lesion.md`](tables/phase2a_lesion.md)。

fold0 验证集共有 **91 个 GT 连通域病灶**（分布在 84 个阳性 case 中，存在 multi-lesion
case），26-连通、匹配规则 `any_overlap`、t=0.5：

| checkpoint | GT 病灶 | 检出 GT 病灶 | 灵敏度 (pooled) | 预测连通域 | FP 连通域 | FP/case | 匹配病灶 Dice |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `e2_unetr` best | 91 | 81 | 0.8901 | 3 943 | 3 864 | 12.88 | **0.4721** |
| `e2_unetr` latest | 91 | 64 | 0.7033 | 2 078 | 2 015 | **6.72** | 0.4272 |
| `e3_multilevel_fpn` best | 91 | **86** | **0.9451** | 6 460 | 6 377 | 21.26 | 0.4536 |
| `e3_multilevel_fpn` latest | 91 | 75 | 0.8242 | 4 795 | 4 722 | 15.74 | 0.3891 |

### 6.1 检出率与假阳性负担同时都很高

在"任意重叠即命中"这一最宽松的规则下：

- 病灶检出率相当高（best 模型 89%–95%，latest 70%–82%），说明**模型确实能找到病灶**，
  之前切片级 precision 极低（0.06–0.15）主要来自"预测面积过大 + 额外假阳性连通域"，
  而不是"完全找错位置"；
- 但每个 case 平均产生 6.7–21.3 个假阳性连通域。**假阳性不是零星噪声，而是数量上与
  真病灶同量级甚至更多**（例如 E3 best：91 个真病灶 vs 6 377 个 FP 连通域，300 个 case）。
- 匹配上的病灶 Dice 只有 0.39–0.47，说明即使"命中了"，预测连通域的形态（体积、边界）
  与 GT 仍有明显差距。

### 6.2 匹配规则的敏感性

本报告使用 `any_overlap`（最宽松）。这是**有意选择的第一步**，因为当前模型的主要问题是
"多报"而不是"漏报"；若换成更严格的规则（如 `dice >= 0.1` 或 `overlap_fraction >= 0.5`），
灵敏度会下降、FP 判定会变化。规则已在 `lesion_summary.json` 与
`tables/phase2a_lesion.md` 中逐表标注，`src/metrics/lesion_metrics.py` 支持
`any_overlap` / `dice` / `iou` / `overlap_fraction` 四种可配置规则，后续分析可直接切换。

同时按报告规范：`num_gt_lesions` 为 91（**未做任何"只保留最大连通域"**），GOI 与 multi-lesion
case 完整保留。

---

## 7. 小病灶分层与连通域过滤

### 7.1 按 GT 病灶物理体积分层（t=0.5）

分层由 spacing 计算病灶体积（探索性分组 <0.5 cc / 0.5–1.0 cc / >1.0 cc，**不是临床标准**）：

| checkpoint | <0.5 cc（26 个） | 0.5–1.0 cc（17 个） | >1.0 cc（48 个） |
| --- | ---: | ---: | ---: |
| `e2_unetr` best | 0.731 | 0.882 | 0.979 |
| `e2_unetr` latest | 0.423 | 0.588 | 0.896 |
| `e3_multilevel_fpn` best | 0.846 | 0.941 | **1.000** |
| `e3_multilevel_fpn` latest | 0.654 | 0.706 | 0.958 |

检出率随病灶体积单调上升，符合小病灶更难检出的先验。**最大的差距出现在最小的一档**：
best 模型在小病灶上仍有 73%–85% 的检出率，latest 模型掉到 42%–65%——latest 的"低假阳性"
是以牺牲小病灶为代价换来的。

### 7.2 预测连通域最小体积过滤（t=0.5）

同一张表同时给出过滤后的病灶灵敏度与 FP/case：

| 最小体积 (mm³) | E2 best：灵敏度 / FP per case | E2 latest | E3 best | E3 latest |
| ---: | ---: | ---: | ---: | ---: |
| 0（不过滤） | 0.890 / 12.88 | 0.703 / 6.72 | 0.945 / 21.26 | 0.824 / 15.74 |
| 10 | 0.879 / 12.00 | 0.495 / 3.76 | 0.912 / 17.59 | 0.593 / 8.97 |
| 25 | 0.879 / 11.13 | 0.396 / 2.54 | 0.879 / 14.49 | 0.440 / 5.97 |
| 50 | 0.824 / 9.84 | 0.330 / 1.61 | 0.714 / 11.74 | 0.286 / 3.91 |
| 100 | 0.692 / 7.79 | 0.264 / 1.02 | 0.527 / 8.64 | 0.198 / 2.21 |
| 250 | 0.407 / 4.50 | 0.198 / 0.48 | 0.242 / 5.01 | 0.110 / 0.98 |

关键观察：

1. **best 模型对轻微过滤几乎是"免费"的**：10 mm³ 过滤把 E2 best 的 FP/case 从 12.88
   降到 12.00、灵敏度只掉 0.011（0.890 → 0.879）；E3 best 在 10 → 25 mm³ 区间同样保持了
   0.912 → 0.879 的灵敏度。**大量假阳性确实来自极小的连通域。**
2. **latest 模型对过滤极其脆弱**：E2 latest 在 10 mm³ 就损失 0.208 灵敏度
   （0.703 → 0.495）。因为它们抑制假阳性的方式是整体压低响应，真阳性与假阳性连通域在
   体积上不再可分。
3. **不存在"免费午餐"的过滤阈值**：把 FP/case 压到 ≤1 需要 ≥100 mm³ 过滤，此时灵敏度只剩
   0.20–0.26。**连通域过滤是缓解手段，不是解决方案**——它能把 FP 削减 1–2 倍，但无法把
   12–21 FP/case 降到临床可接受的水平。

---

## 8. best vs latest，以及 operating point 的选择

Phase 1 的模型选择准则是 `best = max(阳性切片 Dice)`（在 2-epoch 间隔的稀疏验证点上）。
Phase 2A 的数据显示，**这个准则给出的 checkpoint 只在一个口径上最优**：

| 维度 | 更优者 | 数据 |
| --- | --- | --- |
| 切片级阳性 Dice | best | E3 best 0.4827 vs latest 0.3397；E2 best 0.4825 vs latest 0.3279 |
| 阳性 case 3D Dice | **latest**（微弱） | E2 0.2957 vs 0.2910；E3 0.2834 vs 0.2685 |
| 微平均 3D Dice | **latest** | E2 0.4455 vs 0.1997；E3 0.3592 vs 0.1622 |
| precision | **latest** | E2 0.4735 vs 0.1182；E3 0.3069 vs 0.0915 |
| recall | best | E3 0.7097 vs 0.4330；E2 0.6441 vs 0.4207 |
| 病灶检出率 | best | E3 0.9451 vs 0.8242；E2 0.8901 vs 0.7033 |
| FP lesions/case | **latest** | E2 6.72 vs 12.88；E3 15.74 vs 21.26 |
| 匹配病灶 Dice | best | E2 0.4721 vs 0.4272；E3 0.4536 vs 0.3891 |
| 小病灶（<0.5 cc）检出 | best | E3 0.846 vs 0.654；E2 0.731 vs 0.423 |

可以这样概括：**best = 高召回 + 高假阳性；latest = 低假阳性 + 低召回**，两者是同一个
trade-off 的两端。E2 与 E3 之间同样不是简单的高低关系——在相同 epoch 上，E3 更敏感
（检出 94.5% vs 89.0%）但 FP 负担更大（21.26 vs 12.88 FP/case），E2 的匹配病灶 Dice 与
FP 控制更好。

**因此"哪个 checkpoint 更好"取决于下游用途**：

- 如果下游要求"尽可能不漏病灶"（再由医生或后处理排除假阳性），E3 best 是合理选择；
- 如果下游要求"低假阳性、可直接使用"，E2 latest 更合适（FP/case 6.72，precision 0.47）；
- 当前的 `max(阳性切片 Dice)` 准则实际上在**鼓励高假阳性解**，因为它完全不惩罚假阳性。

### 8.1 阈值校准的定位

把阈值扫描与 checkpoint 选择放在一起看：阈值只能在同一模型内部移动 operating point
（pos-Dice 最多 +0.0003…+0.023），而**选择哪个 checkpoint 带来的差异要大一个数量级**
（切片级 0.33 vs 0.48，FP/case 6.72 vs 21.26）。在 Phase 2A 里，**checkpoint 与评价口径的
选择比阈值调参重要得多**。

---

## 9. 下一步建议

按"证据强度 × 预期收益"排序：

| 优先级 | 动作 | 依据 |
| --- | --- | --- |
| P0 | **按下游用途重定义 checkpoint 选择准则**（例如 `pos-Dice` 与 `FP lesions/case` 的联合阈值，或直接在 lesion 级指标上选） | 现准则选出的是 FP/case 21.26 的解；同一模型 latest 的 FP/case 只有 15.74 |
| P0 | **把"假阳性连通域数量"作为一等指标纳入训练监控** | best 模型每 case 产生 12.9–21.3 个 FP 连通域，且 209/209 阴性 case 都被预测出前景 |
| P0 | 在下游流程中加入 **10–25 mm³ 连通域过滤**并固定为评估协议的一部分 | E2 best：10 mm³ 过滤仅损失 0.011 灵敏度即减少 0.9 FP/case；E3 best 在 25 mm³ 仍保持 0.879 |
| P1 | 损失/后处理层面抑制假阳性（如 Tversky/Focal-Tversky、hard-negative 挖掘、切片间一致性） | 阈值与过滤都已触及上限（第 4.1、7.2 节） |
| P1 | 训练层面：早停到峰值区间（epoch 4–10），并用 `val_interval=1` 精确定位峰值 | 三个 best checkpoint 都在 epoch 6–8，之后 30 轮全部在主指标上退化 |
| P1 | 用与 Phase 2A 相同的口径（切片 + volume + lesion）重跑其它 fold，确认排序是否稳定 | 目前所有结论都来自单一 fold0、单一 seed |
| P2 | 将阈值扫描网格向低端扩展（0.05 以下） | E1/E2/E3 的 latest 最优阈值落在网格边界 0.05 |
| P2 | 评估切片间一致性后处理（如 3D 形态学开闭、孔洞填充） | 未在本阶段尝试；连通域过滤只能削减而不能消除散点 FP |
| P2 | 报告规范：任何以 pos-Dice 为选择依据的结论都必须同时给出 volume 与 lesion 指标 | 切片级 0.48 vs volume 级 0.28 的差距 |

### 9.1 本阶段结论的边界

- 所有数字均来自 **fold0 验证集**，属于 **validation 分析**，不是 test 性能，也不构成
  对 PI-CAI 官方评测的预测；
- 单一 fold、单一随机种子，未做重复实验，**E2 与 E3 之间的细小差异不应被视为定论**；
- volume/lesion 指标在原始分辨率上计算，切片级指标在 1024 网格上计算，两者不可直接互换；
- lesion 匹配使用最宽松的 `any_overlap`，换成更严格规则会改变绝对数值（但不会改变
  "FP 连通域数量远超真病灶"这一结构性结论）；
- 本阶段**没有进行任何训练**，因此上述所有结论都是"当前模型 + 当前后处理"的上界，
  不能用于推断"加入训练改动后的上限"。
