# MedSAM-PCa 项目约束

本文件是项目的**硬约束清单**，任何脚本、实验与文档都必须遵守。若与本文档冲突，
以本文档为准。

---

## 1. 项目定位

| 项目 | 内容 |
| --- | --- |
| 任务 | 前列腺 MRI 上临床显著前列腺癌（csPCa）病灶的 **2D 轴位切片级分割** |
| 数据 | PI-CAI（1500 case / 1476 patient），Phase 1–2 使用 **T2W 单模态**（复制 3 通道） |
| 方法 | **MedSAM ViT-B 编码器冻结** + 任务专用轻量 decoder（E0–E3） |
| 主指标 | 阳性切片平均 Dice（`positive_slice_dice`，仅统计 GT 非空切片） |
| 阶段协议 | Phase 1：fold0 = 验证，fold1–4 = 训练；5 折划分文件已固定 |

**不做的事**：本仓库不是 MedSAM 的再训练项目，不微调图像编码器（除非作为显式对照实验
并记录在案）。

## 2. 数据约束

1. **原始数据只读**：`/opt/data/private/lm/data/Prostate/PI-CAI` 下的 `.mha` 影像与
   `labels/` 标注**禁止修改、移动、重命名**。
2. mask 与 T2W 几何不一致时，一律以 **T2W 为 reference** 做 nearest-neighbor 重采样，
   结果写入 `data/cache/masks_aligned/`（md5 命名、原子写入），**绝不回写原始标注**。
3. `data/` 下只把**可复现元数据**纳入 Git：`data/manifests/`、`data/splits/`；
   `data/cache/`、影像、权重、`outputs/` 一律排除（见 `.gitignore`）。
4. 影像读取优先按需读单层（`SetExtractIndex`/`SetExtractSize`），避免整卷常驻内存。
5. 划分必须使用 `StratifiedGroupKFold(group=patient_id, stratify=case_csPCa, seed=42)`，
   保证 **patient 不跨折**；任何新划分都要写出 `split_summary.json`。

## 3. 训练约束

1. **正式训练只能由项目所有者手动启动**，AI 助手不得执行 `train.py`、`resume`、
   `optimizer.step()`、`backward()` 或任何形式的重新训练（冒烟测试之外的自动化训练同样禁止）。
2. 训练产物只写 `outputs/runs/<run_name>/`，不得覆盖他人已有 run 目录。
3. checkpoint 只保存**可训练参数**（encoder 已冻结，权重由 `weights/medsam_vit_b.pth`
   恢复）；**已产出的 checkpoint 禁止修改、覆盖或删除**。
4. 固定随机种子（默认 42），并记录到 checkpoint 的 `seed` 字段。
5. 模型选择准则：`best = max(val positive_slice_dice)`，但报告必须同时给出该 epoch 的
   precision / recall / FP-slice rate，禁止只报单一指标。

## 4. 评估与指标约束

1. **指标输入类型必须显式声明**：使用
   `SegmentationMetrics.update_logits()` / `update_probabilities()` / `update_binary()`；
   禁止用数值范围猜测输入是 logits 还是概率。
2. 空 GT 的处理必须显式：`positive_slice_dice` 只统计 GT 非空切片；
   `all_slice_dice` 含"空对空 = 1.0"的情形，**不得单独作为性能结论**。
3. 报告必须区分 **slice 级 / volume 级 / lesion 级** 三种口径，并写明聚合方式
   （逐切片平均 vs 像素级微平均）。
4. 阈值一旦被用于报告，必须说明其来源（默认 0.5 / 扫描得到 / 按指标选定），
   且阈值扫描属于 **validation 分析**，不得表述为最终 test 性能。
5. 假阳性必须单独报告：`fp_slice_rate`（切片级）、FP lesions / case（病灶级）；
   不允许用大量"空对空"样本抬高总体指标。
6. 预测 mask 回原始分辨率一律用 **nearest**；概率体积可用 bilinear；
   **二值 mask 禁止 bilinear**。
7. 评估使用 `torch.no_grad()`，禁止在评估路径中产生梯度。

## 5. 代码与工程约定

1. 所有耗时操作（评估、阈值扫描、volume 重建、连通域分析、写盘、数据处理）
   **必须带 tqdm 进度条**。
2. 张量形状、架构描述、文档中的结构图必须与**真实前向**一致；中间尺寸由
   `tests/test_decoder_shapes.py` 之类的测试锁定，不允许凭直觉描述。
3. 保持历史 checkpoint 可复现：**不修改会改变已有 checkpoint 语义的网络结构**；
   命名不准的地方通过文档澄清而非改架构。
4. 新增依赖需登记；环境为 `python 3.10.6`、`torch 2.10.0+cu128`、单卡 RTX 3090。
5. 单元测试放 `tests/`，只做前向/指标验证，不得在测试中训练。

## 6. 禁止事项（红线）

- 修改原始影像 / 标注 / 已产出的 checkpoint。
- 在未记录的情况下改动数据划分、评估阈值或指标定义。
- 未经允许执行训练、微调编码器或 resume。
- 在报告中给出未由代码产出的数字，或把 validation 结论包装成 test 结论。
- 删除 `.codebuddy/`、`data/`、`outputs/` 中的历史产物。
