# Phase 2A：阈值扫描 —— 每个 checkpoint 的最优 operating point

> 数据源：`outputs/evaluation/threshold_scan/summary.json`；阈值在 fold0 验证集扫描（**validation 分析，不是 test 性能**）。
> `best-by-pos-Dice` = 在扫描网格上使阳性切片 Dice 最大的阈值；同一行的 precision/recall/FP 率取自该阈值。

| checkpoint | tag | epoch | argmax 阈值 | 阳性切片 Dice | precision | recall | FP-slice rate | micro-Dice 最优阈值 | micro-Dice |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `e0_linear` | best | 40 | 0.40 | 0.3400 | 0.0524 | 0.5404 | 0.8679 | 0.85 | 0.1617 |
| `e0_linear` | latest | 40 | 0.40 | 0.3400 | 0.0524 | 0.5404 | 0.8679 | 0.85 | 0.1617 |
| `e1_simple_pyramid` | best | 8 | 0.35 | 0.4469 | 0.0833 | 0.6768 | 0.9984 | 0.95 | 0.2196 |
| `e1_simple_pyramid` | latest | 40 | 0.05 | 0.3554 | 0.1998 | 0.5708 | 0.9701 | 0.95 | 0.3754 |
| `e2_unetr` | best | 6 | 0.10 | 0.4825 | 0.1176 | 0.6965 | 0.8334 | 0.95 | 0.2823 |
| `e2_unetr` | latest | 40 | 0.05 | 0.3279 | 0.4290 | 0.4401 | 0.4955 | 0.40 | 0.4432 |
| `e3_multilevel_fpn` | best | 6 | 0.65 | 0.4827 | 0.1265 | 0.6987 | 0.9110 | 0.95 | 0.2573 |
| `e3_multilevel_fpn` | latest | 40 | 0.05 | 0.3397 | 0.3067 | 0.4581 | 0.7640 | 0.95 | 0.3840 |
