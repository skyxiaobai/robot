# iPhone RGB-D → 小模型训练对比（路径 A 第一张表）

**日期**：2026-10-11 · **机器**：Ubuntu RTX 4090  
**数据**：xhey 播放列表 3 段 iPhone 13 Pro Max + LiDAR mcap → WiLoR+LiDAR 融合（手腕用 LiDAR、手型用 WiLoR）  
**评测**：留一片段；H=16；指标 = `egodex_act_eval.metrics_for` ADE（cm）  
**脚本**：`scripts/iphone_train_compare.py`

## 数据量

| 集合 | 片段 | 帧 | 标签覆盖（QC） | 有效动作比 | 完整 chunk@16 |
|---|---|---|---|---|---|
| QC 通过 2 段 | 08-22, 08-24 | 358 | ~57–61% | 0.64 | ~0.08 |
| 全部 3 段 | +09-05 | 540 | 含 41% 一段 | — | 更低 |

LeRobot 导出（QC 2 段）：`outputs/iphone_mcap/run_all/lerobot`（358 帧，valid_action_ratio≈0.64）。

## 结果：ADE（cm，越低越好）

### QC 通过 2 段（主表）

| 方法 | ADE cm | 相对零动作 | 95% CI |
|---|---|---|---|
| **零动作** | **8.38** | 0 | [7.46, 9.36] |
| 线性 BC（iPhone） | 41.94 | +33.6 | [38.1, 45.6] |
| MLP（iPhone，5 种子） | 24.27 | +15.9 | [23.3, 25.3] |

### 全部 3 段（参考）

| 方法 | ADE cm | 相对零动作 |
|---|---|---|
| **零动作** | **8.54** | 0 |
| 线性 BC | 11.12 | +2.58 |
| MLP | 18.76 | +10.2 |

## 结论（路径 A）

1. **管线能出可训练张量**：mcap → 融合 → episode → LeRobot 已打通；深度尺度修完后覆盖约 53%。
2. **当前 3 段（约 6 秒×3）还训不出优于「手不动」的策略**——和早期 HOT3D 4 段/596 帧结论一致。
3. **原因更像数据量与有效 chunk，不是导出坏了**：完整 16 步 chunk 比例只有 ~8%；两段之间动作分布差，留一上线性/MLP 易过拟合。
4. **下一步（仍属 A）**：再采/再下 10+ 段同类深度 clip；或先降 horizon / 用单步 MSE 做冒烟；图像 ACT 等帧数过千再上。

指标 JSON：`outputs/iphone_mcap/train/metrics.json`、`outputs/iphone_mcap/train_all3/metrics.json`。
