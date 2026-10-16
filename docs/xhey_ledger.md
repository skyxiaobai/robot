# XHEY 实验台账（自动迭代）

目标：**手腕 ADE 稳定低于「手不动」基线（overall 11.03 cm / 手在动 14.08 cm）**。  
评测固定：3 段 iPhone LiDAR 留一法，seeds=3，horizon=16 @15fps。数字越小越好。

> **坐标约定变更（PR #55）：** 从这版起，XHEY 的 iPhone 片段在进训练前会规范到相机光学系（OpenCV `camera`：x 右、y 下、z 前），不再沿用之前的世界系。上面 overall **11.03 cm** / 手在动 **14.08 cm** 是世界系下的「手不动」基线，**不能和这次改动之后的结果直接比**。需要在相机系下重新跑一遍 zero，立一条新基线。

---

## 已完成（种子台账）

| id | 日期 | 配比 / 改动 | Overall ADE | 手在动 ADE | vs zero | 结论 | 出处 |
|---|---|---|---:|---:|---:|---|---|
| E0 | — | zero（不动） | 11.03 | 14.08 | 0 | 基线 | 每次跑都带 |
| E1 | 10-13 | iPhone-only | 11.29 | 14.19 | +0.27 | 差 | PR #47 / Kaggle |
| E2 | 10-13 | iPhone+aug | 11.19 | — | +0.16 | 差 | PR #47 |
| E3 | 10-13 | iPhone+xhey_rgb | 11.11 | — | +0.08 | 接近 | PR #47 |
| E4 | 10-13 | **iPhone+xhey_rgb+aug** | **11.06** | **14.06** | **+0.04** | 目前最好，仍略差于 zero | PR #47 |
| E5 | 10-14 | +OpenAoE nano | 11.25 | 14.19 | +0.22 | **不帮** | `docs/openaoe_pretrain.md` |
| E6 | 10-14 | +OpenAoE nano +xhey_rgb | 11.23 | 14.18 | +0.20 | **不帮** | 同上 |
| E7 | 10-14 | +OpenAoE tiny | 11.24 | 14.17 | +0.21 | **不帮** | 同上 |
| E8 | 10-14 | +OpenAoE tiny +xhey_rgb | 11.25 | 14.18 | +0.22 | **不帮** | 同上 |

机器可读：[`docs/xhey/ledger.csv`](xhey/ledger.csv)（由 `experiments/ledger.jsonl` 镜像，手写的 E/C 行保留）。日更图：[`docs/xhey/daily_progress.png`](xhey/daily_progress.png)。进度图：[`experiments/progress.png`](../experiments/progress.png)。

**C10（2026-10-15）**：C10 = 按质检分段裁掉坏片段后的 iPhone 训练（`qc_segments: true`）；C10ctl = 同一套代码、不裁剪的对照（`qc_segments: false`）。两边用同一 3 段留出评测片段，内核都是 `justinz2010/xhey-rsi-c10-segtrim`。这是相机系结果（zero overall 9.78 cm），不能和上面世界系 11.03 cm 基线直接比。C10 整体 ADE 9.735 cm（比 zero −0.044）、手在动 12.599 cm（−0.084）；C10ctl 整体 9.759 cm（−0.020）、手在动 12.546 cm（−0.137）。两边的 95% 区间都在 0 以下。

**C11（2026-10-15）**：C11 = 加 DynamicIntelligence（DI，已转相机坐标系、质检 25/25 通过）做预训练（`iphone+di+xhey_rgb+aug`）；C11ctl = 同一内核、同一套代码、不加 DI 的对照（`iphone+xhey_rgb+aug`）。两边用同一 3 段留出评测片段，内核 `justinz2010/xhey-rsi-c11-di-cam`，3 个种子。C11 整体 ADE 9.772 cm（比 zero −0.008，95% 区间 [−0.015, −0.000]）、手在动 12.551 cm（−0.132，[−0.141, −0.122]）；C11ctl 整体 9.759 cm（−0.020，[−0.027, −0.014]）、手在动 12.546 cm（−0.137，[−0.146, −0.129]）。结论：加 DI 没有比不加更好（整体差 +0.013 cm、手在动差 +0.005 cm，都略差），两边都只比「猜手不动」好约 1%。对照图：[`experiments/report/C11_vs_ctl/`](../experiments/report/C11_vs_ctl/)。

**C9（已完成，未质检 / 世界系）**：内核 `justinz2010/xhey-rsi-c9-stera-di`（2026-10-15 21:24 提交）没有 `--qc-filter`，DI 用的是 `xhey-di-pretrain` 的原始世界系，不是 C11 的相机系。未过滤 3 段留出：zero 11.026 cm；C9ctl（`iphone+xhey_rgb+aug`）11.062 cm（+0.037）、手在动相对 zero −0.021；C9（`iphone+stera+di+xhey_rgb+aug`）11.100 cm（+0.075）、手在动相对 zero −0.005。C9 整体和手在动都不如同一次对照，队列结论 **done_worse**。没有另给绝对手在动 ADE、置信区间或逐片段数字，账本不补这些。`ledger.csv` 的 status 列按镜像规则写成 better_moving，只表示手在动相对 zero 为负、且没有置信区间；这不是实验结论。质检 + 相机系的重跑是 C9b，已经在 [Kaggle](https://www.kaggle.com/code/justinz2010/xhey-rsi-c9b-stera-di-qc) 上跑，不要再交一份。

**C12（2026-10-16）**：C12 = HOI4D 预训练（MANO GT 手、相机光学系，`iphone+hoi4d+xhey_rgb+aug`），内核 `justinz2010/xhey-rsi-c12-hoi4d`。评测集与 C11ctl 相同（iPhone-only 对照，手在动 −0.137）。C12 整体 ADE 9.845 cm（比 zero 9.779 +0.066）、手在动 12.619 cm（−0.064，95% 区间 [−0.071, −0.056]）。HOI4D 在手在动上赢了「猜手不动」，但没有超过不加 HOI4D 的对照，整体更差。队列结论 **done_worse_vs_ctl**。图：[`experiments/report/C12/`](../experiments/report/C12/)。

唯一循环驱动是 `scripts/xhey_auto_iter.py`：先收已完成的固定内核，再按 [`docs/xhey/queue.json`](xhey/queue.json) 提交就绪候选。新候选默认必须带 `--qc-filter`。正在占 GPU 的是 **C9b**（2026-10-16 09:23 因 main 仍显示 ready，循环重交了 kernel version 2）；有内核在跑时不抢占。

---

## 候选下一轮（预排序）

便宜代理：同一评测、`--seeds 1 --steps 800 --pre-steps 1500`（本地 CPU 或短 Kaggle），看 vs-zero 符号再上全预算。

| 优先 | id | 假设 | 为什么排前 | 预算 |
|---:|---|---|---|---|
| 1 | C1 | **只在「手在动」帧上算损失**（`--move-cm 5` 加权） | 最好配比在 moving 上已微赢 zero；整体被静止帧拖累 | 短训→全量 |
| 2 | C2 | **加长微调**（steps 8k–12k，E4 配比） | 现 3k 可能欠拟合；改动最小 | 全量 Kaggle |
| 3 | C3 | **残差缩放 / 去 shrink** 或直接预测绝对手腕 | 模型可能被 shrink 压成「几乎不动」 | 短训对比 |
| 4 | C4 | **解冻 ResNet 最后 block** + 更强图像条件 | 有画面的 xhey_rgb 有帮助，图像侧可能还弱 | 全量 |
| 5 | C5 | 再采 **≥10 段 iPhone Pro 深度**（数据，不是训） | 3 段 LOO 方差大；论文路线本来就要加数据 | 采集 |
| 6 | C6 | 停做更大 Open-AoE / 无画面姿态预训练 | E5–E8 已证伪 | 跳过 |
| 7 | C7 | **Stera-10M 预训练 → XHEY 精调**（iPhone Pro LiDAR + ARKit 重力世界系，WiLoR 21 点） | 设备/坐标系和 XHEY 最像，直接避开 Open-AoE 的两个疑似失败原因（无重力、手是重建的） | 全量 Kaggle；**卡在 gated 授权** |
| 8 | C8 | **DynamicIntelligence 预训练 → 精调**（Record3D iPhone 13 Pro，MediaPipe+深度，每手 3 点） | 已就绪；但只有 25 段「手在动」、约 4 分钟，姿态不可用（相机系），预期增益小 | 全量 Kaggle |
| 9 | C9 | C7 + C8 一起预训练 | 看两份是否叠加 | 等 C7 |

**新的提交顺序**（机器可读：[`docs/xhey/queue.json`](xhey/queue.json)）：C2 → **C8** →（Stera 授权后）C7 → C9。C1 不抢占。  
数据来源与许可见 [`docs/public_iphone_lidar.md`](public_iphone_lidar.md)。

本轮启动：**C1、C2**（见 Kaggle slug）；C3 做短代理后再决定。

---

## 自动迭代规则（给 bot）

1. 台账追加一行，不改历史数字。`xhey_auto_iter.py` 把 `experiments/ledger.jsonl` 镜像进 `docs/xhey/ledger.csv`。  
2. 候选预排序 → 最多同时 1 个 Kaggle GPU kernel。固定队列（C2、C8；Stera 就绪后 C7、C9）先于提议器。  
3. 跑完：更新 CSV + 重画进度图 + 写一句结论。  
4. ADE 低于 zero 且 CI 不跨 0 → 标「胜出」，开 PR。  
5. 连续 3 次同方向失败 → 该方向降权或关掉。
