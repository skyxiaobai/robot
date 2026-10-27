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

**C9（已完成，未质检 / 世界系）**：内核 `justinz2010/xhey-rsi-c9-stera-di`（2026-10-15 21:24 提交）没有 `--qc-filter`，DI 用的是 `xhey-di-pretrain` 的原始世界系，不是 C11 的相机系。未过滤 3 段留出：zero 11.026 cm；C9ctl（`iphone+xhey_rgb+aug`）11.062 cm（+0.037）、手在动相对 zero −0.021；C9（`iphone+stera+di+xhey_rgb+aug`）11.100 cm（+0.075）、手在动相对 zero −0.005。C9 整体和手在动都不如同一次对照，队列结论 **done_worse**。没有另给绝对手在动 ADE、置信区间或逐片段数字，账本不补这些。`ledger.csv` 的 status 列按镜像规则写成 better_moving，只表示手在动相对 zero 为负、且没有置信区间；这不是实验结论。质检 + 相机系的重跑是 C9b，结果见下，不要再交一份。

**C12（2026-10-16）**：C12 = HOI4D 预训练（MANO GT 手、相机光学系，`iphone+hoi4d+xhey_rgb+aug`），内核 `justinz2010/xhey-rsi-c12-hoi4d`。评测集与 C11ctl 相同（iPhone-only 对照，手在动 −0.137）。C12 整体 ADE 9.845 cm（比 zero 9.779 +0.066）、手在动 12.619 cm（−0.064，95% 区间 [−0.071, −0.056]）。HOI4D 在手在动上赢了「猜手不动」，但没有超过不加 HOI4D 的对照，整体更差。队列结论 **done_worse_vs_ctl**。图：[`experiments/report/C12/`](../experiments/report/C12/)。

**C9b（2026-10-16）**：C9b = Stera 五批 + DI（相机系脚本）+ iPhone + xhey_rgb + aug，内核 `justinz2010/xhey-rsi-c9b-stera-di-qc`。评测是质检后的 2 段留出（`2026-08-22_18-21`、`2026-08-24_21-18`），数字来自 `xhey_c9b/metrics.json` 的 `ade_cm_pooled`，不是日志里四舍五入到两位的那一行。zero 整体 7.227 cm；C9bctl（`iphone+xhey_rgb+aug`）7.221 cm（−0.006）、手在动相对 zero −0.068；C9b（`iphone+stera+di+xhey_rgb+aug`）7.235 cm（+0.007，95% 区间 [0.004, 0.011]）、手在动 10.383 cm（−0.050，[−0.056, −0.044]）。口头报的整体 7.24 对不上：pooled 是 7.234595，内核日志印成 7.23。C9b 整体和手在动都不如同一次对照，整体还差于 zero。和 C12 同一条规则（没超过对照，且整体比 zero 差）队列结论 **done_worse_vs_ctl**。C11 是另一条：没超过对照，但整体仍不差于 zero，所以是 done_no_gain_vs_ctl。`ledger.csv` 的 status 列按镜像规则是 win_moving，只表示手在动的置信区间整段落在 0 以下；这不是实验结论。内核日志里 DI 的 25 段不在传给 `--qc-filter` 的成绩单里，被原样保留。

**C13（2026-10-17）**：C13 = EgoDex 预训练（相机光学系，Vision Pro ARKit 关节，`iphone+egodex+xhey_rgb+aug`），内核 `justinz2010/xhey-rsi-c13-egodex`（v2，2026-10-17 09:20 重交，21:27 入账）。同一次对照 C13ctl 不加 EgoDex（`iphone+xhey_rgb+aug`）。C13 整体 ADE 9.803 cm（比 zero 9.779 +0.024）、手在动 12.557 cm（−0.126，95% 区间 [−0.134, −0.117]）；C13ctl 整体 9.759 cm、手在动相对 zero −0.137（[−0.146, −0.129]）。EgoDex 在手在动上赢了「猜手不动」，但没有超过同一次对照，整体更差。队列结论 **done_worse_vs_ctl**。`ledger.csv` 的 status 列按镜像规则是 win_moving，只表示手在动的置信区间整段落在 0 以下；这不是实验结论。图：[`experiments/report/C13/`](../experiments/report/C13/)。

**C14（2026-10-18）**：C14 = xPhi-Ego-100 预训练（单目 WiLoR 伪标签，假定 HFOV 90° 内参，`iphone+xphi+xhey_rgb+aug`），内核 `justinz2010/xhey-rsi-c14-xphi`（2026-10-17 21:27 提交，2026-10-18 09:22 入账）。同一次对照 C14ctl 不加 xPhi（`iphone+xhey_rgb+aug`）。xPhi 臂整体 ADE 9.750 cm，同一次对照 C14ctl 9.759 cm，zero 9.779 cm；手在动相对 zero −0.149 cm（95% 区间 [−0.159, −0.139]），对照 −0.137 cm（[−0.146, −0.129]）。这是第一份超过同一次对照的外部数据集，优势约 0.012 cm；标签是单目 WiLoR 伪标签，内参按 HFOV 90° 假定。队列结论 **done_win**。`ledger.csv` 的 status 列按镜像规则是 win_moving，只表示手在动的置信区间整段落在 0 以下；队列结论才是 done_win。图：[`experiments/report/C14/`](../experiments/report/C14/)。

**C16（2026-10-19）**：C16 = EgoVerse/Mecka 预训练（MANO GT 关键点 + 头部位姿 + 真实内参，原生 10 Hz，相机光学系，`iphone+egoverse+xhey_rgb+aug`），内核 `justinz2010/xhey-rsi-c16-egoverse`（~2026-10-19 12:33 UTC 完成，2026-10-19 21:42 Asia/Shanghai 入账）。同一次对照 C16ctl 不加 EgoVerse（`iphone+xhey_rgb+aug`）。C16 整体 ADE 9.782 cm（比 zero 9.779 +0.003）、手在动 12.554 cm（−0.129，95% 区间 [−0.139, −0.120]）；C16ctl 整体 9.759 cm、手在动相对 zero −0.137（[−0.146, −0.129]）。EgoVerse 在手在动上赢了「猜手不动」，但没有超过同一次对照，整体略差于 zero。队列结论 **done_worse_vs_ctl**。`ledger.csv` 的 status 列按镜像规则是 win_moving，只表示手在动的置信区间整段落在 0 以下；这不是实验结论。图：[`experiments/report/C16/`](../experiments/report/C16/)、[`C16ctl/`](../experiments/report/C16ctl/)。

**iter005（2026-10-19）**：自动迭代第 5 轮，内核 `justinz2010/xhey-auto-iter`（`auto_state` 记录 2026-10-19 09:27 提交，Kaggle last run 2026-10-19T01:27:06Z）。三个候选的 mix 名仍写成 `iphone+xphi+xhey_rgb+aug`，只为和提议配置连续。内核日志写的是 `skip xphi: missing data/xphi/unified`——这一轮实际没有吃到 xPhi 数据。zero 整体 9.779 cm、手在动 12.683 cm。按手在动相对 zero，最好的是 **iter005_c0**（hidden 512，dropout 0.1，aug_noise 0.2）：整体 ADE 9.762 cm（Δ −0.017 cm）、手在动 12.545 cm（Δ −0.138 cm，95% 区间 [−0.147, −0.129]）。c2（hidden 512，dropout 0.2，aug_noise 0.05）整体 9.760 cm（Δ −0.019）、手在动 Δ −0.137（[−0.145, −0.128]）。c1（hidden 256，dropout 0.2，aug_noise 0.2）整体 9.815 cm（Δ +0.036）、手在动 Δ −0.111（[−0.121, −0.101]）。`ledger.csv` 的 status 按镜像规则是 win_moving。`experiments/auto_state.json` 的 `running` 已清空，`last_iteration=5`。C15 仍是 ready。这次只入账，没有提交任何 Kaggle 内核。

唯一循环驱动是 `scripts/xhey_auto_iter.py`：先收已完成的固定内核，再按 [`docs/xhey/queue.json`](xhey/queue.json) 提交就绪候选。新候选默认必须带 `--qc-filter`。C14 已入账；C16 EgoVerse 亦已入账（done_worse_vs_ctl）。`queue_running` 已清空。自动迭代第 3–5 轮已入账。第 5 轮没有实际使用 xPhi（日志 `skip xphi: missing data/xphi/unified`）。`running` 为空，没有自动迭代内核占着 GPU。C15 仍是 ready，有内核在跑时不抢占。

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
