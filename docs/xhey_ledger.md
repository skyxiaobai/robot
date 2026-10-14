# XHEY 实验台账（自动迭代）

目标：**手腕 ADE 稳定低于「手不动」基线（overall 11.03 cm / 手在动 14.08 cm）**。  
评测固定：3 段 iPhone LiDAR 留一法，seeds=3，horizon=16 @15fps。数字越小越好。

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

机器可读：[`docs/xhey/ledger.csv`](xhey/ledger.csv)。日更图：[`docs/xhey/daily_progress.png`](xhey/daily_progress.png)。

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

本轮启动：**C1、C2**（见 Kaggle slug）；C3 做短代理后再决定。

---

## 自动迭代规则（给 bot）

1. 台账追加一行，不改历史数字。  
2. 候选预排序 → 最多同时 1 个 Kaggle GPU kernel。  
3. 跑完：更新 CSV + 重画 `daily_progress.png` + 写一句结论。  
4. ADE 低于 zero 且 CI 不跨 0 → 标「胜出」，开 PR。  
5. 连续 3 次同方向失败 → 该方向降权或关掉。
