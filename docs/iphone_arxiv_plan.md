# iPhone 头戴数据 → 可训练策略：论文路径（A → B → C）

面向「用 iPhone Pro Max + LiDAR 采的第一人称数据，能不能训出能用的动作模型」这条线。
先证明 **A 能训**，再讲 **B 数据质量**，最后才谈 **C 预训练增益**。

---

## A. 先证明：这套管线能训出「还能用」的策略（当前优先）

**一句话目标**：同一套指标下，用我们导出的标签训出来的小模型，至少不比「手永远不动」差，最好明显更好。

| 步骤 | 做什么 | 产出 |
|---|---|---|
| A1 | mcap → 会话 → WiLoR+LiDAR 融合 → episode | 已有：覆盖约 53%（修尺度后） |
| A2 | 通过 QC 的片段 → LeRobot | `outputs/iphone_mcap/run_all/lerobot` |
| A3 | 小模型 BC：线性岭回归 + MLP（与 HOT3D/EgoDex 同一套 ADE 指标） | `docs/iphone_mcap/train_compare.md` |
| A4 |（可选）图像 ACT，数据够再上 | 成功率 / ADE |
| A5 |（可选）仿真里重放或简单评测 | 视频 + 表 |

**成功标准（A 过关）**：
- 留一片段或固定划分上，MLP/线性 **ADE ≤ 零动作**，或至少与零动作持平且置信区间不更差；
- 有效动作比例、覆盖率写清楚，不藏「用了多少垃圾帧」。

**现在卡过的坑（已修，写进论文 Related/Method）**：
- WiLoR 单目手腕 ~0.4 m，LiDAR ~1.0 m：手型尺寸对，错的是平移 → **LiDAR 定手腕 + WiLoR 手型不缩放**。

---

## B. 数据质量故事（A 有数字之后）

**一句话目标**：讲清楚「什么样的帧配得上训练」，以及双目/LiDAR/单目差在哪。

| 内容 | 图/表 |
|---|---|
| 覆盖率、坏帧原因（模糊、出画、fit、depth） | QC 表 |
| 手腕深度：WiLoR mono vs LiDAR（修前/修后） | 直方图 |
| 与 HOT3D 双目标签对比（有真值时） | 中位误差 cm |
| 时序平滑 / 缺口填充对 ADE 的影响 | 消融 |

**成功标准**：审稿人能复现「从原始 mcap 到可训练张量」的每一刀。

---

## C. 预训练增益（B 清楚之后）

**一句话目标**：EgoDex / HOT3D / 自采 iPhone 数据，谁作预训练、谁作微调，下游 ADE 或成功率涨多少。

| 实验 | 说明 |
|---|---|
| 只 iPhone | A 的基线 |
| EgoDex 预训练 → iPhone 微调 | 域迁移 |
| HOT3D 双目标签预训练 → iPhone | 与双目管线统一格式的好处 |
| 数据量曲线 | 1 / 2 / 3 段，以及以后 10+ 段 |

**成功标准**：至少一条预训练曲线稳定高于「从头训」，并报告失败案例（负迁移）。

---

## 和仓库其它线的关系

- **HOT3D 扩大训练**：并行跑，用来回答「标签更准是否等于更好训」；不挡 A。
- **头戴双目硬件**：采到足够片段后，把 A 的脚本原样重跑即是「真机」版本的 A。

## 近期排期（建议）

1. **本周**：A2–A3 出第一张表（本文档对应的实验）。
2. **下周**：A 稳定后写 B 的质量图；补更多 iPhone 片段。
3. **再后**：C 的预训练对比；考虑图像 ACT。

## 复现入口

```bash
# 融合（已修尺度）
python scripts/run_stereo_pipeline.py --iphone outputs/iphone_mcap/sessions/*_s1 \
  --out outputs/iphone_mcap/run_all --backend wilor

# 训练对比（与 HOT3D 同指标）
python scripts/iphone_train_compare.py --run outputs/iphone_mcap/run_all --out outputs/iphone_mcap/train
```
