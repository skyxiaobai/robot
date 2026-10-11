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

| 内容 | 图/表 | 状态 |
|---|---|---|
| iPhone vs EgoDex vs HOT3D 总览 | [`docs/iphone_mcap/quality_compare.md`](iphone_mcap/quality_compare.md) + `quality_*.png` | **已完成**（2026-10-11） |
| 覆盖率、坏帧原因（模糊、出画、静止） | `outputs/iphone_mcap/run_s2/report.md` | 已有 |
| 手腕/关节误差：HOT3D 双目 1.22–1.24 cm；DexYCB iPhone-sim RGB-D 1.51 cm；EgoDex WiLoR 相对根 ≈3.97 cm / 绝对手腕 ≈22 cm | `quality_wrist_error.png` | 已有（真机头戴无 GT，用代理） |
| 检出：ARKit 双手 2D 81–91%；WiLoR 整段有效手偏低 | `quality_hand_detect.png` | 已有 |
| 深度覆盖 81–91%、RGB=Depth、≈30 fps | `quality_depth_coverage.png` / `quality_fps_stability.png` | 已有 |
| 手腕深度：WiLoR mono vs LiDAR（修前/修后）直方图 | — | 待补 |
| 时序平滑 / 缺口填充对 ADE 的影响 | — | 待补（Path A 消融） |

**成功标准**：审稿人能复现「从原始 mcap 到可训练张量」的每一刀；质量表能直接进论文 Figure。

**B 节关键数字（给摘要用）**：
- 采集：3× Pro Max，RGB≈30.0 fps，深度帧数=彩色，LiDAR conf≥1 均值 ≈87%，机载双手 2D 均值 ≈86%。
- 精度代理：RGB-D 手腕中位 **1.51 cm**（DexYCB iPhone-sim）；HOT3D 双目 **1.24 cm**；EgoDex 单目绝对手腕 **~22 cm**。
- 诚实披露：整段 WiLoR+LiDAR「任一手有效」均值只有约 **17%**（出画/静止）；短活跃窗约 **77%**。

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
2. **本周**：B 质量对比表已写入 `docs/iphone_mcap/quality_compare.md`；下一步补更多有深度片段 + mono vs LiDAR 直方图。
3. **再后**：C 的预训练对比；考虑图像 ACT。

## 复现入口

```bash
# 融合（已修尺度）
python scripts/run_stereo_pipeline.py --iphone outputs/iphone_mcap/sessions/*_s1 \
  --out outputs/iphone_mcap/run_all --backend wilor

# 训练对比（与 HOT3D 同指标）
python scripts/iphone_train_compare.py --run outputs/iphone_mcap/run_all --out outputs/iphone_mcap/train
```
