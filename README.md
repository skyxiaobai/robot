# Robot 数据采集与训练（robot 工程）

> 本仓库沉淀「**机器人模仿学习训练管线 + 头戴式数据采集设备**」的完整分析、规格与采购依据。
> 训练用基准：LeRobot ACT（pusht）+ robomimic BC-RNN（MimicGen Square）。

## 训练管线（现状）

| 管线 | 框架 | 数据集 | 输入 → 输出 | 结果 |
|---|---|---|---|---|
| ACT | LeRobot v0.6.1 | pusht（206 集，10fps） | image[3,96,96]+state[2] → action[2]（chunk 100） | loss 收敛 0.074@100K；评估 0-5% |
| BC-RNN | robomimic | MimicGen Square（1000 条 × 160 步 @20Hz） | obs+image → actions(160,7) | image 68% / lowdim 76% |

## 分析文档导航

| 文档 | 内容 |
|---|---|
| `docs/ANALYSIS_PROCESS.md` | **全部分析过程记录**（推导、调研、评审修正，建议从这读起） |
| `docs/headcam_data_spec.md`（v9） | 头戴设备规格：模型训练 → 数据采集 → 硬件器件需求；v9 增加世界系手部标签、语言分段、无感佩戴与 EgoScale 配比 |
| `docs/headcam_bom.csv` | **可下单采购清单**：15 列证据链版（提供的数据/格式标准/数据契约/依据/采购原因），Excel 可直接打开 |
| `docs/twostage_pretrain_finetune_plan.md` | 两段式训练方案评估（合成预训练 → 头戴真实微调；路线 C 引用 headcam 规格 §7） |
| `scripts/build_headcam_bom.py` | BOM 生成脚本（可复现再生成 CSV） |
| `scripts/scaling_law.py` | 多次 run 的最优验证损失对 ln(数据量) 拟合，写入 HTML；无输入则跳过。示例 `examples/scaling_law_runs.yaml` |
| `scripts/collect_datasize_evidence.py` 等 | 分析取证脚本 |
| `outputs/mimicgen_*.log` | 训练/生成日志（已提交部分） |

## 核心结论（速览）

- **头戴设备只需 3 件核心**：头戴摄像头 + 头戴 IMU（仅 ego-motion 补偿）+ SOC；`state` 来自机器人侧、`action` 由画面内 MediaPipe 关键点 retargeting 得到
- **成本**：A 档 370-555 元 / B 档 700-945 元（均 ≤1000 元）；B 档 = AR0234 全局快门升级
- **数据量**：微调 50-150 条/任务（有预训练底座），从零训练 500-1000 条/任务
- **两段式可行但不可直接迁移**：现有 Square/pusht 是第三视角机械臂/2D 仿真，头戴是第一视角人手，需走「视觉表征迁移」或「同视角合成预训练」

## 目录说明

```
docs/                分析文档（规格/BOM/方案/过程记录）
scripts/             分析取证与生成脚本
outputs/             训练产物（checkpoints 等大文件不入库）
data/ hf_cache/      数据集与缓存（不入库）
lerobot/ mimicgen/ robomimic/ robosuite/   第三方源码（独立 git 历史，不入库）
```

> 训练检查点（`outputs/checkpoints/`，28G）与第三方源码仓库按 `.gitignore` 有意不提交。
