# Robot 数据采集与训练（robot 工程）

> 本仓库沉淀「**机器人模仿学习训练管线 + 头戴式数据采集设备**」的完整分析、规格与采购依据。
> 训练用基准：LeRobot ACT（pusht）+ robomimic BC-RNN（MimicGen Square）。

## 训练管线（现状）

| 管线 | 框架 | 数据集 | 输入 → 输出 | 结果 |
|---|---|---|---|---|
| ACT | LeRobot v0.6.1 | pusht（206 集，10fps） | image[3,96,96]+state[2] → action[2]（chunk 100） | loss 收敛 0.074@100K；评估 0-5% |
| BC-RNN | robomimic | MimicGen Square（1000 条 × 160 步 @20Hz） | obs+image → actions(160,7) | image 68% / lowdim 76% |

## 在 Colab 上复现 ACT · pusht

[`notebooks/act_pusht_colab.ipynb`](notebooks/act_pusht_colab.ipynb) 在 Colab GPU 上按 `train.log` 的超参训练并评估 LeRobot v0.6.1 ACT（pusht，共 100K 步）：检查 GPU、可选挂载 Google Drive、安装 `lerobot[pusht,training]==0.6.1` 与 `gym-pusht`、克隆本仓库、支持从 `checkpoints/last` 断点续训、无头评估并把 rollout 存成 mp4、再用 `scripts/analyze_training.py --root` 生成报告。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/skyxiaobai/robot/blob/main/notebooks/act_pusht_colab.ipynb)

## 开放数据集上的质检与覆盖（还没有自有头戴设备时）

规格 v10（`docs/headcam_data_spec.md` §8）定义了四级标注、产出率和覆盖词表。差距分析在 `docs/gap_analysis.md`。当前能跑通的是 EgoDex 测试集（带 ARKit 世界系手部位姿）的抽样，不是 pusht。

依赖：`pip install numpy h5py pyarrow pandas`。导出和线性 BC 还要本机有 `ffmpeg`。

合成样本（不下载数据）：

```bash
python scripts/demo_open_dataset_pipeline.py --out outputs/open_data_demo
```

真实 EgoDex 测试集约 16.1GB，许可 CC BY-NC-ND，不要提交进本仓库：

```bash
curl -L "https://ml-site.cdn-apple.com/datasets/egodex/test.zip" -o test.zip
unzip test.zip   # 得到 test/<任务>/<序号>.hdf5 与同名 mp4
python scripts/convert_egodex.py --input test --out outputs/egodex_unified --limit 20
python scripts/egodata_qc.py --episodes outputs/egodex_unified \
    --html outputs/yield_report.html --csv outputs/yield_episodes.csv
python scripts/egodata_coverage.py --episodes outputs/egodex_unified \
    --html outputs/coverage_report.html --csv outputs/coverage_counts.csv
python scripts/validate_hierarchy.py examples/hierarchy_annotation.json --strict
```

Colab 上同一条链路：[`notebooks/open_dataset_pipeline_colab.ipynb`](notebooks/open_dataset_pipeline_colab.ipynb)。笔记本默认跑合成样本；要换真实抽样时，把解压后的 `test/` 路径传给演示脚本的 `--input`。

EgoDex 转换只会填 ENVIRONMENT 和 TASK。时间分段 SUBTASK 和分手 INSTRUCTION 在源数据里没有，校验不加 `--strict` 时只报告缺级。

QC 通过的片段可以导出成 **LeRobot v3.0**（与本仓库的 lerobot 0.6.1 一致）。`observation.state` 是双手关节加当前手腕，`action` 是下一步手腕增量；线性 BC 把连续 16 步拼成一块，并在固定的 episode 验证集上写下 `val_loss`。真实 mp4 默认缩到 224。ACT 配方在 `examples/ego_act_train.yaml`。转换可以加 `--workers`。

```bash
python scripts/convert_egodex.py --input test --out outputs/egodex_unified --limit 20 --workers 4
python scripts/ego_to_lerobot.py \
    --episodes outputs/egodex_unified \
    --yield-csv outputs/yield_episodes.csv \
    --out outputs/egodex_lerobot \
    --horizon 16 --video-size 224
python scripts/ego_pretrain_bc.py --dataset outputs/egodex_lerobot \
    --max-frames 1000 --log outputs/ego_pretrain.log
```

## 分析文档导航

| 文档 | 内容 |
|---|---|
| `docs/ANALYSIS_PROCESS.md` | **全部分析过程记录**（推导、调研、评审修正，建议从这读起） |
| `docs/headcam_data_spec.md`（v10） | 头戴设备规格：v9 的世界系手部标签与语言分段；v10 增加四级标注、产出率 QC、覆盖词表 |
| `docs/gap_analysis.md` | 开放数据下载 → 统一格式 → QC → 覆盖 → 标注 → LeRobot → 缩放律：已有、缺失、优先级 |
| `docs/headcam_bom.csv` | **可下单采购清单**：15 列证据链版（提供的数据/格式标准/数据契约/依据/采购原因），Excel 可直接打开 |
| `docs/twostage_pretrain_finetune_plan.md` | 两段式训练方案评估（合成预训练 → 头戴真实微调；路线 C 引用 headcam 规格 §7） |
| `scripts/build_headcam_bom.py` | BOM 生成脚本（可复现再生成 CSV） |
| `scripts/scaling_law.py` | 多次 run 的最优验证损失对 ln(数据量) 拟合，写入 HTML；无输入则跳过。示例 `examples/scaling_law_runs.yaml` |
| `scripts/convert_egodex.py` | EgoDex HDF5 → 统一 episode JSON。`--workers` 多进程；缺置信度记为未知 |
| `scripts/egodata_qc.py` | 手出画 / 视线飘移 / 模糊代理 / 静止 的 QC，以及训练产出率 HTML、CSV |
| `scripts/egodata_coverage.py` | 按环境、物体、任务、动作类型统计覆盖并标出空档 |
| `scripts/validate_hierarchy.py` | 校验四级标注。样例 `examples/hierarchy_annotation.json` |
| `scripts/demo_open_dataset_pipeline.py` | 用合成 EgoDex 式样本把上面几步串起来 |
| `scripts/ego_to_lerobot.py` | QC 通过的统一 episode → LeRobot v3.0。动作是手腕增量，mp4 可缩放到 224 |
| `scripts/ego_pretrain_bc.py` | 岭回归线性 BC。固定 episode 验证集，日志有 `val_loss` 和保持不动的基线 |
| `examples/ego_pretrain_bc.yaml` | 线性 BC 配方：增量动作、horizon、不同数据量 |
| `examples/ego_act_train.yaml` | 同一导出上的 lerobot 0.6.1 ACT（`chunk_size=16`，Colab GPU） |
| `scripts/collect_datasize_evidence.py` 等 | 分析取证脚本 |
| `outputs/mimicgen_*.log` | 训练/生成日志（已提交部分） |

## 核心结论（速览）

- **头戴设备只需 3 件核心**：头戴摄像头 + 头戴 IMU（仅 ego-motion 补偿）+ SOC；`state` 来自机器人侧、`action` 由画面内 MediaPipe 关键点 retargeting 得到
- **成本**：A 档 370-555 元 / B 档 700-945 元（均 ≤1000 元）；B 档 = AR0234 全局快门升级
- **数据量**：微调 50-150 条/任务（有预训练底座），从零训练 500-1000 条/任务
- **两段式可行但不可直接迁移**：现有 Square/pusht 是第三视角机械臂/2D 仿真，头戴是第一视角人手，需走「视觉表征迁移」或「同视角合成预训练」

## 目录说明

```
notebooks/           Colab：ACT·pusht，以及开放数据集质检/覆盖
docs/                分析文档（规格/BOM/方案/差距/过程记录）
scripts/egodata/     统一 episode、EgoDex 适配、QC、覆盖、标注校验
scripts/             分析取证与命令行入口
outputs/             训练产物与本地报告（大数据不入库）
data/ hf_cache/      数据集与缓存（不入库）
lerobot/ mimicgen/ robomimic/ robosuite/   第三方源码（独立 git 历史，不入库）
```

> 训练检查点（`outputs/checkpoints/`，28G）与第三方源码仓库按 `.gitignore` 有意不提交。
