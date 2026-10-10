# Robot 数据采集与训练（robot 工程）

> 本仓库沉淀「**机器人模仿学习训练管线 + 头戴式数据采集设备**」的完整分析、规格与采购依据。
> 训练用基准：LeRobot ACT（pusht）+ robomimic BC-RNN（MimicGen Square）。
> 第一次看这个仓库，先读 [`docs/PROGRESS.md`](docs/PROGRESS.md)：一张图、做到哪一步、下一步是什么。

## 训练管线（现状）

| 管线 | 框架 | 数据集 | 输入 → 输出 | 结果 |
|---|---|---|---|---|
| ACT | LeRobot v0.6.1 | pusht（206 集，10fps） | image[3,96,96]+state[2] → action[2]（chunk 100） | loss 收敛 0.074@100K；评估 0-5% |
| BC-RNN | robomimic | MimicGen Square（1000 条 × 160 步 @20Hz） | obs+image → actions(160,7) | image 68% / lowdim 76% |

## 在 Colab 上复现 ACT · pusht

[`notebooks/act_pusht_colab.ipynb`](notebooks/act_pusht_colab.ipynb) 在 Colab GPU 上按 `train.log` 的超参训练并评估 LeRobot v0.6.1 ACT（pusht，共 100K 步）：检查 GPU、可选挂载 Google Drive、安装 `lerobot[pusht,training]==0.6.1` 与 `gym-pusht`（并固定 `pymunk<7`）、预检 `gym.make("gym_pusht/PushT-v0")`、克隆本仓库、支持从 `checkpoints/last` 断点续训。无头评估默认 `batch_size=10`；异步 worker 报 `Namespace gym_pusht not found` 时改用 `--eval.use_async_envs=false`，断线后可以直接评估已保存的 checkpoint，并把 rollout 存成 mp4。再用 `scripts/analyze_training.py --root` 生成报告。

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

Colab 上同一条链路：[`notebooks/open_dataset_pipeline_colab.ipynb`](notebooks/open_dataset_pipeline_colab.ipynb)。笔记本默认跑合成样本；要换真实抽样时，把解压后的 `test/` 路径传给演示脚本的 `--input`。同一本笔记本里可以看骨架叠加、质检对照和世界系轨迹。

```bash
python scripts/ego_visualize.py overlay \
    --episode outputs/open_data_demo/unified/basic_pick_place/0.json \
    --out outputs/open_data_demo/viz/overlay.png --width 640
python scripts/ego_visualize.py qc_compare \
    --episodes outputs/open_data_demo/unified \
    --out outputs/open_data_demo/viz/qc_compare.png
python scripts/ego_visualize.py traj3d \
    --episode outputs/open_data_demo/unified/basic_pick_place/0.json \
    --out outputs/open_data_demo/viz/traj3d.png
```

自有头戴会话（设备还没有，目录见规格 §7.7）收成同一份 JSON。有 `hands.json` 和标定时会做双目尺度校正；默认推荐 WiLoR（需 GPU 和仓库外的 MANO 权重）；`--backend mediapipe` 只作为 CPU 兜底，HaMeR 也需要 MANO 权重。MediaPipe 单目手腕深度只是先验（默认 0.55 m），公制深度要靠双目。`mediapipe>=0.10.30` 用 Tasks `HandLandmarker`；无头环境若缺 `libEGL` / `libGLESv2`，先装系统库，推理再用 CPU delegate。和 EgoDex 比的时候，`Hand` / `ThumbKnuckle` 不是 MediaPipe 的腕点和拇指 CMC，可以用 `EGODEX_NONCORRESPONDING_JOINTS` 排除。WiLoR / HaMeR 的公制平移用 `calib['K_left']` 的 fx、fy 和主点；二维点仍按虚拟焦距投影，传不传 K 都一样。不传 K 时平移用约 37500 px（1920 宽），手腕深度不是米。`mano_mean_params.npz` 不在官方 MANO 压缩包里，放在 `MANO_MODEL_DIR` 或 WiLoR 仓库的 `mano_data/`。Colab GPU 上优先用 WiLoR。安装见 `hand_pose.MANO_LICENSE`：`ultralytics==8.1.34`、`pip install --no-build-isolation chumpy`；pyrender 只用于可视化。HaMeR 还要编译 detectron2 和 mmcv。

```bash
python scripts/convert_headcam.py --session /path/to/session --out outputs/headcam/episode.json
python scripts/egodata_qc.py --episodes outputs/headcam/episode.json \
    --html outputs/headcam/yield.html --csv outputs/headcam/yield.csv
python scripts/ego_visualize.py overlay --episode outputs/headcam/episode.json \
    --out outputs/headcam/overlay.png --width 640
```

检测沿时间会抖、会丢帧。`scripts/headcam/hand_track_refine.py` 可以在检测之后做四件事：One Euro 平滑（或常速度卡尔曼）、短缺口插值、按这一段的骨长中位数固定手型、用手腕轨迹把左右标签拉齐。默认关闭。补上的帧会标 `filled`，置信度写成 0，质检会拒绝它们。

```bash
python scripts/convert_headcam.py --session /path/to/session --out outputs/headcam/episode.json --refine
python scripts/eval_hand_refine.py --root data/egodex_hand_eval --out docs/hand_refine_egodex.md
```

`--smooth kalman`、`--gap-fill 5`、`--fixed-shape` 可以分开开。EgoDex 上的五档对比写在 [`docs/hand_refine_egodex.md`](docs/hand_refine_egodex.md)。没有 MANO 时评测用 MediaPipe，不能把那张表当成 WiLoR 的误差。

Colab：[`notebooks/headcam_hand_pose_colab.ipynb`](notebooks/headcam_hand_pose_colab.ipynb)。默认用合成双目检查三角化误差（厘米），再跑 QC 和骨架叠加。HaMeR / WiLoR 单元默认关闭。和 EgoDex 比的那一格在 `HAND_REFINE=1` 时才会做时序精修。

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
| `docs/PROGRESS.md` | **项目进度**（流水线图、时间线、已有实验数字、待办）。建议第一次从这里读 |
| `docs/retarget_sim.md` | 人手腕重定向到 MuJoCo 抓放：标定、IK、重放和行为克隆。成功率以结果 JSON 为准 |
| `docs/PIPELINE_ARCHITECTURE.md` | 整体架构：录制 → 校验 → 标定 → 认手 → 双目 → 世界系 → QC → 标注 → LeRobot → 训练，每步的状态和文件格式 |
| `docs/stereo_pipeline.md` | 双目一条命令管线 `scripts/run_stereo_pipeline.py` 的用法与 HOT3D 实测 |
| `docs/hardware_day1_checklist.md` | 硬件到货第一天：同步、标定、首段录制和合格线 |
| `docs/ANALYSIS_PROCESS.md` | **全部分析过程记录**（推导、调研、评审修正） |
| `docs/headcam_data_spec.md`（v10） | 头戴设备规格：v9 的世界系手部标签与语言分段；v10 增加四级标注、产出率 QC、覆盖词表 |
| `docs/gap_analysis.md` | 开放数据下载 → 统一格式 → QC → 覆盖 → 标注 → LeRobot → 缩放律：已有、缺失、优先级 |
| `docs/headcam_bom.csv` | 采购清单（**单目方案，留作对照**）：15 列证据链版（提供的数据/格式标准/数据契约/依据/采购原因），Excel 可直接打开 |
| `docs/headcam_stereo_bom.md` / `docs/headcam_stereo_bom.csv` | **1000 元内双目采购清单**（推荐方案 A 约 727–982 元）。说明见 md，明细见 csv。下单向卖家要 8–10 cm 基线，镜头选约 100° |
| `docs/hot3d_stereo/results.md` | HOT3D Quest3 上的双目手腕评测和硬件选型。双目手腕中位数 1.22 cm，75% 在 2 cm 内 |
| `docs/twostage_pretrain_finetune_plan.md` | 两段式训练方案评估（早期版本写的是合成预训练 → 头戴真实微调；当前路线改为 EgoDex 第一视角预训练 → 头戴微调，见上文速览） |
| `scripts/build_headcam_bom.py` | BOM 生成脚本（可复现再生成 CSV） |
| `scripts/scaling_law.py` | 对数直线和饱和幂律都拟合，R² 更高的作为默认。多种子画均值和标准差。无输入则跳过 |
| `scripts/convert_egodex.py` | EgoDex HDF5 → 统一 episode JSON。`--workers` 多进程；缺置信度记为未知 |
| `scripts/egodata_qc.py` | 手出画 / 视线飘移 / 模糊代理 / 静止 的 QC，以及训练产出率 HTML、CSV |
| `scripts/egodata_coverage.py` | 按环境、物体、任务、动作类型统计覆盖并标出空档。逐条累加，不把全部 JSON 放进内存 |
| `scripts/validate_hierarchy.py` | 校验四级标注。样例 `examples/hierarchy_annotation.json` |
| `scripts/demo_open_dataset_pipeline.py` | 用合成 EgoDex 式样本把上面几步串起来 |
| `scripts/ego_visualize.py` | 手骨架叠加、质检对照、世界系轨迹。投影与 QC 相同。不把画面提交进仓库 |
| `scripts/convert_headcam.py` | 头戴会话目录 → 统一 episode。HaMeR / WiLoR / MediaPipe，双目三角化，TUM 位姿 |
| `scripts/headcam/hand_pose.py` | 上面三个后端、Kalibr/OpenCV 标定、尺度校正、手腕 6DoF。MediaPipe 走 Tasks，单目深度是先验。MANO 权重不入库 |
| `scripts/headcam/hand_track_refine.py` | 可选时序精修：One Euro 或卡尔曼、短缺口补洞、固定骨长、左右轨迹。默认关闭 |
| `scripts/eval_hand_refine.py` | 在固定的 EgoDex 测试片段上做五档消融。指标用 `evaluate_hand_frames` |
| `notebooks/headcam_hand_pose_colab.ipynb` | 合成双目的厘米误差、QC 和叠加；可选 GPU 上的 HaMeR / WiLoR。和 EgoDex 比时用每段内参、全部帧和全部关节 |
| `scripts/ego_to_lerobot.py` | QC 通过的统一 episode → LeRobot v3.0。动作是手腕增量，mp4 可缩放到 224 |
| `scripts/ego_pretrain_bc.py` | 岭回归线性 BC。特征和动作都用全训练池统计量，可扫描 l2，多种子写均值和标准差 |
| `examples/ego_pretrain_bc.yaml` | 线性 BC 配方：增量动作、horizon、不同数据量 |
| `examples/ego_act_train.yaml` | 同一导出上的 lerobot 0.6.1 ACT（`chunk_size=16`，Colab GPU） |
| `scripts/ego_act_scaling.py` | 按固定验证 episode 安排多档 ACT，训练后写 val_loss 和保持不动基线 |
| `notebooks/egodex_act_scaling_colab.ipynb` | 下载 EgoDex test、224 导出，并在 T4/L4 上跑 3–4 档 ACT 缩放 |
| `scripts/collect_datasize_evidence.py` 等 | 分析取证脚本 |
| `outputs/mimicgen_*.log` | 训练/生成日志（已提交部分） |

## 核心结论（速览）

> **当前状态：软件链路已基本就绪，等双目头戴设备做出来后用自采数据验证。**

- **设备是头戴双目**：同步的全局快门双目 + RGB + IMU，并且要做好标定。两个摄像头像人的两只眼睛，用来算出手离相机的真实距离；IMU 配合 SLAM/VIO 算出相机在世界坐标系里的位置和朝向，不只是用来抵消头部晃动（规格 §7.1–§7.2）。
- **每帧要产出的标签**：21 个手部关节点、世界系手腕 6DoF（位置 + 朝向），以及四级语言标注：环境 → 任务 → 子任务 → 单手指令（规格 v10 §8）。
- **手部识别用 WiLoR**：在 EgoDex 上和 MediaPipe 对比，WiLoR 约 95% 的帧能认出手（MediaPipe 约 50%），关节误差约 3.5 cm（MediaPipe 约 8 cm）。WiLoR 需要 GPU 和 MANO 权重（非商业许可，不入库）；MediaPipe 只作为 CPU 兜底。
- **单目算不准距离，所以必须双目**：只用一个摄像头时，手腕位置会差 4–10 cm，目标是 2 cm 以内。`scripts/headcam/` 已实现双目三角化、尺度校正和世界系手腕位姿。
- **数据先筛再用**：自动 QC 能算出可用于训练的比例（产出率），覆盖报告能指出哪些环境、物体、任务或动作还没采到，`scripts/egodata/` 已实现。
- **先用开放数据预训练**：EgoDex（第一视角、带世界系手部位姿）→ 统一格式 → QC → LeRobot v3.0 → 线性 BC / ACT，并用 `scripts/scaling_law.py` 画数据量和效果的缩放曲线（见 `notebooks/egodex_act_scaling_colab.ipynb`）。
- **两段式训练**：先用 EgoDex 这类第一视角人手数据预训练，再用自有头戴数据微调。旧的 Square/pusht 是第三视角机械臂或 2D 仿真，只用来验证训练框架，不能直接迁移到头戴人手数据。
- **成本**：单目旧表仍在 `docs/headcam_bom.csv`（A 档约 370–555 元，B 档换 AR0234 全局快门约 700–945 元；表内双目可选项 OAK-D-LR 或双 AR0234 + 外部触发 + RK3566，标价 ≥1500–3000 元）。**1000 元内双目清单**在 `docs/headcam_stereo_bom.md`：推荐方案 A（OV9281 硬件同步双目 + Orange Pi 3B）合计约 727–982 元。方案 B 约 837–1077 元，只有按低价买才可能压在 1000 元内。
- **数据量（规格附录的参考值，不是本仓库的实测结果）**：已有预训练底座时，每个任务约需 50–150 条用于微调；从零训练约需 500–1000 条。

## 目录说明

```
notebooks/           Colab：ACT·pusht，以及开放数据集质检/覆盖
docs/                分析文档（规格/BOM/方案/差距/过程记录）
scripts/egodata/     统一 episode、EgoDex 适配、QC、覆盖、标注校验
scripts/headcam/     头戴手部后端、双目三角化、TUM 位姿。不含 MANO 权重
scripts/             分析取证与命令行入口
outputs/             训练产物与本地报告（大数据不入库）
data/ hf_cache/      数据集与缓存（不入库）
lerobot/ mimicgen/ robomimic/ robosuite/   第三方源码（独立 git 历史，不入库）
```

> 训练检查点（`outputs/checkpoints/`，28G）与第三方源码仓库按 `.gitignore` 有意不提交。
