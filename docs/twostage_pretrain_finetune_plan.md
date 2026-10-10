# 两段式训练方案：EgoDex 第一视角预训练 → 头戴真实数据微调

> 文档日期：2026-10-10。早期版本（2026-08-02，2026-10-09 曾增补 EgoScale 参照）评估的是合成预训练，见 §6，不再作为当前路线。
> 适用工程：`/mnt/sda/app/robot`
> **当前路线**：用 EgoDex 这类第一视角、带世界系手部位姿的开放数据预训练，再用自有头戴数据微调。Square / pusht 只用来验证训练框架，不能直接迁到头戴人手数据。
> 采集字段、世界系手部标签和双目 / IMU 要求以 `docs/headcam_data_spec.md` 为准。流水线已做到哪一步，以 `docs/gap_analysis.md` 为准。本文不新造数据量或损失数字。

---

## 0. 结论速览（TL;DR）

| 问题 | 结论 |
|---|---|
| 当前两段式是什么？ | **EgoDex 第一视角预训练 → 自有头戴数据微调**。链路是：EgoDex HDF5 → 统一 episode JSON → QC → LeRobot v3.0 → 线性 BC / ACT。自有头戴会话走同一份 JSON（`scripts/convert_headcam.py`），设备还没有 |
| Square / pusht 呢？ | 第三视角机械臂（Square）或 2D 仿真（pusht）。它们验证 ACT / BC-RNN 能跑。本体、视角、动作空间和头戴人手不一致，**不能把现成权重直接迁过去** |
| 预训练标签 | 不是 pusht 的 `action[2]`。导出后 `observation.state` 为双手关节加当前手腕（140 维），`action` 为下一步双手手腕增量（14 维）。ACT 的 `chunk_size=16`。见 `examples/ego_pretrain_bc.yaml`、`examples/ego_act_train.yaml` |
| 真实数据要多少？ | **没有新的实测**。规格附录的参考值：已有预训练底座时每任务约 50–150 条用于微调；从零约 500–1000 条。这不是 EgoDex 跑出来的微调条数 |
| 还没做完的 | 完整 EgoDex 测试集上的 ACT 还没训完。线性 BC 的测量和饱和现象写在 `docs/gap_analysis.md`，这里不复述成新的结论 |
| 历史方案 | 合成预训练（表征迁移，或同视角 MimicGen）是早期评估，见 §6 |

---

## 1. 预训练吃什么

优先 EgoDex 测试集：录制时的 ARKit 世界系手部位姿，和规格 §7 的世界系手腕是同一类量，测试包可以公开下载。许可 CC BY-NC-ND，数据不进本仓库。核对过的下载地址、约 16.1GB、条数和字段见 `docs/gap_analysis.md` §2。

已经接上的步骤：

1. `scripts/convert_egodex.py`：HDF5 → 统一 episode。只填 ENVIRONMENT 和 TASK；时间分段 SUBTASK 和分手 INSTRUCTION 源数据里没有，不在转换器里编造。
2. `scripts/egodata_qc.py` / `scripts/egodata_coverage.py`：产出率和覆盖。导出只收 QC 通过的片段。
3. `scripts/ego_to_lerobot.py`：写成 lerobot 0.6.1 的 codebase v3.0。真实 mp4 默认可缩到 224。
4. `scripts/ego_pretrain_bc.py`：岭回归线性 BC，作为同一动作目标上的基线。
5. `examples/ego_act_train.yaml` 与 `notebooks/egodex_act_scaling_colab.ipynb`：同一导出上的 ACT。lerobot 0.6.1 的 ACT 只条件于图像和 state，没有语言编码器。

动作空间在这条链上已经写死，供以后的头戴微调对齐：

- `observation.state` 长度 140 = 左手 21×3、右手 21×3、左手腕 7、右手腕 7（世界系；无效手的关节块置 0）。
- 每一行 `action` 长度 14 = 下一步左手腕增量（`dxyz` + 相对四元数）再接右手。保持不动是 `dxyz` 全 0、四元数 `0,0,0,1`。
- 线性 BC 把连续 `horizon` 行（默认 16）拼成一块；ACT 用同样的 `chunk_size=16`。

自有头戴数据要能微调这份权重，导出后的 state / action 必须和上面一致。维度一改，预训练 checkpoint 对不上。

---

## 2. 怎么做

### 阶段 1：EgoDex 预训练（设备还没有时就能跑）

命令与 README 相同。测试集约 16.1GB，不要提交进仓库。

```bash
python scripts/convert_egodex.py --input test --out outputs/egodex_unified --limit 20 --workers 4
python scripts/egodata_qc.py --episodes outputs/egodex_unified \
    --html outputs/yield_report.html --csv outputs/yield_episodes.csv
python scripts/ego_to_lerobot.py \
    --episodes outputs/egodex_unified \
    --yield-csv outputs/yield_episodes.csv \
    --out outputs/egodex_lerobot \
    --horizon 16 --video-size 224
python scripts/ego_pretrain_bc.py --dataset outputs/egodex_lerobot \
    --max-frames 1000 --log outputs/ego_pretrain.log
```

ACT 的单档命令在 `examples/ego_act_train.yaml`（`chunk_size=16`，`steps: 2000`，学习率 `1.0e-5`）。多档和缩放在 `notebooks/egodex_act_scaling_colab.ipynb`。不同数据量的验证损失用 `scripts/scaling_law.py` 拟合；没有足够的 run 时该脚本跳过。线性 BC 在全量测试集上的曲线以 `docs/gap_analysis.md` 已写下的测量为准，本文不另报一个数。

### 阶段 2：头戴采集（设备做出来之后）

字段按 `docs/headcam_data_spec.md`。硬件侧：同步双目给出手到相机的度量距离；头戴 IMU 与 SLAM/VIO 一起给出世界系相机位姿，不只做头部运动补偿。仓库不跑 SLAM，也不求解标定，只读已经算好的 TUM 和标定 yaml。

```bash
python scripts/convert_headcam.py --session /path/to/session --out outputs/headcam/episode.json
python scripts/egodata_qc.py --episodes outputs/headcam/episode.json \
    --html outputs/headcam/yield.html --csv outputs/headcam/yield.csv
```

通过 QC 的片段用同一套 `ego_to_lerobot.py` 导出。落盘后重算 `stats.json`（EgoDex 或 ImageNet 的统计不能直接套到自采数据上；规格 §6 风险 5）。条数没有新的实测，仍只引用规格附录的参考值：有预训练底座时约 50–150 条/任务，从零约 500–1000 条/任务。

### 阶段 3：用头戴数据微调

`scripts/ego_act_scaling.py` 用 `ACTPolicy.from_pretrained` 加载已有 ACT，并用 `pretrained_path` 建立 preprocessor。头戴导出与 EgoDex 导出的 state / action 一致时，可以沿这条加载方式做初始化。本仓库还没有把头戴数据接上微调的现成命令。

微调的步数和学习率**还没有单独的配方**，不要把 `examples/ego_act_train.yaml` 里的 `steps: 2000` 当成微调步数。评估协议（真实任务是否成功）也还没有；现有 pusht 0–5%、Square 68% / 76% 都是仿真 benchmark，不能当作头戴微调的成绩。

---

## 3. 为什么不拿 Square / pusht 当预训练

银河通用 GraspVLA 能用大量合成数据再做短微调，前提是合成数据与部署时的本体、相机和任务族一致。套到本仓库，现有合成数据和头戴数据对不上：

| 维度 | Square / pusht | 头戴 / EgoDex | 后果 |
|---|---|---|---|
| 本体 | Square：Panda 机械臂 + 夹爪。pusht：2D 推块 | 人手，21 个关节点 | 视觉和动作语义不同 |
| 视角 | Square：第三视角 agentview，另有腕部 eye_in_hand。pusht：俯视 2D | 第一视角 | 遮挡和尺度不同 |
| 动作 | Square：7 维（3 位置 + 3 轴角 + 1 夹爪），形状 (160, 7)。pusht：`action[2]` | 双手手腕增量，14 维，chunk 16 | 输出头对不上时 checkpoint 不能直接加载 |

直接把 Square 或 pusht 上训好的 ACT 指到头戴数据集，会在 features 契约上不匹配。即便只借用视觉编码器，也只是「看过机械臂或 2D 画面的 resnet18」，不是第一视角人手预训练。

---

## 4. 数据量（只引用已有参考，不新造）

| 说法 | 数字 | 出处 | 能不能当成 EgoDex 微调的实测 |
|---|---|---|---|
| 有预训练底座时的微调 | 约 50–150 条/任务 | 规格附录；ALOHA 50 条与银河通用「<1 人天」是该附录的依据 | 否 |
| 从零训练 | 约 500–1000 条/任务 | 同上。Square 1000 条 → image 68% / lowdim 76%；pusht 206 集 → ACT 评估 0–5% | 否。这是 benchmark，不是头戴人手 |
| EgoScale 公开口径 | 预训练约 21,000 小时第一视角视频；动作微调约 50 小时动捕 + 约 4 小时遥操作 | 规格 §7.5，公开演讲，不是本仓库的数据 | 否。本仓库没有这批小时数 |
| EgoDex 线性 BC | 全量测试集上的基线、饱和位置和相对「保持不动」的差距 | `docs/gap_analysis.md` | 那是状态-only 线性模型的验证损失，不是每任务需要多少条头戴演示 |

原始采集的留存率，早期版本写过 50–70%（录 100–250 条得到可用 50–150 条）。那是方案估计，不是本仓库在头戴数据上测得的留存率。

---

## 5. 坑

1. **用错预训练数据**：Square / pusht 权重要么加载失败，要么只提供不匹配的视觉初始化。预训练数据用 EgoDex 这一类第一视角人手数据。
2. **微调时改动作定义**：14 维手腕增量或 `chunk_size=16` 一变，EgoDex checkpoint 的输出头对不上。动作空间在阶段 1 已经定死。
3. **标签噪声**：自采手腕若尺度或坐标系错了，预训练帮不上。单目深度不是米；公制尺度靠双目。没有 TUM 时相机位姿是单位阵，坐标留在相机系，不能当成世界系真值。
4. **统计量错配**：微调前重算 `stats.json`。
5. **把没跑完的 ACT 当成已经预训练好**：完整测试集上的 ACT 还没训完（`docs/gap_analysis.md`）。线性 BC 会饱和，不能拿它的对数斜率外推「再多采多少小时损失还降」。
6. **语言**：EgoDex 转换不产生时间分段子任务和分手指令。lerobot 0.6.1 的 ACT 也不吃语言。不要在转换器里把整段描述切成假时间段。
7. **评估**：头戴微调之后用什么判成功，还没定。仿真成功率不能填这个空。

---

## 6. 历史说明：合成预训练（不再作为当前路线）

2026-08-02 的版本问的是：能否复刻银河通用「十亿帧合成预训练 + 每任务 <1 人天真实微调」，用当时已有的 MimicGen / Square 预训练，再用头戴数据微调。

当时的结论是不能直接复刻。Square 生成成功率 47.8%（1000 成功 / 2091 尝试），数据是第三视角机械臂加腕部相机；pusht 是 2D。和头戴第一视角人手在本体、视角、动作空间上都不一致。

曾列过两条未实施的变体，这里只留名字：

- **A. 视觉表征迁移**：用 Square 图像适应 resnet18，动作头在真实数据上重学。真实条数沿用规格附录的 50–150 条/任务。
- **B. 同视角合成**：在 robosuite / MimicGen 里做头戴视角再预训练。当时估计微调可到 30–100 条/任务。仿真第一视角人手没有现成管线，这条没做。

工作量估计（A 约 1–2 天，B 约 1–2 周）和上面的 30–100 条都是当时的方案判断，不是测量结果。EgoScale 的约 21,000 小时 / 约 50 小时 / 约 4 小时仍以规格 §7.5 为准，本仓库没有这批数据。

这些变体不替代 §1–§2。当前预训练是 EgoDex，不是合成数据。

---

## 7. 一句话

先在 EgoDex 上把第一视角、世界系手腕增量的预训练跑通，再用同一套导出微调自有头戴数据。Square 和 pusht 只证明训练框架能跑；早期的合成预训练方案留在 §6，不再当路线。
