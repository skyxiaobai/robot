# 两段式训练方案评估：合成数据预训练 → 头戴真实数据微调

> 文档日期：2026-08-02（2026-10-09 增补 EgoScale 参照）｜ 适用工程：`/mnt/sda/app/robot`
> 起因：银河通用（Galbot）验证"十亿帧合成预训练 + 每任务 <1 人天真实微调"可行。本方案评估：**这个工程能否复刻该路线——用现有 MimicGen/Square 合成数据预训练，再用头戴设备采集的真实数据微调**，给出具体做法、所需真实数据量、以及坑。
> 推导原则（沿用 headcam_data_spec.md）：每条结论尽量引用**工程内真实存在的 checkpoint / 训练配置 / 数据集**作为依据；无法落地的部分明确标注为"方案设计/需验证"。
> **采集契约**：头戴数据字段、世界系手部标签、语言分段、无感佩戴和 EgoScale 式小时配比以 `docs/headcam_data_spec.md` **v9 §7** 为准。下文路线 C 只引用该节，不另写一套字段。

---

## 0. 结论速览（TL;DR）

| 问题 | 结论 |
|---|---|
| 两段式路线能否走？ | **能走，但不能"拿现成 Square/pusht 直接预训练→头戴微调"**。银河通用成功的前提是**合成数据与部署本体/视角/动作空间一致**（同机器人、同相机、同任务族）。本工程现有合成数据是**第三视角机器人臂（Square）+ 2D 仿真（pusht）**，头戴数据是**第一视角人手**——本体+视角+动作空间三重错位，直接权重迁移基本无效 |
| 怎么落地？ | 分两条可行路线：**A. 视觉表征迁移**（用现有合成数据预训练 resnet18 编码器 → 头戴数据上微调 action 头，最快最稳）；**B. 同视角合成预训练**（用 MimicGen/robosuite 自定义头戴视角相机生成第一视角合成数据 → 预训练 ACT → 真实微调，这才是银河式两段式，需先在仿真里建目标任务） |
| 需要多少真实数据？ | 场景 A（表征迁移后微调）：**50-150 条/任务**（与 §2.1 一致）；场景 B（同视角预训练后微调）：**30-100 条/任务**可更低；从零训练仍要 500-1000 条 |
| 最大坑 | **本体/视角错位**——Square 机器人臂权重无法直接用于人手策略；**动作空间必须提前定死**（ACT 输出维度一改，预训练权重作废）；**真实 action 标签噪声**（MediaPipe+retargeting 出的标签若错，预训练救不了） |
| 工作量 | A 路线 ≈ 1-2 天（代码改动小）；B 路线 ≈ 1-2 周（需建仿真环境+生成数据+预训练+微调） |
| EgoScale 参照（路线 C） | 不是用 Square/pusht 硬迁。预训练吃约 21,000 小时第一视角人类视频，标签是世界系手部关节 + 手腕 6DoF（headcam_data_spec.md v9 §7.1）；动作微调参照约 50 小时动捕 + 约 4 小时遥操作。最优验证损失对预训练小时数呈 log-linear，用 `scripts/scaling_law.py` 拟合 |

---

## 1. 现状盘点：工程内已有与缺失的资产

### 1.1 已有（工程内证据）

| 资产 | 位置/证据 | 规格 |
|---|---|---|
| ACT 策略（LeRobot v0.6.1） | `outputs/checkpoints/003000~100000`（38 个目录：37 个编号 checkpoint + `last` 符号链接） | `input_features`: image[3,96,96] + state[2]；`output`: action[2]；resnet18；chunk=100；MEAN_STD 归一化；ImageNet stats |
| ACT 训练曲线 | train.log + /tmp/train_*_log 合并 | 0.261@10K → 0.118@50K → 0.074@100K，收敛；pusht 评估 0-5% |
| BC-RNN 策略（robomimic） | /tmp/core_train_configs/bc_rnn_* | image 68% @240ep；lowdim 76% @300ep（Square） |
| MimicGen 生成器 | `mimicgen/mimicgen/scripts/generate_dataset.py` | 支持自定义相机（agentview/eye_in_hand/frontview/sideview 均有）；Square 生成成功率 **47.8%**（1000 成功/2091 尝试） |
| Square 数据集 | `/tmp/core_datasets/square/demo_src_square_task_D1/demo.hdf5` | 1000 条；ep_length 151±12；action(160,7)=3pos+3rot(轴角)+1夹爪；**第三视角 agentview + 腕部 eye_in_hand 两路单视图** |
| pusht 数据集 | `data/pusht/meta/info.json` | 206 集；10fps；2D 仿真；无 depth/stereo |
| 微调入口 | `lerobot/src/lerobot/scripts/lerobot_train.py` + `policies/factory.py:152` | **`pretrained_path` 参数确认存在**（加载预训练 processor+policy），`resume` 也支持 → 两段式在代码层面可行 |

### 1.2 缺失（两段式真正的门槛）

| 缺失项 | 说明 |
|---|---|
| **同视角/同本体的合成数据** | 现有 Square=机器人臂第三视角、pusht=2D；头戴=人手第一视角。**没有"第一视角人手"的合成数据**——这是与银河通用的根本差距（银河的十亿帧是它自家机器人的抓取画面） |
| **目标动作空间的定义** | 头戴数据的 action 由 MediaPipe 人手关键点 retargeting 而来，其维度/语义（末端 6DoF？关节角？指尖点？）**至今未定**。ACT 输出维度一变，预训练权重作废——这是先决问题 |
| **真实数据评估协议** | 现有评估都在仿真上跑（pusht 0-5%、Square 68-76%）；真实数据微调后**没有对应的真实评估协议**（任务成功率怎么判） |

---

## 2. 为什么"拿现有 Square/pusht 直接预训练→头戴微调"走不通

银河通用 GraspVLA 成功的关键链：**同本体（自家机械臂）+ 同相机视角（抓取视角）+ 同任务族（抓取）+ 十亿帧量级** → 合成学到通用抓取表征 → 真实微调只需覆盖任务边界。

把这条链套到本工程，三重错位：

| 维度 | Square 合成数据 | 头戴真实数据 | 错位后果 |
|---|---|---|---|
| 本体 | Panda 机械臂 + 夹爪 | 人手（21 关键点） | 视觉特征（手 vs 夹爪）完全不同；策略动作语义不同 |
| 视角 | 第三视角 agentview / 腕部 eye_in_hand | 第一视角 egocentric（头戴） | 相机坐标系、遮挡模式、尺度全不同 |
| 动作空间 | 7 维（3pos+3rot+1gripper，轴角） | 未定（retargeting 后可能是 6DoF 末端或手指空间） | ACT 输出头维度/语义一旦不匹配，预训练权重直接作废 |

**推论**：直接 `pretrained_path=Square 训的 ACT` 加载到头戴数据集上，会因 `input_features`/`output_features` 不匹配而在**数据加载阶段就报错**（LeRobot 用 features 契约校验），或即便强行对齐也只是拿一个"看过机械臂画面的 resnet18"当初始化——价值远低于预期。

---

## 3. 可行的两段式：两条路线

### 路线 A：视觉表征迁移（低成本，1-2 天，推荐先做）

**思路**：放弃"整个策略预训练"，只让合成数据强化**视觉编码器**（resnet18 的通用视觉表征），动作头在真实数据上从头学。

**做法**：
1. 用现有 Square 图像数据（1000 条 × 160 帧 ≈ 16 万帧，含 agentview + eye_in_hand）预训练/续训 resnet18——直接用 ImageNet 预训练权重起步（工程已 `use_imagenet_stats=true`，本身已是强初始化），再用 Square 图像做**任务域自适应**（可选：MAE/对比学习等自监督，或直接当 ImageNet 微调）。
2. 头戴数据采集后，加载预训练权重作为 ACT 初始化。**注意：LeRobot 的 `pretrained_path` 加载的是整个策略（vision backbone + transformer + action 头）+ processor 权重，不是只加载 resnet18**（`factory.py:152`）。两种衔接方式：**(a)** 若预训练的是完整 ACT 策略（Square 上训的 ACT），直接 `pretrained_path` 加载；若头戴 action 维度与预训练不同，`load_state_dict` 会在输出头上因尺寸不匹配而报错——LeRobot 不会自动重置输出头，**需自定义加载**（`strict=False` 加载 + 随机初始化新的 action 头）。**(b)** 若第 1 步只预训练了独立的 resnet18，`pretrained_path` 无法注入——**需自定义代码把 resnet18 权重加载进 ACT 的 `vision_backbone` 子模块**（对 backbone 子模块 `strict=False` 加载）。
3. 真实数据（50-150 条）上训练 ACT：**resnet18 用较小 LR（如 backbone lr × 0.1），transformer + action 头正常 LR**，防过拟合小数据集。

**预期收益**：视觉特征比纯 ImageNet 更贴近"操作任务画面"；但提升幅度有限（ImageNet 已覆盖通用视觉）。**风险低、见效快**。

**证据支撑**：
- ACT config 确认 `vision_backbone: resnet18` + `use_imagenet_stats=true`（工程内），说明当前即用 ImageNet 初始化——表征迁移路线只是"加一层任务域自适应"。
- LeRobot `factory.py:152` 的 `pretrained_path` 支持加载预训练 processor（含归一化统计与图像预处理管线）与完整策略权重。

### 路线 B：同视角合成预训练（银河式两段式，1-2 周，需建仿真环境）

**思路**：让合成数据**和头戴数据同视角同本体**，再走"预训练→微调"。这才是银河通式两段式，也是唯一能让真实数据降到 30-100 条/任务的路线。

**做法**：
1. **在仿真里建目标任务**：用 robosuite/MimicGen 定义与真实任务同构的场景（目标物体、桌面、手部工作区），把相机设成**头戴第一视角**——MimicGen/robosuite 支持自定义相机名（工程内 `mimicgen/mimicgen/envs/robosuite/coffee.py` 等已用 agentview/eye_in_hand/frontview，可新增 `headview` 相机挂在"头部"位置）。若目标是真实机器人执行，仿真本体应选目标机械臂；若目标是"人手策略"，则需人手/灵巧手仿真本体（如 robosuite 的 `PandaHand`/`Aloha` 或 MuJoCo 人手模型）。
2. **生成合成数据**：跑 `mimicgen/scripts/generate_dataset.py` 生成数百至数千条第一视角演示；**只保留 success=1 的**（MimicGen 47.8% 成功率意味着过半生成样本是失败轨迹，必须过滤）。
3. **预训练 ACT**：在合成第一视角数据上训练 ACT 至收敛（action 空间 = 目标动作空间，**这一步就把动作空间定死了**）。
4. **头戴采集真实数据**：50-100 条/任务（同视角预训练后，微调需求比 §2.1 的 50-150 更少）。
5. **微调**：`pretrained_path` 加载合成预训练 checkpoint → 真实数据上小 LR 微调 → 在**真实留出集**上评估。

**关键约束**：**动作空间必须在步骤 2 之前确定**——仿真里 action 的定义必须与真实 retargeting 出的 action 完全一致（维度、语义、归一化范围），否则预训练白做。

**证据支撑**：
- 工程内 MimicGen 已跑通（47.8% 生成成功率、`ep_length 151`、action 7 维），生成器代码可用。
- robosuite/MimicGen 相机可配置（多文件确认 `camera` 参数存在）。
- EgoMimic/EgoDex（headcam_data_spec §1.4 已列）验证了"第一视角数据 → 策略"路线本身可行；但它们的预训练数据是**真实大规模第一视角数据**（EgoDex 829h），不是仿真——**仿真第一视角人手数据生成目前无现成开源管线**，是这条路线最大的不确定性（需自行搭建或借用灵巧手仿真）。

### 路线 A vs B 对比

| 维度 | A 表征迁移 | B 同视角预训练 |
|---|---|---|
| 真实数据需求 | 50-150 条/任务 | 30-100 条/任务 |
| 代码改动 | 小（改 pretrained_path + LR 分层） | 大（建仿真任务 + 自定义相机 + 生成 + 预训练） |
| 工作量 | 1-2 天 | 1-2 周 |
| 风险 | 低 | 中-高（仿真第一视角人手数据无现成管线） |
| 与银河通用路线匹配度 | 低（只迁移表征） | 高（同构预训练） |

### 路线 C：EgoScale 式人类视频预训练（参考，本工程尚无这批数据）

公开口径（Jim Fan「Robotics End Game」/ NVIDIA EgoScale），字段与配比全部以 **`docs/headcam_data_spec.md` v9 §7** 为准，这里只列本方案要跟着走的几点：

1. **预训练标签**不是机器人 `action`，而是逐帧手部关节关键点 + **世界系手腕 6DoF**（§7.1）。头戴 IMU 单独不够，采集侧要有双目和/或 SLAM（§7.2）。当前 ACT 的 image+state 契约仍然要落盘，但它替代不了这组标签。
2. **语言**是稠密时间分段的子任务，不是一集一句（§7.3，`meta/language_segments.jsonl`）。
3. **设备无感**：头上重量要低、能长时间佩戴、episode 自动上传（§7.4）。否则第一视角动作分布会偏。
4. **数据配比（参考）**：约 21,000 小时第一视角人类视频做预训练，再用约 50 小时动捕 + 约 4 小时遥操作做动作微调（§7.5）。这和下面 §5 的「50–150 条/任务」不是同一个数：50–150 条是没有这批预训练时的真实演示下限。
5. **缩放律**：最优验证损失对预训练小时数 log-linear。多次不同小时数的 run 用 `scripts/scaling_law.py --runs <yaml或csv>` 取每条日志的最小验证损失，拟合 `L = a + b·ln(N)`，并写进 HTML 报告。没有输入或有效 run 少于 2 个时该节跳过。示例：`examples/scaling_law_runs.yaml`。

路线 C 在本工程里**还跑不起来**：仓库里没有万小时级第一视角视频，也没有世界系手腕标签。它用来约束头戴采集规格（已经写进 headcam_data_spec v9），以及以后有了不同数据量的预训练 run 时怎么看损失是否还在沿 log 下降。在那之前，仍按路线 A 先打通真实微调。

---

## 4. 具体怎么做（可执行步骤）

### 阶段 0：定动作空间（所有路线的前置，1 天）

1. 明确目标机器人：真机执行 → 末端 6DoF（pos+rot）+ 夹爪 = 7 维（对齐 Square 的 action 形状）；纯人手策略 → 指尖关键点空间。
2. 定 retargeting 映射：MediaPipe 21 关键点 → 目标动作空间的具体公式（对齐/IK），**先在离线视频上跑通并人工抽检 action 轨迹平滑度**。
3. 据此写 headcam 数据集的 `info.json` features 声明（action shape/type）。

### 阶段 1：合成预训练（路线 A：1-2 天；路线 B：1 周+）

路线 A：
```bash
# 复用现有 Square 数据 + ImageNet 初始化，可选做任务域自适应；核心是保存 resnet18 视觉权重
# （若只做表征迁移，也可跳过正式预训练，直接 ImageNet 初始化 + 真实微调时 backbone 低 LR）
```

路线 B：
```bash
# 1) 新建 robosuite 任务环境 + headview 相机
#    ⚠️ MimicGen 的 generate_dataset.py 是 config 驱动，相机在 env config 的 camera_names 里配置，
#    没有 --camera/--num-demo 这类 CLI 参数——先跑 python mimicgen/mimicgen/scripts/generate_dataset.py --help 确认实际参数。
# 2) 生成第一视角合成演示（只留 success）
python mimicgen/mimicgen/scripts/generate_dataset.py --config <your_env_config.json> ...
# 3) 转 LeRobot 格式（robomimic HDF5 -> parquet+mp4），生成 info.json/stats.json
# 4) 预训练 ACT
python lerobot/src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=local/your_synthetic_task \
  --policy.type=act --policy.pretrained_path=null \
  --steps=<N> --save_checkpoint=true ...
```

### 阶段 2：头戴采集真实数据（字段见 headcam_data_spec.md v9 §2 与 §7）

- 50-150 条/任务（A 路线）或 30-100 条/任务（B 路线），每条 8-30s 完整闭环。
- 覆盖 ≥5 种起始状态变体 + 边界样本；手部全程在画面内。
- 落盘后**重算 stats.json**（合成/ImageNet 统计不适用于真实数据——headcam 文档 §6 风险 5 已列）。
- 若这段数据还要服务路线 C 的预训练：按 headcam_data_spec.md §7 同时写下世界系 `observation.hand_joints`、`observation.wrist_pose`、稠密 `language_segments`，并满足无感佩戴（重量、长时佩戴、自动上传）。只靠头戴 IMU 的轨迹不能填手腕 6DoF。

### 阶段 3：微调 + 评估

```bash
python lerobot/src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=local/headcam_real_task \
  --policy.type=act \
  --policy.pretrained_path=<stage1 checkpoint 目录> \
  --resume=false \
  --steps=<10K~30K 小步数> \
  --batch_size=16 \
  # backbone 低 LR：如 --optimizer.lr=1e-5（backbone 分层 LR 需自行配置）
```

- 真实数据留出 10-20% 作评估集；定义真实任务成功率判据（如"放置成功=物体进入目标区域"）。
- 与 from-scratch 基线对比（同数据量从零训练），量化预训练增益。

---

## 5. 需要多少真实数据（汇总）

| 路线 | 每任务真实演示数（筛选后） | 依据 |
|---|---|---|
| 从零训练（现状，无预训练） | **500-1000 条** | Square 实测 1000 条 → 68-76%；pusht 206 集 → ACT 0-5% |
| 路线 A（表征迁移微调） | **50-150 条** | ALOHA 50 条业界基线；银河通用 <1 人天；与 §2.1 场景 A 一致 |
| 路线 B（同视角预训练微调） | **30-100 条** | 预训练已学"操作先验"，微调只补任务边界；银河 GraspVLA 实证 <1 人天 |
| 路线 C（EgoScale 参照，见 headcam_data_spec.md §7.5） | 预训练 **~21,000 小时**第一视角视频；动作微调 **~50 小时**动捕 + **~4 小时**遥操作 | 公开演讲口径，不是本工程已有数据。最优验证损失对预训练小时数 log-linear |

原始采集需按 50-70% 留存率折算（录 100-250 条 → 可用 50-150 条）。

---

## 6. 坑清单（按严重程度排序）

### 🔴 严重（会直接导致预训练白做或训练失败）

1. **本体/视角错位**：Square（机械臂第三视角）预训练权重迁移到头戴（人手第一视角），视觉特征与动作语义双重不匹配。**要么走路线 A 只迁表征，要么走路线 B 生成同视角数据**——没有第三条捷径。
2. **动作空间未定就预训练**：ACT 的 `output_features` 维度/语义一变，预训练 checkpoint 全部作废。**动作空间是两段式的"先决条件"，必须在任何预训练之前定死**（阶段 0）。
3. **真实 action 标签噪声**：MediaPipe+retargeting 出的 action 若位置/姿态标错，预训练学到的"通用操作"也会被错误监督覆盖。**必须离线抽检真实 action 轨迹质量**（平滑度、与画面手部位置一致性）。

### 🟡 中等（影响效果或可实施性）

4. **归一化统计错配**：合成/ImageNet 的 stats.json 不适用于真实数据，微调前必须重算（headcam 文档 §6 风险 5）。
5. **MimicGen 数据质量**：47.8% 生成成功率 → 过半合成演示是失败轨迹；生成后必须按 success 过滤，否则预训练学到失败行为。
6. **小数据过拟合**：50-150 条真实数据上直接训练容易过拟合 → 用 backbone 低 LR、数据增强（工程 config 已有 affine/brightness/contrast 变换可开启）、early stopping。
7. **评估协议缺失**：真实数据微调后没有真实成功率判据，无法判断预训练是否有增益。必须先定评估标准再做微调。

### 🟢 低（工程细节）

8. **数据格式转换**：robomimic HDF5 → LeRobot parquet+mp4（v0.6.1 有 `convert_dataset_v21_to_v30.py` 可参考）。
9. **图像尺寸**：Square 84×84 / pusht 96×96 / 头戴 720p——LeRobot 内部 resize，但需确认输入契约一致（96×96）。
10. **双路 vs 单路**：Square 有 agentview+eye_in_hand 两路，头戴只有一路——`input_features` 里的 image 特征数必须与数据集一致。

---

## 7. 建议路线图

1. **本周（阶段 0）**：定目标机器人 + retargeting 公式 + action 空间；用现有视频离线验证 MediaPipe 关键点质量。
2. **第 1-2 周**：走路线 A（表征迁移微调）——成本最低，先把"真实数据 50-150 条 → 可用策略"跑通，建立真实评估协议。
3. **第 3-4 周（可选）**：评估路线 A 的增益是否值得——若 A 的增益有限且真实数据采集成本高，再投入路线 B（建同视角仿真任务 + 生成第一视角合成数据），目标把真实数据需求降到 30-100 条/任务。
4. **持续**：真实数据入库后重算 stats、抽检 action 标签、跑 A/B 对比（预训练 vs 从零）。

---

## 8. 一句话总结

**两段式路线在这个工程能走，但前提是"合成数据与头戴数据同视角同本体"——银河通用做得到，是因为它的十亿帧就是自家机器人的画面；你现有 Square/pusht 不是。** 最快路径：先走"ImageNet+任务域表征 → 真实微调"（路线 A，1-2 天，50-150 条真实数据），同时把动作空间和评估协议定死；若真实数据成本成为瓶颈，再投入"同视角合成预训练"（路线 B，1-2 周，真实数据可降到 30-100 条/任务）。
