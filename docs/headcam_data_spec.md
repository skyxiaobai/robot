# 头戴式数据采集设备规格（纯头显 · v10）

> **适用工程**：`/mnt/sda/app/robot` ｜ **文档日期**：2026-10-09（v8 正文 2026-08-01 保留）
>
> **推导逻辑（一条链，一一对应）**：模型训练消费什么 → 数据采集必须产出什么 → 头戴硬件器件需求是什么。
> **每一条「必须」都引用工程内真实存在的 checkpoint / 训练配置 / 数据集字段作为依据，不靠猜测**。v9 新增的 EgoScale 对齐项来自公开演讲口径，在正文标明「参考方案」，不写成工程内已有 checkpoint。
>
> **v8 变更**：按「纯头显（无手腕相机、无手套）」收敛范围，删除训练数据报告、外部产品规格（Ego 头显 / iPhone）、未来 Pipeline 接入检查流程、版本历史等非设备规格内容。
>
> **v9 变更**（对照 Jim Fan「Robotics End Game」中的 NVIDIA EgoScale）：保留 v8 全文。把**逐帧手部关节关键点 + 世界坐标系手腕 6DoF 位姿**升为核心标签，并写明头戴 IMU 单独不够、需要双目和/或 SLAM。增加稠密时间分段子任务语言标注、无感佩戴要求（重量、长时间佩戴、自动上传），以及 EgoScale 式数据配比参考方案。详见 §7。当前 ACT/BC-RNN 输入契约仍以 §1–§3 为准。
>
> **v10 变更**（对照覆盖度采集、产出率 QC、四级语言标注）：保留 v9 全文。增加 §8：四级标注（ENVIRONMENT / TASK / 时间分段 SUBTASK / 分手 INSTRUCTION）及 JSON 样例、训练产出率与自动 QC 的定义、按环境/物体/任务/动作类型统计的覆盖词表。没有自有头戴设备时，用开放数据集上的位姿和元数据先把这三件事跑起来，工具见 `scripts/egodata/`。

---

## 0. 结论速览（TL;DR）

| 推导层 | 结论 |
|---|---|
| ① 模型训练（真实配置） | 2 套：LeRobot ACT（`image[3,96,96]` + `state[2]` → `action[2]`）+ robomimic BC-RNN（obs + `actions(160,7)`=6DoF 含姿态） |
| ② 数据采集（数据集契约） | 必须字段：RGB 图像流、状态向量、动作向量、时间戳、集号/帧号、`next.done`；手部动作从第一视角画面内估计（MediaPipe），retargeting 后成为 action |
| ③ 头戴硬件（一一对应） | **头戴摄像头**（RGB 图像流+手部关键点）+ **头戴 IMU**（ego-motion 补偿，非数据集字段）+ **SOC**（落盘+手部推理）。A 档 ≈370-555 元、B 档 ≈700-945 元，均 ≤1000 元 |
| ④ v9 核心几何标签 | 逐帧 `observation.hand_joints`（21×3，世界系）+ `observation.wrist_pose`（xyz + 四元数，世界系 6DoF）。头戴 IMU 单独不够，必须双目和/或 SLAM（§7.1–§7.2） |
| ⑤ v9 语言与佩戴 | 稠密时间分段子任务标注；无感佩戴：头戴重量、可长时间佩戴、采集段自动上传（§7.3–§7.4） |
| ⑥ v10 质检与覆盖 | 四级标注、片段级产出率、环境/物体/任务/动作覆盖空档（§8）。开放数据集上的实现见 `scripts/egodata/` |
| 明确不需要 | 手腕相机、Flex 手套、按钮/LED、麦克风/显示屏。双目在「只满足当前 ACT 输入契约」时仍非数据集字段（§3.3）；一旦要写世界系手腕 6DoF，双目/SLAM 变为必需（§7.2） |

---

## 1. ① 模型训练消费什么（依据：工程内真实配置）

### 1.1 LeRobot ACT（pusht，你训练并评估的模型）

依据：`outputs/train/act_pusht_official/checkpoints/100000/pretrained_model/config.json`（已核实）+ `train.log`

```json
"input_features": {
  "observation.image": { "type": "VISUAL", "shape": [3, 96, 96] },
  "observation.state":  { "type": "STATE",  "shape": [2] }
},
"output_features": {
  "action": { "type": "ACTION", "shape": [2] }
},
"n_obs_steps": 1, "n_action_steps": 100, "chunk_size": 100,
"vision_backbone": "resnet18"
```

**模型消费的数据契约**：输入 = 图像 + 本体状态；输出 = 动作（100 步 chunk）。训练时每帧需 `timestamp`、`episode_index`、`frame_index`，归一化需各字段 `min/max/mean/std`。

### 1.2 robomimic BC-RNN（MimicGen Square）

依据：`demo.hdf5` 实测（1000 episodes；注：`/tmp/core_datasets` 已清理，数值以历史实测记录为准）；Square 生成规模可另见仍存在的 `outputs/mimicgen_gen.log`（`success_rate 47.8%`、`num_success 1000 / num_attempts 2091`）

```json
"dataset_keys": ["actions", "rewards", "dones"],
"seq_length": 10, "frame_stack": 1
```

- obs 含 `robot0_eef_pos`（3D 位置）+ `robot0_eef_quat`（4D 四元数姿态）+ 速度 + 夹爪开度
- **`action` 形状 (160, 7) = 3 位置 + 3 旋转（轴角）+ 1 夹爪 → 动作空间 6DoF 且含姿态**

**含义**：真实采集的 `action` 必须包含**末端姿态**。姿态分量由单目视觉（MediaPipe 关键点几何/PnP/MANO）估计，精度粗；**头戴 IMU 只能补偿头部运动，无法直接测量手部旋转**（头与手是不同刚体）。

### 1.3 官方采集范式（工程内现成示例）

依据：`lerobot/examples/phone_to_so100/record.py`

```python
FPS = 30
camera_config = {"front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS)}
```

→ 官方范式为 **640×480 @ 30fps** + `LeRobotDataset`（parquet + mp4）落盘。

---

## 2. ② 数据采集必须产出什么（依据：数据集 features 契约）

依据：`data/pusht/meta/info.json`（features 全字段，已核实：`fps=10`、`total_episodes=206`）

| # | 必须的数据类型 | 由哪个模型/环节强制 | 头戴设备角色 |
|---|---|---|---|
| 1 | `observation.image` RGB 图像流 | ACT `VISUAL` / BC-RNN image | **头戴摄像头产出** |
| 2 | `observation.state` 状态向量 | ACT `STATE` / BC-RNN obs | **非头戴**：机器人侧写入，与图像同时间戳对齐 |
| 3 | `action` 动作向量（6DoF 含姿态） | ACT `ACTION` / BC-RNN actions | **非直接采集**：由画面内人手关键点经 retargeting/IK 映射得到 |
| 4 | `timestamp` 每帧时间戳 | LeRobot 格式 + 多流对齐 | 采集软件单调时钟打标（10-50ms 级；<5ms 需硬件触发） |
| 5 | `episode_index` / `frame_index` | LeRobot 分集存储 | 采集软件编号 |
| 6 | `next.done` / `next.reward` | BC-RNN / LeRobot 分集 | 软件生成（`next.done` 仅末帧 True） |
| 7 | 归一化统计（min/max/mean/std） | ACT `MEAN_STD` | 采集后离线生成（非采集时要求） |
| 8 | 手部动作（人手关键点，中间表示） | 纯头显定位新增 | **头戴摄像头画面内 MediaPipe 估计**，retargeting 后写入 `action` |
| 9 | `observation.hand_joints` 逐帧 21×3，**世界系** | v9 预训练核心标签（§7.1） | 画面内手部几何 × 相机位姿；不能只靠头戴 IMU |
| 10 | `observation.wrist_pose` 逐帧 7 维（xyz+四元数），**世界系 6DoF** | v9 预训练核心标签（§7.1） | 双目（或等价度量深度）+ SLAM/VIO；头戴 IMU 只辅助，不单独构成该标签 |
| 11 | `language_segments` 稠密时间分段子任务 | v9 语言监督（§7.3） | 采集后标注；与帧时钟同一时间基准 |

**频度结论**（已核实）：pusht 数据集实际 **10fps**（`info.json` `fps=10`）。采集 **≥10fps 即满足当前契约**；≥30fps 是 LeRobot 官方范式与手部追踪质量推荐值而非硬性。state/action 采样频率由机器人控制频率决定（MimicGen/robosuite 默认 20Hz），硬性要求是**与图像同一时间基准对齐**。注：BC-RNN 的 `rollout.n=50` 是评估 episode 数、`horizon=400` 是单次 rollout 最大步数——均不是采样频率。

**分辨率结论**：≥640×480 是模型输入下限（模型内部 resize 到 96×96）；纯头显路线推荐 ≥720p（画面内手部像素越多，MediaPipe 越稳）。

---

## 3. ③ 头戴硬件器件需求（数据集/训练契约 → 硬件，一一对应）

### 3.0 严格对照表（无追溯 → 不加硬件）

**判别标准**：凡影响 `action` / `observation.image` **正确性**的硬件即必需，不要求其数据本身进数据集字段（IMU 即属此类）。

| 契约项 | 对应硬件 | 是否必须 | 证据 |
|---|---|---|---|
| `observation.image`（RGB 视频流） | **头戴摄像头** | ✅ 必须 | ACT `VISUAL` 输入；数据集必须有视频流 |
| `observation.image` 帧稳定（相机随头运动） | **头戴 IMU**（VIO/ego-motion 补偿） | ✅ 必须（非数据集字段） | 无补偿则画面内手部轨迹被头部晃动污染（对标 EgoDex/EgoMimic 头显 SLAM 方案） |
| `action` 位置（3D） | 头戴摄像头 + 画面内 MediaPipe（PnP 恢复） | ✅ 必须 | Square `action` 实测 6DoF；位置由视觉估计 |
| `action` 姿态（3D 旋转） | 同上（姿态精度⚠️ 依赖单目视觉） | ⚠️ 部分覆盖 | 头戴 IMU ≠ 手部 IMU；**姿态升级 = 双目视差三角测量（§3.3 可选），非 IMU** |
| `observation.hand_joints` / `observation.wrist_pose`（世界系，v9） | 头戴相机 + **双目或 SLAM**（IMU 仅辅助 VIO） | ✅ 写该标签时必须 | 头戴 IMU 测的是头，不是手，也给不出度量世界系。见 §7.2 |
| `action` 夹爪维度 | MediaPipe 手指屈伸（来自画面） | ✅ 随头戴相机 | 无需手套 |
| `observation.state` | — | ⛔ 非头戴 | 来自机器人本体 |
| `timestamp` / `episode_index` / `frame_index` | — | ⛔ 非硬件 | 采集软件生成 |
| `next.done` / `next.reward` / `next.success` | — | ⛔ 非硬件 | 软件/录入生成 |
| （无契约） | 手腕相机 | ⛔ 不需要 | 纯头显路线手部关键点从头戴画面取 |
| （无契约） | Flex 手套 | ⛔ 不需要 | 只能测单轴屈伸，无法提供 6DoF 姿态；且无手套 |
| （无契约） | 按钮/LED | ⛔ 非必需 | 纯便利件 |

### 3.1 器件规格 → 选型（按「严格必需」控制成本）

**档次 A：严格必需（头戴相机 + 头戴 IMU + SOC，合计 ≈370-555 元，中位 ≈460）**

| # | 部件 | 选型 | 参考价 | 对应数据契约 |
|---|---|---|---|---|
| 1 | 头戴摄像头 | USB 720p 卷帘快门 30fps 广角 ≥110°（镜头集成） | ~50-100 元 | `observation.image` + 画面内手部关键点 |
| 2 | 头戴 IMU | MPU-6050 / BMI270 6 轴 I2C | ~15-40 元 | 相机 ego-motion 补偿（非数据集字段） |
| 3 | SOC | 树莓派 Zero 2W（算力瓶颈见 §5） | ~150-200 元 | LeRobot 落盘（parquet+mp4）+ MediaPipe 推理 |
| 4 | OTG 转接线 | 单相机 OTG 线 | ~5-25 元 | 相机 → SOC 通路 |
| 5 | 存储 | 64GB microSD U3 | ~50 元 | 连续落盘 |
| 6 | 电池 | 10000mAh 充电宝（分体放口袋） | ~60-100 元 | 供电 |
| 7 | 头戴支架 | 头箍 + 3D 打印相机座 | ~40 元 | 结构固定 |

**档次 B：质量升级（推荐，中位 ≈825 元，≤1000 元）**——替换 #1：

| 升级项 | 选型 | 增量价 | 理由 |
|---|---|---|---|
| 头戴相机 | AR0234 1080p 全局快门模组 + M12 2.1mm 广角（≈130°~150°） | +~330-390 元 | 快速手部运动无果冻效应；分辨率高 → 手部关键点像素多，MediaPipe 更稳 |

**档次 C：算力升级（仅压测丢帧时启用）**——升级 Orange Pi Zero 3（官方价 1GB=99/2GB=149/4GB=199 元，2GB 已低于 Zero 2W 的 150 元，**同价位或更低**）或将 MediaPipe 推理下放 PC 端后处理。

**镜头焦距 ↔ FOV 对照**（采购勿混）：2.8mm≈100°~120°；2.1mm≈130°~150°；要 160°+ 需 1.2~1.8mm 鱼眼。

### 3.2 明确不加的硬件

**手腕相机、手腕 IMU**（纯头显定位，手部关键点从头戴画面取）、**Flex 手套**（无手套且只能测单轴屈伸）、**按钮/LED**（便利件）、**麦克风/显示屏**。**双目默认不加**（当前数据集/训练无要求，仅作为姿态精度升级路线可选启用，见 §3.3）。**例外**：若落盘 §7 的世界系手腕 6DoF，双目和/或 SLAM 不再是可选项。

### 3.3 双目升级路线（可选，非必需）

**数据集/训练侧证据（已核实）**：pusht features 无 depth/stereo 字段；Square `agentview_image` + `robot0_eye_in_hand_image` 是两路独立单视图、训练配置 `depth: []`；ACT `input_features` 仅单 RGB + state。**结论：当前数据集与训练管线均不要求双目**——双目只提升采集阶段手部关键点 3D 估计质量（姿态精度），是**质量升级项，不是达标项**。

采用须注意（成本/同步/算力连锁，预算大概率破 1000 元）：两路需帧级同步（带外部触发引脚模组或 OAK-D 成品）；立体匹配 + 双路解码 + 双路 MediaPipe 推理，Zero 2W 确定不够，需 RK3566（NPU）或 OAK-D（VPU）。**归因不变：解决姿态的是双目视觉三角测量，不是 IMU；头戴 IMU 仍只做 ego-motion 补偿。**

### 3.4 可下单 BOM 采购清单（CSV，证据链版）

**文件**：`docs/headcam_bom.csv`（UTF-8 with BOM，Excel 双击直接打开；生成脚本 `scripts/build_headcam_bom.py`，可复现再生成）。

**每行 15 列 = 一条完整采购证据链**：`档位 / 序号 / 部件 / 具体型号规格 / 参考价 / 数量 / 淘宝关键词 / 1688关键词 / 提供的数据（训练用途） / 数据格式标准 / 对应数据契约（数据集字段） / 依据（证据来源） / 必需性 / 采购原因 / 采购注意`。即：**这件器件产出什么训练数据 → 用什么格式标准落盘 → 对应数据集哪个字段 → 依据什么证据 → 为什么必须买**——CSV 本身就是采购审批的依据材料。

**依据列口径**：A 档（严格必需）行的依据是**工程内真实配置**（ACT config.json、pusht info.json、LeRobot record.py）；B/C/可选行的依据是**文档内分析结论（§3.1/§3.3）+ 外部行情/官方价**——因为工程内训练契约对这些升级项本无要求（无追溯则不要求）。**采购关键词为「—」的行**（软件替代、双目可选）是软件/方案项，无需采购，不是漏填。

| 档位 | 部件 | 具体型号/规格 | 参考价 | 必需性 |
|---|---|---|---|---|
| A 严格必需 | 头戴摄像头 | USB UVC 720p 广角模组（OV5640/OV2710，卷帘，≥110°） | 50-100 元 | ✅ |
| A 严格必需 | 头戴 IMU | MPU-6050（GY-521）/ BMI270 6 轴 I2C | 15-40 元 | ✅（非数据集字段） |
| A 严格必需 | SOC | 树莓派 Zero 2W | 150-200 元 | ✅ |
| A 严格必需 | OTG 转接线 | Micro-USB/Type-C OTG 转 USB-A 母座 | 5-25 元 | ✅ |
| A 严格必需 | 存储 | 64GB microSD U3 | 50 元 | ✅ |
| A 严格必需 | 电池 | 10000mAh 充电宝（分体放口袋） | 60-100 元 | ✅ |
| A 严格必需 | 头戴支架 | 运动头带 + 3D 打印相机座 | 40 元 | ✅ |
| B 质量升级（推荐） | 头戴摄像头（替换 A#1） | AR0234 1080p 全局快门 + M12 2.1mm（≈130-150°） | +330-390（整机 380-490） | 🔶 可替换 |
| C 算力升级（丢帧时启用） | SOC（替换 A#3） | Orange Pi Zero 3（2GB/4GB，带 USB3） | 99-199 元 | 🔶 可替换 |
| C 软件替代（零成本） | MediaPipe 推理下放 PC | 采集端只落盘原始帧，PC 后处理 | 0 元 | 🔶 软件替代 |
| 可选（§3.3） | 双目模组 | OAK-D-LR 或双 AR0234+外部触发+RK3566 | ≥1500-3000 元 | 🔶 可选 |

**成本汇总（与 §3.1 一致）**：A 档 370-555 元（中位 462.5）；B 档 700-945 元（中位 822.5≈825）；两档均 ≤1000 元。C 档 SOC 升级不增加总价（Orange Pi Zero 3 与 Zero 2W 同价位且算力更强）。

**采购注意（关键）**：① 摄像头必须选 **UVC 免驱协议**（Linux `v4l2` 直接可读），避开仅 Windows 驱动的工业型号；② AR0234 全局快门模组淘宝行情约 300-500 元（深圳泓嘉精密等厂家在售），务必确认 M12 接口可换 2.1mm 镜头；③ Orange Pi Zero 3 官方价 1GB=99/2GB=149/4GB=199 元，与 Zero 2W 同价位或更低且算力更强，**建议优先选它**；④ 树莓派 Zero 国内电商约 140+ 元。

---

## 4. 采集软件必须满足的格式契约（落盘要求）

输出必须能被现有训练代码直接消费（LeRobot v3.0）：

```
数据集根目录/
├── meta/
│   ├── info.json          # features 声明（image/state/action 的 type/shape/fps）、total_episodes
│   ├── episodes.jsonl     # 每集：任务描述、时长
│   └── stats.json         # 各字段 min/max/mean/std（采集后离线生成）
├── data/chunk-000/
│   └── file-000.parquet   # observation.state / action / timestamp / episode_index / frame_index
└── videos/observation.image/
    └── chunk-000/file-000.mp4   # H.264/AV1 图像流
```

对齐参考：工程内 `data/pusht/meta/info.json` 的 features 结构（`observation.image` dtype=video、`observation.state`/`action`/`timestamp` dtype=float32）。

**手部字段落盘约定（纯头显）**：
- **方式 A**：MediaPipe 手部关键点（21×3 float32）写入 `observation.hand`，retargeting 训练前离线完成——保留原始信息，便于换映射方法。
- **方式 B**：采集端直接 retargeting/IK 写入 `action`——采集即训练可用，但映射固化。
- 两者均须与图像**同时间戳**写入同一 parquet，并在 `info.json` 声明字段。
- **坐标系一致性**：PnP/MANO 恢复的关键点坐标须先做**相机内参 + 外参标定**（头戴相机 ↔ 世界/头部坐标），否则同一物理点跨帧坐标系不一致。

**v9 追加字段**（与上面的 image/state/action 并列，不替换它们）：

- `observation.hand_joints`：`float32[21,3]`，米，世界系，MediaPipe 21 点顺序。
- `observation.wrist_pose`：`float32[7]`，世界系手腕位置（米）+ 四元数 `xyzw`。
- `meta/language_segments.jsonl`：每集一行，稠密子任务分段（§7.3）。

三者都要在 `info.json` 里声明。世界系的定义、为什么头戴 IMU 不够、以及无感佩戴要求见 §7。

---

## 5. 验证清单（组装后确认硬件达标）

- [ ] 采集一集视频，确认分辨率 ≥720p、帧率 ≥30fps、彩色
- [ ] 快速挥动手部录制 10s，无果冻/撕裂（全局快门验证）
- [ ] 头戴 IMU 数据率 ≥200Hz，与视频时间戳同一时钟基准（软件 10-50ms 级）
- [ ] 标定头戴相机内参（棋盘格）+ 外参，多次佩戴/重启后坐标系一致
- [ ] 产出文件能被 `lerobot` 训练脚本直接读取（`info.json` 结构自检）
- [ ] 戴好头戴，MediaPipe 跑握拳/张开/抓取 3 组动作：关键点连续、无长时间跟丢、遮挡恢复 <1s、手不出画
- [ ] （v9）同一段视频能导出世界系 `hand_joints` 与 `wrist_pose`：关掉双目/SLAM、只留头戴 IMU 时，该标签应被标为无效而不是静默写成 IMU 读数
- [ ] （v9）一集的 `language_segments` 覆盖整段时间，相邻段首尾相接
- [ ] （v9）头戴部分不用手扶也能连续佩戴 ≥1 小时；封口后的 episode 在有网络时自动上传，无需拷卡

---

## 6. 已知约束与风险（纯头显）

1. **手部动作精度是最大风险点**：关键点从头戴第一视角画面估计（MediaPipe），短板：手出画即丢失（需头跟随手）、遮挡/手指重叠时跟丢、单目深度幻觉与姿态歧义（无手腕 IMU 冗余）。粗粒度抓取/操作可用（EgoDex/EgoMimic 已验证 30Hz 可行）；精细灵巧操作需预算大幅上升（≥3,000-5,000 元上电磁/光学动捕）。
2. **卷帘快门风险**：预算紧选卷帘相机时，快速手部运动产生果冻效应，需训练侧数据增强缓解。
3. **软件时间戳上限**：1000 元内无法做 PTP/亚毫秒硬件同步，软件时钟误差 10-50ms；对纯视频模仿学习可用，对精密 VLA 需升级同步方案。
4. **Zero 2W 算力瓶颈**：四核 A53@1GHz + 512MB，USB UVC 采集无硬编码通道，1080p@30 全链路（抓帧+编码+parquet+IMU+MediaPipe 推理）可能丢帧。缓解：720p@30、v4l2m2m 硬编码、推理下放 PC；**组装后先做 10 分钟满载压测**，丢帧则走档次 C。
5. **当前模型均为 benchmark 仿真数据训练**（pusht 96×96、MimicGen Square），真实采集数据接入后需重新做归一化统计（stats.json）与图像尺寸适配。
6. **IMU 必需但非数据集字段，且头戴 IMU ≠ 手部 IMU**：头戴 IMU 只做 ego-motion 补偿、只测头部姿态，不能提供手部旋转自由度——6DoF action 姿态分量由单目视觉估计，精度粗。需要手部姿态冗余只能回手腕 IMU/手套路线，与纯头显约束冲突。v9 的世界系手腕 6DoF 同样不能由头戴 IMU 单独给出，路径是双目/SLAM（§7.2），不是改戴手腕 IMU。

---

## 7. EgoScale 对齐（v9）：世界系手部标签、语言、无感佩戴、数据配比

本节是预训练数据契约的**参考方案**，依据 Jim Fan「Robotics End Game」里 NVIDIA EgoScale 的公开口径，不是工程内已有 checkpoint。v8 的 ACT/pusht、BC-RNN/Square 契约（§1–§3）继续有效：那些模型今天不读 depth，也不读下面这些字段。两套契约并存——当前策略训练看 §2 的 image/state/action；要做 EgoScale 式预训练，采集必须额外交出本节的标签。

### 7.1 核心标签：逐帧手部关节 + 世界系手腕 6DoF

每一帧都要有，并且和 `observation.image` 同一时间戳：

| 字段 | shape / dtype | 坐标系 | 语义 |
|---|---|---|---|
| `observation.hand_joints` | `float32[21, 3]` | **世界系**，单位米 | 21 个手部关节位置，顺序与 MediaPipe Hands 一致（0 为手腕根） |
| `observation.wrist_pose` | `float32[7]` | **世界系** | 手腕 6DoF：`xyz`（米）+ 四元数 `xyzw` |

世界系指这一集里重力对齐、不跟相机一起动的 SLAM 世界系。相机系关键点、头部 IMU 系姿态都**不算**这条标签已经完成。

落盘：与 image 同 timestamp 写入 parquet，并在 `meta/info.json` 声明 shape。缺测帧写入 NaN，并另有 `observation.hand_valid`（`bool`，该帧手部几何是否可用），禁止用 0 填充冒充有效姿态。

### 7.2 头戴 IMU 单独不够：要双目和/或 SLAM

头戴 IMU 测的是头部刚体的角速度和比力。手是另一刚体，IMU 看不到手指，也给不出手腕在世界系里的位置。

世界系手腕 6DoF 至少要两条几何：

1. **手在相机系里的度量三维**：双目视差（或等价的度量深度）把关节从像素三角化到相机系。单目 MediaPipe + PnP 有尺度歧义，不能单独当世界系米制标签。
2. **相机在世界系里的位姿** `T_world_cam`：视觉 SLAM，或视觉惯性里程计（VIO）。头戴 IMU 可以给 VIO 提供短期旋转、估计零偏，但没有视觉地图时，IMU 双重积分的位置会漂，不能当作世界系。

合成：`p_world = T_world_cam · p_cam`。手腕朝向同样左乘相机位姿。只录头戴 IMU、不做双目也不做 SLAM，这条核心标签无效。

这和 §3.3 不冲突：§3.3 说的是**当前 ACT/Square 训练不消费 depth 字段**，所以「只为了今天的 ACT 输入」不必买双目。§7 的标签是另一份契约，写它的时候双目和/或 SLAM 是必需路径。

### 7.3 稠密时间分段的子任务语言

不是一集一个任务名。把一集切成首尾相接的短时段，每段一句正在做的子任务。

文件：`meta/language_segments.jsonl`，每集一行。

```json
{"episode_index": 0, "segments": [
  {"t_start": 0.00, "t_end": 1.35, "subtask": "伸手接近杯柄"},
  {"t_start": 1.35, "t_end": 2.80, "subtask": "握住杯柄"},
  {"t_start": 2.80, "t_end": 5.10, "subtask": "把杯子放到托盘上"}
]}
```

要求：

- `t_start` / `t_end` 使用与帧 `timestamp` 相同的时钟，单位秒。
- 稠密：段与段首尾相接（上一段 `t_end` = 下一段 `t_start`），并盖住该集第一帧到最后一帧。空档若短于 0.5 秒，并入相邻段；更长的空档写成明确子任务（如「停住」），不留未标注空洞。
- `episodes.jsonl` 里的整集任务描述可以保留，但不能代替分段。

### 7.4 无感佩戴（unobtrusiveness）

采集设备要让人忘记自己戴着它，否则第一视角视频里的动作会变形。三条验收，不引用未经公开的克重指标：

| 要求 | 含义 | 本工程验收 |
|---|---|---|
| 重量 | 头上只留相机和必要支架，电池与 SOC 不堆在头上 | 与 §3.1 一致：充电宝分体放口袋。戴好后不需要用手扶着设备 |
| 长时间佩戴 | 能连续戴着做日常操作，而不是每几分钟摘一次 | 连续佩戴 ≥ 1 小时，不勒、不滑、不热到必须摘下。设计目标是一天内累计戴数小时 |
| 自动上传 | 佩戴者不拷贝存储卡、不手动整理文件 | episode 封口后，有网络时自动上传到数据集目录；失败重试。本地副本保留到上传确认 |

### 7.5 EgoScale 式数据配比（参考方案）

公开口径的两段式配比，用作采集规划的参照，不是本仓库已经存下的小时数：

| 阶段 | 数据 | 规模（参考） | 监督 |
|---|---|---|---|
| 预训练 | 第一视角人类视频 | **约 21,000 小时** | 预测逐帧手部关节 + 手腕位姿（§7.1） |
| 动作微调 | 动捕（mocap） | **约 50 小时** | 带动作标签的人体运动 |
| 动作微调 | 遥操作（teleop） | **约 4 小时** | 机器人动作 |

预训练阶段的最优验证损失对预训练小时数呈 **log-linear**：损失大约是 `a + b·ln(小时)`。小时数按数量级增加，损失按直线下降，而不是按小时线性下降。本仓库用 `scripts/scaling_law.py` 在多次不同数据量的 run 上拟合这条线（示例配置 `examples/scaling_law_runs.yaml`）。有效 run 不足时脚本跳过，不改其余训练报告。

和文末「微调 50–150 条」的关系：那是**当前没有大规模第一视角预训练时**，每个任务的真实演示下限。§7.5 的 21k / 50h / 4h 是另一条路线的参照配比；两段式方案里怎么引用，见 `docs/twostage_pretrain_finetune_plan.md`。

### 7.6 和 v8 契约怎么同时成立

- 今天的 ACT 仍吃 `observation.image` + `observation.state`，输出 `action`。这些字段继续按 §2、§4 落盘。
- §7.1 的关节和手腕位姿是预训练标签，可以离线从视频估计后再写回 parquet；估计失败的帧用 `hand_valid=false`，不要把单目猜测标成世界系米制真值。
- 语言分段是标注产物，不要求头戴硬件上有麦克风。v8「不加麦克风」仍然成立。

### 7.7 自有头戴会话目录

设备还没有。采集软件按这个目录落盘后，`scripts/convert_headcam.py` 写成和 EgoDex 适配器相同的统一 episode，QC、覆盖、导出、可视化不用改。

```
session/
  metadata.json     episode_id、fps、task、instruction、environment、图像宽高、objects、verbs
  timestamps.csv    frame_index,timestamp_s。没有则用 frame/fps
  imu.csv           角速度和比力。转换时只记路径和行数，不写入 JSON，也不积分
  calib.yaml        Kalibr camchain、简单 YAML，或 OpenCV FileStorage（cameraMatrix1/2、R、T）
  rgb.mp4           单目或彩色参考。没有时用 stereo/left.mp4
  stereo/left.mp4   左目
  stereo/right.mp4  右目，只提供 2D
  slam.tum          可选。TUM：timestamp tx ty tz qx qy qz qw
  hands.json        可选。后端已经算好的每帧 21 点
```

几何按 §7.2 合成，实现在 `scripts/headcam/hand_pose.py`：

1. 手部后端给出左相机系 21 点（MediaPipe 顺序）和 2D。主后端是 [HaMeR](https://github.com/geopavlakos/hamer) 或 [WiLoR](https://github.com/rolpotamias/WiLoR)，都要 MANO 右手模型。MANO 是马普所非商业许可，从 https://mano.is.tue.mpg.de 注册下载，**权重不进仓库**。没有权重时用 MediaPipe Hands（CPU，无 MANO）。它的三维不是公制相机系：单目路径用可配置的手腕深度先验（默认 0.55 m）把 2D 抬进相机系，这不是米制深度，公制尺度要靠双目三角化。HaMeR / WiLoR 把 crop 相机变到全图时使用 `K_left` 的 fx、fy 和主点；不传内参时虚拟焦距是 `5000/256*max(宽,高)`（1920 宽约 37500 px），手腕会落到几十米。推理不导入 renderer（避免 pyrender / EGL），并传 `init_renderer=False`。`MANO_MODEL_DIR` 会覆盖 WiLoR 写死的 `./mano_data`。Colab GPU 上优先 WiLoR；安装 `ultralytics==8.1.34` 和 `pip install --no-build-isolation chumpy`。HaMeR 还要编译 detectron2 与 mmcv。pyrender 只用于网格可视化。`mediapipe>=0.10.30` 没有 `mp.solutions`，后端改用 Tasks `HandLandmarker`（需下载 `hand_landmarker.task`）。无头环境可能缺 `libGLESv2`，默认 CPU delegate；仍有旧接口时会退回 `mp.solutions.hands`。
2. 左右目 2D 用标定三角化到左相机系，修正单目尺度，并给出每个关节的重投影置信度。点在相机后面或已校正双目视差符号不对时置信度为 0。
3. `p_world = T_world_cam · p_cam`。`T` 来自 TUM 里时间最近的一帧。没有轨迹时 `T` 是单位阵，`coordinate_frame` 写 `camera`，这还不是 §7.1 的世界系。有轨迹时写 `slam_world`。
4. 手腕朝向：x 从手腕指向食指 MCP，掌面法向由食指 MCP 与小指 MCP 叉乘得到。缺测帧写空值，不用 0 填充。
5. 子任务和分手指令留空。语言不在这个转换器里编造。
6. 可选的时序精修在 `scripts/headcam/hand_track_refine.py`，`convert_headcam.py` 不传开关时不会改关节。`--refine` 同时打开四件事：One Euro 平滑（也可换成常速度卡尔曼）、最多补 5 帧缺测、按这一段的骨长中位数重摆关节、用手腕轨迹修正左右标签。有 TUM 时平滑在世界系里做，避免把头的转动当成手抖。补上的帧带 `filled=true`，置信度写成 0。QC 看到 `filled`，或者置信度低于 0.5，都不把这帧当成跟踪成功。没有 MANO 文件时形状就是 20 段骨长，不会把骨长写成 betas。

整机标定和 SLAM 仍在仓库外做。这里只读已经算好的 yaml 和 TUM。

---

## 8. 覆盖度、产出率与四级标注（v10）

没有自有头戴设备时，先在开放数据集上按同一套定义做质检和覆盖统计。EgoDex 测试集的位姿已经在 ARKit 世界系里（设备端 SLAM 的结果），所以这一节的工具**消费**世界系轨迹，不在本仓库里重跑 SLAM。自有会话按 §7.7 转成同一份 JSON 之后，也走这些工具。

### 8.1 四级标注

v9 §7.3 只有「一集切成首尾相接的子任务」。v10 把它放进四层，由粗到细：

| 层级 | 粒度 | 写什么 |
|---|---|---|
| ENVIRONMENT | 整段 | 场景类别：桌面、厨房、起居、工作间、户外，或未知 |
| TASK | 整段 | 任务名 + 一句整段指令 |
| SUBTASK | 时间分段 | 与 v9 `language_segments` 相同：首尾相接，盖住整段 |
| INSTRUCTION | 时间分段、分手 | 这一小段里左手、右手或双手具体在做什么 |

样例（亦可直接校验 `examples/hierarchy_annotation.json`）：

```json
{
  "duration_s": 2.0,
  "environment": {
    "name": "tabletop",
    "detail": "table:wood, position:sitting, background:brown",
    "source": "egodex_attr"
  },
  "task": {
    "name": "open_close_insert_remove_case",
    "instruction": "打开盒子，取出垫子和鸭子，再把盒子盖上。"
  },
  "subtasks": [
    {"t_start": 0.0, "t_end": 1.0, "text": "打开盒盖"},
    {"t_start": 1.0, "t_end": 2.0, "text": "取出盒内物品并合盖"}
  ],
  "instructions": [
    {"t_start": 0.0, "t_end": 1.0, "hand": "right", "text": "右手扳开卡扣"},
    {"t_start": 1.0, "t_end": 2.0, "hand": "left", "text": "左手取出垫子和鸭子"}
  ]
}
```

约束：

- `environment.name`、`task.name`、`task.instruction` 不能空。
- `subtasks` 按时间排序后从 0 开始，上一段 `t_end` 等于下一段 `t_start`（容差 1ms），最后一段的 `t_end` 等于片段时长。
- `instructions[].hand` 只能是 `left`、`right`、`both`。时间必须落在片段内。分手指令不必彼此首尾相接：两只手可以只在其中一段时间有指令。
- EgoDex 的 HDF5 属性只有整段的 `environment`、`task`、`llm_description`。转换器填 ENVIRONMENT 和 TASK，`subtasks` 与 `instructions` 留空，不把整段指令匀成假的时间分段。校验器用 `--strict` 时把缺级当成失败。

校验：`python scripts/validate_hierarchy.py examples/hierarchy_annotation.json --strict`

### 8.2 产出率与自动 QC

**产出率（yield）** = 通过片段级 QC 的帧数 / 原始帧数。被拒绝的整段不进入训练集，哪怕其中有几帧是好的。报告里同时给出坏帧比例，方便以后改成「切掉坏段再留好段」。

一条片段被拒绝，当且仅当坏帧比例 **> 20%**（`max_bad_fraction`）。一帧只要命中下面任一条，就是坏帧。阈值是 `scripts/egodata/qc.py` 里的默认值，调用时可以改。

| 标记 | 含义 | 现在用的信号 | 默认阈值 |
|---|---|---|---|
| `hands_out_of_frame` | 手出画或跟丢 | 两只手的手腕都投影到画面外、在相机后方，或手腕置信度低于下限。投影用相机位姿的逆，按 OpenCV 约定（+Z 向前、+Y 向下）。主点两倍取整得到宽高：EgoDex 实测内参主点 (960, 540) → 1920×1080 | 置信度 < 0.5 |
| `view_drift` | 视线飘离手 | 仍在画面内的手腕中点，与相机前向（位姿矩阵第三列）的夹角过大 | > 50° |
| `blur` | 运动模糊 | 相邻帧相机旋转的角速度。没有像素时用它代替拉普拉斯方差 | > 1.5 rad/s |
| `staged_static` | 摆拍或干等 | 两只手腕的世界系速度都低于下限，且连续时长达到阈值。更短的停顿保留 | < 0.015 m/s 且连续 ≥ 1.0 s |

EgoDex 手腕置信度的含义来自数据集说明：它表示这只手是否被整体检测到，低置信度即跟丢或出画，不能把缺测写成 0。

报告：`python scripts/egodata_qc.py --episodes <统一JSON目录> --html outputs/yield_report.html --csv outputs/yield_episodes.csv`

### 8.3 覆盖词表

采集和抽查都按四根轴计数，空档是封闭词表里片段数为 0 的取值。任务名和原始物体名是开放词表，只统计出现过的值。

| 轴 | 封闭取值 |
|---|---|
| environment | `tabletop` `kitchen` `living_room` `workshop` `outdoor` `unknown` |
| object_class | `container` `tool` `cloth` `food` `electronics` `toy` `furniture` `tableware` `other` `unknown` |
| action_type | `pick` `place` `open` `close` `pour` `wipe` `fold` `tie` `cut` `stir` `insert` `remove` `stack` `screw` `throw` `type` `other` `unknown` |
| task / object | 开放词表。EgoDex 用目录名或属性 `task`，物体用 `llm_objects` |

EgoDex 的 `environment` 属性是自由文本（例如 `table:wood, position:sitting, background:brown`）。含 table / desk / sitting 归入 `tabletop`，含 kitchen / fridge / sink 归入 `kitchen`，对不上则为 `unknown`。`llm_verbs` 用子串归入动作类型，归不上的记为 `other`。

报告：`python scripts/egodata_coverage.py --episodes <统一JSON目录> --html outputs/coverage_report.html --csv outputs/coverage_counts.csv`

---

## 附：数据量参考（一次有效微调需要的最少演示数）

| 场景 | 每任务最少演示数 | 依据 |
|---|---|---|
| A. 微调已有预训练模型 | **50-150 条**（建议 100 起步） | ALOHA 50 条业界基线；银河通用「十亿帧合成预训练 + <1 人天真实微调」 |
| B. 从零训练（当前管线） | **500-1000 条** | Square 1000 条 → BC-RNN 68-76%；pusht 206 集 → ACT 收敛但评估 0-5% |

单条时长 8-30s 完整闭环（接近→接触→操作→释放），覆盖 ≥5 种起始状态变体；图像 30fps / action 按机器人控制频率。存储：64GB microSD 可存数百条（150 条 ≈2-3GB），**不是瓶颈**。
