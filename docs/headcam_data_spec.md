# 头戴式数据采集设备规格（纯头显 · v8）

> **适用工程**：`/mnt/sda/app/robot` ｜ **文档日期**：2026-08-01
>
> **推导逻辑（一条链，一一对应）**：模型训练消费什么 → 数据采集必须产出什么 → 头戴硬件器件需求是什么。
> **每一条「必须」都引用工程内真实存在的 checkpoint / 训练配置 / 数据集字段作为依据，不靠猜测**。
>
> **v8 变更**：按「纯头显（无手腕相机、无手套）」收敛范围，删除训练数据报告、外部产品规格（Ego 头显 / iPhone）、未来 Pipeline 接入检查流程、版本历史等非设备规格内容。

---

## 0. 结论速览（TL;DR）

| 推导层 | 结论 |
|---|---|
| ① 模型训练（真实配置） | 2 套：LeRobot ACT（`image[3,96,96]` + `state[2]` → `action[2]`）+ robomimic BC-RNN（obs + `actions(160,7)`=6DoF 含姿态） |
| ② 数据采集（数据集契约） | 必须字段：RGB 图像流、状态向量、动作向量、时间戳、集号/帧号、`next.done`；手部动作从第一视角画面内估计（MediaPipe），retargeting 后成为 action |
| ③ 头戴硬件（一一对应） | **头戴摄像头**（RGB 图像流+手部关键点）+ **头戴 IMU**（ego-motion 补偿，非数据集字段）+ **SOC**（落盘+手部推理）。A 档 ≈370-555 元、B 档 ≈700-945 元，均 ≤1000 元 |
| 明确不需要 | 手腕相机、Flex 手套、按钮/LED、双目（可选升级，非必需）、麦克风/显示屏 |

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

**手腕相机、手腕 IMU**（纯头显定位，手部关键点从头戴画面取）、**Flex 手套**（无手套且只能测单轴屈伸）、**按钮/LED**（便利件）、**麦克风/显示屏**。**双目默认不加**（当前数据集/训练无要求，仅作为姿态精度升级路线可选启用，见 §3.3）。

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

---

## 5. 验证清单（组装后确认硬件达标）

- [ ] 采集一集视频，确认分辨率 ≥720p、帧率 ≥30fps、彩色
- [ ] 快速挥动手部录制 10s，无果冻/撕裂（全局快门验证）
- [ ] 头戴 IMU 数据率 ≥200Hz，与视频时间戳同一时钟基准（软件 10-50ms 级）
- [ ] 标定头戴相机内参（棋盘格）+ 外参，多次佩戴/重启后坐标系一致
- [ ] 产出文件能被 `lerobot` 训练脚本直接读取（`info.json` 结构自检）
- [ ] 戴好头戴，MediaPipe 跑握拳/张开/抓取 3 组动作：关键点连续、无长时间跟丢、遮挡恢复 <1s、手不出画

---

## 6. 已知约束与风险（纯头显）

1. **手部动作精度是最大风险点**：关键点从头戴第一视角画面估计（MediaPipe），短板：手出画即丢失（需头跟随手）、遮挡/手指重叠时跟丢、单目深度幻觉与姿态歧义（无手腕 IMU 冗余）。粗粒度抓取/操作可用（EgoDex/EgoMimic 已验证 30Hz 可行）；精细灵巧操作需预算大幅上升（≥3,000-5,000 元上电磁/光学动捕）。
2. **卷帘快门风险**：预算紧选卷帘相机时，快速手部运动产生果冻效应，需训练侧数据增强缓解。
3. **软件时间戳上限**：1000 元内无法做 PTP/亚毫秒硬件同步，软件时钟误差 10-50ms；对纯视频模仿学习可用，对精密 VLA 需升级同步方案。
4. **Zero 2W 算力瓶颈**：四核 A53@1GHz + 512MB，USB UVC 采集无硬编码通道，1080p@30 全链路（抓帧+编码+parquet+IMU+MediaPipe 推理）可能丢帧。缓解：720p@30、v4l2m2m 硬编码、推理下放 PC；**组装后先做 10 分钟满载压测**，丢帧则走档次 C。
5. **当前模型均为 benchmark 仿真数据训练**（pusht 96×96、MimicGen Square），真实采集数据接入后需重新做归一化统计（stats.json）与图像尺寸适配。
6. **IMU 必需但非数据集字段，且头戴 IMU ≠ 手部 IMU**：头戴 IMU 只做 ego-motion 补偿、只测头部姿态，不能提供手部旋转自由度——6DoF action 姿态分量由单目视觉估计，精度粗。需要手部姿态冗余只能回手腕 IMU/手套路线，与纯头显约束冲突。

---

## 附：数据量参考（一次有效微调需要的最少演示数）

| 场景 | 每任务最少演示数 | 依据 |
|---|---|---|
| A. 微调已有预训练模型 | **50-150 条**（建议 100 起步） | ALOHA 50 条业界基线；银河通用「十亿帧合成预训练 + <1 人天真实微调」 |
| B. 从零训练（当前管线） | **500-1000 条** | Square 1000 条 → BC-RNN 68-76%；pusht 206 集 → ACT 收敛但评估 0-5% |

单条时长 8-30s 完整闭环（接近→接触→操作→释放），覆盖 ≥5 种起始状态变体；图像 30fps / action 按机器人控制频率。存储：64GB microSD 可存数百条（150 条 ≈2-3GB），**不是瓶颈**。
