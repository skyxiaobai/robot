# 机器人数据采集与训练：全部分析过程记录

> **适用工程**：`/mnt/sda/app/robot` ｜ **文档日期**：2026-08-02
>
> 本文档汇总本工程围绕「**头戴式数据采集设备**」与「**训练管线**」的完整分析过程：从数据集/模型现状盘点 → 业界数据集调研 → 模型训练消费的数据契约 → 数据采集必须产出的字段 → 头戴硬件器件需求（一一对应、可下单）→ 两段式训练方案评估 → 每轮评审与修正记录。所有「必须」均追溯到工程内真实配置或公开证据，不靠猜测。
>
> **配套文档**：
> - `headcam_data_spec.md`（v8）——头戴设备规格主文档（模型训练→数据采集→硬件一条链）
> - `headcam_bom.csv`（证据链版）——可下单采购清单（15 列，每行自证采购理由）
> - `twostage_pretrain_finetune_plan.md`——两段式训练方案评估
> - 生成脚本：`scripts/build_headcam_bom.py`（BOM 可复现再生成）

---

## 1. 工程现状盘点（训练了什么）

### 1.1 两套训练管线（已核实）

| 管线 | 框架 | 数据集 | 输入 | 输出 | 训练结果 |
|---|---|---|---|---|---|
| ACT | LeRobot v0.6.1 | pusht（206 集，10fps） | `observation.image[3,96,96]` + `observation.state[2]` | `action[2]`（100 步 chunk） | loss 0.261@10K → 0.074@100K 收敛；评估 0-5% |
| BC-RNN | robomimic | MimicGen Square（1000 条 × 160 步 @20Hz） | obs（eef 位置+四元数+夹爪）+ image | `actions(160,7)`=6DoF+夹爪 | image 68% / lowdim 76%（Square） |

### 1.2 关键事实（磁盘实读）

- pusht `info.json`：`fps=10`、`total_episodes=206`、features **无 IMU / 无 depth 字段**
- ACT `config.json`：`input_features` 仅 image+state，`vision_backbone=resnet18`，`chunk_size=100`，`use_imagenet_stats=true`
- Square 生成：`success_rate 47.8%`（1000 成功 / 2091 尝试），`actions(160,7)` = 3 位置 + 3 旋转（轴角）+ 1 夹爪
- LeRobot 官方采集范式：`lerobot/examples/phone_to_so100/record.py` = 640×480 @ 30fps，parquet+mp4 落盘

---

## 2. 业界数据集调研（回答了哪些是「机器人公司常用」）

### 2.1 结论：ACT（模型）业界常用，pusht（数据集）不是业界数据

- **ACT**：出自 Stanford ALOHA 论文（Zhao et al. 2023），是双臂/桌面遥操作训练事实标准；LeRobot 官方核心 SoTA 策略之一；国内头部公司（智元/银河通用/加速进化/宇树/优必选）模型栈基本包含 ACT 家族，训练数据为**自采 ALOHA 式遥操作数据**，不是 pusht
- **pusht（Push-T）**：2D 仿真 benchmark，出自 Diffusion Policy 论文，用途是**验证训练代码能跑通**（smoke test），非生产数据集
- **通用性高的数据集**：Open X-Embodiment（100 万+轨迹、22 机器人、21 机构）> DROID（7.6 万轨迹、564 场景）> BridgeData V2（6 万轨迹）> 自采遥操作数据

### 2.2 银河通用（Galaxy General）数据策略：合成数据派领军者

公开证据（媒体/官方口径）：
- 训练数据约 **90% 仿真 + 少量真实**；「海量合成预训练 + 有目标的真实微调」
- **GraspVLA**：十亿帧合成「视觉-语言-动作」对预训练，后训练只需 **<1 人天遥操作**真实数据即可迁移真机
- **SynGrasp-1B**：全球首个十亿帧合成抓取数据集（Galbot+北大+港大+智源）
- 推论：小真实数据集能产出可用模型的前提是**已在十亿帧合成数据上预训练**

### 2.3 数据量反推（写入 headcam_data_spec.md §附）

| 场景 | 每任务最少演示数 | 依据 |
|---|---|---|
| A. 微调已有预训练模型 | **50-150 条**（建议 100 起步） | ALOHA 50 条业界基线 + 银河通用 <1 人天 |
| B. 从零训练（当前管线） | **500-1000 条** | Square 1000 条 → BC-RNN 68-76%；pusht 206 集 → ACT 评估 0-5% |

单条 8-30s 完整闭环、覆盖 ≥5 种起始状态、图像 30fps / action 按机器人控制频率；64GB microSD 可存数百条，非瓶颈。

---

## 3. 头戴设备规格推导（模型训练 → 数据采集 → 硬件）

### 3.1 推导链（headcam_data_spec.md 全文结构）

```
① 模型训练消费什么（ACT/BC-RNN 真实配置）
   → ② 数据采集必须产出什么（8 个必须字段，标注头戴/机器人侧/软件）
   → ③ 头戴硬件器件需求（严格对照表：契约项→硬件→是否必须→证据）
```

### 3.2 必须数据字段（每个字段标注来源角色）

| # | 字段 | 头戴设备角色 |
|---|---|---|
| 1 | `observation.image` | **头戴摄像头产出** |
| 2 | `observation.state` | 非头戴：机器人侧写入 |
| 3 | `action`（6DoF 含姿态） | 非直接采集：画面内 MediaPipe 关键点 retargeting 得到 |
| 4 | `timestamp` | 采集软件单调时钟（10-50ms 级） |
| 5 | `episode_index`/`frame_index` | 采集软件编号 |
| 6 | `next.done`/`next.reward` | 软件生成（`next.done` 仅末帧 True） |
| 7 | 归一化统计 | 采集后离线生成 |
| 8 | 手部关键点（中间表示） | 头戴画面内 MediaPipe 估计 |

### 3.3 关键物理判断（多轮评审后确认）

- **头戴 IMU ≠ 手部 IMU**：IMU 测的是头部/相机自身运动，只能做 **ego-motion 补偿**，不能提供手部旋转自由度（头与手是不同刚体）
- **单目姿态精度受限**：MediaPipe+PnP 需手部尺寸先验，深度幻觉与姿态歧义
- **双目是姿态升级路线（非必需）**：视差三角测量直接恢复关键点 3D，不需要尺寸先验；但当前数据集/训练均不要求双目（pusht 无 depth、Square 是两路独立单视图、ACT 仅单 RGB）——**双目只提升采集质量，不是达标项**
- **归因纠错**：解决姿态的是双目视觉三角测量，不是 IMU；IMU 职责收窄回 ego-motion 补偿

### 3.4 硬件选型与成本（与 headcam_bom.csv 一一对应）

| 档位 | 内容 | 成本 |
|---|---|---|
| A 严格必需 | 头戴摄像头 + 头戴 IMU + SOC + OTG 线 + 64GB microSD + 充电宝 + 支架 | 370-555 元（中位 462.5） |
| B 质量升级（推荐） | AR0234 1080p 全局快门 + M12 2.1mm（替换 A#1） | 700-945 元（中位 822.5≈825） |
| C 算力升级 | Orange Pi Zero 3（同价位或更低，算力更强） | 99-199 元 |
| C 软件替代 | MediaPipe 推理下放 PC | 0 元 |
| 可选（非必需） | 双目模组（OAK-D-LR 等） | ≥1500-3000 元（突破预算） |

价格锚点（联网查证）：Orange Pi Zero 3 官方价 1GB=99/2GB=149/4GB=199；树莓派 Zero 国内电商 140+；AR0234 全局快门模组深圳厂家（泓嘉精密等）在售 300-500 元。

---

## 4. 两段式训练方案评估（twostage_pretrain_finetune_plan.md）

### 4.1 核心结论

**能走，但不能「拿现成 Square/pusht 直接预训练→头戴微调」**。银河通用成功前提是合成数据与部署本体/视角/动作空间一致；本工程现有合成数据是**第三视角机器人臂（Square）+ 2D 仿真（pusht）**，头戴数据是**第一视角人手**——三重错位。

### 4.2 两条可行路线

| | 路线 A：视觉表征迁移 | 路线 B：同视角合成预训练 |
|---|---|---|
| 做法 | 用现有合成数据预训练 resnet18 编码器 → 头戴数据微调 action 头 | MimicGen 自定义头戴视角相机生成第一视角合成数据 → 预训练 ACT → 真实微调 |
| 真实数据 | 50-150 条/任务 | 30-100 条/任务 |
| 工作量 | 1-2 天 | 1-2 周 |
| 风险 | 低 | 中-高（第一视角人手仿真无现成开源管线） |

### 4.3 主要坑（按严重度排序）

1. 本体/视角错位无捷径
2. 动作空间未定就预训练 = 白做（ACT 输出维度一改，权重作废）
3. 真实 action 标签噪声（MediaPipe+retargeting 出的标签若错，预训练救不了）
4. 归一化统计错配需重算 stats.json
5. MimicGen 47.8% 成功率 → 过半是失败轨迹，必须过滤
6. 小数据过拟合（backbone 低 LR + 增强）
7. 缺真实评估协议

### 4.4 落地前提（阶段 0）

**动作空间必须定死**：MediaPipe retargeting 公式、action 维度/语义。否则预训练权重全部作废。

---

## 5. 评审与修正记录（每轮发现的问题 → 修复）

### 5.1 headcam_data_spec.md（v4→v8）

| 轮次 | 发现的问题 | 修复 |
|---|---|---|
| v4 | 「<5ms 同步」与软件时钟 10-50ms 自相矛盾 | 全文统一为 10-50ms；<5ms 明确标注需 PTP/硬件触发 |
| v4 | `rollout.n=50` 误读为 50Hz | 拆解 `n`(评估 episode 数)/`horizon`/`rate`，明确都不是采样频率 |
| v4 | 镜头焦距↔FOV 混成一个区间 | 按焦距分档：2.8mm≈100-120°、2.1mm≈130-150°、160°+ 需 1.2-1.8mm |
| v4 | Zero 2W 算力风险未评估 | 写入 §6 风险 5 |
| v4 | OTG HUB 命名与用途不符 | 改「OTG 转接线」，备注单相机场景 |
| v5 | 头戴 IMU 被列为解决手部姿态的必需件（逻辑跳跃） | 明确「IMU 只做 ego-motion 补偿」；姿态升级 = 双目三角测量 |
| v6 | 双目与 IMU 归因混淆 | 归因收窄：解决姿态的是双目，不是 IMU |
| v7 | 50-150 条算术错误（100 vs 120/小时）、chunk 时长锚定频率矛盾、筛选口径不清 | 全部修正并加注 |
| v8 | 评审发现 2 处（双目进「明确不加」与 §3.3 矛盾；Square 依据 /tmp 已清） | 双目改「默认不加，可选升级」；依据补 `outputs/mimicgen_gen.log` |

### 5.2 BOM 证据链（本会话最新轮）

| 发现 | 修复 |
|---|---|
| B 档成本 708-945 与表格明细对不齐 | 精确重算为 700-945（A 档 370-555 换相机 +330-390，同端配对） |
| §3.1 写「升级 Orange Pi Zero 3 +50-80 元」偏高 | 改为官方价实证（99-199 元，2GB 已低于 Zero 2W 的 150 元，同价位或更低） |
| 「依据（工程内证据）」列对 B/C/可选行名不副实 | 改列名「依据（证据来源）」，§3.4 说明口径（A 档=工程内配置；升级/可选=文档分析+外部行情） |
| 软件/可选行关键词「—」易误认为漏填 | §3.4 明确「软件/方案项无需采购」 |

### 5.3 twostage_pretrain_finetune_plan.md

| 发现 | 修复 |
|---|---|
| checkpoint 数 37 vs 38 | 修正为 38 个目录（37 编号 + `last` 符号链接） |
| `pretrained_path` 加载整个策略而非仅 backbone | 补充两种机制（a）整策略 strict=False+重置 action 头；（b）独立 resnet18 需自定义代码注入 |
| MimicGen 命令虚构 `--camera/--num-demo` 参数 | 改为 config 驱动（`--config <env_config.json>`） |

---

## 6. 最终交付物清单

| 文件 | 说明 |
|---|---|
| `docs/headcam_data_spec.md`（v8） | 头戴设备规格主文档，224 行，模型训练→数据采集→硬件一条链 |
| `docs/headcam_bom.csv` | 可下单采购清单，15 列证据链版（UTF-8 with BOM，Excel 可开） |
| `docs/twostage_pretrain_finetune_plan.md` | 两段式训练方案评估，209 行 |
| `docs/ANALYSIS_PROCESS.md` | 本文档 |
| `scripts/build_headcam_bom.py` | BOM 生成脚本（可复现再生成） |
| `scripts/collect_datasize_evidence.py` / `collect_twostage_evidence.py` / `check_finetune_support*.py` | 分析取证脚本 |
| `outputs/twostage_evidence.txt` | 两段式评估证据输出 |

**未提交内容（有意排除）**：`outputs/checkpoints/`（28G 训练检查点）、`outputs/train/`、`data/`、`hf_cache/`、以及第三方源码仓库 `lerobot/` `mimicgen/` `robomimic/` `robosuite/`（独立 clone 各有 git 历史）——全部在 `.gitignore` 中。
