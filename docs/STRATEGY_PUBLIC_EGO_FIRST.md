# 策略重评估：公开第一视角数据为主，自研头戴后接（2026-11-01）

> Strategy reassessment: **public egocentric first, self-HMD later.**
> 适用范围：XHEY（第一视角人手 → 手腕动作）这条训练线。只是评估和建议，本次**没有**下载数据、**没有**提交 Kaggle、**没有**做大规模转换。
> 硬约束（Justin）：自己用 iPhone 录的 Record3D / LiDAR 片段**不再作为主训练数据**，最多当过渡期验证；以后的自研头戴（以及创维系设备，国内 20–30 台，东南亚可能再加 50 台）属于硬件采购，和这里的训练数据策略分开处理。

---

## 0. 结论先说（TL;DR）

1. **iPhone 自采数据的定位：降级 + 冻结，不删除。**
   - 3 段 LiDAR 深度片段（实际 QC 只过 2 段）**冻结**成「深度金标 / 目标视角探针」（domain probe），只用于评测和标签噪声分析，**不再进训练**，也**不再作为唯一的胜负标准**。
   - 17 段无深度的 `xhey_rgb`（WiLoR 单目）**冻结**：C17 实测单目 WiLoR 手腕深度只有真实的 1/2.36，平移尺度有系统误差。今后不放进默认对照配比。
   - 「再录 ≥10 段 iPhone Pro 深度片段」（台账 C5）**取消**。
2. **新的主训练数据**：**EgoDex**（Apple Vision Pro 设备追踪的手，带视频、内参、头部位姿）为第一主干；**EgoVerse/Mecka**（MANO 真值关键点 + 头部位姿 + 真实内参）为第二主干，前提是先把许可确认清楚；**HOI4D**（RGB-D + MANO 真值）和 **HOT3D**（动捕级真值）主要用来做**公开的深度/金标评测**，顺带少量训练。**Open-AoE** 规模最大、许可对商用最友好，在新评测协议下重测一次再决定。
3. **现在 C10–C16 测出来的差异不可信。** 这不是因为公开数据没用，而是评测本身分不出差别：所有配比之间只差 0.005–0.02 cm（ADE 约 12.5 cm，不到 0.2%）。置信区间是在 3 段片段的**步长 1、互相重叠**的窗口上做独立同分布 bootstrap 算出来的，明显偏窄（见 §2.2）。所以 C14「done_win」（比对照好 0.012 cm）、C11/C12/C13/C16 的「没超过对照」，都**不能**当成数据集好坏的结论。
4. **训练流程本身也挡住了公开数据的作用**：`train_xhey.py` 里 `iphone` 必选；公开数据**不带画面**（`has_image=0`），只训练手部状态分支；预训练完再在 2 段 iPhone 片段上微调，用同一批 iPhone 片段做早停，再选 shrink 系数，预测结果被收向「手不动」。公开数据的主要价值（大量视频 + 设备追踪的手）基本没用上。
5. **找到一个会让结果失真的 bug**：自动迭代内核（`xhey_auto_iter.py` 的 `KERNEL_TEMPLATE`）既不挂 `xhey-xphi-cam` 数据集，也不传 `--xphi`，但提议器从 C14 的最佳配置出发，会一直提出 `iphone+xphi+…` 的配比。`train_xhey.py` 的 `try_external` 找不到目录时**静默跳过**，所以第 3、4、5 轮里带 xPhi 的候选实际上都**没有用到 xPhi**。证据：`iter004_c2`（xPhi，噪声 0.2，lr 1.0）和 `iter003_c1`（没有 xPhi，同样超参）的 overall / moving ADE 完全相同（9.752460214243373 / 12.535582195742764）。
6. **保留的东西**：统一 episode 格式、各数据集转换器、`common/frames.py` 坐标规范化和 `sanity_check`、分段 QC 成绩单、`action_valid` 掩码、LeRobot v3 导出、双目管线（留给头戴）、「同一次运行带对照」的台账纪律、默认 `--qc-filter`。这些已经能用，换数据源时直接沿用。
7. **停止投入的东西**：针对 iPhone 3 段留一法的自动调参、更多 iPhone 采集、xPhi / DI / zeno / Hubdata 扩量、Flora（C18：非商用许可，T4 跑不动）、Dyn-HaMR 继续修（C17）。
8. **本 PR 已做的最小改动**：`docs/xhey/queue.json` 加 `launch_paused`，`xhey_auto_iter.py step` 看到它就只收结果、不推新内核（取代人工记得停 cron）；把几份文档里「iPhone = 主数据 / 唯一评测」的说法改成指向本文；修掉一条早已和 `queue.json` 对不上的测试断言。

---

## 1. 现状架构：哪些稳，哪些脆

### 1.1 数据流（现在实际怎么跑）

```
公开数据集 ──convert_*.py──► 统一 episode JSON ──qc_scorecard.py──► 成绩单（整段 / 分段）
                                   │                                        │
iPhone mcap ─► mcap_adapter ─► rgbd_pipeline（WiLoR + LiDAR 放手腕）          │
                                   │                                        ▼
                       common/frames.canonicalize_for_xhey（统一到相机光学系，米）
                                   │
                                   ▼
            train_xhey.py：预训练（公开数据 + xhey_rgb，无画面） → 微调（2 段 iPhone，有画面）
                                   │
                                   ▼
            评测：3 段 iPhone 深度片段留一法，16 步 ADE / 「手在动」子集，对比 zero / 对照
                                   │
                                   ▼
            xhey_auto_iter.py → Kaggle → experiments/ledger.jsonl → docs/xhey/ledger.csv
```

### 1.2 已经能用的（可以放心沿用）

| 模块 | 代码 | 证据 |
|---|---|---|
| 统一 episode 格式 + 10 多个转换器 | `scripts/egodata/schema.py`、`scripts/convert_{egodex,egoverse,hoi4d,openaoe,stera,di,hf_iphone}.py`、`scripts/wilor_video_to_episodes.py` | C11–C16 都通过同一格式进训练 |
| 坐标规范化 | `scripts/common/frames.py`（`camera` / `camera_yup` / `world_yup` / `world`，`sanity_check`） | HOI4D 重投影中位误差 0.76 px；EgoDex 3243/3243、EgoVerse 前 15 段投影检查通过（`queue.json`） |
| QC 成绩单（整段 + 分段） | `scripts/qc_scorecard.py`、`scripts/egodata/qc.py` | 能挡住 xPhi 的高抖动（全量 jitter 中位 9.4 cm → 保留段 2.4 cm，C18） |
| 缺测动作掩码 | `scripts/egodata/action_valid.py`、`lerobot_export.py`、`ego_act_train.py` | 修过 ACT 掩码在 lerobot 0.6.1 从未生效的 bug（`docs/egodex_act_results.md`） |
| 双目手部管线 | `scripts/headcam/stereo_pipeline.py`、`run_stereo_pipeline.py` | HOT3D Quest3 手腕中位 1.13 cm、p90 2.38 cm（`docs/PROGRESS.md` §4.8） |
| RGB-D 手部管线 | `scripts/headcam/rgbd_pipeline.py`、`scripts/iphone/*` | DexYCB 实测手腕中位 1.41 cm（单目 2.75 cm） |
| 实验纪律 | `docs/xhey/queue.json`、`experiments/ledger.jsonl`、同次对照臂 | C11–C16 每次都有同代码对照，结论按「是否超过对照」判 |

### 1.3 脆弱 / 已经拖后腿的

| 问题 | 位置 | 影响 |
|---|---|---|
| **评测太窄**：3 段 iPhone 深度片段留一法；其中 `09-05_17-16` QC 不过（抖动 4.07 cm、左右手互换 134 次），C17 发现它现行的 LiDAR 标签文件是 0 字节 | `train_xhey.py main()`、`docs/xhey/c17_label_methods.md` | 实际可信的测试片段只有 2 段 |
| **置信区间过窄**：按 16 步、步长 1 的重叠窗口做独立同分布 bootstrap | `hot3d_train_compare.bootstrap_ci`、`train_xhey.summarize_results` | 「95% 区间不跨 0」几乎总能成立，见 §2.2 |
| **基线换了三次**：世界系 11.03 → 相机系 9.779 → QC 两段 7.227 | `docs/xhey_ledger.md` | 跨批次的数字不能放在一起比 |
| **`iphone` 必选** | `train_mix()` 始终用 iPhone 训练片段 | 没法做「只用公开数据训练」 |
| **公开数据不带画面** | `build_external()` → `finish(d, None)` | 公开视频完全没用，图像分支只见过 iPhone + xhey_rgb |
| **微调吃掉预训练**：同一批 iPhone 片段早停 + shrink 收缩 | `XHEY.fit()` | 所有臂都收敛到「比不动好约 1%」 |
| **缺数据静默跳过** | `try_external()` | 导致 §0 第 5 条的 xPhi bug |
| **对照里混着有尺度误差的标签**：`xhey_rgb` 是单目 WiLoR，深度约为 LiDAR 的 1/2.36 | C17 | 每一个对照臂都带着这个偏差 |
| **公开数据只用了很小一部分**（见 §3 表格「仓库里已用」一列） | — | 「公开数据不帮忙」的结论是在 0.5%–5% 的规模上得出的 |
| **许可**：WiLoR（CC BY-NC-ND）、MANO（非商用）贯穿打标链路；多数公开集是 NC | 见 §3、§6 | 要商用的话，需要单独走一条合规路线 |

---

## 2. 为什么 C 系列测不出差别

### 2.1 各配比之间的差距

相机系、同一评测集（zero overall 9.779，「手在动」zero 12.683）：

| 实验 | 配比 | 手在动 − zero (cm) | 和对照比 |
|---|---|---:|---:|
| C10ctl / C11ctl / C13ctl / C14ctl / C16ctl | iphone+xhey_rgb+aug | −0.137 | — |
| C14 | +xPhi | −0.149 | 好 0.012 |
| C11 | +DI | −0.132 | 差 0.005 |
| C16 | +EgoVerse | −0.129 | 差 0.008 |
| C13 | +EgoDex | −0.126 | 差 0.011 |
| C12 | +HOI4D | −0.064 | 差 0.073 |

除了 C12，其余差距都在 ±0.012 cm 以内，大约是 ADE 的 0.1%。

### 2.2 置信区间偏窄的原因（估算，不是重新跑出来的数）

- 窗口长 16 步、步长 1：相邻两个窗口有 15/16 的目标是同一段手腕轨迹，彼此高度相关。只算重叠这一项，有效样本数最多是名义样本数的 1/16，区间半宽至少要放大约 √16 = 4 倍。
- 测试片段只有 2–3 段，片段之间的差异（录制场景、动作快慢）根本没有进入 bootstrap。
- 现在报告的区间半宽大约 0.01 cm（例如 C14 [−0.159, −0.139]）。只按重叠校正，半宽也会到约 0.04 cm，已经盖过 C14 和对照之间 0.012 cm 的差距。再把片段之间的方差算进去还会更宽。
- 结论：现有台账里的 `done_win` / `done_worse_vs_ctl` 只能当「方向上没有明显信号」，不能当「这个数据集有用 / 没用」。

### 2.3 三个结构性原因

1. **任务太难、信号太弱**：未来约 1 秒的手腕位移中位只有 5.5–6.2 cm，逐帧跳变噪声 1.2–1.4 cm（C15）。在 2 段片段上，模型最稳的策略就是接近不动。
2. **公开数据只喂了手部状态**：图像特征在公开数据上全是 0，`has_image=0`。EgoDex、HOI4D、Stera、Open-AoE 都有视频，但没有用。
3. **最后一步总是在 iPhone 上微调、在 iPhone 上选模型**：预训练学到的东西，会被 3000 步微调和 shrink 选择洗掉一大半。

---

## 3. iPhone 自采数据：留什么、降什么、冻结什么

| 资产 | 现状 | 建议 | 原因 |
|---|---|---|---|
| 3 段 LiDAR 深度片段（`run_s2`，约 4.66 万帧 @15fps） | 唯一训练+评测主线 | **冻结为评测探针**：记下文件哈希、QC 成绩单和标签版本，以后不再改。只报 zero-shot（直接用公开数据训的模型测）和可选的 few-shot。**不进训练、不作为胜负门槛** | 这是仓库里唯一一份「目标视角 + 公制深度」的手腕数据，丢了就没法量化域差。但 2 段的统计功效撑不起主评测 |
| `09-05_17-16`（QC 不过，现行 LiDAR 标签 0 字节） | 在 3 段留一法里 | **剔除出探针**，或者重打标签后单独标注 | 已知坏数据 |
| 17 段 `xhey_rgb`（WiLoR 单目） | 每个对照臂都有 | **冻结**，只留作消融，不放进默认配比 | 单目深度约为真实的 1/2.36（C17），平移尺度有系统误差 |
| iPhone 管线代码（`scripts/iphone/*`、`rgbd_pipeline.py`、`--iphone`） | 已能跑 | **保留，保持测试通过** | 以后带深度的头戴（或 iPhone 当过渡采集器）还能直接用；C15/C17 的标签噪声分析也靠它 |
| 「再录 ≥10 段」（C5） | 候选 | **取消** | 和硬约束冲突，而且再多 10 段也解决不了评测设计的问题 |
| `docs/iphone_arxiv_plan.md`（A→B→C 论文路线） | 以 iPhone 为主线 | **冻结**，文首加说明 | B 节的质量对比表仍有参考价值 |

**iPhone 数据还剩下的价值**（为什么不直接删）：
- 唯一的「公制深度 + 目标头戴视角」探针，可以检查公开数据训出来的模型能不能迁移到自己的设备视角；
- 给 C15/C17 这类标签噪声研究提供 LiDAR 参照（单目尺度 2.0–2.6 倍误差就是这样测出来的）；
- 头戴到货之前，它是唯一能演练「自采 → QC → 导出 → 评测」全链路的真实数据。

---

## 4. 新的主数据栈（按可行性 × 许可 × 格式契合 × 预期收益排序）

「仓库里已用」一列是本仓库实际转换、训练过的量。「全量」来自各数据集文档或仓库内记录；仓库里没写的标 **待核实**。

| 优先级 | 数据集 | 手部标签 | 视频 / 深度 / 头部位姿 | 许可 | 仓库代码 | 仓库里已用 / 全量 | 定位 |
|---:|---|---|---|---|---|---|---|
| **1** | **EgoDex**（Apple） | Vision Pro 设备追踪的 ARKit 关节（不是伪标签） | 有 mp4 / 无深度 / 每帧相机位姿 + 内参 | CC BY-NC-ND（研究用，不能分发改写版） | `convert_egodex.py`、`egodata/egodex.py`、`egodex_act_eval.py`、固定划分 `docs/egodex_act/split.json`；Kaggle `xhey-egodex-cam`（私有） | test 部分 3243 条、7.65 h（C13 只取了 4 h）/ 全量 829 h（据 `2607.24744v1.pdf` 综述表） | **主训练 + 主评测之一** |
| **2** | **EgoVerse / Mecka** | MANO 真值关键点 | 有视频 / 无深度 / 头部位姿 + 真实内参；**原生 10 Hz** | freeform 行 license 字段为空（旗舰行 CC BY-SA 4.0）→ **扩量前必须先确认** | `convert_egoverse.py`（已处理 10 Hz 插值陷阱、`head_const`）；Kaggle `xhey-egoverse-cam` | 401 段、保留 3.99 h / mecka 目录 828 h | **主训练之二**（许可确认后） |
| **3** | **HOI4D** | MANO 真值（只标交互的那只手，多为右手） | RGB-D / 有深度 / 相机内参 | CC BY-NC 4.0 | `convert_hoi4d.py`；Kaggle `xhey-hoi4d-pretrain` | 1487 段、8.26 h / 发布的 4000 段（待核实小时数） | **公开深度金标评测**（替代 iPhone 的「有深度」角色）+ 少量训练 |
| **4** | **HOT3D**（Meta，Quest3 / Aria） | 动捕级真值手 + 物体 6DoF | 双目 / 无深度图 / 标定位姿 | HOT3D 许可（研究）+ MANO | `headcam/hot3d_adapter.py`、`eval_hot3d_stereo.py`、`hot3d_train_compare.py` | 24 段（训练对比）/ 13.9 h（综述表） | **金标评测 + 双目管线基准**（和自研头戴最像） |
| **5** | **Open-AoE**（OpenAoE-2000h） | MANO 重建 | 有视频 / 无深度 / SLAM 相机轨迹 | Open-AoE 许可**允许商用**（需署名）；MANO 派生部分非商用 | `convert_openaoe.py`（world / camera 两种） | nano 2.5 h + tiny 切片 13.9 h / 2000 h | **在新评测下重测**；以后要商用时是首选候选 |
| 6 | Stera-10M | WiLoR + LiDAR 锚定 | 有视频 / 有深度 / ARKit 重力世界系 | CC BY-NC 4.0，gated | `stera_fetch.py`、`convert_stera.py` | 转换 19.66 h、QC 保留 7.14 h / 200 h | 备选。本质还是「iPhone 头戴 + WiLoR」，和硬约束的方向不一致 |
| 7 | xPhi-Ego-100 | 单目 WiLoR 伪标签，内参是**假设的** HFOV 90° | 有视频 / 无 / 无 | **Apache-2.0**（最宽松） | `kaggle/xphi_wilor`、`wilor_video_to_episodes.py` | 5.03 h，QC 保留 0.32 h | **冻结**。动作标签太差；以后可能只拿视频做视觉预训练 |
| 8 | DynamicIntelligence | 每只手 3 个点 | — | CC BY-NC 4.0 | `convert_di.py` | 25 段「手在动」、约 4.3 分钟 | **停** |
| 9 | zeno / Hubdata（HF 手机数据） | MediaPipe+LiDAR 或 WiLoR | — | CC BY-NC 4.0 / other | `convert_hf_iphone.py` | 小样本 | **停** |
| — | EgoX-Data-6M | — | — | — | **仓库里没有**（只出现过 EgoXperience，C17 核实为代码和权重都未发布） | — | 本轮不纳入；要评估时先核查许可和格式。仓库里的综述 PDF 列了 Xperience-10M（10,000 h），和 EgoXperience 是否同源**未核实** |

**为什么 EgoDex 排第一**：标签是设备追踪的，不是 WiLoR 伪标签；带视频、内参、每帧相机位姿；仓库里转换、QC、LeRobot 导出、ACT 评测和固定划分都已经有了；量级是综述里同类数据集中最大的之一（829 h），而仓库只用了 test 部分。缺点：许可是 NC-ND，只能研究用；没有深度；Vision Pro 的视角和 FOV 跟自研头戴不同。

**为什么 HOI4D / HOT3D 主要当评测**：它们提供了 iPhone 片段原本承担的「有深度 / 有金标」角色，而且是公开的、可复现的、片段数多得多。拿它们当评测，就能把 iPhone 片段从「唯一裁判」降成「域探针」。

---

## 5. 新的评测协议（Eval v2，提议）

| 评测集 | 来源 | 用途 | 是否作为胜负门槛 |
|---|---|---|---|
| E1 EgoDex 留出 | 按 episode **和**按任务各留一份（沿用 `docs/egodex_act/split.json` 的做法，扩到更多任务） | 主指标 | 是 |
| E2 HOI4D 留出 | 按序列留出 | 有深度的公开金标 | 是 |
| E3 HOT3D | 固定片段 | 动捕金标，量标签噪声底线 | 参考 |
| E4 EgoVerse 留出 | 按任务 / 场景留出（排除 `head_const`） | 跨数据集泛化 | 许可确认后作为门槛 |
| E5 iPhone 探针 | 冻结的 2 段 QC 合格深度片段 | 目标视角 zero-shot | **只报告，不作门槛** |

统计口径：
- bootstrap 按 **episode / 片段整组重采样**（cluster bootstrap），不再按重叠窗口重采样；报告有效样本数。
- 报告「可检测的最小差距」（MDE）。小于 MDE 的差距一律记成「不确定」。
- 胜出规则：在 ≥2 个门槛评测上同时超过 zero **和**同次对照，且整组 bootstrap 区间不跨 0；E5 不能变差超过一个事先写好的容差。
- 每个评测集配一份冻结的划分 JSON 进仓库（只放 id，不放数据）。

---

## 6. 保留 vs 停止投入

**保留并继续用（换数据源时原样复用）**
- 统一 episode 格式、全部转换器、`common/frames.py`（尤其 `sanity_check`，跨数据集坐标出错最容易翻车）
- `qc_scorecard.py` 整段 + 分段成绩单，新实验默认带 `--qc-filter`
- `action_valid` 掩码、LeRobot v3 导出、ACT 训练入口
- 同次对照臂、台账只追加不改、`queue.json` 驱动的固定队列
- 双目管线和 HOT3D 适配器（给自研头戴用）
- iPhone / RGB-D 管线代码（冻结，但保持测试通过）

**停止 / 暂停**
- 在 iPhone 3 段留一法上继续自动调参（第 5 轮之后不再推新的；本 PR 用 `launch_paused` 卡住）
- 再录 iPhone 片段用于训练（C5）
- xPhi / DI / zeno / Hubdata 扩量
- C15（累计位移损失权重）：假设本身值得测，但在旧评测上测出来也读不出结论 → 等 Eval v2 再跑
- C18 Flora（NVIDIA 非商用许可 + T4 一个窗口都跑不完）、C17 Dyn-HaMR 继续修
- 把 0.01 cm 级别的差距写成 `done_win` / `done_worse`

---

## 7. 接下来的执行顺序（对应你说的 2–4 周窗口）

按依赖排序。每一阶段有一个「过了才进下一步」的门槛，而不是按天排期。

### 阶段 0：先把评测和代码修对（不需要 GPU）

| # | 改什么 | 代码位置 |
|---|---|---|
| 0.1 | `iphone` 改成可选：加 `--train-sources` / `--eval-sets`，评测集从冻结的划分 JSON 读 | `scripts/train_xhey.py`（`train_mix`、`main`） |
| 0.2 | 按 episode 整组 bootstrap，报告有效样本数和 MDE | `scripts/hot3d_train_compare.py::bootstrap_ci` 或在 `train_xhey.summarize_results` 新加一个函数 |
| 0.3 | 配比里写了某个数据源但目录不存在时**直接报错**（至少在 Kaggle / 自动迭代模式下），不再静默跳过 | `scripts/train_xhey.py::try_external` |
| 0.4 | 修自动迭代的 xPhi bug：内核模板按配比挂数据集、传 `--xphi` 等参数；或者把提议器的配比限制在 `SPACE["mix"]` 里 | `scripts/xhey_auto_iter.py`（`KERNEL_TEMPLATE`、`DATASETS`、`propose`） |
| 0.5 | 第 5 轮（`auto_state.json` 里那 3 个 xPhi 候选）入账时注明「xPhi 未加载」；第 3、4 轮里带 xPhi 的行同样注明 | `experiments/ledger.jsonl`（只追加备注行，不改历史数字） |
| 0.6 | 公开数据可选带画面：有视频的数据源（EgoDex 同名 mp4、HOI4D、Stera `rgb.mp4`、Open-AoE）在 `build_external` 里算 ResNet 特征，注意和 `subsample_episode` 的帧对齐 | `scripts/train_xhey.py::build_external`、`image_features` |
| 0.7 | 写出 E1–E5 的冻结划分 JSON | `docs/xhey/eval_v2/*.json`（新建） |

**门槛**：在已有的 Kaggle 数据集上做一次 CPU 冒烟（小步数），能输出 E1–E5 的整组 bootstrap 区间。

### 阶段 1：只用公开数据训练（用 Kaggle 上已经有的私有数据集，不新下载）

| 实验 | 训练 | 评测 | 要回答的问题 |
|---|---|---|---|
| P1 | EgoDex test（`xhey-egodex-cam`），只用手部状态 | E1 + E5 | 新协议下，公开数据训出来的模型能不能稳定超过 zero |
| P2 | P1 + 画面特征（阶段 0.6） | E1 + E5 | 公开视频对图像分支有没有用 |
| P3 | EgoDex + EgoVerse（`xhey-egoverse-cam`） | E1 + E4 + E5 | 两个主干叠加是否更好；交叉评测矩阵（A 训 B 测） |
| P4 | P3 + HOI4D（`xhey-hoi4d-pretrain`） | E1 + E2 + E5 | 带深度的金标数据加进训练后，在深度评测上有没有改善 |
| P5 | P1 的数据量曲线：1 h / 2 h / 4 h / 全部 test | E1 | 现有规模下还在涨，还是已经饱和；决定要不要下 train 部分 |
| P6 | 同一个 P3 模型在 E5 上 zero-shot，以及可选的 2 段 few-shot | E5 | 公开 → 目标视角的域差有多大 |

**门槛**：P1 或 P3 在 E1 上以整组区间超过 zero，并且 P5 的曲线还没有饱和，才进入阶段 2。

### 阶段 2：扩量（需要 Justin 拍板：许可、存储、GPU）

- EgoDex train 部分分批下载 → 转换 → QC → 接到 P5 曲线上。**下载前先确认存储和 Kaggle 数据集大小上限**（本 PR 没有动）。
- EgoVerse：先确认 freeform 行的许可，再从 4 h 扩到几十小时。
- Open-AoE：在 E1/E2 上用相机系重测一次 nano/tiny；如果有正信号，它是唯一一个对商用相对友好的大规模来源。
- C15（累计位移损失）放到这里，用 Eval v2 重测。

### 阶段 3：给自研头戴铺路（和硬件采购并行，不挡上面）

- 头戴的数据规格（`docs/headcam_data_spec.md` §7.7）里加一条：同时出 Eval v2 格式的冻结划分，第一批头戴数据先当 E6 评测，不急着训练。
- 拿 HOT3D 双目继续当头戴管线的回归测试。
- 头戴到货后的第一个问题是「E5（iPhone）和 E6（头戴）的域差，哪个更接近公开数据」，不是「微调多少步」。

---

## 8. 过早放弃 iPhone 数据的风险

| 风险 | 后果 | 缓解 |
|---|---|---|
| 失去唯一的「公制深度 + 目标视角」评测 | 公开数据训出来的模型到了自己的设备上不行，直到头戴到货才发现 | 冻结 2 段作为 E5，每次都报 zero-shot；E2（HOI4D）、E3（HOT3D）补上公开的深度 / 金标 |
| 失去标签噪声的参照 | 没法再量单目 WiLoR 的尺度误差（C17 的 2.36 倍就是靠 LiDAR 测的） | 保留 C15/C17 脚本和冻结片段 |
| 头戴延期 | 很长一段时间完全没有自有视角的数据 | iPhone 管线保持测试通过；真到了需要的时候，只录**评测**片段，不录训练片段 |
| 公开数据的视角 / FOV 和头戴差很多（Vision Pro、Aria、Quest、手持） | 迁移效果打折 | P6 量化域差；头戴规格里记好 FOV 和安装角度，必要时按 FOV 做裁剪增强 |
| 坐标约定在不同数据集之间出错 | 一个翻转就足以让结果反向 | 所有转换都走 `common/frames.py`，`sanity_check` 失败就拒绝进训练 |
| **许可** | 大部分公开集是 NC / NC-ND；WiLoR（CC BY-NC-ND）和 MANO（非商用）贯穿整条打标链路，**连 iPhone 自采数据的标签也是用 NC 模型打的** | 研究路线和商用路线分开记账；商用候选看 Open-AoE（去掉 MANO 派生部分）、xPhi（Apache-2.0，但标签要重做）、EgoVerse 旗舰（CC BY-SA）；商用打标器需要单独选型，交法务确认 |

---

## 9. 做 / 不做 清单

**接下来做**
1. 先修评测：`iphone` 可选、按 episode 整组 bootstrap、缺数据报错、修自动迭代的 xPhi bug（阶段 0）。
2. 冻结 iPhone 2 段深度片段为 E5 探针；剔除 `09-05_17-16`。
3. 用 Kaggle 上已有的 `xhey-egodex-cam`、`xhey-egoverse-cam`、`xhey-hoi4d-pretrain` 跑 P1–P6，**不新下载**。
4. 公开数据带画面训练（阶段 0.6 → P2）。
5. 确认 EgoVerse freeform 行的许可，以及 EgoDex train 部分的存储方案，再决定阶段 2。
6. 研究 / 商用两条许可路线分开记账。

**不要做**
1. 不要再录 iPhone 片段来训练。
2. 不要在 3 段 iPhone 留一法上继续自动调参，也不要按 0.01 cm 级别的差距下结论。
3. 不要在新评测协议跑出正信号之前下载 TB 级数据。
4. 不要把单目 WiLoR 的绝对深度当成公制标签（没有 LiDAR / 双目锚定时）。
5. 不要再扩 xPhi / DI / zeno / Hubdata，不要再修 Flora / Dyn-HaMR。
6. 不要删 iPhone 管线代码和冻结片段。

---

## 附：本次核查用到的关键证据

- 自动迭代 xPhi bug：`experiments/ledger.jsonl` 里 `iter004_c2`（`iphone+xphi+xhey_rgb+aug`，aug_noise 0.2，lr_scale 1.0）和 `iter003_c1`（`iphone+xhey_rgb+aug`，同样超参）的 `overall_ade_cm` / `moving_ade_cm` 完全相同；`scripts/xhey_auto_iter.py` 的 `DATASETS` 只有 `xhey-openaoe-pretrain` 和 `xhey-iphone-sessions`，`KERNEL_TEMPLATE` 不传 `--xphi`；`train_xhey.try_external` 找不到目录时打印 `skip` 并返回 `None`。专用内核 `kaggle/rsi_c14` 传了 `--xphi`，所以 C14 本身确实用到了 xPhi。
- 公开数据无画面：`scripts/train_xhey.py::build_external` 最后是 `finish(d, None)`，`f_img` 全 0、`f_has` = 0。
- `iphone` 必选：`train_mix` 无论配比如何，都先用 `iph` 的训练片段构建 X / Y，微调永远在 iPhone 上做。
- 置信区间：`hot3d_train_compare.bootstrap_ci` 对每个窗口的误差做独立同分布重采样；窗口来自 `egodex_act_eval.chunk_starts(..., stride=1)`。
- 单目尺度：`docs/xhey/c17_label_methods.md`，WiLoR 单目手腕 z 中位比 LiDAR 小 68.8 cm，尺度系数中位 2.371（2.015–2.571）。
- 测试现状：`tests/test_xhey_auto_iter.py::CommittedState::test_c9_done_worse_and_c9b_already_running` 在 main 上就是失败的（断言 C16 还是 `ready`，而 `queue.json` 早已是 `done_worse_vs_ctl`），本 PR 顺手对齐。
