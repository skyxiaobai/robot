# 开放数据集流水线差距分析

> 日期：2026-10-09。对照一条完整链路：
> **开放数据集下载 → 转成统一格式 → QC / 产出率 → 覆盖统计 → 四级标注 → LeRobot 训练 → 缩放律评估**。
> 前提：还没有自有头戴采集设备，只能先用带手部位姿的开放第一视角数据。
> 规格见 `docs/headcam_data_spec.md` v10 §7–§8。

## 0. 结论

仓库里原来只有「仿真 benchmark 怎么训练」和「将来头戴设备该采什么」。中间从原始第一视角数据到可训练样本的质检、覆盖和标注是空的。这次补上统一 episode、EgoDex 适配器、产出率、覆盖报告和四级标注校验，并且用合成 HDF5 测过。真实 EgoDex 测试集的下载地址和单条 HDF5 结构已经核对过，16GB 的 zip 不进仓库。

在完整 EgoDex 测试集（3,243 条、约 82.6 万帧）上跑过一轮之后，修了几处会让曲线和产出率失真的问题：没有 `confidences` 组时不再当成置信度 0；四元数按相邻帧保持同一半球，不再每帧强制 `w >= 0`；线性 BC 用固定的 episode 验证集和收敛的岭回归，并记下「保持当前手腕」的基线。动作改成多步手腕增量。导出按条释放 JSON，真实 mp4 默认可缩到 224。ACT 的 `lerobot-train` 命令在 `examples/ego_act_train.yaml` 和 Colab 后半段，CPU 冒烟不安装 torch。

第二次全量重跑之后又收紧了三处：`val_loss` 改用科学计数法，小数据量按 episode 轮转抽满预算，损失按训练集逐维标准化并分开报告平移和旋转。覆盖统计逐条累加，并认 `tablecloth:` 和 `lavendar`/`lavender`。

第三次全量重跑说明，按每一档自己的训练子集做标准化会让基线从 2.77 变到 1.09，缩放拟合被抬高。现在均值和标准差只从全部训练池算一次，验证集和基线用同一套。轮转前先在每条 episode 内打乱动作块，小预算不再只看见片段开头，更小预算仍是前缀。缩放律同时读基线、平移、旋转，并对 val/基线比值拟合。

第四次全量重跑里，测量已经稳定：基线固定在 1.088572，损失随数据量下降，但对数直线从大约 6.4 万个样本起就不再下降（全量只比保持不动好约 6%）。仓库外的 MLP 也在 25.6 万之后停住。瓶颈是输入只有手部状态。缩放报告现在同时给饱和幂律 `L = L∞ + A·N^(-α)`。特征和动作都用全训练池的统计量，岭回归系数在训练内部划分上选择。多种子会写成均值和标准差。ACT 的图像缩放在 `notebooks/egodex_act_scaling_colab.ipynb`，lerobot 0.6.1 的 ACT 仍不吃语言。

还没接上的是：在这 3,243 条上真正把 ACT 训完，以及给 EgoDex 补上按时间切开的子任务和分手指令。像素模糊、HOT3D 仍不在这条链路上。笼统动词 `use` 仍然落在 other。

`scripts/ego_visualize.py` 可以把统一 episode 画出来：手骨架叠加、通过/拒绝对照、头和手腕的世界系轨迹。投影和 QC 相同。Colab 合成样本单元会写 PNG 和短 MP4。转换现在会进入符号链接的任务目录；以前 `Path.rglob` 会把这些目录整段跳过。

自有头戴会话（规格 §7.7，设备还没有）现在可以收成同一份 JSON。`scripts/headcam/hand_pose.py` 接 HaMeR、WiLoR 和 CPU 上的 MediaPipe；左右目 2D 用标定三角化，修正单目尺度。相机位姿读 TUM 文件，没有文件时是单位阵。`scripts/convert_headcam.py` 写出的 episode 直接进现有的 QC、覆盖、导出和可视化。笔记本是 `notebooks/headcam_hand_pose_colab.ipynb`。

自有设备仍缺这些：

- SLAM：仓库不跑双目加 IMU 的里程计，只读已经算好的 TUM。没有轨迹时坐标留在相机系
- 标定计算：只读取 Kalibr / OpenCV yaml，不在这里求内参、外参或 IMU 到相机的外参
- 自动语言标注：按时间切开的子任务和分手指令。转换器仍然留空
- HaMeR / WiLoR 的权重：MANO 是非商业许可，不入库。没装权重时测试走合成双目，MediaPipe 测试在未安装时跳过

## 1. 逐段对照

| 阶段 | 现在有什么 | 还缺什么 | 优先级 |
|---|---|---|---|
| 下载开放数据 | 已核对 EgoDex `test.zip`（见 §2）。没有下载脚本，避免把 16GB 拉进 CI | 需要时用 README 里的 `curl`。HOT3D clips、`lerobot/umi_cup_in_the_wild` 没有适配器 | P2：再加一个数据集 |
| 转成统一格式 | `scripts/egodata/schema.py` + `egodex.py`，`scripts/convert_egodex.py`。关节矩阵整段读取，`--workers` 可多进程。没有 confidences 时置信度为未知。四元数在片段内连续。任务目录是符号链接时也会进入。`ego_to_lerobot.py` 写成 LeRobot v3.0，真实 mp4 可缩放到 224。自有会话用 `scripts/convert_headcam.py`：HaMeR / WiLoR / MediaPipe、双目三角化、TUM 位姿，输出同一份 JSON | 开放数据只接了 EgoDex。没有源 mp4 时视频仍是 16×16 占位。头戴侧不跑 SLAM，也不做标定求解 | P2：再加一个数据集；SLAM 仍在设备侧 |
| QC / 产出率 | `scripts/egodata/qc.py`，`scripts/egodata_qc.py`。手出画、视线飘移、相机角速度（模糊代理）、长时间静止。产出率 = 通过片段的帧数 / 原始帧数。`ego_visualize.py qc_compare` 用同一套逐帧标记画对照 | 模糊还没看像素（拉普拉斯）。坏段只能整段丢掉，不能把好的子段切出来留用 | P1：有 mp4 时加像素模糊；子段裁剪 |
| 可视化 | `scripts/ego_visualize.py`：`overlay`（21 点骨架、手腕轨迹、任务文字，PNG/MP4）、`qc_compare`、`traj3d`。投影与 QC 相同。开放数据笔记本里有合成样本单元 | 不读像素，也不做手部估计。生成的图不入库 | P2：抽样时看几条真实 mp4 |
| 覆盖统计 | `scripts/egodata/coverage.py`。桌布 `tablecloth:`、坐姿和背景都留在环境名里，`lavendar` 收成 `lavender`。物体类别在出报告时按当前词表重算，脚本逐条累加、不把全部 JSON 留在内存里。动作词表含组装、滚动、推动、涂色等，take/gather、stock/add 收到已有类别 | 词表仍是手写的，笼统的 `use` 仍算 other。没有「该采多少才算补上」的数量目标 | P2：按目标小时数做配额 |
| 四级标注 | 规格 §8.1，样例 `examples/hierarchy_annotation.json`，校验 `scripts/validate_hierarchy.py`。EgoDex 转换只填 ENVIRONMENT 和 TASK | EgoDex 没有时间分段 SUBTASK，也没有分手 INSTRUCTION。不能把整段描述切成假时间段 | **P1：标注**（人工或模型），不要在转换器里编造 |
| LeRobot 训练 | pusht ACT 笔记本仍在（image+state[2] → action[2]）。手部线性 BC 用全训练池标准化，并可扫描岭回归系数、多种子。ACT 配方在 `examples/ego_act_train.yaml` 和 `notebooks/egodex_act_scaling_colab.ipynb`：224 图像、16 步增量、固定验证 episode、三到四档 | 还没有在完整测试集上把这几档 ACT 训完。lerobot 0.6.1 的 ACT 没有语言编码器。pusht 的 action[2] 没有改 | **P1：在 Colab GPU 上跑完 ACT 四档** |
| 缩放律 | `scripts/scaling_law.py`。对数直线和饱和幂律都报告，R² 更高的作为默认曲线。同一数据量的多种子画均值和标准差。靠得近的点标签会错开 | 还没有真实 EgoDex 的 ACT 曲线。状态-only 的线性模型在 6.4 万之后饱和，不要用对数斜率外推 | P1：用 ACT 笔记本的日志再拟合 |

优先级的意思：P1 是下一条数据链路还没通的地方；P2 是数据集种类、配额和自有设备 SLAM，不挡住现在用开放数据做质检。

## 2. 数据集怎么选的

优先 EgoDex 测试集（[apple/ml-egodex](https://github.com/apple/ml-egodex)），因为位姿是录制时的 ARKit 世界系 SE(3)，和规格 §7 的「世界系手腕」同一类量，而且测试包可以公开下载。

2026-10-09 核对结果：

- `https://ml-site.cdn-apple.com/datasets/egodex/test.zip` 返回 HTTP 200，`content-length` = 17,304,529,397 字节（约 16.1GB），`accept-ranges: bytes`。
- ZIP64 中央目录有 6598 条：`test/<任务>/<序号>.hdf5` 与同名 `.mp4` 各 3243 个，111 个任务目录。
- 用 Range 取出最小的一条 `test/open_close_insert_remove_case/8.hdf5`（压缩后约 49KB）。结构与上游 README 一致：`camera/intrinsic` 为 3×3，`transforms/<关节>` 为 N×4×4（这条 N=15），`confidences/<关节>` 为 N。属性里有 `environment`、`task`、`llm_description`、`llm_description2`、`which_llm_description=2`、`llm_objects`、`llm_verbs`。
- 这条的内参主点是 (960, 540)。用相机位姿的逆、按 OpenCV（+Z 向前、+Y 向下）投影，食指尖落在 1920×1080 里面，手腕靠近画面下沿。QC 的出画判断用的就是这个约定。
- 许可是 CC BY-NC-ND。工具只在本地读，不把数据或改写后的数据提交进仓库。

没选另外两个的原因：

- HOT3D clips 也能下，但是 tar 里的手是 UmeTrack/MANO，还要另下 MANO 模型才有网格。比 EgoDex 多一道授权。
- `lerobot/umi_cup_in_the_wild` 已经是 LeRobot 格式，适合夹爪遥操作，没有人手 21 关节，盖不住规格 §7。

## 3. 统一 episode 里有什么

一条 JSON（`schema_version` 1.0）包含：

- 来源、帧率、`coordinate_frame=arkit_world`（集内静止，集与集的原点不必相同）
- 相机内参、每帧 4×4 相机位姿、由主点推出来的图像宽高
- 左右手各 21×3 关节（MediaPipe 顺序；对不上的 EgoDex 掌骨点不塞进去）、手腕 7 维（xyz + xyzw）、手腕置信度。没有 confidences 组时置信度是 null，四元数在片段内与上一帧同半球
- `annotation`：四级标注。EgoDex 只填环境和整段任务，子任务和分手指令是空数组
- `coverage`：归一后的环境、原始物体名、粗类别、任务名、动作类型

21 点对照写在 `scripts/egodata/egodex.py`：手腕用 `leftHand`/`rightHand`，食指尖用 `*IndexFingerTip`，拇指用 Knuckle → IntermediateBase → IntermediateTip → Tip。这是 ARKit 名字到 MediaPipe 的近似，不是逐点解剖注册。`Hand` 在前臂上，比 MediaPipe 腕点更靠肘；`ThumbKnuckle` 也不是拇指 CMC。比较误差时用 `EGODEX_NONCORRESPONDING_JOINTS`（0 和 1）把这两点排除。

## 4. LeRobot 导出和预训练（已接上）

`scripts/ego_to_lerobot.py` 只读 `egodata_qc` 产出率 CSV 里 `accepted=yes` 的片段（同一 id 多行时以最后一行为准），写成 lerobot 0.6.1 的 **codebase v3.0**。导出时一次只展开一条 JSON，打包后即丢掉，避免把全部嵌套列表留在内存里。读 parquet 的线性 BC 用 pyarrow 扁平数组，不再 `to_pylist()`。

转换侧（全量测试集上暴露出来的问题）：

- 454 条 HDF5 没有 `confidences` 组。以前记成 0，QC 把它们全部当成手出画，导出时关节也被置 0。现在记为未知（JSON `null`）：关节齐全就算这只手可用，出画只看投影。读到了低于 0.5 的数字仍然算低置信。
- 四元数不再每帧强制 `w >= 0`。`q` 和 `-q` 是同一个旋转，强制符号会在半球边界上让相邻帧翻转。现在在一条片段内部让后一帧与前一帧点积不小于 0。
- HDF5 的 N×4×4 按关节一次读出。`convert_egodex.py --workers N` 按文件多进程，id 和输出路径与单进程相同。任务文件夹若是符号链接，转换仍会走进去；`Path.rglob` 不跟随目录链接，以前会把整个任务丢掉。

向量（世界系）：

- `observation.state` 长度 140 = 左手 21×3、右手 21×3、左手腕 7、右手腕 7。某只手有缺测关节，或置信度是数字且低于 0.5 时，该手 63 维关节置 0。置信度未知不算无效。
- `action` 每一行是**下一步**相对当前手腕的增量，长度 14：左手 `dxyz` + 相对四元数，再接右手。相对四元数是 `q_next * conj(q_now)`，取 `w >= 0`。保持不动是 dxyz 全 0、四元数 `0,0,0,1`。片段最后一帧丢掉。`--include-hand` 时再接双手关节 xyz 增量。多步目标不摊进这一列：线性 BC 拼连续 `horizon` 行（默认 16），ACT 用同样的 `chunk_size`。
- 语言仍是该帧 SUBTASK，否则 `TASK.instruction`。

没有源 mp4 时写 16×16 占位视频。有同名 mp4 时裁到保留帧数，并用 `--video-size`（默认 224，偶数）缩成正方形，避免 1080p 重编码。

`scripts/ego_pretrain_bc.py` 做岭回归（闭式解）。动作和特征的逐维均值、标准差都只从全部训练池算一次，所有 `--max-frames` 共用，并同样作用在验证目标和保持不动的基线上。岭回归系数默认在该档训练样本内部留出 20% 来选，`--l2` 可以钉死。`--seeds 0,1,2` 时每个种子重抽验证 episode，日志写 `val_loss_mean` 和 `val_loss_std`。平移和旋转分开记。抽样前先在每条 episode 内打乱动作块，再轮转直到 N 个样本。

`scaling_law.py` 同时拟合 `a + b·ln(N)` 和 `L∞ + A·N^(-α)`（不依赖 scipy）。两条都写出 R²，更高的画成实线。同一 `size` 的多次运行合成一个点。第四次重跑的线性曲线在约 6.4 万样本后变平，默认曲线会落到饱和幂律，而不再把斜率外推成「数据翻倍就降多少」。

ACT：`notebooks/egodex_act_scaling_colab.ipynb` 把 EgoDex test 下到 Drive，转换、质检、按 224 导出，再按固定验证 episode 训三到四档 lerobot 0.6.1 ACT。T4/L4 上正式四档训练大约 80 分钟，连同 16GB 下载和导出大约 2.5–3.5 小时。ACT 这一版只条件于图像和 state。

覆盖报告逐条读取 JSON 后立刻丢掉，全量时不再把数 GB 的 episode 同时留在内存里。环境字段认 `tablecloth:`，并把 `lavendar` 收成 `lavender`。物体粗类别按当前关键词从物体名重算。`use` 这种笼统动词仍然算 other，不另造一个类别。

导出的 `stats.json` 含 `observation.image`。lerobot 0.6.1 打开 ImageNet 统计量时会改写它的 mean/std，键不存在会直接报错。

ACT 训练入口是 `scripts/ego_act_scaling.py` 和 `notebooks/egodex_act_scaling_colab.ipynb`。`examples/ego_act_train.yaml` 仍是单档命令。验证损失在反归一化之后的原始动作上算 L1，和保持不动基线同一单位。

## 5. 仍然不做的事

- 不下载、不提交 test.zip 或任何 HDF5/MP4/导出数据集，也不提交 EgoDex 画面。可视化测试用临时生成的小视频。真实数据上的 PNG/MP4 只留在本机或 Colab。
- 不在转换器里伪造时间分段语言。SUBTASK 为空时整段用 TASK.instruction。
- 不在仓库里跑 SLAM，也不把 IMU 双重积分。`convert_headcam.py` 只读取已有的 TUM；没有轨迹时相机位姿是单位阵，`coordinate_frame` 为 `camera`。标定文件只读，不求解。
- 不把 pusht ACT 改成吃手部关节。手部预训练用单独的配方，动作是手腕增量。
- 单元测试不导入 torch。装了 lerobot 0.6.1 时，用合成数据跑 1 步 CPU 的 `ego_act_scaling.py --run`，只确认命令能启动。完整四档仍在 Colab GPU。
- 不在这一轮用 3,243 条把 ACT 训完。

## 6. 建议的下一步

1. 用修好的转换器重跑 EgoDex test（`--workers` 大于 1），再看产出率。没有 confidences 的 12 个任务不应再整任务被拒。
2. 导出时加上 `--video-size 224`，在 GPU 上按 `examples/ego_act_train.yaml` 训 ACT。不同小时数的 `val_loss` 交给 `scripts/scaling_law.py`。线性 BC 只作基线，并和 `copy_current_wrist` 比。
3. 时间分段和分手指令单独做标注，过 `validate_hierarchy.py --strict` 再进训练。导出已经会读 SUBTASK。
4. 像素模糊和 HOT3D 仍排在这条链路之后。
5. 自有头戴还缺：在设备上跑 SLAM 并把 TUM 写进会话、做 Kalibr/OpenCV 标定、以及自动语言标注。手部后端、双目三角化和会话转换已经在 `scripts/headcam/` 与 `scripts/convert_headcam.py`。MANO / HaMeR / WiLoR 权重不要提交。
