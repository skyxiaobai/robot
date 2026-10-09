# 开放数据集流水线差距分析

> 日期：2026-10-09。对照一条完整链路：
> **开放数据集下载 → 转成统一格式 → QC / 产出率 → 覆盖统计 → 四级标注 → LeRobot 训练 → 缩放律评估**。
> 前提：还没有自有头戴采集设备，只能先用带手部位姿的开放第一视角数据。
> 规格见 `docs/headcam_data_spec.md` v10 §7–§8。

## 0. 结论

仓库里原来只有「仿真 benchmark 怎么训练」和「将来头戴设备该采什么」。中间从原始第一视角数据到可训练样本的质检、覆盖和标注是空的。这次补上统一 episode、EgoDex 适配器、产出率、覆盖报告和四级标注校验，并且用合成 HDF5 测过。真实 EgoDex 测试集的下载地址和单条 HDF5 结构已经核对过，16GB 的 zip 不进仓库。

QC 通过的统一 episode 已经能写成 LeRobot 0.6.1 的 v3.0 数据集（`scripts/ego_to_lerobot.py`）。CPU 上的线性 BC（`scripts/ego_pretrain_bc.py`）会打出 `val_loss`，`scripts/scaling_law.py` 可以直接吃不同帧数的两次 run。动作是下一帧双手手腕，不把当前姿态复制成标签。

还没接上的是：用完整的 `lerobot-train` / ACT 在 GPU 上吃这份 140 维状态，以及给 EgoDex 补上真正按时间切开的子任务和分手指令。像素模糊、HOT3D、自有设备 SLAM 仍不在这条链路上。截断帧数只证明缩放律接口能读第一视角日志，不是真实小时数的拟合。

## 1. 逐段对照

| 阶段 | 现在有什么 | 还缺什么 | 优先级 |
|---|---|---|---|
| 下载开放数据 | 已核对 EgoDex `test.zip`（见 §2）。没有下载脚本，避免把 16GB 拉进 CI | 需要时用 README 里的 `curl`。HOT3D clips、`lerobot/umi_cup_in_the_wild` 没有适配器 | P2：再加一个数据集 |
| 转成统一格式 | `scripts/egodata/schema.py` + `scripts/egodata/egodex.py`，`scripts/convert_egodex.py`。世界系 21 点关节、手腕 xyz+xyzw、相机位姿、整段语言。`scripts/ego_to_lerobot.py` 写成 LeRobot v3.0 | 只接了 EgoDex。没有源 mp4 时视频是 16×16 占位 | P2：再加一个数据集 |
| QC / 产出率 | `scripts/egodata/qc.py`，`scripts/egodata_qc.py`。手出画、视线飘移、相机角速度（模糊代理）、长时间静止。产出率 = 通过片段的帧数 / 原始帧数 | 模糊还没看像素（拉普拉斯）。坏段只能整段丢掉，不能把好的子段切出来留用 | P1：有 mp4 时加像素模糊；子段裁剪 |
| 覆盖统计 | `scripts/egodata/coverage.py`，`scripts/egodata_coverage.py`。环境 / 物体 / 物体类别 / 任务 / 动作类型，HTML+CSV，封闭词表里计数为 0 的算空档 | 词表是手写的一小套，不是从 111 个 EgoDex 任务自动长出来的目标配额。没有「该采多少才算补上」的数量目标 | P2：按目标小时数做配额 |
| 四级标注 | 规格 §8.1，样例 `examples/hierarchy_annotation.json`，校验 `scripts/validate_hierarchy.py`。EgoDex 转换只填 ENVIRONMENT 和 TASK | EgoDex 没有时间分段 SUBTASK，也没有分手 INSTRUCTION。不能把整段描述切成假时间段 | **P1：标注**（人工或模型），不要在转换器里编造 |
| LeRobot 训练 | pusht ACT 笔记本仍在（image+state[2] → action[2]）。另外 `scripts/ego_pretrain_bc.py` 在导出的手部数据上做线性 BC（CPU），日志有 `val_loss`。配方 `examples/ego_pretrain_bc.yaml` | 还没有用 lerobot 0.6.1 的 ACT / `lerobot-train` 吃 140 维状态和 14 维手腕动作。pusht 的 action[2] 没有改 | **P1：有 GPU 时把同一导出接上 ACT** |
| 缩放律 | `scripts/scaling_law.py`：最优验证损失对 ln(数据量)。线性 BC 用 `--max-frames` 打出不同数据量，测试里已经喂给拟合 | 还没有真实 EgoDex 不同小时数的预训练 run。截断帧数只验证接口 | P2：有了多组真实 run 再拟合，不必改公式 |

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
- 左右手各 21×3 关节（MediaPipe 顺序；对不上的 EgoDex 掌骨点不塞进去）、手腕 7 维（xyz + xyzw）、手腕置信度
- `annotation`：四级标注。EgoDex 只填环境和整段任务，子任务和分手指令是空数组
- `coverage`：归一后的环境、原始物体名、粗类别、任务名、动作类型

21 点对照写在 `scripts/egodata/egodex.py`：手腕用 `leftHand`/`rightHand`，食指尖用 `*IndexFingerTip`，拇指用 Knuckle → IntermediateBase → IntermediateTip → Tip。这是 ARKit 名字到 MediaPipe 的近似，不是逐点解剖注册。

## 4. LeRobot 导出（已接上）

`scripts/ego_to_lerobot.py` 只读 `egodata_qc` 产出率 CSV 里 `accepted=yes` 的片段（同一 id 多行时以最后一行为准），写成 lerobot 0.6.1 的 **codebase v3.0**：`meta/info.json`、`meta/stats.json`、`meta/tasks.parquet`（索引名是句子）、`meta/episodes/`、`data/chunk-000/file-000.parquet`、按片段一条的 mp4。冒烟不安装 torch；文件按 0.6.1 的读写器落盘，训练脚本用 pyarrow 读 parquet。

向量（世界系，与统一 episode 相同）：

- `observation.state` 长度 140 = 左手 21×3、右手 21×3、左手腕 7、右手腕 7。某只手有缺测关节或手腕置信度低于 0.5 时，该手 63 维关节置 0，`observation.hand_valid` 对应侧为 0。手腕 7 维只要数字齐全就保留。
- `action` 长度 14 = **下一帧**左手腕 7 维再接右手腕 7 维（xyz 米 + 四元数 xyzw）。片段最后一帧丢掉，不把当前姿态复制成动作。
- 语言在 `task_index`：帧时刻落在某条 SUBTASK 的半开区间里就用子任务句子，否则用 `TASK.instruction`，再否则用任务名。

没有旁边的源 mp4 时写 16×16 占位视频，并在特征 info 里标 `egodata.image_source=placeholder`。有同名 mp4 时裁到保留的帧数。

`scripts/ego_pretrain_bc.py` 是 `action ≈ state @ W` 的全批量线性 BC（CPU，学习率 0.01）。日志第一行有帧数，后面是 `step=K train_loss: … val_loss: …`。`--max-frames` 截断帧数，用来给 `scripts/scaling_law.py` 提供不同数据量。配方在 `examples/ego_pretrain_bc.yaml`。Colab 单元在 `notebooks/open_dataset_pipeline_colab.ipynb` 后半段，接着合成演示的产出率 CSV 往下跑。

## 5. 仍然不做的事

- 不下载、不提交 test.zip 或任何 HDF5/MP4/导出数据集。测试用临时合成文件，形状和属性按上面那条真实文件来。
- 不在转换器里伪造时间分段语言。SUBTASK 为空时整段用 TASK.instruction。
- 不跑 SLAM。EgoDex 的世界系是设备上算好的；自有头戴还没有相机，规格 §7.2 的双目/SLAM 仍然是硬件到位以后的事。
- 不把 pusht ACT 改成吃手部关节。那是另一条训练契约。线性 BC 只证明导出和验证损失日志能接上。
- 不安装 `lerobot==0.6.1`（它会拉 torch）来做 CPU 冒烟。

## 6. 建议的下一步

1. 在有磁盘的机器上解压 EgoDex test，跑 `scripts/demo_open_dataset_pipeline.py --input <test目录> --limit 20`，再跑 `ego_to_lerobot.py`，看真实数据的产出率和导出帧数，而不是只看合成样本。
2. 有 GPU 时用同一份 v3.0 导出接 `lerobot-train` 的 ACT，输入仍是手部状态和图像，动作仍是下一帧手腕，不要改回 pusht 的 action[2]。
3. 时间分段和分手指令单独做标注，过 `validate_hierarchy.py --strict` 再进训练。导出已经会读 SUBTASK，不需要改格式。
4. 真实数据上至少两次不同小时数的预训练日志，直接交给现成的 `scripts/scaling_law.py`。`--max-frames` 只适合接口验证。
