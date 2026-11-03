# Astra Phase-0：公开数据入口、失效关闭和证据隔离

本页及账本末尾的 `review_note` **覆盖 PR #85 中过强的结论和旧启动指令**。
仅实现 Phase-0；`docs/xhey/queue.json` 的 `launch_paused` 保持不变。
不使用 Cursor Cloud Agents，不下载数据、不提交 Kaggle、不合并 main。

## 当前可用的路径

- `scripts/train_xhey.py` 默认 `--mix egodex`，调用小型 `xhey_public.py` 入口。
  **不加载 iPhone**，也不存在隐式 iPhone 微调。`iphone` / `xhey_rgb` 训练请求直接报错；
  旧 `train_mix()` API 同样拒绝训练。iPhone 转换、RGB-D、探针加载及回归测试保留。
- 显式 `--split` 中的训练、验证、测试 episode 和独立单元不可交叉。
  先按这些单元确定归属，再使用 episode 内窗口。normalization 只用 train，
  最佳步和 shrink 只用 val；保存已选模型，**不在 train+val/test 上重训**。
- Phase-0 一次一个训练配比/图像模式，只有单阶段 `--steps`。
  旧 `--ablation` / `--mixes` / `--pre-steps` 拒绝而不是悄悄忽略。
  `--no-final` 表示不保存 checkpoint；`--prepare-only` 只校验本地数据、输出实际清单。
- `egodex / egoverse / hoi4d / hot3d / openaoe` 是格式入口，不是许可批准或效果排名。
  多源须使用相同实际 FPS；不混合 10 Hz / 15 Hz 后假称统一一秒 horizon。
  **优先用一个获准且有视频的 EgoDex 小分片验证，不假定现有 Kaggle 缓存有视频。**

## 显式划分格式

以下只有格式示例，**不是已锁定的科研划分，也不是可判断胜负的样本量**：

```json
{
  "unit": "session",
  "train": ["egodex/task/train_a", "egodex/task/train_b"],
  "val": ["egodex/task/val_a"],
  "test": ["egodex/task/test_a", "egodex/task/test_b"],
  "groups": {
    "egodex/task/train_a": "subject_A_session_1",
    "egodex/task/train_b": "subject_B_session_1",
    "egodex/task/val_a": "subject_C_session_1",
    "egodex/task/test_a": "subject_D_session_1",
    "egodex/task/test_b": "subject_E_session_1"
  }
}
```

`unit` 支持 `episode/session/subject`。`groups` 必须由数据管理员按真实依赖关系提供，
使用最大的相关单元；代码不可能从文件名推断独立主体。多个剪辑来自同一会话时应同组。
跨源重复主体/素材也必须同组；不能靠改 ID 消除依赖。所有指定 ID 必须存在且经过 QC 后有有效窗口。
不随机划分窗口，不自动创造 E1–E5 的正式划分；正式协议、最小有用改进和样本量门槛后续锁定。

## RGB 与实际数据清单

- EgoDex 转换器记录 HDF5 同名 `.mp4` 的绝对路径和 `video_fps=30`；搬迁后需正确重绑定路径。
  也可在 episode 中写相对 `video_path`（相对该 JSON 文件）。路径缺失不能默认为无图。
- 下采样使用 `video_frame_indices`：30→15 Hz 选择原视频 0、2、4…帧；动作行使用当前帧的画面，
  不使用未来目标帧。检查时间戳间隔、原视频 FPS、可解码帧；短视频报错，**不补零**。
  自定义抽帧需提供真实索引；不支持把未对齐/VFR 视频当 CFR 悄悄使用。
- `--state-only` 是显式无图消融，不解码视频、不加载视觉权重。
  RGB 模式使用已有的 torchvision ImageNet ResNet18 特征。骨干仍冻结；训练更新的是
  MLP 的图像输入权重，**没有宣称端到端微调 ResNet**。
  实际 RGB 运行需事先准备可合法使用的本地视频和 torchvision 权重缓存；权重缺失立即报错，
  特征提取不会自动联网下载。
- `actual_data.json`：代码 revision/执行脚本 SHA256、参数、划分及其 SHA256、QC 输入 SHA256、
  选中 episode 的路径/SHA256/原始来源及许可备注、
  视频路径/尺寸/mtime、实际 FPS、每分区独立单元数、分源有效窗口数/图像窗口数/候选权重。
  每个 seed 另外记录真实随机抽样次数、唯一已抽窗口数、实际权重、有图抽样次数。
  图像覆盖率可由 `image_windows/windows` 或 `image_rows/rows` 得到。
  新输出目录强制避免复用旧 metrics/checkpoint 假装本轮成功。
- `metrics.json` 包含清单、逐 seed 训练曲线、zero / constant_velocity / 模型结果；
  `window_errors.npz` 保留配对窗口/手误差、episode/group ID 和 moving mask，便于独立复核。
  源 JSON 重新读取，不再信任旧 array cache；昂贵 RGB 特征仍按视频指纹和对齐信息缓存。

## 统计与账本

`xhey_stats.cluster_summary` 对**完整 episode/session/subject**重采样，组内保留全部窗口和双手。
同窗口 seed 先平均，seed 数不会膨胀独立样本数。比较使用同一有限值/moving mask 的配对差。
点估计仍是 pooled window-hand ADE，bootstrap 估计量是重采样组的误差和/计数和，
不是把窗口当独立样本，也没有悄悄把 pooled ADE 改成等权 episode ADE。
报告 `ci_method`、`ci_estimand`、`n_eval_units`、`n_moving_eval_units`；少于两个组不给 CI。

**2–3 个独立片段即使换成 cluster bootstrap 仍不足以支持微小差距。**
Phase-0 输出一律 `decision_eligible=false` / `evidence_status=evidence_insufficient`。
旧窗口级区间、缺统计元数据或未锁最小有用改进的记录，不再自动产生 win/loss。
这是 XHEY 决策链的修复；其他历史独立评测脚本的旧窗口 CI 不能拿来代替本协议。

`experiments/ledger.jsonl` 的原始 40 条指标保持逐字节不变，只追加三条审查事件：

| 对象 | 当前结论/处置 |
|---|---|
| **C10–C16（含已有对照臂）** | **证据不足**，撤回胜/负、数据源有效/无效的判断；保留数值 |
| C15 | 没有伪造结果；旧候选仍暂停，不以旧评测恢复运行 |
| C14 与涉 xPhi 的自动迭代行 | 隔离，`proposer_eligible=false`，等待数据/运行审计 |
| iter003_c2、iter004_c0、iter004_c2 | `xphi_not_loaded_suspected`：模板缺挂载和 `--xphi` 是强证据，不以同 ADE 当铁证 |
| iter005_c0/c1/c2 | 前瞻审查目标；没有查询远端状态，也没有声称已完成/已加载 |

`xhey_ledger.reviewed_rows` 为提议器、CSV 镜像、汇总图统一应用追加事件，事件本身不当训练样本。
历史 `status: ok` 仅表示原记录的执行状态；结论看审查字段。新摄入的旧结果同样证据不足。
旧 CSV、旧图表和 queue 中历史胜负字段不重写为伪造的新实验；本页和 review notes 撤回其结论。
运行 `mirror` 时才会按审查视图刷新派生 CSV。汇总图不再把负数一律着绿称“赢”。

## 自动迭代和 Kaggle

- 保持 `launch_paused`；`propose` 在暂停时不提议，`launch` / 固定队列直接调用也受暂停保护。
- 即使临时移除暂停，提议器只接受已有挂载/参数契约的显式候选空间；不能从历史最佳 C14
  继承 xPhi。xPhi 固定候选也单独拒绝。被隔离/未批准用于 proposer 的行不进入代理拟合。
- 生成的 Kaggle 模板对缺挂载路径报出源名；训练 shell 使用 `check=True` 和 Bash
  `set -euo pipefail`。各历史 Kaggle XHEY driver 的**训练命令**也做同样退出码传播，
  不再被 `; tail` 或 `| tail` 隐藏失败。依赖安装/打标等非训练命令不在本轮改动范围。
- 模板和历史 driver 仍描述旧 iPhone 实验，**没有改造成可恢复的公开训练调度器**；
  不能直接重放旧 tar 包/旧候选。当前 checkout 的训练入口会拒绝 iPhone 配比。
  自动迭代最后恢复，本轮没有挂载 xPhi、没有 Kaggle 提交。

## iPhone、许可、后续边界

- 自采 iPhone 深度与 `xhey_rgb` **冻结为回归/标签噪声探针，仅评测**；不参与训练、调参、
  early stop、shrink、选型、few-shot，也不追加采集。冻结不能消除历史测试污染。
  管线不删除，既有测试继续运行。
- 不笼统称“金标”：LiDAR 深度、设备追踪、MANO 拟合的误差及标定证据分别核对。
  C17 的尺度比例仍需排除内参、单位、转换等混杂因素。
- EgoDex/EgoVerse 是**候选顺序，不是实验排名**。未知授权的 freeform 分片不进入新训练。
  数据、代码、打标权重、MANO 资产、派生标签、checkpoint 逐项保留许可血缘；研究/商用分账。
- 许可、容量和获准小样本核验可与代码修复并行，不以“全部 Phase-0 完成”阻塞所有工作。
  后续才锁正式 train/val/test、有效样本量和最小有用改进，做固定预算、多 seed 单源有图/无图
  对照（加静止/恒速基线），有证据再考虑第二源/扩量，自动迭代最后恢复。

## 本地 smoke-test（不需要真实数据或网络）

本环境已存在的依赖环境是 `/workspace/egodex_act/lrenv/bin/python`；其他机器替换为安装了
numpy/scipy/pandas/pyarrow/h5py/OpenCV/torch/torchvision/pytest/matplotlib 的 Python。

```bash
PY=/workspace/egodex_act/lrenv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
$PY -m pytest -q tests/test_phase0_astra.py
$PY -m pytest -q tests
$PY scripts/xhey_auto_iter.py propose  # 只打印暂停，不访问 Kaggle
```

fixture 使用自造的小 episode 和真实可解码视频，以本地微型 encoder 代替 ImageNet 权重，
实际执行帧对齐、缓存校验、MLP 训练/验证、图像输入权重更新、抽样日志及 CLI state-only smoke。
它证明管线接通，**不证明真实 EgoDex 分片的许可/标定/泛化收益已经通过验证**。

已有获准本地数据的手工准备检查（不会下载数据，不会提交任务）：

```bash
$PY scripts/train_xhey.py --mix egodex --egodex /approved/egodex/unified \
  --split /approved/split.json --state-only --prepare-only --out /tmp/astra-state-check
# RGB: 去掉 --state-only，换一个新 --out；需本地同名视频/正确 video_path 和视觉权重缓存。
# 极小训练：去掉 --prepare-only，添加 --steps 2 --seeds 1 --hidden 8 --no-final。
```

全套离线测试：**336 passed、9 skipped、2 subtests passed**。跳过项为缺失可选依赖/权重
（liblzfse、MuJoCo、MediaPipe、HaMeR、WiLoR）；没有为测试下载数据或模型。
`git diff --check` 通过；账本原始 40 行和三份旧说明文档均验证为字节前缀不变、仅追加。
