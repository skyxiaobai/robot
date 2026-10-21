# C18 Flora 筛片段 vs 现行「运动阈值 + QC」

Justin 2026-10-21：「两个都排上：标签方法对比和 Flora 筛片段」。队列状态 **done_blocked_hw**（研究项，不会被 auto-iter 提交）。

## Flora 核实（2026-10-21）
- 真实仓库：HF `nvidia/Flora-TempSeg`（Flora-segmenter-9B = Qwen/Qwen3.5-9B + LoRA r32）和 `nvidia/Flora-Captioner-27B`，HF 发布 2026-10-15，论文 2026-10-20，未设访问门槛。
  推理代码 `batch_utils/` 在 main 版本 `78a0e4ac`（适配器版本 `114a1076` 没有代码）。
- **许可：NVIDIA OneWay Noncommercial License，只能研究用。XHEY 是商业研发，不能用于生产筛片段**；本实验只作研究评估，结果不进训练数据管线。
- 官方验证硬件：Ampere 及以上、80 GB 显存；单视频 ≤120 s、≥4 fps。人类视角配置：16 s 窗口、4 Hz、12 s 核心区，输出 `{"segments":[{s,e,lo,ro}]}`（20 Hz tick）。
- 只用分段模型（9B），不跑 27B 打字幕模型。

## 现行选择（对照，盒子上算好）
| 数据 | 片段 | 有标签时长 | 现行保留 | 保留段数 | 保留部分 QC 通过 | 保留部分 jitter 中位 | 1 s 位移中位 |
|---|---:|---:|---:|---:|---:|---:|---:|
| xPhi-Ego-100（QC 3 s 窗口全规则通过） | 102 | 5.03 h | 0.315 h（6.3%） | 196 | 100% | 2.40 cm | 2.90 cm |
| Open-AoE nano（20 s 段，运动中位 ≥3 cm/s） | 111 | 2.94 h | 2.54 h（86.5%） | 129 | 100% | 0.71 cm | 8.46 cm |

（xPhi 全部 5.03 h 的 QC 通过率只有 6.3%，jitter 中位 9.4 cm——xPhi 标签本身抖动大，现行 QC 主要在挡这个。）

## Flora 运行（Kaggle，justinz2010/xhey-c18-flora）
这个账号没有 L4/A100：申请 NvidiaL4X1 实际退回到 2×T4（每张 14.6 GB，sm75，没有原生 bf16）。
做法：同 vendor `tempseg_fp32_loader` 逻辑（BF16 底座 + FP32 LoRA + FP32 GDN），但用 accelerate device_map 把 9B 拆到两张 T4；vendor 文件不改。

| 版本 | 结果 | 卡住的步骤 |
|---|---|---|
| v1 | 失败 | snapshot 没下到 batch_utils；L4 退回 T4 |
| v2 | 失败 | 适配器版本 114a1076 没有 batch_utils → 固定代码版本 78a0e4ac |
| v3 | 跑完但 0 段 | 我的内核 bug：probe() 要 pathlib.Path，传了 str（盒子上用假模型端到端验证修复） |
| v4 | 冒烟失败（设计如此，快速失败） | 模型加载 822 s 成功（GPU0 7.3 GiB / GPU1 10.6 GiB，适配器和 GDN 值逐一校验通过）；**一个 16 s 窗口（1811 token）的 prefill 用了 651 s**，超过 Flora 自己 generate 的 max_time=300 s，只生成 1 个 token →「did not return a complete output」 |
| v5 | 冒烟失败 → **关闭** | 装上 flash-linear-attention 0.4.1（import 成功），模型加载 938 s；同一个窗口 prefill **823 s**，仍只生成 1 个 token、输出不完整。fla 没有带来加速（T4 sm75 无原生 bf16；GDN 按 vendor 要求保持 FP32） |
可行性：两个数据集共 7.97 h 视频 ≈ 2,400 个 16 s 窗口（12 s 步进，未计修补重试）。按 v4 实测每窗口 ≥651 s，
光 prefill 就要 ≈430 GPU·h，远超 Kaggle 单次 12 h、每周配额。v5 实测 823 s/窗口 → ≈546 GPU·h。两次都连一个窗口都没跑完，而且每窗口还要再生成几十到几百个 token。

## 结论
**done_blocked_hw**：Flora-segmenter-9B 在这个 Kaggle 账号能拿到的硬件（2×T4）上，连一个 16 s 窗口都在 Flora 自己的 300 s 生成时限内跑不完，
外推全量需要 430–550 GPU·h；按 Justin 的规则（不硬跑）关闭，不再迭代。
所以本次**没有** Flora 与现行选择的段数/保留时长/重叠/质量对比数字——只有上面现行选择这一侧的基线。
要真正比较，需要 Ampere+ 80 GB（A100/H100）；即使跑通，许可也只允许研究用，不能接进 XHEY 生产筛片段。
盒子上的对比管线（c18_compare.py / c18_charts.py）已用假 Flora 输出测过，拿到输出即可直接出 IoU、互相召回、段数、保留时长、QC 通过率和 jitter。

训练对比（可选项）：没有 Flora 选择结果，不做。

## 图
- [`experiments/report/C18/c18_feasibility.png`](../../experiments/report/C18/c18_feasibility.png)：实测耗时与外推
（同图左/中两栏：现行选择保留时长与质量）

## 复现
- 内核 `kaggle/c18_flora/`（当前版本是 v5 冒烟诊断：默认 `SMOKE_ONLY=1`，设 0 才跑全量；数据集 justinz2010/xhey-xphi-raw；Open-AoE nano 在内核里从 HF inclusionAI/OpenAoE-2000h rev `nano` 直接下载——从盒子上传到 Kaggle 被自动审核拦了）。
- 盒子：`scripts/c18_compare.py --flora <dir> --out <dir>`、`scripts/c18_charts.py <summary_dir> <out>`（Flora 输出到位后可直接出对比表/重叠/质量/时间线图）。
