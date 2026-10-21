# C17 手部打标方法对比（WiLoR / Dyn-HaMR / EgoXperience）

Justin 2026-10-21：「两个都排上：标签方法对比和 Flora 筛片段」。本文件是标签方法对比部分（Flora 见 [c18_flora.md](c18_flora.md)）。
队列状态 **done_blocked_dynhamr_wilor_only**（不会被 auto-iter 再提交）。

## 方法与许可
| 方法 | 代码 / 权重 | 许可 | 结果 |
|---|---|---|---|
| WiLoR（基线，现行打标器） | rolpotamias/WiLoR + HF space 权重 | 代码 CC BY-NC-ND（权重非商用），MANO 需单独许可 | 跑了：①现行 iPhone 标签 = WiLoR + LiDAR 放置；②WiLoR 单目（xhey_rgb 同一流程，真实内参） |
| Dyn-HaMR（CVPR 2025） | github.com/ZhengdiYu/Dyn-HaMR（含 hamer 子模块） | 代码 MIT，README 写「non-commercial research」；依赖 HaMeR 数据、MANO（非商用）、Hand-BMC | **blocked，未产出标签**：v6 卡在 dh_run → HaMeR 预处理子进程（见下「Dyn-HaMR 尝试记录」） |
| EgoXperience | github.com/Ropedia/EgoXperience 只有 README/CITATION/demo.mp4，无 LICENSE；HF ropedia-ai/EgoX 只有 README（"Model weights coming soon"）；论文 arXiv 2610.26625 | 未声明 | **跳过：代码和权重都未发布**（2026-10-21 核实） |

另：LambdaHumanoid/openlambda（λ0，Apache-2.0）只引用 Dyn-HaMR + ViPE 相机，不包含 Dyn-HaMR 代码，不能替代。

## 评测设置
- 片段：3 段 iPhone LiDAR 片（2026-08-22_18-21 / 08-24_21-18 / 09-05_17-16），每段在 20%/50%/80% 分位附近取手有效率最高的 20 s 窗口（300 帧，15 fps）→ 9 个深度窗口；另 3 段 xhey_rgb 无深度片各 1 个窗口（帧 750–1050）。
- 指标（1）：scripts/wrist_noise.py（C15 定义）：逐帧跳变、jitter、静止漂移 RMS，单位 cm，相机系。
- 指标（2）：手腕投影到 LiDAR 深度图（256×192，5×5 窗口、conf≥1 中位数）比较 z。
- 09-05 的现行 LiDAR 标签文件为空（0 字节），所以「现行 iPhone」只有 6 个窗口（12 条手序列）。
- WiLoR+LiDAR 本来就用 LiDAR 放置，深度误差是**循环**的，只作下限参考。

## 结果（1）抖动 / 静止漂移（cm，中位数）
| 方法 [窗口] | 手序列 | 逐帧跳变 | jitter | 静止漂移 RMS | 1 s 位移 |
|---|---:|---:|---:|---:|---:|
| 现行 iPhone（WiLoR+LiDAR）[深度] | 12 | 1.30 | 1.95 | 1.16 | 6.07 |
| WiLoR 单目原始 [深度] | 18 | 1.06 | 1.08 | 0.88 | 3.84 |
| WiLoR 单目 × 每窗 LiDAR 尺度 [深度] | 18 | 2.46 | 2.55 | 1.26 | 8.83 |
| WiLoR 单目 [无深度 xhey_rgb] | 6 | 0.54 | 0.69 | 0.45 | 2.47 |
| 参考 C15 iphone（整段） | | 1.38 | 1.80 | 1.06 | 6.17 |
| 参考 C15 xhey_rgb（整段） | | 1.18 | 1.42 | 0.99 | 5.49 |

单目原始的数字小是**假象**：单目 WiLoR 把手放得太近（z 只有真实的 1/2.36），cm 噪声被同比例缩小。
按每个窗口/手一个 LiDAR 尺度系数还原后（中位 2.36×，范围 2.01–2.57），单目的跳变 2.46、jitter 2.55 cm，
约为现行 LiDAR 标签的 1.3–1.9 倍。

## 结果（2）手腕绝对深度 vs LiDAR
| 方法 | n 帧·手 | z 误差中位（有符号） | \|dz\| 中位 | \|dz\| p90 | ≤2 cm |
|---|---:|---:|---:|---:|---:|
| 现行 iPhone（循环） | 3600 | +1.00 | 1.37 | 3.77 | 67% |
| WiLoR 单目原始 | 5394 | −68.8 | 68.8 | 82.8 | 0% |
| WiLoR 单目 × LiDAR 尺度（尺度用 LiDAR 拟合，偏乐观） | 5394 | 0.0 | 3.24 | 12.4 | 34% |

结论：单目 WiLoR 的绝对深度在 XHEY 头戴视角下不可用（手实际在 1.2–1.3 m，单目给 ~0.53 m）；
即使给它一个正确的全局尺度，帧间深度仍有 3 cm 级误差、p90 12 cm。LiDAR 放置仍是现行最好的绝对深度来源。

## 结果（3）每套标签训练一次 XHEY
**跳过。** 原因：
- WiLoR 单目：绝对深度错 2.36×，用它训练再拿 LiDAR 标签评测，差异主要是尺度错，不是标签方法本身；而且整 3 段深度片重打需 ≈10.4 h（单 T4，0.8 s/帧）。
  若评测也换成单目标签，GT 变了，ADE 不能与 C11–C16 比较。
- Dyn-HaMR：没有在任何窗口产出结果，无从训练。
- EgoXperience：未发布。

## 结果（4）单张 Kaggle GPU 能否跑、耗时
- WiLoR 单目：T4 上 300 帧窗口 227–256 s（中位 ~240 s，≈0.8 s/帧，含检测）。3 段深度片 46,620 帧 ≈ 10.4 h（单 T4）。
- Dyn-HaMR（T4，ARKit 相机跳过 SLAM）：安装链在 T4 上全部通过且耗时可控——uv py3.10 venv 11 s、torch 2.1.2 cu121 30 s、requirements 32 s、
  detectron2 sm75 CUDA 编译 108 s、hamer+ViTPose 3 s、HaMeR demo 数据 6 GB 213 s、Hand-BMC 计算 324 s（合计 ≈12.5 min）。每窗口优化耗时：**未测到**——v6 进入 run_opt 后 69 s 就在 HaMeR 预处理失败，没有任何窗口跑到优化阶段，因此无法与 WiLoR（~240 s/窗口）比较速度。

## Dyn-HaMR 尝试记录（每步 rc 见 c17_steps.png）
| 内核版本 | 首个失败步骤 | 原因 / 修复 |
|---|---|---|
| v1 | dh_import_check | uv venv 没有 pip，mmcv/chumpy 没装上；管道 `| tail` 吞了返回码 → 加 pip、`bash -o pipefail` |
| v2 | dh_pkg | 仓库 requirements 钉 setuptools==59.5.0，不支持 PEP 660 editable 安装 → 去掉这一行、装完再升到 69.5.1（盒子 CPU 复刻验证） |
| v3 | dh_run（第一个窗口 6 s 就退出） | run_opt.py → optim/base_scene.py `import mano`：ModuleNotFoundError（仓库没写这个依赖，代码里只在注释中用） |
| v4 | （被 v5 取代，无输出）| 装 otaheri/MANO + libEGL（pyrender 需要）+ `import run_opt` 检查 |
| v5 修复内容 | （修复周期 1）| 盒子复刻继续发现缺 loguru、plyfile、human_body_prior 需 git main 版（PyPI 版无 load_model，且声明 py≥3.11，用 --ignore-requires-python）；盒子上 `import run_opt` 通过 |
| v6 | **dh_run（最终，关闭）** | 所有安装步骤和 import run_opt 都通过；进入第一个窗口 69 s 后失败：preproc/launch_hamer.py 用裸 `python -u run.py` 起 HaMeR 子进程，解析到 Kaggle 系统 Python 而不是 py3.10 venv → `No module named 'pyrender'` → HaMeR 结果 pkl 没生成 → `os.rename` FileNotFoundError。按规则不再重发；修法很可能是把 venv 的 bin 放到 PATH 最前（未验证） |
- EgoXperience：无法评估（未发布）。

## 图
- [`experiments/report/C17/c17_noise.png`](../../experiments/report/C17/c17_noise.png)：抖动 / 静止漂移
- [`c17_depth_error.png`](../../experiments/report/C17/c17_depth_error.png)：深度误差 CDF + 直方图
- [`c17_mono_scale.png`](../../experiments/report/C17/c17_mono_scale.png)：单目尺度误差（每窗/手）
- [`c17_steps.png`](../../experiments/report/C17/c17_steps.png)：Kaggle 步骤 / 耗时 / 首个失败步骤

## 复现
- 内核 `kaggle/c17_labels/`（slug justinz2010/xhey-c17-labels；数据集 xhey-iphone-sessions 私有、xhey-c15-wrist-data；MANO 只在私有数据集里，不公开）。
- 盒子：`scripts/c17_eval.py --kout <kernel output> --out <dir>`，`scripts/c17_dyn_to_cam.py <kout>`，`scripts/c17_charts.py <results> <kout> <out>`。
