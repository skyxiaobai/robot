# C16 EgoVerse/Mecka 预训练（已入队 ready，未提交 Kaggle）

假设：EgoVerse 自带 MANO 真值 3D 关键点 + 头部位姿 + 真实内参，不需要 WiLoR 打标，
且静止漂移比 WiLoR 小，可能比 xPhi（猜内参）/HOI4D 更适合做 XHEY 预训练。无深度，结果只作方向参考。

## 数据与质检（盒子，2026-10-19，原生 10 Hz）
- 访问：公开只读（EgoVerse 仓库 README 公开 key → setup_secret.sh 回退 public read-only），
  按 SQL `lab='mecka'` 选段（sync_s3 的 mecka 预设按 embodiment 过滤已失效）。目录 41,617 段 = 828.4 h。
- ⚠️ 导出 30 fps 是 10 Hz 线性插值（等速段恰 3 帧）；转换只取关键帧输出 10 fps，按 30 fps 做 QC 是假合格。
- 无偏估计（seed 42 按 task 分层随机 100 段，2.028 h）：整段通过 57/100（失败：抖动大 42、手几乎不动 1），
  3 秒分段保留 1.019 h = 50.2%；内参 100/100，头部位姿恒定 13/100。
  抖动（原生率）：逐帧跳变中位 0.84 cm、jitter 中位 2.32 cm、静止漂移 RMS 0.46 cm、1 秒位移中位 2.80 cm（动起来 8.70 cm）。
  外推 828 h：分段保留 ≈416 h、整段通过 ≈444 h（注意：分层按段抽样后按小时加权；任务构成/段长分布变化会漂移，只作量级估计）。
- 训练池：随机批次共 600 段（12.00 h）原生率 QC：整段通过 50.7%，分段保留 5.63 h（46.9%），头恒定 42/600（7.0%）。
  排除头恒定、按手腕速度从快到慢选，分段保留封顶 ~4 h → **401 段（总 8.03 h，保留 3.994 h，整段通过 227/401）**。
- 相机系校验（前 15 段）：逆头部位姿转 OpenCV 系后投影，z>0 100%、关节落在 640×360 内 99.2–100%。

## 训练（kaggle/rsi_c16，dataset justinz2010/xhey-egoverse-cam 私有）
臂 A `iphone+egoverse+xhey_rgb+aug`（--qc-filter + --qc-segments），臂 B 对照 `iphone+xhey_rgb+aug`；
其余同 C11–C14：3 段深度片 LOO、3 seeds、pre 6000、steps 3000、move-cm 5、--no-final。
状态 ready，等待自动迭代/空闲 GPU 槽提交，不要重复提交。许可：目录 freeform 行 license 为空
（旗舰行 CC BY-SA 4.0），仅私有研究用，不再分发。
