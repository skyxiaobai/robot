# iPhone mcap STATUS（2026-10-11 13:01 CST）

## 已完成（路径 A）
- [x] 3× Pro Max mcap 下载 + mcap_adapter
- [x] WiLoR↔LiDAR 尺度修复（LiDAR 手腕 + WiLoR 手型）→ 覆盖 ~53%
- [x] RGB-D 融合 3 段；QC 通过 2/3
- [x] LeRobot 导出：2 ep / 358 frames（`run_all/lerobot`）
- [x] 小模型训练对比：`scripts/iphone_train_compare.py`
  - 2 段：zero **8.38** cm；linear 41.9；mlp 24.3（均差于零动作）
  - 3 段：zero **8.54**；linear 11.1；mlp 18.8
- [x] `docs/iphone_arxiv_plan.md`（A→B→C）
- [x] `docs/iphone_mcap/train_compare.md`

## 未完成
- [ ] 更多深度片段（目标 10+）后再训，争取 ADE ≤ zero
- [ ] 图像 ACT / 仿真评测（数据够了再上）
- [ ] 代码合入 main（本轮脚本 + 文档）

## HOT3D（后台）
- 流水线仍在跑；见 `outputs/hot3d_scale/`
