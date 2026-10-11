# iPhone mcap STATUS（2026-10-11 13:30 CST）

## 已完成
- [x] 180 帧冒烟：zero ADE **8.38 cm**（PR #37）
- [x] **全长 stride=10**：3 段共 **9325** 帧（覆盖 14–22 分钟整段录制）
  - 标签覆盖 96–99%（很好）
  - 但有效 fps≈3 → `valid_action_ratio` 0.003–0.09，`valid_full_chunk@16`=**0**
  - 训练 ADE = null（无法与零动作比较）
  - QC 因 staged_static / hands_out_of_frame 全拒（低帧率阈值）

## 进行中
- [ ] 全长 **stride=2**（~15 fps）重转+融合+训练，保留时序密度后再比零动作

## 结论（当前）
全长抽稀（stride=10）**不能**替代「更多近 30fps 片段」；要更多帧且 ADE 可算，需更高时序密度。
