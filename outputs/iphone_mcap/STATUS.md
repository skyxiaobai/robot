# iPhone mcap STATUS（2026-10-11 14:16 CST）

## 已完成
- [x] 180 帧冒烟：zero ADE **8.38 cm**（PR #37）
- [x] 全长 stride=10：9325 帧，ADE N/A（3 fps）
- [x] HOT3D 24clips（PR #38）：zero **5.026 cm**
- [x] 全长 stride=2：46620 帧 @15 fps → 融合+训练
  - zero **16.68 cm**；linear 24.5；**MLP 16.56（≈零动作，CI 重叠）**
  - 全长没有改变「打不过 / 持平零动作」的结论

## 结论（路径 A）
管线可训；当前 3 段全长仍不足以宣称 usable policy。需要更多近 30fps 数据或换模型。
