# PROGRESS

## 2026-10-11 13:02 CST · iPhone 路径 A 第一张表
- LeRobot：2 ep / 358 帧（QC 通过）
- 训练：zero ADE **8.38 cm**；线性/MLP 均差于零动作（数据太少，同早期 HOT3D）
- 计划：`docs/iphone_arxiv_plan.md`；结果：`docs/iphone_mcap/train_compare.md`

## 2026-10-11 13:30 CST · HOT3D 24clips + iPhone 全长
- HOT3D：3576 帧；手腕 vs GT 中位 1.24cm；zero ADE **5.026**；linear/MLP 均更差
- iPhone 全长 stride=10：9325 帧覆盖高但 valid_chunk@16=0，ADE 无法比；改 stride=2 重跑中
