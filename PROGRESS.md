# PROGRESS

## 2026-10-11 17:42 CST · XHEY 模型 + 开源/合成配比消融

- 新模型名 **XHEY**（xhey App iPhone 数据）；脚本 `scripts/train_xhey.py`（`--mix` / `--ablation`）。
- 消融（留一段 iPhone）：zero ADE **16.68 cm**；最好的 XHEY=iphone-only **17.60**（未赢 zero）；EgoDex/HOT3D/aug 均无帮助。
- 文档 `docs/xhey_model.md`，图 `docs/xhey/xhey_results.png`。ckpt：`outputs/xhey/xhey.pt`（不进 git）。


## 2026-10-11 13:02 CST · iPhone 路径 A 第一张表
- LeRobot：2 ep / 358 帧（QC 通过）
- 训练：zero ADE **8.38 cm**；线性/MLP 均差于零动作（数据太少，同早期 HOT3D）
- 计划：`docs/iphone_arxiv_plan.md`；结果：`docs/iphone_mcap/train_compare.md`

## 2026-10-11 13:30 CST · HOT3D 24clips + iPhone 全长
- HOT3D：3576 帧；手腕 vs GT 中位 1.24cm；zero ADE **5.026**；linear/MLP 均更差
- iPhone 全长 stride=10：9325 帧覆盖高但 valid_chunk@16=0，ADE 无法比；改 stride=2 重跑中

## 2026-10-11 14:16 CST · iPhone 全长 stride=2
- 46620 帧 @15fps；zero ADE **16.68**；MLP 16.56（≈持平）；线性 24.5
- 全长未推翻「需更多数据/换模型」

## 2026-10-11 14:50 CST · Path B 质量对比
- 文档：`docs/iphone_mcap/quality_compare.md` + `quality_*.png` / `quality_metrics.json`
- 脚本：`scripts/iphone_quality_compare.py`
- 要点：采集层稳（≈30fps、RGB=Depth、LiDAR≈87%）；精度代理 RGB-D 手腕 1.51cm vs HOT3D 双目 1.24cm vs EgoDex 绝对≈22cm；整段 WiLoR 有效手≈17%（需披露）
