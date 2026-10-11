# HOT3D scale STATUS（2026-10-11 13:30 CST）

## 完成
- [x] 24 clips 下载 + WiLoR 双目流水线（每段 max 150 帧）
- [x] QC 21/24 通过；覆盖均值 79%；手腕 vs GT 中位 **1.24 cm**
- [x] `hot3d_train_compare` GPU：zero ADE **5.026 cm**；linear/MLP stereo 与 GT 均差于零动作
- [x] 文档 `docs/hot3d_stereo/train_compare.md` + chart

## 结论
扩大数据后标签更准，但 BC 仍打不过零动作 → 换模型/表示，不单堆帧。
