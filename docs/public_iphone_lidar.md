# 公开 iPhone LiDAR 手部数据 → XHEY 预训练候选（C7 / C8 / C9）

```
公开数据 ──转换(15fps, 21 点, y 朝上, 只留手在动)──► Kaggle 私有数据集 ──预训练──► XHEY 精调 ──► 留一段评测 + 「手在动」子集
```

| | Stera-10M（C7） | DynamicIntelligence（C8） |
|---|---|---|
| 链接 | https://huggingface.co/datasets/fpvlabs/stera-10m · https://www.fpvlabs.ai/stera | https://huggingface.co/datasets/DynamicIntelligence/humanoid-robots-training-dataset |
| 许可 | Stera-10M License = **CC BY-NC 4.0**（非商用），WiLoR / MANO 部分另按各自许可 | **CC BY-NC 4.0**（非商用） |
| 访问 | **gated**：需 HF 登录并同意条款（会把邮箱/用户名给作者） | 公开 |
| 体量 | 584 段、200 h、全量 1.6 TB；只用的 `annotation.hdf5` 合计约 322 GB（每段 0.2–1.6 GB，内含深度帧） | 151 集（2 集为空），parquet 只有 16 MB；视频 4.3 GB（不用） |
| 设备 | iPhone Pro 头戴，LiDAR + ARKit 6DoF + IMU | Record3D，iPhone 13 Pro 头戴 |
| 手 | WiLoR 21 点（MANO），LiDAR 锚定，相机系 → 世界系 | MediaPipe + 深度，**每手只 3 点**（腕 + 2 指尖），cm，相机系 |
| 世界系 | ARKit 重力对齐 → 统一成 y 朝上（与 iPhone 管线一致） | 数据里的欧拉角约定不明，实测没有一种能让手腕轨迹更平滑 → **只能用相机系翻成 y 朝上，不是严格重力对齐** |
| 转换脚本 | `scripts/stera_fetch.py`（只下 annotation.hdf5 + calibrations，默认取最小的 40 GB）+ `scripts/convert_stera.py` | `scripts/convert_di.py` |
| 转换结果 | 待授权 | 148 集 → **25 集「手在动」**（≥3 cm/s），3,862 帧 @15fps ≈ 4.3 分钟；手检测率全集只有 13.6%，集中在后段任务（摆椅子、放碗） |
| Kaggle | `justinz2010/xhey-stera-pretrain`（待建） | `justinz2010/xhey-di-pretrain`（私有，已上传） |

评测与 PR #47 / Open-AoE 相同：3 段 iPhone LiDAR 留一法，seeds=3，horizon 16 @15fps，并报「手在动」子集（`--move-cm 5`）。

## 致谢 / Credits

- **Stera-10M**：Palanisamy et al., *MobileEgo Anywhere: Open Infrastructure for long horizon egocentric data on commodity hardware*, arXiv:2605.05945, FPV Labs. 同时致谢 WiLoR（Potamias et al., 2024）、MANO（Romero et al., 2017）、EgoBlur、rerun-sdk。
- **DynamicIntelligence**：Dynamic Intelligence, *Humanoid Robot Training Dataset: Egocentric Hand Manipulation Demonstrations*, Hugging Face, 2025.

两份都是 **非商用**：只能用于研究 / 论文；如果 XHEY 将来要商用，必须去掉这两份预训练权重或另取得授权。转换结果不提交进仓库。
