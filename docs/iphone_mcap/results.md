# iPhone mcap（带 LiDAR 深度）管线结果

## 结论先说
1. **三段 mcap 都下好了**，里面**有深度**（不是只有 RGB）。
2. 原来仓库只会吃 Record3D 的 `.r3d`，**不会吃 xhey 的 mcap** → 加了 `scripts/iphone/mcap_adapter.py`。
3. WiLoR 单目算出来的手腕大约在 **0.4 m**，LiDAR 同一位置大约 **0.9–1.0 m**。单目手型大小是对的，错的是平移深度。
4. 修好融合后：用 **LiDAR 定手腕深度 + WiLoR 手型（不缩放）**，标注覆盖从 **0% → 约 53%**（三段共 540 帧）。

## 数据
| 片段 | 大小 | 会话目录（前 180 帧） |
|---|---|---|
| 2026-08-22_18-21 | 1.63 GB | `outputs/iphone_mcap/sessions/2026-08-22_18-21_s1` |
| 2026-08-24_21-18 | 1.58 GB | `…/2026-08-24_21-18_s1` |
| 2026-09-05_17-16 | 2.51 GB | `…/2026-09-05_17-16_s1` |

深度：`16UC1` 大端 uint16 / `depth_scale=10000` → 米，256×192；confidence 0/1/2；与 RGB（1920×1440）同视场（焦距比≈分辨率比）。

## 融合怎么改的
旧逻辑：把 WiLoR 3D 去贴「每个关节各自的 LiDAR 反投影」。单目整体偏近时，相似变换会把**手掌拉到 18–23 cm**，几乎全部判成 `fit` 失败。

新逻辑（`scripts/headcam/rgbd_pipeline.py` · `depth_hand`）：
1. 在手的 2D 凸包里取 LiDAR **近层**深度；
2. 沿 RGB 射线放手腕；
3. WiLoR 关节减去手腕后**平移**过去（手掌保持约 9.5 cm）。

单目深度 / LiDAR 深度中位数比约 **2.3×**（0.4 m → 0.9 m）。

## 数字（修复后，`--no-export`）
| 片段 | 左 ok% | 右 ok% | 标注覆盖（QC） |
|---|---|---|---|
| 08-22 | 80.6 | 63.3 | 52.8% |
| 08-24 | 84.4 | 55.6 | 60.6% |
| 09-05 | 46.1 | 20.6 | 16.1%（运动模糊多） |
| **合计** | | | **约 53%**（285/540） |

修复前三段覆盖均为 **0%**。

## 还没做完 / 缺口
- mcap 里没有 depth↔color 外参 TF，目前假定同光轴、同视场按分辨率缩放；
- H265 不能直接跳帧抽包（会缺参考帧），适配器要连续解码；
- onboard `ego_hand_pose` 只有归一化 2D，不能当 3D 真值；
- 09-05 段仍偏弱；LeRobot 导出需整数 fps（已写进 metadata）。

## 复现
```bash
source outputs/hot3d_scale/env.sh
python scripts/iphone/mcap_adapter.py data/iphone_mcap/XXX.mcap \
  --out outputs/iphone_mcap/sessions/XXX_s1 --stride 1 --max-frames 180
python scripts/run_stereo_pipeline.py \
  --iphone outputs/iphone_mcap/sessions/*_s1 \
  --out outputs/iphone_mcap/run_all --backend wilor --no-export
```
