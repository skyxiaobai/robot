# Hugging Face iPhone / smartphone egocentric data (local mirror)

Downloaded for XHEY pretraining ablations. **Do not commit the media** (too large); only converters and docs go in git.

| Local dir | HF repo | License | Hands | Role in XHEY |
|---|---|---|---|---|
| `zeno_lidar/` | [zeno-labs/egostation-iphone-lidar-household-v1](https://huggingface.co/datasets/zeno-labs/egostation-iphone-lidar-household-v1) | **CC BY-NC 4.0** | Metric 3D wrists (LiDAR + MediaPipe), LeRobot | Primary external mix `zeno_lidar` |
| `zeno_phone_sample/` | [zeno-labs/egostation-smartphone-lerobot-sample](https://huggingface.co/datasets/zeno-labs/egostation-smartphone-lerobot-sample) | **CC BY-NC 4.0** | Normalized landmarks (not metres) | WiLoR → `unified/zeno_phone_wilor` (optional) |
| `zeno_raw/` | [zeno-labs/egostation-smartphone-raw-v1](https://huggingface.co/datasets/zeno-labs/egostation-smartphone-raw-v1) | **CC BY-NC 4.0** | 2D pose JSON + open sample videos | WiLoR → mix `zeno_raw` (samples only) |
| `hub_rgb/` | [Hubdata/egocentric-iphone-rgb](https://huggingface.co/datasets/Hubdata/egocentric-iphone-rgb) | **other** (read `LICENSE`) | None | WiLoR → mix `hub_rgb` |
| `hub_rgb_imu/` | [Hubdata/egocentric-iphone-rgb-imu](https://huggingface.co/datasets/Hubdata/egocentric-iphone-rgb-imu) | **other** | None (RGB+IMU; MCAPs skipped locally) | WiLoR → mix `hub_imu` |

## Unified episodes

```bash
# Metric LiDAR wrists (no WiLoR needed)
python scripts/convert_hf_iphone.py --source zeno_lidar

# RGB videos → WiLoR camera-frame hands
source outputs/hot3d_scale/env.sh
python scripts/wilor_video_to_episodes.py \
  --videos 'data/hf_iphone/hub_rgb/*.mp4' \
  --out data/hf_iphone/unified/hub_rgb --source hub_rgb \
  --stride 2 --max-frames 600
```

Outputs land in `data/hf_iphone/unified/<source>/`.

## Train

```bash
python scripts/train_xhey.py --ablation
# or a single mix:
python scripts/train_xhey.py --mix iphone+zeno_lidar+aug
```

Held-out eval is always Justin's iPhone Pro Max clips (`outputs/iphone_mcap/run_s2`).
