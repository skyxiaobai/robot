#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DynamicIntelligence/humanoid-robots-training-dataset（Record3D, iPhone 13 Pro）→ XHEY 统一 episode JSON。

只读 data/*.parquet（约 16 MB），不下视频。
- 每只手只有 3 个点（cm, 相机系 x右 y下 z前）：左手 腕/拇指尖/食指尖，右手 腕/食指尖/中指尖。
  放进 WiLoR/OpenPose 21 点顺序：0 腕，4 拇指尖，8 食指尖，12 中指尖；其余关节填成手腕
  （局部偏移 = 0）。XHEY 只预测手腕，所以主要信号是手腕轨迹。
- 30 fps → 15 fps，cm → m。
- 坐标系：数据集的 observation.camera_pose 是欧拉角，约定未写明；实测 48 种约定都没有让
  世界系手腕轨迹比相机系更平滑（中位帧间位移 1.21 cm vs 相机系 0.93 cm），说明姿态不可靠。
  因此用 **相机系翻成 y 朝上**（x右 y上 z后，和 XHEY 头戴近似一致），**不是严格重力对齐**，
  coordinate_frame 标为 camera_yup。
- 「手在动」：1 秒内手腕位移中位数 > --min-move-cm 的整段保留。

许可：CC BY-NC 4.0（非商用，需署名 Dynamic Intelligence）。转换结果不要提交进仓库。
用法：python scripts/convert_di.py --root data/di --out data/di/unified
"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from egodata.schema import empty_interaction  # noqa: E402
from headcam.hand_pose import wrist_poses_from_joints  # noqa: E402

FLIP = np.diag([1.0, -1.0, -1.0])  # 相机系 y下 z前 → y上 z后
SLOTS = {"left": (0, 4, 8), "right": (0, 8, 12)}  # 3 个点在 21 点里的位置


def to21(p9, side):
    J = np.repeat(p9[:, None, 0:3], 21, axis=1).copy()
    for k, slot in enumerate(SLOTS[side]):
        J[:, slot] = p9[:, 3 * k:3 * k + 3]
    return J


def convert(pq, out, min_move_cm, stats):
    import pandas as pd
    df = pd.read_parquet(pq)
    if len(df) < 40:
        return 0
    idx = np.arange(0, len(df), 2); n = len(idx)
    hands, wpos = {}, {}
    for side, col in (("left", "observation.left_hand"), ("right", "observation.right_hand")):
        P = np.stack(df[col].values)[idx].astype(np.float64)
        valid = (np.abs(P).sum(1) > 0) & (P[:, 2] > 1.0)
        J = to21(P / 100.0, side) @ FLIP.T
        J[~valid] = np.nan
        poses = wrist_poses_from_joints(list(J))
        # 只有 3 个点时常算不出手腕朝向：用单位四元数占位（XHEY 只预测手腕位置）
        poses = [p if p[3] is not None or p[0] is None else p[:3] + [0.0, 0.0, 0.0, 1.0] for p in poses]
        hands[side] = {"joints": [jj.tolist() if v else [[None] * 3] * 21 for jj, v in zip(J, valid)],
                       "wrist_pose": [p if v else [None] * 7 for p, v in zip(poses, valid)],
                       "confidence": [1.0 if v else None for v in valid], "valid": valid.tolist(),
                       "filled": [False] * n, "label_status": ["ok" if v else "missing" for v in valid]}
        wpos[side] = np.where(valid[:, None], J[:, 0], np.nan)
    mv = np.concatenate([np.linalg.norm(w[15:] - w[:-15], axis=1) for w in wpos.values() if len(w) > 15])
    stats["episodes"] += 1; stats["frames_all"] += n
    med = float(np.nanmedian(mv)) * 100 if np.isfinite(mv).any() else 0.0
    nvalid = int(sum(np.isfinite(w[:, 0]).sum() for w in wpos.values()))
    if med < min_move_cm or nvalid < 20:
        return 0
    ei = int(df["episode_index"].iloc[0]); instr = str(df["language_instruction"].iloc[0])
    ep = {"schema_version": "1.0", "episode_id": "di/%06d" % ei, "source": "dynamic_intelligence",
          "source_path": str(pq), "video_path": None, "fps": 15.0, "coordinate_frame": "camera_yup",
          "num_frames": n, "timestamps": (np.arange(n) / 15.0).tolist(), "camera_intrinsic": None,
          "camera_poses": [np.eye(4).tolist()] * n, "hands": hands,
          "annotation": {"task": {"name": instr, "instruction": instr}, "subtasks": [], "instructions": [],
                         "environment": {"name": "", "detail": "", "source": "missing"}},
          "hand_pose": {"backend": "mediapipe_depth", "frame": "camera_yup", "metric_translation": True,
                        "joints_present": "wrist+2 fingertips", "wrist_quat": "identity placeholder when degenerate", "move_median_cm_per_s": med},
          "license_note": "CC BY-NC 4.0, Dynamic Intelligence humanoid-robots-training-dataset (non-commercial)"}
    ep.update(empty_interaction(n))
    (out / ("di_%06d.json" % ei)).write_text(json.dumps(ep))
    stats["kept"] += 1; stats["frames"] += n; stats["hand_frames"] += nvalid
    return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--min-move-cm", type=float, default=3.0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stats = {"episodes": 0, "frames_all": 0, "kept": 0, "frames": 0, "hand_frames": 0}
    for f in sorted(glob.glob(str(Path(a.root) / "data/**/*.parquet"), recursive=True)):
        try:
            convert(f, out, a.min_move_cm, stats)
        except Exception as e:  # noqa: BLE001
            print("skip", f, repr(e)[:200], flush=True)
    (out / "convert_summary.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
