#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Open-AoE（inclusionAI/OpenAoE-2000h）→ XHEY 统一 episode JSON（预训练用，无画面）。

只读 hands.npz + camera_traj.npz（MANO 参数 + 米制 SLAM 相机轨迹），不用视频。
- MANO 正向运动学（smplx，需本地 MANO_RIGHT/LEFT.pkl，遵守 MANO 许可）→ 21 点，
  按 WiLoR/OpenPose 顺序（腕、拇指 4、食指 4…，指尖取顶点）。
- 世界系（SLAM，米），30 fps → 15 fps（与 XHEY iPhone 一致）。
- 「手在动」过滤：切成 --seg-s 秒一段，1 秒内手腕位移中位数 > --min-move-cm 才保留。

数据来源与许可：Open-AoE Dataset License（需署名 Open-AoE / OpenAoE-2000h，arXiv:2607.14183）；
hands.npz / camera_traj.npz 另受 MANO 许可约束。转换结果不要提交进仓库。

用法：
  python scripts/convert_openaoe.py --root data/openaoe/nano --out data/openaoe/unified/nano \
      --mano-dir /path/to/mano_parent   # 该目录下有 mano/MANO_RIGHT.pkl
"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from egodata.schema import empty_interaction  # noqa: E402
from headcam.hand_pose import wrist_poses_from_joints  # noqa: E402
from common import frames as FR  # noqa: E402

TIPS = {"right": [745, 317, 444, 556, 673], "left": [745, 317, 444, 556, 673]}
FLIP = FR.CV_TO_YUP
MANO2OP = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]


def mano_joints(models, side, d, h, idx, frame="world"):
    import torch
    m = models[side]
    with torch.no_grad():
        rk, tk = ("pred_rot_cam", "pred_trans_cam") if frame == "camera" else ("pred_rot", "pred_trans")
        o = m(global_orient=torch.tensor(d[rk][h, idx]).float(),
              hand_pose=torch.tensor(d["pred_hand_pose"][h, idx]).float(),
              betas=torch.tensor(d["pred_betas"][h, idx]).float(),
              transl=torch.tensor(d[tk][h, idx]).float())
    j = torch.cat([o.joints, o.vertices[:, TIPS[side]]], 1).numpy()
    return j[:, MANO2OP]


def convert(npz, out, models, seg_s, min_move_cm, stats, frame="world"):
    d = np.load(npz); c = np.load(str(npz).replace("hands.npz", "camera_traj.npz"))
    T = d["pred_valid"].shape[1]
    idx = np.arange(0, T, 2)  # 30 → 15 fps
    n = len(idx)
    hands, wpos = {}, {}
    for h, side in enumerate(("left", "right")):
        valid = d["pred_valid"][h, idx] > 0.5
        J = mano_joints(models, side, d, h, idx, frame)
        if frame == "world":
            J = J @ FLIP.T  # 首帧相机系（y 朝下）→ 近似 ARKit 世界（y 朝上，与 iPhone 一致）
        J[~valid] = np.nan
        poses = wrist_poses_from_joints(list(J))
        hands[side] = {"joints": [jj.tolist() if v else [[None] * 3] * 21 for jj, v in zip(J, valid)],
                       "wrist_pose": [p if v else [None] * 7 for p, v in zip(poses, valid)],
                       "confidence": [1.0 if v else None for v in valid], "valid": valid.tolist(),
                       "filled": [False] * n, "label_status": ["ok" if v else "missing" for v in valid]}
        wpos[side] = np.where(valid[:, None], J[:, 0], np.nan)
    seg = int(seg_s * 15); name = Path(npz).parents[2].name; kept = 0
    for s0 in range(0, n - 17, seg):
        s1 = min(n, s0 + seg)
        mv = []
        for side in wpos:
            w = wpos[side][s0:s1]
            if len(w) > 15:
                mv.append(np.linalg.norm(w[15:] - w[:-15], axis=1))
        mv = np.concatenate(mv) if mv else np.array([])
        med = float(np.nanmedian(mv)) * 100 if np.isfinite(mv).any() else 0.0
        stats["segments"] += 1
        if med < min_move_cm:
            continue
        sl = slice(s0, s1); m = s1 - s0
        ep = {"schema_version": "1.0", "episode_id": "openaoe/%s_%04d" % (name, s0), "source": "openaoe",
              "source_path": str(npz), "video_path": None, "fps": 15.0, "coordinate_frame": "camera" if frame == "camera" else "slam_world",
              "num_frames": m, "timestamps": (np.arange(m) / 15.0).tolist(),
              "camera_intrinsic": c["intrinsic"].tolist(),
              "camera_poses": [np.eye(4).tolist()] * m if frame == "camera" else [(np.block([[FLIP, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]]) @ P).tolist()
                               for P in c["cam_c2w"][idx[sl]]],
              "hands": {k: {f: v[f][sl] for f in v} for k, v in hands.items()},
              "annotation": {"task": {"name": name, "instruction": ""}, "subtasks": [], "instructions": [],
                             "environment": {"name": "", "detail": "", "source": "missing"}},
              "hand_pose": {"backend": "openaoe_mano", "frame": frame, "metric_translation": True,
                            "move_median_cm_per_s": med},
              "license_note": "Open-AoE Dataset License (attribution: Open-AoE / OpenAoE-2000h, arXiv:2607.14183); MANO-derived"}
        ep.update(empty_interaction(m))
        # camera：MANO 相机系（光学系）；world：首帧相机系翻 y 朝上，没有重力 → 记成 world（上轴未核实）
        FR.tag(ep, FR.CAMERA if frame == "camera" else FR.WORLD, gravity_aligned=False, source_frame="openaoe_" + frame)
        if not FR.sanity_check(ep)["ok"]:
            stats.setdefault("frame_flagged", []).append(ep["episode_id"])
        (out / ("%s_%04d.json" % (name, s0))).write_text(json.dumps(ep))
        kept += 1; stats["kept"] += 1; stats["frames"] += m
    return kept


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--mano-dir", required=True, help="其下有 mano/MANO_RIGHT.pkl、MANO_LEFT.pkl")
    ap.add_argument("--seg-s", type=float, default=20.0)
    ap.add_argument("--min-move-cm", type=float, default=3.0, help="1 秒手腕位移中位数阈值（cm）")
    ap.add_argument("--frame", default="world", choices=["world", "camera"],
                    help="world=SLAM 世界系（y 翻转近似 ARKit）；camera=每帧相机系（同 xhey_rgb，避开重力方向问题）")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    import smplx
    models = {s: smplx.create(a.mano_dir, model_type="mano", is_rhand=(s == "right"), use_pca=False,
                              flat_hand_mean=True, batch_size=1) for s in ("left", "right")}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(Path(a.root) / "**/hands.npz"), recursive=True))[: a.limit]
    stats = {"clips": 0, "segments": 0, "kept": 0, "frames": 0}
    for f in files:
        try:
            convert(f, out, models, a.seg_s, a.min_move_cm, stats, a.frame); stats["clips"] += 1
        except Exception as e:  # noqa: BLE001
            print("skip", f, repr(e)[:200], flush=True)
    (out / "convert_summary.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
