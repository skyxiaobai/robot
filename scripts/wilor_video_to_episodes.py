#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对纯 RGB 视频跑 WiLoR，导出统一 episode JSON（相机系米制近似）。

用于 Hubdata RGB / RGB+IMU，以及 zeno raw、phone 样例视频。
没有 SLAM 时手腕留在各帧相机系；短时域增量会混一点头部运动，适合预训练配比。

用法：
  source outputs/hot3d_scale/env.sh
  python scripts/wilor_video_to_episodes.py \
      --videos 'data/hf_iphone/hub_rgb/*.mp4' \
      --out data/hf_iphone/unified/hub_rgb \
      --stride 2 --max-frames 900 --source hub_rgb
"""
from __future__ import annotations

import argparse
import json
import sys
from glob import glob
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _empty_hand(n):
    return {
        "joints": [[[None] * 3] * 21 for _ in range(n)],
        "wrist_pose": [[None] * 7 for _ in range(n)],
        "confidence": [None] * n,
        "valid": [False] * n,
        "filled": [False] * n,
        "label_status": ["missing"] * n,
    }


def _fill_hand(hand, i, joints_cam, conf_list):
    from headcam.hand_pose import wrist_poses_from_joints
    joints = np.asarray(joints_cam, float).reshape(21, 3)
    if not np.isfinite(joints).all():
        return False
    pose = wrist_poses_from_joints([joints])[0]
    if any(v is None for v in pose):
        return False
    if isinstance(conf_list, (list, tuple, np.ndarray)):
        conf = float(np.mean(np.asarray(conf_list, float)))
    elif conf_list is None:
        conf = 1.0
    else:
        conf = float(conf_list)
    hand["joints"][i] = joints.tolist()
    hand["wrist_pose"][i] = [float(x) for x in pose]
    hand["confidence"][i] = conf
    hand["valid"][i] = True
    hand["label_status"][i] = "ok"
    return True


def process_video(path: Path, out_dir: Path, backend, source: str, stride: int,
                  max_frames: int, min_conf: float, license_note: str):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print("skip open fail", path, flush=True)
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames_rgb, kept_src = [], []
    idx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            frames_rgb.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            kept_src.append(idx)
            if max_frames and len(frames_rgb) >= max_frames:
                break
        idx += 1
    cap.release()
    n = len(frames_rgb)
    if n < 8:
        print("skip too short", path, n, flush=True)
        return None

    hands = {"left": _empty_hand(n), "right": _empty_hand(n)}
    det = 0
    for i, rgb in enumerate(frames_rgb):
        pred = backend.predict(rgb)
        for side in ("left", "right"):
            h = pred.get(side) or {}
            joints = h.get("joints_cam")
            conf = h.get("confidence")
            if joints is None:
                continue
            if isinstance(conf, (list, tuple, np.ndarray)):
                cmean = float(np.mean(conf))
            else:
                cmean = float(conf) if conf is not None else 1.0
            if cmean < min_conf:
                continue
            if _fill_hand(hands[side], i, joints, conf):
                det += 1
        if (i + 1) % 50 == 0 or i + 1 == n:
            print(" ", path.name, i + 1, "/", n, "hand_slots", det, flush=True)

    out_fps = fps / max(stride, 1)
    eid = "%s/%s" % (source, path.stem)
    ep = {
        "schema_version": "1.0",
        "episode_id": eid,
        "source": source,
        "source_path": str(path),
        "video_path": str(path),
        "fps": float(out_fps),
        "coordinate_frame": "camera",
        "image_width": width,
        "image_height": height,
        "num_frames": n,
        "timestamps": [s / fps for s in kept_src],
        "camera_intrinsic": None,
        "camera_poses": None,
        "hands": hands,
        "objects": [],
        "contact": {"source": "none", "left": [], "right": []},
        "grasp": {"source": "none", "left": [], "right": []},
        "events": [],
        "annotation": {"environment": "", "task": path.stem, "subtasks": [], "instructions": []},
        "coverage": {},
        "hand_pose": {"backend": "wilor", "frame": "camera", "stride": stride,
                      "src_frames": total, "kept": n},
        "license_note": license_note,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    # include parent folder(s) so video.mp4 / RGB.mp4 under different dirs do not collide
    parts = []
    for part in list(path.parts)[-3:]:
        if part.lower() in {path.suffix.lstrip(".").lower()}:
            continue
        parts.append(part)
    raw = "__".join(parts) if parts else path.stem
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in raw)[:160]
    out_path = out_dir / (safe + ".json")
    out_path.write_text(json.dumps(ep), encoding="utf-8")
    vf = float(np.mean([hands["left"]["valid"], hands["right"]["valid"]]))
    print("wrote", out_path.name, "frames", n, "valid", round(vf, 3), flush=True)
    return {"id": eid, "path": str(out_path), "frames": n, "valid_frac": vf}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default="hub_rgb")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=900,
                    help="每段最多输出帧数（stride 后）。900≈30s@30fps stride2")
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--license-note", default="")
    a = ap.parse_args()

    paths = []
    for pattern in a.videos:
        p = Path(pattern)
        if p.is_file():
            paths.append(p)
        else:
            paths.extend(Path(x) for x in sorted(glob(pattern)))
    seen, ordered = set(), []
    for p in paths:
        if p.suffix.lower() not in {".mp4", ".mov", ".mkv"}:
            continue
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp); ordered.append(rp)
    paths = ordered
    if not paths:
        raise SystemExit("no videos matched: %s" % a.videos)

    from headcam.hand_pose import WiLoRBackend
    backend = WiLoRBackend()
    out = Path(a.out)
    manifest = []
    for p in paths:
        print("video", p, flush=True)
        row = process_video(p, out, backend, a.source, a.stride, a.max_frames, a.min_conf, a.license_note)
        if row:
            manifest.append(row)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("done", len(manifest), "episodes", flush=True)


if __name__ == "__main__":
    main()
