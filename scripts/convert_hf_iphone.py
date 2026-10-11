#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 Hugging Face 上的 iPhone / 手机第一视角数据转成仓库统一 episode JSON。

输出目录默认：data/hf_iphone/unified/<source>/*.json
（状态/动作约定与 EgoDex / iPhone 管线相同：hands.left/right 的 joints + wrist_pose。）

来源与是否可直接用：
  zeno_lidar   zeno-labs/egostation-iphone-lidar-household-v1
               LeRobot，双手腕米制 3D + valid 旗标（优先，无需 WiLoR）
  zeno_phone   zeno-labs/egostation-smartphone-lerobot-sample
               归一化关键点，不是米制 → 默认跳过；--force-phone 才导出（仅作对照）
  zeno_raw     zeno-labs/egostation-smartphone-raw-v1 开放样例
               pose_keypoints 是像素 2D → 需要 WiLoR（见 scripts/wilor_video_to_episodes.py）
  hub_rgb*     Hubdata RGB / RGB+IMU → 无手标签，走 WiLoR

许可：zeno-* 为 CC BY-NC 4.0（非商用）；Hubdata 为 other，用前请读 LICENSE。

用法：
  python scripts/convert_hf_iphone.py --source zeno_lidar
  python scripts/convert_hf_iphone.py --source all_labeled
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
HF = ROOT / "data" / "hf_iphone"
OUT = HF / "unified"


def rot6d_to_quat_xyzw(r6: np.ndarray) -> list:
    """连续 6D 旋转（旋转矩阵前两列）→ xyzw 四元数。"""
    a1 = np.asarray(r6[:3], float)
    a2 = np.asarray(r6[3:6], float)
    n1 = np.linalg.norm(a1)
    if n1 < 1e-8:
        return [0.0, 0.0, 0.0, 1.0]
    b1 = a1 / n1
    a2 = a2 - float(np.dot(b1, a2)) * b1
    n2 = np.linalg.norm(a2)
    if n2 < 1e-8:
        # 退化：造一个正交轴
        tmp = np.array([1.0, 0.0, 0.0]) if abs(b1[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        a2 = np.cross(b1, tmp)
        n2 = np.linalg.norm(a2)
    b2 = a2 / n2
    b3 = np.cross(b1, b2)
    R = np.stack([b1, b2, b3], axis=1)  # columns
    # Shepperd
    t = float(np.trace(R))
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w], float)
    q /= np.linalg.norm(q) + 1e-12
    if q[3] < 0:
        q = -q
    return [float(v) for v in q]


def _canonical_joints(wrist_xyz, quat_xyzw):
    """只有手腕时：21 点都放在手腕上（相对特征为 0，手腕增量仍可学）。"""
    w = np.asarray(wrist_xyz, float).reshape(3)
    return [w.tolist() for _ in range(21)]


def _empty_hand(n):
    return {
        "joints": [[[None] * 3] * 21 for _ in range(n)],
        "wrist_pose": [[None] * 7 for _ in range(n)],
        "confidence": [None] * n,
        "valid": [False] * n,
        "filled": [False] * n,
        "label_status": ["missing"] * n,
    }


def _hand_from_wrist(n, xyz_list, quat_list, valid_list, conf=1.0):
    hand = _empty_hand(n)
    for i in range(n):
        if not valid_list[i]:
            continue
        xyz = xyz_list[i]
        quat = quat_list[i]
        if xyz is None or quat is None:
            continue
        if not np.isfinite(xyz).all() or not np.isfinite(quat).all():
            continue
        hand["wrist_pose"][i] = [float(xyz[0]), float(xyz[1]), float(xyz[2]),
                                 float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
        hand["joints"][i] = _canonical_joints(xyz, quat)
        hand["confidence"][i] = float(conf)
        hand["valid"][i] = True
        hand["label_status"][i] = "ok"
    return hand


def _make_episode(episode_id, source, fps, n, hands, video_path=None, extra=None):
    ep = {
        "schema_version": "1.0",
        "episode_id": episode_id,
        "source": source,
        "source_path": str(video_path) if video_path else "",
        "video_path": str(video_path) if video_path else "",
        "fps": float(fps),
        "coordinate_frame": "dataset_world",
        "image_width": None,
        "image_height": None,
        "num_frames": int(n),
        "timestamps": [i / float(fps) for i in range(n)],
        "camera_intrinsic": None,
        "camera_poses": None,
        "hands": hands,
        "objects": [],
        "contact": {"source": "none", "left": [], "right": []},
        "grasp": {"source": "none", "left": [], "right": []},
        "events": [],
        "annotation": {"environment": "", "task": "", "subtasks": [], "instructions": []},
        "coverage": {},
        "license_note": extra.get("license_note") if extra else None,
    }
    if extra:
        ep.update({k: v for k, v in extra.items() if k != "license_note"})
        if "license_note" in extra:
            ep["license_note"] = extra["license_note"]
    return ep


def convert_zeno_lidar(root: Path, out: Path, stride: int = 2):
    """米制双手腕 → 统一 episode。默认 30→15 fps（stride=2）。"""
    data = root / "data" / "chunk-000"
    videos = root / "videos" / "chunk-000" / "observation.images.head"
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for pq_path in sorted(data.glob("episode_*.parquet")):
        df = pq.read_table(pq_path).to_pandas()
        idx = int(pq_path.stem.split("_")[-1])
        actions = np.stack(df["action"].to_numpy()).astype(np.float64)
        lv = np.asarray(df["action.left_valid"].tolist()).astype(bool)
        rv = np.asarray(df["action.right_valid"].tolist()).astype(bool)
        # subsample
        actions, lv, rv = actions[::stride], lv[::stride], rv[::stride]
        n = len(actions)
        l_xyz = [actions[i, 0:3] for i in range(n)]
        r_xyz = [actions[i, 10:13] for i in range(n)]
        l_q = [rot6d_to_quat_xyzw(actions[i, 3:9]) for i in range(n)]
        r_q = [rot6d_to_quat_xyzw(actions[i, 13:19]) for i in range(n)]
        # 连续四元数同半球
        for qs in (l_q, r_q):
            for i in range(1, n):
                if np.dot(qs[i][:4], qs[i - 1][:4]) < 0:
                    qs[i] = [-c for c in qs[i]]
        hands = {
            "left": _hand_from_wrist(n, l_xyz, l_q, lv),
            "right": _hand_from_wrist(n, r_xyz, r_q, rv),
        }
        vid = videos / ("episode_%06d.mp4" % idx)
        eid = "zeno_lidar/episode_%06d" % idx
        ep = _make_episode(
            eid, "zeno_lidar", 30.0 / stride, n, hands,
            video_path=str(vid) if vid.exists() else "",
            extra={
                "license_note": "CC BY-NC 4.0 (zeno-labs/egostation-iphone-lidar-household-v1); non-commercial only",
                "hf_repo": "zeno-labs/egostation-iphone-lidar-household-v1",
                "original_fps": 30,
                "stride": stride,
            },
        )
        # image size from README
        ep["image_width"], ep["image_height"] = 1920, 1440
        path = out / ("episode_%06d.json" % idx)
        path.write_text(json.dumps(ep), encoding="utf-8")
        valid = float(np.mean([hands["left"]["valid"], hands["right"]["valid"]]))
        written.append({"id": eid, "frames": n, "valid_frac": valid, "path": str(path)})
        print("zeno_lidar", path.name, "frames", n, "valid", round(valid, 3), flush=True)
    (out / "manifest.json").write_text(json.dumps(written, indent=2), encoding="utf-8")
    return written


def convert_zeno_phone_normalized(root: Path, out: Path, stride: int = 2):
    """归一化关键点（非米制）——仅在 --force-phone 时导出，默认不进训练。"""
    data = root / "data" / "chunk-000"
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for pq_path in sorted(data.glob("episode_*.parquet")):
        df = pq.read_table(pq_path).to_pandas()
        idx = int(pq_path.stem.split("_")[-1])
        st = np.stack(df["observation.state"].to_numpy()).astype(np.float64)[::stride]
        n = len(st)
        # 126 = 21*3 left + 21*3 right，已是「归一化 xyz」；z≈0
        # 不当成米制手腕；把整掌尺度粗暴当成「伪米」（不推荐训练）
        hands = {"left": _empty_hand(n), "right": _empty_hand(n)}
        for side, sl in (("left", slice(0, 63)), ("right", slice(63, 126))):
            for i in range(n):
                pts = st[i, sl].reshape(21, 3)
                if not np.isfinite(pts).all() or np.abs(pts).sum() < 1e-6:
                    continue
                # 伪手腕：用归一化坐标，单位不是米
                w = pts[0]
                hands[side]["joints"][i] = pts.tolist()
                hands[side]["wrist_pose"][i] = [float(w[0]), float(w[1]), float(w[2]), 0, 0, 0, 1]
                hands[side]["confidence"][i] = 0.6
                hands[side]["valid"][i] = True
                hands[side]["label_status"][i] = "normalized_not_metric"
        eid = "zeno_phone/episode_%06d" % idx
        ep = _make_episode(
            eid, "zeno_phone_normalized", 30.0 / stride, n, hands,
            extra={
                "license_note": "CC BY-NC 4.0; landmarks are NORMALIZED (not metres) — do not mix into metric ADE without WiLoR",
                "hf_repo": "zeno-labs/egostation-smartphone-lerobot-sample",
                "usable_for_metric_xhey": False,
            },
        )
        path = out / ("episode_%06d.json" % idx)
        path.write_text(json.dumps(ep), encoding="utf-8")
        written.append({"id": eid, "frames": n, "usable_metric": False, "path": str(path)})
        print("zeno_phone(normalized)", path.name, "frames", n, flush=True)
    (out / "manifest.json").write_text(json.dumps(written, indent=2), encoding="utf-8")
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-root", default=str(HF))
    ap.add_argument("--out-root", default=str(OUT))
    ap.add_argument("--source", default="zeno_lidar",
                    choices=["zeno_lidar", "zeno_phone", "all_labeled"])
    ap.add_argument("--stride", type=int, default=2, help="30fps→15fps 用 2")
    ap.add_argument("--force-phone", action="store_true",
                    help="导出 zeno_phone 归一化关键点（默认跳过：不是米制）")
    a = ap.parse_args()
    hf, out_root = Path(a.hf_root), Path(a.out_root)
    results = {}
    if a.source in ("zeno_lidar", "all_labeled"):
        results["zeno_lidar"] = convert_zeno_lidar(hf / "zeno_lidar", out_root / "zeno_lidar", a.stride)
    if a.source in ("zeno_phone", "all_labeled") and a.force_phone:
        results["zeno_phone"] = convert_zeno_phone_normalized(
            hf / "zeno_phone_sample", out_root / "zeno_phone", a.stride)
    elif a.source == "zeno_phone" and not a.force_phone:
        raise SystemExit("zeno_phone 关键点不是米制；若只要对照请加 --force-phone，否则请对该视频跑 WiLoR")
    summary = {k: {"episodes": len(v), "frames": sum(x["frames"] for x in v)} for k, v in results.items()}
    (out_root / "convert_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("summary", summary, flush=True)


if __name__ == "__main__":
    main()
