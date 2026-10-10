# -*- coding: utf-8 -*-
"""HOT3D-Clips（Quest3 / Aria）→ 自有头戴会话目录（docs/headcam_data_spec.md §7.7）。

转完之后，HOT3D 和你自己的设备走同一个入口 ``scripts/run_stereo_pipeline.py``。

做的事：

1. 两路 SLAM 黑白鱼眼图（1201-1 左 / 1201-2 右）去畸变成针孔图。Quest3 的 SLAM 相机是侧放的，
   再转 -90° 让手是竖直的（WiLoR 对竖直手更稳，见 ``eval_hot3d_stereo.pinhole_of``）。
   写成 ``stereo/left.mp4`` 和 ``stereo/right.mp4``（近无损 H.264）。
2. ``calib.yaml``：两个虚拟针孔相机的内参和 ``T_right_left``（简单 YAML，``p_right = R p_left + T``）。
   去畸变后畸变系数为 0。
3. ``slam.tum``：HOT3D 自带的每帧左目世界位姿（动捕 + 设备定位），当作“SLAM 轨迹”。
4. ``timestamps.csv``：左目曝光时间（秒）。``metadata.json``：fps、图像宽高等。
5. ``gt/hands_gt.json``（可选，需要 MANO）：每帧双手 21 点世界坐标真值，只用来评测，
   不是设备规格的一部分，管线主体不读它。

HOT3D-Clips 没有 IMU 和彩色图，所以不写 ``imu.csv`` / ``rgb.mp4``。
依赖：``hand_tracking_toolkit``（相机模型、去畸变、MANO 解码）、imageio-ffmpeg；真值还要 smplx。
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

STREAM_LEFT, STREAM_RIGHT = "1201-1", "1201-2"


def _scripts_dir():
    return str(Path(__file__).resolve().parents[1])


def rotmat_to_quat_xyzw(rotation):
    if _scripts_dir() not in sys.path:
        sys.path.insert(0, _scripts_dir())
    from egodata.schema import rotmat_to_quat_xyzw as convert
    return convert(rotation)


def relative_extrinsics(T_world_left, T_world_right):
    """两个相机的世界位姿 → OpenCV 约定 ``p_right = R @ p_left + T``。"""
    T_right_left = np.linalg.inv(np.asarray(T_world_right, dtype=float)) @ np.asarray(T_world_left, dtype=float)
    return T_right_left[:3, :3], T_right_left[:3, 3]


def write_tum(path, timestamps, poses):
    lines = ["# timestamp tx ty tz qx qy qz qw  (T_world_leftcam)"]
    for stamp, pose in zip(timestamps, poses):
        pose = np.asarray(pose, dtype=float)
        q = rotmat_to_quat_xyzw(pose[:3, :3])
        lines.append("%.9f %s" % (stamp, " ".join("%.9f" % v for v in list(pose[:3, 3]) + list(q))))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


class _VideoWriter(object):
    """用系统 ffmpeg 写近无损 H.264（crf 默认 10）。帧率固定按 30 写，真实时间以 timestamps.csv 为准。"""

    def __init__(self, path, fps=30, crf=10):
        self.path, self.fps, self.crf, self.proc = str(path), fps, crf, None

    def write(self, frame):
        import subprocess
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        if self.proc is None:
            h, w = frame.shape[:2]
            self.proc = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", "%dx%d" % (w, h), "-r", str(self.fps), "-i", "-", "-c:v", "libx264",
                 "-crf", str(self.crf), "-pix_fmt", "yuv420p", self.path],
                stdin=subprocess.PIPE)
        self.proc.stdin.write(frame.tobytes())

    def close(self):
        if self.proc is not None:
            self.proc.stdin.close()
            if self.proc.wait() != 0:
                raise RuntimeError("ffmpeg 写视频失败：%s" % self.path)


def clip_keys(clip):
    return sorted(os.path.basename(p).split(".")[0] for p in glob.glob(os.path.join(clip, "*.info.json")))


def convert_clip(clip, out_session, mano_dir=None, max_frames=None, crf=10):
    """把一个 HOT3D clip 目录写成会话目录。返回摘要字典。"""
    sys.path.insert(0, _scripts_dir())
    import imageio.v2 as imageio
    from eval_hot3d_stereo import GTHands, K_of, load_frame_cameras, pinhole_of, undistort

    clip = str(clip)
    out = Path(out_session)
    (out / "stereo").mkdir(parents=True, exist_ok=True)
    keys = clip_keys(clip)
    if max_frames:
        keys = keys[: int(max_frames)]
    if not keys:
        raise ValueError("HOT3D clip 里没有帧：%s" % clip)
    first_info = json.load(open(os.path.join(clip, keys[0] + ".info.json")))
    device = first_info["device"]
    rotate = -90 if device == "Quest3" else 0

    writers = {s: _VideoWriter(out / "stereo" / name, crf=crf)
               for s, name in ((STREAM_LEFT, "left.mp4"), (STREAM_RIGHT, "right.mp4"))}
    gt = GTHands(mano_dir) if mano_dir else None
    stamps, stamps_r, poses, gt_frames, rel = [], [], [], [], []
    pin0 = None
    for key in keys:
        cams = load_frame_cameras(clip, key)
        pins = {s: pinhole_of(cams[s], 1.0, rotate) for s in (STREAM_LEFT, STREAM_RIGHT)}
        if pin0 is None:
            pin0 = pins
        for s in (STREAM_LEFT, STREAM_RIGHT):
            img = imageio.imread(os.path.join(clip, "%s.image_%s.jpg" % (key, s)))
            if img.ndim == 2:
                img = np.stack([img] * 3, -1)
            und = undistort(img, cams[s], pins[s])
            writers[s].write(np.ascontiguousarray(und[..., :3]).astype(np.uint8))
        info = json.load(open(os.path.join(clip, key + ".info.json")))
        stamps.append(info["image_timestamps_ns"][STREAM_LEFT] * 1e-9)
        stamps_r.append(info["image_timestamps_ns"][STREAM_RIGHT] * 1e-9)
        poses.append(np.asarray(pins[STREAM_LEFT].T_world_from_eye, dtype=float))
        rel.append(relative_extrinsics(pins[STREAM_LEFT].T_world_from_eye, pins[STREAM_RIGHT].T_world_from_eye))
        if gt is not None:
            joints, _ = gt.joints(clip, key)
            gt_frames.append({side: (joints[side].tolist() if side in joints else None) for side in ("left", "right")})
    for w in writers.values():
        w.close()

    R, T = rel[0]
    drift_t = max(float(np.linalg.norm(t - T)) for _, t in rel)
    left, right = pin0[STREAM_LEFT], pin0[STREAM_RIGHT]
    Kl, Kr = K_of(left), K_of(right)
    T_rl = np.eye(4)
    T_rl[:3, :3], T_rl[:3, 3] = R, T
    lines = [
        "# HOT3D %s SLAM 左右目去畸变后的虚拟针孔相机（hot3d_adapter.py 生成）" % device,
        "image_width: %d" % left.width,
        "image_height: %d" % left.height,
        "left:",
        "  intrinsics: [%.10g, %.10g, %.10g, %.10g]" % (Kl[0, 0], Kl[1, 1], Kl[0, 2], Kl[1, 2]),
        "  dist_coeffs: [0, 0, 0, 0, 0]",
        "right:",
        "  intrinsics: [%.10g, %.10g, %.10g, %.10g]" % (Kr[0, 0], Kr[1, 1], Kr[0, 2], Kr[1, 2]),
        "  dist_coeffs: [0, 0, 0, 0, 0]",
        "T_right_left:",
    ] + ["  - [%s]" % ", ".join("%.10g" % v for v in row) for row in T_rl]
    (out / "calib.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_tum(out / "slam.tum", stamps, poses)
    (out / "timestamps.csv").write_text(
        "frame_index,timestamp_s\n" + "".join("%d,%.9f\n" % (i, t) for i, t in enumerate(stamps)), encoding="utf-8")
    (out / "stereo" / "timestamps_lr.csv").write_text(
        "frame_index,left_s,right_s\n" + "".join(
            "%d,%.9f,%.9f\n" % (i, a, b) for i, (a, b) in enumerate(zip(stamps, stamps_r))), encoding="utf-8")
    dt = float(np.median(np.diff(stamps))) if len(stamps) > 1 else 1 / 30.0
    name = os.path.basename(clip.rstrip("/"))
    meta = {
        "episode_id": "hot3d/%s/%s" % (device.lower(), name),
        "fps": float(round(1.0 / dt)),
        "task": "hot3d_object_manipulation",
        "instruction": "",
        "environment": "lab",
        "image_width": int(left.width),
        "image_height": int(left.height),
        "objects": [],
        "verbs": [],
        "source": {"dataset": "HOT3D-Clips", "clip": clip, "device": device,
                   "sequence_id": first_info.get("sequence_id"), "participant_id": first_info.get("participant_id")},
    }
    (out / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if gt is not None:
        (out / "gt").mkdir(exist_ok=True)
        (out / "gt" / "hands_gt.json").write_text(json.dumps(
            {"frame": "world", "order": "mediapipe21", "unit": "m", "frames": gt_frames}), encoding="utf-8")
    return {"session": str(out), "frames": len(keys), "device": device, "baseline_m": float(np.linalg.norm(T)),
            "extrinsic_drift_m": drift_t, "fps": meta["fps"], "gt": gt is not None}
