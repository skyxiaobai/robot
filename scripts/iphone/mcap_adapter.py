# -*- coding: utf-8 -*-
"""xhey iPhone mcap（带 LiDAR 深度）→ iPhone 会话目录（同 record3d_adapter 输出格式）。

mcap 里是 protobuf（FileDescriptorSet 嵌在 schema 里）：
  /cam_head_l/color/image_compressed_h265  H265
  /cam_head_l/depth/image_raw              16UC1，大端 uint16 / depth_scale → 米
  /cam_head_l/depth/confidence             8UC1，0/1/2
  /cam_head_l/color|depth/camera_info
  /embosa_tf                               arkit_world → color optical frame
  /cam_head_l/ego_hand_pose                可选：onboard 2D 21 点（无深度）

用法::
    python scripts/iphone/mcap_adapter.py data/iphone_mcap/foo.mcap --out outputs/iphone_mcap/sessions/foo \\
        --stride 5 --max-frames 300
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

TOPICS = {
    "color": "/cam_head_l/color/image_compressed_h265",
    "depth": "/cam_head_l/depth/image_raw",
    "conf": "/cam_head_l/depth/confidence",
    "color_info": "/cam_head_l/color/camera_info",
    "depth_info": "/cam_head_l/depth/camera_info",
    "tf": "/embosa_tf",
    "hands": "/cam_head_l/ego_hand_pose",
}


def _pool_from_mcap(path):
    from mcap.reader import make_reader
    from google.protobuf import descriptor_pb2, descriptor_pool

    with open(path, "rb") as f:
        summary = make_reader(f).get_summary()
    schema = next(iter(summary.schemas.values()))
    fds = descriptor_pb2.FileDescriptorSet()
    fds.ParseFromString(schema.data)
    pool = descriptor_pool.DescriptorPool()
    for fd in fds.file:
        pool.Add(fd)
    return pool


def _cls(pool, name):
    from google.protobuf import message_factory

    return message_factory.GetMessageClass(pool.FindMessageTypeByName(name))


def _ts_sec(header):
    t = header.timestamp
    return float(t.sec) + float(t.nanosec) * 1e-9


def _quat_xyzw_from_proto(q):
    return [float(q.x), float(q.y), float(q.z), float(q.w)]


def _K_from_info(info):
    k = list(info.k)
    return np.array([[k[0], k[1], k[2]], [k[3], k[4], k[5]], [k[6], k[7], k[8]]], dtype=float)


def decode_depth(data, depth_scale, height, width):
    """大端 uint16 / depth_scale → 米。"""
    arr = np.frombuffer(data, dtype=">u2")
    if arr.size != height * width:
        raise ValueError("depth 字节数不对：%d vs %dx%d" % (len(data), height, width))
    scale = float(depth_scale) if depth_scale else 10000.0
    return (arr.astype(np.float32) / scale).reshape(height, width)


def decode_conf(data, height, width):
    arr = np.frombuffer(data, dtype=np.uint8)
    if arr.size != height * width:
        raise ValueError("conf 字节数不对：%d vs %dx%d" % (len(data), height, width))
    return arr.reshape(height, width)


def _nearest_pose(poses, t):
    if not poses:
        return None
    # poses: list of (t, tx,ty,tz,qx,qy,qz,qw)
    i = min(range(len(poses)), key=lambda j: abs(poses[j][0] - t))
    if abs(poses[i][0] - t) > 0.05 or not np.isfinite(poses[i]).all() or np.linalg.norm(poses[i][4:8]) < 1e-8:
        return None
    return poses[i]


def _write_selected_video(color_pkts, selected, path, width, height, fps, ffmpeg):
    """先解码完整 HEVC 参考链，再按 RGB 帧号抽帧，输出与标签相同的 CFR。"""
    wanted = {row[0] for row in selected}
    last = max(wanted)
    frame_bytes = int(width) * int(height) * 3
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "color.h265"
        with source.open("wb") as fh:
            for _, blob, _ in color_pkts:
                fh.write(blob)
        decode = [ffmpeg, "-loglevel", "error", "-f", "hevc", "-i", str(source),
                  "-vsync", "0", "-frames:v", str(last + 1), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
        encode = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                  "-s", "%dx%d" % (width, height), "-r", str(fps), "-i", "pipe:0",
                  "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(path)]
        with tempfile.TemporaryFile() as decode_err, tempfile.TemporaryFile() as encode_err:
            with subprocess.Popen(decode, stdout=subprocess.PIPE, stderr=decode_err) as reader, \
                    subprocess.Popen(encode, stdin=subprocess.PIPE, stderr=encode_err) as writer:
                try:
                    for index in range(last + 1):
                        frame = reader.stdout.read(frame_bytes)
                        if len(frame) != frame_bytes:
                            raise ValueError("HEVC 解码在 RGB 帧 %d 提前结束，无法和深度对齐" % index)
                        if index in wanted:
                            writer.stdin.write(frame)
                    writer.stdin.close()
                    reader.stdout.close()
                    for proc, command, err in ((reader, decode, decode_err), (writer, encode, encode_err)):
                        if proc.wait() != 0:
                            err.seek(0)
                            raise subprocess.CalledProcessError(proc.returncode, command, stderr=err.read().decode())
                except BaseException:
                    for proc in (reader, writer):
                        if proc.poll() is None:
                            proc.terminate()
                    raise


def convert_mcap(path, out, stride=1, max_frames=None, ffmpeg="ffmpeg"):
    from mcap.reader import make_reader

    path = Path(path)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pool = _pool_from_mcap(path)
    Compressed = _cls(pool, "xhey.sensor_proto.CompressedImage")
    CamInfo = _cls(pool, "xhey.sensor_proto.CameraInfo")
    TF = _cls(pool, "xhey.tf2_proto.TF2Message")
    Hand = _cls(pool, "xhey.sensor_proto.EgoHandPoseFrame")

    color_info = depth_info = None
    poses = []  # (t, tx,ty,tz,qx,qy,qz,qw) T_world_cam, OpenCV optical
    depth_by_t = {}
    conf_by_t = {}
    color_pkts = []  # (t, bytes)
    hand_frames = []

    want = set(TOPICS.values())
    with open(path, "rb") as f:
        for schema, channel, message in make_reader(f).iter_messages(topics=list(want)):
            topic = channel.topic
            data = message.data
            if topic == TOPICS["color_info"] and color_info is None:
                color_info = CamInfo(); color_info.ParseFromString(data)
            elif topic == TOPICS["depth_info"] and depth_info is None:
                depth_info = CamInfo(); depth_info.ParseFromString(data)
            elif topic == TOPICS["tf"]:
                msg = TF(); msg.ParseFromString(data)
                for tr in msg.transforms:
                    if tr.child_frame_id.endswith("color_optical_frame") or "color_optical" in tr.child_frame_id:
                        t = _ts_sec(tr.header)
                        T = tr.transform
                        # empty translation → zeros
                        tx = float(getattr(T.translation, "x", 0.0) or 0.0)
                        ty = float(getattr(T.translation, "y", 0.0) or 0.0)
                        tz = float(getattr(T.translation, "z", 0.0) or 0.0)
                        q = _quat_xyzw_from_proto(T.rotation)
                        poses.append((t, tx, ty, tz, q[0], q[1], q[2], q[3]))
            elif topic == TOPICS["depth"]:
                msg = Compressed(); msg.ParseFromString(data)
                t = _ts_sec(msg.header)
                depth_by_t[t] = (msg.data, msg.depth_scale, msg.format)
            elif topic == TOPICS["conf"]:
                msg = Compressed(); msg.ParseFromString(data)
                t = _ts_sec(msg.header)
                conf_by_t[t] = msg.data
            elif topic == TOPICS["color"]:
                msg = Compressed(); msg.ParseFromString(data)
                color_pkts.append((_ts_sec(msg.header), msg.data, msg.format))
            elif topic == TOPICS["hands"]:
                msg = Hand(); msg.ParseFromString(data)
                hand_frames.append(msg)

    if color_info is None or depth_info is None:
        raise RuntimeError("mcap 缺 camera_info")
    if not color_pkts:
        raise RuntimeError("mcap 缺彩色帧")
    if not depth_by_t:
        raise RuntimeError("mcap 缺深度帧")

    w, h = int(color_info.width), int(color_info.height)
    dw, dh = int(depth_info.width), int(depth_info.height)
    K = _K_from_info(color_info)
    depth_times = np.array(sorted(depth_by_t.keys()), dtype=float)

    # 对齐：按彩色时间戳找最近深度；再 stride / max_frames
    selected = []
    for i, (t, blob, fmt) in enumerate(color_pkts):
        j = int(np.argmin(np.abs(depth_times - t)))
        td = float(depth_times[j])
        if abs(td - t) > 0.05:
            continue
        selected.append((i, t, td, blob))
    selected = selected[:: max(1, int(stride))]
    if max_frames:
        selected = selected[: int(max_frames)]
    if not selected:
        raise RuntimeError("对齐后没有帧（彩色/深度时间戳对不上）")

    stamps = [t for _, t, _, _ in selected]
    dt = np.diff(stamps)
    if not np.isfinite(stamps).all() or np.any(dt <= 0):
        raise ValueError("RGB 时间戳必须有限且严格递增")
    fps = float(1.0 / np.median(dt)) if len(dt) else 30.0
    pose_rows = [_nearest_pose(poses, t) for t in stamps]
    missing = [i for i, pose in enumerate(pose_rows) if pose is None]
    if missing:
        raise ValueError("缺少 %d 帧的相机位姿（第一帧号 %d），无法生成世界系监督；请检查 /embosa_tf" % (
            len(missing), missing[0]))
    _write_selected_video(color_pkts, selected, out / "rgb.mp4", w, h, fps, ffmpeg)
    depths, confs = [], []
    for _, _, td, _ in selected:
        raw, scale, _fmt = depth_by_t[td]
        depths.append(decode_depth(raw, scale, dh, dw).astype(np.float16))
        confs.append(decode_conf(conf_by_t.get(td, b"\x02" * (dh * dw)), dh, dw))

    np.savez_compressed(out / "depth.npz", depth=np.stack(depths), conf=np.stack(confs))
    with open(out / "slam.tum", "w", encoding="utf-8") as fh:
        for i, p in enumerate(pose_rows):
            fh.write("%.6f %.8f %.8f %.8f %.8f %.8f %.8f %.8f\n" % (
                stamps[i], p[1], p[2], p[3], p[4], p[5], p[6], p[7]))
    with open(out / "timestamps.csv", "w", encoding="utf-8") as fh:
        fh.write("frame,timestamp_s\n")
        for i, t in enumerate(stamps):
            fh.write("%d,%.6f\n" % (i, t))

    # 深度有效性统计
    # 统计只抽样（整段 stack 成 float32 会占几 GB 内存）
    step = max(1, len(depths) // 500)
    depth_stack = np.stack([d.astype(np.float32) for d in depths[::step]])
    conf_stack = np.stack(confs[::step])
    valid = (depth_stack > 0.05) & (depth_stack < 8.0) & (conf_stack >= 1)
    depth_ok_frac = float(valid.mean())
    med = float(np.median(depth_stack[valid])) if valid.any() else None

    # onboard 手出现率（2D only）
    hand_rate = None
    if hand_frames:
        n_ok = sum(1 for h in hand_frames if len(h.hands) > 0)
        hand_rate = float(n_ok) / len(hand_frames)

    calib = {
        "sensor": "iphone_lidar",
        "image_width": w,
        "image_height": h,
        "K_rgb": K.tolist(),
        "depth_width": dw,
        "depth_height": dh,
        "depth_units": "m",
        "pose_convention": "T_world_cam, OpenCV camera (x right, y down, z forward)",
        "source": "xhey_mcap",
        "mcap": str(path),
    }
    (out / "calib.yaml").write_text(json.dumps(calib, indent=1), encoding="utf-8")
    md = {
        "fps": fps,
        "device": "iphone_pro_lidar",
        "capture_app": "xhey_mcap",
        "num_frames": len(stamps),
        "rgb": [w, h],
        "depth": [dw, dh],
        "stride": int(stride),
        "depth_valid_frac": depth_ok_frac,
        "depth_median_m": med,
        "onboard_hand_frame_rate": hand_rate,
        "n_tf_poses": len(poses),
        "n_color_raw": len(color_pkts),
        "n_depth_raw": len(depth_by_t),
    }
    (out / "metadata.json").write_text(json.dumps(md, indent=1), encoding="utf-8")
    # 保存一小段 onboard 手，供对照（不是 GT）
    if hand_frames:
        sample = []
        for h in hand_frames[:: max(1, len(hand_frames) // 50)][:50]:
            sample.append({
                "t": _ts_sec(h.header),
                "hands": [
                    {
                        "landmarks": [
                            {
                                "name": lm.joint_name,
                                "u": lm.normalized_image_position.x,
                                "v": lm.normalized_image_position.y,
                                "conf": lm.confidence,
                            }
                            for lm in hand.landmarks
                        ]
                    }
                    for hand in h.hands
                ],
            })
        (out / "onboard_hands_sample.json").write_text(json.dumps(sample, indent=1), encoding="utf-8")
    return md


def main(argv=None):
    ap = argparse.ArgumentParser(description="xhey iPhone mcap → iPhone 会话目录")
    ap.add_argument("mcap")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--ffmpeg", default="ffmpeg")
    args = ap.parse_args(argv)
    md = convert_mcap(args.mcap, args.out, stride=args.stride, max_frames=args.max_frames, ffmpeg=args.ffmpeg)
    print(json.dumps(md, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
