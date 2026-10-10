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
6. ``gt/objects_gt.json``：有 ``<帧>.objects.json`` 时写下物体在世界系里的 6DoF。
   再给了物体表面（顶点或 glb）和手部点之后，用网格距离写 ``gt/interaction_gt.json``
   （接触、抓取、事件）。没有表面就不编造接触。

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


def se3_from_hot3d(se3_dict):
    """HOT3D 的 ``quaternion_wxyz`` + ``translation_xyz`` → 4×4。

    四元数是 wxyz，和平时 episode 里的 xyzw 顺序不同。平移单位是米。
    """
    qw, qx, qy, qz = [float(v) for v in se3_dict["quaternion_wxyz"]]
    tx, ty, tz = [float(v) for v in se3_dict["translation_xyz"]]
    norm = float(np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz))
    if norm == 0.0:
        qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
    else:
        qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    rotation = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=float)
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = [tx, ty, tz]
    return matrix


def _pose7_from_matrix(matrix):
    matrix = np.asarray(matrix, dtype=float)
    quat = rotmat_to_quat_xyzw(matrix[:3, :3])
    return [float(v) for v in list(matrix[:3, 3]) + list(quat)]


def _instance_pose(instance):
    raw = instance.get("T_world_from_object")
    if raw is None:
        raise ValueError("HOT3D 物体缺少 T_world_from_object")
    if isinstance(raw, dict):
        return se3_from_hot3d(raw)
    matrix = np.asarray(raw, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError("T_world_from_object 必须是 4x4 或 quaternion/translation")
    return matrix


def parse_objects_json(payload):
    """读一帧 ``.objects.json``。

    HOT3D-Clips 里这是一个字典，值是实例列表。每个实例有 ``object_bop_id`` 和
    ``T_world_from_object``（``quaternion_wxyz`` + ``translation_xyz``）。
    也接受直接的实例列表，方便小夹具。
    """
    if isinstance(payload, dict):
        groups = list(payload.items())
    elif isinstance(payload, list):
        groups = [(str(index), [item]) for index, item in enumerate(payload)]
    else:
        raise ValueError("objects.json 必须是对象或列表")
    found = []
    for key, value in groups:
        seq = value if isinstance(value, list) else [value]
        for index, instance in enumerate(seq):
            if not isinstance(instance, dict):
                continue
            bop = instance.get("object_bop_id", key)
            track_id = str(key) if len(seq) == 1 else "%s#%d" % (key, index)
            name = instance.get("object_name") or ("bop_%s" % bop)
            found.append({
                "id": track_id,
                "bop_id": int(bop) if str(bop).lstrip("-").isdigit() else bop,
                "category": str(name),
                "pose7": _pose7_from_matrix(_instance_pose(instance)),
            })
    return found


def _continuous_poses(poses):
    from egodata.schema import make_quaternions_continuous
    quats = []
    for pose in poses:
        if pose is None:
            quats.append([np.nan, np.nan, np.nan, np.nan])
        else:
            quats.append(pose[3:])
    aligned = make_quaternions_continuous(quats)
    out = []
    for pose, quat in zip(poses, aligned):
        if pose is None or not np.isfinite(quat).all():
            out.append(pose)
        else:
            out.append([float(v) for v in list(pose[:3]) + list(quat)])
    return out


def tracks_from_frames(frames):
    """每帧的实例列表 → schema 里的物体轨迹。某一帧没有这个物体就记成无效。"""
    order = []
    meta = {}
    for frame in frames:
        for inst in frame:
            if inst["id"] not in meta:
                order.append(inst["id"])
                meta[inst["id"]] = inst
    num_frames = len(frames)
    objects = []
    for obj_id in order:
        pose = [None] * num_frames
        for index, frame in enumerate(frames):
            for inst in frame:
                if inst["id"] == obj_id:
                    pose[index] = inst["pose7"]
                    meta[obj_id] = inst
        pose = _continuous_poses(pose)
        objects.append({
            "id": obj_id,
            "category": meta[obj_id]["category"],
            "source": "hot3d",
            "bop_id": meta[obj_id]["bop_id"],
            "pose": pose,
            "confidence": [1.0 if item is not None else None for item in pose],
            "valid": [item is not None for item in pose],
        })
    return objects


def read_frame_objects(clip, key):
    path = Path(clip) / ("%s.objects.json" % key)
    if not path.is_file():
        return []
    return parse_objects_json(json.loads(path.read_text(encoding="utf-8")))


def load_object_surfaces(models_dir):
    """从目录读物体表面。支持 ``12.npy`` / ``obj_000012.npy`` 顶点，可选 ``*_faces.npy``，以及 glb。

    glb 需要 trimesh。读不到就跳过，不假装已经有网格。
    """
    directory = Path(models_dir)
    surfaces = {}
    if not directory.is_dir():
        return surfaces
    for path in sorted(directory.glob("*.npy")):
        if path.name.endswith("_faces.npy"):
            continue
        stem = path.stem
        if stem.startswith("obj_"):
            stem = stem.split("obj_", 1)[1]
        if not stem.lstrip("-").isdigit():
            continue
        bop_id = str(int(stem))
        vertices = np.load(path)
        faces_path = path.with_name("%s_faces.npy" % path.stem)
        faces = np.load(faces_path) if faces_path.is_file() else None
        surfaces[bop_id] = {"vertices": vertices, "faces": faces}
    for path in sorted(directory.glob("*.glb")):
        stem = path.stem
        if stem.startswith("obj_"):
            stem = stem.split("obj_", 1)[1]
        if not stem.lstrip("-").isdigit():
            continue
        bop_id = str(int(stem))
        if bop_id in surfaces:
            continue
        try:
            import trimesh
        except ImportError:
            break
        loaded = trimesh.load(path, force="mesh", process=False)
        surfaces[bop_id] = {
            "vertices": np.asarray(loaded.vertices, dtype=float),
            "faces": np.asarray(loaded.faces, dtype=int),
        }
    return surfaces


def _surfaces_for_tracks(objects, surfaces):
    """表面字典的键可以是轨迹 id，也可以是 bop id。"""
    mapped = {}
    for obj in objects:
        if obj["id"] in surfaces:
            mapped[obj["id"]] = surfaces[obj["id"]]
            continue
        bop = str(obj.get("bop_id"))
        if bop in surfaces:
            mapped[obj["id"]] = surfaces[bop]
    return mapped


def export_hot3d_objects(clip, out_dir, keys, timestamps, hand_points=None, surfaces=None, object_models_dir=None):
    """把 clip 目录里的物体位姿写成 ``gt/objects_gt.json``。

    给了手部点和物体表面时，再用网格距离写 ``gt/interaction_gt.json``。
    没有表面就只写位姿，接触通道保持无效，不编造抓取。
    """
    if _scripts_dir() not in sys.path:
        sys.path.insert(0, _scripts_dir())
    from egodata.interaction import derive_gt_interaction
    from egodata.schema import empty_interaction

    frames = [read_frame_objects(clip, key) for key in keys]
    objects = tracks_from_frames(frames)
    out = Path(out_dir)
    (out / "gt").mkdir(parents=True, exist_ok=True)
    (out / "gt" / "objects_gt.json").write_text(
        json.dumps({"coordinate_frame": "world", "unit": "m", "objects": objects}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    resolved = {}
    if surfaces:
        resolved.update(_surfaces_for_tracks(objects, surfaces))
    if object_models_dir:
        resolved.update(_surfaces_for_tracks(objects, load_object_surfaces(object_models_dir)))
    if hand_points is not None and resolved:
        interaction = derive_gt_interaction(objects, hand_points, timestamps, resolved)
    else:
        interaction = empty_interaction(len(timestamps))
        interaction["contact"]["source"] = "hot3d_pose_only"
        interaction["grasp"]["source"] = "hot3d_pose_only"
    (out / "gt" / "interaction_gt.json").write_text(
        json.dumps(interaction, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"objects": len(objects), "interaction": interaction["contact"]["source"], "session": str(out)}


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


def convert_clip(clip, out_session, mano_dir=None, max_frames=None, crf=10, object_models_dir=None):
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
    object_keys = [key for key in keys if (Path(clip) / ("%s.objects.json" % key)).is_file()]
    objects_summary = None
    if object_keys:
        hand_points = None
        if gt_frames:
            hand_points = {
                side: [None if frame.get(side) is None else np.asarray(frame[side], dtype=float) for frame in gt_frames]
                for side in ("left", "right")
            }
        objects_summary = export_hot3d_objects(
            clip, out, keys, stamps,
            hand_points=hand_points,
            object_models_dir=object_models_dir,
        )
        names = []
        tracks = json.loads((out / "gt" / "objects_gt.json").read_text(encoding="utf-8"))
        for item in tracks["objects"]:
            if item["category"] not in names:
                names.append(item["category"])
        meta["objects"] = names
        (out / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"session": str(out), "frames": len(keys), "device": device, "baseline_m": float(np.linalg.norm(T)),
            "extrinsic_drift_m": drift_t, "fps": meta["fps"], "gt": gt is not None,
            "objects": None if objects_summary is None else objects_summary["objects"]}
