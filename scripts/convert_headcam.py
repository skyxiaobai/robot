#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把头戴会话目录收成现有的统一 episode JSON。

会话目录（设备还没有，按这个落盘，规格见 ``docs/headcam_data_spec.md`` §7.7）::

    session/
      metadata.json          可选。episode_id、fps、task、instruction、environment、image_width/height、objects、verbs
      timestamps.csv         可选。frame_index,timestamp_s；没有就用 frame/fps
      imu.csv                可选。只记录路径和行数，不把样本写进 JSON，也不做积分
      calib.yaml             可选。Kalibr camchain、简单 YAML，或 OpenCV FileStorage
      rgb.mp4                可选。单目或彩色参考
      stereo/left.mp4        可选。没有 rgb 时当作主目
      stereo/right.mp4       可选。右目只提供 2D，用来三角化
      slam.tum               可选。TUM 轨迹。没有则相机位姿是单位阵
      hands.json             可选。后端已经算好的每帧 21 点

``hands.json`` 的每一帧::

    {
      "left": {"joints_cam": [[x,y,z] * 21], "keypoints_2d": [[u,v] * 21], "confidence": [21 个数]},
      "right": null,
      "stereo_keypoints_right_view": {"left": [[u,v] * 21], "right": null}
    }

``joints_cam`` 在左相机（或 rgb 相机）系，可以是没对齐尺度的单目结果。
有右目 2D 和标定时，三角化会把尺度收到米。再乘 SLAM 的相机位姿得到世界系。
子任务和分手指令留空，不在这里编造。

示例::

    python scripts/convert_headcam.py --session /path/to/session --out outputs/headcam/episode.json
    python scripts/convert_headcam.py --session /path/to/session --out episode.json --backend mediapipe
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from egodata.coverage import coarse_object_class, normalize_action, normalize_environment  # noqa: E402
from egodata.schema import SCHEMA_VERSION, empty_interaction, save_episode, validate_episode  # noqa: E402
from headcam.hand_pose import (  # noqa: E402
    JOINTS,
    _confidence_vector,
    _joints_array,
    _uv_array,
    associate_camera_poses,
    correct_monocular_with_stereo,
    get_backend,
    load_calibration,
    transform_points,
    wrist_poses_from_joints,
    xyz_to_json,
)
from headcam.hand_track_refine import RefineParams, refine_hands  # noqa: E402


def _first_file(session, relative_paths):
    for relative in relative_paths:
        path = session / relative
        if path.is_file():
            return path
    return None


def _count_rows(path):
    if path is None or not path.is_file():
        return 0
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("timestamp") or line.lower().startswith("frame"):
            continue
        count += 1
    return count


def _read_timestamps(path, num_frames, fps):
    if path is None or not path.is_file():
        return [index / float(fps) for index in range(num_frames)]
    stamps = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        lowered = line.lower()
        if lowered.startswith("frame") or lowered.startswith("timestamp"):
            continue
        parts = [part for part in line.replace(",", " ").split() if part]
        stamps.append(float(parts[-1]))
    if len(stamps) != num_frames:
        raise ValueError("timestamps.csv 有 %d 行，帧数是 %d" % (len(stamps), num_frames))
    return stamps


def _load_metadata(session):
    path = session / "metadata.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    data.setdefault("episode_id", "headcam/%s" % session.name)
    data.setdefault("fps", 30.0)
    data.setdefault("task", session.name)
    data.setdefault("instruction", "")
    data.setdefault("environment", "unknown")
    data.setdefault("objects", [])
    data.setdefault("verbs", [])
    return data


def _load_hands(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    frames = payload["frames"] if isinstance(payload, dict) else payload
    if not frames:
        raise ValueError("hands.json 没有帧：%s" % path)
    return frames


def _iter_rgb(path):
    import cv2
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError("打不开视频 %s" % path)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def _keypoints_or_none(hand):
    if not isinstance(hand, dict) or hand.get("keypoints_2d") is None:
        return None
    return _uv_array(hand["keypoints_2d"])


def _right_view_uv(stereo, side):
    """右目 2D。hands.json 里是点表，视频后端里是带 keypoints_2d 的手。"""
    if not isinstance(stereo, dict):
        return None
    value = stereo.get(side)
    if value is None:
        return None
    if isinstance(value, dict):
        return _keypoints_or_none(value)
    return _uv_array(value)


def _frames_from_backend(session, backend_name, calib):
    rgb = _first_file(session, ("rgb.mp4", "color.mp4"))
    left = _first_file(session, ("stereo/left.mp4", "left.mp4"))
    right = _first_file(session, ("stereo/right.mp4", "right.mp4"))
    primary = rgb or left
    if primary is None:
        raise RuntimeError("会话里没有 rgb.mp4 或 stereo/left.mp4，无法跑后端 %s" % backend_name)
    backend = get_backend(backend_name)
    left_predictions = [backend.predict(frame, calib=calib) for frame in _iter_rgb(primary)]
    right_predictions = None
    if right is not None and primary != right:
        right_predictions = [backend.predict(frame, calib=calib) for frame in _iter_rgb(right)]
        count = min(len(left_predictions), len(right_predictions))
        if len(left_predictions) != len(right_predictions):
            left_predictions = left_predictions[:count]
            right_predictions = right_predictions[:count]
    if not left_predictions:
        raise RuntimeError("视频里没有帧：%s" % primary)
    frames = []
    for index, prediction in enumerate(left_predictions):
        stereo = None
        if right_predictions is not None:
            other = right_predictions[index]
            stereo = {"left": other["left"], "right": other["right"]}
        frames.append({
            "left": prediction["left"],
            "right": prediction["right"],
            "stereo_keypoints_right_view": stereo,
        })
    return frames, primary


def _metric_for_hand(hand, right_uv, calib):
    if not hand or (hand.get("joints_cam") is None and _keypoints_or_none(hand) is None):
        return None, None, None
    mono = _joints_array(hand.get("joints_cam")) if hand.get("joints_cam") is not None else None
    if mono is not None and not np.isfinite(mono).any():
        mono = None
    left_uv = _keypoints_or_none(hand)
    if calib is not None and left_uv is not None and right_uv is not None and np.isfinite(right_uv).any():
        fused, confidence, scale = correct_monocular_with_stereo(mono, left_uv, right_uv, calib)
        return fused, confidence, scale
    if mono is None:
        return None, None, None
    return mono, _confidence_vector(hand.get("confidence")), None


def _confidence_lists(confidence):
    """关节置信度变成 JSON 列表。非有限值记为未知（None），不写成 0。"""
    if confidence is None:
        return [None] * JOINTS, None
    vector = []
    for value in np.asarray(confidence, dtype=float).reshape(-1):
        if not np.isfinite(value):
            vector.append(None)
        else:
            vector.append(float(value))
    if len(vector) != JOINTS:
        raise ValueError("confidence 必须有 21 个数")
    return vector, vector[0]


def _pack_side(camera_joints, confidences, camera_poses, filled=None, swapped=None):
    world_sequence = []
    joint_confidence = []
    scalar = []
    valid = []
    for joints, confidence, pose in zip(camera_joints, confidences, camera_poses):
        if joints is None:
            world_sequence.append(np.full((JOINTS, 3), np.nan))
            joint_confidence.append([0.0] * JOINTS)
            scalar.append(0.0)
            valid.append(False)
            continue
        world = transform_points(joints, pose)
        world_sequence.append(world)
        vector, wrist_confidence = _confidence_lists(confidence)
        joint_confidence.append(vector)
        scalar.append(wrist_confidence)
        if wrist_confidence is None:
            tracked = bool(np.isfinite(world[0]).all())
        else:
            tracked = bool(np.isfinite(world[0]).all() and wrist_confidence >= 0.5)
        valid.append(tracked)
    packed = {
        "joints": [xyz_to_json(frame) for frame in world_sequence],
        "wrist_pose": wrist_poses_from_joints(world_sequence),
        "confidence": scalar,
        "joint_confidence": joint_confidence,
        "valid": valid,
    }
    if filled is not None:
        packed["filled"] = [bool(value) for value in filled]
    if swapped is not None:
        packed["label_swapped"] = [bool(value) for value in swapped]
    return packed


def refine_params_from_args(args):
    """命令行没有打开任何精修开关时返回 None，输出与原来一致。"""
    if args is None:
        return None
    requested = bool(args.refine or args.smooth or args.gap_fill is not None or args.fixed_shape
                     or getattr(args, "gap_fill_s", None) is not None)
    if not requested:
        return None
    params = RefineParams()
    if args.refine:
        params.smooth = "one_euro"
        params.gap_fill = True
        params.max_gap = 5
        params.max_gap_s = 5.0 / 30.0
        params.gap_unit = "seconds"
        params.fixed_shape = True
        params.lr_consistency = True
    if args.smooth:
        params.smooth = args.smooth
    if args.gap_fill is not None:
        params.max_gap = int(args.gap_fill)
        params.gap_unit = "frames"
        params.gap_fill = params.max_gap > 0
        if params.gap_fill:
            params.lr_consistency = True
    if getattr(args, "gap_fill_s", None) is not None:
        params.max_gap_s = float(args.gap_fill_s)
        params.gap_unit = "seconds"
        params.gap_fill = params.max_gap_s > 0
        if params.gap_fill:
            params.lr_consistency = True
    if getattr(args, "legacy_frames", False):
        params.gap_unit = "frames"
    if args.fixed_shape:
        params.fixed_shape = True
    if args.no_lr_consistency:
        params.lr_consistency = False
    if args.min_cutoff is not None:
        params.min_cutoff = float(args.min_cutoff)
    if args.beta is not None:
        params.beta = float(args.beta)
    if args.d_cutoff is not None:
        params.d_cutoff = float(args.d_cutoff)
    if args.kalman_accel_std is not None:
        params.kalman_accel_std = float(args.kalman_accel_std)
    if args.kalman_meas_std is not None:
        params.kalman_meas_std = float(args.kalman_meas_std)
    return RefineParams(
        smooth=params.smooth,
        min_cutoff=params.min_cutoff,
        beta=params.beta,
        d_cutoff=params.d_cutoff,
        kalman_accel_std=params.kalman_accel_std,
        kalman_meas_std=params.kalman_meas_std,
        gap_fill=params.gap_fill,
        max_gap=params.max_gap,
        max_gap_s=params.max_gap_s,
        gap_unit=params.gap_unit,
        track_memory_s=params.track_memory_s,
        fixed_shape=params.fixed_shape,
        lr_consistency=params.lr_consistency,
    )


def build_episode(session_dir, backend=None, hands_path=None, refine=None):
    """读会话目录，返回统一 episode 字典。不写文件。"""
    session = Path(session_dir)
    if not session.is_dir():
        raise FileNotFoundError("会话目录不存在：%s" % session)
    metadata = _load_metadata(session)
    calib_path = _first_file(session, ("calib.yaml", "calib.yml", "stereo/calib.yaml"))
    calib = load_calibration(calib_path) if calib_path is not None else None
    slam_path = _first_file(session, ("slam.tum", "trajectory.tum"))
    imu_path = _first_file(session, ("imu.csv", "imu.txt"))
    video = _first_file(session, ("rgb.mp4", "color.mp4", "stereo/left.mp4", "left.mp4"))

    if backend:
        frames, predicted_video = _frames_from_backend(session, backend, calib)
        backend_name = backend
        if video is None:
            video = predicted_video
    else:
        hands_file = Path(hands_path) if hands_path else _first_file(session, ("hands.json",))
        if hands_file is None:
            frames, predicted_video = _frames_from_backend(session, "mediapipe", calib)
            backend_name = "mediapipe"
            if video is None:
                video = predicted_video
        else:
            frames = _load_hands(hands_file)
            backend_name = "hands_json"

    num_frames = len(frames)
    fps = float(metadata["fps"])
    timestamps = _read_timestamps(_first_file(session, ("timestamps.csv",)), num_frames, fps)
    poses, gaps = associate_camera_poses(timestamps, slam_path)

    width = metadata.get("image_width")
    height = metadata.get("image_height")
    if width is None or height is None:
        if calib is None:
            raise ValueError("metadata.json 缺少 image_width/image_height，且没有 calib.yaml")
        width = calib["image_width"]
        height = calib["image_height"]
    width, height = int(width), int(height)
    if calib is not None:
        intrinsic = np.asarray(calib["K_left"], dtype=float)
    else:
        focal = float(max(width, height))
        intrinsic = np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]])

    per_side = {side: {"joints": [], "confidence": [], "scale": []} for side in ("left", "right")}
    used_stereo = False
    for frame, pose in zip(frames, poses):
        stereo = frame.get("stereo_keypoints_right_view") or {}
        for side in ("left", "right"):
            joints, confidence, scale = _metric_for_hand(frame.get(side), _right_view_uv(stereo, side), calib)
            if scale is not None:
                used_stereo = True
            per_side[side]["joints"].append(joints)
            per_side[side]["confidence"].append(confidence if confidence is not None else np.zeros(JOINTS))
            per_side[side]["scale"].append(None if scale is None else float(scale))

    refine_info = None
    if refine is not None:
        stacked = {}
        conf_stack = {}
        for side in ("left", "right"):
            joints = np.full((num_frames, JOINTS, 3), np.nan, dtype=np.float64)
            confidence = np.full((num_frames, JOINTS), np.nan, dtype=np.float64)
            for index, (frame_joints, frame_conf) in enumerate(
                zip(per_side[side]["joints"], per_side[side]["confidence"])
            ):
                if frame_joints is not None:
                    joints[index] = np.asarray(frame_joints, dtype=np.float64)
                if frame_conf is not None:
                    confidence[index] = np.asarray(frame_conf, dtype=np.float64).reshape(-1)
            stacked[side] = joints
            conf_stack[side] = confidence
        refined = refine_hands(
            stacked["left"], stacked["right"], conf_stack["left"], conf_stack["right"],
            timestamps, refine, camera_poses=np.stack(poses),
        )
        for side in ("left", "right"):
            per_side[side]["joints"] = [
                None if not np.isfinite(frame).any() else frame
                for frame in refined[side]["joints"]
            ]
            per_side[side]["confidence"] = list(refined[side]["confidence"])
            per_side[side]["filled"] = refined[side]["filled"]
            per_side[side]["swapped"] = refined[side]["swapped"]
        refine_info = {
            "params": refine.to_dict(),
            "coordinate": refined["coordinate"],
            "filled_frames": {
                side: int(np.count_nonzero(refined[side]["filled"])) for side in ("left", "right")
            },
            "swapped_frames": {
                side: int(np.count_nonzero(refined[side]["swapped"])) for side in ("left", "right")
            },
            "shape": "bone_lengths" if refine.fixed_shape else None,
        }

    hands = {}
    for side in ("left", "right"):
        hands[side] = _pack_side(
            per_side[side]["joints"],
            per_side[side]["confidence"],
            poses,
            filled=per_side[side].get("filled"),
            swapped=per_side[side].get("swapped"),
        )

    environment_raw = metadata.get("environment") or ""
    environment = normalize_environment(environment_raw)
    objects = [str(name) for name in metadata.get("objects") or []]
    verbs = [str(name) for name in metadata.get("verbs") or []]
    action_types = []
    for verb in verbs:
        mapped = normalize_action(verb)
        if mapped not in action_types:
            action_types.append(mapped)
    object_classes = []
    for name in objects:
        mapped = coarse_object_class(name)
        if mapped not in object_classes:
            object_classes.append(mapped)
    task_name = str(metadata.get("task") or session.name)

    episode = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": metadata["episode_id"],
        "source": "headcam",
        "source_path": str(video if video is not None else session),
        "video_path": str(video) if video is not None else None,
        "fps": fps,
        "coordinate_frame": "slam_world" if slam_path is not None else "camera",
        "image_width": width,
        "image_height": height,
        "num_frames": num_frames,
        "timestamps": timestamps,
        "camera_intrinsic": intrinsic.tolist(),
        "camera_poses": [np.asarray(pose, dtype=float).tolist() for pose in poses],
        "hands": hands,
        **empty_interaction(num_frames),
        "annotation": {
            "environment": {
                "name": environment,
                "detail": environment_raw,
                "source": "metadata" if environment_raw else "missing",
            },
            "task": {"name": task_name, "instruction": metadata.get("instruction") or ""},
            "subtasks": [],
            "instructions": [],
        },
        "coverage": {
            "environment": environment,
            "objects": objects,
            "object_classes": object_classes,
            "task": task_name,
            "action_types": action_types,
        },
        "hand_pose": {
            "backend": backend_name,
            "stereo": used_stereo,
            "scale": {side: per_side[side]["scale"] for side in ("left", "right")},
            "slam_time_gap_s": gaps,
            "imu_csv": None if imu_path is None else imu_path.name,
            "imu_samples": _count_rows(imu_path),
            "calib": None if calib_path is None else calib_path.name,
            "refine": refine_info,
        },
    }
    errors = validate_episode(episode)
    if errors:
        raise ValueError("统一 episode 未通过校验：%s" % "; ".join(errors))
    return episode


def main(argv=None):
    parser = argparse.ArgumentParser(description="头戴会话目录 → 统一 episode JSON")
    parser.add_argument("--session", required=True, help="会话目录")
    parser.add_argument("--out", required=True, help="输出的 episode JSON")
    parser.add_argument("--backend", default=None, choices=["mediapipe", "hamer", "wilor"], help="从视频重算手部。省略时优先用 hands.json")
    parser.add_argument("--hands", default=None, help="预计算的 hands.json。默认用会话目录里的 hands.json")
    parser.add_argument("--refine", action="store_true",
                        help="打开 One Euro、最多补 5/30 秒、固定骨长和左右轨迹一致性")
    parser.add_argument("--smooth", default=None, choices=["none", "one_euro", "kalman"], help="时序平滑。one_euro 或常速度 kalman")
    parser.add_argument("--min-cutoff", type=float, default=None, help="One Euro 静止时的截止频率，单位 Hz")
    parser.add_argument("--beta", type=float, default=None, help="One Euro 速度系数，单位 1/(米/秒)")
    parser.add_argument("--d-cutoff", type=float, default=None, help="One Euro 导数截止频率，单位 Hz")
    parser.add_argument("--kalman-accel-std", type=float, default=None, help="卡尔曼加速度噪声，单位 米/秒²")
    parser.add_argument("--kalman-meas-std", type=float, default=None, help="卡尔曼测量噪声，单位米")
    parser.add_argument("--gap-fill", type=int, default=None, help="旧行为：最多插值多少帧缺测。0 表示不补")
    parser.add_argument("--gap-fill-s", type=float, default=None, help="最多插值多少秒的缺测。默认按 5/30 秒")
    parser.add_argument("--legacy-frames", action="store_true", help="补洞和左右手记忆按帧数，不按秒")
    parser.add_argument("--fixed-shape", action="store_true", help="用这一段的骨长中位数重摆关节")
    parser.add_argument("--no-lr-consistency", action="store_true", help="不要按轨迹修正左右手标签")
    args = parser.parse_args(argv)
    episode = build_episode(
        args.session,
        backend=args.backend,
        hands_path=args.hands,
        refine=refine_params_from_args(args),
    )
    save_episode(episode, args.out)
    print(
        "wrote %s frames=%d backend=%s stereo=%s frame=%s"
        % (
            args.out,
            episode["num_frames"],
            episode["hand_pose"]["backend"],
            episode["hand_pose"]["stereo"],
            episode["coordinate_frame"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
