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
from egodata.schema import SCHEMA_VERSION, save_episode, validate_episode  # noqa: E402
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


def _pack_side(camera_joints, confidences, camera_poses):
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
        vector = [float(value) if np.isfinite(value) else 0.0 for value in confidence]
        joint_confidence.append(vector)
        wrist_confidence = vector[0]
        scalar.append(wrist_confidence)
        valid.append(bool(np.isfinite(world[0]).all() and wrist_confidence >= 0.5))
    return {
        "joints": [xyz_to_json(frame) for frame in world_sequence],
        "wrist_pose": wrist_poses_from_joints(world_sequence),
        "confidence": scalar,
        "joint_confidence": joint_confidence,
        "valid": valid,
    }


def build_episode(session_dir, backend=None, hands_path=None):
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

    hands = {}
    for side in ("left", "right"):
        hands[side] = _pack_side(per_side[side]["joints"], per_side[side]["confidence"], poses)

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
    args = parser.parse_args(argv)
    episode = build_episode(args.session, backend=args.backend, hands_path=args.hands)
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
