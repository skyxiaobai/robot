# -*- coding: utf-8 -*-
"""统一 episode：世界系手部关节、手腕 6DoF，以及标注和覆盖度字段。"""
import json
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "1.0"

# 与 MediaPipe Hands 的 21 点顺序一致。EgoDex 的 Hand / ThumbKnuckle
# 只是按这个顺序摆进来，解剖上并不等于腕点和拇指 CMC，见
# egodex.EGODEX_NONCORRESPONDING_JOINTS。
MEDIAPIPE_21 = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)


def rotmat_to_quat_xyzw(rotation):
    """旋转矩阵 → 四元数 (x, y, z, w)。

    不强制 ``w >= 0``。``q`` 和 ``-q`` 是同一个旋转，按符号截断会让相邻帧
    在半球边界上翻转。一条轨迹里的连续性由 ``make_quaternions_continuous`` 保证。
    """
    matrix = np.asarray(rotation, dtype=float)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / scale
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / scale
        x = (matrix[0, 2] + matrix[2, 0]) / scale
        y = (matrix[1, 2] + matrix[2, 1]) / scale
        z = 0.25 * scale
    quat = np.array([x, y, z, w], dtype=float)
    norm = np.linalg.norm(quat)
    if norm == 0:
        return [0.0, 0.0, 0.0, 1.0]
    quat /= norm
    return [float(v) for v in quat]


def make_quaternions_continuous(quats):
    """让序列里每个四元数与上一帧落在同一半球（点积 >= 0）。

    缺测帧（非有限值）不参与，也不打断已经建立的参考。返回新的 float 数组。
    """
    out = np.asarray(quats, dtype=float).copy()
    if out.ndim != 2 or out.shape[1] != 4:
        raise ValueError("四元数序列必须是 (N, 4)")
    previous = None
    for index in range(out.shape[0]):
        current = out[index]
        if not np.isfinite(current).all():
            continue
        if previous is not None and float(np.dot(current, previous)) < 0.0:
            current = -current
            out[index] = current
        previous = current
    return out


def _as_floats(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [_as_floats(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if number != number:  # NaN
            return None
        return number
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, np.ndarray):
        return _as_floats(value.tolist())
    return value


def validate_episode(episode):
    """返回结构错误列表。空列表表示可以进入 QC。"""
    errors = []
    if not isinstance(episode, dict):
        return ["episode 必须是对象"]
    if episode.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version 必须是 %s" % SCHEMA_VERSION)
    num_frames = episode.get("num_frames")
    if not isinstance(num_frames, int) or num_frames < 1:
        errors.append("num_frames 必须是正整数")
        return errors
    timestamps = episode.get("timestamps")
    if not isinstance(timestamps, list) or len(timestamps) != num_frames:
        errors.append("timestamps 长度必须等于 num_frames")
    poses = episode.get("camera_poses")
    if not isinstance(poses, list) or len(poses) != num_frames:
        errors.append("camera_poses 长度必须等于 num_frames")
    else:
        for index, pose in enumerate(poses):
            if not isinstance(pose, list) or len(pose) != 4 or any(len(row) != 4 for row in pose):
                errors.append("camera_poses[%d] 必须是 4x4" % index)
                break
    hands = episode.get("hands")
    if not isinstance(hands, dict):
        errors.append("缺少 hands")
        return errors
    for side in ("left", "right"):
        hand = hands.get(side)
        if not isinstance(hand, dict):
            errors.append("缺少 hands.%s" % side)
            continue
        joints = hand.get("joints")
        wrist = hand.get("wrist_pose")
        confidence = hand.get("confidence")
        if not isinstance(joints, list) or len(joints) != num_frames:
            errors.append("hands.%s.joints 长度必须等于 num_frames" % side)
        else:
            for index, frame in enumerate(joints):
                if not isinstance(frame, list) or len(frame) != 21:
                    errors.append("hands.%s.joints[%d] 必须有 21 个点" % (side, index))
                    break
                if any(point is not None and (not isinstance(point, list) or len(point) != 3) for point in frame):
                    errors.append("hands.%s.joints[%d] 的点必须是长度为 3 的坐标" % (side, index))
                    break
        if not isinstance(wrist, list) or len(wrist) != num_frames:
            errors.append("hands.%s.wrist_pose 长度必须等于 num_frames" % side)
        else:
            for index, pose in enumerate(wrist):
                if not isinstance(pose, list) or len(pose) != 7:
                    errors.append("hands.%s.wrist_pose[%d] 必须是 xyz + xyzw 共 7 个数" % (side, index))
                    break
        if not isinstance(confidence, list) or len(confidence) != num_frames:
            errors.append("hands.%s.confidence 长度必须等于 num_frames" % side)
    return errors


def save_episode(episode, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _as_floats(episode)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_episode(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def iter_episode_paths(path):
    """文件则返回自身；目录则返回其中的 episode JSON（跳过 index.json）。"""
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(
        item for item in path.rglob("*.json") if item.name != "index.json"
    )
