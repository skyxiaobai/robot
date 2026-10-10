# -*- coding: utf-8 -*-
"""统一 episode：世界系手部关节、手腕 6DoF、物体 6DoF、接触和抓取。"""
import json
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "1.0"
GRASP_STATES = ("open", "pre_grasp", "grasp", "release")
EVENT_TYPES = ("contact_start", "contact_end", "grasp", "release")
HAND_SIDES = ("left", "right")

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


def _blank_hand(num_frames, field):
    return {
        field: [None] * num_frames,
        "confidence": [None] * num_frames,
        "valid": [False] * num_frames,
    }


def empty_interaction(num_frames):
    """没有物体位姿时的空通道。有效位全是 false，不能把 0 当成真位姿。"""
    num_frames = int(num_frames)
    return {
        "objects": [],
        "contact": {
            "source": "unknown",
            "left": _blank_hand(num_frames, "object_id"),
            "right": _blank_hand(num_frames, "object_id"),
        },
        "grasp": {
            "source": "unknown",
            "left": _blank_hand(num_frames, "state"),
            "right": _blank_hand(num_frames, "state"),
        },
        "events": [],
    }


def _finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating, np.integer)):
        return False
    return bool(np.isfinite(value))


def _pose7_ok(pose):
    if not isinstance(pose, (list, tuple)) or len(pose) != 7:
        return False
    return all(_finite_number(item) for item in pose)


def _validate_interaction(episode, num_frames, errors):
    objects = episode.get("objects")
    if not isinstance(objects, list):
        errors.append("缺少 objects")
        object_ids = set()
    else:
        object_ids = set()
        for index, obj in enumerate(objects):
            if not isinstance(obj, dict):
                errors.append("objects[%d] 必须是对象" % index)
                continue
            obj_id = obj.get("id")
            if not isinstance(obj_id, str) or not obj_id:
                errors.append("objects[%d].id 必须是非空字符串" % index)
            elif obj_id in object_ids:
                errors.append("objects id 重复：%s" % obj_id)
            else:
                object_ids.add(obj_id)
            if not isinstance(obj.get("category"), str) or not obj.get("category"):
                errors.append("objects[%d].category 必须是非空字符串" % index)
            if not isinstance(obj.get("source"), str) or not obj.get("source"):
                errors.append("objects[%d].source 必须是非空字符串" % index)
            pose = obj.get("pose")
            confidence = obj.get("confidence")
            valid = obj.get("valid")
            if not isinstance(pose, list) or len(pose) != num_frames:
                errors.append("objects[%d].pose 长度必须等于 num_frames" % index)
            if not isinstance(confidence, list) or len(confidence) != num_frames:
                errors.append("objects[%d].confidence 长度必须等于 num_frames" % index)
            if not isinstance(valid, list) or len(valid) != num_frames:
                errors.append("objects[%d].valid 长度必须等于 num_frames" % index)
            elif isinstance(pose, list) and len(pose) == num_frames:
                for frame_index, flag in enumerate(valid):
                    if not isinstance(flag, (bool, np.bool_)):
                        errors.append("objects[%d].valid[%d] 必须是布尔值" % (index, frame_index))
                        break
                    if flag and not _pose7_ok(pose[frame_index]):
                        errors.append("objects[%d].pose[%d] 必须是 xyz + xyzw 共 7 个数" % (index, frame_index))
                        break
    for channel, field, allowed in (
        ("contact", "object_id", None),
        ("grasp", "state", GRASP_STATES),
    ):
        block = episode.get(channel)
        if not isinstance(block, dict):
            errors.append("缺少 %s" % channel)
            continue
        if "source" in block and not isinstance(block.get("source"), str):
            errors.append("%s.source 必须是字符串" % channel)
        for side in HAND_SIDES:
            hand = block.get(side)
            if not isinstance(hand, dict):
                errors.append("缺少 %s.%s" % (channel, side))
                continue
            values = hand.get(field)
            confidence = hand.get("confidence")
            valid = hand.get("valid")
            if not isinstance(values, list) or len(values) != num_frames:
                errors.append("%s.%s.%s 长度必须等于 num_frames" % (channel, side, field))
                continue
            if not isinstance(confidence, list) or len(confidence) != num_frames:
                errors.append("%s.%s.confidence 长度必须等于 num_frames" % (channel, side))
            if not isinstance(valid, list) or len(valid) != num_frames:
                errors.append("%s.%s.valid 长度必须等于 num_frames" % (channel, side))
                continue
            for frame_index, flag in enumerate(valid):
                if not isinstance(flag, (bool, np.bool_)):
                    errors.append("%s.%s.valid[%d] 必须是布尔值" % (channel, side, frame_index))
                    break
                if not flag:
                    continue
                value = values[frame_index]
                if channel == "contact":
                    if value is not None and not isinstance(value, str):
                        errors.append("contact.%s.object_id[%d] 必须是字符串或空" % (side, frame_index))
                        break
                    if isinstance(value, str) and object_ids and value not in object_ids:
                        errors.append("contact.%s.object_id[%d] 不在 objects 里" % (side, frame_index))
                        break
                elif value not in allowed:
                    errors.append("grasp.%s.state[%d] 必须是 %s" % (side, frame_index, "/".join(GRASP_STATES)))
                    break
    events = episode.get("events")
    if not isinstance(events, list):
        errors.append("缺少 events")
        return
    timestamps = episode.get("timestamps")
    t0 = t1 = None
    if isinstance(timestamps, list) and timestamps:
        try:
            t0 = float(timestamps[0])
            t1 = float(timestamps[-1])
        except (TypeError, ValueError):
            t0 = t1 = None
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            errors.append("events[%d] 必须是对象" % index)
            continue
        if event.get("type") not in EVENT_TYPES:
            errors.append("events[%d].type 必须是 %s" % (index, "/".join(EVENT_TYPES)))
        if event.get("hand") not in HAND_SIDES:
            errors.append("events[%d].hand 必须是 left 或 right" % index)
        stamp = event.get("timestamp")
        if not _finite_number(stamp):
            errors.append("events[%d].timestamp 必须是数" % index)
        elif t0 is not None and (float(stamp) < t0 - 1e-6 or float(stamp) > t1 + 1e-6):
            errors.append("events[%d].timestamp 必须落在片段时间范围内" % index)


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
    _validate_interaction(episode, num_frames, errors)
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
