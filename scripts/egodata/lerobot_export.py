# -*- coding: utf-8 -*-
"""把统一 episode 写成 LeRobot 0.6.1 能读的 v3.0 数据集。

只写入 QC 产出率 CSV 里 ``accepted=yes`` 的片段。一次只展开一条 JSON，
打包成数组后就丢掉原文，避免把全部 episode 的嵌套列表留在内存里。

向量约定（世界系，与统一 episode 相同）：

- ``observation.state`` 长度 140 =
  左手 21×3 关节、右手 21×3 关节、左手腕 7 维、右手腕 7 维。
  某只手有缺测关节，或手腕置信度是数字且低于 0.5 时，该手的 63 维关节置 0，
  ``observation.hand_valid`` 对应侧为 0。置信度缺失（未知）不算无效。
  手腕 7 维只要数字齐全就保留。
- ``action`` 是**下一步**相对当前手腕的增量，不是下一帧的绝对位姿。
  每只手 7 维：``dxyz``（米）+ 相对四元数 xyzw（``q_next * conj(q_now)``，
  取 ``w >= 0`` 的那支）。左手在前，右手在后。默认长度 14。
  ``include_hand=True`` 时再接双手关节的 xyz 增量（各 63 维）。
  片段最后一帧没有下一步，直接丢掉。多步目标不摊进这一列：
  线性 BC 把连续 ``horizon`` 帧的 action 拼成一块，ACT 用同样的
  ``chunk_size`` 向后看。保持不动的标签是 dxyz=0、相对四元数 0,0,0,1。
- ``action_valid`` 形状 (2,)，左手、右手各一个。当前帧和下一帧的手腕
  都是实测（有限、不是补帧、不是丢掉的标注、置信度不是低于 0.5 的数字）
  才为 1。缺测仍写入上面的占位增量，但这一位是 0，训练损失不算它。
  显式的 ``good_frame_mask`` 为 false 时，这一帧两只手都无效。
- 语言写在 ``task_index``。该帧时刻落在某条 SUBTASK 里就用子任务句子，
  否则用 TASK.instruction。
- ``observation.object_pose`` 长度 28 = 最多 4 个物体 ×（xyz + xyzw）。
  无效槽位写 0，同时 ``observation.object_pose_valid`` 为 0。0 不是原点上的真物体。
- ``observation.contact`` 长度 4：左手槽位、左手置信度、右手槽位、右手置信度。
  没有接触时槽位是 -1。``observation.contact_valid`` 长度 2。
- ``action.grasp`` 长度 2，是这一帧的抓取状态，不是下一步增量：
  0 张开、1 预备、2 抓住、3 放开。``action.grasp_valid`` 长度 2。
  掩码的写法和 ``observation.hand_valid`` 一样：0 表示这一维不要拿去训练。

没有现成 mp4 时写 16×16 的占位视频，并在特征 info 里标
``egodata.image_source=placeholder``。旁边若有与 ``source_path`` 同名的
``.mp4``，则裁到保留的帧数，并默认缩到正方形（默认 224，边长为偶数），
避免按 1080p 重编码。

``hands_measured is False`` 的片段走另一条落盘：不写 ``observation.state``、
``action`` 和手腕列，也不把缺测手填成 0。每一行有 ``slam_status`` 和
``observation.pose``（``tx ty tz qx qy qz qw``）。状态不是 ``ok`` 时位姿格是
null。``timestamp`` 按 LeRobot 0.6.1 的写法是 ``frame_index / fps``（float32），
视频解码用它做容差检查。图像时刻另外放在 ``observation.capture_time_ns``
（int64 纳秒），避免 1e9 秒量级被 float32 吃掉。``accuracy`` 只出现在导出说明里，值是 null。

有手部测量、并且 episode 上已经有逐帧 ``pose_status`` / ``T_world_cam`` 时，
同一份包再加这三列。没有这组状态的旧导出保持原来的列，不补 0 位姿。
最后一帧没有下一步动作，整行不写，所以包里的帧数比 episode 少 1。
"""
import csv
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from egodata.action_valid import (
    action_dim_mask,
    action_valid_flags,
    chunk_ratio_map,
    validity_summary,
)
from egodata.schema import iter_episode_paths, load_episode, rotmat_to_quat_xyzw

CODEBASE_VERSION = "v3.0"
JOINTS = 21
WRIST_DIM = 7
STATE_DIM = 2 * (JOINTS * 3 + WRIST_DIM)
WRIST_STEP_DIM = 2 * WRIST_DIM
HAND_STEP_DIM = 2 * JOINTS * 3
ACTION_DIM = WRIST_STEP_DIM
DEFAULT_HORIZON = 16
DEFAULT_VIDEO_SIZE = 224
OBJECT_SLOTS = 4
OBJECT_POSE_DIM = OBJECT_SLOTS * WRIST_DIM
CONTACT_DIM = 4
GRASP_DIM = 2
_GRASP_CODE = {"open": 0.0, "pre_grasp": 1.0, "grasp": 2.0, "release": 3.0}
IMAGE_KEY = "observation.image"
PLACEHOLDER_SIZE = 16
_CONFIDENCE_MIN = 0.5
_IDENTITY_DELTA = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)

_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
_EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"


def action_step_dim(include_hand=False):
    return WRIST_STEP_DIM + (HAND_STEP_DIM if include_hand else 0)


def accepted_episode_ids(yield_csv):
    """读 ``egodata_qc`` 写出的 CSV。同一 id 多行时以最后一行为准。"""
    status = {}
    with open(yield_csv, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "episode_id" not in reader.fieldnames or "accepted" not in reader.fieldnames:
            raise ValueError("产出率 CSV 需要 episode_id 和 accepted 列")
        for row in reader:
            episode_id = (row.get("episode_id") or "").strip()
            if not episode_id:
                continue
            flag = (row.get("accepted") or "").strip().lower()
            status[episode_id] = flag in ("yes", "true", "1")
    return {episode_id for episode_id, ok in status.items() if ok}


def _finite_vec(value, width):
    if not isinstance(value, (list, tuple)) or len(value) != width:
        return None
    if any(item is None for item in value):
        return None
    array = np.asarray(value, dtype=np.float32)
    if not np.isfinite(array).all():
        return None
    return array


def _joints_ok(joints):
    if not isinstance(joints, list) or len(joints) != JOINTS:
        return False
    for point in joints:
        if _finite_vec(point, 3) is None:
            return False
    return True


def _confidence_ok(confidence):
    """缺失表示未知，视为通过；只有读到了低于阈值的数才失败。"""
    if confidence is None:
        return True
    return float(confidence) >= _CONFIDENCE_MIN


def hand_valid_flags(episode, frame_index):
    """左手、右手是否整手可用。返回 float32[2]。"""
    flags = []
    for side in ("left", "right"):
        hand = episode["hands"][side]
        confidence = hand["confidence"][frame_index]
        joints = hand["joints"][frame_index]
        wrist = _finite_vec(hand["wrist_pose"][frame_index], WRIST_DIM)
        usable = wrist is not None and _joints_ok(joints) and _confidence_ok(confidence)
        flags.append(1.0 if usable else 0.0)
    return np.asarray(flags, dtype=np.float32)


def pack_state(episode, frame_index):
    """140 维状态。无效手的关节块为 0，手腕仍写入（缺测则为 0）。"""
    flags = hand_valid_flags(episode, frame_index)
    parts = []
    for side, valid in zip(("left", "right"), flags):
        if valid < 0.5:
            parts.append(np.zeros(JOINTS * 3, dtype=np.float32))
        else:
            points = episode["hands"][side]["joints"][frame_index]
            parts.append(np.asarray(points, dtype=np.float32).reshape(JOINTS * 3))
    for side in ("left", "right"):
        wrist = _finite_vec(episode["hands"][side]["wrist_pose"][frame_index], WRIST_DIM)
        parts.append(wrist if wrist is not None else np.zeros(WRIST_DIM, dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)


def _quat_conj(quat):
    return np.array([-quat[0], -quat[1], -quat[2], quat[3]], dtype=np.float64)


def _quat_mul(left, right):
    ax, ay, az, aw = left
    bx, by, bz, bw = right
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dtype=np.float64)


def relative_quat(current, future):
    """``future * conj(current)``，并取 w >= 0。单位旋转是 (0, 0, 0, 1)。"""
    rel = _quat_mul(np.asarray(future, dtype=np.float64), _quat_conj(np.asarray(current, dtype=np.float64)))
    norm = float(np.linalg.norm(rel))
    if norm == 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    rel /= norm
    if rel[3] < 0.0:
        rel = -rel
    return rel


def wrist_delta(current, future):
    """当前手腕到下一步手腕的 7 维增量。缺测时用「保持不动」。"""
    current = _finite_vec(current, WRIST_DIM)
    future = _finite_vec(future, WRIST_DIM)
    if current is None or future is None:
        return _IDENTITY_DELTA.copy()
    delta = np.empty(WRIST_DIM, dtype=np.float64)
    delta[:3] = np.asarray(future[:3], dtype=np.float64) - np.asarray(current[:3], dtype=np.float64)
    delta[3:] = relative_quat(current[3:], future[3:])
    return delta.astype(np.float32)


def _joint_delta(episode, side, frame_index, next_index):
    current = episode["hands"][side]["joints"][frame_index]
    future = episode["hands"][side]["joints"][next_index]
    if not _joints_ok(current) or not _joints_ok(future):
        return np.zeros(JOINTS * 3, dtype=np.float32)
    now = np.asarray(current, dtype=np.float32).reshape(JOINTS * 3)
    nxt = np.asarray(future, dtype=np.float32).reshape(JOINTS * 3)
    return (nxt - now).astype(np.float32)


def pack_action(episode, frame_index, include_hand=False):
    """下一步双手手腕增量。已经是最后一帧时返回 None。"""
    nxt = frame_index + 1
    if nxt >= episode["num_frames"]:
        return None
    parts = []
    for side in ("left", "right"):
        parts.append(wrist_delta(
            episode["hands"][side]["wrist_pose"][frame_index],
            episode["hands"][side]["wrist_pose"][nxt],
        ))
    if include_hand:
        for side in ("left", "right"):
            parts.append(_joint_delta(episode, side, frame_index, nxt))
    return np.concatenate(parts).astype(np.float32)


def _sorted_objects(episode):
    objects = episode.get("objects") or []
    return sorted(objects, key=lambda item: str(item.get("id")))


def _series_valid(hand, frame_index):
    valid = (hand or {}).get("valid") or []
    if frame_index >= len(valid) or not isinstance(valid[frame_index], (bool, np.bool_, int, float)):
        return False
    return bool(valid[frame_index])


def pack_object_pose(episode, frame_index):
    """物体 6DoF。最多 4 个槽位，按 id 排序。无效槽位是 0，掩码也是 0。"""
    pose = np.zeros(OBJECT_POSE_DIM, dtype=np.float32)
    valid = np.zeros(OBJECT_SLOTS, dtype=np.float32)
    for slot, obj in enumerate(_sorted_objects(episode)[:OBJECT_SLOTS]):
        if not _series_valid(obj, frame_index):
            continue
        poses = obj.get("pose") or []
        if frame_index >= len(poses):
            continue
        vec = _finite_vec(poses[frame_index], WRIST_DIM)
        if vec is None:
            continue
        pose[slot * WRIST_DIM:(slot + 1) * WRIST_DIM] = vec
        valid[slot] = 1.0
    return pose, valid


def pack_contact(episode, frame_index):
    """左右手各一个物体槽位和置信度。没接触且有效时槽位是 -1。"""
    vec = np.zeros(CONTACT_DIM, dtype=np.float32)
    mask = np.zeros(2, dtype=np.float32)
    slots = {
        str(obj.get("id")): index
        for index, obj in enumerate(_sorted_objects(episode)[:OBJECT_SLOTS])
    }
    contact = episode.get("contact") or {}
    for hand_index, side in enumerate(("left", "right")):
        hand = contact.get(side) or {}
        if not _series_valid(hand, frame_index):
            continue
        ids = hand.get("object_id") or []
        confs = hand.get("confidence") or []
        object_id = ids[frame_index] if frame_index < len(ids) else None
        confidence = confs[frame_index] if frame_index < len(confs) else None
        if object_id is None:
            slot = -1.0
        elif str(object_id) not in slots:
            continue
        else:
            slot = float(slots[str(object_id)])
        if confidence is None:
            conf_value = 0.0
        else:
            conf_value = float(confidence)
            if not np.isfinite(conf_value):
                conf_value = 0.0
        mask[hand_index] = 1.0
        vec[hand_index * 2] = slot
        vec[hand_index * 2 + 1] = conf_value
    return vec, mask


def pack_grasp(episode, frame_index):
    """当前帧的抓取状态编码。无效手是 0，并且掩码是 0。"""
    vec = np.zeros(GRASP_DIM, dtype=np.float32)
    mask = np.zeros(2, dtype=np.float32)
    grasp = episode.get("grasp") or {}
    for hand_index, side in enumerate(("left", "right")):
        hand = grasp.get(side) or {}
        if not _series_valid(hand, frame_index):
            continue
        states = hand.get("state") or []
        state = states[frame_index] if frame_index < len(states) else None
        if state not in _GRASP_CODE:
            continue
        vec[hand_index] = _GRASP_CODE[state]
        mask[hand_index] = 1.0
    return vec, mask


def frame_task(episode, timestamp):
    """SUBTASK 半开区间 [t_start, t_end)；对不上时用整段指令。"""
    subtasks = (episode.get("annotation") or {}).get("subtasks") or []
    ordered = sorted(subtasks, key=lambda item: float(item["t_start"]))
    for segment in ordered:
        start = float(segment["t_start"])
        end = float(segment["t_end"])
        if start <= float(timestamp) < end:
            return str(segment["text"])
    if ordered and abs(float(timestamp) - float(ordered[-1]["t_end"])) <= 1e-3:
        return str(ordered[-1]["text"])
    task = (episode.get("annotation") or {}).get("task") or {}
    instruction = str(task.get("instruction") or "").strip()
    if instruction:
        return instruction
    return str(task.get("name") or "unknown")


def _feature(dtype, shape, names=None, info=None, fps=None):
    payload = {"dtype": dtype, "shape": list(shape), "names": names}
    if info is not None:
        payload["info"] = info
    if fps is not None and dtype != "video":
        payload["fps"] = fps
    return payload


def _stats(array):
    values = np.asarray(array, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    count = int(values.shape[0])
    mean = values.mean(axis=0)
    return {
        "min": values.min(axis=0),
        "max": values.max(axis=0),
        "mean": mean,
        "std": values.std(axis=0),
        "count": np.array([count]),
    }


def _stats_json(array):
    raw = _stats(array)
    return {key: np.asarray(value).reshape(-1).tolist() for key, value in raw.items()}


def _stats_where(array, mask):
    """按维只用 mask>=0.5 的行。某一维完全没有有效行时，退回这一维的全部存储值。"""
    values = np.asarray(array, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    weights = np.asarray(mask, dtype=np.float64)
    if weights.shape != values.shape or float(np.min(weights)) >= 1.0 - 1e-8:
        return _stats(values)
    width = values.shape[1]
    mins = np.empty(width)
    maxs = np.empty(width)
    means = np.empty(width)
    stds = np.empty(width)
    for dim in range(width):
        chosen = values[weights[:, dim] >= 0.5, dim]
        if chosen.size == 0:
            chosen = values[:, dim]
        mins[dim] = chosen.min()
        maxs[dim] = chosen.max()
        means[dim] = chosen.mean()
        stds[dim] = chosen.std()
    return {
        "min": mins,
        "max": maxs,
        "mean": means,
        "std": stds,
        "count": np.array([int(values.shape[0])]),
    }


def _stats_json_where(array, mask):
    raw = _stats_where(array, mask)
    return {key: np.asarray(value).reshape(-1).tolist() for key, value in raw.items()}


def _image_stats_json():
    """图像统计量。lerobot 0.6.1 默认用 ImageNet 的 mean/std 覆盖这两项，但键必须先存在。"""
    return {
        "min": [[[0.0]], [[0.0]], [[0.0]]],
        "max": [[[1.0]], [[1.0]], [[1.0]]],
        "mean": [[[0.485]], [[0.456]], [[0.406]]],
        "std": [[[0.229]], [[0.224]], [[0.225]]],
        "count": [1],
    }


def _write_placeholder_video(path, num_frames, fps):
    size = PLACEHOLDER_SIZE
    raw = bytearray()
    for index in range(num_frames):
        pixel = bytes((index * 40 % 256, 30, 180))
        raw.extend(pixel * (size * size))
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", "%dx%d" % (size, size),
            "-r", str(int(fps)),
            "-i", "pipe:0",
            "-frames:v", str(num_frames),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
            str(path),
        ],
        input=bytes(raw),
        check=True,
    )
    return size, size, "placeholder"


def _even_size(video_size):
    size = int(video_size)
    if size < 2:
        raise ValueError("video_size 至少为 2")
    if size % 2:
        size -= 1
    return size


def _trim_source_video(source, path, num_frames, fps, video_size):
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(source),
        "-frames:v", str(num_frames),
        "-r", str(int(fps)),
    ]
    if video_size:
        side = _even_size(video_size)
        command.extend(["-vf", "scale=%d:%d" % (side, side)])
    command.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(path)])
    subprocess.run(command, check=True)
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    width, height = probe.stdout.strip().split(",")
    return int(width), int(height), "source_mp4"


def _source_video(episode):
    explicit = episode.get("video_path")
    if explicit and Path(explicit).is_file():
        return Path(explicit)
    source = episode.get("source_path")
    if not source:
        return None
    candidate = Path(source).with_suffix(".mp4")
    if candidate.is_file():
        return candidate
    return None


def _fixed_list(values, width):
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    return pa.FixedSizeListArray.from_arrays(pa.array(flat, type=pa.float32()), width)


def _pack_episode(episode, include_hand):
    kept = int(episode["num_frames"]) - 1
    if kept < 1:
        raise ValueError("%s 至少需要 2 帧才能形成下一步动作" % episode["episode_id"])
    step = action_step_dim(include_hand)
    states = np.empty((kept, STATE_DIM), dtype=np.float32)
    actions = np.empty((kept, step), dtype=np.float32)
    valids = np.empty((kept, 2), dtype=np.float32)
    object_pose = np.empty((kept, OBJECT_POSE_DIM), dtype=np.float32)
    object_valid = np.empty((kept, OBJECT_SLOTS), dtype=np.float32)
    contact = np.empty((kept, CONTACT_DIM), dtype=np.float32)
    contact_valid = np.empty((kept, 2), dtype=np.float32)
    grasp = np.empty((kept, GRASP_DIM), dtype=np.float32)
    grasp_valid = np.empty((kept, 2), dtype=np.float32)
    action_valid = np.empty((kept, 2), dtype=np.float32)
    timestamps = np.empty(kept, dtype=np.float32)
    texts = []
    for frame_index in range(kept):
        timestamp = float(episode["timestamps"][frame_index])
        texts.append(frame_task(episode, timestamp))
        states[frame_index] = pack_state(episode, frame_index)
        actions[frame_index] = pack_action(episode, frame_index, include_hand=include_hand)
        valids[frame_index] = hand_valid_flags(episode, frame_index)
        object_pose[frame_index], object_valid[frame_index] = pack_object_pose(episode, frame_index)
        contact[frame_index], contact_valid[frame_index] = pack_contact(episode, frame_index)
        grasp[frame_index], grasp_valid[frame_index] = pack_grasp(episode, frame_index)
        action_valid[frame_index] = action_valid_flags(episode, frame_index)
        # 视频是 CFR，按帧号对齐；原始时钟继续保存在源 episode，语言仍按源时钟匹配。
        timestamps[frame_index] = frame_index / float(episode["fps"])
    return {
        "episode_id": episode["episode_id"],
        "fps": float(episode["fps"]),
        "video": _source_video(episode),
        "state": states,
        "action": actions,
        "valid": valids,
        "object_pose": object_pose,
        "object_valid": object_valid,
        "contact": contact,
        "contact_valid": contact_valid,
        "grasp": grasp,
        "grasp_valid": grasp_valid,
        "events": list(episode.get("events") or []),
        "object_truncated": max(0, len(episode.get("objects") or []) - OBJECT_SLOTS),
        "action_valid": action_valid,
        "timestamp": timestamps,
        "text": texts,
        "pose_columns": _pose_columns_for_episode(episode, kept),
    }


def _pose7(matrix):
    """4×4 → ``tx ty tz qx qy qz qw``。矩阵不合法时返回 None，不写零向量。"""
    pose = np.asarray(matrix, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        return None
    quat = rotmat_to_quat_xyzw(pose[:3, :3])
    return [
        float(pose[0, 3]), float(pose[1, 3]), float(pose[2, 3]),
        float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]),
    ]


def _hands_unmeasured(episode):
    return episode.get("hands_measured") is False


def _export_accuracy(episodes):
    """测过的 Umeyama 结构原样写入说明。手写数字和头戴真值来源保持 null。"""
    from headcam.hot3d_check import shared_measured_accuracy
    return shared_measured_accuracy(episodes)


def _pose_columns_for_episode(episode, kept):
    """有逐帧位姿状态时返回要写入的列。没有这组字段就返回 None，不补 0。"""
    slam = (episode.get("hand_pose") or {}).get("slam") or {}
    statuses = slam.get("pose_status")
    matrices = slam.get("T_world_cam")
    if statuses is None and matrices is None:
        return None
    if not isinstance(statuses, list) or not isinstance(matrices, list):
        raise ValueError("%s 的 slam 位姿列不完整" % episode.get("episode_id"))
    if len(statuses) < kept or len(matrices) < kept:
        raise ValueError("%s 的 slam 位姿列短于要导出的帧" % episode.get("episode_id"))
    stamps = episode.get("timestamps")
    if not isinstance(stamps, list) or len(stamps) < kept:
        raise ValueError("%s 缺少图像时间戳，不能写 capture_time_ns" % episode.get("episode_id"))
    pose_rows = []
    status_rows = []
    capture_rows = []
    for frame_index in range(kept):
        status = statuses[frame_index]
        if status not in ("ok", "tracking_lost", "pose_missing"):
            raise ValueError("%s 第 %d 帧的 slam_status 无法识别：%s" % (
                episode.get("episode_id"), frame_index, status,
            ))
        if status == "ok":
            pose = _pose7(matrices[frame_index])
            if pose is None:
                raise ValueError("%s 第 %d 帧是 ok，但没有有限的 T_world_cam" % (
                    episode.get("episode_id"), frame_index,
                ))
        else:
            pose = None
        pose_rows.append(pose)
        status_rows.append(status)
        capture_rows.append(_capture_time_ns(stamps[frame_index]))
    return {
        "pose": pose_rows,
        "slam_status": status_rows,
        "capture_time_ns": capture_rows,
        "pose_source": slam.get("pose_source"),
    }


def _capture_time_ns(stamp):
    """图像时刻的纳秒。episode 里是秒；``round(秒 * 1e9)`` 回到整数纳秒。"""
    return int(round(float(stamp) * 1e9))


def _export_unmeasured_hands(episodes, out_dir, repo_id, video_size, horizon):
    """手部没有测量时的 LeRobot v3.0 目录。

    不写 state / action / 手腕。位姿只在 ``slam_status=ok`` 时有值，否则是 null。
    ``timestamp`` 是 ``frame_index / fps`` 的 float32，和 0.6.1 写数据集、解视频的
    容差一致。图像时刻在 ``observation.capture_time_ns``。
    """
    episodes = sorted(episodes, key=lambda item: str(item.get("episode_id")))
    fps_values = {float(item["fps"]) for item in episodes}
    if len(fps_values) != 1:
        raise ValueError("一次导出里的片段帧率必须相同，实际是 %s" % sorted(fps_values))
    fps_value = next(iter(fps_values))
    fps = int(round(fps_value))
    if abs(fps - fps_value) > 1e-3:
        raise ValueError("LeRobot 0.6.1 的 fps 是整数，无法表示 %s" % fps_value)
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon 至少为 1")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    task_to_index = {}
    pose_rows = []
    status_rows = []
    timestamp_rows = []
    capture_ns_rows = []
    frame_indices = []
    episode_indices = []
    global_indices = []
    task_indices = []
    episode_rows = []
    cursor = 0
    image_h = image_w = None
    image_source = None
    ok_frames = 0

    for episode_index, episode in enumerate(episodes):
        num_frames = int(episode["num_frames"])
        if num_frames < 1:
            raise ValueError("%s 没有帧" % episode.get("episode_id"))
        stamps = episode.get("timestamps")
        slam = (episode.get("hand_pose") or {}).get("slam") or {}
        statuses = slam.get("pose_status")
        matrices = slam.get("T_world_cam")
        if not isinstance(stamps, list) or len(stamps) != num_frames:
            raise ValueError("%s 的 timestamps 长度必须等于 num_frames" % episode.get("episode_id"))
        if not isinstance(statuses, list) or len(statuses) != num_frames:
            raise ValueError("%s 缺少逐帧 slam_status" % episode.get("episode_id"))
        if not isinstance(matrices, list) or len(matrices) != num_frames:
            raise ValueError("%s 缺少逐帧 T_world_cam" % episode.get("episode_id"))
        start = cursor
        texts = []
        for frame_index in range(num_frames):
            status = statuses[frame_index]
            if status not in ("ok", "tracking_lost", "pose_missing"):
                raise ValueError("%s 第 %d 帧的 slam_status 无法识别：%s" % (
                    episode.get("episode_id"), frame_index, status,
                ))
            if status == "ok":
                pose = _pose7(matrices[frame_index])
                if pose is None:
                    raise ValueError("%s 第 %d 帧是 ok，但没有有限的 T_world_cam" % (
                        episode.get("episode_id"), frame_index,
                    ))
                ok_frames += 1
            else:
                pose = None
            stamp = float(stamps[frame_index])
            text = frame_task(episode, stamp)
            texts.append(text)
            if text not in task_to_index:
                task_to_index[text] = len(task_to_index)
            pose_rows.append(pose)
            status_rows.append(status)
            timestamp_rows.append(np.float32(frame_index) / np.float32(fps))
            capture_ns_rows.append(_capture_time_ns(stamp))
            frame_indices.append(frame_index)
            episode_indices.append(episode_index)
            global_indices.append(cursor)
            task_indices.append(task_to_index[text])
            cursor += 1
        video_path = out_dir / "videos" / IMAGE_KEY / "chunk-000" / ("file-%03d.mp4" % episode_index)
        source = _source_video(episode)
        if source is None:
            width, height, source_kind = _write_placeholder_video(video_path, num_frames, fps)
        else:
            width, height, source_kind = _trim_source_video(
                source, video_path, num_frames, fps, video_size,
            )
        if image_w is None:
            image_w, image_h, image_source = width, height, source_kind
        elif (width, height, source_kind) != (image_w, image_h, image_source):
            raise ValueError(
                "视频尺寸不一致：%s 是 %dx%d (%s)，之前是 %dx%d (%s)"
                % (episode.get("episode_id"), width, height, source_kind, image_w, image_h, image_source)
            )
        unique_tasks = []
        for text in texts:
            if text not in unique_tasks:
                unique_tasks.append(text)
        episode_rows.append({
            "episode_index": episode_index,
            "tasks": unique_tasks,
            "length": num_frames,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": start,
            "dataset_to_index": cursor,
            "videos/%s/chunk_index" % IMAGE_KEY: 0,
            "videos/%s/file_index" % IMAGE_KEY: episode_index,
            "videos/%s/from_timestamp" % IMAGE_KEY: 0.0,
            "videos/%s/to_timestamp" % IMAGE_KEY: num_frames / float(fps),
        })

    # 固定长度列表的 null 在这份 pyarrow 里写成空列表，读回来会报错。
    # 用变长列表：ok 帧正好 7 个数，其他帧整格是 null，不是 7 个 0。
    for pose in pose_rows:
        if pose is not None and len(pose) != 7:
            raise ValueError("observation.pose 必须是 7 个数或 null")
    pose_type = pa.list_(pa.float64())
    data_path = out_dir / "data" / "chunk-000" / "file-000.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "timestamp": pa.array(np.asarray(timestamp_rows, dtype=np.float32)),
        "frame_index": pa.array(frame_indices, type=pa.int64()),
        "episode_index": pa.array(episode_indices, type=pa.int64()),
        "index": pa.array(global_indices, type=pa.int64()),
        "task_index": pa.array(task_indices, type=pa.int64()),
        "slam_status": pa.array(status_rows, type=pa.string()),
        "observation.capture_time_ns": pa.array(capture_ns_rows, type=pa.int64()),
        "observation.pose": pa.array(pose_rows, type=pose_type),
    })
    pq.write_table(table, data_path)

    names = [None] * len(task_to_index)
    for text, index in task_to_index.items():
        names[index] = text
    tasks_frame = pd.DataFrame(
        {"task_index": list(range(len(names)))},
        index=pd.Index(names, name="task"),
    )
    tasks_path = out_dir / "meta" / "tasks.parquet"
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    tasks_frame.to_parquet(tasks_path)

    episodes_frame = pd.DataFrame(episode_rows)
    episodes_path = out_dir / _EPISODES_PATH.format(chunk_index=0, file_index=0)
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    episodes_frame.to_parquet(episodes_path, index=False)

    finite_pose = np.asarray([row for row in pose_rows if row is not None], dtype=np.float64)
    timestamps = np.asarray(timestamp_rows, dtype=np.float64)
    capture_ns = np.asarray(capture_ns_rows, dtype=np.float64)
    stats = {
        IMAGE_KEY: _image_stats_json(),
        "timestamp": _stats_json(timestamps),
        "observation.capture_time_ns": _stats_json(capture_ns),
        "observation.pose": None if finite_pose.size == 0 else _stats_json(finite_pose),
    }
    (out_dir / "meta" / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    features = {
        IMAGE_KEY: _feature(
            "video",
            (image_h, image_w, 3),
            ["height", "width", "channel"],
            info={
                "video.fps": float(fps),
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
                "egodata.image_source": image_source,
            },
        ),
        "timestamp": _feature("float32", (1,), fps=fps),
        "frame_index": _feature("int64", (1,), fps=fps),
        "episode_index": _feature("int64", (1,), fps=fps),
        "index": _feature("int64", (1,), fps=fps),
        "task_index": _feature("int64", (1,), fps=fps),
        "slam_status": _feature("string", (1,), ["ok", "tracking_lost", "pose_missing"], fps=fps),
        "observation.capture_time_ns": _feature("int64", (1,), ["nanoseconds"], fps=fps),
        "observation.pose": _feature(
            "float64", (7,), ["tx", "ty", "tz", "qx", "qy", "qz", "qw"], fps=fps,
        ),
    }
    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": fps,
        "features": features,
        "total_episodes": len(episodes),
        "total_frames": cursor,
        "total_tasks": len(names),
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "data_path": _DATA_PATH,
        "video_path": _VIDEO_PATH,
        "robot_type": "headcam_pose_only",
        "splits": {"train": "0:%d" % len(episodes)},
    }
    pose_sources = {
        ((episode.get("hand_pose") or {}).get("slam") or {}).get("pose_source")
        for episode in episodes
    }
    pose_sources.discard(None)
    if len(pose_sources) == 1:
        info["pose_source"] = next(iter(pose_sources))
    (out_dir / "meta" / "info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    note = {
        "repo_id": repo_id,
        "lerobot": "0.6.1",
        "codebase_version": CODEBASE_VERSION,
        "hands": "absent",
        "hands_note": (
            "手部没有测量。parquet 里没有 observation.state、action、"
            "observation.hand_joints、observation.wrist_pose，也不写 0。"
        ),
        "accuracy": _export_accuracy(episodes),
        "pose_source": info.get("pose_source"),
        "accuracy_note": (
            "accuracy 有值时是相对真值轨迹的 Umeyama ATE/RPE，方法写在这个对象里。"
            "没有另给真值、来源是头戴真值、或只是手写数字时为 null。"
        ),
        "slam_status": "逐帧标签：ok、tracking_lost 或 pose_missing。frame_label 就是这一列。",
        "pose": (
            "observation.pose 是 tx ty tz qx qy qz qw，只在 slam_status=ok 时有值。"
            "其他帧是 null，不是零向量，也不是单位四元数。"
        ),
        "timestamp_doc": (
            "timestamp 是 frame_index/fps，float32，给 LeRobot 0.6.1 的视频容差检查用。"
            "图像时刻在 observation.capture_time_ns，单位纳秒，int64。"
            "不把绝对时间放进 timestamp：加载器拿它去对视频时间轴，"
            "容差默认 1e-4 秒，1e9 秒量级对不上，而且 float32 会把这一量级收成同一个数。"
        ),
        "ok_frames": ok_frames,
        "frames": cursor,
        "horizon": horizon,
        "qc": "只含 yield CSV 中 accepted=yes 的 episode_id",
        "image_source": image_source,
        "video_size": None if image_source != "source_mp4" else video_size,
        "language": "没有指令时 task 文本是 unknown，不编造任务名。",
    }
    (out_dir / "meta" / "egodata_export.json").write_text(
        json.dumps(note, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (out_dir / "meta" / "interaction_events.jsonl").write_text("", encoding="utf-8")
    return {
        "episodes": len(episodes),
        "frames": cursor,
        "ok_frames": ok_frames,
        "tasks": len(names),
        "out": str(out_dir),
        "horizon": horizon,
        "hands": "absent",
        "accuracy": _export_accuracy(episodes),
        "valid_action_ratio": None,
        "valid_action_count": 0,
        "action_count": 0,
        "valid_full_chunk_ratio": {16: None, 50: None, 100: None},
    }


def export_lerobot(
    episodes_dir,
    yield_csv,
    out_dir,
    repo_id="local/egodex",
    include_hand=False,
    video_size=DEFAULT_VIDEO_SIZE,
    horizon=DEFAULT_HORIZON,
):
    """导出 QC 通过的片段。返回写入的片段数和帧数。

    ``video_size`` 只作用于真实 mp4，写成正方形。``None`` 表示不缩放。
    ``horizon`` 写入导出说明，供线性 BC 和 ACT 的 chunk 使用；parquet 里
    每一行仍是一步增量。
    """
    accepted = accepted_episode_ids(yield_csv)
    selected = []
    for path in iter_episode_paths(episodes_dir):
        episode = load_episode(path)
        if episode.get("episode_id") not in accepted:
            continue
        selected.append(episode)
    if not selected:
        raise ValueError("产出率 CSV 里没有可导出的片段")
    unmeasured = [_hands_unmeasured(episode) for episode in selected]
    if any(unmeasured) and not all(unmeasured):
        raise ValueError(
            "同一次导出不能混有手部测量和未测量手部。未测量的片段不写 0 手。"
        )
    if all(unmeasured):
        return _export_unmeasured_hands(selected, out_dir, repo_id, video_size, horizon)
    packed = []
    for episode in selected:
        packed.append(_pack_episode(episode, include_hand))
    packed.sort(key=lambda item: item["episode_id"])
    if not packed:
        raise ValueError("产出率 CSV 里没有可导出的片段")
    fps_values = {item["fps"] for item in packed}
    if len(fps_values) != 1:
        raise ValueError("一次导出里的片段帧率必须相同，实际是 %s" % sorted(fps_values))
    fps_value = next(iter(fps_values))
    fps = int(round(fps_value))
    if abs(fps - fps_value) > 1e-3:
        raise ValueError("LeRobot 0.6.1 的 fps 是整数，无法表示 %s" % fps_value)
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon 至少为 1")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    step = action_step_dim(include_hand)
    task_to_index = {}
    state_blocks = []
    action_blocks = []
    valid_blocks = []
    object_blocks = []
    object_valid_blocks = []
    contact_blocks = []
    contact_valid_blocks = []
    grasp_blocks = []
    grasp_valid_blocks = []
    action_valid_blocks = []
    timestamp_blocks = []
    pose_blocks = []
    event_rows = []
    truncated = 0
    frame_indices = []
    episode_indices = []
    global_indices = []
    task_indices = []
    episode_rows = []
    cursor = 0
    image_h = image_w = None
    image_source = None

    for episode_index, item in enumerate(packed):
        kept = int(item["state"].shape[0])
        start = cursor
        texts = item["text"]
        for text in texts:
            if text not in task_to_index:
                task_to_index[text] = len(task_to_index)
        state_blocks.append(item["state"])
        action_blocks.append(item["action"])
        valid_blocks.append(item["valid"])
        object_blocks.append(item["object_pose"])
        object_valid_blocks.append(item["object_valid"])
        contact_blocks.append(item["contact"])
        contact_valid_blocks.append(item["contact_valid"])
        grasp_blocks.append(item["grasp"])
        grasp_valid_blocks.append(item["grasp_valid"])
        action_valid_blocks.append(item["action_valid"])
        timestamp_blocks.append(item["timestamp"])
        if item.get("pose_columns") is not None:
            pose_blocks.append(item["pose_columns"])
        truncated += int(item["object_truncated"])
        event_rows.append({
            "episode_index": episode_index,
            "episode_id": item["episode_id"],
            "events": item["events"],
        })
        frame_indices.extend(range(kept))
        episode_indices.extend([episode_index] * kept)
        global_indices.extend(range(cursor, cursor + kept))
        task_indices.extend(task_to_index[text] for text in texts)
        cursor += kept
        video_path = out_dir / "videos" / IMAGE_KEY / "chunk-000" / ("file-%03d.mp4" % episode_index)
        if item["video"] is None:
            width, height, source_kind = _write_placeholder_video(video_path, kept, fps)
        else:
            width, height, source_kind = _trim_source_video(
                item["video"], video_path, kept, fps, video_size,
            )
        if image_w is None:
            image_w, image_h, image_source = width, height, source_kind
        elif (width, height, source_kind) != (image_w, image_h, image_source):
            raise ValueError(
                "视频尺寸不一致：%s 是 %dx%d (%s)，之前是 %dx%d (%s)"
                % (item["episode_id"], width, height, source_kind, image_w, image_h, image_source)
            )
        unique_tasks = []
        for text in texts:
            if text not in unique_tasks:
                unique_tasks.append(text)
        episode_rows.append({
            "episode_index": episode_index,
            "tasks": unique_tasks,
            "length": kept,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": start,
            "dataset_to_index": cursor,
            "videos/%s/chunk_index" % IMAGE_KEY: 0,
            "videos/%s/file_index" % IMAGE_KEY: episode_index,
            "videos/%s/from_timestamp" % IMAGE_KEY: 0.0,
            "videos/%s/to_timestamp" % IMAGE_KEY: kept / float(fps),
        })

    states = np.concatenate(state_blocks, axis=0)
    actions = np.concatenate(action_blocks, axis=0)
    valids = np.concatenate(valid_blocks, axis=0)
    object_pose = np.concatenate(object_blocks, axis=0)
    object_valid = np.concatenate(object_valid_blocks, axis=0)
    contact = np.concatenate(contact_blocks, axis=0)
    contact_valid = np.concatenate(contact_valid_blocks, axis=0)
    grasp = np.concatenate(grasp_blocks, axis=0)
    grasp_valid = np.concatenate(grasp_valid_blocks, axis=0)
    action_valids = np.concatenate(action_valid_blocks, axis=0)
    timestamps = np.concatenate(timestamp_blocks, axis=0)
    validity = validity_summary(action_valids)
    if pose_blocks and len(pose_blocks) != len(packed):
        raise ValueError("同一次导出里，有的片段有逐帧位姿，有的没有。")
    pose_rows = []
    status_rows = []
    capture_rows = []
    pose_source = None
    if pose_blocks:
        sources = []
        for block in pose_blocks:
            pose_rows.extend(block["pose"])
            status_rows.extend(block["slam_status"])
            capture_rows.extend(block["capture_time_ns"])
            sources.append(block.get("pose_source"))
        unique_sources = {item for item in sources if item}
        if len(unique_sources) > 1:
            raise ValueError("同一次导出里的 pose_source 不一致：%s" % sorted(unique_sources))
        pose_source = sources[0] if sources else None

    data_path = out_dir / "data" / "chunk-000" / "file-000.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    columns = {
        "observation.state": _fixed_list(states, STATE_DIM),
        "observation.hand_valid": _fixed_list(valids, 2),
        "observation.object_pose": _fixed_list(object_pose, OBJECT_POSE_DIM),
        "observation.object_pose_valid": _fixed_list(object_valid, OBJECT_SLOTS),
        "observation.contact": _fixed_list(contact, CONTACT_DIM),
        "observation.contact_valid": _fixed_list(contact_valid, 2),
        "action": _fixed_list(actions, step),
        "action.grasp": _fixed_list(grasp, GRASP_DIM),
        "action.grasp_valid": _fixed_list(grasp_valid, 2),
        "action_valid": _fixed_list(action_valids, 2),
        "timestamp": pa.array(timestamps.astype(np.float32)),
        "frame_index": pa.array(frame_indices, type=pa.int64()),
        "episode_index": pa.array(episode_indices, type=pa.int64()),
        "index": pa.array(global_indices, type=pa.int64()),
        "task_index": pa.array(task_indices, type=pa.int64()),
    }
    if pose_blocks:
        for pose in pose_rows:
            if pose is not None and len(pose) != 7:
                raise ValueError("observation.pose 必须是 7 个数或 null")
        columns["slam_status"] = pa.array(status_rows, type=pa.string())
        columns["observation.capture_time_ns"] = pa.array(capture_rows, type=pa.int64())
        columns["observation.pose"] = pa.array(pose_rows, type=pa.list_(pa.float64()))
    table = pa.table(columns)
    pq.write_table(table, data_path)

    names = [None] * len(task_to_index)
    for text, index in task_to_index.items():
        names[index] = text
    tasks_frame = pd.DataFrame(
        {"task_index": list(range(len(names)))},
        index=pd.Index(names, name="task"),
    )
    tasks_path = out_dir / "meta" / "tasks.parquet"
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    tasks_frame.to_parquet(tasks_path)

    episodes_frame = pd.DataFrame(episode_rows)
    episodes_path = out_dir / _EPISODES_PATH.format(chunk_index=0, file_index=0)
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    episodes_frame.to_parquet(episodes_path, index=False)

    stats = {
        IMAGE_KEY: _image_stats_json(),
        "observation.state": _stats_json(states),
        "observation.hand_valid": _stats_json(valids),
        "observation.object_pose": _stats_json(object_pose),
        "observation.object_pose_valid": _stats_json(object_valid),
        "observation.contact": _stats_json(contact),
        "observation.contact_valid": _stats_json(contact_valid),
        "action": _stats_json_where(actions, action_dim_mask(action_valids, step)),
        "action.grasp": _stats_json(grasp),
        "action.grasp_valid": _stats_json(grasp_valid),
        "action_valid": _stats_json(action_valids),
        "timestamp": _stats_json(timestamps),
    }
    if pose_blocks:
        finite_pose = np.asarray([row for row in pose_rows if row is not None], dtype=np.float64)
        stats["observation.capture_time_ns"] = _stats_json(np.asarray(capture_rows, dtype=np.float64))
        stats["observation.pose"] = None if finite_pose.size == 0 else _stats_json(finite_pose)
    (out_dir / "meta" / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    features = {
        IMAGE_KEY: _feature(
            "video",
            (image_h, image_w, 3),
            ["height", "width", "channel"],
            info={
                "video.fps": float(fps),
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
                "egodata.image_source": image_source,
            },
        ),
        "observation.state": _feature("float32", (STATE_DIM,), fps=fps),
        "observation.hand_valid": _feature("float32", (2,), fps=fps),
        "observation.object_pose": _feature("float32", (OBJECT_POSE_DIM,), fps=fps),
        "observation.object_pose_valid": _feature("float32", (OBJECT_SLOTS,), ["slot_%d" % i for i in range(OBJECT_SLOTS)], fps=fps),
        "observation.contact": _feature(
            "float32", (CONTACT_DIM,), ["left_slot", "left_confidence", "right_slot", "right_confidence"], fps=fps,
        ),
        "observation.contact_valid": _feature("float32", (2,), ["left", "right"], fps=fps),
        "action": _feature("float32", (step,), fps=fps),
        "action.grasp": _feature("float32", (GRASP_DIM,), ["left", "right"], fps=fps),
        "action.grasp_valid": _feature("float32", (2,), ["left", "right"], fps=fps),
        "action_valid": _feature("float32", (2,), names=["left", "right"], fps=fps),
        "timestamp": _feature("float32", (1,), fps=fps),
        "frame_index": _feature("int64", (1,), fps=fps),
        "episode_index": _feature("int64", (1,), fps=fps),
        "index": _feature("int64", (1,), fps=fps),
        "task_index": _feature("int64", (1,), fps=fps),
    }
    if pose_blocks:
        features["slam_status"] = _feature("string", (1,), ["ok", "tracking_lost", "pose_missing"], fps=fps)
        features["observation.capture_time_ns"] = _feature("int64", (1,), ["nanoseconds"], fps=fps)
        features["observation.pose"] = _feature(
            "float64", (7,), ["tx", "ty", "tz", "qx", "qy", "qz", "qw"], fps=fps,
        )
    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": fps,
        "features": features,
        "total_episodes": len(packed),
        "total_frames": cursor,
        "total_tasks": len(names),
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "data_path": _DATA_PATH,
        "video_path": _VIDEO_PATH,
        "robot_type": "egodex_human",
        "splits": {"train": "0:%d" % len(packed)},
    }
    if pose_source:
        info["pose_source"] = pose_source
    (out_dir / "meta" / "info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    note = {
        "repo_id": repo_id,
        "lerobot": "0.6.1",
        "codebase_version": CODEBASE_VERSION,
        "action": "wrist_pose_delta",
        "action_doc": (
            "每一行 action 是相对当前手腕的下一步增量："
            "左手 dxyz+相对四元数，再接右手。保持不动 = dxyz 0 且四元数 0,0,0,1。"
            "多步目标是连续 horizon 行，不把整段 chunk 摊进这一列。"
        ),
        "horizon": horizon,
        "include_hand": bool(include_hand),
        "action_step_dim": step,
        "timestamp_doc": "timestamp 是视频内 frame_index/fps；原始采集时间戳和语言时段保留在源 episode JSON。",
        "state_doc": "state[0:63] 左手关节，[63:126] 右手关节，[126:133] 左手腕，[133:140] 右手腕。",
        "language": "SUBTASK 覆盖该帧时用子任务，否则 TASK.instruction",
        "qc": "只含 yield CSV 中 accepted=yes 的 episode_id",
        "image_source": image_source,
        "video_size": None if image_source != "source_mp4" else video_size,
        "confidence": "缺失的手腕置信度视为未知，不把关节置 0",
        "object_pose": (
            "observation.object_pose 长度 %d = %d 个槽位 × (xyz + xyzw)。"
            "按物体 id 排序，多出来的丢掉。无效槽位是 0，要看 observation.object_pose_valid。"
            % (OBJECT_POSE_DIM, OBJECT_SLOTS)
        ),
        "object_slots": OBJECT_SLOTS,
        "object_tracks_truncated": truncated,
        "contact": "observation.contact：左手槽位和置信度，再接右手。没有接触时槽位是 -1。掩码是 observation.contact_valid。",
        "grasp": "action.grasp 是当前帧状态，不是下一步增量。0 张开，1 预备，2 抓住，3 放开。掩码是 action.grasp_valid。",
        "validity_masks": "object_pose_valid、contact_valid、grasp_valid、action_valid 与 observation.hand_valid 一样，0 表示不要把对应的 0 当成测量值。",
        "action_valid_doc": (
            "action_valid[左, 右] 为 1 时，这一步该手的增量来自实测手腕："
            "当前帧和下一帧都有限、不是 filled、逐帧标注不是丢掉的状态、"
            "置信度不是低于 0.5 的数字。episode 上若有 good_frame_mask，"
            "false 的帧两只手都为 0。占位的保持不动仍然写在 action 里。"
        ),
        "valid_action_ratio": validity["valid_action_ratio"],
        "valid_full_chunk_ratio": {
            str(length): validity["valid_full_chunk_ratio_%d" % length]
            for length in (16, 50, 100)
        },
    }
    if pose_blocks:
        note["pose_source"] = pose_source
        note["accuracy"] = _export_accuracy(selected)
        note["accuracy_note"] = (
            "accuracy 有值时是相对真值轨迹的 Umeyama ATE/RPE。"
            "没有这份测量、来源是头戴真值、或 episode 里只写了一个数字时为 null。"
        )
        note["pose"] = (
            "observation.pose 是 tx ty tz qx qy qz qw，只在 slam_status=ok 时有值。"
            "其他帧是 null，不是零向量。"
        )
        note["capture_time_ns"] = (
            "observation.capture_time_ns 是图像时刻的纳秒。"
            "timestamp 仍是 frame_index/fps，给 LeRobot 0.6.1 对视频用。"
        )
        note["frame_count_note"] = (
            "parquet 行数比 episode 的 num_frames 少 1：最后一帧没有下一步 action，整行不写。"
        )
    events_path = out_dir / "meta" / "interaction_events.jsonl"
    events_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in event_rows),
        encoding="utf-8",
    )
    (out_dir / "meta" / "egodata_export.json").write_text(
        json.dumps(note, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return {
        "episodes": len(packed),
        "frames": cursor,
        "tasks": len(names),
        "out": str(out_dir),
        "horizon": horizon,
        "action_step_dim": step,
        "valid_action_ratio": validity["valid_action_ratio"],
        "valid_action_count": validity["valid_action_count"],
        "action_count": validity["action_count"],
        "valid_full_chunk_ratio": chunk_ratio_map(validity),
    }
