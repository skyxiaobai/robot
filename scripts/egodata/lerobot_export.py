# -*- coding: utf-8 -*-
"""把统一 episode 写成 LeRobot 0.6.1 能读的 v3.0 数据集。

只写入 QC 产出率 CSV 里 ``accepted=yes`` 的片段。

向量约定（世界系，与统一 episode 相同）：

- ``observation.state`` 长度 140 =
  左手 21×3 关节、右手 21×3 关节、左手腕 7 维、右手腕 7 维。
  某只手有缺测关节或手腕置信度 < 0.5 时，该手的 63 维关节置 0，
  ``observation.hand_valid`` 对应侧为 0。手腕 7 维只要数字齐全就保留。
- ``action`` 长度 14 = **下一帧**的左手腕 7 维再接右手腕 7 维
  （xyz 米 + 四元数 xyzw）。每个片段的最后一帧没有下一帧，直接丢掉，
  不把当前姿态复制成动作。
- 语言写在 ``task_index``。该帧时刻落在某条 SUBTASK 里就用子任务句子，
  否则用 TASK.instruction。

没有现成 mp4 时写 16×16 的占位视频，并在特征 info 里标
``egodata.image_source=placeholder``。旁边若有与 ``source_path`` 同名的
``.mp4``，则裁到保留的帧数后放进数据集。
"""
import csv
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from egodata.schema import iter_episode_paths, load_episode

CODEBASE_VERSION = "v3.0"
JOINTS = 21
WRIST_DIM = 7
STATE_DIM = 2 * (JOINTS * 3 + WRIST_DIM)
ACTION_DIM = 2 * WRIST_DIM
IMAGE_KEY = "observation.image"
PLACEHOLDER_SIZE = 16
_CONFIDENCE_MIN = 0.5

_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
_EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"


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


def hand_valid_flags(episode, frame_index):
    """左手、右手是否整手可用。返回 float32[2]。"""
    flags = []
    for side in ("left", "right"):
        hand = episode["hands"][side]
        confidence = hand["confidence"][frame_index]
        joints = hand["joints"][frame_index]
        wrist = _finite_vec(hand["wrist_pose"][frame_index], WRIST_DIM)
        usable = (
            wrist is not None
            and _joints_ok(joints)
            and confidence is not None
            and float(confidence) >= _CONFIDENCE_MIN
        )
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


def pack_action(episode, frame_index):
    """下一帧双手手腕。已经是最后一帧时返回 None。"""
    nxt = frame_index + 1
    if nxt >= episode["num_frames"]:
        return None
    parts = []
    for side in ("left", "right"):
        wrist = _finite_vec(episode["hands"][side]["wrist_pose"][nxt], WRIST_DIM)
        parts.append(wrist if wrist is not None else np.zeros(WRIST_DIM, dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)


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
    std = values.std(axis=0)
    return {
        "min": mean * 0 + values.min(axis=0),  # keep ndarray
        "max": values.max(axis=0),
        "mean": mean,
        "std": std,
        "count": np.array([count]),
    }


def _stats_json(array):
    raw = _stats(array)
    return {key: np.asarray(value).reshape(-1).tolist() for key, value in raw.items()}


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


def _trim_source_video(source, path, num_frames, fps):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(source),
            "-frames:v", str(num_frames),
            "-r", str(int(fps)),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
            str(path),
        ],
        check=True,
    )
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


def _fixed_list(rows, width):
    flat = np.asarray(rows, dtype=np.float32).reshape(-1)
    return pa.FixedSizeListArray.from_arrays(pa.array(flat, type=pa.float32()), width)


def export_lerobot(episodes_dir, yield_csv, out_dir, repo_id="local/egodex"):
    """导出 QC 通过的片段。返回写入的片段数和帧数。"""
    accepted = accepted_episode_ids(yield_csv)
    episodes = []
    for path in iter_episode_paths(episodes_dir):
        episode = load_episode(path)
        if episode.get("episode_id") in accepted:
            episodes.append(episode)
    episodes.sort(key=lambda item: item["episode_id"])
    if not episodes:
        raise ValueError("产出率 CSV 里没有可导出的片段")
    fps_values = {float(item["fps"]) for item in episodes}
    if len(fps_values) != 1:
        raise ValueError("一次导出里的片段帧率必须相同，实际是 %s" % sorted(fps_values))
    fps = int(round(next(iter(fps_values))))
    if abs(fps - next(iter(fps_values))) > 1e-3:
        raise ValueError("LeRobot 0.6.1 的 fps 是整数，无法表示 %s" % next(iter(fps_values)))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    task_to_index = {}
    states, actions, valids = [], [], []
    timestamps, frame_indices, episode_indices, global_indices, task_indices = [], [], [], [], []
    episode_rows = []
    cursor = 0
    image_h = image_w = PLACEHOLDER_SIZE
    image_source = "placeholder"

    for episode_index, episode in enumerate(episodes):
        kept = episode["num_frames"] - 1
        if kept < 1:
            raise ValueError("%s 至少需要 2 帧才能形成下一帧动作" % episode["episode_id"])
        start = cursor
        texts = []
        for frame_index in range(kept):
            timestamp = float(episode["timestamps"][frame_index])
            text = frame_task(episode, timestamp)
            if text not in task_to_index:
                task_to_index[text] = len(task_to_index)
            texts.append(text)
            states.append(pack_state(episode, frame_index))
            actions.append(pack_action(episode, frame_index))
            valids.append(hand_valid_flags(episode, frame_index))
            timestamps.append(timestamp)
            frame_indices.append(frame_index)
            episode_indices.append(episode_index)
            global_indices.append(cursor)
            task_indices.append(task_to_index[text])
            cursor += 1
        video_path = (
            out_dir / "videos" / IMAGE_KEY / "chunk-000" / ("file-%03d.mp4" % episode_index)
        )
        source = _source_video(episode)
        if source is None:
            image_w, image_h, image_source = _write_placeholder_video(video_path, kept, fps)
        else:
            image_w, image_h, image_source = _trim_source_video(source, video_path, kept, fps)
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

    data_path = out_dir / "data" / "chunk-000" / "file-000.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "observation.state": _fixed_list(states, STATE_DIM),
        "observation.hand_valid": _fixed_list(valids, 2),
        "action": _fixed_list(actions, ACTION_DIM),
        "timestamp": pa.array(np.asarray(timestamps, dtype=np.float32)),
        "frame_index": pa.array(frame_indices, type=pa.int64()),
        "episode_index": pa.array(episode_indices, type=pa.int64()),
        "index": pa.array(global_indices, type=pa.int64()),
        "task_index": pa.array(task_indices, type=pa.int64()),
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

    state_arr = np.stack(states)
    action_arr = np.stack(actions)
    valid_arr = np.stack(valids)
    stats = {
        "observation.state": _stats_json(state_arr),
        "observation.hand_valid": _stats_json(valid_arr),
        "action": _stats_json(action_arr),
        "timestamp": _stats_json(np.asarray(timestamps, dtype=np.float32)),
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
        "observation.state": _feature("float32", (STATE_DIM,), fps=fps),
        "observation.hand_valid": _feature("float32", (2,), fps=fps),
        "action": _feature("float32", (ACTION_DIM,), fps=fps),
        "timestamp": _feature("float32", (1,), fps=fps),
        "frame_index": _feature("int64", (1,), fps=fps),
        "episode_index": _feature("int64", (1,), fps=fps),
        "index": _feature("int64", (1,), fps=fps),
        "task_index": _feature("int64", (1,), fps=fps),
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
        "robot_type": "egodex_human",
        "splits": {"train": "0:%d" % len(episodes)},
    }
    (out_dir / "meta" / "info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (out_dir / "meta" / "egodata_export.json").write_text(
        json.dumps({
            "repo_id": repo_id,
            "lerobot": "0.6.1",
            "codebase_version": CODEBASE_VERSION,
            "action": "next_wrist_pose",
            "action_doc": "action[0:7] 下一帧左手腕 xyz+xyzw，action[7:14] 右手腕。片段最后一帧已丢弃。",
            "state_doc": "state[0:63] 左手关节，[63:126] 右手关节，[126:133] 左手腕，[133:140] 右手腕。",
            "language": "SUBTASK 覆盖该帧时用子任务，否则 TASK.instruction",
            "qc": "只含 yield CSV 中 accepted=yes 的 episode_id",
            "image_source": image_source,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"episodes": len(episodes), "frames": cursor, "tasks": len(names), "out": str(out_dir)}
