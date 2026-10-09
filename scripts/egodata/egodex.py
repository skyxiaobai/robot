# -*- coding: utf-8 -*-
"""把 EgoDex 的 HDF5（ARKit 世界系 SE(3)）收成统一 episode。

已对照公开 test.zip 里的
``test/open_close_insert_remove_case/8.hdf5``：
``transforms/<joint>`` 为 N×4×4，平移在最后一列；
``camera/intrinsic`` 主点 (960, 540) 对应 1920×1080；
属性里有 environment、task、llm_description、llm_objects、llm_verbs。
位姿是录制时 ARKit 的静止世界系，本模块不重新跑 SLAM。

没有 ``confidences`` 组时，手腕置信度记为未知（JSON 里是 null），不当成 0。
未知表示「源数据没写这个通道」：关节齐全就算这只手可用，QC 只靠投影判断出画。
读到了低于 0.5 的数字才算低置信。
"""
from pathlib import Path

import numpy as np

from egodata.coverage import (
    coarse_object_class,
    normalize_action,
    normalize_environment,
    normalize_object_name,
)
from egodata.schema import (
    SCHEMA_VERSION,
    make_quaternions_continuous,
    rotmat_to_quat_xyzw,
    save_episode,
)

EGODEX_FPS = 30.0

# MediaPipe 21 点 ← EgoDex / ARKit 关节名（不含左右前缀）。
_JOINT_SUFFIXES = (
    "Hand",
    "ThumbKnuckle",
    "ThumbIntermediateBase",
    "ThumbIntermediateTip",
    "ThumbTip",
    "IndexFingerKnuckle",
    "IndexFingerIntermediateBase",
    "IndexFingerIntermediateTip",
    "IndexFingerTip",
    "MiddleFingerKnuckle",
    "MiddleFingerIntermediateBase",
    "MiddleFingerIntermediateTip",
    "MiddleFingerTip",
    "RingFingerKnuckle",
    "RingFingerIntermediateBase",
    "RingFingerIntermediateTip",
    "RingFingerTip",
    "LittleFingerKnuckle",
    "LittleFingerIntermediateBase",
    "LittleFingerIntermediateTip",
    "LittleFingerTip",
)


def _text(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.ndarray):
        return [_text(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _text(value.item())
    return str(value)


def _attr(handle, key, default=""):
    if key not in handle.attrs:
        return default
    return _text(handle.attrs[key])


def active_language(attrs):
    """可逆任务用 which_llm_description 选择 llm_description / llm_description2。"""
    llm_type = _text(attrs.get("llm_type", "")).strip().lower()
    primary = _text(attrs.get("llm_description", "")).strip()
    secondary = _text(attrs.get("llm_description2", "")).strip()
    if llm_type == "reversible":
        which = _text(attrs.get("which_llm_description", "1")).strip()
        if which in ("2", "02"):
            return secondary or primary
    return primary or secondary


def _string_list(value):
    text = _text(value)
    if isinstance(text, list):
        return [item.strip() for item in text if str(item).strip()]
    if not text:
        return []
    return [part.strip() for part in str(text).replace(";", ",").split(",") if part.strip()]


def _image_size(intrinsic):
    width = int(round(float(intrinsic[0, 2]) * 2))
    height = int(round(float(intrinsic[1, 2]) * 2))
    return max(width, 1), max(height, 1)


def _column_translations(transforms, name, num_frames):
    """一次读出 N×4×4，只留平移。没有这个关节时返回 None。"""
    if name not in transforms:
        return None
    matrices = np.asarray(transforms[name])
    if matrices.shape[0] != num_frames or matrices.shape[-2:] != (4, 4):
        raise ValueError("%s 的形状应为 (%d, 4, 4)，实际是 %s" % (name, num_frames, matrices.shape))
    return np.asarray(matrices[:, :3, 3], dtype=np.float64)


def _wrist_poses(transforms, wrist_name, num_frames):
    """手腕 xyz + 连续四元数。整段缺失时返回 None。"""
    if wrist_name not in transforms:
        return None
    matrices = np.asarray(transforms[wrist_name], dtype=np.float64)
    if matrices.shape[0] != num_frames or matrices.shape[-2:] != (4, 4):
        raise ValueError("%s 的形状应为 (%d, 4, 4)，实际是 %s" % (wrist_name, num_frames, matrices.shape))
    quats = np.stack([rotmat_to_quat_xyzw(matrices[index, :3, :3]) for index in range(num_frames)])
    quats = make_quaternions_continuous(quats)
    return np.concatenate([matrices[:, :3, 3], quats], axis=1)


def _confidence_column(conf_group, wrist_name, num_frames):
    """没有 confidences 组、或没有该手腕时，整列记为未知（None），不当成 0。"""
    if conf_group is None or wrist_name not in conf_group:
        return [None] * num_frames
    values = np.asarray(conf_group[wrist_name], dtype=np.float64).reshape(-1)
    if values.shape[0] != num_frames:
        raise ValueError("%s 置信度长度应为 %d，实际是 %d" % (wrist_name, num_frames, values.shape[0]))
    column = []
    for value in values:
        if not np.isfinite(value):
            column.append(None)
        else:
            column.append(float(value))
    return column


def _hand_series(handle, prefix, num_frames):
    transforms = handle["transforms"]
    conf_group = handle["confidences"] if "confidences" in handle else None
    wrist_name = prefix + "Hand"
    columns = [_column_translations(transforms, prefix + suffix, num_frames) for suffix in _JOINT_SUFFIXES]
    poses = _wrist_poses(transforms, wrist_name, num_frames)
    confidence = _confidence_column(conf_group, wrist_name, num_frames)
    joints = []
    wrist_pose = []
    valid = []
    for frame_index in range(num_frames):
        frame_joints = []
        for column in columns:
            if column is None or not np.isfinite(column[frame_index]).all():
                frame_joints.append(None)
            else:
                frame_joints.append([float(value) for value in column[frame_index]])
        if poses is None or not np.isfinite(poses[frame_index]).all():
            pose = None
        else:
            pose = [float(value) for value in poses[frame_index]]
        conf = confidence[frame_index]
        finite = pose is not None and all(point is not None for point in frame_joints)
        # 置信度未知时，关节齐全即视为可用；读到了数字才用 0.5 阈值。
        confident = conf is None or conf >= 0.5
        joints.append(frame_joints)
        wrist_pose.append(pose if pose is not None else [None] * 7)
        valid.append(bool(finite and confident))
    return {
        "joints": joints,
        "wrist_pose": wrist_pose,
        "confidence": confidence,
        "valid": valid,
    }


def load_episode_hdf5(path, episode_id=None):
    """读取一条 EgoDex HDF5，返回统一 episode 字典。"""
    import h5py

    path = Path(path)
    with h5py.File(path, "r") as handle:
        intrinsic = np.asarray(handle["camera/intrinsic"], dtype=float)
        camera = np.asarray(handle["transforms/camera"], dtype=float)
        num_frames = int(camera.shape[0])
        width, height = _image_size(intrinsic)
        task_name = _attr(handle, "task", path.parent.name).strip() or path.parent.name
        environment_raw = _attr(handle, "environment", "")
        objects = _string_list(handle.attrs["llm_objects"]) if "llm_objects" in handle.attrs else []
        objects = [normalize_object_name(name) for name in objects]
        verbs = _string_list(handle.attrs["llm_verbs"]) if "llm_verbs" in handle.attrs else []
        instruction = active_language({key: handle.attrs[key] for key in handle.attrs})
        hands = {
            "left": _hand_series(handle, "left", num_frames),
            "right": _hand_series(handle, "right", num_frames),
        }
    if episode_id is None:
        episode_id = "egodex/%s/%s" % (task_name, path.stem)
    environment = normalize_environment(environment_raw)
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
    return {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "source": "egodex",
        "source_path": str(path),
        "fps": EGODEX_FPS,
        "coordinate_frame": "arkit_world",
        "image_width": width,
        "image_height": height,
        "num_frames": num_frames,
        "timestamps": [frame_index / EGODEX_FPS for frame_index in range(num_frames)],
        "camera_intrinsic": intrinsic.tolist(),
        "camera_poses": camera.tolist(),
        "hands": hands,
        "annotation": {
            "environment": {
                "name": environment,
                "detail": environment_raw,
                "source": "egodex_attr" if environment_raw else "missing",
            },
            "task": {"name": task_name, "instruction": instruction},
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
    }


def _convert_one(job):
    path, episode_id, destination = job
    episode = load_episode_hdf5(path, episode_id=episode_id)
    save_episode(episode, destination)
    return destination


def convert_tree(root, out_dir, limit=None, workers=1):
    """把目录下的 ``*.hdf5`` 写成统一 JSON。``limit`` 只转换前若干条。

    ``workers > 1`` 时按文件多进程转换。输出路径和 episode_id 与单进程相同。
    """
    root = Path(root)
    out_dir = Path(out_dir)
    files = sorted(root.rglob("*.hdf5"))
    if limit is not None:
        files = files[: int(limit)]
    jobs = []
    for path in files:
        relative = path.relative_to(root).with_suffix("")
        episode_id = "egodex/" + relative.as_posix()
        destination = out_dir / relative.with_suffix(".json")
        jobs.append((str(path), episode_id, str(destination)))
    if not jobs:
        return []
    worker_count = max(1, int(workers))
    if worker_count == 1:
        written = [_convert_one(job) for job in jobs]
    else:
        import multiprocessing

        with multiprocessing.Pool(worker_count) as pool:
            written = pool.map(_convert_one, jobs, chunksize=4)
    return [Path(path) for path in written]
