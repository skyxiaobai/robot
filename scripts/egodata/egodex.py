# -*- coding: utf-8 -*-
"""把 EgoDex 的 HDF5（ARKit 世界系 SE(3)）收成统一 episode。

已对照公开 test.zip 里的
``test/open_close_insert_remove_case/8.hdf5``：
``transforms/<joint>`` 为 N×4×4，平移在最后一列；
``camera/intrinsic`` 主点 (960, 540) 对应 1920×1080；
属性里有 environment、task、llm_description、llm_objects、llm_verbs。
位姿是录制时 ARKit 的静止世界系，本模块不重新跑 SLAM。
"""
from pathlib import Path

import numpy as np

from egodata.coverage import coarse_object_class, normalize_action, normalize_environment
from egodata.schema import SCHEMA_VERSION, rotmat_to_quat_xyzw, save_episode

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


def _hand_series(handle, prefix, num_frames):
    joints = []
    wrist_pose = []
    confidence = []
    valid = []
    transforms = handle["transforms"]
    conf_group = handle["confidences"] if "confidences" in handle else None
    wrist_name = prefix + "Hand"
    for frame_index in range(num_frames):
        frame_joints = []
        for suffix in _JOINT_SUFFIXES:
            name = prefix + suffix
            if name not in transforms:
                frame_joints.append(None)
                continue
            translation = transforms[name][frame_index][:3, 3]
            frame_joints.append([float(v) for v in translation])
        if wrist_name in transforms:
            matrix = transforms[wrist_name][frame_index]
            quat = rotmat_to_quat_xyzw(matrix[:3, :3])
            translation = [float(v) for v in matrix[:3, 3]]
            pose = translation + quat
        else:
            pose = None
        if conf_group is not None and wrist_name in conf_group:
            conf = float(conf_group[wrist_name][frame_index])
        else:
            conf = 0.0
        finite = pose is not None and all(point is not None for point in frame_joints)
        confidence.append(conf)
        wrist_pose.append(pose if pose is not None else [None] * 7)
        joints.append(frame_joints)
        valid.append(bool(finite and conf >= 0.5))
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


def convert_tree(root, out_dir, limit=None):
    """把目录下的 ``*.hdf5`` 写成统一 JSON。``limit`` 只转换前若干条。"""
    root = Path(root)
    out_dir = Path(out_dir)
    files = sorted(root.rglob("*.hdf5"))
    if limit is not None:
        files = files[: int(limit)]
    written = []
    for path in files:
        relative = path.relative_to(root).with_suffix("")
        episode_id = "egodex/" + relative.as_posix()
        episode = load_episode_hdf5(path, episode_id=episode_id)
        destination = out_dir / relative.with_suffix(".json")
        save_episode(episode, destination)
        written.append(destination)
    return written
