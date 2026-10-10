# -*- coding: utf-8 -*-
"""从 HOT3D-Clips 的手腕和物体位姿里切出「拿起再放下」。

这些阈值在看仿真成功率之前就定好，不按结果再改：

- 物体在 Y 轴升高至少 6 cm，再落回到起点高度 4 cm 以内（Y 朝上，和已核对过的 EgoDex 一样）。
- 两个低点的水平距离（XZ）至少 8 cm。
- 抬起期间更近的那只手，到物体包围盒中心的中位距离不超过 25 cm。
- 整段 0.5–8 秒。每只物体只留水平位移最大的一段。

没有 MANO，不用皮肤接触冒充网格真值。手腕来自 UmeTrack 的 ``T_world_from_wrist``。
"""
import json
from pathlib import Path

import numpy as np

from retarget.frames import matrix_from_quat_xyzw

LIFT_M = 0.06
PLACE_TOL_M = 0.04
TRAVEL_XZ_M = 0.08
HAND_NEAR_M = 0.25
MIN_SEGMENT_S = 0.5
MAX_SEGMENT_S = 8.0
FPS = 30.0
MARGIN_S = 0.5


def _quat_xyzw(wxyz):
    quat = np.asarray(wxyz, dtype=float).reshape(4)
    return np.array([quat[1], quat[2], quat[3], quat[0]], dtype=float)


def transform_local(translation, quat_wxyz, local):
    """物体坐标系里的点 → 世界系。(T, 3)。"""
    translation = np.asarray(translation, dtype=float)
    local = np.asarray(local, dtype=float).reshape(3)
    out = np.empty_like(translation)
    for index in range(translation.shape[0]):
        rotation = matrix_from_quat_xyzw(_quat_xyzw(quat_wxyz[index]))
        out[index] = rotation @ local + translation[index]
    return out


def find_pick_place(centers, left_xyz, right_xyz, fps=FPS):
    """返回一段，或 None。``centers`` / 手腕都是 (T, 3)，Y 朝上。"""
    centers = np.asarray(centers, dtype=float)
    left_xyz = np.asarray(left_xyz, dtype=float)
    right_xyz = np.asarray(right_xyz, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("物体中心必须是 (T, 3)")
    valid = np.isfinite(centers).all(axis=1)
    height = centers[:, 1]
    best = None
    for peak in _peak_indices(height, valid):
        before = np.flatnonzero(valid & (np.arange(height.shape[0]) <= peak))
        after = np.flatnonzero(valid & (np.arange(height.shape[0]) >= peak))
        if before.size < 2 or after.size < 2:
            continue
        start = int(before[np.argmin(height[before])])
        end = int(after[np.argmin(height[after])])
        if not (start < peak < end):
            continue
        lift = float(height[peak] - height[start])
        if lift < LIFT_M:
            continue
        if float(height[end]) > float(height[start]) + PLACE_TOL_M:
            continue
        travel = float(np.linalg.norm(centers[end, [0, 2]] - centers[start, [0, 2]]))
        if travel < TRAVEL_XZ_M:
            continue
        duration = (end - start) / float(fps)
        if duration < MIN_SEGMENT_S or duration > MAX_SEGMENT_S:
            continue
        side, distance = _near_hand(centers, left_xyz, right_xyz, start, end)
        if side is None or distance > HAND_NEAR_M:
            continue
        candidate = {
            "start": start,
            "end": end,
            "peak": peak,
            "side": side,
            "hand_distance_m": float(distance),
            "travel_xz_m": travel,
            "lift_m": lift,
            "duration_s": float(duration),
        }
        if best is None or candidate["travel_xz_m"] > best["travel_xz_m"]:
            best = candidate
    return best


def _peak_indices(height, valid):
    peaks = []
    count = height.shape[0]
    for index in range(1, count - 1):
        if not valid[index]:
            continue
        if height[index] >= height[index - 1] and height[index] >= height[index + 1]:
            peaks.append(index)
    if not peaks and np.any(valid):
        peaks.append(int(np.nanargmax(np.where(valid, height, -np.inf))))
    return peaks


def _near_hand(centers, left_xyz, right_xyz, start, end):
    distances = []
    for side, wrist in (("left", left_xyz), ("right", right_xyz)):
        delta = wrist[start:end + 1] - centers[start:end + 1]
        dist = np.linalg.norm(delta, axis=1)
        dist = dist[np.isfinite(dist)]
        if dist.size == 0:
            continue
        distances.append((side, float(np.median(dist))))
    if not distances:
        return None, None
    return min(distances, key=lambda item: item[1])


def _pose_translation(pose):
    if not isinstance(pose, dict):
        return None, None
    translation = pose.get("translation_xyz")
    quat = pose.get("quaternion_wxyz")
    if translation is None or quat is None:
        return None, None
    xyz = np.asarray(translation, dtype=float).reshape(3)
    wxyz = np.asarray(quat, dtype=float).reshape(4)
    if not np.isfinite(xyz).all() or not np.isfinite(wxyz).all():
        return None, None
    return xyz, wxyz


def load_clip(directory):
    """读一个已解开的 clip 目录（只要 json，不要图像）。"""
    directory = Path(directory)
    hand_files = sorted(directory.glob("*.hands.json"))
    if not hand_files:
        raise FileNotFoundError("没有 *.hands.json：%s" % directory)
    count = len(hand_files)
    left = np.full((count, 3), np.nan)
    right = np.full((count, 3), np.nan)
    left_q = np.full((count, 4), np.nan)
    right_q = np.full((count, 4), np.nan)
    objects = {}
    sequence_id = None
    device = None
    for index, hand_path in enumerate(hand_files):
        stem = hand_path.name[: -len(".hands.json")]
        hands = json.loads(hand_path.read_text(encoding="utf-8"))
        for side, store, store_q in (("left", left, left_q), ("right", right, right_q)):
            pose = ((hands.get(side) or {}).get("umetrack_pose") or {}).get("T_world_from_wrist")
            xyz, quat = _pose_translation(pose)
            if xyz is None:
                continue
            store[index] = xyz
            store_q[index] = quat
        obj_path = hand_path.with_name(stem + ".objects.json")
        payload = json.loads(obj_path.read_text(encoding="utf-8"))
        for key, instances in payload.items():
            if not instances:
                continue
            inst = instances[0]
            xyz, quat = _pose_translation(inst.get("T_world_from_object"))
            if xyz is None:
                continue
            slot = objects.get(key)
            if slot is None:
                slot = {
                    "bop_id": str(inst.get("object_bop_id") or key),
                    "name": inst.get("object_name") or str(key),
                    "uid": str(inst.get("object_uid") or ""),
                    "origin": np.full((count, 3), np.nan),
                    "quat_wxyz": np.full((count, 4), np.nan),
                }
                objects[key] = slot
            slot["origin"][index] = xyz
            slot["quat_wxyz"][index] = quat
        if sequence_id is None:
            info_path = hand_path.with_name(stem + ".info.json")
            if info_path.exists():
                info = json.loads(info_path.read_text(encoding="utf-8"))
                sequence_id = info.get("sequence_id")
                device = info.get("device")
    return {
        "directory": str(directory),
        "clip": directory.name,
        "num_frames": count,
        "fps": FPS,
        "sequence_id": sequence_id,
        "device": device,
        "left_xyz": left,
        "right_xyz": right,
        "left_wxyz": left_q,
        "right_wxyz": right_q,
        "objects": objects,
    }


def bbox_centers(origin, quat_wxyz, center_local):
    """模型原点的轨迹 → 包围盒中心的轨迹。缺帧保持 NaN。"""
    origin = np.asarray(origin, dtype=float)
    quat_wxyz = np.asarray(quat_wxyz, dtype=float)
    center = np.full_like(origin, np.nan)
    finite = np.isfinite(origin).all(axis=1) & np.isfinite(quat_wxyz).all(axis=1)
    if not np.any(finite):
        return center
    center[finite] = transform_local(origin[finite], quat_wxyz[finite], center_local)
    return center


def window_slice(count, start, end, fps=FPS, margin_s=MARGIN_S):
    margin = int(round(float(margin_s) * float(fps)))
    lo = max(0, int(start) - margin)
    hi = min(int(count), int(end) + margin + 1)
    return lo, hi


def iter_clip_dirs(root):
    root = Path(root)
    dirs = [path for path in sorted(root.iterdir()) if path.is_dir() and list(path.glob("*.hands.json"))]
    return dirs
