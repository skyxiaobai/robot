# -*- coding: utf-8 -*-
"""夹爪开合改由抓取状态决定，而不是只看开合距离。

两路都写清楚来源，不把近似值叫成网格真值：

- 位姿耦合（HOT3D，没有 MANO）：被选中的那只手是更近的一只，距离不超过
  ``HAND_NEAR_M``，物体在动（≥ 5 cm/s），而且手腕速度和物体速度同向（余弦 ≥ 0.5）。
  夹爪在第一帧耦合前 3 帧（0.1 秒）合上，直到最后一帧耦合。这不是皮肤接触。
- 估计器：调用 ``egodata.interaction.estimate_interaction``（PR #27 的启发式）。
  HOT3D 没有 MANO 的 21 点，指尖用手腕坐标系里的固定偏移，表面用包围盒 8 个角。
  来源仍是 ``heuristic``，另外注明关节和表面是近似的。EgoDex 有真的 21 点，
  物体是抓住那一帧指尖中心上的一个方块，不是数据集里的物体位姿。
"""
import numpy as np

from egodata.interaction import estimate_interaction
from retarget.events import normalized_grip
from retarget.frames import GRIP_CLOSE, GRIP_OPEN, matrix_from_quat_xyzw
from retarget.hot3d_clip import HAND_NEAR_M, _quat_xyzw

COUPLE_SPEED_M_S = 0.05
COUPLE_COS = 0.5
LEAD_FRAMES = 3

# 固定模板，不是 MANO。单位米，手腕在原点，手指大致朝 +X，拇指在 +Y。
# 拇指尖到食指尖约 3.3 cm，低于启发式里 8 cm 的张开上限，所以张合这一关是常开的。
_TEMPLATE = np.array(
    [
        [0.000, 0.000, 0.000],
        [0.020, 0.015, 0.010],
        [0.035, 0.025, 0.015],
        [0.048, 0.032, 0.018],
        [0.058, 0.038, 0.015],
        [0.040, 0.018, 0.000],
        [0.055, 0.018, 0.000],
        [0.068, 0.018, 0.000],
        [0.080, 0.018, 0.000],
        [0.040, 0.000, 0.000],
        [0.058, 0.000, 0.000],
        [0.072, 0.000, 0.000],
        [0.085, 0.000, 0.000],
        [0.038, -0.016, 0.000],
        [0.054, -0.016, 0.000],
        [0.066, -0.016, 0.000],
        [0.078, -0.016, 0.000],
        [0.034, -0.032, 0.000],
        [0.048, -0.032, 0.000],
        [0.058, -0.032, 0.000],
        [0.068, -0.032, 0.000],
    ],
    dtype=float,
)


def canonical_joints(side="right"):
    joints = _TEMPLATE.copy()
    if side == "left":
        joints[:, 1] *= -1.0
    return joints


def approximate_joints(wrist_xyz, quat_wxyz, side):
    """(T, 21, 3)。缺手腕的帧是 NaN。关节是模板，不随 UmeTrack 的 22 个关节角弯曲。"""
    wrist_xyz = np.asarray(wrist_xyz, dtype=float)
    quat_wxyz = np.asarray(quat_wxyz, dtype=float)
    local = canonical_joints(side)
    out = np.full((wrist_xyz.shape[0], 21, 3), np.nan, dtype=float)
    for index in range(wrist_xyz.shape[0]):
        if not np.isfinite(wrist_xyz[index]).all() or not np.isfinite(quat_wxyz[index]).all():
            continue
        rotation = matrix_from_quat_xyzw(_quat_xyzw(quat_wxyz[index]))
        out[index] = (rotation @ local.T).T + wrist_xyz[index]
    return out


def coupling_mask(wrist_xyz, object_xyz, other_xyz=None, fps=30.0, near_m=HAND_NEAR_M):
    """逐帧：这只手带着物体在动。静止或离得远都是 False。"""
    wrist = np.asarray(wrist_xyz, dtype=float)
    obj = np.asarray(object_xyz, dtype=float)
    dist = np.linalg.norm(wrist - obj, axis=1)
    if other_xyz is None:
        nearest = np.ones(wrist.shape[0], dtype=bool)
    else:
        other = np.linalg.norm(np.asarray(other_xyz, dtype=float) - obj, axis=1)
        nearest = dist <= other + 1e-9
    dt = 1.0 / float(fps)
    obj_v = np.gradient(obj, dt, axis=0)
    wrist_v = np.gradient(wrist, dt, axis=0)
    obj_speed = np.linalg.norm(obj_v, axis=1)
    wrist_speed = np.linalg.norm(wrist_v, axis=1)
    cosine = np.sum(obj_v * wrist_v, axis=1) / np.maximum(obj_speed * wrist_speed, 1e-8)
    finite = np.isfinite(dist) & np.isfinite(obj_speed) & np.isfinite(cosine)
    return finite & nearest & (dist <= float(near_m)) & (obj_speed >= COUPLE_SPEED_M_S) & (cosine >= COUPLE_COS)


def grip_from_closed(closed, lead_frames=LEAD_FRAMES):
    """闭合区间略提前，好让指垫在物体离桌之前合上。提前量是固定的 3 帧。"""
    flags = np.asarray(closed, dtype=bool).copy()
    hits = np.flatnonzero(flags)
    if hits.size and int(lead_frames) > 0:
        start = max(0, int(hits[0]) - int(lead_frames))
        flags[start:int(hits[0])] = True
    grip = np.full(flags.shape[0], GRIP_OPEN, dtype=float)
    grip[flags] = GRIP_CLOSE
    return grip


def grip_from_states(states, lead_frames=LEAD_FRAMES):
    closed = np.array([state == "grasp" for state in states], dtype=bool)
    return grip_from_closed(closed, lead_frames=lead_frames)


def grip_from_distance(wrist_xyz, object_xyz):
    """没有真实开合时的刚性基线：手腕到物体中心的距离，远则张开。

    这不是抓取状态，只是一条跟开合距离同一形状的连续指令。
    """
    dist = np.linalg.norm(np.asarray(wrist_xyz, dtype=float) - np.asarray(object_xyz, dtype=float), axis=1)
    if not np.isfinite(dist).all():
        dist = np.where(np.isfinite(dist), dist, np.nanmax(dist) if np.isfinite(dist).any() else 1.0)
    grip = normalized_grip(dist)
    if grip is None:
        return np.full(dist.shape[0], GRIP_OPEN, dtype=float)
    return np.asarray(grip, dtype=float)


def _bbox_corners(minimum, size):
    minimum = np.asarray(minimum, dtype=float).reshape(3)
    size = np.asarray(size, dtype=float).reshape(3)
    corners = []
    for ix in (0.0, 1.0):
        for iy in (0.0, 1.0):
            for iz in (0.0, 1.0):
                corners.append(minimum + np.array([ix, iy, iz]) * size)
    return np.asarray(corners, dtype=float)


def _pose7(origin, quat_wxyz):
    """xyz + xyzw。缺帧返回 None。"""
    if not np.isfinite(origin).all() or not np.isfinite(quat_wxyz).all():
        return None
    xyzw = _quat_xyzw(quat_wxyz)
    return [float(origin[0]), float(origin[1]), float(origin[2]), float(xyzw[0]), float(xyzw[1]), float(xyzw[2]), float(xyzw[3])]


def estimator_grip_hot3d(joints_by_side, objects, timestamps, bbox_by_id):
    """启发式抓取。``bbox_by_id`` 是 bop id → (min_xyz, size_xyz)，角点在模型坐标系。

    返回 ``{"grip": {side: array}, "source": ..., "states": {side: list}}``。
    """
    count = len(timestamps)
    obj_rows = []
    surfaces = {}
    for obj in objects:
        poses = []
        valid = []
        for index in range(count):
            pose = _pose7(obj["origin"][index], obj["quat_wxyz"][index])
            poses.append(pose if pose is not None else [0, 0, 0, 0, 0, 0, 1])
            valid.append(pose is not None)
        obj_rows.append({"id": obj["bop_id"], "pose": poses, "valid": valid})
        minimum, size = bbox_by_id[obj["bop_id"]]
        surfaces[obj["bop_id"]] = {"vertices": _bbox_corners(minimum, size)}
    hands = {}
    for side in ("left", "right"):
        series = joints_by_side.get(side)
        frames = []
        if series is None:
            frames = [None] * count
        else:
            for index in range(count):
                frame = np.asarray(series[index], dtype=float)
                frames.append(None if not np.isfinite(frame).all() else frame)
        hands[side] = {"joints": frames}
    estimated = estimate_interaction(hands, obj_rows, list(timestamps), surfaces)
    grip = {}
    states = {}
    for side in ("left", "right"):
        states[side] = list(estimated["grasp"][side]["state"])
        grip[side] = grip_from_states(states[side])
    return {
        "grip": grip,
        "states": states,
        "source": estimated["grasp"]["source"],
        "joint_source": "fixed_wrist_frame_offsets",
        "surface_source": "bbox_corners",
        "contact_source": estimated["contact"]["source"],
    }


def estimator_grip_egodex(joints, side, grasp_index, timestamps, half=0.018):
    """EgoDex 没有物体位姿。方块放在抓住那一帧的指尖中心，边长和仿真方块一样。

    这是估计器输入，不是真值。另一只手没有关节。
    """
    joints = np.asarray(joints, dtype=float)
    count = joints.shape[0]
    tips = joints[int(grasp_index)][np.array([4, 8, 12, 16, 20])]
    center = tips.mean(axis=0)
    if not np.isfinite(center).all():
        return {
            "grip": np.full(count, GRIP_OPEN),
            "states": [None] * count,
            "source": "heuristic",
            "note": "抓住帧的指尖不是有限值，夹爪保持张开",
        }
    poses = [[float(center[0]), float(center[1]), float(center[2]), 0.0, 0.0, 0.0, 1.0]] * count
    objects = [{"id": "inferred_cube", "pose": poses, "valid": [True] * count}]
    h = float(half)
    corners = np.array([[sx * h, sy * h, sz * h] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)])
    surfaces = {"inferred_cube": {"vertices": corners}}
    series = []
    for index in range(count):
        frame = joints[index]
        series.append(None if not np.isfinite(frame).all() else frame)
    empty = [None] * count
    hands = {
        "left": {"joints": series if side == "left" else empty},
        "right": {"joints": series if side == "right" else empty},
    }
    estimated = estimate_interaction(hands, objects, list(timestamps), surfaces)
    states = list(estimated["grasp"][side]["state"])
    return {
        "grip": grip_from_states(states),
        "states": states,
        "source": estimated["grasp"]["source"],
        "joint_source": "egodex_21",
        "surface_source": "inferred_cube_at_fingertips",
        "object_center_xyz": [float(v) for v in center],
    }
