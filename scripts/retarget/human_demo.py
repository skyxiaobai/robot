# -*- coding: utf-8 -*-
"""合成一条 EgoDex 式的 basic_pick_place。

仓库里没有 16GB 的 EgoDex test.zip（许可也不允许提交）。这里按统一 episode
的字段写一条右手抓放：世界系手腕 7 个数、21 个关节、开合随阶段变化。
坐标在人手盒子里，要经过 ``AxisCalibration`` 才落到机械臂工作空间。
"""
import numpy as np

from egodata.schema import SCHEMA_VERSION, empty_interaction, make_quaternions_continuous, rotmat_to_quat_xyzw
from retarget.frames import GRIP_OPEN, HUMAN_FPS, identity_wrist_quat
from retarget.paths import sample_path


def _finger_chain(tip, count):
    """从掌根到指尖均匀排 ``count`` 个点，不含手腕。"""
    points = []
    for index in range(1, count + 1):
        points.append(tip * (index / float(count)))
    return points


def hand_joints(wrist_xyz, aperture_m, rotation):
    """构造 21 个点，使拇指尖到食指尖的距离等于 ``aperture_m``。"""
    half = float(aperture_m) * 0.5
    thumb_tip = np.array([half, 0.02, 0.08], dtype=float)
    index_tip = np.array([-half, 0.02, 0.08], dtype=float)
    middle_tip = np.array([0.0, 0.0, 0.10], dtype=float)
    ring_tip = np.array([-0.015, -0.012, 0.09], dtype=float)
    pinky_tip = np.array([-0.03, -0.02, 0.08], dtype=float)
    local = np.zeros((21, 3), dtype=float)
    local[1:5] = _finger_chain(thumb_tip, 4)
    local[5:9] = _finger_chain(index_tip, 4)
    local[9:13] = _finger_chain(middle_tip, 4)
    local[13:17] = _finger_chain(ring_tip, 4)
    local[17:21] = _finger_chain(pinky_tip, 4)
    # 指尖用精确坐标，避免 linspace 的端点误差。
    local[4] = thumb_tip
    local[8] = index_tip
    world = (np.asarray(rotation, dtype=float) @ local.T).T + np.asarray(wrist_xyz, dtype=float)
    return world


def _static_hand(num_frames, wrist_xyz):
    rotation = np.eye(3)
    joints = hand_joints(wrist_xyz, 0.09, rotation)
    quat = identity_wrist_quat()
    pose = np.concatenate([np.asarray(wrist_xyz, dtype=float), quat])
    return {
        "joints": [joints.tolist() for _ in range(num_frames)],
        "wrist_pose": [pose.tolist() for _ in range(num_frames)],
        "confidence": [0.99 for _ in range(num_frames)],
        "valid": [True for _ in range(num_frames)],
    }


def make_basic_pick_place(cube_xy, goal_xy, calibration, arc_m=0.03, fps=HUMAN_FPS, episode_id=None):
    """生成一条右手 basic_pick_place。``cube_xy`` / ``goal_xy`` 在机器人桌面坐标。"""
    times, robot_ee, grip, _duration = sample_path(cube_xy, goal_xy, dt=1.0 / float(fps), arc_m=arc_m)
    human_ee = calibration.unmap_points(robot_ee)
    num_frames = int(human_ee.shape[0])
    # 张开 9 cm，夹紧 2 cm。grip 从 GRIP_OPEN 到 0，线性对应这两档，便于阈值切开。
    open_gap = 0.09
    close_gap = 0.02
    aperture = close_gap + (grip / GRIP_OPEN) * (open_gap - close_gap)
    rotation = np.eye(3)
    quat = np.array(rotmat_to_quat_xyzw(rotation), dtype=float)
    quats = np.repeat(quat.reshape(1, 4), num_frames, axis=0)
    quats = make_quaternions_continuous(quats)
    joints = []
    wrist_pose = []
    for index in range(num_frames):
        frame_joints = hand_joints(human_ee[index], aperture[index], rotation)
        joints.append(frame_joints.tolist())
        wrist_pose.append(np.concatenate([human_ee[index], quats[index]]).tolist())
    if episode_id is None:
        episode_id = "synthetic/basic_pick_place/0"
    return {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "source": "synthetic_basic_pick_place",
        "source_path": "",
        "fps": float(fps),
        "coordinate_frame": "arkit_world",
        "image_width": 1920,
        "image_height": 1080,
        "num_frames": num_frames,
        "timestamps": [float(value) for value in times],
        "camera_intrinsic": [[736.6339, 0.0, 960.0], [0.0, 736.6339, 540.0], [0.0, 0.0, 1.0]],
        "camera_poses": [np.eye(4, dtype=float).tolist() for _ in range(num_frames)],
        "hands": {
            "right": {
                "joints": joints,
                "wrist_pose": wrist_pose,
                "confidence": [0.99 for _ in range(num_frames)],
                "valid": [True for _ in range(num_frames)],
            },
            "left": _static_hand(num_frames, np.array([-0.25, 0.2, 0.9])),
        },
        # 合成片段同样没有物体 6DoF。空通道表示「没测到」，不能写成全 0。
        **empty_interaction(num_frames),
        "annotation": {
            "environment": {"name": "table", "detail": "table:wood", "source": "synthetic"},
            "task": {
                "name": "basic_pick_place",
                "instruction": "Pick up the block and place it on the marked spot.",
            },
            "subtasks": [],
            "instructions": [],
        },
        "coverage": {
            "environment": "table",
            "objects": ["block"],
            "object_classes": ["block"],
            "actions": ["pick", "place"],
        },
        "retarget_task": {
            "cube_xy": [float(cube_xy[0]), float(cube_xy[1])],
            "goal_xy": [float(goal_xy[0]), float(goal_xy[1])],
            "side": "right",
            "arc_m": float(arc_m),
            "note": "合成任务在机器人桌面坐标里的对应摆放。真实 EgoDex 没有这个字段时，用闭合/松开的手腕位置推断。",
        },
    }
