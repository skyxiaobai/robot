# -*- coding: utf-8 -*-
"""把一条统一 episode 的手腕和开合变成机械臂关节轨迹。"""
from dataclasses import dataclass

import numpy as np

from retarget.aperture import apertures_from_joints, grip_command
from retarget.calibrate import default_calibration
from retarget.frames import R_DOWN, site_rotation


@dataclass
class RetargetResult:
    times: np.ndarray
    q: np.ndarray
    grip: np.ndarray
    ee_target: np.ndarray
    pos_err: np.ndarray
    rot_err: np.ndarray
    n_colliding: int
    n_lifted: int
    side: str

    @property
    def median_pos_err(self):
        return float(np.median(self.pos_err))

    @property
    def max_pos_err(self):
        return float(np.max(self.pos_err))


def _as_array(frames):
    array = np.asarray(frames, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("手腕或关节里有缺失值，重定向前需要补齐")
    return array


def retarget_episode(episode, env, calibration=None, side="right"):
    """右手（默认）手腕 6DoF + 指尖开合 → 关节轨迹。

    位置用标定做缩放和平移。朝向用固定工具旋转。夹爪开合来自拇指尖到食指尖的距离。
    每一帧做阻尼最小二乘 IK，关节角夹在限位里；连杆撞桌子时把目标抬高再解一次。
    """
    if calibration is None:
        calibration = default_calibration()
    hand = episode["hands"][side]
    wrist = _as_array(hand["wrist_pose"])
    joints = _as_array(hand["joints"])
    if wrist.ndim != 2 or wrist.shape[1] != 7:
        raise ValueError("wrist_pose 必须是 (T, 7)")
    human_xyz = wrist[:, :3]
    robot_xyz = np.asarray(calibration.map_points(human_xyz), dtype=float)
    aperture = apertures_from_joints(joints)
    grip = np.asarray(grip_command(aperture), dtype=float)
    times = np.asarray(episode["timestamps"], dtype=float)
    q = env.home_q.copy()
    qs = []
    pos_err = []
    rot_err = []
    n_colliding = 0
    n_lifted = 0
    for index in range(robot_xyz.shape[0]):
        rotation = site_rotation(wrist[index, 3:])
        # 合成掌心朝下时 rotation 就是 R_DOWN。数值漂移时仍用解出的矩阵。
        if not np.isfinite(rotation).all():
            rotation = R_DOWN
        solved = env.ik(q, robot_xyz[index], rotation)
        q = solved.q
        qs.append(q.copy())
        pos_err.append(solved.pos_err)
        rot_err.append(solved.rot_err)
        n_colliding += int(solved.collided)
        n_lifted += int(solved.lifted)
    return RetargetResult(
        times=times,
        q=np.vstack(qs),
        grip=grip,
        ee_target=robot_xyz,
        pos_err=np.asarray(pos_err, dtype=float),
        rot_err=np.asarray(rot_err, dtype=float),
        n_colliding=n_colliding,
        n_lifted=n_lifted,
        side=side,
    )


def task_xy(episode):
    """优先用合成片段写明的桌面坐标；否则用夹爪闭合和松开时的末端 xy。"""
    task = episode.get("retarget_task")
    if isinstance(task, dict) and "cube_xy" in task and "goal_xy" in task:
        return (
            np.asarray(task["cube_xy"], dtype=float),
            np.asarray(task["goal_xy"], dtype=float),
        )
    return None


def infer_scene_xy(ee_target, grip, close_below=0.008):
    """没有机器人坐标标注时：第一次夹紧的 xy 放方块，之后第一次张开的 xy 当目标。"""
    grip = np.asarray(grip, dtype=float)
    closed = grip < float(close_below)
    if not np.any(closed):
        return None
    grasp = int(np.flatnonzero(closed)[0])
    opened = np.flatnonzero(~closed[grasp + 1:])
    if opened.size == 0:
        return None
    release = grasp + 1 + int(opened[0])
    ee = np.asarray(ee_target, dtype=float)
    return ee[grasp, :2].copy(), ee[release, :2].copy()
