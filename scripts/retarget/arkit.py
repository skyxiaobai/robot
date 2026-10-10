# -*- coding: utf-8 -*-
"""ARKit 世界系（Y 朝上）到桌面机械臂（Z 朝上）的一把共用标定。

位置轴的对应是固定的：人手 X → 机器人左右，人手 Z → 机器人前后，人手 Y → 机器人高度。
每条录像的 ARKit 原点都不一样，所以先减去这条录像手腕的中位数，再拟合缩放。
缩放全数据只有一把，不会按每一条演示把方块拉到同一个点上。
"""
import numpy as np

from retarget.calibrate import fit_from_boxes
from retarget.frames import TASK_HIGH, TASK_LOW, Z_GRASP, Z_HOVER

# 人手 (x, y, z) 重排成机器人 (前后, 左右, 上) = (人手 z, 人手 x, 人手 y)。
ARKIT_TO_ROBOT_AXES = (2, 0, 1)

# 映射进夹爪真正够得到的桌面盒子，而不是手臂连杆的整个活动范围。
ROBOT_BOX_MIN = np.array([float(TASK_LOW[0]), float(TASK_LOW[1]), float(Z_GRASP)], dtype=float)
ROBOT_BOX_MAX = np.array([float(TASK_HIGH[0]), float(TASK_HIGH[1]), float(Z_HOVER)], dtype=float)


def permute_arkit_xyz(human_xyz):
    """(T, 3) 或 (3,) 的 ARKit 坐标 → 机器人轴顺序，尚未缩放。"""
    points = np.asarray(human_xyz, dtype=float)
    single = points.ndim == 1
    if single:
        points = points.reshape(1, 3)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("坐标必须是 (T, 3)")
    ordered = points[:, list(ARKIT_TO_ROBOT_AXES)]
    if single:
        return ordered[0]
    return ordered


def height_gain(human_xyz, grasp_index, release_index):
    """抓住之后到松开之前，各轴相对抓住帧升高了多少。Y 朝上时第 1 轴应为正。"""
    points = np.asarray(human_xyz, dtype=float)
    grasp = int(grasp_index)
    release = int(release_index)
    window = points[grasp:release + 1]
    return window.max(axis=0) - points[grasp]


def fit_shared_calibration(permuted_points, grasp_heights, carry_heights):
    """水平轴用 5%–95% 分位贴到桌面盒子；高度用抓住高度的中位数贴到抓取高度，搬运高点贴到悬停。

    ``permuted_points`` 已经是机器人轴顺序。(N, 3)。
    ``grasp_heights`` / ``carry_heights`` 是同一顺序下的第 3 维。
    """
    points = np.asarray(permuted_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 2:
        raise ValueError("标定至少需要两个手腕点")
    grasp_heights = np.asarray(grasp_heights, dtype=float).reshape(-1)
    carry_heights = np.asarray(carry_heights, dtype=float).reshape(-1)
    if grasp_heights.shape[0] < 1 or carry_heights.shape[0] < 1:
        raise ValueError("标定需要抓住高度和搬运高度")
    horizontal_lo = np.percentile(points[:, :2], 5, axis=0)
    horizontal_hi = np.percentile(points[:, :2], 95, axis=0)
    if np.any(horizontal_hi - horizontal_lo <= 1e-4):
        raise ValueError("手腕在水平方向上几乎没有展开，无法标定")
    z_lo = float(np.median(grasp_heights))
    z_hi = float(np.median(carry_heights))
    used_grasp_median = True
    if z_hi - z_lo <= 0.02:
        z_lo = float(np.percentile(points[:, 2], 10))
        z_hi = float(np.percentile(points[:, 2], 90))
        used_grasp_median = False
    if z_hi - z_lo <= 1e-4:
        raise ValueError("手腕高度没有展开，无法标定")
    human_min = np.array([horizontal_lo[0], horizontal_lo[1], z_lo], dtype=float)
    human_max = np.array([horizontal_hi[0], horizontal_hi[1], z_hi], dtype=float)
    calibration = fit_from_boxes(human_min, human_max, ROBOT_BOX_MIN, ROBOT_BOX_MAX)
    return calibration, {
        "human_min": human_min,
        "human_max": human_max,
        "z_from_grasp_median": used_grasp_median,
    }


def episode_center(permuted_points):
    """这一条录像自己的中位数。ARKit 的原点每条录像不同，不能把绝对坐标混在一起拟合。"""
    points = np.asarray(permuted_points, dtype=float)
    return np.median(points, axis=0)


def map_centered(calibration, human_xyz, center):
    """先换成机器人轴顺序，减去这条录像的中心，再做共用的缩放和平移，最后夹进桌面盒子。"""
    ordered = permute_arkit_xyz(human_xyz)
    centered = ordered - np.asarray(center, dtype=float)
    mapped = np.asarray(calibration.map_points(centered), dtype=float)
    return np.clip(mapped, ROBOT_BOX_MIN, ROBOT_BOX_MAX)
