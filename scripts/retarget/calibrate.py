# -*- coding: utf-8 -*-
"""把人手腕的 xyz 按轴缩放并平移到机械臂工作空间。

每一轴单独做 ``robot = scale * human + offset``。缩放和偏移用两组对应点
（例如标定动作的角点）做最小二乘，或者直接用人手盒子和机器人盒子的边界。
"""
from dataclasses import dataclass

import numpy as np

from retarget.frames import HUMAN_MAX, HUMAN_MIN, ROBOT_MAX, ROBOT_MIN


@dataclass(frozen=True)
class AxisCalibration:
    """三轴各自的 scale 和 offset。``robot = scale * human + offset``。"""

    scale: np.ndarray
    offset: np.ndarray

    def map_points(self, human_xyz):
        points = np.asarray(human_xyz, dtype=float)
        return points * self.scale + self.offset

    def unmap_points(self, robot_xyz):
        points = np.asarray(robot_xyz, dtype=float)
        return (points - self.offset) / self.scale


def fit_from_correspondences(human_xyz, robot_xyz):
    """每轴最小二乘：已知若干人手点和对应的机器人点。"""
    human = np.asarray(human_xyz, dtype=float)
    robot = np.asarray(robot_xyz, dtype=float)
    if human.ndim != 2 or human.shape[1] != 3 or human.shape != robot.shape:
        raise ValueError("对应点必须是同样形状的 (N, 3)")
    if human.shape[0] < 2:
        raise ValueError("至少需要 2 个对应点")
    return _fit_independent(human, robot)


def _fit_independent(human, robot):
    scale = np.zeros(3, dtype=float)
    offset = np.zeros(3, dtype=float)
    for axis in range(3):
        design = np.column_stack([human[:, axis], np.ones(human.shape[0])])
        coeff, _, _, _ = np.linalg.lstsq(design, robot[:, axis], rcond=None)
        scale[axis] = float(coeff[0])
        offset[axis] = float(coeff[1])
    return AxisCalibration(scale=scale, offset=offset)


def fit_from_boxes(human_min, human_max, robot_min, robot_max):
    """用两只轴对齐盒子的边界确定缩放和平移。"""
    human_min = np.asarray(human_min, dtype=float)
    human_max = np.asarray(human_max, dtype=float)
    robot_min = np.asarray(robot_min, dtype=float)
    robot_max = np.asarray(robot_max, dtype=float)
    span_h = human_max - human_min
    if np.any(span_h <= 0):
        raise ValueError("人手盒子每一轴都要有正的长度")
    scale = (robot_max - robot_min) / span_h
    offset = robot_min - scale * human_min
    return AxisCalibration(scale=scale, offset=offset)


def default_calibration():
    """合成 basic_pick_place 使用的固定标定。真实片段也可以先用它，再按数据重拟合。"""
    return fit_from_boxes(HUMAN_MIN, HUMAN_MAX, ROBOT_MIN, ROBOT_MAX)
