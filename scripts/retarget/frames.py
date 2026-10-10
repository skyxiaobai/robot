# -*- coding: utf-8 -*-
"""人手工作空间和机械臂工作空间的常数，以及朝向换算。

机器人是桌面前方的 6 轴臂。人手坐标故意用另一套范围（更大、更高），
这样重定向必须做缩放和平移，不能把手腕位置原样塞进仿真。
"""
import numpy as np

from egodata.schema import rotmat_to_quat_xyzw

# 桌面顶面 z、方块半边长。抓取高度是调过接触之后能稳定夹起的指尖高度。
TABLE_TOP = 0.22
CUBE_HALF = 0.018
Z_REST = TABLE_TOP + CUBE_HALF
Z_GRASP = 0.242
Z_HOVER = 0.36

# 夹爪滑动关节：0 夹紧，GRIP_OPEN 张开。
GRIP_OPEN = 0.024
GRIP_CLOSE = 0.0

# 成功：方块中心的水平距离，以及还在桌面上的高度带。
SUCCESS_XY_M = 0.04
SUCCESS_Z = (0.20, 0.30)

# 末端在机器人基座坐标系里允许活动的盒子（米）。
ROBOT_MIN = np.array([0.40, -0.16, 0.24], dtype=float)
ROBOT_MAX = np.array([0.60, 0.16, 0.40], dtype=float)

# 合成 EgoDex 式手腕使用的人手工作空间（米）。和机器人盒子不是同一套数。
HUMAN_MIN = np.array([-0.20, -0.30, 0.75], dtype=float)
HUMAN_MAX = np.array([0.40, 0.30, 1.10], dtype=float)

# 方块起点和目标点的采样范围（只含 xy）。
TASK_LOW = np.array([0.42, -0.10], dtype=float)
TASK_HIGH = np.array([0.56, 0.12], dtype=float)
TASK_MIN_SEPARATION = 0.12

# 复位时末端停在桌面中上方。
HOME_EE = np.array([0.48, 0.0, Z_HOVER], dtype=float)

# 指尖朝下：site 的三列是世界系坐标轴。z 列指向世界 -Z。
R_DOWN = np.array(
    [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
    dtype=float,
)
# 人手腕旋转为单位阵时，工具坐标把「掌心朝下」变成上面的 site 朝向。
R_TOOL = R_DOWN.copy()

ARM_JOINTS = ("j1", "j2", "j3", "j4", "j5", "j6")
GRIP_JOINTS = ("grip_l", "grip_r")
# 这些连杆碰到桌子算碰撞。指垫贴着桌面是抓取，不算。
ARM_LINK_GEOMS = ("g1", "g2", "g3", "g4", "g5", "g6")
PAD_GEOMS = ("pad_l", "pad_r", "palm")

CONTROL_DT = 0.1
HUMAN_FPS = 30.0


def matrix_from_quat_xyzw(quat):
    """四元数 (x, y, z, w) → 3×3 旋转矩阵。"""
    x, y, z, w = [float(v) for v in quat]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def site_rotation(wrist_quat_xyzw, tool=None):
    """人手腕朝向 → 夹爪 site 的旋转矩阵。

    ``R_site = R_wrist @ R_tool``。工具旋转是固定的：人手「掌心朝下」
    （腕部旋转接近单位阵）对应夹爪从上方接近。
    """
    if tool is None:
        tool = R_TOOL
    return matrix_from_quat_xyzw(wrist_quat_xyzw) @ np.asarray(tool, dtype=float)


def identity_wrist_quat():
    """掌心朝下、无偏航时的手腕四元数。"""
    return np.array(rotmat_to_quat_xyzw(np.eye(3)), dtype=float)


def goal_xyz(goal_xy):
    """策略看到的目标点：桌面上的 xy，加上方块静止高度。"""
    return np.array([float(goal_xy[0]), float(goal_xy[1]), Z_REST], dtype=float)


def sample_task(rng):
    """抽一个起点和一个目标，两者至少隔开 ``TASK_MIN_SEPARATION`` 米。"""
    for _ in range(200):
        cube = rng.uniform(TASK_LOW, TASK_HIGH)
        goal = rng.uniform(TASK_LOW, TASK_HIGH)
        if float(np.linalg.norm(cube - goal)) >= TASK_MIN_SEPARATION:
            return cube.astype(float), goal.astype(float)
    raise RuntimeError("在采样盒子里抽不到足够分开的起点和目标")
