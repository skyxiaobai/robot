# -*- coding: utf-8 -*-
"""把手腕轨迹写成相对物体的量。

开环、而且基座用的是记录里的物体轨迹时，``物体 + (手腕 - 物体)`` 就是手腕本身。
所以物体坐标系的差别要落在两处：

1. 相对偏移用物体几何的统一缩放，不用工作空间那把缩放。
   这样缩小后的物体边上，手腕仍贴着物体，而不是停在十几厘米外。
   物体在桌面上的起点和终点仍用工作空间标定，搬运距离不会被缩成 0。
2. 在线修正看的是仿真里物体的当前位置，而不是记录里的位置。

EgoDex 没有物体位姿，也没有几何缩放。能做的是：抓住那一帧的高度对齐到夹取高度，
搬运仍跟着手腕。在线修正时偏移是 0，末端伺服到仿真方块的当前位置。
"""
import numpy as np

from retarget.arkit import ROBOT_BOX_MAX, ROBOT_BOX_MIN, permute_arkit_xyz
from retarget.frames import Z_GRASP


def robot_relative(wrist_yup, object_yup, geometry_scale):
    """Y 朝上的手腕和物体中心 → 机器人轴顺序下、按几何缩放后的相对位移 (T, 3)。"""
    delta = np.asarray(wrist_yup, dtype=float) - np.asarray(object_yup, dtype=float)
    return permute_arkit_xyz(delta) * float(geometry_scale)


def object_centric_ee(object_robot, relative, grasp_index, release_index, grasp_z=Z_GRASP):
    """接近段贴在抓住时的物体上；搬运段用固定的抓住偏移跟着物体走。

    ``object_robot`` 和 ``relative`` 已经是机器人坐标，单位米。
    抓住那一帧的高度再平移到 ``grasp_z``，好让指垫落在物体半高。
    """
    obj = np.asarray(object_robot, dtype=float)
    rel = np.asarray(relative, dtype=float)
    if obj.shape != rel.shape:
        raise ValueError("物体轨迹和相对位移长度不一致")
    grasp = int(grasp_index)
    release = int(release_index)
    offset = rel[grasp].copy()
    ee = np.empty_like(obj)
    ee[:grasp + 1] = obj[grasp] + rel[:grasp + 1]
    ee[grasp:release + 1] = obj[grasp:release + 1] + offset
    if release + 1 < obj.shape[0]:
        # 松开之后仍贴着物体，但用松开后的相对位移，避免手腕已经拿开还粘在物体上。
        ee[release + 1:] = obj[release + 1:] + rel[release + 1:]
    ee[:, 2] += float(grasp_z) - float(ee[grasp, 2])
    return np.clip(ee, ROBOT_BOX_MIN, ROBOT_BOX_MAX)


def align_grasp_height(wrist_robot, grasp_index, grasp_z=Z_GRASP):
    """没有物体位姿时：整段高度平移，使抓住那一帧落在夹取高度。水平路径不动。"""
    ee = np.asarray(wrist_robot, dtype=float).copy()
    ee[:, 2] += float(grasp_z) - float(ee[int(grasp_index), 2])
    return np.clip(ee, ROBOT_BOX_MIN, ROBOT_BOX_MAX)
