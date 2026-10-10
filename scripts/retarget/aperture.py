# -*- coding: utf-8 -*-
"""用拇指尖和食指尖的距离当作夹爪开合。

21 点顺序与 ``egodata.schema.MEDIAPIPE_21`` 相同：4 是拇指尖，8 是食指尖。
"""
import numpy as np

from retarget.frames import GRIP_CLOSE, GRIP_OPEN

THUMB_TIP = 4
INDEX_TIP = 8

# 合成演示里张开约 9 cm、捏紧约 2 cm。阈值落在两者中间，并带一点滞回。
APERTURE_OPEN_M = 0.08
APERTURE_CLOSE_M = 0.025
GRASP_CLOSE_M = 0.04
GRASP_OPEN_M = 0.065


def apertures_from_joints(joints):
    """``joints`` 为 (21, 3) 或 (T, 21, 3)，返回开合距离（米）。"""
    points = np.asarray(joints, dtype=float)
    single = points.ndim == 2
    if single:
        points = points[None, ...]
    if points.shape[-2:] != (21, 3):
        raise ValueError("关节必须是 21×3，实际是 %s" % (points.shape,))
    distance = np.linalg.norm(points[:, THUMB_TIP] - points[:, INDEX_TIP], axis=1)
    if single:
        return float(distance[0])
    return distance


def grip_command(aperture, open_m=APERTURE_OPEN_M, close_m=APERTURE_CLOSE_M):
    """开合距离 → 夹爪滑动关节位置。0 是夹紧，``GRIP_OPEN`` 是张开。"""
    values = np.asarray(aperture, dtype=float)
    span = float(open_m - close_m)
    if span <= 0:
        raise ValueError("张开距离必须大于闭合距离")
    unit = np.clip((values - close_m) / span, 0.0, 1.0)
    command = GRIP_CLOSE + unit * (GRIP_OPEN - GRIP_CLOSE)
    if np.isscalar(aperture) or (isinstance(aperture, np.ndarray) and aperture.ndim == 0):
        return float(command)
    return command


def grasp_closed_mask(aperture, close_m=GRASP_CLOSE_M, open_m=GRASP_OPEN_M):
    """带滞回的闭合标记：降到 close_m 以下算握住，升到 open_m 以上算松开。"""
    values = np.asarray(aperture, dtype=float).reshape(-1)
    closed = np.zeros(values.shape[0], dtype=bool)
    flag = False
    for index, value in enumerate(values):
        if not flag and value <= close_m:
            flag = True
        elif flag and value >= open_m:
            flag = False
        closed[index] = flag
    return closed
