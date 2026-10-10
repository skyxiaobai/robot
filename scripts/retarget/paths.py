# -*- coding: utf-8 -*-
"""桌面抓放的末端路径：从复位高度接近、下降、夹紧、搬到目标、松开。

机器演示走直线。人手演示在搬运段加一个侧向弧，重定向之后弧还在，
这样两条数据不是同一条折线。
"""
from dataclasses import dataclass

import numpy as np

from retarget.frames import (
    GRIP_CLOSE,
    GRIP_OPEN,
    HOME_EE,
    Z_GRASP,
    Z_HOVER,
)


@dataclass(frozen=True)
class Segment:
    start: np.ndarray
    end: np.ndarray
    grip_start: float
    grip_end: float
    duration: float
    bow: float


def min_jerk(alpha):
    a = float(alpha)
    return 10 * a ** 3 - 15 * a ** 4 + 6 * a ** 5


def scripted_segments(cube_xy, goal_xy, arc_m=0.0):
    """返回按时间排列的末端线段。``arc_m`` 是搬运段的侧向最大偏移（米）。"""
    cube = np.asarray(cube_xy, dtype=float)
    goal = np.asarray(goal_xy, dtype=float)
    hover_c = np.array([cube[0], cube[1], Z_HOVER])
    hover_g = np.array([goal[0], goal[1], Z_HOVER])
    grasp = np.array([cube[0], cube[1], Z_GRASP])
    place = np.array([goal[0], goal[1], Z_GRASP])
    home = np.array(HOME_EE, dtype=float)
    return [
        Segment(home, hover_c, GRIP_OPEN, GRIP_OPEN, 0.8, 0.0),
        Segment(hover_c, grasp, GRIP_OPEN, GRIP_OPEN, 1.0, 0.0),
        Segment(grasp, grasp, GRIP_OPEN, GRIP_CLOSE, 0.5, 0.0),
        Segment(grasp, hover_c, GRIP_CLOSE, GRIP_CLOSE, 0.8, 0.0),
        Segment(hover_c, hover_g, GRIP_CLOSE, GRIP_CLOSE, 1.0, float(arc_m)),
        Segment(hover_g, place, GRIP_CLOSE, GRIP_CLOSE, 0.8, 0.0),
        Segment(place, place, GRIP_CLOSE, GRIP_OPEN, 0.4, 0.0),
        Segment(place, hover_g, GRIP_OPEN, GRIP_OPEN, 0.6, 0.0),
    ]


def path_duration(segments):
    return float(sum(segment.duration for segment in segments))


def _bow_offset(start, end, bow, alpha):
    if bow == 0.0:
        return 0.0, 0.0
    direction = np.asarray(end[:2], dtype=float) - np.asarray(start[:2], dtype=float)
    length = float(np.linalg.norm(direction))
    if length < 1e-8:
        return 0.0, 0.0
    lateral = np.array([-direction[1], direction[0]], dtype=float) / length
    shift = lateral * (bow * np.sin(np.pi * alpha))
    return float(shift[0]), float(shift[1])


def query_path(segments, time_s):
    """时间（秒）→ ``(xyz, grip)``。超出两端时夹在起点或终点。"""
    if not segments:
        raise ValueError("路径是空的")
    remaining = float(time_s)
    if remaining <= 0.0:
        first = segments[0]
        return first.start.copy(), float(first.grip_start)
    for segment in segments:
        if remaining <= segment.duration or segment is segments[-1]:
            alpha = 1.0 if segment.duration <= 0 else min(remaining / segment.duration, 1.0)
            blend = min_jerk(alpha)
            position = (1.0 - blend) * segment.start + blend * segment.end
            dx, dy = _bow_offset(segment.start, segment.end, segment.bow, alpha)
            position = position.copy()
            position[0] += dx
            position[1] += dy
            grip = (1.0 - blend) * segment.grip_start + blend * segment.grip_end
            return position, float(grip)
        remaining -= segment.duration
    last = segments[-1]
    return last.end.copy(), float(last.grip_end)


def sample_path(cube_xy, goal_xy, dt, arc_m=0.0):
    """按固定步长采样。返回 times, xyz (N,3), grip (N,)。包含 t=0。"""
    segments = scripted_segments(cube_xy, goal_xy, arc_m=arc_m)
    duration = path_duration(segments)
    times = [0.0]
    positions = []
    grips = []
    pos0, grip0 = query_path(segments, 0.0)
    positions.append(pos0)
    grips.append(grip0)
    t = float(dt)
    while t < duration - 1e-9:
        pos, grip = query_path(segments, t)
        times.append(t)
        positions.append(pos)
        grips.append(grip)
        t += float(dt)
    pos, grip = query_path(segments, duration)
    if times[-1] < duration - 1e-9:
        times.append(duration)
        positions.append(pos)
        grips.append(grip)
    return (
        np.asarray(times, dtype=float),
        np.vstack(positions),
        np.asarray(grips, dtype=float),
        duration,
    )
