# -*- coding: utf-8 -*-
"""用物体包围盒生成抓放场景。

夹爪完全合上时，两指垫内侧面相距 2.4 cm（指根在 y=±1.6 cm，指垫半厚 0.4 cm）。
比这更薄的物体夹不住。张开约 8 cm。桌子上最长边限制在 8 cm。

缩放是统一的，而且不会把物体放大。某一轴落在 2.6–7.0 cm 里才算夹得住，
取最接近方块宽度 3.6 cm 的那一轴对准夹爪（机器人 Y）。对不准就仍生成场景，
``pinchable`` 为假，重放失败就记失败，不用焊接把物体粘在夹爪上。

连续对称（碗、盘子、笔）用圆柱，其余用盒子。仿真里的朝向是轴对齐的，
不用 HOT3D 的物体四元数。成功仍只看水平 4 cm 和桌面高度，不看 6DoF 朝向。
"""
import re
from pathlib import Path

import numpy as np

from retarget.frames import TABLE_TOP

XML_PATH = Path(__file__).with_name("franka_like_pick.xml")

# 合拢间隙 0.024 m。下限留 2 mm，避免刚好蹭到却没有夹紧力。
PINCH_LO_M = 0.026
PINCH_HI_M = 0.070
PINCH_TARGET_M = 0.036
MAX_EXTENT_M = 0.080
# 方块体积，用来按体积给一个差不多的质量。
_CUBE_VOLUME = (2 * 0.018) ** 3


def model_sizes(info):
    """``models_info`` 的一条。返回 (min_xyz, size_xyz)。"""
    minimum = np.array([info["min_x"], info["min_y"], info["min_z"]], dtype=float)
    size = np.array([info["size_x"], info["size_y"], info["size_z"]], dtype=float)
    if np.any(size <= 0):
        raise ValueError("包围盒边长要为正")
    return minimum, size


def plan_geometry(info):
    """由包围盒决定盒子或圆柱、统一缩放，以及哪一轴对着夹爪。"""
    minimum, size = model_sizes(info)
    scale = min(1.0, float(MAX_EXTENT_M / np.max(size)))
    scaled = size * scale
    candidates = [i for i in range(3) if PINCH_LO_M <= float(scaled[i]) <= PINCH_HI_M]
    pinchable = bool(candidates)
    if candidates:
        pinch_i = min(candidates, key=lambda i: abs(float(scaled[i]) - PINCH_TARGET_M))
    else:
        pinch_i = int(np.argmax(scaled))
    remaining = [i for i in range(3) if i != pinch_i]

    def _vertical_cost(index):
        length = float(scaled[index])
        # 太高会让静止中心超出成功高度带（桌面 0.22，上限 0.30）。
        penalty = 0.0 if length <= 0.10 else 10.0
        return penalty + abs(length - PINCH_TARGET_M)

    vert_i = min(remaining, key=_vertical_cost)
    long_i = remaining[0] if remaining[1] == vert_i else remaining[1]
    continuous = bool(info.get("symmetries_continuous"))
    kind = "box"
    axis = None
    radius = None
    halfheight = None
    hx = float(scaled[long_i] / 2.0)
    hy = float(scaled[pinch_i] / 2.0)
    hz = float(scaled[vert_i] / 2.0)
    if continuous:
        # 对称轴是和另外两条最不像的那条。另外两条是直径。
        spread = [abs(float(scaled[i] - np.median(scaled))) for i in range(3)]
        sym_i = int(np.argmax(spread))
        diameters = [i for i in range(3) if i != sym_i]
        d0, d1 = float(scaled[diameters[0]]), float(scaled[diameters[1]])
        if abs(d0 - d1) <= 0.2 * max(d0, d1, 1e-6):
            kind = "cylinder"
            radius = 0.5 * 0.5 * (d0 + d1)
            halfheight = float(scaled[sym_i] / 2.0)
            if sym_i == pinch_i:
                # 夹的是圆柱的高度：轴线沿夹爪开合方向（Y）。
                axis = "y"
                hy = halfheight
                hx = radius
                hz = radius
            else:
                # 夹的是直径：圆柱立在桌上，轴线沿 Z。
                axis = "z"
                hy = radius
                hx = radius
                hz = halfheight
    grasp_z = float(np.clip(TABLE_TOP + hz + 0.004, 0.230, 0.280))
    rest_z = float(TABLE_TOP + hz + 0.004)
    return {
        "kind": kind,
        "axis": axis,
        "scale": float(scale),
        "pinchable": pinchable,
        "pinch_m": float(scaled[pinch_i]),
        "half_x": hx,
        "half_y": hy,
        "half_z": hz,
        "radius": None if radius is None else float(radius),
        "halfheight": None if halfheight is None else float(halfheight),
        "grasp_z": grasp_z,
        "rest_z": rest_z,
        "bbox_min": minimum,
        "bbox_size": size,
        "bbox_center_local": minimum + size / 2.0,
        "mass": _mass(hx, hy, hz),
    }


def _mass(hx, hy, hz):
    volume = 8.0 * float(hx) * float(hy) * float(hz)
    if _CUBE_VOLUME <= 0:
        return 0.05
    return float(np.clip(0.05 * volume / _CUBE_VOLUME, 0.03, 0.20))


def _geom_xml(plan):
    mass = plan["mass"]
    common = (
        'rgba="0.9 0.75 0.1 1" friction="1.8 0.05 0.01" condim="4" mass="%.4f"' % mass
    )
    if plan["kind"] == "cylinder" and plan["axis"] == "z":
        return '<geom name="cube" type="cylinder" size="%.6f %.6f" %s/>' % (
            plan["radius"], plan["halfheight"], common
        )
    if plan["kind"] == "cylinder" and plan["axis"] == "y":
        # 局部 +Z 转到世界 +Y：绕 X 转 90 度。
        return (
            '<geom name="cube" type="cylinder" size="%.6f %.6f" quat="0.70710678 0.70710678 0 0" %s/>'
            % (plan["radius"], plan["halfheight"], common)
        )
    return '<geom name="cube" type="box" size="%.6f %.6f %.6f" %s/>' % (
        plan["half_x"], plan["half_y"], plan["half_z"], common
    )


def scene_xml(plan, template=None):
    """换成新的物体几何，手臂和夹爪不动。"""
    text = Path(template).read_text(encoding="utf-8") if template else XML_PATH.read_text(encoding="utf-8")
    body = (
        '    <body name="cube" pos="0.46 0.0 0.24">\n'
        '      <freejoint name="cube_free"/>\n'
        "      %s\n"
        "    </body>"
    ) % _geom_xml(plan)
    replaced, count = re.subn(
        r"    <body name=\"cube\" pos=\"0\.46 0\.0 0\.24\">.*?</body>",
        body,
        text,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise RuntimeError("场景模板里没有找到方块刚体")
    return replaced


def geometry_key(plan):
    """相同几何共用一个 MuJoCo 模型。"""
    return (
        plan["kind"],
        plan["axis"],
        round(plan["half_x"], 4),
        round(plan["half_y"], 4),
        round(plan["half_z"], 4),
    )
