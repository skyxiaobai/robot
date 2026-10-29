# -*- coding: utf-8 -*-
"""用相邻正步长区分秒和纳秒。

会话规范是秒。ORB-SLAM3 的 EuRoC 例子通常已经换成秒，Basalt 和一部分
导出仍是纳秒。只看绝对值会把「从 0 起算的纳秒」和 Unix 秒混在一起，
所以这里看中位正步长：30 fps 的秒大约是 0.033，纳秒大约是 3.3e7。

本模块不改写调用者的数组，也不在两种单位之间自动换算。
"""
import numpy as np

UNIT_SECONDS = "seconds"
UNIT_NANOSECONDS = "nanoseconds"
UNIT_UNKNOWN = "unknown"

# 30 fps 的纳秒步长约 3.3e7。1e5 仍远大于任何按秒计的帧间隔（10 秒一帧）。
_NS_STEP = 1e5
# 慢于大约 0.1 Hz 的秒时间戳不再猜测，避免把毫秒步长当成秒。
_SEC_STEP = 10.0
# 只有一个样本时才看数量级。1e6 秒约 11 天；1e14 纳秒约 1.2 天。
_SINGLE_SEC = 1e6
_SINGLE_NS = 1e14


def finite_values(values):
    array = np.asarray(list(values) if values is not None else [], dtype=float).reshape(-1)
    return array[np.isfinite(array)]


def median_positive_step(values):
    """相邻正差分的中位数。不足两个有限样本，或没有正步长时返回 None。"""
    array = finite_values(values)
    if array.size < 2:
        return None
    step = np.diff(array)
    step = step[step > 0]
    if step.size == 0:
        return None
    return float(np.median(step))


def clock_unit(values):
    """返回 ``seconds``、``nanoseconds`` 或 ``unknown``。不改写数值。"""
    array = finite_values(values)
    if array.size == 0:
        return UNIT_UNKNOWN
    step = median_positive_step(array)
    if step is not None:
        if step >= _NS_STEP:
            return UNIT_NANOSECONDS
        if step <= _SEC_STEP:
            return UNIT_SECONDS
        return UNIT_UNKNOWN
    magnitude = float(np.max(np.abs(array)))
    if magnitude >= _SINGLE_NS:
        return UNIT_NANOSECONDS
    if magnitude <= _SINGLE_SEC:
        return UNIT_SECONDS
    return UNIT_UNKNOWN


def parse_time_scale(value):
    """显式缩放系数。缺省是 1。非正数直接拒绝，避免把时间轴翻过去。"""
    if value is None:
        return 1.0
    scale = float(value)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("tum_time_scale 必须是正数，本模块不自动猜测单位")
    return scale


def describe_clocks(frame_times, tum_times, tum_time_scale=None):
    """比较会话帧时间和轨迹时间。返回关联用的时间以及是否单位不一致。

    ``tum_time_scale`` 先乘到轨迹时间上，再比较单位。默认 1，不自动把纳秒除成秒。
    单位不一致时调用方仍可用返回的时间做关联，但不应采纳跨单位的位姿。
    """
    scale = parse_time_scale(tum_time_scale)
    raw = np.asarray(list(tum_times) if tum_times is not None else [], dtype=float).reshape(-1)
    scaled = raw * scale
    frame_unit = clock_unit(frame_times)
    file_unit = clock_unit(raw) if raw.size else UNIT_UNKNOWN
    used_unit = clock_unit(scaled) if scaled.size else UNIT_UNKNOWN
    known = (UNIT_SECONDS, UNIT_NANOSECONDS)
    mismatch = frame_unit in known and used_unit in known and frame_unit != used_unit
    notes = []
    if raw.size and scale != 1.0:
        notes.append(
            "已按显式 tum_time_scale=%g 缩放轨迹时间后再关联。文件单位记为 %s。"
            % (scale, file_unit)
        )
    if mismatch:
        notes.append(
            "时间单位不一致：会话帧时间是 %s，用于关联的轨迹时间是 %s（文件单位 %s，scale=%g）。"
            "不自动缩放。跨单位不采纳位姿，这些帧是 tracking_lost，T_world_cam 为 null。"
            "轨迹是纳秒、会话是秒时，请显式传入 --tum-time-scale 1e-9，不要放宽关联门限。"
            % (frame_unit, used_unit, file_unit, scale)
        )
    elif raw.size and (frame_unit == UNIT_UNKNOWN or used_unit == UNIT_UNKNOWN) and frame_unit != used_unit:
        notes.append(
            "有一边时间单位无法判断（帧=%s，轨迹=%s）。只区分秒和纳秒，不猜测毫秒，也不自动缩放。"
            % (frame_unit, used_unit)
        )
    if frame_unit == UNIT_NANOSECONDS:
        notes.append(
            "timestamps.csv 的步长像纳秒。规范要求秒。本模块不把会话时间戳除以 1e9，"
            "下游速度和半帧门限都按文件里的数字解释。"
        )
    return {
        "frame_time_unit": frame_unit,
        "tum_time_unit": file_unit,
        "tum_time_scale": scale,
        "time_unit_mismatch": mismatch,
        "times_for_association": scaled,
        "note": " ".join(notes),
    }
