# -*- coding: utf-8 -*-
"""从真实 EgoDex 片段里找出「哪只手、何时抓住、何时松开」。

EgoDex 没有物体位姿。方块起点用抓住那一帧的手腕水平位置，目标用松开那一帧。
只保留主动手 ``wrist_measured`` 连续为真的一段：这段里相邻帧满足 ``action_valid``。
"""
import numpy as np

from egodata.action_valid import wrist_measured
from retarget.aperture import INDEX_TIP, THUMB_TIP
from retarget.frames import GRIP_CLOSE, GRIP_OPEN, HUMAN_FPS

# 平滑后的开合距离，10% 到 90% 分位至少要拉开这么多，才认为有张有合。
APERTURE_MIN_SPAN_M = 0.012
# 滞回：降到区间的 35% 算握住，升到 65% 算松开。阈值跟着这一段自己的开合走，
# 不用合成演示里 2.5 cm / 8 cm 的绝对数（真实片段经常整段都比那更开或更闭）。
CLOSE_FRAC = 0.35
OPEN_FRAC = 0.65
SMOOTH_WINDOW = 5
# 抓住和松开在水平面上至少隔开这么远，才像一次放下而不是原地张合。
MIN_HUMAN_TRAVEL_M = 0.05
# 映射到机器人桌子之后还要隔开这么远。成功半径是 4 cm，隔开不到 4 cm 时方块不动也会被算成功。
# 6 cm 比成功半径多 2 cm，方块不动一定失败。
MIN_ROBOT_SEPARATION_M = 0.06
MIN_MEASURED_FRAMES = 45
MARGIN_S = 1.0
DOWNSAMPLE_STRIDE = 3


def smooth_aperture(aperture, window=SMOOTH_WINDOW):
    values = np.asarray(aperture, dtype=float).reshape(-1)
    window = int(window)
    if window <= 1 or values.shape[0] < window:
        return values.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values, kernel, mode="same")


def grasp_release_pairs(aperture, min_span_m=APERTURE_MIN_SPAN_M, close_frac=CLOSE_FRAC, open_frac=OPEN_FRAC):
    """返回 (抓住帧, 松开帧) 列表。信号从闭合开始时，抓住帧是 0。

    没有足够的张合，或者闭合之后没有再张开，返回空列表。
    """
    values = smooth_aperture(aperture)
    if values.shape[0] < 2 or not np.isfinite(values).all():
        return []
    lo = float(np.percentile(values, 10))
    hi = float(np.percentile(values, 90))
    if hi - lo < float(min_span_m):
        return []
    close_th = lo + float(close_frac) * (hi - lo)
    open_th = lo + float(open_frac) * (hi - lo)
    if not (close_th < open_th):
        return []
    pairs = []
    grasp = 0 if values[0] <= close_th else None
    for index, value in enumerate(values):
        if grasp is None:
            if value <= close_th:
                grasp = int(index)
            continue
        if value >= open_th and index > grasp:
            pairs.append((int(grasp), int(index)))
            grasp = None
    return pairs


def longest_true_run(mask):
    """返回最长连续 True 的 [start, end)。空掩码返回 (0, 0)。"""
    flags = np.asarray(mask, dtype=bool).reshape(-1)
    best = (0, 0)
    start = None
    for index, flag in enumerate(flags):
        if flag and start is None:
            start = index
        if start is not None and (not flag or index == flags.shape[0] - 1):
            end = index + 1 if flag and index == flags.shape[0] - 1 else index
            if end - start > best[1] - best[0]:
                best = (int(start), int(end))
            start = None
    return best


def horizontal_travel(wrist_xyz, grasp, release):
    """ARKit 世界系里水平面是 XZ（Y 朝上）。返回抓住点到松开点的水平距离。"""
    points = np.asarray(wrist_xyz, dtype=float)
    delta = points[int(release), [0, 2]] - points[int(grasp), [0, 2]]
    return float(np.linalg.norm(delta))


def pickup_index(human_xyz, grip, grasp_index, release_index, grip_closed=0.012):
    """闭合区间里手腕最低的那一帧。阈值刚跨过的那一帧常常还在空中。

    ``human_xyz`` 用 ARKit 坐标，第 1 轴朝上。没有明显闭合时退回抓住帧。
    """
    heights = np.asarray(human_xyz, dtype=float)[:, 1]
    grip = np.asarray(grip, dtype=float).reshape(-1)
    grasp = int(grasp_index)
    release = int(release_index)
    indices = np.arange(grip.shape[0])
    closed = np.flatnonzero((indices >= grasp) & (indices <= release) & (grip <= float(grip_closed)))
    pool = closed if closed.size else np.arange(grasp, release + 1)
    return int(pool[int(np.argmin(heights[pool]))])


def choose_grasp_release(aperture, wrist_xyz):
    """在所有闭合-张开对里，取水平位移最大、且超过 ``MIN_HUMAN_TRAVEL_M`` 的一对。"""
    best = None
    best_travel = -1.0
    for grasp, release in grasp_release_pairs(aperture):
        travel = horizontal_travel(wrist_xyz, grasp, release)
        if travel + 1e-9 < MIN_HUMAN_TRAVEL_M:
            continue
        if travel > best_travel:
            best = (int(grasp), int(release), float(travel))
            best_travel = travel
    return best


def normalized_grip(aperture):
    """把这一段自己的开合距离线性映射到夹爪关节。张得最大的一档是张开。"""
    values = np.asarray(aperture, dtype=float).reshape(-1)
    lo = float(np.percentile(values, 10))
    hi = float(np.percentile(values, 90))
    if hi - lo < APERTURE_MIN_SPAN_M:
        return None
    unit = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    return GRIP_CLOSE + unit * (GRIP_OPEN - GRIP_CLOSE)


def _finite_joints(frame):
    if frame is None:
        return False
    array = np.asarray(frame, dtype=float)
    return array.shape == (21, 3) and bool(np.isfinite(array).all())


def aperture_series(hand):
    """拇指尖到食指尖。关节缺失的帧是 NaN。"""
    distances = np.full(len(hand["joints"]), np.nan, dtype=float)
    for index, frame in enumerate(hand["joints"]):
        if not _finite_joints(frame):
            continue
        array = np.asarray(frame, dtype=float)
        distances[index] = float(np.linalg.norm(array[THUMB_TIP] - array[INDEX_TIP]))
    return distances


def wrist_xyz_series(hand):
    count = len(hand["wrist_pose"])
    xyz = np.full((count, 3), np.nan, dtype=float)
    quat = np.full((count, 4), np.nan, dtype=float)
    for index, pose in enumerate(hand["wrist_pose"]):
        if pose is None:
            continue
        array = np.asarray(pose, dtype=float).reshape(-1)
        if array.shape[0] != 7 or not np.isfinite(array).all():
            continue
        xyz[index] = array[:3]
        quat[index] = array[3:]
    return xyz, quat


def tagged_hand(episode):
    """读 environment 里的 ``hand:left`` / ``hand:right``。没有或两个都写了就返回 None。"""
    annotation = episode.get("annotation") or {}
    environment = annotation.get("environment") or {}
    detail = str(environment.get("detail") or "").lower().replace(" ", "")
    has_left = "hand:left" in detail
    has_right = "hand:right" in detail
    if has_left and not has_right:
        return "left"
    if has_right and not has_left:
        return "right"
    return None


def _usable_mask(episode, side):
    hand = episode["hands"][side]
    count = int(episode["num_frames"])
    mask = np.zeros(count, dtype=bool)
    for index in range(count):
        if not wrist_measured(episode, side, index):
            continue
        if not _finite_joints(hand["joints"][index]):
            continue
        mask[index] = True
    return mask


def _window(run_start, run_end, grasp, release, count, fps=HUMAN_FPS, margin_s=MARGIN_S):
    margin = int(round(float(margin_s) * float(fps)))
    start = max(int(run_start), int(grasp) - margin)
    end = min(int(run_end), int(release) + margin + 1, int(count))
    if end - start < 2:
        return None
    return int(start), int(end)


def _downsample_indices(start, end, grasp, release, stride):
    keep = list(range(int(start), int(end), int(stride)))
    if not keep or keep[-1] != int(end) - 1:
        # 把窗口最后一帧留住，避免松开点落在被丢掉的尾巴上。
        if int(end) - 1 not in keep:
            keep.append(int(end) - 1)
    keep = sorted(set(keep))

    def nearest(frame):
        return min(keep, key=lambda item: (abs(item - int(frame)), item))

    grasp_frame = nearest(grasp)
    release_frame = nearest(release)
    if release_frame <= grasp_frame:
        return None
    return keep, keep.index(grasp_frame), keep.index(release_frame)


def prepare_side(episode, side):
    """切出这一只手的可用窗口。失败时返回 ``(None, reason)``。"""
    mask = _usable_mask(episode, side)
    run_start, run_end = longest_true_run(mask)
    if run_end - run_start < MIN_MEASURED_FRAMES:
        return None, "short_measured_run"
    hand = episode["hands"][side]
    aperture = aperture_series(hand)
    xyz, quat = wrist_xyz_series(hand)
    segment_aperture = aperture[run_start:run_end]
    segment_xyz = xyz[run_start:run_end]
    if not np.isfinite(segment_aperture).all() or not np.isfinite(segment_xyz).all():
        return None, "non_finite_in_run"
    chosen = choose_grasp_release(segment_aperture, segment_xyz)
    if chosen is None:
        return None, "no_grasp_release"
    grasp, release, travel = chosen
    grasp += run_start
    release += run_start
    bounds = _window(run_start, run_end, grasp, release, episode["num_frames"])
    if bounds is None:
        return None, "short_window"
    start, end = bounds
    sampled = _downsample_indices(start, end, grasp, release, DOWNSAMPLE_STRIDE)
    if sampled is None:
        return None, "downsample_collapsed"
    frames, grasp_index, release_index = sampled
    window_aperture = aperture[frames]
    grip = normalized_grip(window_aperture)
    if grip is None:
        return None, "flat_aperture"
    times = (np.asarray(frames, dtype=float) - float(frames[0])) / float(HUMAN_FPS)
    return {
        "side": side,
        "frames": np.asarray(frames, dtype=int),
        "wrist_xyz": xyz[frames],
        "wrist_quat": quat[frames],
        "grip": np.asarray(grip, dtype=float),
        "times": times,
        "grasp_index": int(grasp_index),
        "release_index": int(release_index),
        "human_travel_m": float(travel),
        "aperture_span_m": float(np.percentile(window_aperture, 90) - np.percentile(window_aperture, 10)),
        "n_measured_run": int(run_end - run_start),
        "n_kept": int(len(frames)),
    }, None


def prepare_episode(episode):
    """优先用标注里的那只手。那只手没有抓放时再试另一只，并记 ``side_fallback``。"""
    preferred = tagged_hand(episode)
    order = [preferred] if preferred in ("left", "right") else []
    for side in ("right", "left"):
        if side not in order:
            order.append(side)
    errors = {}
    for side in order:
        prepared, reason = prepare_side(episode, side)
        if prepared is not None:
            prepared["side_fallback"] = bool(preferred in ("left", "right") and side != preferred)
            prepared["tagged_hand"] = preferred
            return prepared, None
        errors[side] = reason
    return None, errors
