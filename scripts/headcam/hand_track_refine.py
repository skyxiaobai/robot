# -*- coding: utf-8 -*-
"""把手部 21 点沿时间修稳：平滑、补短缺口、固定手型。

检测器逐帧出结果，相邻帧会抖，手被挡住时会空几帧，左右标签偶尔对调，
骨长也会跟着噪声变。这一层不重新跑 WiLoR / MediaPipe，只吃已经得到的
``(T, 21, 3)`` 关节。默认什么都不做；调用方打开开关才改数。

处理顺序（每一项都可以单独关）：

1. 左右手轨迹一致性。用上一帧手腕的位置认人，而不是只信这一帧的标签。
2. 时序平滑。只把高置信的观测送进滤波器。One Euro 是默认；也可以换成
   常速度卡尔曼。缺测帧先留空，不拿预测值冒充观测。
3. 补洞。两头都有观测、中间连续缺测不超过 ``max_gap`` 帧时，线性插值。
   这些帧写 ``filled=True``，置信度写成 ``filled_confidence``（默认 0）。
   QC 看到 ``filled`` 或低于 0.5 的置信度，都不会把这帧当成跟踪成功。
4. 固定手型。在这一段的高置信帧上，对每段骨头取稳健中位数，再按骨架把
   关节重摆到这个长度。手腕不动。这是「这个人的手有多长」，一段视频算一次。

有相机位姿时，平滑和补洞在世界系里做，再变回相机系。头在转时，相机系里的
手会跟着画幅跑，那不是手抖。没有位姿时就在传入的坐标系里做。

MANO 的 betas 是官方形状空间里的 10 维系数，需要 ``MANO_RIGHT.pkl`` 和
能输出 21 点的前向模型。本模块不内置 MANO，也不把骨长假装成 betas。
调用方如果传入 ``mano_length_fn(betas) -> 21 段骨长``，会顺带用最小二乘
拟合 betas，关节仍按测到的骨长重摆。没有这个函数时，形状就是骨长。
"""
import math

import numpy as np

from headcam.hand_pose import (
    JOINTS,
    transform_points,
    world_to_camera,
    wrist_poses_from_joints,
)

# MediaPipe 21 点的父节点。下标 0 是手腕，没有父节点。
# 子节点的下标总是大于父节点，所以顺着 1..20 重摆骨头不会用到还没更新的父节点。
MEDIAPIPE_PARENTS = np.array([
    -1,
    0, 1, 2, 3,
    0, 5, 6, 7,
    0, 9, 10, 11,
    0, 13, 14, 15,
    0, 17, 18, 19,
], dtype=np.int64)


class RefineParams(object):
    """时序精修的开关和数值。长度单位是米，时间单位是秒，截止频率是 Hz。

    ``beta`` 的单位是 1 / (米/秒)：手移动越快，One Euro 的截止频率越高，越少滞后。
    ``kalman_accel_std`` 是加速度噪声（米/秒²），``kalman_meas_std`` 是测量噪声（米）。
    """

    def __init__(
        self,
        smooth="none",
        min_cutoff=1.0,
        beta=0.5,
        d_cutoff=1.0,
        kalman_accel_std=5.0,
        kalman_meas_std=0.02,
        gap_fill=False,
        max_gap=5,
        confidence_min=0.5,
        filled_confidence=0.0,
        fixed_shape=False,
        lr_consistency=False,
        lr_margin_m=0.02,
        lr_max_match_m=0.35,
        track_memory_frames=15,
        fps=30.0,
        mano_length_fn=None,
        n_betas=10,
    ):
        smooth = str(smooth or "none").lower()
        if smooth not in ("none", "one_euro", "kalman"):
            raise ValueError("smooth 只能是 none、one_euro 或 kalman")
        if float(min_cutoff) <= 0.0 or float(d_cutoff) <= 0.0:
            raise ValueError("One Euro 的截止频率必须为正")
        if float(beta) < 0.0:
            raise ValueError("One Euro 的 beta 不能为负")
        if float(kalman_accel_std) <= 0.0 or float(kalman_meas_std) <= 0.0:
            raise ValueError("卡尔曼的噪声标准差必须为正")
        if int(max_gap) < 0:
            raise ValueError("max_gap 不能为负")
        if float(fps) <= 0.0:
            raise ValueError("fps 必须为正")
        self.smooth = smooth
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.kalman_accel_std = float(kalman_accel_std)
        self.kalman_meas_std = float(kalman_meas_std)
        self.gap_fill = bool(gap_fill)
        self.max_gap = int(max_gap)
        self.confidence_min = float(confidence_min)
        self.filled_confidence = float(filled_confidence)
        self.fixed_shape = bool(fixed_shape)
        self.lr_consistency = bool(lr_consistency)
        self.lr_margin_m = float(lr_margin_m)
        self.lr_max_match_m = float(lr_max_match_m)
        self.track_memory_frames = int(track_memory_frames)
        self.fps = float(fps)
        self.mano_length_fn = mano_length_fn
        self.n_betas = int(n_betas)

    def enabled(self):
        return bool(
            self.smooth != "none" or self.gap_fill or self.fixed_shape or self.lr_consistency
        )

    def to_dict(self):
        return {
            "smooth": self.smooth,
            "min_cutoff": self.min_cutoff,
            "beta": self.beta,
            "d_cutoff": self.d_cutoff,
            "kalman_accel_std": self.kalman_accel_std,
            "kalman_meas_std": self.kalman_meas_std,
            "gap_fill": self.gap_fill,
            "max_gap": self.max_gap,
            "confidence_min": self.confidence_min,
            "filled_confidence": self.filled_confidence,
            "fixed_shape": self.fixed_shape,
            "lr_consistency": self.lr_consistency,
            "lr_margin_m": self.lr_margin_m,
            "lr_max_match_m": self.lr_max_match_m,
            "track_memory_frames": self.track_memory_frames,
            "fps": self.fps,
            "mano_betas": self.mano_length_fn is not None,
        }


def ablation_presets():
    """消融用的五档。参数是事先定的默认值，不是在测试片段上搜出来的。"""
    return (
        ("baseline", RefineParams()),
        ("smoothing", RefineParams(smooth="one_euro")),
        ("gap_fill", RefineParams(gap_fill=True, max_gap=5, lr_consistency=True)),
        ("fixed_shape", RefineParams(fixed_shape=True)),
        ("all", RefineParams(
            smooth="one_euro", gap_fill=True, max_gap=5, fixed_shape=True, lr_consistency=True,
        )),
    )


def _as_joints(value, count):
    out = np.full((count, JOINTS, 3), np.nan, dtype=np.float64)
    if value is None:
        return out
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (count, JOINTS, 3):
        raise ValueError("关节序列必须是 (%d, 21, 3)，得到 %s" % (count, array.shape))
    finite = np.isfinite(array)
    out[finite] = array[finite]
    return out


def _as_confidence(value, count):
    """返回 ``(T, 21)``。未知置信度是 NaN，不当成 0。"""
    out = np.full((count, JOINTS), np.nan, dtype=np.float64)
    if value is None:
        return out
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (count,):
        out[:] = array.reshape(count, 1)
        return out
    if array.shape == (count, JOINTS):
        finite = np.isfinite(array)
        out[finite] = array[finite]
        return out
    raise ValueError("置信度必须是 (%d,) 或 (%d, 21)" % (count, count))


def _as_timestamps(timestamps, count, params):
    if timestamps is None:
        return np.arange(count, dtype=np.float64) / float(params.fps)
    array = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if array.shape[0] != count:
        raise ValueError("timestamps 长度必须是 %d" % count)
    return array


def _as_poses(camera_poses, count):
    if camera_poses is None:
        return None
    poses = np.asarray(camera_poses, dtype=np.float64)
    if poses.shape != (count, 4, 4):
        raise ValueError("camera_poses 必须是 (%d, 4, 4)" % count)
    return poses


def _map_frame(joints, poses, to_world):
    out = np.full_like(joints, np.nan)
    for index in range(joints.shape[0]):
        if to_world:
            out[index] = transform_points(joints[index], poses[index])
        else:
            out[index] = world_to_camera(joints[index], poses[index])
    return out


def _frame_observed(joints, confidence, params):
    wrist = joints[:, 0, :]
    finite = np.isfinite(wrist).all(axis=1)
    score = confidence[:, 0]
    unknown = ~np.isfinite(score)
    high = score >= float(params.confidence_min)
    return finite & (unknown | high)


def _swap_frame(left, right, left_conf, right_conf, index):
    left_row = left[index].copy()
    right_row = right[index].copy()
    left[index] = right_row
    right[index] = left_row
    left_score = left_conf[index].copy()
    right_score = right_conf[index].copy()
    left_conf[index] = right_score
    right_conf[index] = left_score


def _track_wrist(track, frame_index, memory):
    if track is None:
        return None
    wrist, seen = track
    if frame_index - int(seen) > int(memory):
        return None
    return wrist


def enforce_left_right(left, right, left_conf, right_conf, observed_left, observed_right, params):
    """按手腕轨迹把对调的左右标签换回来。返回交换标记。"""
    left = np.array(left, copy=True)
    right = np.array(right, copy=True)
    left_conf = np.array(left_conf, copy=True)
    right_conf = np.array(right_conf, copy=True)
    swapped_left = np.zeros(left.shape[0], dtype=bool)
    swapped_right = np.zeros(right.shape[0], dtype=bool)
    tracks = {"left": None, "right": None}
    for index in range(left.shape[0]):
        have = {}
        if observed_left[index]:
            have["left"] = left[index, 0].copy()
        if observed_right[index]:
            have["right"] = right[index, 0].copy()
        if not have:
            continue
        current = {
            side: _track_wrist(tracks[side], index, params.track_memory_frames)
            for side in ("left", "right")
        }
        if len(have) == 2 and current["left"] is not None and current["right"] is not None:
            keep = (
                float(np.linalg.norm(have["left"] - current["left"]))
                + float(np.linalg.norm(have["right"] - current["right"]))
            )
            crossed = (
                float(np.linalg.norm(have["left"] - current["right"]))
                + float(np.linalg.norm(have["right"] - current["left"]))
            )
            if crossed + params.lr_margin_m < keep:
                _swap_frame(left, right, left_conf, right_conf, index)
                swapped_left[index] = True
                swapped_right[index] = True
                have["left"], have["right"] = have["right"], have["left"]
        elif len(have) == 1:
            side = next(iter(have))
            other = "right" if side == "left" else "left"
            wrist = have[side]
            same = current[side]
            opposite = current[other]
            distance_same = float("inf") if same is None else float(np.linalg.norm(wrist - same))
            distance_other = float("inf") if opposite is None else float(np.linalg.norm(wrist - opposite))
            if (
                distance_other + params.lr_margin_m < distance_same
                and distance_other <= params.lr_max_match_m
            ):
                _swap_frame(left, right, left_conf, right_conf, index)
                swapped_left[index] = True
                swapped_right[index] = True
                side = other
                have = {side: wrist}
        for side in ("left", "right"):
            if side in have:
                tracks[side] = (have[side], index)
    return left, right, left_conf, right_conf, swapped_left, swapped_right


def _alpha(cutoff, dt):
    cutoff = np.maximum(np.asarray(cutoff, dtype=np.float64), 1e-6)
    dt = np.maximum(np.asarray(dt, dtype=np.float64), 1e-6)
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


def smooth_one_euro(values, valid, timestamps, params):
    """对 ``(T, D)`` 的每个通道做 One Euro。``valid`` 为假的地方不更新，输出保持 NaN。"""
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    count, width = values.shape
    out = np.full((count, width), np.nan, dtype=np.float64)
    filtered = np.zeros(width, dtype=np.float64)
    speed = np.zeros(width, dtype=np.float64)
    seen = np.full(width, np.nan, dtype=np.float64)
    ready = np.zeros(width, dtype=bool)
    for index in range(count):
        active = valid[index] & np.isfinite(values[index])
        if not np.any(active):
            continue
        fresh = active & ~ready
        if np.any(fresh):
            filtered[fresh] = values[index, fresh]
            speed[fresh] = 0.0
            out[index, fresh] = values[index, fresh]
            seen[fresh] = timestamps[index]
            ready[fresh] = True
        keep = active & ready & np.isfinite(seen) & (seen < timestamps[index] - 1e-12)
        # fresh 刚刚写入 seen，上面的比较会把它们排除。再补一次「上一帧就存在」的通道。
        keep = active & ready & (seen < timestamps[index] - 1e-12)
        if not np.any(keep):
            seen[active] = timestamps[index]
            continue
        dt = timestamps[index] - seen[keep]
        derivative = (values[index, keep] - filtered[keep]) / dt
        alpha_d = _alpha(params.d_cutoff, dt)
        speed_hat = alpha_d * derivative + (1.0 - alpha_d) * speed[keep]
        cutoff = params.min_cutoff + params.beta * np.abs(speed_hat)
        alpha = _alpha(cutoff, dt)
        filtered_hat = alpha * values[index, keep] + (1.0 - alpha) * filtered[keep]
        filtered[keep] = filtered_hat
        speed[keep] = speed_hat
        out[index, keep] = filtered_hat
        seen[active] = timestamps[index]
    return out


def smooth_kalman(values, valid, timestamps, params):
    """常速度卡尔曼，每个通道独立。缺测帧不输出预测值。"""
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    count, width = values.shape
    out = np.full((count, width), np.nan, dtype=np.float64)
    position = np.zeros(width, dtype=np.float64)
    velocity = np.zeros(width, dtype=np.float64)
    p00 = np.full(width, params.kalman_meas_std ** 2, dtype=np.float64)
    p01 = np.zeros(width, dtype=np.float64)
    p11 = np.full(width, params.kalman_accel_std ** 2, dtype=np.float64)
    seen = np.full(width, np.nan, dtype=np.float64)
    ready = np.zeros(width, dtype=bool)
    meas_var = float(params.kalman_meas_std) ** 2
    accel_var = float(params.kalman_accel_std) ** 2
    for index in range(count):
        active = valid[index] & np.isfinite(values[index])
        if not np.any(active):
            continue
        fresh = active & ~ready
        if np.any(fresh):
            position[fresh] = values[index, fresh]
            velocity[fresh] = 0.0
            p00[fresh] = meas_var
            p01[fresh] = 0.0
            p11[fresh] = accel_var
            out[index, fresh] = values[index, fresh]
            ready[fresh] = True
        keep = active & ready & np.isfinite(seen) & (seen < timestamps[index] - 1e-12)
        if np.any(keep):
            dt = np.maximum(timestamps[index] - seen[keep], 1e-6)
            dt2 = dt * dt
            dt3 = dt2 * dt
            dt4 = dt2 * dt2
            pos = position[keep] + dt * velocity[keep]
            vel = velocity[keep]
            # P = F P F^T + Q，Q 来自离散白加速度。
            q00 = accel_var * dt4 / 4.0
            q01 = accel_var * dt3 / 2.0
            q11 = accel_var * dt2
            pred00 = p00[keep] + dt * (2.0 * p01[keep] + dt * p11[keep]) + q00
            pred01 = p01[keep] + dt * p11[keep] + q01
            pred11 = p11[keep] + q11
            innovation = values[index, keep] - pos
            variance = pred00 + meas_var
            gain_pos = pred00 / variance
            gain_vel = pred01 / variance
            position[keep] = pos + gain_pos * innovation
            velocity[keep] = vel + gain_vel * innovation
            p00[keep] = np.maximum((1.0 - gain_pos) * pred00, 1e-12)
            p01[keep] = (1.0 - gain_pos) * pred01
            p11[keep] = np.maximum(pred11 - gain_vel * pred01, 1e-12)
            out[index, keep] = position[keep]
        seen[active] = timestamps[index]
    return out


def _channel_valid(joints, observed):
    finite = np.isfinite(joints).all(axis=2)
    return finite & observed.reshape(-1, 1)


def _smooth_joints(joints, observed, timestamps, params):
    count = joints.shape[0]
    flat = joints.reshape(count, JOINTS * 3)
    joint_valid = _channel_valid(joints, observed)
    valid = np.repeat(joint_valid, 3, axis=1)
    if params.smooth == "one_euro":
        smoothed = smooth_one_euro(flat, valid, timestamps, params)
    elif params.smooth == "kalman":
        smoothed = smooth_kalman(flat, valid, timestamps, params)
    else:
        raise ValueError("未知平滑 %s" % params.smooth)
    out = np.full_like(joints, np.nan)
    reshaped = smoothed.reshape(count, JOINTS, 3)
    # 三个坐标一起才算这个关节滤完了。有一个通道没更新就整点留空，避免半个点。
    complete = np.isfinite(reshaped).all(axis=2)
    use = complete & joint_valid
    out[use] = reshaped[use]
    return out


def fill_gaps(joints, observed, max_gap):
    """在观测之间线性补上不超过 ``max_gap`` 的缺测。开头和结尾不外推。"""
    out = np.array(joints, copy=True)
    filled = np.zeros(out.shape[0], dtype=bool)
    if max_gap <= 0:
        return out, filled
    count = out.shape[0]
    index = 0
    while index < count:
        if observed[index]:
            index += 1
            continue
        start = index
        while index < count and not observed[index]:
            index += 1
        end = index
        gap = end - start
        left = start - 1
        if left < 0 or end >= count or gap > int(max_gap):
            continue
        for joint in range(JOINTS):
            before = out[left, joint]
            after = out[end, joint]
            if not np.isfinite(before).all() or not np.isfinite(after).all():
                continue
            for step, frame in enumerate(range(start, end)):
                weight = float(step + 1) / float(gap + 1)
                out[frame, joint] = (1.0 - weight) * before + weight * after
        for frame in range(start, end):
            if np.isfinite(out[frame, 0]).all():
                filled[frame] = True
    return out, filled


def bone_lengths(joints):
    """每段骨头的长度。下标 0 没有骨头，记为 NaN。``joints`` 是 ``(21, 3)``。"""
    lengths = np.full(JOINTS, np.nan, dtype=np.float64)
    for child in range(1, JOINTS):
        parent = int(MEDIAPIPE_PARENTS[child])
        start = joints[parent]
        end = joints[child]
        if np.isfinite(start).all() and np.isfinite(end).all():
            lengths[child] = float(np.linalg.norm(end - start))
    return lengths


def robust_median_bone_lengths(joints, valid):
    """高置信帧上每段骨头的中位数。离群点用 MAD 丢掉。没有任何观测时返回 None。"""
    valid = np.asarray(valid, dtype=bool)
    if not np.any(valid):
        return None
    samples = []
    for index in np.flatnonzero(valid):
        lengths = bone_lengths(joints[index])
        if np.isfinite(lengths[1:]).any():
            samples.append(lengths)
    if not samples:
        return None
    stack = np.stack(samples, axis=0)
    target = np.full(JOINTS, np.nan, dtype=np.float64)
    for joint in range(1, JOINTS):
        column = stack[:, joint]
        column = column[np.isfinite(column)]
        if column.size == 0:
            continue
        center = float(np.median(column))
        deviation = np.abs(column - center)
        mad = float(np.median(deviation))
        if mad > 1e-8:
            keep = deviation <= (3.0 * 1.4826 * mad)
            if int(np.count_nonzero(keep)) >= 1:
                column = column[keep]
        target[joint] = float(np.median(column))
    return target


def apply_fixed_bone_lengths(joints, lengths):
    """手腕留在原地，其余关节沿原来的方向放到固定骨长上。"""
    original = np.asarray(joints, dtype=np.float64)
    out = original.copy()
    if lengths is None or not np.isfinite(out[0]).all():
        return out
    for child in range(1, JOINTS):
        parent = int(MEDIAPIPE_PARENTS[child])
        length = float(lengths[child])
        if not np.isfinite(length):
            continue
        if not np.isfinite(out[parent]).all() or not np.isfinite(original[parent]).all():
            continue
        if not np.isfinite(original[child]).all():
            continue
        direction = original[child] - original[parent]
        norm = float(np.linalg.norm(direction))
        if norm < 1e-8:
            out[child] = out[parent]
        else:
            out[child] = out[parent] + direction / norm * length
    return out


def fit_shape_coefficients(target_lengths, length_fn, n_coeff, eps=1e-3, ridge=1e-4):
    """用骨长对 betas 的数值雅可比做一次线性最小二乘。

    ``length_fn(betas)`` 要返回 21 个数，下标 0 可以是 NaN。这是给真正的 MANO
    前向准备的接口。没有前向函数时不要调用。
    """
    if n_coeff < 1:
        raise ValueError("n_coeff 必须为正")
    origin = np.zeros(n_coeff, dtype=np.float64)
    base = np.asarray(length_fn(origin), dtype=np.float64).reshape(-1)
    if base.shape != (JOINTS,):
        raise ValueError("mano_length_fn 必须返回 21 个骨长")
    jacobian = np.zeros((JOINTS, n_coeff), dtype=np.float64)
    for index in range(n_coeff):
        step = np.zeros(n_coeff, dtype=np.float64)
        step[index] = float(eps)
        shifted = np.asarray(length_fn(step), dtype=np.float64).reshape(-1)
        jacobian[:, index] = (shifted - base) / float(eps)
    target = np.asarray(target_lengths, dtype=np.float64).reshape(-1)
    mask = np.isfinite(target) & np.isfinite(base)
    if int(np.count_nonzero(mask)) < 1:
        return origin
    design = jacobian[mask]
    residual = target[mask] - base[mask]
    system = design.T @ design + float(ridge) * np.eye(n_coeff)
    return np.linalg.solve(system, design.T @ residual)


def _temporal(joints, observed, timestamps, params):
    if params.smooth == "none":
        base = np.array(joints, copy=True)
    else:
        base = _smooth_joints(joints, observed, timestamps, params)
    if params.gap_fill:
        filled_joints, filled = fill_gaps(base, observed, params.max_gap)
    else:
        filled_joints = base
        filled = np.zeros(joints.shape[0], dtype=bool)
    if params.smooth != "none":
        # 长缺口保留原来的低置信检测，不把它们抹成 NaN。短缺口已经是插值。
        restore = (~observed) & (~filled)
        filled_joints[restore] = joints[restore]
    return filled_joints, filled


def _apply_shape(joints, observed, params):
    target = None
    betas = None
    if not params.fixed_shape:
        return joints, target, betas
    target = robust_median_bone_lengths(joints, observed)
    if target is None:
        return joints, None, None
    if params.mano_length_fn is not None:
        betas = fit_shape_coefficients(target, params.mano_length_fn, params.n_betas)
    out = np.array(joints, copy=True)
    for index in range(out.shape[0]):
        if np.isfinite(out[index, 0]).all():
            out[index] = apply_fixed_bone_lengths(out[index], target)
    return out, target, betas


def refine_hands(
    left_joints, right_joints, left_confidence, right_confidence,
    timestamps, params, camera_poses=None,
):
    """精修左右手。返回相机系（或调用方传入的坐标系）里的关节、置信度、补洞标记。

    ``camera_poses`` 是每帧的 ``T_world_cam``。传入之后，轨迹一致性、平滑、补洞
    和骨长都在世界系里计算，返回值再变回原来的相机系。
    """
    left_in = np.asarray(left_joints, dtype=np.float64)
    if left_in.ndim != 3:
        raise ValueError("left_joints 必须是 (T, 21, 3)")
    count = int(left_in.shape[0])
    left = _as_joints(left_joints, count)
    right = _as_joints(right_joints, count)
    left_conf = _as_confidence(left_confidence, count)
    right_conf = _as_confidence(right_confidence, count)
    times = _as_timestamps(timestamps, count, params)
    poses = _as_poses(camera_poses, count)
    if poses is not None:
        left = _map_frame(left, poses, to_world=True)
        right = _map_frame(right, poses, to_world=True)
    observed_left = _frame_observed(left, left_conf, params)
    observed_right = _frame_observed(right, right_conf, params)
    swapped_left = np.zeros(count, dtype=bool)
    swapped_right = np.zeros(count, dtype=bool)
    if params.lr_consistency:
        left, right, left_conf, right_conf, swapped_left, swapped_right = enforce_left_right(
            left, right, left_conf, right_conf, observed_left, observed_right, params,
        )
        observed_left = _frame_observed(left, left_conf, params)
        observed_right = _frame_observed(right, right_conf, params)
    left, filled_left = _temporal(left, observed_left, times, params)
    right, filled_right = _temporal(right, observed_right, times, params)
    left, lengths_left, betas_left = _apply_shape(left, observed_left, params)
    right, lengths_right, betas_right = _apply_shape(right, observed_right, params)
    left_conf = left_conf.copy()
    right_conf = right_conf.copy()
    left_conf[filled_left] = float(params.filled_confidence)
    right_conf[filled_right] = float(params.filled_confidence)
    if poses is not None:
        left = _map_frame(left, poses, to_world=False)
        right = _map_frame(right, poses, to_world=False)
    return {
        "left": _pack_result(left, left_conf, filled_left, swapped_left),
        "right": _pack_result(right, right_conf, filled_right, swapped_right),
        "shape_bone_lengths": {"left": lengths_left, "right": lengths_right},
        "betas": {"left": betas_left, "right": betas_right},
        "coordinate": "world_then_camera" if poses is not None else "input",
    }


def _pack_result(joints, confidence, filled, swapped):
    return {
        "joints": joints,
        "confidence": confidence,
        "filled": filled,
        "swapped": swapped,
        "wrist_pose": wrist_poses_from_joints(joints),
    }


def _joints_from_hand(hand):
    if not isinstance(hand, dict) or hand.get("joints_cam") is None:
        return None
    value = hand["joints_cam"]
    if isinstance(value, np.ndarray) and value.shape == (JOINTS, 3):
        return np.asarray(value, dtype=np.float64)
    array = np.full((JOINTS, 3), np.nan, dtype=np.float64)
    if len(value) != JOINTS:
        return None
    for index, point in enumerate(value):
        if point is None or any(coord is None for coord in point):
            continue
        array[index] = [float(point[0]), float(point[1]), float(point[2])]
    return array


def _confidence_from_hand(hand):
    if not isinstance(hand, dict) or hand.get("confidence") is None:
        return np.full(JOINTS, np.nan, dtype=np.float64)
    value = hand["confidence"]
    if isinstance(value, (int, float)):
        return np.full(JOINTS, float(value), dtype=np.float64)
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 1:
        return np.full(JOINTS, float(array[0]), dtype=np.float64)
    out = np.full(JOINTS, np.nan, dtype=np.float64)
    if array.size != JOINTS:
        return out
    finite = np.isfinite(array)
    out[finite] = array[finite]
    return out


def refine_prediction_sequence(frames, params, timestamps=None, camera_poses=None):
    """``predict`` 返回的一串帧。几何被平滑、补洞或重摆之后，2D 关键点清掉。

    旧的 2D 点和修过的 3D 对不上。下游要用的话，按相机内参重新投影。
    只做左右标签修正、没有改几何时，2D 点跟着标签对调，坐标仍是原来的。
    """
    count = len(frames)
    left = np.full((count, JOINTS, 3), np.nan, dtype=np.float64)
    right = np.full((count, JOINTS, 3), np.nan, dtype=np.float64)
    left_conf = np.full((count, JOINTS), np.nan, dtype=np.float64)
    right_conf = np.full((count, JOINTS), np.nan, dtype=np.float64)
    for index, frame in enumerate(frames):
        for side, joints, confidence in (
            ("left", left, left_conf),
            ("right", right, right_conf),
        ):
            hand = frame.get(side) if isinstance(frame, dict) else None
            parsed = _joints_from_hand(hand)
            if parsed is not None:
                joints[index] = parsed
            confidence[index] = _confidence_from_hand(hand)
    refined = refine_hands(
        left, right, left_conf, right_conf, timestamps, params, camera_poses=camera_poses,
    )
    geometry_changed = params.smooth != "none" or params.gap_fill or params.fixed_shape
    output = []
    for index, frame in enumerate(frames):
        source = dict(frame) if isinstance(frame, dict) else {}
        if refined["left"]["swapped"][index]:
            source["left"], source["right"] = source.get("right"), source.get("left")
        packed = {}
        for side in ("left", "right"):
            previous = source.get(side) if isinstance(source.get(side), dict) else {}
            hand = dict(previous or {})
            joints = refined[side]["joints"][index]
            if not np.isfinite(joints).any():
                hand["joints_cam"] = None
            else:
                hand["joints_cam"] = joints
            confidence = refined[side]["confidence"][index]
            hand["confidence"] = confidence
            hand["filled"] = bool(refined[side]["filled"][index])
            hand["label_swapped"] = bool(refined[side]["swapped"][index])
            if geometry_changed:
                hand["keypoints_2d"] = None
            packed[side] = hand
        for key, value in source.items():
            if key not in ("left", "right"):
                packed[key] = value
        output.append(packed)
    output_meta = {
        "shape_bone_lengths": refined["shape_bone_lengths"],
        "betas": refined["betas"],
        "coordinate": refined["coordinate"],
        "params": params.to_dict(),
    }
    return output, output_meta


def accel_samples(points, timestamps, mask):
    """相邻三帧都在 ``mask`` 里时，收集加速度向量的模长。

    ``points`` 是 ``(T, 3)`` 或 ``(T, J, 3)``。加速度用前后两段速度的差，
    再除以两段时间的平均，所以帧间隔不均匀也能用。单位是米/秒²，如果点的单位是米。
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 2:
        pts = pts[:, None, :]
    times = np.asarray(timestamps, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    samples = []
    for index in range(1, pts.shape[0] - 1):
        if not (mask[index - 1] and mask[index] and mask[index + 1]):
            continue
        dt0 = float(times[index] - times[index - 1])
        dt1 = float(times[index + 1] - times[index])
        if dt0 <= 0.0 or dt1 <= 0.0:
            continue
        before = (pts[index] - pts[index - 1]) / dt0
        after = (pts[index + 1] - pts[index]) / dt1
        accel = (after - before) / (0.5 * (dt0 + dt1))
        if not np.isfinite(accel).all():
            continue
        samples.append(np.linalg.norm(accel, axis=-1))
    if not samples:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(samples)


def mean_accel(points, timestamps, mask):
    samples = accel_samples(points, timestamps, mask)
    if samples.size == 0:
        return float("nan")
    return float(np.mean(samples))


def _format_percent(value):
    if value is None or not np.isfinite(value):
        return "n/a"
    return "%.1f%%" % (100.0 * float(value))


def _format_fixed(value, digits):
    if value is None or not np.isfinite(value):
        return "n/a"
    return ("%." + str(int(digits)) + "f") % float(value)


def format_ablation_table(rows):
    """把消融结果写成 Markdown 表。缺测写成 n/a，不填 0。"""
    header = (
        "| 方法 | 检出率 | MPJPE 均值 (cm) | MPJPE 中位 (cm) | "
        "手腕无尺度 中位/均值 (cm) | 手腕对齐后 中位/均值 (cm) | "
        "抖动 (m/s²) | 左右交换率 | 高置信检出率 |"
    )
    rule = "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    lines = [header, rule]
    for row in rows:
        lines.append("| %s | %s | %s | %s | %s / %s | %s / %s | %s | %s | %s |" % (
            row["name"],
            _format_percent(row.get("detection_rate")),
            _format_fixed(row.get("mpjpe_cm"), 2),
            _format_fixed(row.get("mpjpe_median_cm"), 2),
            _format_fixed(row.get("wrist_cm_median"), 2),
            _format_fixed(row.get("wrist_cm_mean"), 2),
            _format_fixed(row.get("wrist_scaled_cm_median"), 2),
            _format_fixed(row.get("wrist_scaled_cm_mean"), 2),
            _format_fixed(row.get("jitter_mps2"), 3),
            _format_percent(row.get("swap_rate")),
            _format_percent(row.get("confident_detection_rate")),
        ))
    return "\n".join(lines)


def metrics_row(name, report, jitter, confident_report):
    """把 ``evaluate_hand_frames`` 的米换成厘米，并附上抖动。"""

    def centimeters(value):
        if value is None or not np.isfinite(value):
            return float("nan")
        return 100.0 * float(value)

    return {
        "name": name,
        "detection_rate": float(report["detection_rate"]),
        "mpjpe_cm": centimeters(report["root_relative_m_mean"]),
        "mpjpe_median_cm": centimeters(report["root_relative_m_median"]),
        "wrist_cm_median": centimeters(report["wrist_error_m_median"]),
        "wrist_cm_mean": centimeters(report["wrist_error_m_mean"]),
        "wrist_scaled_cm_median": centimeters(report["wrist_error_scaled_m_median"]),
        "wrist_scaled_cm_mean": centimeters(report["wrist_error_scaled_m_mean"]),
        "jitter_mps2": float(jitter),
        "swap_rate": float(report["swap_rate"]),
        "confident_detection_rate": float(confident_report["detection_rate"]),
        "gt_visible": int(report["gt_visible"]),
        "matched": int(report["matched"]),
        "exclude_joints": tuple(report.get("exclude_joints") or ()),
    }
