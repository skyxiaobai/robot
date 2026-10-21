# -*- coding: utf-8 -*-
"""对照 HOT3D 真值，报告手腕误差和轨迹误差。

相机系误差：预测手腕（相机系 episode）相对 ``inv(T) @ gt_world``。
世界系误差：预测手腕（已经乘过位姿的 episode）相对 gt_world。
``T`` 用的是这次导出的位姿来源。来源是头戴真值时，两个误差应接近。
来源是 ORB-SLAM3 时，世界系误差还包含这条轨迹和真值的差。

``trajectory_accuracy`` 在 SLAM 轨迹和另一份真值轨迹都在、并且对上至少 3 个
不同时间戳时，给出 Umeyama SE3 / Sim3 的 ATE 和相对位姿误差。调用方可以把
这份结构放进 ``slam_status.accuracy``。对不上、来源是头戴真值、或只是一个
手写数字时，accuracy 保持 null。
"""
import json
import math
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import numpy as np

from headcam.hand_pose import load_tum_trajectory


def _wrist(frame):
    if not isinstance(frame, list) or not frame:
        return None
    point = frame[0]
    if not isinstance(point, (list, tuple)) or len(point) < 3 or any(coord is None for coord in point[:3]):
        return None
    array = np.asarray(point[:3], dtype=float)
    if not np.isfinite(array).all():
        return None
    return array


def _stats_cm(errors_m):
    if not errors_m:
        return {"n": 0}
    values = np.asarray(errors_m, dtype=float) * 100.0
    return {
        "n": int(values.size),
        "median_cm": float(np.median(values)),
        "p90_cm": float(np.percentile(values, 90)),
        "mean_cm": float(np.mean(values)),
    }


def _umeyama(source, target, with_scale):
    """把 source 对齐到 target。返回 scale, R, t，使 ``scale * R @ p + t`` 贴近 target。

    ``with_scale=False`` 时尺度固定为 1（SE3）。为 True 时尺度自由（Sim3）。
    点几乎重合、方差为 0 时尺度没有定义，返回 None。
    """
    src = np.asarray(source, dtype=float)
    dst = np.asarray(target, dtype=float)
    count = src.shape[0]
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    # cov 的 SVD：dst 在左，source 在右，R 把 source 转到 target。
    covariance = (dst_c.T @ src_c) / count
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1.0
    rotation = u @ sign @ vt
    if with_scale:
        variance = float(np.mean(np.sum(src_c ** 2, axis=1)))
        if variance <= 1e-18:
            return None
        scale = float(np.trace(np.diag(singular) @ sign) / variance)
    else:
        scale = 1.0
    translation = dst_mean - scale * rotation @ src_mean
    return scale, rotation, translation


def _umeyama_fixed_scale(source, target):
    """source 对齐到 target。尺度固定为 1。返回 R, t。"""
    _scale, rotation, translation = _umeyama(source, target, with_scale=False)
    return rotation, translation


def wrist_errors(camera_episode, world_episode, hands_gt, poses):
    """poses 是和帧对齐的 4×4，缺的帧是 None。"""
    gt_frames = hands_gt["frames"]
    camera_errors = []
    world_measured = []
    world_filled = []
    for side in ("left", "right"):
        camera_hand = camera_episode["hands"][side]
        world_hand = world_episode["hands"][side]
        filled_flags = world_hand.get("filled") or []
        for index, gt_frame in enumerate(gt_frames):
            gt = gt_frame.get(side) if isinstance(gt_frame, dict) else None
            if gt is None:
                continue
            gt_world = np.asarray(gt, dtype=float)[0]
            pose = poses[index] if index < len(poses) else None
            pred_cam = _wrist(camera_hand["joints"][index]) if index < len(camera_hand["joints"]) else None
            if pred_cam is not None and pose is not None:
                rotation = pose[:3, :3]
                translation = pose[:3, 3]
                gt_cam = rotation.T @ (gt_world - translation)
                camera_errors.append(float(np.linalg.norm(pred_cam - gt_cam)))
            pred_world = _wrist(world_hand["joints"][index]) if index < len(world_hand["joints"]) else None
            if pred_world is None:
                continue
            error = float(np.linalg.norm(pred_world - gt_world))
            filled = bool(filled_flags[index]) if index < len(filled_flags) else False
            if filled:
                world_filled.append(error)
            else:
                world_measured.append(error)
    declared = hands_gt.get("wrist_definition") if isinstance(hands_gt, dict) else None
    return {
        "camera_frame_wrist_m_vs_hot3d_gt": _stats_cm(camera_errors),
        "world_frame_wrist_m_vs_hot3d_gt_measured": _stats_cm(world_measured),
        "world_frame_wrist_m_vs_hot3d_gt_filled": _stats_cm(world_filled),
        "measured_against": declared or (
            "HOT3D MANO 手腕，OpenPose 第 0 点，米。不是头戴设备精度，也不是 accuracy 字段。"
        ),
    }


def _load_scaled_tum(path, scale):
    times, poses = load_tum_trajectory(path)
    times = np.asarray(times, dtype=float) * float(scale)
    return times, poses


def _unique_trajectory(times, poses):
    """同一时间戳只留第一行。返回时间、位姿、丢掉的重复行数。"""
    unique_t = []
    unique_p = []
    seen = set()
    for index, stamp in enumerate(times):
        key = round(float(stamp), 9)
        if key in seen:
            continue
        seen.add(key)
        unique_t.append(times[index])
        unique_p.append(poses[index])
    duplicate_rows = int(len(times) - len(unique_t))
    return np.asarray(unique_t, dtype=float), unique_p, duplicate_rows


def _pair_by_time(est_t, est_poses, gt_t, gt_poses, max_dt_s):
    """估计轨迹的每一行找最近的真值。超过 max_dt_s 的不配。"""
    paired_est = []
    paired_gt = []
    paired_t = []
    for index, stamp in enumerate(est_t):
        nearest = int(np.argmin(np.abs(gt_t - stamp)))
        if abs(float(gt_t[nearest] - stamp)) > float(max_dt_s):
            continue
        paired_est.append(est_poses[index])
        paired_gt.append(gt_poses[nearest])
        paired_t.append(float(stamp))
    return paired_t, paired_est, paired_gt


def _prepare_pair(estimate_tum, gt_tum, estimate_scale, gt_scale, max_dt_s):
    est_t, est_poses = _load_scaled_tum(estimate_tum, estimate_scale)
    gt_t, gt_poses = _load_scaled_tum(gt_tum, gt_scale)
    est_t, est_poses, duplicate_rows = _unique_trajectory(est_t, est_poses)
    paired_t, paired_est, paired_gt = _pair_by_time(est_t, est_poses, gt_t, gt_poses, max_dt_s)
    return {
        "estimate_rows": int(duplicate_rows + len(est_t)),
        "unique_timestamps": int(len(est_t)),
        "duplicate_rows_ignored": duplicate_rows,
        "gt_frames": int(len(gt_t)),
        "paired_t": paired_t,
        "paired_est": paired_est,
        "paired_gt": paired_gt,
    }


def _rotation_angle_deg(rotation):
    cosine = (float(np.trace(rotation)) - 1.0) / 2.0
    cosine = max(-1.0, min(1.0, cosine))
    return float(np.degrees(np.arccos(cosine)))


def trajectory_ate(estimate_tum, gt_tum, estimate_scale=1.0, gt_scale=1.0, alignment="se3", max_dt_s=0.05):
    """两条轨迹的相机中心。时间用最近邻。

    ``alignment="se3"`` 尺度固定为 1。``alignment="sim3"`` 尺度自由。
    估计轨迹里的重复时间戳只留第一行，不用重复行凑数量。
    对上的不同时间戳少于 3 时不写 rmse。
    """
    if alignment not in ("se3", "sim3"):
        raise ValueError("alignment 只能是 se3 或 sim3，收到 %s" % alignment)
    prepared = _prepare_pair(estimate_tum, gt_tum, estimate_scale, gt_scale, max_dt_s)
    paired_est = [pose[:3, 3] for pose in prepared["paired_est"]]
    paired_gt = [pose[:3, 3] for pose in prepared["paired_gt"]]
    if len(paired_est) < 3:
        return {
            "n": len(paired_est),
            "estimate_rows": prepared["estimate_rows"],
            "unique_timestamps": prepared["unique_timestamps"],
            "duplicate_rows_ignored": prepared["duplicate_rows_ignored"],
            "note": "不同时间戳对上的帧少于 3，不算 ATE。重复行没有拿来凑数。",
        }
    with_scale = alignment == "sim3"
    fitted = _umeyama(paired_est, paired_gt, with_scale=with_scale)
    if fitted is None:
        return {
            "n": len(paired_est),
            "estimate_rows": prepared["estimate_rows"],
            "unique_timestamps": prepared["unique_timestamps"],
            "duplicate_rows_ignored": prepared["duplicate_rows_ignored"],
            "note": "估计轨迹的位置没有展开，尺度没有定义，不算 ATE。",
        }
    scale, rotation, translation = fitted
    aligned = (scale * (rotation @ np.asarray(paired_est).T)).T + translation
    errors = np.linalg.norm(aligned - np.asarray(paired_gt), axis=1)
    report = {
        "n": int(errors.size),
        "estimate_frames": prepared["unique_timestamps"],
        "gt_frames": prepared["gt_frames"],
        "rmse_m": float(np.sqrt(np.mean(errors ** 2))),
        "median_m": float(np.median(errors)),
        "max_m": float(np.max(errors)),
        "alignment": "umeyama_sim3" if with_scale else "umeyama_se3",
        "measured_against": "相机中心，最近邻配对。短片段或长序列都只是这次轨迹和头戴定位的差，不是另一次实验的精度。",
    }
    if with_scale:
        report["scale"] = "free"
        report["similarity_scale"] = float(scale)
        report["scale_meaning"] = "similarity_scale 乘在估计轨迹上，使它贴近真值。小于 1 表示估计的位移比真值大。"
    else:
        report["scale"] = "fixed_1"
    return report


def trajectory_rpe(estimate_tum, gt_tum, estimate_scale=1.0, gt_scale=1.0, delta_s=1.0, max_dt_s=0.05, max_delta_error_s=0.05):
    """相对位姿误差。不做全局对齐。

    先按时间把估计和真值配上。再在配上的估计时间里，为每一帧找大约晚 ``delta_s`` 的另一帧。
    误差是两段相对运动的差：``inv(gt_i^{-1} gt_j) @ (est_i^{-1} est_j)``。
    对上的不同时间戳少于 3，或者凑不出间隔接近 delta 的配对时，不写 rmse。
    """
    prepared = _prepare_pair(estimate_tum, gt_tum, estimate_scale, gt_scale, max_dt_s)
    times = prepared["paired_t"]
    if len(times) < 3:
        return {
            "n_poses": len(times),
            "duplicate_rows_ignored": prepared["duplicate_rows_ignored"],
            "note": "不同时间戳对上的帧少于 3，不算 RPE。",
        }
    trans_errors = []
    rot_errors = []
    stamps = np.asarray(times, dtype=float)
    for index, stamp in enumerate(stamps):
        later = np.where(stamps > stamp)[0]
        if later.size == 0:
            continue
        gaps = np.abs(stamps[later] - (stamp + float(delta_s)))
        choice = int(later[int(np.argmin(gaps))])
        if abs(float(stamps[choice] - stamp) - float(delta_s)) > float(max_delta_error_s):
            continue
        est_i = prepared["paired_est"][index]
        est_j = prepared["paired_est"][choice]
        gt_i = prepared["paired_gt"][index]
        gt_j = prepared["paired_gt"][choice]
        est_rel = np.linalg.inv(est_i) @ est_j
        gt_rel = np.linalg.inv(gt_i) @ gt_j
        error = np.linalg.inv(gt_rel) @ est_rel
        trans_errors.append(float(np.linalg.norm(error[:3, 3])))
        rot_errors.append(_rotation_angle_deg(error[:3, :3]))
    if not trans_errors:
        return {
            "n_poses": len(times),
            "n_pairs": 0,
            "delta_s": float(delta_s),
            "duplicate_rows_ignored": prepared["duplicate_rows_ignored"],
            "note": "没有间隔接近 delta 的配对，不算 RPE。",
        }
    trans = np.asarray(trans_errors, dtype=float)
    rot = np.asarray(rot_errors, dtype=float)
    return {
        "n_poses": len(times),
        "n_pairs": int(trans.size),
        "delta_s": float(delta_s),
        "trans_rmse_m": float(np.sqrt(np.mean(trans ** 2))),
        "trans_median_m": float(np.median(trans)),
        "rot_rmse_deg": float(np.sqrt(np.mean(rot ** 2))),
        "rot_median_deg": float(np.median(rot)),
        "alignment": "none_relative",
        "duplicate_rows_ignored": prepared["duplicate_rows_ignored"],
        "note": "相对位姿误差，没有先做全局 SE3/Sim3 对齐。平移是相对运动的差，旋转是那个差的角度。",
    }


def _finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def _vector(value, length):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return False
    return all(_finite_number(item) for item in value)


def _ate_block(block, alignment, scale):
    if not isinstance(block, dict):
        return False
    if block.get("alignment") != alignment or block.get("scale") != scale:
        return False
    return all(_finite_number(block.get(key)) for key in ("n", "rmse_m", "median_m", "max_m")) and int(block["n"]) >= 3


def trajectory_accuracy(
    estimate_tum, gt_tum, estimate_scale=1.0, gt_scale=1.0, max_dt_s=0.05,
    rpe_deltas=(1.0, 5.0), max_delta_error_s=0.05,
    estimate_pose_source=None, reference_pose_source=None,
):
    """SLAM 轨迹相对另一份真值的 ATE 和 RPE。对不上 3 帧时返回 None。

    SE3 的旋转和平移是 Umeyama、尺度固定为 1：``p_ref ~= R @ p_est + t``。
    只保存这次对齐，不把真值轨迹抄进结果。
    """
    if not isinstance(estimate_pose_source, str) or not estimate_pose_source:
        return None
    if estimate_pose_source.startswith("hot3d_gt"):
        return None
    if not isinstance(reference_pose_source, str) or not reference_pose_source:
        return None
    if reference_pose_source == estimate_pose_source:
        return None
    se3 = trajectory_ate(
        estimate_tum, gt_tum, estimate_scale=estimate_scale, gt_scale=gt_scale,
        alignment="se3", max_dt_s=max_dt_s,
    )
    sim3 = trajectory_ate(
        estimate_tum, gt_tum, estimate_scale=estimate_scale, gt_scale=gt_scale,
        alignment="sim3", max_dt_s=max_dt_s,
    )
    if "rmse_m" not in se3 or "rmse_m" not in sim3:
        return None
    prepared = _prepare_pair(estimate_tum, gt_tum, estimate_scale, gt_scale, max_dt_s)
    fitted = _umeyama(
        [pose[:3, 3] for pose in prepared["paired_est"]],
        [pose[:3, 3] for pose in prepared["paired_gt"]],
        with_scale=False,
    )
    if fitted is None:
        return None
    _scale, rotation, translation = fitted
    se3_block = {
        "n": se3["n"],
        "rmse_m": se3["rmse_m"],
        "median_m": se3["median_m"],
        "max_m": se3["max_m"],
        "alignment": "umeyama_se3",
        "scale": "fixed_1",
        "rotation_row_major": [float(value) for value in np.asarray(rotation, dtype=float).reshape(-1)],
        "translation": [float(value) for value in np.asarray(translation, dtype=float).reshape(-1)],
        "applied_as": "p_ref = R @ p_est + t，尺度固定为 1。R 按行展开。",
    }
    sim3_block = {
        "n": sim3["n"],
        "rmse_m": sim3["rmse_m"],
        "median_m": sim3["median_m"],
        "max_m": sim3["max_m"],
        "alignment": "umeyama_sim3",
        "scale": "free",
        "similarity_scale": sim3["similarity_scale"],
    }
    rpe_rows = []
    for delta in rpe_deltas:
        row = trajectory_rpe(
            estimate_tum, gt_tum, estimate_scale=estimate_scale, gt_scale=gt_scale,
            delta_s=float(delta), max_dt_s=max_dt_s, max_delta_error_s=max_delta_error_s,
        )
        compact = {
            "delta_s": float(delta),
            "max_delta_error_s": float(max_delta_error_s),
            "alignment": "none_relative",
            "n_poses": row.get("n_poses"),
            "n_pairs": row.get("n_pairs", 0),
        }
        for key in ("trans_rmse_m", "trans_median_m", "rot_rmse_deg", "rot_median_deg"):
            if key in row:
                compact[key] = row[key]
        rpe_rows.append(compact)
    return {
        "kind": "trajectory_error_v1",
        "method": "umeyama",
        "estimate_pose_source": estimate_pose_source,
        "reference_pose_source": reference_pose_source,
        "max_dt_s": float(max_dt_s),
        "ate": {"se3": se3_block, "sim3": sim3_block},
        "rpe": rpe_rows,
        "note": (
            "相机中心，最近邻，超过 max_dt_s 不配。重复时间戳只留第一行。"
            "SE3 尺度固定为 1。Sim3 尺度自由，similarity_scale 乘在估计轨迹上。"
            "RPE 不做这次全局对齐。"
        ),
    }


def is_measured_accuracy(value, pose_source=None):
    """只有 ``trajectory_accuracy`` 这种结构算测过。手写数字和头戴真值来源都不算。"""
    if not isinstance(value, dict):
        return False
    if value.get("kind") != "trajectory_error_v1" or value.get("method") != "umeyama":
        return False
    estimate = value.get("estimate_pose_source")
    reference = value.get("reference_pose_source")
    if not isinstance(estimate, str) or not estimate or estimate.startswith("hot3d_gt"):
        return False
    if isinstance(pose_source, str) and (pose_source.startswith("hot3d_gt") or pose_source != estimate):
        return False
    if not isinstance(reference, str) or not reference or reference == estimate:
        return False
    if not _finite_number(value.get("max_dt_s")) or float(value["max_dt_s"]) <= 0:
        return False
    ate = value.get("ate")
    if not isinstance(ate, dict):
        return False
    se3 = ate.get("se3")
    sim3 = ate.get("sim3")
    if not _ate_block(se3, "umeyama_se3", "fixed_1") or not _ate_block(sim3, "umeyama_sim3", "free"):
        return False
    if not _finite_number(sim3.get("similarity_scale")) or float(sim3["similarity_scale"]) <= 0:
        return False
    if not _vector(se3.get("rotation_row_major"), 9) or not _vector(se3.get("translation"), 3):
        return False
    rpe = value.get("rpe")
    if not isinstance(rpe, list) or not rpe:
        return False
    for row in rpe:
        if not isinstance(row, dict) or row.get("alignment") != "none_relative":
            return False
        if not _finite_number(row.get("delta_s")):
            return False
        if "trans_rmse_m" in row and not _finite_number(row["trans_rmse_m"]):
            return False
    return True


def shared_measured_accuracy(episodes):
    """每份 episode 都有同一份测过的 accuracy 才返回它。缺一份、或来源是头戴真值，返回 None。"""
    chosen = None
    kept = None
    if not episodes:
        return None
    for episode in episodes:
        slam = (episode.get("hand_pose") or {}).get("slam") if isinstance(episode, dict) else None
        if not isinstance(slam, dict):
            return None
        value = slam.get("accuracy")
        if not is_measured_accuracy(value, slam.get("pose_source")):
            return None
        blob = json.dumps(value, sort_keys=True)
        if chosen is None:
            chosen = blob
            kept = value
        elif blob != chosen:
            return None
    return kept


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="HOT3D 手腕误差。不写入 accuracy。")
    parser.add_argument("--camera-episode", required=True)
    parser.add_argument("--world-episode", required=True)
    parser.add_argument("--hands-gt", required=True)
    parser.add_argument("--poses-json", required=True, help="每帧 4×4 或 null 的 JSON 列表")
    parser.add_argument("--slam-tum", default=None)
    parser.add_argument("--gt-tum", default=None)
    parser.add_argument("--slam-time-scale", type=float, default=1.0)
    parser.add_argument("--gt-time-scale", type=float, default=1.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    camera = json.loads(Path(args.camera_episode).read_text(encoding="utf-8"))
    world = json.loads(Path(args.world_episode).read_text(encoding="utf-8"))
    hands = json.loads(Path(args.hands_gt).read_text(encoding="utf-8"))
    poses = []
    for item in json.loads(Path(args.poses_json).read_text(encoding="utf-8")):
        if item is None:
            poses.append(None)
        else:
            poses.append(np.asarray(item, dtype=float))
    report = wrist_errors(camera, world, hands, poses)
    if args.slam_tum and args.gt_tum:
        report["orbslam3_ate_vs_hot3d_gt"] = trajectory_ate(
            args.slam_tum, args.gt_tum, args.slam_time_scale, args.gt_time_scale,
        )
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
