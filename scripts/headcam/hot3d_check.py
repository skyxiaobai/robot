# -*- coding: utf-8 -*-
"""对照 HOT3D 真值，报告手腕误差。不把这些数写进导出报告的 accuracy。

相机系误差：预测手腕（相机系 episode）相对 ``inv(T) @ gt_world``。
世界系误差：预测手腕（已经乘过位姿的 episode）相对 gt_world。
``T`` 用的是这次导出的位姿来源。来源是头戴真值时，两个误差应接近。
来源是 ORB-SLAM3 时，世界系误差还包含这条短轨迹和真值的差。

ATE 只在同时给出 SLAM 轨迹和真值轨迹时计算，尺度固定，并写明比的是哪两条轨迹。
"""
import json
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


def _umeyama_fixed_scale(source, target):
    """source 对齐到 target。尺度固定为 1。返回 R, t。"""
    src = np.asarray(source, dtype=float)
    dst = np.asarray(target, dtype=float)
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    matrix = (src - src_mean).T @ (dst - dst_mean) / src.shape[0]
    u, _s, vt = np.linalg.svd(matrix)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = dst_mean - rotation @ src_mean
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


def trajectory_ate(estimate_tum, gt_tum, estimate_scale=1.0, gt_scale=1.0):
    """两条轨迹的相机位置，SE3 对齐、尺度固定。时间用最近邻，门限 0.05 秒。"""
    est_t, est_p = load_tum_trajectory(estimate_tum)
    gt_t, gt_p = load_tum_trajectory(gt_tum)
    est_t = np.asarray(est_t, dtype=float) * float(estimate_scale)
    gt_t = np.asarray(gt_t, dtype=float) * float(gt_scale)
    # 同一时间戳重复写出时只留第一次。否则几十行拷贝会被最近邻配成“很多帧”，ATE 会看起来很小。
    unique_est = []
    unique_p = []
    seen = set()
    for index, stamp in enumerate(est_t):
        key = round(float(stamp), 9)
        if key in seen:
            continue
        seen.add(key)
        unique_est.append(est_t[index])
        unique_p.append(est_p[index])
    duplicate_rows = int(len(est_t) - len(unique_est))
    est_t = np.asarray(unique_est, dtype=float)
    est_p = unique_p
    paired_est = []
    paired_gt = []
    for index, stamp in enumerate(est_t):
        nearest = int(np.argmin(np.abs(gt_t - stamp)))
        if abs(float(gt_t[nearest] - stamp)) > 0.05:
            continue
        paired_est.append(est_p[index][:3, 3])
        paired_gt.append(gt_p[nearest][:3, 3])
    if len(paired_est) < 3:
        return {
            "n": len(paired_est),
            "estimate_rows": int(duplicate_rows + len(est_t)),
            "unique_timestamps": int(len(est_t)),
            "duplicate_rows_ignored": duplicate_rows,
            "note": "不同时间戳对上的帧少于 3，不算 ATE。重复行没有拿来凑数。",
        }
    rotation, translation = _umeyama_fixed_scale(paired_est, paired_gt)
    aligned = (rotation @ np.asarray(paired_est).T).T + translation
    errors = np.linalg.norm(aligned - np.asarray(paired_gt), axis=1)
    return {
        "n": int(errors.size),
        "estimate_frames": int(len(est_t)),
        "gt_frames": int(len(gt_t)),
        "rmse_m": float(np.sqrt(np.mean(errors ** 2))),
        "median_m": float(np.median(errors)),
        "max_m": float(np.max(errors)),
        "scale": "fixed_1",
        "alignment": "umeyama_se3",
        "measured_against": "HOT3D 左目虚拟针孔位姿。短片段管线检查，不是头戴精度。",
    }


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
