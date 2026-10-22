# -*- coding: utf-8 -*-
"""逐帧轨迹误差，以及能从现有数据算出来的对照。

误差是 SE3 Umeyama、尺度固定为 1 之后的相机中心距离，和 trajectory_ate 同一套对齐。
速度用的是头戴真值，不是 SLAM。接缝、关键帧间隔、亮度和模糊只和误差放在一起看。
相关高不高都不是原因。日志里没有逐帧特征数时，这一项是 null，不写成 0。
"""
import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import numpy as np

from headcam.hot3d_check import _prepare_pair, _rotation_angle_deg, _umeyama  # noqa: E402
from headcam.slam_batch import (  # noqa: E402
    ESTIMATE_POSE_SOURCE,
    REFERENCE_POSE_SOURCE,
    _header_pose_source,
)

NOTE = (
    "逐帧误差是 SE3 Umeyama、尺度固定为 1 之后的相机中心距离。"
    "线速度和角速度来自头戴真值相邻帧，不是 SLAM 轨迹。"
    "Pearson 在样本少于 3 或某一列没有起伏时是 null，不写成 0。"
    "这些相关和分组只是同一批帧上的数字，不是原因。"
)


def pearson(left, right):
    """两组等长数列的 Pearson。不够 3 个有限值，或有一列没有起伏，返回 None。"""
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("pearson 要两组等长的一维数列")
    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < 3:
        return None
    x = x[ok] - float(np.mean(x[ok]))
    y = y[ok] - float(np.mean(y[ok]))
    denom = float(np.sqrt(np.sum(x ** 2) * np.sum(y ** 2)))
    if denom <= 1e-18:
        return None
    return float(np.sum(x * y) / denom)


def near_seam_mask(n_frames, clip_lengths, radius):
    """clip 相接处两侧 radius 帧（含接缝两边的那两帧）为 True。

    序列开头和结尾不是接缝。只有一个 clip 时全是 False。
    radius 0 只标出紧贴切口的两帧。
    """
    lengths = [int(value) for value in clip_lengths]
    if any(value <= 0 for value in lengths):
        raise ValueError("clip 帧数要是正数")
    if sum(lengths) != int(n_frames):
        raise ValueError("clip 帧数加起来是 %d，不是 %d" % (sum(lengths), int(n_frames)))
    radius = int(radius)
    if radius < 0:
        raise ValueError("接缝半径不能是负数")
    mask = np.zeros(int(n_frames), dtype=bool)
    cut = 0
    for length in lengths[:-1]:
        cut += length
        start = cut - 1 - radius
        stop = cut + radius
        mask[max(0, start):min(int(n_frames), stop + 1)] = True
    return mask


def _error_stats(errors):
    values = np.asarray(errors, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return {
        "n": int(values.size),
        "rmse_m": float(np.sqrt(np.mean(values ** 2))),
        "median_m": float(np.median(values)),
        "max_m": float(np.max(values)),
    }


def group_stats(errors, mask):
    """mask 为 True / False 的两组误差。某一组没有帧时是 null。"""
    values = np.asarray(errors, dtype=float)
    flags = np.asarray(mask, dtype=bool)
    if values.shape != flags.shape or values.ndim != 1:
        raise ValueError("误差和 mask 要等长")
    return {"near": _error_stats(values[flags]), "away": _error_stats(values[~flags])}


def sse_share_worst_fraction(errors, fraction=0.1):
    """最大的 ceil(n * fraction) 帧（至少 1 帧）占全部平方和的比例。"""
    values = np.asarray(errors, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    total = float(np.sum(values ** 2))
    if total <= 0.0:
        return {"n_worst": int(values.size), "share": 0.0}
    count = max(1, int(np.ceil(float(fraction) * values.size)))
    ordered = np.sort(values)
    worst = ordered[-count:]
    return {"n_worst": int(count), "share": float(np.sum(worst ** 2) / total)}


def motion_series(times, poses):
    """头戴真值相邻帧的线速度（米/秒）和角速度（度/秒）。第 0 帧没有前一帧，是 nan。"""
    stamps = np.asarray(times, dtype=float)
    if len(poses) != stamps.size:
        raise ValueError("时间和位姿数量不一致")
    linear = np.full(stamps.size, np.nan, dtype=float)
    angular = np.full(stamps.size, np.nan, dtype=float)
    for index in range(1, stamps.size):
        dt = float(stamps[index] - stamps[index - 1])
        if dt <= 0.0:
            continue
        prev = np.asarray(poses[index - 1], dtype=float)
        curr = np.asarray(poses[index], dtype=float)
        linear[index] = float(np.linalg.norm(curr[:3, 3] - prev[:3, 3]) / dt)
        delta = prev[:3, :3].T @ curr[:3, :3]
        angular[index] = float(_rotation_angle_deg(delta) / dt)
    return linear, angular


def seconds_since_keyframe(frame_times, keyframe_times):
    """每帧距离不晚于它的最近关键帧过了多少秒。前面还没有关键帧时是 nan。"""
    frames = np.asarray(frame_times, dtype=float)
    keys = np.asarray(list(keyframe_times), dtype=float)
    keys = np.sort(keys[np.isfinite(keys)])
    out = np.full(frames.size, np.nan, dtype=float)
    if keys.size == 0:
        return out
    for index, stamp in enumerate(frames):
        earlier = keys[keys <= float(stamp) + 1e-9]
        if earlier.size == 0:
            continue
        out[index] = float(stamp) - float(earlier[-1])
    return out


def brightness_and_blur(image):
    """灰度均值，以及四邻域拉普拉斯的方差。方差越小，这块图越糊。不做 SLAM。"""
    array = np.asarray(image)
    if array.ndim == 3:
        gray = array.astype(float).mean(axis=2)
    elif array.ndim == 2:
        gray = array.astype(float)
    else:
        raise ValueError("图像要是二维或三维")
    brightness = float(np.mean(gray))
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return {"brightness": brightness, "laplacian_var": None}
    center = gray[1:-1, 1:-1]
    lap = (
        gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4.0 * center
    )
    return {"brightness": brightness, "laplacian_var": float(np.var(lap))}


def _tercile_stats(errors, signal):
    values = np.asarray(errors, dtype=float)
    series = np.asarray(signal, dtype=float)
    ok = np.isfinite(values) & np.isfinite(series)
    if int(ok.sum()) < 3:
        return None
    order = np.argsort(series[ok], kind="mergesort")
    chosen = values[ok][order]
    count = int(chosen.size)
    width = max(1, count // 3)
    return {
        "n_each": width,
        "low": _error_stats(chosen[:width]),
        "high": _error_stats(chosen[-width:]),
    }


def _bins(times, errors, bin_s):
    stamps = np.asarray(times, dtype=float)
    values = np.asarray(errors, dtype=float)
    if stamps.size == 0:
        return []
    width = float(bin_s)
    if width <= 0.0:
        raise ValueError("时间箱宽度要是正数")
    start = float(stamps[0])
    bins = []
    last = float(np.max(stamps))
    edge = start
    while edge <= last + 1e-12:
        nxt = edge + width
        chosen = (stamps >= edge) & (stamps < nxt if nxt <= last + 1e-9 else stamps <= last + 1e-9)
        # 最后一箱包含右端点。前面的箱是半开区间。
        if nxt <= last + 1e-9:
            chosen = (stamps >= edge) & (stamps < nxt)
        else:
            chosen = (stamps >= edge) & (stamps <= last + 1e-12)
        stats = _error_stats(values[chosen])
        if stats is not None:
            stats["t0_s"] = edge - start
            stats["t1_s"] = min(nxt, last) - start
            bins.append(stats)
        edge = nxt
        if edge > last + 1e-9:
            break
    return bins


def _tail_runs(times, errors, threshold):
    stamps = np.asarray(times, dtype=float)
    values = np.asarray(errors, dtype=float)
    runs = []
    current = []
    for index, error in enumerate(values):
        if np.isfinite(error) and float(error) > float(threshold):
            current.append(index)
            continue
        if current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    out = []
    for indexes in runs:
        chunk = values[indexes]
        out.append({
            "t0_s": float(stamps[indexes[0]] - stamps[0]),
            "t1_s": float(stamps[indexes[-1]] - stamps[0]),
            "n": int(len(indexes)),
            "rmse_m": float(np.sqrt(np.mean(chunk ** 2))),
            "max_m": float(np.max(chunk)),
        })
    return out


def _clip_at(index, clip_lengths, clip_ids):
    if not clip_lengths:
        return None, None
    acc = 0
    ids = list(clip_ids) if clip_ids is not None else [None] * len(clip_lengths)
    for length, clip_id in zip(clip_lengths, ids):
        if index < acc + int(length):
            return (None if clip_id is None else int(clip_id)), int(index - acc)
        acc += int(length)
    return None, None


def _signal_block(errors, signal):
    return {
        "pearson": pearson(errors, signal),
        "tercile": _tercile_stats(errors, signal),
    }


def summarize_timeline(
    times, errors, gt_poses, clip_lengths=None, clip_ids=None, seam_radius=2,
    bin_s=1.0, keyframe_times=None, brightness=None, blur=None, tail_factor=3.0,
):
    """把已经对齐好的逐帧误差和可测的对照收成一份记录。不重新估计轨迹。"""
    stamps = np.asarray(times, dtype=float)
    values = np.asarray(errors, dtype=float)
    if stamps.shape != values.shape or stamps.ndim != 1:
        raise ValueError("时间和误差要等长")
    if len(gt_poses) != stamps.size:
        raise ValueError("头戴位姿数量和误差不一致")
    stats = _error_stats(values)
    if stats is None:
        raise ValueError("没有有限的逐帧误差")
    max_index = int(np.nanargmax(values))
    clip_id, offset = _clip_at(max_index, clip_lengths or [], clip_ids)
    linear, angular = motion_series(stamps, gt_poses)
    threshold = max(0.02, float(tail_factor) * stats["median_m"])
    report = {
        "kind": "trajectory_error_timeline_v1",
        "estimate_pose_source": ESTIMATE_POSE_SOURCE,
        "reference_pose_source": REFERENCE_POSE_SOURCE,
        "n": stats["n"],
        "rmse_m": stats["rmse_m"],
        "median_m": stats["median_m"],
        "max_m": stats["max_m"],
        "max_error": {
            "index": max_index,
            "t_s": float(stamps[max_index] - stamps[0]),
            "error_m": float(values[max_index]),
            "clip_id": clip_id,
            "offset_in_clip": offset,
        },
        "sse_share_worst_10pct": sse_share_worst_fraction(values, 0.1),
        "bins_s": _bins(stamps, values, bin_s),
        "tail_threshold_m": threshold,
        "tail_runs": _tail_runs(stamps, values, threshold),
        "headset_motion": {
            "linear_speed_m_s": _error_stats(linear) and {
                "median": float(np.nanmedian(linear)),
                "p90": float(np.nanpercentile(linear, 90)),
                "max": float(np.nanmax(linear)),
            },
            "angular_speed_deg_s": {
                "median": float(np.nanmedian(angular)),
                "p90": float(np.nanpercentile(angular, 90)),
                "max": float(np.nanmax(angular)),
            },
            "error_vs_linear": _signal_block(values, linear),
            "error_vs_angular": _signal_block(values, angular),
        },
        "seams": None,
        "keyframes": None,
        "images": None,
        "tracked_features": None,
        "tracked_features_note": (
            "上游 stereo_euroc 在跟踪成功时不打印该帧的 inlier 个数。"
            "日志里没有逐帧特征数，这一项是 null，不是 0。"
        ),
        "note": NOTE,
    }
    if clip_lengths:
        mask = near_seam_mask(values.size, clip_lengths, seam_radius)
        seams = group_stats(values, mask)
        seams["radius_frames"] = int(seam_radius)
        seams["n_near"] = int(mask.sum())
        report["seams"] = seams
    if keyframe_times is not None:
        gaps = seconds_since_keyframe(stamps, keyframe_times)
        finite = gaps[np.isfinite(gaps)]
        report["keyframes"] = {
            "n": int(len(list(keyframe_times))),
            "median_seconds_since_keyframe": None if finite.size == 0 else float(np.median(finite)),
            "error_vs_seconds_since_keyframe": _signal_block(values, gaps),
        }
    if brightness is not None or blur is not None:
        report["images"] = {
            "error_vs_brightness": None if brightness is None else _signal_block(values, brightness),
            "error_vs_laplacian_var": None if blur is None else _signal_block(values, blur),
            "blur_note": "laplacian_var 是四邻域拉普拉斯的方差。越小越糊。",
        }
    return report


def _aligned_errors(estimate_tum, gt_tum, estimate_scale, gt_scale, max_dt_s):
    estimate_source = _header_pose_source(estimate_tum)
    if isinstance(estimate_source, str) and estimate_source.startswith("hot3d_gt"):
        return None, "估计轨迹的文件头是头戴真值，不能当作 SLAM 位姿。"
    if isinstance(estimate_source, str) and estimate_source != ESTIMATE_POSE_SOURCE:
        return None, "估计轨迹的 pose_source=%s，不是 %s。" % (estimate_source, ESTIMATE_POSE_SOURCE)
    reference_source = _header_pose_source(gt_tum)
    if reference_source != REFERENCE_POSE_SOURCE:
        return None, "真值文件头的 pose_source=%s，不是 %s。" % (reference_source, REFERENCE_POSE_SOURCE)
    prepared = _prepare_pair(estimate_tum, gt_tum, estimate_scale, gt_scale, max_dt_s)
    if len(prepared["paired_est"]) < 3:
        return None, "对上的不同时间戳少于 3，不算逐帧误差。"
    fitted = _umeyama(
        [pose[:3, 3] for pose in prepared["paired_est"]],
        [pose[:3, 3] for pose in prepared["paired_gt"]],
        with_scale=False,
    )
    if fitted is None:
        return None, "估计轨迹的位置没有展开，尺度没有定义，不算逐帧误差。"
    _scale, rotation, translation = fitted
    est = np.asarray([pose[:3, 3] for pose in prepared["paired_est"]], dtype=float)
    gt = np.asarray([pose[:3, 3] for pose in prepared["paired_gt"]], dtype=float)
    aligned = (rotation @ est.T).T + translation
    errors = np.linalg.norm(aligned - gt, axis=1)
    return {
        "times": prepared["paired_t"],
        "errors": errors,
        "gt_poses": prepared["paired_gt"],
    }, None


def _image_series(times, image_dir, estimate_scale):
    import cv2

    folder = Path(image_dir)
    brightness = []
    blur = []
    missing = 0
    scale = float(estimate_scale)
    for stamp in times:
        raw = int(round(float(stamp) / scale)) if scale != 0.0 else int(round(float(stamp)))
        path = folder / ("%d.png" % raw)
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            missing += 1
            brightness.append(np.nan)
            blur.append(np.nan)
            continue
        stats = brightness_and_blur(image)
        brightness.append(stats["brightness"])
        blur.append(stats["laplacian_var"] if stats["laplacian_var"] is not None else np.nan)
    return np.asarray(brightness, dtype=float), np.asarray(blur, dtype=float), missing


def _keyframe_times(path, scale):
    from headcam.hot3d_check import _load_scaled_tum

    times, _poses = _load_scaled_tum(path, scale)
    return [float(value) for value in times]


def diagnose_files(
    estimate_tum, gt_tum, estimate_scale=1e-9, gt_scale=1.0, max_dt_s=0.05,
    clip_lengths=None, clip_ids=None, seam_radius=2, bin_s=1.0,
    keyframe_tum=None, image_dir=None,
):
    """从轨迹文件写出时间线。失败时不填 rmse。"""
    paired, failure = _aligned_errors(estimate_tum, gt_tum, estimate_scale, gt_scale, max_dt_s)
    if failure:
        return {
            "kind": "trajectory_error_timeline_v1",
            "status": "failed",
            "failure": failure,
            "rmse_m": None,
            "tracked_features": None,
            "note": NOTE,
        }
    keyframe_times = None
    if keyframe_tum and Path(keyframe_tum).is_file():
        keyframe_times = _keyframe_times(keyframe_tum, estimate_scale)
    brightness = None
    blur = None
    if image_dir and Path(image_dir).is_dir():
        brightness, blur, _missing = _image_series(paired["times"], image_dir, estimate_scale)
    report = summarize_timeline(
        paired["times"], paired["errors"], paired["gt_poses"],
        clip_lengths=clip_lengths, clip_ids=clip_ids, seam_radius=seam_radius,
        bin_s=bin_s, keyframe_times=keyframe_times, brightness=brightness, blur=blur,
    )
    report["status"] = "ok"
    report["failure"] = None
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="逐帧 SE3 误差对照头戴运动、接缝和关键帧间隔。")
    parser.add_argument("--estimate", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--estimate-scale", type=float, default=1e-9)
    parser.add_argument("--gt-scale", type=float, default=1.0)
    parser.add_argument("--clip-lengths", default="", help="逗号分隔。例如 150,150,150")
    parser.add_argument("--clip-ids", default="")
    parser.add_argument("--seam-radius", type=int, default=2)
    parser.add_argument("--bin-s", type=float, default=1.0)
    parser.add_argument("--keyframes", default=None)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    lengths = [int(part) for part in args.clip_lengths.split(",") if part != ""]
    ids = [int(part) for part in args.clip_ids.split(",") if part != ""]
    try:
        report = diagnose_files(
            args.estimate, args.gt,
            estimate_scale=args.estimate_scale, gt_scale=args.gt_scale,
            clip_lengths=lengths or None,
            clip_ids=ids or None,
            seam_radius=args.seam_radius,
            bin_s=args.bin_s,
            keyframe_tum=args.keyframes,
            image_dir=args.image_dir,
        )
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(report.get("status"), "rmse", report.get("rmse_m"))
    return 0 if report.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
