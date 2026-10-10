#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查一个头戴会话目录是否符合 docs/headcam_data_spec.md §7.7，设备一到就能用来验收每次录制。

检查项（ERROR 会让退出码为 1；WARN 只提示）：

- 必需文件：stereo/left.mp4、stereo/right.mp4、calib.yaml；metadata.json 建议有
- 标定：能被 load_calibration 读出；基线长度（默认要求 4–15 cm）；内参主点在图内；图像尺寸与视频一致
- 视频：左右帧数一致；与 timestamps.csv 行数一致
- 时间戳：严格递增；帧间隔抖动（相对中位数 >50% 的帧数）；推算帧率与 metadata.fps 一致
- 左右同步：有 stereo/timestamps_lr.csv 时，逐帧 |左-右|，默认最大 ≤1 ms（HOT3D 仿真：10 ms 时 2 cm 内比例从 67% 掉到 62%）
- SLAM：slam.tum 覆盖全部帧，最近位姿时间差 ≤ 半帧
- IMU（可选）：imu.csv 行数与采样率（≥100 Hz 建议）、时间覆盖录像区间

iPhone 会话（有 depth.npz，见 scripts/iphone/record3d_adapter.py）改走 ``validate_iphone``：
rgb.mp4 / depth.npz / timestamps.csv 帧数一致；内参主点在图内；深度有效像素比例、高置信比例、
深度落在激光雷达量程 0.25–5 m 的比例；ARKit 位姿覆盖全部帧、相邻帧跳变（代替 Record3D 不导出的跟踪状态）。

用法::

    python scripts/validate_session.py /path/to/session [--json report.json]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULTS = {"baseline_min_m": 0.04, "baseline_max_m": 0.15, "sync_max_ms": 1.0, "jitter_frac": 0.5,
            "max_jitter_frames": 0.01, "imu_min_hz": 100.0}


def _video_info(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    n = 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while cap.grab():
        n += 1
    cap.release()
    return {"frames": n, "width": w, "height": h}


def _read_numeric_csv(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or not (line[0].isdigit() or line[0] in "-."):
                continue
            rows.append([float(x) for x in line.replace(",", " ").split()])
    return np.asarray(rows, dtype=float)


IPHONE_DEFAULTS = {"min_valid_frac": 0.5, "min_high_conf_frac": 0.3, "lidar_range_m": (0.25, 5.0),
                   "max_pose_jump_m": 0.10, "max_pose_jump_deg": 20.0, "max_jump_frames": 0.01}


def validate_iphone(session, **overrides):
    opt = dict(IPHONE_DEFAULTS)
    opt.update(overrides)
    s = Path(session)
    errors, warns, facts = [], [], {}
    for rel in ("rgb.mp4", "depth.npz", "calib.yaml"):
        if not (s / rel).is_file():
            errors.append("缺少 %s" % rel)
    if errors:
        return {"session": str(s), "kind": "iphone", "ok": False, "errors": errors, "warnings": warns, "facts": facts}
    meta = json.loads((s / "metadata.json").read_text(encoding="utf-8")) if (s / "metadata.json").is_file() else {}
    fps = float(meta.get("fps", 30.0))
    calib = json.loads((s / "calib.yaml").read_text(encoding="utf-8"))
    K = np.asarray(calib["K_rgb"], dtype=float)
    w, h = int(calib["image_width"]), int(calib["image_height"])
    if not (0 < K[0, 2] < w and 0 < K[1, 2] < h):
        errors.append("K_rgb 主点不在图像里")
    vid = _video_info(s / "rgb.mp4")
    data = np.load(s / "depth.npz")
    depth, conf = data["depth"].astype(np.float32), data["conf"]
    n = len(depth)
    facts.update(frames=n, rgb=vid, depth_shape=list(depth.shape[1:]))
    if vid is None:
        errors.append("打不开 rgb.mp4")
    else:
        if vid["frames"] != n:
            errors.append("rgb.mp4 有 %d 帧，深度有 %d 帧" % (vid["frames"], n))
        if (vid["width"], vid["height"]) != (w, h):
            errors.append("视频尺寸 %dx%d 与 calib %dx%d 不一致" % (vid["width"], vid["height"], w, h))
    valid = np.isfinite(depth) & (depth > 0)
    lo, hi = opt["lidar_range_m"]
    facts["depth_valid_frac"] = float(valid.mean())
    facts["depth_high_conf_frac"] = float((conf >= 2).mean())
    facts["depth_in_range_frac"] = float(((depth >= lo) & (depth <= hi) & valid).sum() / max(valid.sum(), 1))
    facts["depth_median_m"] = float(np.median(depth[valid])) if valid.any() else None
    if facts["depth_valid_frac"] < opt["min_valid_frac"]:
        errors.append("有效深度像素只有 %.0f%%（不是激光雷达录制？用了前置摄像头？）" % (100 * facts["depth_valid_frac"]))
    if facts["depth_high_conf_frac"] < opt["min_high_conf_frac"]:
        warns.append("高置信深度只有 %.0f%%（太暗、太远或反光表面多）" % (100 * facts["depth_high_conf_frac"]))
    stamps = None
    if (s / "timestamps.csv").is_file():
        arr = _read_numeric_csv(s / "timestamps.csv")
        stamps = arr[:, -1]
        if len(stamps) != n:
            errors.append("timestamps.csv 有 %d 行，深度 %d 帧" % (len(stamps), n))
        d = np.diff(stamps)
        if len(d) and (d <= 0).any():
            errors.append("时间戳不是严格递增（%d 处）" % int((d <= 0).sum()))
        if len(d):
            med = float(np.median(d))
            facts["measured_fps"] = 1.0 / med
            if abs(1.0 / med - fps) > 0.05 * fps:
                errors.append("实测帧率 %.2f 与 metadata.fps %.2f 不一致" % (1.0 / med, fps))
    else:
        warns.append("没有 timestamps.csv")
    if (s / "slam.tum").is_file() and stamps is not None:
        from headcam.hand_pose import associate_camera_poses
        from headcam.rgbd_pipeline import pose_jumps
        poses, gaps = associate_camera_poses(list(stamps), s / "slam.tum")
        facts["pose_max_gap_s"] = float(np.max(gaps))
        if facts["pose_max_gap_s"] > 0.5 / fps:
            errors.append("ARKit 位姿与帧时间最大相差 %.1f ms，超过半帧" % (1e3 * facts["pose_max_gap_s"]))
        jumps = pose_jumps(poses, opt["max_pose_jump_m"], opt["max_pose_jump_deg"])
        facts["pose_jump_frames"] = int(jumps.sum())
        if jumps.sum() > opt["max_jump_frames"] * n:
            warns.append("ARKit 位姿跳变 %d 帧（跟踪丢失/重定位？录制时别遮挡镜头、别对着白墙）" % int(jumps.sum()))
    else:
        errors.append("没有 slam.tum（ARKit 位姿）：Record3D 导出里缺 poses？")
    return {"session": str(s), "kind": "iphone", "ok": not errors, "errors": errors, "warnings": warns, "facts": facts}


def validate(session, **overrides):
    if (Path(session) / "depth.npz").is_file():
        return validate_iphone(session, **{k: v for k, v in overrides.items() if k in IPHONE_DEFAULTS})
    opt = dict(DEFAULTS)
    opt.update(overrides)
    s = Path(session)
    errors, warns, facts = [], [], {}

    def need(rel):
        p = s / rel
        if not p.is_file():
            errors.append("缺少 %s" % rel)
            return None
        return p

    left, right, calib_path = need("stereo/left.mp4"), need("stereo/right.mp4"), need("calib.yaml")
    meta = {}
    if (s / "metadata.json").is_file():
        meta = json.loads((s / "metadata.json").read_text(encoding="utf-8"))
    else:
        warns.append("没有 metadata.json（fps 会按 30 处理）")
    fps = float(meta.get("fps", 30.0))

    calib = None
    if calib_path:
        try:
            from headcam.hand_pose import load_calibration
            calib = load_calibration(calib_path)
            base = float(np.linalg.norm(calib["T"]))
            facts["baseline_m"] = base
            if not (opt["baseline_min_m"] <= base <= opt["baseline_max_m"]):
                errors.append("基线 %.1f cm 不在 %.0f–%.0f cm（单位写错了？毫米当成米？）" % (
                    100 * base, 100 * opt["baseline_min_m"], 100 * opt["baseline_max_m"]))
            for key in ("K_left", "K_right"):
                K = calib[key]
                if not (0 < K[0, 2] < calib["image_width"] and 0 < K[1, 2] < calib["image_height"]):
                    errors.append("%s 主点不在图像里" % key)
        except Exception as exc:  # noqa: BLE001
            errors.append("calib.yaml 读不出来：%s" % exc)

    vids = {}
    for name, p in (("left", left), ("right", right)):
        if p:
            info = _video_info(p)
            if info is None:
                errors.append("打不开 %s" % p.name)
            else:
                vids[name] = info
    facts["videos"] = vids
    if len(vids) == 2:
        if vids["left"]["frames"] != vids["right"]["frames"]:
            errors.append("左右帧数不一致：%d vs %d" % (vids["left"]["frames"], vids["right"]["frames"]))
        if calib is not None and (vids["left"]["width"], vids["left"]["height"]) != (
                calib["image_width"], calib["image_height"]):
            errors.append("视频尺寸 %dx%d 与标定 %dx%d 不一致" % (
                vids["left"]["width"], vids["left"]["height"], calib["image_width"], calib["image_height"]))
    n = vids.get("left", {}).get("frames")

    stamps = None
    if (s / "timestamps.csv").is_file():
        arr = _read_numeric_csv(s / "timestamps.csv")
        stamps = arr[:, -1] if arr.size else np.zeros(0)
        if n is not None and len(stamps) != n:
            errors.append("timestamps.csv 有 %d 行，视频 %d 帧" % (len(stamps), n))
        d = np.diff(stamps)
        if len(d) and (d <= 0).any():
            errors.append("时间戳不是严格递增（%d 处）" % int((d <= 0).sum()))
        if len(d):
            med = float(np.median(d))
            jitter = int((np.abs(d - med) > opt["jitter_frac"] * med).sum())
            facts.update(measured_fps=1.0 / med, dropped_or_jitter_frames=jitter)
            if abs(1.0 / med - fps) > 0.05 * fps:
                errors.append("实测帧率 %.2f 与 metadata.fps %.2f 不一致" % (1.0 / med, fps))
            if jitter > opt["max_jitter_frames"] * len(d):
                warns.append("帧间隔异常 %d 处（丢帧或卡顿）" % jitter)
    else:
        warns.append("没有 timestamps.csv：只能按 frame/fps 推时间，无法和 SLAM/IMU 对齐")

    lr = s / "stereo" / "timestamps_lr.csv"
    if lr.is_file():
        arr = _read_numeric_csv(lr)
        off = np.abs(arr[:, 1] - arr[:, 2]) * 1e3
        facts["sync_ms"] = {"max": float(off.max()), "median": float(np.median(off))}
        if off.max() > opt["sync_max_ms"]:
            errors.append("左右曝光时间差最大 %.2f ms，超过 %.1f ms（硬件同步没生效？）" % (off.max(), opt["sync_max_ms"]))
    else:
        warns.append("没有 stereo/timestamps_lr.csv：无法核对左右同步")

    if (s / "slam.tum").is_file() and stamps is not None:
        from headcam.hand_pose import associate_camera_poses
        _, gaps = associate_camera_poses(list(stamps), s / "slam.tum")
        gmax = float(np.max(gaps)) if len(gaps) else 0.0
        facts["slam_max_gap_s"] = gmax
        if gmax > 0.5 / fps:
            errors.append("SLAM 位姿与帧时间最大相差 %.1f ms，超过半帧" % (1e3 * gmax))
    else:
        warns.append("没有 slam.tum：输出只能在相机系（coordinate_frame=camera）")

    if (s / "imu.csv").is_file():
        arr = _read_numeric_csv(s / "imu.csv")
        if len(arr) > 1:
            t = arr[:, 0]
            hz = 1.0 / float(np.median(np.diff(t)))
            facts["imu_hz"] = hz
            if hz < opt["imu_min_hz"]:
                warns.append("IMU 采样率 %.0f Hz 偏低" % hz)
            if stamps is not None and (t[0] > stamps[0] or t[-1] < stamps[-1]):
                warns.append("IMU 时间没有覆盖整段录像")
    else:
        warns.append("没有 imu.csv（HOT3D 适配会话没有 IMU，属正常）")
    return {"session": str(s), "ok": not errors, "errors": errors, "warnings": warns, "facts": facts}


def main(argv=None):
    ap = argparse.ArgumentParser(description="检查头戴会话目录格式")
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--json", default=None)
    ap.add_argument("--sync-max-ms", type=float, default=DEFAULTS["sync_max_ms"])
    a = ap.parse_args(argv)
    reports = [validate(p, sync_max_ms=a.sync_max_ms) for p in a.sessions]
    for r in reports:
        print("%s  %s" % ("通过" if r["ok"] else "不通过", r["session"]))
        for e in r["errors"]:
            print("  [ERROR] " + e)
        for w in r["warnings"]:
            print("  [WARN]  " + w)
        print("  " + json.dumps(r["facts"], ensure_ascii=False))
    if a.json:
        Path(a.json).write_text(json.dumps(reports, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0 if all(r["ok"] for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
