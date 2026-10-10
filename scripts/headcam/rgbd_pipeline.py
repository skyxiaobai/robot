# -*- coding: utf-8 -*-
"""iPhone Pro 单目 RGB + 激光雷达深度的手部管线（和双目管线共用 episode / QC / LeRobot 导出）。

    iPhone 会话（rgb.mp4、depth.npz、calib.yaml、slam.tum=ARKit 位姿，见 scripts/iphone/record3d_adapter.py）
      → WiLoR 在 RGB 上逐帧认手（21 点 2D + 带尺度的单目 3D 手型）
      → 每只手：在 2D 关键点凸包（手的区域）里取高置信度的激光雷达深度，
        先取“近层”（手在物体前面：区域内深度的低分位数），只保留近层附近的深度，排除背景和物体边缘
      → 逐关节在 3x3 深度邻域取中位数 → 沿视线反投影成公制 3D（加上皮肤到关节中心的固定偏移）
      → 把 WiLoR 单目 21 点稳健相似变换对齐到这些深度点（同双目 rigid_fit_hand），得到 21 个关节
      → 一致性检查（有效深度关节数、手腕深度 0.25–1.5 m、对齐残差、尺度、手掌长度、深度置信度、ARKit 位姿跳变）
      → 世界系（ARKit 位姿）→ 速度门限 + RTS 平滑 → QC → LeRobot

状态码沿用双目（stereo_qc 只区分 none / one_view / ok / 其他），新增：
``lowconf``（手区域高置信深度太少）、``fit``（深度与 WiLoR 手型对不上）、``tracking``（ARKit 位姿跳变）。

逐手状态同时写入 ``stereo.per_frame``、``iphone.per_frame`` 和 ``hands[side].label_status``。
LeRobot 导出与双目共用 ``action_valid``：当前帧和下一帧都是实测手腕才为 1。
"""
import json
import time
from pathlib import Path

import numpy as np

from headcam.hand_pose import JOINTS, associate_camera_poses, get_backend
from headcam.stereo_pipeline import MIDDLE_MCP, SIDES, _hand_to_json, build_stereo_episode, rigid_fit_hand


class RGBDParams(object):
    """深度取样与检查的参数。默认值是按传感器常识先定的，不是在评测真值上调出来的。"""

    def __init__(self, min_conf=2, hand_band_m=0.10, near_percentile=10.0, patch=1, joint_offset_m=0.01,
                 min_joints=8, depth_range_m=(0.25, 1.5), max_fit_residual_m=0.02, scale_range=(0.75, 1.33),
                 palm_range_m=(0.05, 0.15), min_conf_frac=0.3, max_pose_jump_m=0.10, max_pose_jump_deg=20.0):
        self.min_conf = int(min_conf)            # Record3D：0 低 / 1 中 / 2 高
        self.hand_band_m = float(hand_band_m)    # 近层之后多厚还算手
        self.near_percentile = float(near_percentile)
        self.patch = int(patch)                  # 深度图上的邻域半径（像素，256x192 分辨率）
        self.joint_offset_m = float(joint_offset_m)  # 激光雷达测到皮肤，关节中心大约再深 1 cm
        self.min_joints = int(min_joints)
        self.depth_range_m = tuple(depth_range_m)
        self.max_fit_residual_m = float(max_fit_residual_m)
        self.scale_range = tuple(scale_range)
        self.palm_range_m = tuple(palm_range_m)
        self.min_conf_frac = float(min_conf_frac)
        self.max_pose_jump_m = float(max_pose_jump_m)
        self.max_pose_jump_deg = float(max_pose_jump_deg)

    def to_dict(self):
        return dict(self.__dict__)


def _hull_mask(uv, shape, dilate=1):
    import cv2
    mask = np.zeros(shape, np.uint8)
    pts = uv[np.isfinite(uv).all(1)]
    if len(pts) < 3:
        return mask.astype(bool)
    hull = cv2.convexHull(np.round(pts).astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 1)
    if dilate:
        mask = cv2.dilate(mask, np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8))
    return mask.astype(bool)


def depth_hand(hand, depth, conf, image_size, K, p, mono_override=None):
    """一只手一帧。hand：WiLoR 输出 JSON（keypoints_2d 是 RGB 像素，joints_cam 是单目公制 3D）。
    depth/conf：激光雷达深度（米）和置信度，任意分辨率，按比例对齐到 RGB（Record3D 的深度和 RGB 同视场）。
    返回与 stereo_hand 同样键的字典。"""
    out = {"status": "none", "joints_cam": None, "raw_joints": None, "confidence": None, "reproj_px": None,
           "wrist_depth_m": None, "palm_m": None, "scale": None, "mono_joints": None}
    if hand is None or hand.get("keypoints_2d") is None:
        return out
    uv = np.asarray(hand["keypoints_2d"], dtype=float)
    mono = mono_override if mono_override is not None else (
        None if hand.get("joints_cam") is None else np.asarray(hand["joints_cam"], dtype=float))
    out["mono_joints"] = mono
    out["confidence"] = np.asarray(hand.get("confidence") or [1.0] * JOINTS, dtype=float)
    w, h = image_size
    dh, dw = depth.shape
    sx, sy = dw / float(w), dh / float(h)
    uvd = np.stack([(uv[:, 0] + 0.5) * sx - 0.5, (uv[:, 1] + 0.5) * sy - 0.5], 1)
    good = np.isfinite(depth) & (depth > 0) & (conf >= p.min_conf)
    mask = _hull_mask(uvd, depth.shape, dilate=1)
    area = int(mask.sum())
    vals = depth[mask & good]
    out["conf_frac"] = float(len(vals)) / area if area else 0.0
    if area == 0 or out["conf_frac"] < p.min_conf_frac or len(vals) < 3:
        out["status"] = "lowconf"
        return out
    near = float(np.percentile(vals, p.near_percentile))
    lo, hi = near - 0.02, near + p.hand_band_m
    out["near_depth_m"] = near
    pts = np.full((JOINTS, 3), np.nan)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    r = p.patch
    for j in range(JOINTS):
        u, v = int(round(uvd[j, 0])), int(round(uvd[j, 1]))
        if not (0 <= u < dw and 0 <= v < dh):
            continue
        sl = (slice(max(0, v - r), v + r + 1), slice(max(0, u - r), u + r + 1))
        patch = depth[sl][good[sl]]
        patch = patch[(patch >= lo) & (patch <= hi)]
        if len(patch) == 0:
            continue
        z = float(np.median(patch)) + p.joint_offset_m
        pts[j] = [(uv[j, 0] - cx) / fx * z, (uv[j, 1] - cy) / fy * z, z]
    ok = np.isfinite(pts).all(1)
    out["raw_joints"] = pts
    out["n_joints_ok"] = int(ok.sum())
    if ok.sum() < p.min_joints or mono is None or not np.isfinite(mono).all():
        out["status"] = "few_joints"
        return out
    fit, resid = rigid_fit_hand(mono, pts, out["confidence"] * ok)
    if fit is None:
        out["status"] = "few_joints"
        return out
    # 尺度：对齐后的手掌长度 / 单目手掌长度
    out["scale"] = float(np.linalg.norm(fit[MIDDLE_MCP] - fit[0]) / max(np.linalg.norm(mono[MIDDLE_MCP] - mono[0]), 1e-6))
    out["fit_residual_m"] = resid
    out["wrist_depth_m"] = float(fit[0, 2])
    out["palm_m"] = float(np.linalg.norm(fit[MIDDLE_MCP] - fit[0]))
    status = "ok"
    if not (p.depth_range_m[0] <= fit[0, 2] <= p.depth_range_m[1]):
        status = "depth"
    elif resid > p.max_fit_residual_m or not (p.scale_range[0] <= out["scale"] <= p.scale_range[1]):
        status = "fit"
    elif not (p.palm_range_m[0] <= out["palm_m"] <= p.palm_range_m[1]):
        status = "palm"
    out["status"] = status
    if status == "ok":
        out["joints_cam"] = fit
    return out


def pose_jumps(poses, max_m, max_deg):
    """ARKit 跟踪状态 Record3D 不导出，用相邻帧位姿跳变代替：超过阈值的帧记 True。"""
    bad = np.zeros(len(poses), dtype=bool)
    for i in range(1, len(poses)):
        a, b = np.asarray(poses[i - 1]), np.asarray(poses[i])
        dt = np.linalg.norm(b[:3, 3] - a[:3, 3])
        c = np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2, -1, 1)
        if dt > max_m or np.degrees(np.arccos(c)) > max_deg:
            bad[i] = True
    return bad


def load_rgbd_session(session_dir):
    import convert_headcam as ch
    s = Path(session_dir)
    calib = json.loads((s / "calib.yaml").read_text(encoding="utf-8"))
    K = np.asarray(calib["K_rgb"], dtype=float)
    calib_full = dict(calib, K_left=K)
    data = np.load(s / "depth.npz")

    def first(*names):
        for n in names:
            if (s / n).is_file():
                return s / n
        return None
    return {"session": s, "metadata": ch._load_metadata(s), "calib": calib_full, "calib_path": s / "calib.yaml",
            "left_video": s / "rgb.mp4", "right_video": s / "rgb.mp4", "rgb_video": s / "rgb.mp4",
            "depth": data["depth"], "conf": data["conf"], "slam_path": first("slam.tum"),
            "imu_path": first("imu.csv"), "timestamps_path": first("timestamps.csv"),
            "gt_path": first("gt/hands_gt.json")}


def run_rgb_backend(info, backend_name, cache_path, backend=None, log=print):
    import convert_headcam as ch
    if cache_path is not None and Path(cache_path).is_file():
        data = json.loads(Path(cache_path).read_text(encoding="utf-8"))
        if data.get("backend") == backend_name:
            log("  复用缓存 %s" % cache_path)
            return data["frames"], data.get("seconds")
    backend = backend or get_backend(backend_name)
    frames, t0 = [], time.time()
    for index, image in enumerate(ch._iter_rgb(info["rgb_video"])):
        pred = backend.predict(image, calib=info["calib"])
        frames.append({side: _hand_to_json(pred.get(side)) for side in SIDES})
        if index % 50 == 0:
            log("  RGB 第 %d 帧  %.1fs" % (index, time.time() - t0))
    seconds = time.time() - t0
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(json.dumps({"backend": backend_name, "frames": frames, "seconds": seconds}),
                                    encoding="utf-8")
    return frames, seconds


def make_session_builder(rgbd_params=None):
    """返回给 ``stereo_pipeline.run_pipeline(session_builder=...)`` 用的函数。"""
    rp = rgbd_params or RGBDParams()

    def build(session_dir, out, params, backend_name, backend, log):
        import convert_headcam as ch
        info = load_rgbd_session(session_dir)
        name = info["session"].name
        log("[%s] iPhone RGB-D，手部模型（%s）" % (name, backend_name))
        frames, seconds = run_rgb_backend(info, backend_name, out / "cache" / ("%s.%s.rgbd.json" % (name, backend_name)),
                                          backend=backend, log=log)
        t0 = time.time()
        n = min(len(frames), len(info["depth"]))
        frames = frames[:n]
        fps = float(info["metadata"]["fps"])
        ts = ch._read_timestamps(info["timestamps_path"], n, fps) if info["timestamps_path"] else [i / fps for i in range(n)]
        poses = associate_camera_poses(ts, info["slam_path"])[0]
        jumps = pose_jumps(poses, rp.max_pose_jump_m, rp.max_pose_jump_deg)
        size = (int(info["calib"]["image_width"]), int(info["calib"]["image_height"]))
        K = info["calib"]["K_left"]

        def hand_fn(index, side):
            res = depth_hand(frames[index][side], info["depth"][index].astype(np.float32), info["conf"][index],
                             size, K, rp)
            if jumps[index] and res["status"] == "ok":
                res["status"], res["joints_cam"] = "tracking", None
            return res
        views = {"left": frames, "right": frames}
        episode, extra = build_stereo_episode(info, views, params, hand_fn=hand_fn, source="iphone_lidar",
                                              backend_label="%s_lidar" % backend_name)
        statuses = {side: list(episode["stereo"]["per_frame"][side]) for side in SIDES}
        for side in SIDES:
            episode["hands"][side]["label_status"] = list(statuses[side])
        episode["iphone"] = {"per_frame": {side: list(statuses[side]) for side in SIDES}}
        episode["stereo"]["rgbd"] = {
            "params": rp.to_dict(), "pose_jump_frames": int(jumps.sum()),
            "conf_frac": {s: [d.get("conf_frac") for d in extra["diag"][s]] for s in SIDES},
            "fit_residual_m": {s: [d.get("fit_residual_m") for d in extra["diag"][s]] for s in SIDES}}
        return info, episode, extra, seconds, time.time() - t0
    return build
