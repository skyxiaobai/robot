# -*- coding: utf-8 -*-
"""一条命令的双目手部管线：会话目录 → 统一 episode → QC → LeRobot。

    双目会话（stereo/left.mp4、stereo/right.mp4、calib.yaml、可选 rgb.mp4 / imu.csv / slam.tum）
      → WiLoR 分别跑左目和右目
      → 21 个关节逐点三角化（左相机系，米）
      → 左右一致性检查（两目都认到、重投影误差、深度范围、手掌尺寸；不合格的手这一帧丢掉）
      → One Euro 平滑（默认开）+ 最多补 5 帧（补的帧标 filled，QC 不当成测到）；固定手型默认关
      → 乘相机世界位姿（slam.tum）得到世界系 21 点和手腕 6DoF
      → 统一 episode JSON → 双目 QC（产出率 + 拒绝原因）→ LeRobot v3.0 导出

会话格式见 ``docs/headcam_data_spec.md`` §7.7。HOT3D 通过 ``headcam/hot3d_adapter.py`` 先转成同样的目录。
会话里有 ``gt/hands_gt.json`` 时（HOT3D 适配器会写），额外和真值比手腕误差。
"""
import json
import math
import time
from pathlib import Path

import numpy as np

from egodata.coverage import normalize_environment
from egodata.schema import SCHEMA_VERSION, save_episode, validate_episode
from headcam.hand_pose import (
    JOINTS,
    associate_camera_poses,
    fuse_metric_joints,
    get_backend,
    load_calibration,
    project_pinhole,
    transform_points,
    triangulate_pair,
)
from headcam.hand_track_refine import RefineParams, refine_hands

SIDES = ("left", "right")
# OpenPose / MediaPipe 21 点：0 手腕，9 中指 MCP
MIDDLE_MCP = 9


class StereoParams(object):
    """一致性检查与精修的开关。阈值是事先按常识定的，不是在 HOT3D 真值上调出来的。"""

    def __init__(self, max_reproj_px=10.0, joint_reproj_px=15.0, min_joints=15,
                 depth_range_m=(0.10, 1.20), palm_range_m=(0.05, 0.15),
                 smooth="rts", min_cutoff=3.0, beta=50.0, gap_fill=True, max_gap=5, fixed_shape=False,
                 consistency=True, wrist_mode="rigid_fit", velocity_gate_m=0.02, rts_q=0.3, rts_r=4e-4,
                 max_median_reproj_px=None, max_offaxis_deg=None, assoc=True, recrop=True, assoc_params=None):
        self.max_reproj_px = float(max_reproj_px)
        self.joint_reproj_px = float(joint_reproj_px)
        self.min_joints = int(min_joints)
        self.depth_range_m = tuple(float(v) for v in depth_range_m)
        self.palm_range_m = tuple(float(v) for v in palm_range_m)
        self.smooth = smooth
        # One Euro 参数：EgoDex 精修（PR #19）的默认 min_cutoff=1、beta=0.5 在世界系米制轨迹上滞后严重
        # （HOT3D clip-000000 上手腕中位误差 1.14→3.32 cm）。这里的 3 / 50 是只在 clip-000000（开发段）
        # 上选的，其余三段当留出集报告。
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.gap_fill = bool(gap_fill)
        self.max_gap = int(max_gap)
        self.fixed_shape = bool(fixed_shape)
        self.consistency = bool(consistency)
        # 尾部误差（docs/hot3d_stereo/tail.md）：手腕不用单点三角化，而是把 WiLoR 单目 21 点（带尺度）稳健刚体
        # 对齐到三角化关节后取对齐后的手腕；世界系手腕先做 2 cm 速度门限剔跳点，再做离线 RTS 平滑。
        # 这些值在 HOT3D 上按“一段调参、其余三段测试”选出，4 种轮换选到的组合基本一致。
        # smooth="rts" 时用 RTS 代替 One Euro；wrist_mode="tri" 恢复 PR #22 行为。
        self.wrist_mode = wrist_mode
        self.velocity_gate_m = float(velocity_gate_m or 0.0)
        self.rts_q = float(rts_q)
        self.rts_r = float(rts_r)
        # 可选的严格门限（默认关）：换更低产出换更小 p90，曲线见 docs/hot3d_stereo/tail.md
        self.max_median_reproj_px = max_median_reproj_px
        self.max_offaxis_deg = max_offaxis_deg
        # 手部关联（docs/hot3d_stereo/hand_assoc.md）：WiLoR 多候选（翻转 TTA + 左右两种假设）→ 时序 + 双目关联
        # → 另一目重新裁剪。只对 wilor 后端生效；assoc=False 恢复 PR #23 的逐目 predict。
        self.assoc = bool(assoc)
        self.recrop = bool(recrop)
        self.assoc_params = dict(assoc_params or {})

    def to_dict(self):
        return dict(self.__dict__)

    def refine(self):
        smooth = "none" if self.smooth == "rts" else self.smooth
        if smooth in (None, "none") and not self.gap_fill and not self.fixed_shape:
            return None
        return RefineParams(smooth=smooth or "none", min_cutoff=self.min_cutoff, beta=self.beta,
                            gap_fill=self.gap_fill, max_gap=self.max_gap,
                            fixed_shape=self.fixed_shape, lr_consistency=False)


# ---------------------------------------------------------------- 会话读取

def _first(session, names):
    for name in names:
        path = session / name
        if path.is_file():
            return path
    return None


def load_session(session_dir):
    """读会话目录的元数据、标定、时间戳和位姿（不读视频）。"""
    import convert_headcam as ch
    session = Path(session_dir)
    calib_path = _first(session, ("calib.yaml", "calib.yml", "stereo/calib.yaml"))
    if calib_path is None:
        raise FileNotFoundError("双目会话需要 calib.yaml：%s" % session)
    left = _first(session, ("stereo/left.mp4", "left.mp4"))
    right = _first(session, ("stereo/right.mp4", "right.mp4"))
    if left is None or right is None:
        raise FileNotFoundError("双目会话需要 stereo/left.mp4 和 stereo/right.mp4：%s" % session)
    return {
        "session": session,
        "metadata": ch._load_metadata(session),
        "calib": load_calibration(calib_path),
        "calib_path": calib_path,
        "left_video": left,
        "right_video": right,
        "rgb_video": _first(session, ("rgb.mp4", "color.mp4")),
        "slam_path": _first(session, ("slam.tum", "trajectory.tum")),
        "imu_path": _first(session, ("imu.csv", "imu.txt")),
        "timestamps_path": _first(session, ("timestamps.csv",)),
        "gt_path": _first(session, ("gt/hands_gt.json",)),
    }


# ---------------------------------------------------------------- 手部后端

def _hand_to_json(hand):
    if not isinstance(hand, dict) or hand.get("keypoints_2d") is None:
        return None
    out = {"keypoints_2d": np.asarray(hand["keypoints_2d"], dtype=float).tolist(),
           "confidence": [float(v) for v in np.asarray(hand.get("confidence"), dtype=float).reshape(-1)]}
    if hand.get("joints_cam") is not None:
        out["joints_cam"] = np.asarray(hand["joints_cam"], dtype=float).tolist()
    return out


def run_backend(info, backend_name="wilor", cache_path=None, backend=None, log=print):
    """两目各跑一次手部模型。结果缓存成 JSON，重跑时直接读。"""
    import convert_headcam as ch
    if cache_path is not None and Path(cache_path).is_file():
        data = json.loads(Path(cache_path).read_text(encoding="utf-8"))
        if data.get("backend") == backend_name:
            log("  复用缓存 %s" % cache_path)
            return data["views"], data.get("seconds")
    backend = backend or get_backend(backend_name)
    calib = info["calib"]
    views = {}
    t0 = time.time()
    for view, path in (("left", info["left_video"]), ("right", info["right_video"])):
        # 右目推理用右目内参；只用它的 2D，三维来自三角化
        view_calib = dict(calib, K_left=calib["K_right"]) if view == "right" else calib
        frames = []
        for index, image in enumerate(ch._iter_rgb(path)):
            pred = backend.predict(image, calib=view_calib)
            frames.append({side: _hand_to_json(pred.get(side)) for side in SIDES})
            if index % 50 == 0:
                log("  %s 目 第 %d 帧  %.1fs" % (view, index, time.time() - t0))
        views[view] = frames
    seconds = time.time() - t0
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(json.dumps({"backend": backend_name, "views": views, "seconds": seconds}),
                                    encoding="utf-8")
    return views, seconds


def run_backend_assoc(info, cache_dir, params, backend=None, log=print):
    """WiLoR 多候选 + 关联 + 另一目重新裁剪。候选和重裁结果分别缓存，中断后重跑直接读。返回 (views, seconds)。"""
    import convert_headcam as ch
    from headcam import hand_assoc as HA
    name = info["session"].name
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cand_path = cache_dir / ("%s.wilor_cands.json" % name)
    recrop_path = cache_dir / ("%s.wilor_recrop.json" % name)
    calib = info["calib"]
    wc = None
    seconds = 0.0

    def _wc():
        from headcam.wilor_candidates import WiLoRCandidates
        return WiLoRCandidates(backend=backend)

    if cand_path.is_file():
        log("  复用候选缓存 %s" % cand_path)
        cands = json.loads(cand_path.read_text(encoding="utf-8"))
    else:
        wc = _wc()
        cands, t0 = {}, time.time()
        for view, path in (("left", info["left_video"]), ("right", info["right_video"])):
            view_calib = dict(calib, K_left=calib["K_right"]) if view == "right" else calib
            cands[view] = [wc.frame_candidates(img, calib=view_calib) for img in ch._iter_rgb(path)]
            log("  %s 目候选完成 %.1fs" % (view, time.time() - t0))
        cands["seconds"] = time.time() - t0
        cand_path.write_text(json.dumps(cands), encoding="utf-8")
    seconds += float(cands.get("seconds") or 0.0)
    ap = HA.AssocParams(**params.assoc_params)
    cands = {v: [[c for c in fr if HA._score(c) >= ap.min_det_score] for fr in cands[v]] for v in SIDES}
    n = min(len(cands["left"]), len(cands["right"]))
    fps = float(info["metadata"]["fps"])
    ts = ch._read_timestamps(info["timestamps_path"], n, fps) if info["timestamps_path"] else [i / fps for i in range(n)]
    poses = [np.asarray(p, dtype=float) for p in associate_camera_poses(ts, info["slam_path"])[0]]
    views, chosen = HA.associate(cands, calib, poses, ap)
    if params.recrop:
        if recrop_path.is_file():
            extra = json.loads(recrop_path.read_text(encoding="utf-8"))
        else:
            HA.annotate_chosen_wrists(chosen, cands, calib, poses)
            reqs = HA.cross_view_requests(cands, chosen, calib, poses, ap)
            wc = wc or _wc()
            extra, t0 = {"items": [], "requests": len(reqs)}, time.time()
            by = {}
            for r in reqs:
                by.setdefault((r[0], r[1]), []).append(r)
            import cv2
            for view, path in (("left", info["left_video"]), ("right", info["right_video"])):
                view_calib = dict(calib, K_left=calib["K_right"]) if view == "right" else calib
                for t, img in enumerate(ch._iter_rgb(path)):
                    rs = by.get((t, view))
                    if not rs:
                        continue
                    bgr = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2BGR)
                    preds = wc.predict_boxes(bgr, [r[3] for r in rs], [1.0 if r[2] == "right" else 0.0 for r in rs],
                                             view_calib)
                    for r, pr in zip(rs, preds):
                        extra["items"].append({"t": t, "view": view, "cand": {
                            "box": r[3], "orig": None, "flip": None, "recrop": True, r[2]: pr}})
            extra["seconds"] = time.time() - t0
            recrop_path.write_text(json.dumps(extra), encoding="utf-8")
        seconds += float(extra.get("seconds") or 0.0)
        for e in extra["items"]:
            if e["t"] < n:
                cands[e["view"]][e["t"]].append(e["cand"])
        views, chosen = HA.associate(cands, calib, poses, ap)
    return views, seconds


# ---------------------------------------------------------------- 三角化 + 一致性

def reprojection_errors(points, uv_left, uv_right, calib):
    """每个关节在左右两目的平均重投影误差（像素）。不能算的关节是 NaN。"""
    left = project_pinhole(points, calib["K_left"])
    right = project_pinhole(points, calib["K_right"], calib["R"], calib["T"])
    err = 0.5 * (np.linalg.norm(left - uv_left, axis=1) + np.linalg.norm(right - uv_right, axis=1))
    err[~np.isfinite(points).all(axis=1)] = np.nan
    return err


def _umeyama(src, dst, weights, with_scale=True):
    w = weights / weights.sum()
    ms = (w[:, None] * src).sum(0)
    md = (w[:, None] * dst).sum(0)
    a, b = src - ms, dst - md
    u, sv, vt = np.linalg.svd((w[:, None] * b).T @ a)
    d = np.eye(3)
    d[2, 2] = np.sign(np.linalg.det(u @ vt))
    rot = u @ d @ vt
    var = float((w * (a ** 2).sum(1)).sum())
    scale = float((sv * np.diag(d)).sum() / var) if with_scale and var > 0 else 1.0
    return scale, rot, md - scale * rot @ ms


def rigid_fit_hand(mono, tri, weights, iters=5, huber_m=0.01):
    """把单目 21 点（相似变换：旋转+平移+尺度）稳健对齐到三角化关节（Huber IRLS）。
    返回 (对齐后的 21x3, 残差 RMS 米)；可用关节少于 6 个时返回 (None, None)。"""
    mono = np.asarray(mono, dtype=float)
    tri = np.asarray(tri, dtype=float)
    w0 = np.asarray(weights, dtype=float)
    ok = np.isfinite(tri).all(1) & np.isfinite(mono).all(1) & (w0 > 0)
    if ok.sum() < 6:
        return None, None
    w = w0.copy()
    for _ in range(iters):
        scale, rot, t = _umeyama(mono[ok], tri[ok], w[ok])
        fit = (scale * (rot @ mono.T)).T + t
        r = np.linalg.norm(fit - tri, axis=1)
        w = w0 * np.where(r < huber_m, 1.0, huber_m / np.maximum(r, 1e-9))
    return fit, float(np.sqrt(np.median(r[ok] ** 2)))


def velocity_gate(frames, points, gate_m, window=3, max_frame_gap=6):
    """离群点：离前后各 ``window`` 个邻居（帧号相差不超过 max_frame_gap）的中位数超过 gate_m 米。返回保留掩码。"""
    frames = np.asarray(frames)
    points = np.asarray(points, dtype=float)
    keep = np.ones(len(frames), dtype=bool)
    if gate_m <= 0 or len(frames) < 5:
        return keep
    for i in range(len(frames)):
        nb = [j for j in range(max(0, i - window), min(len(frames), i + window + 1))
              if j != i and abs(frames[j] - frames[i]) <= max_frame_gap]
        if len(nb) >= 2 and np.linalg.norm(points[i] - np.median(points[nb], 0)) > gate_m:
            keep[i] = False
    return keep


def rts_smooth(times, points, q=0.3, r=4e-4):
    """逐轴匀速模型 Kalman + RTS 反向平滑（离线、非因果）。times 秒，可以不等间隔。"""
    times = np.asarray(times, dtype=float)
    points = np.asarray(points, dtype=float)
    n = len(times)
    if n < 3:
        return points.copy()
    out = np.zeros_like(points)
    for ax in range(points.shape[1]):
        xs, ps, xp, pp = [], [], [], []
        x = np.array([points[0, ax], 0.0])
        p = np.diag([r, 1.0])
        for i in range(n):
            if i > 0:
                dt = max(times[i] - times[i - 1], 1e-6)
                f = np.array([[1.0, dt], [0.0, 1.0]])
                qm = q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
                x = f @ x
                p = f @ p @ f.T + qm
            xp.append(x.copy())
            pp.append(p.copy())
            k = p[:, 0] / (p[0, 0] + r)
            x = x + k * (points[i, ax] - x[0])
            p = p - np.outer(k, p[0, :])
            xs.append(x.copy())
            ps.append(p.copy())
        for i in range(n - 2, -1, -1):
            dt = max(times[i + 1] - times[i], 1e-6)
            f = np.array([[1.0, dt], [0.0, 1.0]])
            c = ps[i] @ f.T @ np.linalg.inv(pp[i + 1])
            xs[i] = xs[i] + c @ (xs[i + 1] - xp[i + 1])
            ps[i] = ps[i] + c @ (ps[i + 1] - pp[i + 1]) @ c.T
        out[:, ax] = [v[0] for v in xs]
    return out


def stereo_hand(hand_left_view, hand_right_view, calib, params):
    """一只手一帧。返回 dict：status、joints_cam（左相机系 21x3 或 None）、诊断量。

    status：none（两目都没认到）/ one_view / few_joints / reproj / depth / palm / ok。
    ``raw_joints`` 是一致性检查之前的三角化结果（评测“检查前”误差用）。
    """
    has_l = hand_left_view is not None
    has_r = hand_right_view is not None
    out = {"status": "none", "joints_cam": None, "raw_joints": None, "confidence": None,
           "reproj_px": None, "wrist_depth_m": None, "palm_m": None, "scale": None,
           "mono_joints": None}
    if has_l and hand_left_view.get("joints_cam") is not None:
        out["mono_joints"] = np.asarray(hand_left_view["joints_cam"], dtype=float)
    if not (has_l and has_r):
        out["status"] = "one_view" if (has_l or has_r) else "none"
        return out
    uv_l = np.asarray(hand_left_view["keypoints_2d"], dtype=float)
    uv_r = np.asarray(hand_right_view["keypoints_2d"], dtype=float)
    tri, _ = triangulate_pair(uv_l, uv_r, calib)
    err = reprojection_errors(tri, uv_l, uv_r, calib)
    use = np.isfinite(err) & (err <= params.joint_reproj_px)
    out["raw_joints"] = tri
    finite = np.isfinite(err)
    out["reproj_px"] = float(np.median(err[finite])) if finite.any() else None
    if np.isfinite(tri[0]).all():
        out["wrist_depth_m"] = float(tri[0, 2])
    if np.isfinite(tri[[0, MIDDLE_MCP]]).all():
        out["palm_m"] = float(np.linalg.norm(tri[MIDDLE_MCP] - tri[0]))
    # 关节：重投影合格的用三角化，其他的用 WiLoR 单目形状按三角化对齐尺度后补上
    mono = out["mono_joints"]
    if mono is not None and np.isfinite(mono).all():
        joints, scale = fuse_metric_joints(mono, tri, use.astype(float), threshold=0.5)
        out["scale"] = float(scale)
    else:
        joints = np.where(use[:, None], tri, np.nan)
    conf_l = np.asarray(hand_left_view.get("confidence") or [1.0] * JOINTS, dtype=float)
    conf_r = np.asarray(hand_right_view.get("confidence") or [1.0] * JOINTS, dtype=float)
    out["confidence"] = np.minimum(conf_l, conf_r)
    out["n_joints_ok"] = int(use.sum())
    status = "ok"
    if params.consistency:
        lo, hi = params.depth_range_m
        plo, phi = params.palm_range_m
        if out["n_joints_ok"] < params.min_joints:
            status = "few_joints"
        elif out["reproj_px"] is None or out["reproj_px"] > params.max_reproj_px:
            status = "reproj"
        elif out["wrist_depth_m"] is None or not (lo <= out["wrist_depth_m"] <= hi):
            status = "depth"
        elif out["palm_m"] is None or not (plo <= out["palm_m"] <= phi):
            status = "palm"
    elif not np.isfinite(joints[0]).all():
        status = "few_joints"
    if status == "ok" and params.wrist_mode == "rigid_fit" and out["mono_joints"] is not None:
        weights = out["confidence"] * np.exp(-0.5 * (np.nan_to_num(err, nan=99.0) / 5.0) ** 2)
        fit, residual = rigid_fit_hand(out["mono_joints"], tri, weights)
        if fit is not None:
            joints = joints.copy()
            joints[0] = fit[0]
            out["fit_residual_m"] = residual
    if status == "ok" and params.max_median_reproj_px is not None and out["reproj_px"] > params.max_median_reproj_px:
        status = "strict"
    if status == "ok" and params.max_offaxis_deg is not None and np.isfinite(joints[0]).all():
        if math.degrees(math.atan2(float(np.linalg.norm(joints[0, :2])), float(joints[0, 2]))) > params.max_offaxis_deg:
            status = "strict"
    out["status"] = status
    if status == "ok" and np.isfinite(joints[0]).all():
        out["joints_cam"] = joints
    elif status == "ok":
        out["status"] = "few_joints"
    return out


# ---------------------------------------------------------------- episode

def _temporal_wrist(joints, confs, diag, poses, timestamps, params):
    """世界系手腕：速度门限剔跳点（该手该帧作废，状态记 jump），再 RTS 平滑；整只手按手腕的平滑位移平移。原地修改。"""
    for side in SIDES:
        idx = [i for i in range(len(poses)) if np.isfinite(joints[side][i][0]).all()]
        if not idx:
            continue
        world = np.array([transform_points(joints[side][i][:1], poses[i])[0] for i in idx])
        keep = velocity_gate(idx, world, params.velocity_gate_m)
        for i, k in zip(idx, keep):
            if not k:
                joints[side][i] = np.nan
                confs[side][i] = np.nan
                diag[side][i]["status"] = "jump"
                diag[side][i]["joints_cam"] = None
        idx = [i for i, k in zip(idx, keep) if k]
        world = world[keep]
        if params.smooth != "rts" or len(idx) < 3:
            continue
        smoothed = rts_smooth([timestamps[i] for i in idx], world, params.rts_q, params.rts_r)
        for i, w_new in zip(idx, smoothed):
            pose = np.asarray(poses[i], dtype=float)
            delta_cam = pose[:3, :3].T @ (w_new - (pose[:3, :3] @ joints[side][i][0] + pose[:3, 3]))
            joints[side][i] = joints[side][i] + delta_cam


def build_stereo_episode(info, views, params):
    """返回 (episode, diagnostics)。diagnostics 里有逐帧逐手的检查前/后结果。"""
    import convert_headcam as ch
    from egodata.coverage import coarse_object_class, normalize_action
    metadata = info["metadata"]
    calib = info["calib"]
    num_frames = min(len(views["left"]), len(views["right"]))
    fps = float(metadata["fps"])
    timestamps = ch._read_timestamps(info["timestamps_path"], num_frames, fps) if info["timestamps_path"] else \
        [i / fps for i in range(num_frames)]
    poses, gaps = associate_camera_poses(timestamps, info["slam_path"])
    poses = [np.asarray(p, dtype=float) for p in poses]

    diag = {side: [] for side in SIDES}
    joints = {side: np.full((num_frames, JOINTS, 3), np.nan) for side in SIDES}
    confs = {side: np.full((num_frames, JOINTS), np.nan) for side in SIDES}
    for index in range(num_frames):
        for side in SIDES:
            result = stereo_hand(views["left"][index][side], views["right"][index][side], calib, params)
            diag[side].append(result)
            if result["joints_cam"] is not None:
                joints[side][index] = result["joints_cam"]
                confs[side][index] = result["confidence"]

    if params.smooth == "rts" or params.velocity_gate_m > 0:
        _temporal_wrist(joints, confs, diag, poses, timestamps, params)

    refine = params.refine()
    refine_info = None
    filled = {side: np.zeros(num_frames, dtype=bool) for side in SIDES}
    if refine is not None:
        refined = refine_hands(joints["left"], joints["right"], confs["left"], confs["right"],
                               timestamps, refine, camera_poses=np.stack(poses))
        for side in SIDES:
            joints[side] = np.asarray(refined[side]["joints"], dtype=float)
            confs[side] = np.asarray(refined[side]["confidence"], dtype=float)
            filled[side] = np.asarray(refined[side]["filled"], dtype=bool)
        refine_info = {"params": refine.to_dict(), "coordinate": refined["coordinate"],
                       "filled_frames": {s: int(filled[s].sum()) for s in SIDES}}

    hands = {}
    for side in SIDES:
        per_frame = [None if not np.isfinite(joints[side][i][0]).all() else joints[side][i] for i in range(num_frames)]
        conf_list = [confs[side][i] if per_frame[i] is not None else None for i in range(num_frames)]
        hands[side] = ch._pack_side(per_frame, conf_list, poses, filled=filled[side])

    session = info["session"]
    environment_raw = metadata.get("environment") or ""
    objects = [str(n) for n in metadata.get("objects") or []]
    verbs = [str(n) for n in metadata.get("verbs") or []]
    task_name = str(metadata.get("task") or session.name)
    statuses = {side: [d["status"] for d in diag[side]] for side in SIDES}
    episode = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": metadata["episode_id"],
        "source": "headcam_stereo",
        "source_path": str(info["rgb_video"] or info["left_video"]),
        "video_path": str(info["rgb_video"] or info["left_video"]),
        "fps": fps,
        "coordinate_frame": "slam_world" if info["slam_path"] is not None else "camera",
        "image_width": int(calib["image_width"]),
        "image_height": int(calib["image_height"]),
        "num_frames": num_frames,
        "timestamps": timestamps,
        "camera_intrinsic": np.asarray(calib["K_left"], dtype=float).tolist(),
        "camera_poses": [p.tolist() for p in poses],
        "hands": hands,
        "annotation": {
            "environment": {"name": normalize_environment(environment_raw), "detail": environment_raw,
                            "source": "metadata" if environment_raw else "missing"},
            "task": {"name": task_name, "instruction": metadata.get("instruction") or ""},
            "subtasks": [],
            "instructions": [],
        },
        "coverage": {
            "environment": normalize_environment(environment_raw),
            "objects": objects,
            "object_classes": sorted(set(coarse_object_class(n) for n in objects)),
            "task": task_name,
            "action_types": sorted(set(normalize_action(v) for v in verbs)),
        },
        "hand_pose": {
            "backend": "wilor_stereo",
            "stereo": True,
            "slam_time_gap_s": gaps,
            "imu_csv": None if info["imu_path"] is None else info["imu_path"].name,
            "imu_samples": ch._count_rows(info["imu_path"]),
            "calib": info["calib_path"].name,
            "refine": refine_info,
        },
        "stereo": {
            "params": params.to_dict(),
            "baseline_m": float(np.linalg.norm(np.asarray(calib["T"], dtype=float))),
            "per_frame": statuses,
            "reproj_px": {side: [d["reproj_px"] for d in diag[side]] for side in SIDES},
        },
    }
    errors = validate_episode(episode)
    if errors:
        raise ValueError("统一 episode 未通过校验：%s" % "; ".join(errors))
    return episode, {"diag": diag, "poses": poses, "filled": filled, "joints_cam_final": joints}


# ---------------------------------------------------------------- 真值评测（只在有 gt 时）

def _stats_cm(values):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float) * 100.0
    if len(v) == 0:
        return {"n": 0}
    return {"n": int(len(v)), "median_cm": float(np.median(v)), "p90_cm": float(np.percentile(v, 90)),
            "mean_cm": float(np.mean(v)), "within_2cm": float(np.mean(v <= 2.0))}


def evaluate_against_gt(info, episode, extra):
    """手腕世界坐标误差。分组：检查前（两目都认到）、通过检查、被拒（按原因）、最终（平滑后，测到的帧 / 补的帧）、
    以及同一批手的 WiLoR 单目手腕（参照）。"""
    gt = json.loads(Path(info["gt_path"]).read_text(encoding="utf-8"))["frames"]
    poses = extra["poses"]
    rows = []
    for side in SIDES:
        for index, d in enumerate(extra["diag"][side]):
            if index >= len(gt) or gt[index].get(side) is None:
                continue
            g = np.asarray(gt[index][side], dtype=float)[0]
            row = {"side": side, "frame": index, "status": d["status"]}
            if d["raw_joints"] is not None and np.isfinite(d["raw_joints"][0]).all():
                row["raw"] = float(np.linalg.norm(transform_points(d["raw_joints"][:1], poses[index])[0] - g))
            if d["mono_joints"] is not None and np.isfinite(d["mono_joints"][0]).all():
                row["mono"] = float(np.linalg.norm(transform_points(d["mono_joints"][:1], poses[index])[0] - g))
            if d["joints_cam"] is not None:
                row["checked"] = float(np.linalg.norm(transform_points(d["joints_cam"][:1], poses[index])[0] - g))
            fin = episode["hands"][side]["joints"][index][0]
            if fin is not None and fin[0] is not None:
                row["final"] = float(np.linalg.norm(np.asarray(fin, dtype=float) - g))
                row["filled"] = bool(extra["filled"][side][index])
            rows.append(row)
    return rows


def wrist_jitter_mm(points):
    """手腕世界坐标的二阶差分模长（毫米/帧²）的中位数。只用连续三帧都有值的地方。"""
    vals = []
    for i in range(1, len(points) - 1):
        a, b, c = points[i - 1], points[i], points[i + 1]
        if a is None or b is None or c is None:
            continue
        vals.append(float(np.linalg.norm(np.asarray(a) - 2 * np.asarray(b) + np.asarray(c))) * 1000)
    return {"n": len(vals), "median_mm": float(np.median(vals)) if vals else None}


def jitter_before_after(episode, extra):
    out = {}
    for side in SIDES:
        before = []
        for i, d in enumerate(extra["diag"][side]):
            before.append(None if d["joints_cam"] is None else
                          transform_points(d["joints_cam"][:1], extra["poses"][i])[0])
        after = []
        for i, j in enumerate(episode["hands"][side]["joints"]):
            w = j[0]
            ok = w is not None and w[0] is not None and not extra["filled"][side][i] and before[i] is not None
            after.append(np.asarray(w, dtype=float) if ok else None)
        before = [b if after[i] is not None else None for i, b in enumerate(before)]
        out[side] = {"before_smoothing": wrist_jitter_mm(before), "after_smoothing": wrist_jitter_mm(after)}
    return out


def summarize_eval(rows):
    raw = [r for r in rows if "raw" in r]
    out = {
        "gt_hand_frames": len(rows),
        "before_check": _stats_cm([r["raw"] for r in raw]),
        "mono_same_hands": _stats_cm([r["mono"] for r in raw if "mono" in r]),
        "after_check": _stats_cm([r["checked"] for r in rows if "checked" in r]),
        "after_check_raw_tri": _stats_cm([r["raw"] for r in raw if r["status"] == "ok"]),
        "rejected_by_reason": {},
        "final_measured": _stats_cm([r["final"] for r in rows if "final" in r and not r["filled"]]),
        "final_filled": _stats_cm([r["final"] for r in rows if "final" in r and r["filled"]]),
        "status_counts": {},
    }
    for r in rows:
        out["status_counts"][r["status"]] = out["status_counts"].get(r["status"], 0) + 1
    for reason in sorted(set(r["status"] for r in raw if r["status"] != "ok")):
        out["rejected_by_reason"][reason] = _stats_cm([r["raw"] for r in raw if r["status"] == reason])
    return out


# ---------------------------------------------------------------- 一条命令

def run_pipeline(sessions, out_dir, params=None, backend_name="wilor", repo_id="local/headcam_stereo",
                 export=True, backend=None, log=print, export_python=None, min_label_coverage=None):
    """跑完整条管线。返回报告字典，同时写 ``out_dir/report.json``。"""
    from egodata.qc import write_yield_reports, yield_report
    from egodata.stereo_qc import qc_stereo_episode

    params = params or StereoParams()
    out = Path(out_dir)
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    timing = {"backend_s": 0.0, "stereo_refine_s": 0.0, "qc_s": 0.0, "export_s": 0.0}
    results, evals, per_session = [], [], []
    all_eval_rows = []
    for session_dir in sessions:
        info = load_session(session_dir)
        name = info["session"].name
        log("[%s] 手部模型（%s）" % (name, backend_name))
        cache = out / "cache" / ("%s.%s.json" % (name, backend_name))
        if params.assoc and backend_name == "wilor":
            views, seconds = run_backend_assoc(info, out / "cache", params, backend=backend, log=log)
        else:
            views, seconds = run_backend(info, backend_name, cache_path=cache, backend=backend, log=log)
        timing["backend_s"] += float(seconds or 0.0)
        t0 = time.time()
        episode, extra = build_stereo_episode(info, views, params)
        timing["stereo_refine_s"] += time.time() - t0
        path = out / "episodes" / ("%s.json" % name)
        save_episode(episode, path)
        t0 = time.time()
        qc_overrides = {}
        if min_label_coverage is not None:
            qc_overrides["min_label_coverage"] = min_label_coverage
        qc = qc_stereo_episode(episode, **qc_overrides)
        timing["qc_s"] += time.time() - t0
        results.append({k: v for k, v in qc.items() if k != "good_frame_mask"})
        entry = {"session": str(info["session"]), "episode": str(path), "frames": episode["num_frames"],
                 "backend_seconds": seconds, "qc": results[-1], "jitter": jitter_before_after(episode, extra)}
        if info["gt_path"] is not None:
            rows = evaluate_against_gt(info, episode, extra)
            for r in rows:
                r["session"] = name
            all_eval_rows.extend(rows)
            entry["eval"] = summarize_eval(rows)
        per_session.append(entry)
        log("[%s] 帧 %d，QC %s，坏帧 %.0f%%，标注覆盖 %.0f%%" % (
            name, episode["num_frames"], "通过" if qc["accepted"] else "拒绝",
            100 * qc["bad_fraction"], 100 * qc["label_coverage"]))
    report = yield_report(results)
    html_path, csv_path = write_yield_reports(report, out / "qc" / "yield.html", out / "qc" / "yield.csv")
    reasons = {}
    only = {}
    for r in results:
        for k, v in r["flags"].items():
            reasons[k] = reasons.get(k, 0) + v
        for k, v in r["flags_only_reason"].items():
            only[k] = only.get(k, 0) + v
    dropped_names = ("stereo_one_view", "stereo_inconsistent", "stereo_filled")
    labeled_frames = sum(r.get("labeled_frames", r["num_frames"]) for r in results)
    dropped_frames = sum(r.get("dropped_label_frames", 0) for r in results)
    summary = {
        "params": params.to_dict(),
        "backend": backend_name,
        "yield": report["yield"],
        "raw_frames": report["raw_frames"],
        "usable_frames": report["usable_frames"],
        "episodes": report["episodes"],
        "accepted_episodes": report["accepted_episodes"],
        "bad_frames_by_reason": {k: v for k, v in reasons.items() if k not in dropped_names},
        "bad_frames_only_this_reason": {k: v for k, v in only.items() if k not in dropped_names},
        "dropped_labels_by_reason": {k: v for k, v in reasons.items() if k in dropped_names},
        "dropped_labels_only_this_reason": {k: v for k, v in only.items() if k in dropped_names},
        "labeled_frames": labeled_frames,
        "dropped_label_frames": dropped_frames,
        "label_coverage": 0.0 if report["raw_frames"] == 0 else labeled_frames / float(report["raw_frames"]),
        "min_label_coverage": results[0]["min_label_coverage"] if results else 0.0,
        "sessions": per_session,
        "timing": timing,
    }
    if all_eval_rows:
        summary["eval_all"] = summarize_eval(all_eval_rows)
    if export:
        t0 = time.time()
        if report["accepted_episodes"]:
            summary["lerobot"] = export_accepted(out / "episodes", csv_path, out / "lerobot", repo_id, export_python, log)
        else:
            summary["lerobot"] = {"dir": None, "result": "没有通过 QC 的片段，未导出"}
        timing["export_s"] = time.time() - t0
    (out / "report.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=1), encoding="utf-8")
    return summary


def export_accepted(episodes_dir, csv_path, lerobot_dir, repo_id, export_python=None, log=print):
    """导出 LeRobot v3.0。当前解释器装不了 pyarrow（例如 WiLoR 环境的 numpy 太老）时，
    用 ``export_python`` 指定的另一个 Python 跑 ``scripts/ego_to_lerobot.py``。"""
    import subprocess
    import sys
    if export_python:
        script = Path(__file__).resolve().parents[1] / "ego_to_lerobot.py"
        cmd = [export_python, str(script), "--episodes", str(episodes_dir), "--yield-csv", str(csv_path),
               "--out", str(lerobot_dir), "--repo-id", repo_id]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError("LeRobot 导出失败：%s" % proc.stderr[-2000:])
        return {"dir": str(lerobot_dir), "result": proc.stdout.strip().splitlines()[-1:] if proc.stdout else []}
    try:
        from egodata.lerobot_export import export_lerobot
    except ImportError as exc:
        raise RuntimeError("当前 Python 导入 pyarrow/pandas 失败（%s）。用 --export-python 指定一个装了 "
                           "pyarrow 的 Python，或之后单独跑 scripts/ego_to_lerobot.py" % exc)
    written = export_lerobot(episodes_dir, csv_path, lerobot_dir, repo_id=repo_id)
    return {"dir": str(lerobot_dir), "result": _jsonable(written), "python": sys.executable}


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
