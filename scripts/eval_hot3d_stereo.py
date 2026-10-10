"""HOT3D 双目手腕测距评测：WiLoR 左右目 2D 关键点 → 三角化 → 与 MANO 真值比较。

数据：HuggingFace bop-benchmark/hot3d 的 HOT3D-Clips（train_quest3 / train_aria，tar 解压后的目录）。
依赖：hand_tracking_toolkit（相机模型、MANO 真值），smplx，WiLoR（见 headcam/hand_pose.py）。

用法：
  PYTHONPATH=/path/WiLoR:scripts python scripts/eval_hot3d_stereo.py run \
      --clips data/train_quest3/clip-000000 ... --stride 3 --out out/preds.npz
  python scripts/eval_hot3d_stereo.py report --preds out/preds.npz --out out/
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

STREAMS = ("1201-1", "1201-2")  # SLAM 左 / 右（Quest3 与 Aria 都是这两个 id）
TIP_VERTS = [745, 317, 444, 556, 673]
MANO_TO_OPENPOSE = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]


def load_frame_cameras(clip, key):
    from hand_tracking_toolkit import camera
    raw = json.load(open(os.path.join(clip, "%s.cameras.json" % key)))
    return {s: camera.from_json(raw[s]) for s in STREAMS}


def pinhole_of(cam, focal_scale=1.0, rotate_deg=0, res_scale=1.0):
    """鱼眼 → 针孔虚拟相机。rotate_deg=-90 把 Quest3 侧放的 SLAM 图转正（WiLoR 对竖直手更稳）。"""
    from hand_tracking_toolkit import camera
    k = focal_scale * res_scale
    f = [cam.f[0] * k, cam.f[1] * k]
    W, H = int(round(cam.width * res_scale)), int(round(cam.height * res_scale))
    c = (cam.c[0] * res_scale, cam.c[1] * res_scale)
    if rotate_deg == 0:
        return camera.PinholePlaneCameraModel(
            width=W, height=H, f=f, c=c,
            distort_coeffs=[], T_world_from_eye=cam.T_world_from_eye)
    t = np.radians(rotate_deg)
    Rz = np.eye(4)
    Rz[:2, :2] = [[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]
    return camera.PinholePlaneCameraModel(
        width=H, height=W, f=f, c=(c[1], c[0]),
        distort_coeffs=[], T_world_from_eye=cam.T_world_from_eye @ Rz)


def undistort(image, src, dst):
    from hand_tracking_toolkit.dataset import warp_image
    return warp_image(src, dst, image)


def K_of(pin):
    return np.array([[pin.f[0], 0, pin.c[0]], [0, pin.f[1], pin.c[1]], [0, 0, 1.0]])


class GTHands(object):
    """HOT3D MANO 真值 → OpenPose 顺序 21 点（世界坐标，米），和 WiLoR 关节定义一致。"""

    def __init__(self, mano_dir):
        import smplx
        import torch
        from hand_tracking_toolkit.hand_models.mano_hand_model import MANOHandModel
        self.model = MANOHandModel(mano_dir)
        self.J = smplx.MANO(mano_dir, is_rhand=True, use_pca=False).J_regressor.numpy()
        self.torch = torch

    def joints(self, clip, key):
        from hand_tracking_toolkit.dataset import decode_hand_pose
        from hand_tracking_toolkit.hand_models.mano_hand_model import forward_kinematics
        beta = self.torch.tensor(json.load(open(os.path.join(clip, "__hand_shapes.json__")))["mano"])
        hands = json.load(open(os.path.join(clip, "%s.hands.json" % key)))
        out = {}
        for side, pose in decode_hand_pose(hands).items():
            if pose.mano is None:
                continue
            _, verts, _ = forward_kinematics(pose.mano, beta, self.model)
            v = np.asarray(verts, dtype=np.float64)
            j = np.concatenate([self.J @ v, v[TIP_VERTS]], 0)[MANO_TO_OPENPOSE]
            out[side.value] = j
        return out, hands


def triangulate_world(P_list, uv_list):
    """多视 DLT，P 为 3x4（世界→像素）。"""
    A = []
    for P, (u, v) in zip(P_list, uv_list):
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    _, _, vt = np.linalg.svd(np.asarray(A))
    X = vt[-1]
    return X[:3] / X[3]


def proj_matrix(pin):
    T = np.linalg.inv(pin.T_world_from_eye)
    return K_of(pin) @ T[:3, :]


def cmd_run(args):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from headcam.hand_pose import get_backend
    import imageio.v2 as imageio
    backend = get_backend("wilor")
    gt = GTHands(os.environ["MANO_MODEL_DIR"])
    rec = []
    t0 = time.time()
    for clip in args.clips:
        keys = sorted(os.path.basename(p).split(".")[0] for p in glob.glob(os.path.join(clip, "*.hands.json")))
        keys = keys[::args.stride][: args.max_frames]
        device = json.load(open(os.path.join(clip, keys[0] + ".info.json")))["device"]
        for key in keys:
            cams = load_frame_cameras(clip, key)
            gtj, hands = gt.joints(clip, key)
            pins, preds = {}, {}
            for s in STREAMS:
                img = imageio.imread(os.path.join(clip, "%s.image_%s.jpg" % (key, s)))
                if img.ndim == 2:
                    img = np.stack([img] * 3, -1)
                pins[s] = pinhole_of(cams[s], args.focal_scale, -90 if device == "Quest3" else 0, args.res_scale)
                und = undistort(img, cams[s], pins[s])
                preds[s] = backend.predict(und, {"K_left": K_of(pins[s])})
            for side, jw in gtj.items():
                r = {"clip": clip, "key": key, "device": device, "side": side, "gt_world": jw,
                     "wh": (pins[STREAMS[0]].width, pins[STREAMS[0]].height), "f": float(pins[STREAMS[0]].f[0])}
                for i, s in enumerate(STREAMS):
                    p = preds[s][side]
                    Tc = np.linalg.inv(pins[s].T_world_from_eye)
                    gt_cam = (Tc[:3, :3] @ jw.T).T + Tc[:3, 3]
                    gt_uv = (K_of(pins[s]) @ gt_cam.T).T
                    r["gt_uv%d" % i] = gt_uv[:, :2] / gt_uv[:, 2:3]
                    r["gt_cam%d" % i] = gt_cam
                    r["vis%d" % i] = float(hands[side]["visibilities_modeled"].get(s, 0.0))
                    r["P%d" % i] = proj_matrix(pins[s])
                    r["T_world_cam%d" % i] = pins[s].T_world_from_eye
                    if p["keypoints_2d"] is not None:
                        r["uv%d" % i] = np.asarray(p["keypoints_2d"], dtype=np.float64)
                        r["mono_cam%d" % i] = np.asarray(p["joints_cam"], dtype=np.float64)
                rec.append(r)
            print("%s %s  %.1fs" % (os.path.basename(clip), key, time.time() - t0), flush=True)
    np.save(args.out, np.array(rec, dtype=object), allow_pickle=True)
    print("saved", args.out, len(rec))


# ---------------------------------------------------------------- 评测

def _visible(r, i, width_height):
    uv = r["gt_uv%d" % i][0]
    w, h = width_height
    return bool(r["vis%d" % i] >= 0.5) and bool( r["gt_cam%d" % i][0, 2] > 0.05 and 0 <= uv[0] < w and 0 <= uv[1] < h)


def _stats(x):
    x = np.asarray([v for v in x if np.isfinite(v)], dtype=np.float64) * 100.0
    if len(x) == 0:
        return {"n": 0}
    return {"n": int(len(x)), "median": float(np.median(x)), "mean": float(np.mean(x)), "p90": float(np.percentile(x, 90))}


def evaluate(records, width_height=(1024, 1280)):
    """返回逐手记录的误差（米）和汇总。width_height 是转正后的针孔图尺寸。"""
    rows = []
    for r in records:
        row = {"device": r["device"], "clip": os.path.basename(r["clip"]), "key": r["key"], "side": r["side"],
               "depth": float(r["gt_cam0"][0, 2])}
        for i in (0, 1):
            row["vis%d" % i] = _visible(r, i, r.get("wh", width_height))
            row["det%d" % i] = "uv%d" % i in r
            if row["det%d" % i]:
                d2 = np.linalg.norm(r["uv%d" % i] - r["gt_uv%d" % i], axis=1)
                row["px2d%d" % i] = float(np.median(d2))
                row["px2d_wrist%d" % i] = float(d2[0])
                m, g = r["mono_cam%d" % i], r["gt_cam%d" % i]
                row["mono_wrist%d" % i] = float(np.linalg.norm(m[0] - g[0]))
                row["mono_mpjpe%d" % i] = float(np.mean(np.linalg.norm(m - g, axis=1)))
                row["mono_rr%d" % i] = float(np.mean(np.linalg.norm((m - m[0]) - (g - g[0]), axis=1)))
        if row["det0"] and row["det1"]:
            X = np.array([triangulate_world([r["P0"], r["P1"]], [r["uv0"][j], r["uv1"][j]]) for j in range(21)])
            e = np.linalg.norm(X - r["gt_world"], axis=1)
            row["stereo_wrist"] = float(e[0])
            row["stereo_mpjpe"] = float(np.mean(e))
            # 双目误差沿视线（深度）与横向分解，在左目坐标系
            Tc = np.linalg.inv(r["T_world_cam0"])
            xc = Tc[:3, :3] @ X[0] + Tc[:3, 3]
            row["stereo_wrist_depth_err"] = float(abs(xc[2] - r["gt_cam0"][0, 2]))
            # 只用双目给“尺度”，手型仍用单目（headcam 现有 correct_monocular_with_stereo 思路）
            m = r["mono_cam0"]
            row["stereo_scaled_mono_wrist"] = float(np.linalg.norm(m[0] * (xc[2] / m[0, 2]) - r["gt_cam0"][0]))
        rows.append(row)
    return rows


def summarize(rows):
    both = [r for r in rows if r["vis0"] and r["vis1"]]
    out = {"hands_total": len(rows), "hands_visible_both": len(both)}
    for i in (0, 1):
        vis = [r for r in rows if r["vis%d" % i]]
        out["det_rate%d" % i] = sum(r["det%d" % i] for r in vis) / max(1, len(vis))
        out["px2d%d" % i] = float(np.median([r["px2d%d" % i] for r in vis if r["det%d" % i]]))
        out["mono_wrist%d" % i] = _stats([r["mono_wrist%d" % i] for r in vis if r["det%d" % i]])
        out["mono_mpjpe%d" % i] = _stats([r["mono_mpjpe%d" % i] for r in vis if r["det%d" % i]])
        out["mono_rr%d" % i] = _stats([r["mono_rr%d" % i] for r in vis if r["det%d" % i]])
    out["det_both"] = sum(r["det0"] and r["det1"] for r in both) / max(1, len(both))
    tri = [r for r in both if "stereo_wrist" in r]
    out["mono_wrist0_on_stereo_set"] = _stats([r["mono_wrist0"] for r in tri])
    out["stereo_wrist"] = _stats([r["stereo_wrist"] for r in tri])
    out["stereo_wrist_depth_err"] = _stats([r["stereo_wrist_depth_err"] for r in tri])
    out["stereo_mpjpe"] = _stats([r["stereo_mpjpe"] for r in tri])
    out["stereo_scaled_mono_wrist"] = _stats([r["stereo_scaled_mono_wrist"] for r in tri])
    out["within_2cm_stereo"] = float(np.mean([r["stereo_wrist"] < 0.02 for r in tri])) if tri else None
    out["within_2cm_mono"] = float(np.mean([r["mono_wrist0"] < 0.02 for r in tri])) if tri else None
    bins = [(0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 9)]
    out["by_depth"] = []
    for lo, hi in bins:
        sel = [r for r in tri if lo <= r["depth"] < hi]
        out["by_depth"].append({"bin": [lo, hi], "stereo": _stats([r["stereo_wrist"] for r in sel]),
                                "mono": _stats([r["mono_wrist0"] for r in sel])})
    return out


def cmd_report(args):
    records = list(np.load(args.preds, allow_pickle=True))
    wh = (1024, 1280) if records[0]["device"] == "Quest3" else (640, 480)
    rows = evaluate(records, wh)
    summary = summarize(rows)
    json.dump({"summary": summary, "rows": rows}, open(args.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))


# ---------------------------------------------------------------- 仿真

def gt_trajectories(clips, mano_dir):
    """每个 clip 全部帧的真值关节，表达在“当帧左目”坐标系（含头动），以及时间戳（秒）。"""
    gt = GTHands(mano_dir)
    trajs = []
    for clip in clips:
        keys = sorted(os.path.basename(p).split(".")[0] for p in glob.glob(os.path.join(clip, "*.hands.json")))
        for side in ("left", "right"):
            ts, js = [], []
            for key in keys:
                joints, _ = gt.joints(clip, key)
                if side not in joints:
                    ts.append(None); js.append(None); continue
                cam = load_frame_cameras(clip, key)[STREAMS[0]]
                pin = pinhole_of(cam, 1.0, -90)
                T = np.linalg.inv(pin.T_world_from_eye)
                js.append((T[:3, :3] @ joints[side].T).T + T[:3, 3])
                info = json.load(open(os.path.join(clip, key + ".info.json")))
                ts.append(info["image_timestamps_ns"][STREAMS[0]] * 1e-9)
            trajs.append({"clip": clip, "side": side, "t": ts, "J": js})
    return trajs


def paired_residuals(records, f_meas, outlier_px=60.0):
    """WiLoR 真实二维残差（左右目成对，保留两目误差相关性），单位：弧度（像素/焦距）。

    返回 (成对残差列表, 被剔除的错配比例)。>outlier_px 的是左右手认错/认到别处，
    这部分由检测率与 QC 单独统计，不混进“精度”。"""
    res, bad = [], 0
    for r in records:
        if "uv0" in r and "uv1" in r:
            e0 = (r["uv0"] - r["gt_uv0"]) / f_meas
            e1 = (r["uv1"] - r["gt_uv1"]) / f_meas
            if max(np.median(np.linalg.norm(e0, axis=1)), np.median(np.linalg.norm(e1, axis=1))) * f_meas < outlier_px:
                res.append((e0, e1))
            else:
                bad += 1
    return res, bad / max(1, bad + len(res))


def _interp(J, t, k, dt):
    """手在 t[k]+dt 时刻的位置（相邻帧线性插值；越界返回 None）。"""
    if dt == 0:
        return J[k]
    k2 = k + 1 if dt > 0 else k - 1
    if k2 < 0 or k2 >= len(J) or J[k2] is None:
        return None
    w = dt / (t[k2] - t[k])
    return J[k] + (J[k2] - J[k]) * w


def _ci(values, fn, n=300, seed=1):
    """自助法 95% 置信区间。"""
    v = np.asarray(values)
    if len(v) < 5:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    bs = [fn(v[rng.integers(len(v), size=len(v))]) for _ in range(n)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def simulate(trajs, residuals, f_meas, baseline=0.06, hfov_deg=75.0, width=1280, height=800,
             sync_ms=0.0, readout_ms=0.0, alpha=1.0, noise_scale=1.0, reps=3, seed=0):
    """把 HOT3D 真值手放进一个虚拟平行双目，加“实测 WiLoR 二维误差”，三角化，算三维误差。

    - 二维误差模型：像素误差 ∝ f^alpha（alpha 由真实图像在两种分辨率下实测拟合；
      alpha=1 误差随分辨率等比放大=分辨率没用，alpha=0 像素误差固定=分辨率越高越准）。
    - sync_ms：右目比左目晚曝光多少毫秒。readout_ms：卷帘快门整帧读出时间（0=全局快门），
      每一行的曝光时刻 = 帧时刻 + readout*行号/高度。
    - 覆盖率：真值手腕在左右两目画面内的比例（相机光轴沿用 Quest3 左 SLAM 相机朝向）。
    """
    rng = np.random.default_rng(seed)
    f = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
    K = np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]])
    P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = K @ np.hstack([np.eye(3), np.array([[-baseline], [0], [0]])])
    gain = f_meas * (f / f_meas) ** alpha * noise_scale  # 弧度残差 → 本相机像素
    off = np.array([baseline, 0, 0])

    def proj(X):
        u = (K @ X.T).T
        return u[:, :2] / u[:, 2:]

    def inside(uv):
        return 0 <= uv[0] < width and 0 <= uv[1] < height

    wrist, mpjpe, total, covered = [], [], 0, 0
    for tr in trajs:
        t, J = tr["t"], tr["J"]
        for k in range(len(J)):
            if J[k] is None:
                continue
            total += 1
            uv0, uv1 = proj(J[k]), proj(J[k] - off)
            if J[k][0, 2] < 0.1 or not (inside(uv0[0]) and inside(uv1[0])):
                continue
            covered += 1
            # 每个关节按自己所在行的曝光时刻取位置（卷帘），右目再加同步偏差
            obs = []
            ok = True
            for view, uv, shift in ((0, uv0, 0.0), (1, uv1, sync_ms)):
                pts = np.empty((21, 3))
                for j in range(21):
                    dt = (shift + readout_ms * np.clip(uv[j, 1], 0, height) / height) * 1e-3
                    X = _interp(J, t, k, dt)
                    if X is None:
                        ok = False
                        break
                    pts[j] = X[j]
                if not ok:
                    break
                obs.append(proj(pts - (off if view else 0)))
            if not ok:
                continue
            # 真值取左目“第 0 行”时刻（即帧时间戳），与采集系统打的时间戳一致
            for _ in range(reps):
                e0, e1 = residuals[rng.integers(len(residuals))]
                n0, n1 = obs[0] + e0 * gain, obs[1] + e1 * gain
                X = np.array([triangulate_world([P0, P1], [n0[j], n1[j]]) for j in range(21)])
                err = np.linalg.norm(X - J[k], axis=1)
                wrist.append(err[0]); mpjpe.append(err.mean())
    w = np.asarray(wrist)
    return {
        "f_px": float(f), "coverage": covered / max(1, total), "n_hand_frames": covered,
        "wrist": _stats(w), "mpjpe": _stats(mpjpe),
        "wrist_median_ci": [x * 100 for x in _ci(w, np.median)],
        "wrist_p90_ci": [x * 100 for x in _ci(w, lambda v: np.percentile(v, 90))],
        "within_2cm": float(np.mean(w < 0.02)) if len(w) else None,
        "within_2cm_ci": _ci(w, lambda v: float(np.mean(v < 0.02))),
    }


def hand_speed_offsets(trajs, offsets_ms):
    """RGB 与双目时间差导致的“标签错位”：手腕在 dt 内移动的距离（真值，相机系，含头动）。"""
    out = {}
    for ms in offsets_ms:
        d = []
        for tr in trajs:
            for k in range(len(tr["J"])):
                if tr["J"][k] is None:
                    continue
                X = _interp(tr["J"], tr["t"], k, ms * 1e-3)
                if X is not None:
                    d.append(np.linalg.norm(X[0] - tr["J"][k][0]))
        out[ms] = _stats(d)
        out[ms]["within_2cm"] = float(np.mean(np.asarray(d) < 0.02))
    return out


def fit_alpha(full_records, low_records, res_scale, f_full):
    """真实图像：同一批帧在全分辨率和降采样 res_scale 下的 WiLoR 二维误差 → 像素误差 ∝ f^alpha。"""
    def med(records):
        v = []
        for r in records:
            k = r.get("f", f_full) / f_full
            for i in (0, 1):
                if "uv%d" % i in r:
                    d = np.median(np.linalg.norm(r["uv%d" % i] - r["gt_uv%d" % i], axis=1))
                    if d < 60 * k:
                        v.append(d)
        return float(np.median(v)), len(v)
    keys = set((r["clip"], r["key"], r["side"]) for r in low_records)
    full = [r for r in full_records if (r["clip"], r["key"], r["side"]) in keys]
    (m_full, n_full), (m_low, n_low) = med(full), med(low_records)
    alpha = float(np.log(m_full / m_low) / np.log(1.0 / res_scale))
    return {"px_full": m_full, "n_full": n_full, "px_low": m_low, "n_low": n_low,
            "res_scale": res_scale, "alpha": alpha}


def sim_plan():
    """候选硬件配置。所有分辨率按 16:10 / 4:3 实际传感器尺寸。"""
    plan = []
    R = {"640x400": (640, 400), "1280x800 (OV9281)": (1280, 800), "1600x1200 (AR0234)": (1600, 1200)}
    base = dict(baseline=0.06, hfov_deg=75, width=1280, height=800, sync_ms=0, readout_ms=0)
    for b in (0.04, 0.06, 0.08, 0.10):
        plan.append(("基线", "%d cm" % round(b * 100), dict(base, baseline=b)))
    for fov in (75, 100, 120):
        plan.append(("视场角", "%d°" % fov, dict(base, hfov_deg=fov)))
    for name, (w, h) in R.items():
        for fov in (75, 100):
            plan.append(("分辨率", "%s @%d°" % (name, fov), dict(base, width=w, height=h, hfov_deg=fov)))
    for ms in (0, 1, 10, 30):
        plan.append(("双目同步偏差", "%d ms" % ms, dict(base, sync_ms=ms)))
    # 快门：卷帘读出时间约等于帧周期的 ~70%（典型值，假设）
    for fps in (30, 60):
        plan.append(("快门/帧率", "全局快门 %dfps" % fps, dict(base)))
        plan.append(("快门/帧率", "卷帘快门 %dfps（读出≈%.0fms，两目同步）" % (fps, 0.7e3 / fps), dict(base, readout_ms=0.7e3 / fps)))
        plan.append(("快门/帧率", "两目不同步 %dfps（平均偏差≈%.0fms）" % (fps, 0.25e3 / fps), dict(base, sync_ms=0.25e3 / fps)))
    plan.append(("方案", "BOM 方案A/B：OV9281 75° 1280x800 全局快门 硬件同步 6cm", dict(base)))
    plan.append(("方案", "同上但基线 8cm", dict(base, baseline=0.08)))
    plan.append(("方案", "两颗独立 USB 卷帘摄像头 1080p 90° 软件同步（反例）", dict(base, width=1920, height=1080, hfov_deg=90, readout_ms=25, sync_ms=15)))
    return plan


def cmd_simulate(args):
    records = list(np.load(args.preds, allow_pickle=True))
    f_meas = float(records[0].get("f") or load_frame_cameras(records[0]["clip"], records[0]["key"])[STREAMS[0]].f[0])
    res, outlier = paired_residuals(records, f_meas)
    out = {"f_meas": f_meas, "n_residual_pairs": len(res), "outlier_fraction": outlier}
    if args.low_preds:
        out["alpha_fit"] = fit_alpha(records, list(np.load(args.low_preds, allow_pickle=True)), args.low_scale, f_meas)
    alpha = out.get("alpha_fit", {}).get("alpha", 1.0) if args.alpha is None else args.alpha
    alpha = float(np.clip(alpha, 0.0, 1.0))
    out["alpha_used"] = alpha
    clips = sorted(set(r["clip"] for r in records))
    trajs = gt_trajectories(clips, os.environ["MANO_MODEL_DIR"])
    out["rgb_offset"] = hand_speed_offsets(trajs, (0, 10, 17, 33))
    # 校验：用 Quest3 自身几何仿真，与真实三角化结果对比
    W, H = records[0].get("wh", (1024, 1280))
    out["validate"] = simulate(trajs, res, f_meas, baseline=float(args.real_baseline),
                               hfov_deg=2 * np.degrees(np.arctan(W / 2.0 / f_meas)), width=W, height=H, alpha=alpha)
    print("validate", json.dumps(out["validate"]["wrist"]), flush=True)
    out["runs"] = []
    for group, label, kw in sim_plan():
        rows = {}
        for a_name, a in (("model", alpha), ("alpha0", 0.0), ("alpha1", 1.0)):
            if a_name != "model" and group not in ("分辨率", "视场角"):
                continue
            rows[a_name] = simulate(trajs, res, f_meas, alpha=a, **kw)
        r = dict(rows["model"], group=group, label=label, config=kw)
        if "alpha0" in rows:
            r["bound_alpha0_wrist_median"] = rows["alpha0"]["wrist"]["median"]
            r["bound_alpha1_wrist_median"] = rows["alpha1"]["wrist"]["median"]
        out["runs"].append(r)
        print(group, label, "median %.2f p90 %.2f <2cm %.0f%% cov %.0f%%" % (
            r["wrist"]["median"], r["wrist"]["p90"], 100 * r["within_2cm"], 100 * r["coverage"]), flush=True)
    json.dump(out, open(args.out, "w"), indent=1, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run")
    r.add_argument("--clips", nargs="+", required=True)
    r.add_argument("--stride", type=int, default=3)
    r.add_argument("--max-frames", type=int, default=50)
    r.add_argument("--focal-scale", type=float, default=1.0)
    r.add_argument("--res-scale", type=float, default=1.0, help="降采样，用来标定 WiLoR 二维误差随分辨率的变化")
    r.add_argument("--out", required=True)
    rp = sub.add_parser("report")
    rp.add_argument("--preds", required=True)
    rp.add_argument("--out", required=True)
    sp = sub.add_parser("simulate")
    sp.add_argument("--preds", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--low-preds", help="同一批帧降采样后的 WiLoR 结果（run --res-scale）")
    sp.add_argument("--low-scale", type=float, default=0.5)
    sp.add_argument("--alpha", type=float, default=None)
    sp.add_argument("--real-baseline", type=float, default=0.0637)
    a = ap.parse_args()
    if a.cmd == "simulate":
        cmd_simulate(a)
    if a.cmd == "report":
        cmd_report(a)
    if a.cmd == "run":
        cmd_run(a)


if __name__ == "__main__":
    main()
