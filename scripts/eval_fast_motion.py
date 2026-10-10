"""快速手部运动下的双目硬件要求评测（HOT3D 真值轨迹 + 实测 WiLoR 二维误差）。

【仿真】把 HOT3D 真值手的轨迹在时间轴上压缩 k 倍（速度×k，加速度×k²），放进虚拟双目
（默认 OV9281：1280x800、100°、基线 9 cm、全局快门），按帧率采样，逐项加入：
  - 左右目同步偏差 sync_ms（右目晚曝光）
  - 卷帘快门读出时间 readout_ms（每行曝光时刻不同；0=全局快门）
  - 曝光时间 exposure_ms：运动模糊 → 额外二维误差（系数由 blur 子命令在真实图像上实测）
  - 帧率 fps：影响速度门限和 RTS 平滑
三角化后跑管线里真实的 velocity_gate + rts_smooth，按瞬时手腕速度分档统计误差。

用法：
  python scripts/eval_fast_motion.py blur --clips ... --out blur.json        # 实测：模糊对 WiLoR 的影响
  python scripts/eval_fast_motion.py simulate --preds preds.npy --clips ... --blur blur.json --out sim_fast.json
  python scripts/eval_fast_motion.py measured --preds preds_fast.npy --out measured_fast.json  # 实测：真实快片段按速度分档
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_hot3d_stereo as E  # noqa: E402

SPEED_BINS = [(0.0, 0.3, "≈0.15 m/s（慢）"), (0.3, 0.75, "≈0.5 m/s"), (0.75, 1.5, "≈1 m/s（快）"), (1.5, 3.0, "≈2 m/s（很快）")]


# ---------------------------------------------------------------- 轨迹

def segments(trajs):
    """把真值轨迹切成连续段：(t[秒], J[n,21,3]) ，相机系。"""
    out = []
    for tr in trajs:
        cur_t, cur_j = [], []
        for t, j in zip(tr["t"] + [None], tr["J"] + [None]):
            if j is None or t is None:
                if len(cur_t) >= 8:
                    out.append((np.array(cur_t), np.stack(cur_j)))
                cur_t, cur_j = [], []
            else:
                cur_t.append(t); cur_j.append(j)
    return out


def sample(seg_t, seg_j, tau):
    """压缩后时间 tau（秒，从 0 开始）上的插值关节；越界返回 None。tau 已经乘过 k。"""
    t = seg_t[0] + tau
    if t < seg_t[0] or t > seg_t[-1]:
        return None
    i = int(np.clip(np.searchsorted(seg_t, t) - 1, 0, len(seg_t) - 2))
    w = (t - seg_t[i]) / (seg_t[i + 1] - seg_t[i])
    return seg_j[i] * (1 - w) + seg_j[i + 1] * w


def linear_gate(times, points, gate_m, window=3, degree=1):
    """建议的新门限：用前后邻居按时间做直线拟合（匀速预测），残差超过 gate_m 才剔。
    与旧的“邻居中位数”相比，不会把匀速/边缘快速运动当成跳点。"""
    times = np.asarray(times); points = np.asarray(points)
    keep = np.ones(len(times), bool)
    if gate_m <= 0 or len(times) < 5:
        return keep
    for i in range(len(times)):
        nb = [j for j in range(max(0, i - window), min(len(times), i + window + 1)) if j != i]
        dtn = times[nb] - times[i]
        A = np.stack([dtn ** d for d in range(degree + 1)], 1)
        coef, *_ = np.linalg.lstsq(A, points[nb], rcond=None)
        if np.linalg.norm(points[i] - coef[0]) > gate_m:
            keep[i] = False
    return keep


# ---------------------------------------------------------------- 仿真

def simulate(segs, residuals, f_meas, blur_coef, k=1.0, fps=30.0, baseline=0.09, hfov_deg=100.0,
             width=1280, height=800, sync_ms=0.0, readout_ms=0.0, exposure_ms=2.0, alpha=1.0,
             gate="median", gate_m=0.02, smooth=True, rts_q=0.3, seed=0):
    from headcam.stereo_pipeline import velocity_gate, rts_smooth
    rng = np.random.default_rng(seed)
    f = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
    K = np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]])
    P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = K @ np.hstack([np.eye(3), np.array([[-baseline], [0], [0]])])
    off = np.array([baseline, 0, 0])
    gain = f_meas * (f / f_meas) ** alpha

    def proj(X):
        u = (K @ X.T).T
        return u[:, :2] / u[:, 2:]

    rows = []
    for seg_t, seg_j in segs:
        dur = (seg_t[-1] - seg_t[0]) / k
        n = int(dur * fps)
        if n < 6:
            continue
        times, raw, gt, speed, ok_cov = [], [], [], [], []
        for m in range(n):
            tau = m / fps
            J = sample(seg_t, seg_j, tau * k)
            Jn = sample(seg_t, seg_j, (tau + 1e-3) * k)
            if J is None or Jn is None:
                continue
            v = (Jn - J) / 1e-3  # 相机系关节速度
            uv0, uv1 = proj(J), proj(J - off)
            inside = J[0, 2] > 0.1 and all(0 <= u[0] < width and 0 <= u[1] < height for u in (uv0[0], uv1[0]))
            obs = []
            for view, uv, shift in ((0, uv0, 0.0), (1, uv1, sync_ms)):
                dt = (shift + readout_ms * np.clip(uv[:, 1], 0, height) / height) * 1e-3  # 每个关节自己的行
                Xs = J + v * dt[:, None]  # 1~30 ms 内线性
                o = proj(Xs - (off if view else 0))
                # 运动模糊：模糊长度（像素）= 图像速度×曝光；实测系数给出额外二维误差（沿运动方向）
                o_next = proj(Xs + v * exposure_ms * 1e-3 - (off if view else 0))
                L = np.linalg.norm(o_next - o, axis=1)
                obs.append((o, o_next - o, L))
            e0, e1 = residuals[rng.integers(len(residuals))]
            n0 = obs[0][0] + e0 * gain
            n1 = obs[1][0] + e1 * gain
            if blur_coef > 0:
                # 两目模糊方向/长度几乎相同，但 WiLoR 在两张图上的“偏到哪儿”不同 → 独立随机
                for nn, (o, d, L) in ((n0, obs[0]), (n1, obs[1])):
                    dirn = d / np.maximum(L[:, None], 1e-9)
                    nn += dirn * (rng.standard_normal(21) * blur_coef * L)[:, None]
            X = np.array([E.triangulate_world([P0, P1], [n0[j], n1[j]]) for j in range(21)])
            times.append(tau); raw.append(X[0]); gt.append(J[0]); speed.append(np.linalg.norm(v[0])); ok_cov.append(inside)
        if len(times) < 6:
            continue
        times = np.array(times); raw = np.array(raw); gt = np.array(gt); speed = np.array(speed); cov = np.array(ok_cov)
        idx = np.where(cov)[0]
        if len(idx) < 3:
            continue
        frames = np.round(times[idx] * fps).astype(int)
        if gate == "median":
            keep = velocity_gate(frames, raw[idx], gate_m, mode="median")
        elif gate == "linear":
            keep = linear_gate(times[idx], raw[idx], gate_m)
        elif gate == "quad":
            keep = linear_gate(times[idx], raw[idx], gate_m, degree=2)
        else:
            keep = np.ones(len(idx), bool)
        sm = raw[idx].copy()
        if smooth and keep.sum() >= 3:
            sm[keep] = rts_smooth(times[idx][keep], raw[idx][keep], q=rts_q)
        for a, i in enumerate(idx):
            rows.append({"speed": float(speed[i]), "raw": float(np.linalg.norm(raw[i] - gt[i])),
                         "final": float(np.linalg.norm(sm[a] - gt[i])), "kept": bool(keep[a])})
    return rows


def binned(rows):
    out = []
    for lo, hi, name in SPEED_BINS:
        r = [x for x in rows if lo <= x["speed"] < hi]
        if len(r) < 20:
            out.append({"bin": name, "n": len(r)}); continue
        kept = [x["final"] for x in r if x["kept"]]
        raw = np.array([x["raw"] for x in r])
        d = {"bin": name, "n": len(r), "kept_frac": len(kept) / len(r),
             "raw_median_cm": float(np.median(raw) * 100), "raw_p90_cm": float(np.percentile(raw, 90) * 100)}
        if len(kept) >= 10:
            kept = np.array(kept)
            d.update(final_median_cm=float(np.median(kept) * 100), final_p90_cm=float(np.percentile(kept, 90) * 100),
                     final_within_2cm=float(np.mean(kept < 0.02)))
        out.append(d)
    return out


def cmd_simulate(a):
    records = list(np.load(a.preds, allow_pickle=True))
    f_meas = float(records[0].get("f") or E.load_frame_cameras(records[0]["clip"], records[0]["key"])[E.STREAMS[0]].f[0])
    res, _ = E.paired_residuals(records, f_meas)
    blur = json.load(open(a.blur)) if a.blur else {"coef": 0.0}
    coef = float(blur.get("coef", 0.0))
    trajs = E.gt_trajectories(a.clips, os.environ["MANO_MODEL_DIR"])
    segs = segments(trajs)
    base = dict(fps=30.0, baseline=0.09, hfov_deg=100.0, sync_ms=0.0, readout_ms=0.0, exposure_ms=2.0,
                gate="median", gate_m=0.02)
    ks = [1, 3, 6, 10]
    plan = [("OV9281 基准（全局、硬件同步、30fps、曝光2ms、现管线：中位数门限2cm + RTS q=0.3）", {})]
    plan.append(("只三角化，不剔点不平滑（硬件本身的精度）", dict(gate="none", smooth=False)))
    for s_ in (0.1, 1, 5, 10, 30):
        plan.append(("同步偏差 %g ms" % s_, dict(sync_ms=s_, gate="none", smooth=False)))
    for r in (10, 20, 30):
        plan.append(("卷帘快门 读出 %d ms" % r, dict(readout_ms=r, gate="none", smooth=False)))
    for e in (1, 2, 5, 10):
        plan.append(("曝光 %g ms" % e, dict(exposure_ms=e, gate="none", smooth=False)))
    for fps in (30, 60, 90, 120):
        plan.append(("%d fps 现管线（门限2cm/帧 + q=0.3）" % fps, dict(fps=fps)))
    for fps in (30, 60, 120):
        for q in (30.0, 300.0):
            plan.append(("%d fps 新管线（匀加速预测门限2cm + RTS q=%g）" % (fps, q), dict(fps=fps, gate="quad", rts_q=q)))
            plan.append(("%d fps 只平滑不剔点（RTS q=%g）" % (fps, q), dict(fps=fps, gate="none", rts_q=q)))
    plan.append(("反例：两颗独立卷帘 USB 30fps 软同步 曝光10ms（只三角化）", dict(readout_ms=25, sync_ms=15, exposure_ms=10, gate="none", smooth=False)))
    out = {"blur_coef": coef, "n_residual_pairs": len(res), "k": ks, "n_segments": len(segs), "runs": []}
    for label, kw in plan:
        cfg = dict(base, **kw)
        rows = []
        for k in ks:
            rows += simulate(segs, res, f_meas, coef, k=k, seed=k, **cfg)
        b = binned(rows)
        out["runs"].append({"label": label, "config": cfg, "bins": b})
        print(label, " | ".join("%s kept%.0f%% med%.2f p90%.2f" % (x["bin"][:6], 100 * x.get("kept_frac", 0),
              x.get("final_median_cm", -1), x.get("final_p90_cm", -1)) for x in b), flush=True)
        json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)


# ---------------------------------------------------------------- 实测：模糊

def motion_blur(img, length, angle_deg):
    import cv2
    if length < 1:
        return img
    L = int(np.ceil(length)) | 1
    ker = np.zeros((L, L), np.float32)
    ker[L // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((L / 2 - 0.5, L / 2 - 0.5), angle_deg, 1.0)
    ker = cv2.warpAffine(ker, M, (L, L))
    ker /= max(ker.sum(), 1e-6)
    return cv2.filter2D(img, -1, ker)


def cmd_blur(a):
    """真实图像加线性运动模糊（随机方向），重跑 WiLoR，测二维关键点误差随模糊长度的变化。"""
    from headcam.hand_pose import get_backend
    import imageio.v2 as imageio
    backend = get_backend("wilor")
    gt = E.GTHands(os.environ["MANO_MODEL_DIR"])
    rng = np.random.default_rng(0)
    lengths = [0, 4, 8, 16, 32]
    rows = []
    for clip in a.clips:
        keys = sorted(os.path.basename(p).split(".")[0] for p in glob.glob(os.path.join(clip, "*.hands.json")))[::a.stride][: a.max_frames]
        for key in keys:
            cams = E.load_frame_cameras(clip, key)
            s = E.STREAMS[0]
            pin = E.pinhole_of(cams[s], 1.0, -90)
            img = imageio.imread(os.path.join(clip, "%s.image_%s.jpg" % (key, s)))
            if img.ndim == 2:
                img = np.stack([img] * 3, -1)
            und = E.undistort(img, cams[s], pin)
            gtj, _ = gt.joints(clip, key)
            ang = float(rng.uniform(0, 180))
            for L in lengths:
                pred = backend.predict(motion_blur(und, L, ang), {"K_left": E.K_of(pin)})
                for side, jw in gtj.items():
                    Tc = np.linalg.inv(pin.T_world_from_eye)
                    g = (Tc[:3, :3] @ jw.T).T + Tc[:3, 3]
                    if g[0, 2] < 0.05:
                        continue
                    guv = (E.K_of(pin) @ g.T).T; guv = guv[:, :2] / guv[:, 2:]
                    p = pred[side]
                    det = p["keypoints_2d"] is not None
                    err = float(np.median(np.linalg.norm(np.asarray(p["keypoints_2d"]) - guv, axis=1))) if det else None
                    rows.append({"key": key, "clip": os.path.basename(clip), "side": side, "L": L, "det": det, "px": err})
            print(clip[-6:], key, flush=True)
    # 只用 L=0 能认到且误差<60px 的手做配对
    base = {(r["clip"], r["key"], r["side"]): r["px"] for r in rows if r["L"] == 0 and r["det"] and r["px"] < 60}
    summ = []
    for L in lengths:
        v = [(r["px"], base[(r["clip"], r["key"], r["side"])]) for r in rows if r["L"] == L and (r["clip"], r["key"], r["side"]) in base]
        dets = [r["px"] is not None and r["px"] < 60 for r in rows if r["L"] == L and (r["clip"], r["key"], r["side"]) in base]
        good = [(x, b) for x, b in v if x is not None and x < 60]
        extra = [np.sqrt(max(x * x - b * b, 0)) for x, b in good]
        summ.append({"L_px": L, "n": len(dets), "det_rate": float(np.mean(dets)),
                     "median_px": float(np.median([x for x, _ in good])), "extra_rms_px": float(np.sqrt(np.mean(np.square(extra))))})
    # 额外误差 ≈ coef × L（过原点最小二乘，只用 L>0）
    Ls = np.array([s["L_px"] for s in summ if s["L_px"] > 0]); ex = np.array([s["extra_rms_px"] for s in summ if s["L_px"] > 0])
    coef = float((Ls @ ex) / (Ls @ Ls))
    out = {"note": "实测：HOT3D Quest3 图像（f≈505px）加随机方向线性模糊后重跑 WiLoR", "summary": summ, "coef": coef, "rows": rows}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(summ, indent=1), "coef", coef)


# ---------------------------------------------------------------- 实测：真实快片段

def cmd_measured(a):
    """真实 WiLoR 双目（逐帧）结果，按真值手腕速度分档。速度用相邻帧真值（相机系）算。"""
    records = list(np.load(a.preds, allow_pickle=True))
    rows = E.evaluate(records)
    by = {}
    for r in records:
        by.setdefault((r["clip"], r["side"]), []).append(r)
    speed = {}
    for key, rs in by.items():
        rs.sort(key=lambda r: r["key"])
        for i, r in enumerate(rs):
            nb = [x for x in (rs[i - 1] if i else None, rs[i + 1] if i + 1 < len(rs) else None)
                  if x is not None and abs(int(x["key"]) - int(r["key"])) == 1]
            if nb:
                x = nb[0]
                dt = abs(int(x["key"]) - int(r["key"])) / 30.0
                # 世界系真值速度（HOT3D 片段 30fps）
                speed[(r["clip"], r["key"], r["side"])] = float(np.linalg.norm(x["gt_world"][0] - r["gt_world"][0]) / dt)
    out = []
    for lo, hi, name in SPEED_BINS:
        e = [row["stereo_wrist"] for row, r in zip(rows, records)
             if "stereo_wrist" in row and row["vis0"] and row["vis1"]
             and lo <= speed.get((r["clip"], r["key"], r["side"]), -1) < hi]
        e = np.array(e)
        d = {"bin": name, "n": int(len(e))}
        if len(e) >= 5:
            d.update(median_cm=float(np.median(e) * 100), p90_cm=float(np.percentile(e, 90) * 100), within_2cm=float(np.mean(e < 0.02)))
        out.append(d)
    det = [row for row in rows if row["vis0"] and row["vis1"]]
    res = {"note": "实测：真实图像逐帧 WiLoR 左右目三角化（未做门限/平滑），速度用世界系真值", "bins": out,
           "hands_visible_both": len(det), "both_detected": float(np.mean([("stereo_wrist" in r) for r in det]))}
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simulate"); s.add_argument("--preds", required=True); s.add_argument("--clips", nargs="+", required=True)
    s.add_argument("--blur"); s.add_argument("--out", required=True)
    b = sub.add_parser("blur"); b.add_argument("--clips", nargs="+", required=True); b.add_argument("--stride", type=int, default=10)
    b.add_argument("--max-frames", type=int, default=15); b.add_argument("--out", required=True)
    m = sub.add_parser("measured"); m.add_argument("--preds", required=True); m.add_argument("--out", required=True)
    a = ap.parse_args()
    {"simulate": cmd_simulate, "blur": cmd_blur, "measured": cmd_measured}[a.cmd](a)


if __name__ == "__main__":
    main()
