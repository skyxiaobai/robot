#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C15 hypothesis test (UNVERIFIED): how noisy are wrist labels?

Step 1 measures, per episode/hand, in the camera frame XHEY trains in:
  (a) static/low-motion 1 s windows: wrist drift (detrended RMS, range) --
      "true" motion of a window = ||median(last3)-median(first3)|| of the raw wrist;
  (b) frame-to-frame jump distribution (median/p90/p99, cm) vs the 1 s
      displacement distribution on moving windows (the signal ADE is scored on);
  (c) the QC jitter metric (fps-normalised 2nd difference, same as qc_scorecard).

Step 2 (--depth-session + --episode) replaces wrist z by the median LiDAR
depth in a small window around the projected wrist (Record3D depth.npz),
rebuilds the wrist on the same camera ray (hand shape translated, not
scaled), re-measures the same metrics, and can write corrected episodes.

This is a measurement tool; it makes no claim that noise explains XHEY
results. See docs/xhey/c15_wrist_noise.md.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from egodata.hand_fix import SIDES, _good  # noqa: E402


def load_wrist(ep):
    """episode -> {side: (ok, W_cam_m)} in XHEY camera frame when possible."""
    try:
        from common.frames import canonicalize_for_xhey
        ep = canonicalize_for_xhey(ep)
    except Exception as e:  # noqa: BLE001 - measurement should not die on one clip
        print("canonicalize failed (%s); using stored wrist" % e, flush=True)
    n = int(ep["num_frames"]); out = {}
    for side in SIDES:
        ok, W = _good(ep["hands"][side], n, 0.5)
        out[side] = (ok, W)
    return ep, out


def series_metrics(W, ok, fps):
    """All step-1 numbers for one hand series. Lengths in cm."""
    idx = np.arange(len(W))
    both = ok[1:] & ok[:-1] & (idx[1:] - idx[:-1] == 1)
    jump = np.linalg.norm(np.diff(W, axis=0), axis=1)[both[1:] if False else np.where(both)[0] - 0] if False else None
    # frame-to-frame jumps on adjacent ok frames
    adj = np.where(ok[1:] & ok[:-1])[0]
    jump = np.linalg.norm(W[adj + 1] - W[adj], axis=1) * 100 if len(adj) else np.array([])
    tri = np.where(ok[2:] & ok[1:-1] & ok[:-2])[0]
    resid = np.linalg.norm(W[tri + 2] - 2 * W[tri + 1] + W[tri], axis=1) * (15.0 / fps) ** 2 * 100 if len(tri) else np.array([])
    # 1 s windows, half-overlap
    win = max(3, int(round(fps))); hop = max(1, win // 2); lag = win
    rows = []
    for a in range(0, len(W) - win + 1, hop):
        m = ok[a:a + win]
        if m.mean() < 0.8:
            continue
        seg = W[a:a + win][m]
        if len(seg) < 5:
            continue
        true_disp = float(np.linalg.norm(np.median(seg[-3:], 0) - np.median(seg[:3], 0)) * 100)
        t = np.arange(len(seg), dtype=float); t -= t.mean()
        A = np.vstack([np.ones_like(t), t]).T
        pred, *_ = np.linalg.lstsq(A, seg, rcond=None)
        detr = seg - A @ pred
        rows.append({"true_disp_cm": true_disp,
                     "drift_rms_cm": float(np.sqrt((detr ** 2).sum(1).mean()) * 100),
                     "range_cm": float(np.linalg.norm(seg.max(0) - seg.min(0)) * 100 if False else np.sqrt(((seg.max(0) - seg.min(0)) ** 2).sum()) * 100),
                     "jump_med_cm": float(np.median(np.linalg.norm(np.diff(seg, axis=0), axis=1)) * 100) if len(seg) > 1 else 0.0})
    # 1 s displacement (signal magnitude)
    cand = np.where(ok[lag:] & ok[:-lag])[0]
    disp = np.linalg.norm(W[cand + lag] - W[cand], axis=1) * 100 if len(cand) else np.array([])
    def q(a):
        return {"n": int(len(a)), "median": float(np.median(a)) if len(a) else None,
                "p90": float(np.percentile(a, 90)) if len(a) else None,
                "p99": float(np.percentile(a, 99)) if len(a) else None}
    static = [r for r in rows if r["true_disp_cm"] < 1.0]
    moving = [r for r in rows if r["true_disp_cm"] >= 5.0]
    return {"fps": float(fps), "frames": int(len(W)), "ok_frac": float(ok.mean()),
            "jump_cm": q(jump), "jitter_cm_median": float(np.median(resid)) if len(resid) else None,
            "disp_1s_cm": q(disp),
            "disp_1s_moving_cm": q(disp[disp >= 5.0]) if len(disp) else q(disp),
            "windows": {"n": len(rows), "static_n": len(static), "moving_n": len(moving),
                        "static_drift_rms_cm_median": float(np.median([r["drift_rms_cm"] for r in static])) if static else None,
                        "static_range_cm_median": float(np.median([r["range_cm"] for r in static])) if static else None,
                        "static_jump_med_cm_median": float(np.median([r["jump_med_cm"] for r in static])) if static else None,
                        "moving_jump_med_cm_median": float(np.median([r["jump_med_cm"] for r in moving])) if moving else None},
            "_jump": jump, "_disp": disp, "_rows": rows}


def summarise(per_hand):
    """Pool jumps/displacements across hands; medians of per-hand window stats."""
    jump = np.concatenate([h.pop("_jump") for h in per_hand]) if per_hand else np.array([])
    disp = np.concatenate([h.pop("_disp") for h in per_hand]) if per_hand else np.array([])
    for h in per_hand:
        h.pop("_rows", None)
    def q(a):
        return {"n": int(len(a)), "median": float(np.median(a)) if len(a) else None,
                "p90": float(np.percentile(a, 90)) if len(a) else None,
                "p99": float(np.percentile(a, 99)) if len(a) else None}
    st = [h["windows"]["static_drift_rms_cm_median"] for h in per_hand if h["windows"]["static_drift_rms_cm_median"] is not None]
    sr = [h["windows"]["static_range_cm_median"] for h in per_hand if h["windows"]["static_range_cm_median"] is not None]
    jit = [h["jitter_cm_median"] for h in per_hand if h["jitter_cm_median"] is not None]
    return {"jump_cm": q(jump), "disp_1s_cm": q(disp), "disp_1s_ge5cm": q(disp[disp >= 5.0]) if len(disp) else q(disp),
            "jump_median_over_disp1s_median": float(np.median(jump) / np.median(disp)) if len(jump) and len(disp) and np.median(disp) else None,
            "static_drift_rms_cm_median_of_hands": float(np.median(st)) if st else None,
            "static_range_cm_median_of_hands": float(np.median(sr)) if sr else None,
            "jitter_cm_median_of_hands": float(np.median(jit)) if jit else None,
            "hands": per_hand}, jump, disp


def correct_wrist_depth(W, ok, K, depth, conf, radius=2, min_conf=1, offset_m=0.01):
    """Replace z by median depth in (2r+1)^2 window at projected wrist; rebuild on ray."""
    n = len(W); dh, dw = depth.shape[1], depth.shape[2]
    # K is for RGB size; caller scales K to depth size beforehand
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    out = W.copy(); znew = np.full(n, np.nan); used = np.zeros(n, bool)
    for i in range(n):
        if not ok[i] or not np.isfinite(W[i]).all() or W[i, 2] <= 0.05:
            continue
        u = int(round(fx * W[i, 0] / W[i, 2] + cx)); v = int(round(fy * W[i, 1] / W[i, 2] + cy))
        a, b = max(0, v - radius), min(dh, v + radius + 1); c, d = max(0, u - radius), min(dw, u + radius + 1)
        if b <= a or d <= c:
            continue
        patch = np.asarray(depth[i, a:b, c:d], float); cf = conf[i, a:b, c:d]
        vals = patch[(patch > 0.2) & (patch < 2.5) & (cf >= min_conf)]
        if len(vals) < 3:
            continue
        z = float(np.median(vals)) + offset_m; znew[i] = z; used[i] = True
        out[i] = W[i] * (z / W[i, 2])
    return out, znew, used


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", nargs="+", required=True, help="episode JSON files or dirs")
    ap.add_argument("--label", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth-session", default=None, help="dir with depth.npz+calib.yaml (step 2, one episode)")
    ap.add_argument("--write-corrected", default=None, help="dir to write depth-corrected episode JSON")
    a = ap.parse_args()
    paths = []
    for p in a.episodes:
        p = Path(p)
        paths += sorted(p.glob("*.json")) if p.is_dir() else [p]
    paths = [p for p in paths if p.name not in ("index.json", "manifest.json", "convert_summary.json")]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    result = {"label": a.label, "episodes": []}
    pool_hands = []
    examples = {}
    for p in paths:
        ep = json.loads(p.read_text())
        ep_c, wrists = load_wrist(ep)
        fps = float(ep.get("fps") or 15.0)
        rec = {"clip": p.stem, "source": ep.get("source"), "fps": fps, "hands": {}}
        for side in SIDES:
            ok, W = wrists[side]
            if ok.sum() < 30:
                continue
            m = series_metrics(np.where(ok[:, None], W, 0.0), ok, fps)
            rec["hands"][side] = {k: v for k, v in m.items() if not k.startswith("_")}
            pool_hands.append(m)
            if "static_example" not in examples:
                # first static window trajectory for chart
                for r_i, r in enumerate(m["_rows"]):
                    if r["true_disp_cm"] < 1.0:
                        win = max(3, int(round(fps))); a0 = r_i * max(1, win // 2)
                        examples["static_example"] = {"clip": p.stem, "side": side, "start": int(a0),
                                                      "wrist_cm": (W[a0:a0 + win * 3] * 100).round(3).tolist(),
                                                      "ok": ok[a0:a0 + win * 3].tolist()}
                        break
        result["episodes"].append(rec)
        print(p.stem, json.dumps(rec["hands"], ensure_ascii=False)[:400], flush=True)
        # ---- step 2 for this episode if a matching depth session was given
        if a.depth_session:
            sess = Path(a.depth_session)
            if p.stem not in str(sess) and sess.name != p.stem:
                print("depth session %s does not match %s, skip step2" % (sess, p.stem))
                continue
            data = np.load(sess / "depth.npz"); depth, conf = data["depth"], data["conf"]
            calib = json.loads((sess / "calib.yaml").read_text())
            K_rgb = np.asarray(calib["K_rgb"], float); Wimg, Himg = calib["image_width"], calib["image_height"]
            dh, dw = depth.shape[1], depth.shape[2]
            K = K_rgb.copy(); K[0] *= dw / Wimg; K[1] *= dh / Himg
            step2 = {}
            corrected = json.loads(json.dumps(ep_c))  # camera-frame copy
            for side in SIDES:
                ok, W = wrists[side]
                W2, znew, used = correct_wrist_depth(W, ok, K, depth, conf)
                before = series_metrics(np.where(ok[:, None], W, 0.0), ok, fps)
                after = series_metrics(np.where(ok[:, None], W2, 0.0), ok, fps)
                dz = np.abs(znew - W[:, 2])[used] * 100
                step2[side] = {"corrected_frac": float(used.mean()),
                               "abs_dz_cm": {"median": float(np.median(dz)) if len(dz) else None,
                                             "p90": float(np.percentile(dz, 90)) if len(dz) else None},
                               "before": {k: before[k] for k in ("jump_cm", "jitter_cm_median", "windows")},
                               "after": {k: after[k] for k in ("jump_cm", "jitter_cm_median", "windows")}}
                if a.write_corrected:
                    delta = np.where(used[:, None], W2 - W, 0.0)
                    h = corrected["hands"][side]
                    for i in range(len(W)):
                        if used[i] and h["wrist_pose"][i]:
                            h["wrist_pose"][i] = [float(x) for x in W2[i]] + list(h["wrist_pose"][i][3:])
                            if h["joints"][i] is not None:
                                h["joints"][i] = (np.asarray(h["joints"][i], float) + delta[i]).tolist()
            rec["step2_depth_correction"] = step2
            print("STEP2", p.stem, json.dumps(step2, ensure_ascii=False)[:600], flush=True)
            if a.write_corrected:
                wd = Path(a.write_corrected); wd.mkdir(parents=True, exist_ok=True)
                (wd / p.name).write_text(json.dumps(corrected))
    pooled, jump, disp = summarise(pool_hands)
    result["pooled"] = pooled
    (out / "noise.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    if examples:
        (out / "examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=1))
    # charts
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
        if len(jump): ax[0].hist(jump, bins=np.linspace(0, 8, 41), color="#c44"); ax[0].set_title("frame jump cm (median %.2f p90 %.2f)" % (np.median(jump), np.percentile(jump, 90)))
        if len(disp): ax[1].hist(disp, bins=np.linspace(0, 40, 41), color="#48c"); ax[1].set_title("1s displacement cm (median %.2f)" % np.median(disp))
        for x in ax: x.set_xlabel("cm")
        fig.suptitle("C15 wrist noise: %s (hypothesis test)" % a.label); fig.tight_layout(); fig.savefig(out / "noise_hist.png", dpi=130); plt.close(fig)
        if "static_example" in examples:
            ex = examples["static_example"]; W = np.array(ex["wrist_cm"], float)
            fig, ax = plt.subplots(figsize=(7, 3))
            for k, c in enumerate("rgb"): ax.plot(W[:, k], label="xyz"[k], color=c)
            ax.set_title("static window example %s %s (cm)" % (ex["clip"], ex["side"])); ax.legend(); fig.tight_layout(); fig.savefig(out / "static_traj.png", dpi=130); plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print("chart failed", e)
    print(json.dumps(pooled, ensure_ascii=False)[:1200], flush=True)


if __name__ == "__main__":
    main()
