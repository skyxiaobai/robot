#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""在 DexYCB 真实 RGB-D（RealSense D415，对齐到彩色的深度）+ 3D 手部真值上测激光雷达深度手部求解的精度。

三种深度条件：
- ``realsense``：DexYCB 原始深度 640x480（实测：真实传感器深度）。
- ``iphone_sim``：把同一张深度降到 256x192（iPhone 激光雷达深度分辨率），高斯模糊 1 px（模拟 ARKit 边缘平滑/飞点），
  加 σ = 1 cm × (z / 1 m) 的高斯噪声；置信度：3x3 邻域深度跨度 < 3 cm 记高(2)，否则中(1)。（仿真）
- ``iphone_sim_2x``：同上，噪声 σ 翻倍（2 cm @ 1 m）。（仿真）
- ``iphone_sim_2x_conf1``：同 2x 噪声，但中置信度像素也用（min_conf=1）。（仿真）

WiLoR 在彩色图上只跑一次（缓存），三个条件共用同一份 2D/单目 3D。
指标：相机系手腕误差（cm）中位数 / p90 / ≤2 cm 比例，21 点平均误差，覆盖率（通过检查的手 / 有真值的手）。
参照：同一批手的 WiLoR 单目手腕（相机系，用真实 K 的公制平移）。

用法::

    source /workspace/stereo_env.sh
    python scripts/iphone/eval_rgbd_dexycb.py --root data/dexycb --subject 20200709-subject-01 \
        --serials 932122060861 836212060125 841412060263 --stride 2 --out outputs/iphone_eval
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from headcam.rgbd_pipeline import RGBDParams, depth_hand  # noqa: E402


def simulate_iphone(depth_m, rng, noise_at_1m=0.01, size=(256, 192), blur_px=1.0):
    import cv2
    valid = (depth_m > 0).astype(np.float32)
    d = cv2.resize(depth_m * valid, size, interpolation=cv2.INTER_AREA)
    v = cv2.resize(valid, size, interpolation=cv2.INTER_AREA)
    d = np.where(v > 0.5, d / np.maximum(v, 1e-6), 0.0).astype(np.float32)
    if blur_px > 0:
        m = (d > 0).astype(np.float32)
        num = cv2.GaussianBlur(d, (0, 0), blur_px)
        den = cv2.GaussianBlur(m, (0, 0), blur_px)
        d = np.where(m > 0, num / np.maximum(den, 1e-6), 0.0).astype(np.float32)
    d = np.where(d > 0, d + rng.normal(0.0, 1.0, d.shape).astype(np.float32) * noise_at_1m * d, 0.0)
    mx = cv2.dilate(d, np.ones((3, 3), np.uint8))
    mn = -cv2.dilate(-np.where(d > 0, d, 99.0), np.ones((3, 3), np.uint8))
    conf = np.where(d <= 0, 0, np.where(mx - mn < 0.03, 2, 1)).astype(np.uint8)
    return d, conf


def stats(errs):
    v = np.asarray([e for e in errs if e is not None and np.isfinite(e)]) * 100
    if len(v) == 0:
        return {"n": 0}
    return {"n": int(len(v)), "median_cm": float(np.median(v)), "p90_cm": float(np.percentile(v, 90)),
            "mean_cm": float(v.mean()), "within_2cm": float((v <= 2).mean())}


def load_K(root, serial):
    import yaml
    text = (Path(root) / "calibration" / "intrinsics" / ("%s_640x480.yml" % serial)).read_text()
    c = yaml.safe_load(text.split("extrinsics:")[0])["color"]
    return np.array([[c["fx"], 0, c["ppx"]], [0, c["fy"], c["ppy"]], [0, 0, 1.0]])


def main(argv=None):
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--serials", nargs="+", required=True)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / "wilor_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
    backend = None
    samples = []
    t0 = time.time()
    for seq in sorted((Path(a.root) / a.subject).iterdir()):
        if not seq.is_dir():
            continue
        for serial in a.serials:
            cam = seq / serial
            if not cam.is_dir():
                continue
            K = load_K(a.root, serial)
            for lab in sorted(cam.glob("labels_*.npz"))[::a.stride]:
                idx = lab.stem.split("_")[1]
                gt = np.load(lab)["joint_3d"][0]
                if not np.isfinite(gt).all() or np.abs(gt).sum() == 0 or gt[0, 2] <= 0:
                    continue
                key = "%s/%s/%s" % (seq.name, serial, idx)
                if key not in cache:
                    if backend is None:
                        import torch
                        torch.set_num_threads(a.threads)
                        from headcam.hand_pose import WiLoRBackend
                        backend = WiLoRBackend()
                    img = cv2.cvtColor(cv2.imread(str(cam / ("color_%s.jpg" % idx))), cv2.COLOR_BGR2RGB)
                    pred = backend.predict(img, calib={"K_left": K})
                    cache[key] = {}
                    for side in ("left", "right"):
                        h = pred.get(side)
                        if isinstance(h, dict) and h.get("keypoints_2d") is not None and h.get("joints_cam") is not None:
                            cache[key][side] = {"keypoints_2d": np.asarray(h["keypoints_2d"]).tolist(),
                                                "joints_cam": np.asarray(h["joints_cam"]).tolist(),
                                                "confidence": np.asarray(h.get("confidence"), dtype=float).reshape(-1).tolist()}
                    if len(cache) % 25 == 0:
                        cache_path.write_text(json.dumps(cache))
                        print("WiLoR %d 帧  %.0fs" % (len(cache), time.time() - t0), flush=True)
                samples.append((key, cam, idx, K, gt))
    cache_path.write_text(json.dumps(cache))
    rng = np.random.default_rng(a.seed)
    conds = ("realsense", "iphone_sim", "iphone_sim_2x", "iphone_sim_2x_conf1")
    rows = []
    params = {"realsense": RGBDParams(min_conf=2, patch=2), "iphone_sim": RGBDParams(), "iphone_sim_2x": RGBDParams(), "iphone_sim_2x_conf1": RGBDParams(min_conf=1)}
    for key, cam, idx, K, gt in samples:
        gt2d = gt @ K.T
        gt2d = gt2d[:, :2] / gt2d[:, 2:]
        hand, best = None, 1e9
        for side, h in cache[key].items():
            dpx = np.linalg.norm(np.asarray(h["keypoints_2d"])[0] - gt2d[0])
            if dpx < best:
                hand, best = h, dpx
        row = {"key": key, "detected": bool(hand is not None and best < 80)}
        if row["detected"]:
            mono = np.asarray(hand["joints_cam"])
            row["mono_wrist"] = float(np.linalg.norm(mono[0] - gt[0]))
            row["mono_mpjpe"] = float(np.linalg.norm(mono - gt, axis=1).mean())
            raw = cv2.imread(str(cam / ("aligned_depth_to_color_%s.png" % idx)), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
            for c in conds:
                if c == "realsense":
                    d, conf = raw, np.where(raw > 0, 2, 0).astype(np.uint8)
                else:
                    d, conf = simulate_iphone(raw, rng, noise_at_1m=0.01 if c == "iphone_sim" else 0.02)
                r = depth_hand(hand, d, conf, (640, 480), K, params[c])
                row[c] = {"status": r["status"]}
                if r["raw_joints"] is not None and np.isfinite(r["raw_joints"][0]).all():
                    row[c]["raw_wrist"] = float(np.linalg.norm(r["raw_joints"][0] - gt[0]))
                if r["joints_cam"] is not None:
                    row[c]["wrist"] = float(np.linalg.norm(r["joints_cam"][0] - gt[0]))
                    row[c]["mpjpe"] = float(np.linalg.norm(r["joints_cam"] - gt, axis=1).mean())
        rows.append(row)
    det = [r for r in rows if r["detected"]]
    summary = {"frames_with_gt": len(rows), "wilor_detected": len(det),
               "gt_wrist_depth_m": {"median": float(np.median([s[4][0, 2] for s in samples])),
                                    "min": float(np.min([s[4][0, 2] for s in samples])),
                                    "max": float(np.max([s[4][0, 2] for s in samples]))},
               "mono_all_detected": stats([r["mono_wrist"] for r in det]),
               "mono_all_detected_mpjpe_cm": float(np.mean([r["mono_mpjpe"] for r in det]) * 100) if det else None,
               "conditions": {}}
    for c in conds:
        ok = [r for r in det if r[c]["status"] == "ok"]
        st = {}
        for r in det:
            st[r[c]["status"]] = st.get(r[c]["status"], 0) + 1
        summary["conditions"][c] = {
            "coverage_of_gt": len(ok) / float(len(rows)) if rows else 0.0,
            "coverage_of_detected": len(ok) / float(len(det)) if det else 0.0,
            "wrist": stats([r[c]["wrist"] for r in ok]),
            "mpjpe_cm": float(np.mean([r[c]["mpjpe"] for r in ok]) * 100) if ok else None,
            "raw_depth_wrist_same_hands": stats([r[c].get("raw_wrist") for r in ok]),
            "mono_same_hands": stats([r["mono_wrist"] for r in ok]),
            "status_counts": st}
    (out / "rows.json").write_text(json.dumps(rows))
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
