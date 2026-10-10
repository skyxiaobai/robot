#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ReViV（单目第一视角, ECCV 2026）vs WiLoR 单目，在 EgoDex 留出片段上比较手部 3D。

ReViV 没在 EgoDex 上训练（训练集: HOT3D/H2O/ARCTIC/TACO/HoloAssist/Ego-Exo4D/EgoGen/Nymeria），
所以这是干净的留出评测。

片段：eval_hand_refine.py 的 EVAL_EPISODES 里长度够的 3 条（open_close_insert_remove_case/8 只有 15 帧，跳过），
每条在 1 s、4 s、6.5 s 处各取 2 s（60 帧 @30fps），起点在看误差之前定好。
ReViV 预测 [60,21,3] 相机系米制关节（MANO 顺序），这里换成仓库的 21 点顺序。
WiLoR 用 eval_hand_refine.py 的逐帧缓存（不做精修），取同样的帧。

两种匹配：
  * gated：仓库原 evaluate_hand_frames（按手腕 2D 投影匹配，250 px 门限）。
  * side：按左右手标签直接配对（不设门限），只要 GT 手可见。ReViV 每帧总会输出两只手，
    2D 门限会把深度猜错的手当成"漏检"，所以两种都报。

reviv_zvalid：把 ReViV 手腕 z<=5cm（在相机后面）的退化输出当作未检出。

用法：python scripts/eval_reviv_egodex.py --reviv_out <demo_hand 输出目录> --out docs/reviv_compare.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from egodata.egodex import EGODEX_NONCORRESPONDING_JOINTS, load_episode_hdf5  # noqa
from headcam.hand_pose import evaluate_hand_frames, project_pinhole, _gt_hand_visible  # noqa
from eval_hand_refine import _world_to_camera  # noqa

MANO_TO_REPO = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
EPISODES = ("test/add_remove_lid/14", "test/add_remove_lid/23", "test/add_remove_lid/8")
STARTS = (1.0, 4.0, 6.5)
FPS = 30


def gt_cam(ep, idx, side):
    raw = ep["hands"][side]["joints"][idx]
    a = np.full((21, 3), np.nan)
    for j, p in enumerate(raw):
        if p is not None and p[0] is not None:
            a[j] = p
    if not np.isfinite(a[0]).all():
        return None
    return _world_to_camera(a, np.asarray(ep["camera_poses"][idx], float))


def side_metrics(samples, excl):
    rel, wr, keys = [], [], []
    vis = 0
    for s in samples:
        K = s["K"]
        for side in ("left", "right"):
            g = s["gt"].get(side)
            if g is None:
                continue
            uv = project_pinhole(g, K)
            if not _gt_hand_visible(g, uv, s["width"], s["height"], s["gt_confidence"].get(side), 0.75):
                continue
            vis += 1
            p = s["pred"].get(side)
            if p is None:
                continue
            P = p["joints_cam"]
            keep = [j for j in range(21) if j not in excl]
            rel.append(float(np.nanmean(np.linalg.norm((P[keep] - P[0]) - (g[keep] - g[0]), axis=1))))
            wr.append((s["clip"], side, P[0], g[0]))
    w = np.array([np.linalg.norm(a[2] - a[3]) for a in wr])
    # 每片段每只手一个最小二乘尺度（和仓库口径一致：按 episode_id/side）
    groups = {}
    for c, sd, p, g in wr:
        groups.setdefault((c, sd), []).append((p, g))
    ws = []
    for k, pairs in groups.items():
        P = np.stack([x[0] for x in pairs]); G = np.stack([x[1] for x in pairs])
        sc = float((P * G).sum() / max((P * P).sum(), 1e-12))
        ws += list(np.linalg.norm(sc * P - G, axis=1))
    ws = np.array(ws)
    f = lambda x, q: float(np.percentile(x, q)) if len(x) else float("nan")
    return dict(gt_visible=vis, paired=len(rel), coverage=len(rel) / vis if vis else float("nan"),
                mpjpe_rel_cm_mean=100 * float(np.mean(rel)), mpjpe_rel_cm_median=100 * f(rel, 50),
                wrist_cm_median=100 * f(w, 50), wrist_cm_p90=100 * f(w, 90), wrist_le2cm=float(np.mean(w <= 0.02)),
                wrist_scaled_cm_median=100 * f(ws, 50), wrist_scaled_cm_p90=100 * f(ws, 90),
                wrist_scaled_le2cm=float(np.mean(ws <= 0.02)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/egodex_hand_eval")
    ap.add_argument("--reviv_out", required=True)
    ap.add_argument("--out", default="docs/reviv_compare.json")
    a = ap.parse_args()
    root = Path(a.root)
    samples = {"wilor": [], "reviv": [], "reviv_zvalid": []}
    for rel in EPISODES:
        ep = load_episode_hdf5(root / (rel + ".hdf5"))
        cache = np.load(root / "cache" / (rel.replace("/", "__") + ".hdf5_wilor_d055.npz"))
        K = np.asarray(ep["camera_intrinsic"], float)
        W, H = int(cache["width"]), int(cache["height"])
        for st in STARTS:
            stem = "%s_s%s" % (rel.replace("test/", "").replace("/", "_"), ("%g" % st))
            d = Path(a.reviv_out) / stem
            rv = {s: np.load(d / ("%s_tok_%shand.npy" % (stem.split(".")[0], s[0])))[:, MANO_TO_REPO] for s in ("left", "right")}
            s0 = int(round(st * FPS))
            for i in range(60):
                idx = s0 + i
                base = dict(episode_id=stem, clip=stem, width=W, height=H, K=K, gt={}, gt_confidence={})
                for side in ("left", "right"):
                    base["gt"][side] = gt_cam(ep, idx, side)
                    base["gt_confidence"][side] = ep["hands"][side]["confidence"][idx]
                for m in ("wilor", "reviv", "reviv_zvalid"):
                    s = dict(base, pred={})
                    for side in ("left", "right"):
                        P = cache[side][idx] if m == "wilor" else rv[side][i]
                        if m == "reviv_zvalid" and not P[0, 2] > 0.05:
                            P = np.full_like(P, np.nan)  # 手腕在相机后面 = 退化输出，当作未检出
                        s["pred"][side] = {"joints_cam": P, "keypoints_2d": None} if np.isfinite(P[0]).all() else None
                    samples[m].append(s)
    res = {}
    for m in samples:
        excl = EGODEX_NONCORRESPONDING_JOINTS
        g = evaluate_hand_frames(samples[m], exclude_joints=excl)
        res[m] = {"gated": {k: (list(v) if isinstance(v, tuple) else v) for k, v in g.items()},
                  "side": side_metrics(samples[m], excl)}
    print(json.dumps(res, indent=1, ensure_ascii=False))
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
