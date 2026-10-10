#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""在 HOT3D-Clips 真数据上评测接触 / 抓取启发式。

真值：HOT3D MANO 手表面（778 个顶点）到 HOT3D 物体网格（object_models_eval/*.glb）的距离。
  接触 = 任一手顶点离物体表面 <= --gt-contact-m；
  抓取 = 至少 --gt-min-fingers 根手指的末节（按 MANO 蒙皮权重分顶点）同时 <= 阈值。
估计：scripts/egodata/interaction.estimate_interaction，物体位姿和网格用 HOT3D 真值，
  手用 (a) MANO 真值 21 关节，或 (b) 双目管线写出的 episode 关节。
时间滤波默认开。docs/contact_eval/hot3d_report.json 里的数字是没开滤波跑的，
复现时加 --no-temporal-filter。开着滤波的重跑在
docs/contact_eval/hot3d_report_filtered.json。停留时长扫参（不改默认）见
scripts/dwell.py 和 docs/contact_grasp.md。

    python scripts/eval_hot3d_contact.py --clips-dir .../train_quest3 --models models/ \
        --mano /workspace/mano --stereo-episodes run_final/episodes --tune clip-000000 --out out/ \
        --no-temporal-filter
"""
import argparse
import glob
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "headcam"))

from egodata.interaction import HEURISTIC_DEFAULTS, estimate_interaction, evaluate_interaction, events_from_labels  # noqa: E402

TIP_VERTS = [745, 317, 444, 556, 673]
MANO_TO_OPENPOSE = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
DISTAL = (15, 3, 6, 12, 9)  # MANO 关节号：拇指、食指、中指、无名指、小指的末节


class Surfaces(object):
    def __init__(self, models_dir, spacing_m=0.0008):
        import trimesh
        from scipy.spatial import cKDTree
        self.trees = {}
        self.radius = {}
        for path in sorted(glob.glob(os.path.join(models_dir, "obj_*.glb"))):
            bop = str(int(Path(path).stem.split("_")[1]))
            mesh = trimesh.load(path, force="mesh", process=False)
            n = int(min(400000, max(20000, mesh.area / spacing_m ** 2)))
            pts, _ = trimesh.sample.sample_surface_even(mesh, n, seed=0)
            pts = np.vstack([pts, mesh.vertices])
            self.trees[bop] = cKDTree(pts)
            self.radius[bop] = float(np.linalg.norm(pts, axis=1).max())
            self.mesh = mesh

    def distance(self, bop, T_world_obj, pts_world):
        R, t = T_world_obj[:3, :3], T_world_obj[:3, 3]
        local = (pts_world - t) @ R
        if np.linalg.norm(local, axis=1).min() > self.radius[bop] + 0.05:
            return np.full(len(pts_world), np.inf)
        d, _ = self.trees[bop].query(local)
        return d


def pose_matrix7(p):
    from egodata.interaction import _pose_matrix
    return _pose_matrix(p)


class Mano(object):
    def __init__(self, mano_dir):
        import smplx
        import torch
        from hand_tracking_toolkit.hand_models.mano_hand_model import MANOHandModel
        self.model = MANOHandModel(mano_dir)
        right = smplx.MANO(mano_dir, is_rhand=True, use_pca=False)
        self.J = right.J_regressor.numpy()
        owner = right.lbs_weights.numpy().argmax(1)
        self.finger_sets = [np.where(owner == j)[0] for j in DISTAL]
        self.torch = torch

    def frame(self, clip, key):
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
            out[side.value] = (v, np.concatenate([self.J @ v, v[TIP_VERTS]], 0)[MANO_TO_OPENPOSE])
        return out


def load_clip(clip, mano, surfaces, cache):
    cache_path = os.path.join(cache, os.path.basename(clip) + ".json")
    if os.path.isfile(cache_path):
        return json.load(open(cache_path))
    from hot3d_adapter import read_frame_objects, tracks_from_frames
    keys = sorted(p.split("/")[-1].split(".")[0] for p in glob.glob(os.path.join(clip, "*.info.json")))
    stamps = [json.load(open(os.path.join(clip, k + ".info.json")))["image_timestamps_ns"]["1201-1"] * 1e-9 for k in keys]
    objects = tracks_from_frames([read_frame_objects(clip, k) for k in keys])
    n = len(keys)
    joints = {"left": [None] * n, "right": [None] * n}
    hand_obj_d = {"left": [None] * n, "right": [None] * n}  # 每帧 {obj_id: [hand_min, finger1..5]}
    for i, k in enumerate(keys):
        fr = mano.frame(clip, k)
        for side, (verts, j21) in fr.items():
            joints[side][i] = j21.tolist()
            rec = {}
            for obj in objects:
                if not obj["valid"][i] or str(obj["bop_id"]) not in surfaces.trees:
                    continue
                d = surfaces.distance(str(obj["bop_id"]), pose_matrix7(obj["pose"][i]), verts)
                if not np.isfinite(d).any():
                    continue
                rec[obj["id"]] = [float(d.min())] + [float(d[s].min()) for s in mano.finger_sets]
            hand_obj_d[side][i] = rec
    data = {"keys": keys, "timestamps": stamps, "objects": objects, "joints": joints, "dist": hand_obj_d}
    json.dump(data, open(cache_path, "w"))
    return data


def gt_labels(data, contact_m, min_fingers):
    n = len(data["timestamps"])
    out = {"contact": {}, "grasp": {}, "events": []}
    for side in ("left", "right"):
        ids, states, valid = [None] * n, [None] * n, [False] * n
        gflag = [False] * n
        cflag = [False] * n
        for i in range(n):
            rec = data["dist"][side][i]
            if rec is None:
                continue
            valid[i] = True
            if rec:
                oid, d = min(rec.items(), key=lambda kv: kv[1][0])
                if d[0] <= contact_m:
                    ids[i] = oid
                    cflag[i] = True
                    gflag[i] = sum(x <= contact_m for x in d[1:]) >= min_fingers
        for i in range(n):
            if not valid[i]:
                continue
            if gflag[i]:
                states[i] = "grasp"
            elif i > 0 and gflag[i - 1]:
                states[i] = "release"
            elif cflag[i] or (i + 1 < n and gflag[i + 1]):
                states[i] = "pre_grasp"
            else:
                states[i] = "open"
        out["contact"][side] = {"object_id": ids, "valid": valid}
        out["grasp"][side] = {"state": states, "valid": valid}
        out["events"] += events_from_labels(side, ids, states, data["timestamps"], valid)
    return out


def stereo_joints(path, n):
    ep = json.load(open(path))
    out = {}
    for side in ("left", "right"):
        h = ep["hands"][side]
        js = h["joints"]
        out[side] = {"joints": [js[i] if (h["valid"][i] and not (h.get("filled") or [False] * n)[i]) else None for i in range(n)]}
    return out


def mesh_surfaces(data, surfaces_obj):
    return {o["id"]: surfaces_obj[str(o["bop_id"])] for o in data["objects"] if str(o["bop_id"]) in surfaces_obj}


def summarize(m):
    def f1(c):
        p, r = c["precision"], c["recall"]
        return None if not p or not r else 2 * p * r / (p + r)
    res = {}
    for k in ("contact", "grasp"):
        c = m[k]["both"]
        res[k] = {"tp": c["tp"], "fp": c["fp"], "fn": c["fn"], "precision": c["precision"], "recall": c["recall"], "f1": f1(c)}
    ev = m["events"]
    res["events"] = {"matched": ev["matched"], "unmatched_gt": ev["unmatched_gt"], "unmatched_pred": ev["unmatched_pred"],
                     "timing_median_s": ev["timing_error_median_s"]}
    return res


def pooled(results):
    tot = {}
    for k in ("contact", "grasp"):
        tp = sum(r[k]["tp"] for r in results); fp = sum(r[k]["fp"] for r in results); fn = sum(r[k]["fn"] for r in results)
        p = tp / (tp + fp) if tp + fp else None; r_ = tp / (tp + fn) if tp + fn else None
        tot[k] = {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r_, "f1": (2 * p * r_ / (p + r_)) if p and r_ else None}
    return tot


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-dir", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--mano", required=True)
    ap.add_argument("--stereo-episodes")
    ap.add_argument("--tune", required=True)
    ap.add_argument("--gt-contact-m", type=float, default=0.005)
    ap.add_argument("--gt-min-fingers", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--no-temporal-filter",
        action="store_true",
        help="关掉接触/抓取的滞回和最短停留。文档里的 HOT3D 数字是关着滤波跑的",
    )
    return ap


def main():
    ap = build_parser()
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cache = os.path.join(a.out, "cache"); os.makedirs(cache, exist_ok=True)
    surf = Surfaces(a.models)
    import trimesh
    meshes = {}
    for path in sorted(glob.glob(os.path.join(a.models, "obj_*.glb"))):
        m = trimesh.load(path, force="mesh", process=False)
        meshes[str(int(Path(path).stem.split("_")[1]))] = {"vertices": np.asarray(m.vertices), "faces": np.asarray(m.faces)}
    mano = Mano(a.mano)
    clips = sorted(glob.glob(os.path.join(a.clips_dir, "clip-*")))
    datas = {os.path.basename(c): load_clip(c, mano, surf, cache) for c in clips}
    gts = {k: gt_labels(d, a.gt_contact_m, a.gt_min_fingers) for k, d in datas.items()}

    def run(name, hands, params):
        d = datas[name]
        pred = estimate_interaction(
            hands, d["objects"], d["timestamps"], mesh_surfaces(d, meshes),
            params=params, temporal_filter=not a.no_temporal_filter,
        )
        return summarize(evaluate_interaction(gts[name], pred, d["timestamps"]))

    def gt_hands(name):
        return {s: {"joints": datas[name]["joints"][s]} for s in ("left", "right")}

    stereo = {}
    if a.stereo_episodes:
        for p in sorted(glob.glob(os.path.join(a.stereo_episodes, "clip-*.json"))):
            name = Path(p).stem
            if name in datas:
                stereo[name] = stereo_joints(p, len(datas[name]["timestamps"]))

    gt_stats = {}
    for k, g in gts.items():
        nv = sum(sum(g["contact"][s]["valid"]) for s in ("left", "right"))
        nc = sum(sum(1 for x, v in zip(g["contact"][s]["object_id"], g["contact"][s]["valid"]) if v and x) for s in ("left", "right"))
        ng = sum(sum(1 for x in g["grasp"][s]["state"] if x == "grasp") for s in ("left", "right"))
        gt_stats[k] = {"hand_frames": nv, "contact": nc, "grasp": ng, "events": len(g["events"])}

    grid = {"contact_m": [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04],
            "min_tips": [1, 2, 3],
            "aperture_grasp_m": [0.08, 0.10, 0.12, 0.15]}
    tuned = {}
    for label, hands_of in (("gt_joints", gt_hands), ("stereo", lambda n: stereo[n])):
        if label == "stereo" and a.tune not in stereo:
            continue
        best = None
        for cm, mt, apm in itertools.product(*grid.values()):
            params = {"contact_m": cm, "min_tips": mt, "aperture_grasp_m": apm}
            r = run(a.tune, hands_of(a.tune), params)
            score = (r["contact"]["f1"] or 0) + (r["grasp"]["f1"] or 0)
            if best is None or score > best[0]:
                best = (score, params)
        tuned[label] = best[1]

    report = {"gt_definition": {"contact_m": a.gt_contact_m, "min_fingers": a.gt_min_fingers}, "gt_stats": gt_stats,
              "tune_clip": a.tune, "tuned_params": tuned, "defaults": dict(HEURISTIC_DEFAULTS), "results": {}}
    for label, hands_of, names in (("gt_joints", gt_hands, sorted(datas)), ("stereo", lambda n: stereo[n], sorted(stereo))):
        for setting, params in (("default", None), ("tuned", tuned.get(label))):
            if setting == "tuned" and params is None:
                continue
            per = {n: run(n, hands_of(n), params) for n in names}
            test = [per[n] for n in names if n != a.tune]
            report["results"]["%s/%s" % (label, setting)] = {"per_clip": per, "pooled_test": pooled(test),
                                                             "test_clips": [n for n in names if n != a.tune]}
    json.dump(report, open(os.path.join(a.out, "report.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v["pooled_test"] for k, v in report["results"].items()}, indent=1))
    print(json.dumps({"gt_stats": gt_stats, "tuned": tuned}, indent=1))


if __name__ == "__main__":
    main()
