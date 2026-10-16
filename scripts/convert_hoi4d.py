#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""HOI4D（hoi4d.github.io，头戴 RGB-D 第一视角）→ XHEY 统一 episode JSON。

只读 Hand_pose/*.pickle（MANO 参数）+ camera_params/*/intrin.npy，不需要视频/深度
（XHEY 预训练分量没有画面特征）。
- 每帧 pickle：poseCoeff（3 全局旋转 + 45 轴角，flat_hand_mean，不用 PCA）、beta、
  trans（相机光学系，米）。按官方说明 kps3d = MANO 关节/1000 + trans（manopth 约定，不居中）。
  这里用纯 numpy 实现 MANO 前向（MANO_{LEFT,RIGHT}.pkl 需自行按 MANO 许可获取，不随数据上传）。
- 21 点按 OpenPose 顺序（0 腕，4 拇指尖，8 食指尖 ...），和 WiLoR 一致。
- 坐标系：相机光学系（x右 y下 z前），frame_convention=camera，不需要翻转。
  用 kps2D 做投影校验：MANO 关节投影到图像应和官方 kps2D 重合（像素误差写进 hand_pose）。
- 15 fps（HOI4D 视频 15 fps，20 s/段），没标注的帧记为 missing。
- 「手在动」：1 秒内手腕位移中位数 > --min-move-cm 的序列才保留。

许可：HOI4D CC BY-NC 4.0（非商用，需引用 Liu et al. CVPR 2022）。转换结果不要提交进仓库。
用法：python scripts/convert_hoi4d.py --hand-pose raw/Hand_pose --cam raw/camera_params \
        --mano /workspace/mano --out data/hoi4d/unified
"""
import argparse, glob, json, os, pickle, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from egodata.schema import empty_interaction  # noqa: E402
from headcam.hand_pose import wrist_poses_from_joints  # noqa: E402
from common import frames as FR  # noqa: E402

FPS = 15.0
NFR = 300
TIPS = [745, 317, 444, 556, 673]
ORDER = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]


class _Ch:
    def __init__(self, *a, **k): pass

    def __setstate__(self, s): self.__dict__.update(s if isinstance(s, dict) else {"s": s})


class _U(pickle.Unpickler):
    def find_class(self, m, n):
        if m.startswith("chumpy"):
            return _Ch
        return super().find_class(m, n)


def _arr(v):
    if isinstance(v, _Ch):
        if "x" in v.__dict__:
            return np.asarray(v.x)
        a = _arr(v.__dict__["a"])
        return a.ravel()[v.__dict__["idxs"]].reshape(v.__dict__["preferred_shape"])
    return np.asarray(v.toarray() if hasattr(v, "toarray") else v)


class Mano:
    """numpy MANO（与 manopth flat_hand_mean=True, use_pca=False 一致），输出米。"""
    def __init__(self, path):
        d = _U(open(path, "rb"), encoding="latin1").load()
        self.vt = _arr(d["v_template"]); self.sd = _arr(d["shapedirs"]); self.pd = _arr(d["posedirs"])
        self.Jr = _arr(d["J_regressor"]); self.W = _arr(d["weights"])
        self.par = _arr(d["kintree_table"])[0].astype(int)

    @staticmethod
    def _rod(r):
        th = np.linalg.norm(r, axis=-1, keepdims=True) + 1e-8; k = r / th
        K = np.zeros(r.shape[:-1] + (3, 3))
        K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
        K = K - np.swapaxes(K, -1, -2); th = th[..., None]
        return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K

    def joints(self, pose, beta):
        """pose (T,48), beta (T,10) → (T,21,3) 米，未加 trans。"""
        T = len(pose)
        v = self.vt[None] + np.einsum("vck,tk->tvc", self.sd, beta)
        J = np.einsum("jv,tvc->tjc", self.Jr, v)
        R = self._rod(pose.reshape(T, 16, 3))
        pf = (R[:, 1:] - np.eye(3)).reshape(T, -1)
        v = v + np.einsum("vcp,tp->tvc", self.pd, pf)
        G = np.zeros((T, 16, 4, 4)); G[..., 3, 3] = 1
        G[:, 0, :3, :3] = R[:, 0]; G[:, 0, :3, 3] = J[:, 0]
        for i in range(1, 16):
            L = np.zeros((T, 4, 4)); L[:, 3, 3] = 1; L[:, :3, :3] = R[:, i]
            L[:, :3, 3] = J[:, i] - J[:, self.par[i]]
            G[:, i] = G[:, self.par[i]] @ L
        Jp = G[:, :, :3, 3]
        Gr = G.copy(); Gr[..., :3, 3] -= np.einsum("tjab,tjb->tja", G[..., :3, :3], J)
        A = np.einsum("vj,tjab->tvab", self.W, Gr)
        vh = np.concatenate([v, np.ones((T, 778, 1))], -1)
        vp = np.einsum("tvab,tvb->tva", A, vh)[..., :3]
        return np.concatenate([Jp, vp[:, TIPS]], 1)[:, ORDER]


def load_seq(d):
    fr = {}
    for p in glob.glob(os.path.join(d, "*.pickle")):
        try:
            i = int(os.path.basename(p)[:-7])
        except ValueError:
            continue
        h = pickle.load(open(p, "rb"), encoding="latin1")
        fr[i] = h
    return fr


def convert(seq, hp, cam, mano, out, min_move_cm, stats):
    n = NFR
    hands, wpos, perr = {}, {}, []
    K = np.load(os.path.join(cam, seq.split("/")[0], "intrin.npy"))
    for side in ("left", "right"):
        fr = load_seq(os.path.join(hp, "handpose_%s_hand" % side, seq))
        J = np.full((n, 21, 3), np.nan); valid = np.zeros(n, bool)
        idx = sorted(i for i in fr if 0 <= i < n)
        if idx:
            pose = np.stack([np.asarray(fr[i]["poseCoeff"], float) for i in idx])
            beta = np.stack([np.asarray(fr[i]["beta"], float) for i in idx])
            tr = np.stack([np.asarray(fr[i]["trans"], float) for i in idx])
            jj = mano[side].joints(pose, beta) + tr[:, None]
            ok = np.isfinite(jj).all((1, 2)) & (jj[:, 0, 2] > 0.05)
            J[idx] = jj; valid[idx] = ok
            k2 = np.stack([np.asarray(fr[i]["kps2D"], float) for i in idx])
            uv = jj @ K.T; uv = uv[..., :2] / uv[..., 2:3]
            perr.append(np.nanmedian(np.linalg.norm(uv - k2, axis=-1)))
        J[~valid] = np.nan
        poses = wrist_poses_from_joints(list(J))
        hands[side] = {"joints": [jj.tolist() if v else [[None] * 3] * 21 for jj, v in zip(J, valid)],
                       "wrist_pose": [p if v else [None] * 7 for p, v in zip(poses, valid)],
                       "confidence": [1.0 if v else None for v in valid], "valid": valid.tolist(),
                       "filled": [False] * n, "label_status": ["ok" if v else "missing" for v in valid]}
        wpos[side] = np.where(valid[:, None], J[:, 0], np.nan)
    stats["episodes"] += 1
    mv = np.concatenate([np.linalg.norm(w[15:] - w[:-15], axis=1) for w in wpos.values()])
    med = float(np.nanmedian(mv)) * 100 if np.isfinite(mv).any() else 0.0
    nvalid = int(sum(np.isfinite(w[:, 0]).sum() for w in wpos.values()))
    pe = float(np.nanmax(perr)) if perr else None
    stats.setdefault("proj_err_px", []).append(pe)
    if med < min_move_cm or nvalid < 60:
        return 0
    eid = "hoi4d/" + seq
    ep = {"schema_version": "1.0", "episode_id": eid, "source": "hoi4d", "source_path": seq,
          "video_path": None, "fps": FPS, "coordinate_frame": "camera", "num_frames": n,
          "timestamps": (np.arange(n) / FPS).tolist(), "camera_intrinsic": K.tolist(),
          "camera_poses": [np.eye(4).tolist()] * n, "hands": hands,
          "annotation": {"task": {"name": seq.split("/")[2], "instruction": ""}, "subtasks": [], "instructions": [],
                         "environment": {"name": "", "detail": "", "source": "missing"}},
          "hand_pose": {"backend": "hoi4d_mano_gt", "frame": "camera", "metric_translation": True,
                        "joints_present": "21 MANO", "move_median_cm_per_s": med, "kps2d_reproj_px": pe},
          "license_note": "CC BY-NC 4.0, HOI4D (Liu et al. CVPR 2022), non-commercial; MANO-derived"}
    ep.update(empty_interaction(n))
    FR.tag(ep, FR.CAMERA, gravity_aligned=False, source_frame="hoi4d_camera_optical_m")
    sc = FR.sanity_check(ep)
    if not sc["ok"]:
        stats.setdefault("frame_flagged", []).append(eid)
    stats.setdefault("median_dist_m", []).append(sc["median_dist_m"])
    (out / ("hoi4d_%s.json" % seq.replace("/", "_"))).write_text(json.dumps(ep))
    stats["kept"] += 1; stats["hand_frames"] += nvalid; stats["frames"] += n
    return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hand-pose", required=True); ap.add_argument("--cam", required=True)
    ap.add_argument("--mano", required=True, help="含 MANO_LEFT.pkl / MANO_RIGHT.pkl 的目录")
    ap.add_argument("--out", required=True); ap.add_argument("--min-move-cm", type=float, default=3.0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    mano = {s: Mano(os.path.join(a.mano, "MANO_%s.pkl" % s.upper())) for s in ("left", "right")}
    seqs = set()
    for side in ("left", "right"):
        r = os.path.join(a.hand_pose, "handpose_%s_hand" % side)
        for d in glob.glob(os.path.join(r, "*/*/*/*/*/*/*")):
            seqs.add(os.path.relpath(d, r))
    seqs = sorted(seqs)[: a.limit or None]
    stats = {"episodes": 0, "kept": 0, "frames": 0, "hand_frames": 0}
    for s in seqs:
        try:
            convert(s, a.hand_pose, a.cam, mano, out, a.min_move_cm, stats)
        except Exception as e:  # noqa: BLE001
            print("skip", s, repr(e)[:200], flush=True)
    pe = [x for x in stats.pop("proj_err_px") if x is not None]
    md = [x for x in stats.pop("median_dist_m", []) if x is not None]
    stats["proj_err_px_median"] = float(np.median(pe)) if pe else None
    stats["median_wrist_dist_m"] = float(np.median(md)) if md else None
    stats["hours"] = stats["frames"] / FPS / 3600
    (out / "convert_summary.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
