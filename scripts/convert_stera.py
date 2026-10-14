#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stera-10M（fpvlabs/stera-10m, iPhone Pro LiDAR + ARKit）→ XHEY 统一 episode JSON（预训练用，无画面）。

只读每段 session 的 annotation.hdf5 里 /hand-pose 和 /cam-pose（不读 depth 帧、不下 rgb.mp4/rrd）。
- 手：/hand-pose/{left,right}_joints (N,21,3)，WiLoR 认手 + LiDAR 锚定，相机光学系（x右 y下 z前，米），
  关节顺序 = 腕 + 5×[mcp,pip,dip,tip]，与 XHEY 的 WiLoR/OpenPose 顺序一致。
- 世界系：/cam-pose 的 rotations/translations（ARKit VIO，经 Stera SDK 转成 ROS 风格 camera_link→world）。
  光学系 → link 用 calibrations/R_optical_to_link.npy（缺省用 SDK 常量），再乘 cam-pose 得世界系。
  ARKit 世界本身重力对齐；自动判断世界里哪根轴朝上（camera_link 的 z 轴平均方向），
  再统一转成 XHEY / ARKit 的 y 朝上（与 iPhone 管线一致）。
- 15 fps（原生），切成 --seg-s 秒一段，1 秒内手腕位移中位数 > --min-move-cm 才保留。

许可：Stera-10M License = CC BY-NC 4.0（非商用，需署名 FPV Labs / arXiv:2605.05945），
WiLoR、MANO 部分另受各自许可。HF 上是 gated 数据集，需要登录并同意条款后才能下载。转换结果不要提交进仓库。
用法：python scripts/convert_stera.py --root data/stera --out data/stera/unified   # root 下是 session_data_*/
"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from egodata.schema import empty_interaction  # noqa: E402
from headcam.hand_pose import wrist_poses_from_joints  # noqa: E402


def quat_to_rot(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


R_O2L_DEFAULT = quat_to_rot(0.7071, 0.0, 0.7071, 0.0)  # stera.core.transforms.R_OPTICAL_TO_LINK


def as_rot(r):
    r = np.asarray(r, dtype=np.float64)
    if r.shape[-2:] == (3, 3):
        return r
    if r.shape[-1] == 4:  # xyzw
        return np.stack([quat_to_rot(*q) for q in r])
    raise ValueError("unknown rotation shape %s" % (r.shape,))


def world_to_yup(Rw, R_o2l):
    """世界系 → y 朝上：用光学系 -y（相机「上」）在世界里的平均方向判断哪根轴是上（重力反方向）。"""
    up_world = (Rw @ (R_o2l @ np.array([0.0, -1.0, 0.0]))).mean(0)
    ax = int(np.argmax(np.abs(up_world))); sgn = np.sign(up_world[ax])
    if ax == 1 and sgn > 0:
        return np.eye(3), "y"
    if ax == 2:  # ROS: x前 y左 z上 → ARKit: x右 y上 z后
        return (np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], float) * np.array([1, sgn, 1])[:, None]), "z"
    # 其他情况：把 up 轴旋到 +y
    u = np.zeros(3); u[ax] = sgn
    v = np.cross(u, [0, 1, 0]); s = np.linalg.norm(v); c = float(np.dot(u, [0, 1, 0]))
    if s < 1e-8:
        return (np.diag([1, -1, -1]) if c < 0 else np.eye(3)), "y-"
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2), "axis%d" % ax


def nearest(ts_src, ts_q):
    i = np.clip(np.searchsorted(ts_src, ts_q), 1, len(ts_src) - 1)
    return np.where(np.abs(ts_src[i - 1] - ts_q) <= np.abs(ts_src[i] - ts_q), i - 1, i)


def convert(h5, out, seg_s, min_move_cm, stride, stats, up_axis="auto"):
    import h5py
    sess = Path(h5).parent; name = sess.name
    cal = sess / "calibrations" / "R_optical_to_link.npy"
    R_o2l = np.load(cal).astype(np.float64) if cal.exists() else R_O2L_DEFAULT
    with h5py.File(h5, "r") as f:
        hp, cp = f["hand-pose"], f["cam-pose"]
        if str(hp.attrs.get("coord_frame", "camera_3d")) != "camera_3d":
            raise ValueError("hand-pose 不是 3D（%s）" % hp.attrs.get("coord_frame"))
        hts = np.asarray(hp["timestamps"], np.float64)
        idx = np.arange(0, len(hts), stride)
        Jc = {s: np.asarray(hp["%s_joints" % s][:], np.float64)[idx] for s in ("left", "right")}
        Vd = {s: np.asarray(hp["%s_valid" % s][:]).astype(bool)[idx] for s in ("left", "right")}
        pts, Rw, tw = np.asarray(cp["timestamps"], np.float64), as_rot(cp["rotations"][:]), np.asarray(cp["translations"][:], np.float64)
    hts = hts[idx]; n = len(hts)
    k = nearest(pts, hts); okpose = np.abs(pts[k] - hts) < 0.1
    Rw, tw = Rw[k], tw[k]
    if up_axis == "auto":
        A, up = world_to_yup(Rw, R_o2l)
    elif up_axis == "z":
        A, up = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], float), "z(forced)"
    else:
        A, up = np.eye(3), "y(forced)"
    stats["up_axis"][up] = stats["up_axis"].get(up, 0) + 1
    hands, wpos = {}, {}
    for side in ("left", "right"):
        J = Jc[side]  # (n,21,3) 光学系
        Jw = np.einsum("nij,nkj->nki", Rw @ R_o2l, J) + tw[:, None]
        Jw = Jw @ A.T
        valid = Vd[side] & okpose & np.isfinite(Jw).all((1, 2)) & (np.abs(J).sum((1, 2)) > 0)
        Jw[~valid] = np.nan
        poses = wrist_poses_from_joints(list(Jw))
        hands[side] = {"joints": [jj.tolist() if v else [[None] * 3] * 21 for jj, v in zip(Jw, valid)],
                       "wrist_pose": [p if v else [None] * 7 for p, v in zip(poses, valid)],
                       "confidence": [1.0 if v else None for v in valid], "valid": valid.tolist(),
                       "filled": [False] * n, "label_status": ["ok" if v else "missing" for v in valid]}
        wpos[side] = np.where(valid[:, None], Jw[:, 0], np.nan)
    Tc = np.einsum("ij,njk->nik", A, Rw @ R_o2l)  # 光学系 → y-up 世界
    cam = np.tile(np.eye(4), (n, 1, 1)); cam[:, :3, :3] = Tc; cam[:, :3, 3] = tw @ A.T
    fps = round(1.0 / np.median(np.diff(hts)), 1)
    seg = int(seg_s * fps); kept = 0; one_s = max(1, int(round(fps)))
    for s0 in range(0, n - 17, seg):
        s1 = min(n, s0 + seg)
        mv = [np.linalg.norm(w[s0:s1][one_s:] - w[s0:s1][:-one_s], axis=1) for w in wpos.values() if s1 - s0 > one_s]
        mv = np.concatenate(mv) if mv else np.array([])
        med = float(np.nanmedian(mv)) * 100 if np.isfinite(mv).any() else 0.0
        stats["segments"] += 1
        if med < min_move_cm:
            continue
        sl = slice(s0, s1); m = s1 - s0
        ep = {"schema_version": "1.0", "episode_id": "stera/%s_%05d" % (name, s0), "source": "stera",
              "source_path": str(h5), "video_path": None, "fps": fps, "coordinate_frame": "arkit_world_yup",
              "num_frames": m, "timestamps": (hts[sl] - hts[s0]).tolist(), "camera_intrinsic": None,
              "camera_poses": cam[sl].tolist(),
              "hands": {kk: {ff: vv[ff][sl] for ff in vv} for kk, vv in hands.items()},
              "annotation": {"task": {"name": name, "instruction": ""}, "subtasks": [], "instructions": [],
                             "environment": {"name": "", "detail": "", "source": "missing"}},
              "hand_pose": {"backend": "stera_wilor_lidar", "frame": "arkit_world_yup", "metric_translation": True,
                            "move_median_cm_per_s": med, "world_up_axis": up},
              "license_note": "Stera-10M License (CC BY-NC 4.0; attribution FPV Labs, arXiv:2605.05945); WiLoR/MANO-derived"}
        ep.update(empty_interaction(m))
        (out / ("stera_%s_%05d.json" % (name, s0))).write_text(json.dumps(ep))
        kept += 1; stats["kept"] += 1; stats["frames"] += m
    return kept


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seg-s", type=float, default=20.0)
    ap.add_argument("--min-move-cm", type=float, default=3.0)
    ap.add_argument("--stride", type=int, default=1, help="原生 15 fps，保持 1")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--up-axis", default="auto", choices=["auto", "y", "z"],
                    help="世界系哪根轴朝上；auto = 用相机「上」方向的平均值判断（首次拿到数据请人工核对一段）")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(Path(a.root) / "**/annotation.hdf5"), recursive=True))[: a.limit]
    stats = {"sessions": 0, "segments": 0, "kept": 0, "frames": 0, "up_axis": {}}
    for f in files:
        try:
            convert(f, out, a.seg_s, a.min_move_cm, a.stride, stats, a.up_axis); stats["sessions"] += 1
        except Exception as e:  # noqa: BLE001
            print("skip", f, repr(e)[:200], flush=True)
    (out / "convert_summary.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
