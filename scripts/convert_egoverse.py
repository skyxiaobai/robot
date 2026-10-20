#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EgoVerse（GaTech-RL2，Mecka 等头戴采集）Zarr v3 episode → XHEY 统一 episode JSON（预训练用，无画面）。

数据获取（公开只读，仓库 README 的 AWS Configure + egomimic/utils/aws/setup_secret.sh
会回退到 public read-only 凭证，再用 egomimic/scripts/data_download/sync_s3.py 同步；
sync_s3 的 mecka 预设按 embodiment 过滤在 2026-07-08 合并后已失效，按 SQL 的 lab='mecka' 选段）。

- 关节：left/right.obs_keypoints (T,63) MANO 序（0=腕，与 obs_wrist_pose 平移一致），世界系（z 朝上）。
- 相机系：obs_head_pose (T,7)=[xyz, qw,qx,qy,qz] 的逆变换 → OpenCV 光学系（x右 y下 z前，米）；
  用 zarr.json 的 intrinsics.front_1 投影校验（样例 15/15 段关节 99%+ 落在 640×360 内、z>0 100%）。
- ⚠️ 原生标签率：导出是 30 fps，但关键点由 10 Hz 线性插值而来（等速段恰 3 帧）。
  本转换默认 --stride 3 只取关键帧，输出 10 fps；按 30 fps 做 QC 会得到 jitter≈0 的假合格。
- 头部位姿整段恒定的段（样例 2/15）写 egoverse.head_const=True，训练选段时应排除/复核。
- 无深度、无 IMU：depth_valid 在 QC 里记 null，结果只作方向参考。内参是真实标定（非猜测）。

许可：目录里 freeform 行 license 字段为空（旗舰行 CC BY-SA 4.0），大规模使用前确认条款；
转换结果不要提交进仓库，Kaggle 数据集保持私有。
用法：python scripts/convert_egoverse.py --input /path/to/zarr_dir --out data/egoverse/unified
依赖：zarr（pip install zarr），其余仅 numpy。
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from egodata.schema import empty_interaction, rotmat_to_quat_xyzw  # noqa: E402
from common import frames as FR  # noqa: E402


def quat_wxyz_to_rot(q):
    """(N,4) [qw,qx,qy,qz] → (N,3,3)。"""
    q = np.asarray(q, float); q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z); R[..., 0, 1] = 2 * (x * y - z * w); R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w); R[..., 1, 1] = 1 - 2 * (x * x + z * z); R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w); R[..., 2, 1] = 2 * (y * z + x * w); R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def convert_one(zarr_path, stride=3):
    try:
        import zarr
    except ImportError as e:
        raise SystemExit("convert_egoverse 需要 zarr：pip install zarr") from e
    zarr_path = Path(zarr_path)
    attrs = json.loads((zarr_path / "zarr.json").read_text())["attributes"]
    g = zarr.open_group(str(zarr_path), mode="r")
    head_all = np.asarray(g["obs_head_pose"][:]); n30 = len(head_all)
    idx = np.arange(0, n30, stride); head = head_all[idx]; n = len(idx)
    fps = float(attrs.get("fps", 30)) / stride
    Rh = quat_wxyz_to_rot(head[:, 3:7]); Rt = np.transpose(Rh, (0, 2, 1)); t = head[:, :3]
    head_const = bool(head_all[:, :3].std(0).max() < 1e-9 and head_all[:, 3:].std(0).max() < 1e-9)
    intr = (attrs.get("intrinsics") or {}).get("front_1")
    K = np.array(intr, float)[:3, :3] if intr else None
    hands = {}
    for side in ("left", "right"):
        kp = np.asarray(g[f"{side}.obs_keypoints"][:])[idx].reshape(n, 21, 3)
        cam = np.einsum("nij,nkj->nki", Rt, kp - t[:, None, :])
        wrist = np.asarray(g[f"{side}.obs_wrist_pose"][:])[idx]
        Rcam = np.einsum("nij,njk->nik", Rt, quat_wxyz_to_rot(wrist[:, 3:7]))
        finite = np.isfinite(cam).all(axis=(1, 2))
        hands[side] = {
            "joints": [[[float(v) for v in j] for j in fr] if finite[i] else None for i, fr in enumerate(cam)],
            "wrist_pose": [[float(v) for v in cam[i, 0]] + rotmat_to_quat_xyzw(Rcam[i]) if finite[i] else None for i in range(n)],
            "confidence": [1.0 if finite[i] else 0.0 for i in range(n)],
            "valid": [bool(v) for v in finite], "filled": [False] * n,
            "label_status": ["ok" if finite[i] else "missing" for i in range(n)],
        }
    ep = {"schema_version": "1.0", "episode_id": "egoverse_mecka/%s" % zarr_path.name.replace(".zarr", ""),
          "source": "egoverse_mecka", "source_path": str(zarr_path), "video_path": "", "fps": fps,
          "coordinate_frame": "camera", "image_width": 640, "image_height": 360, "num_frames": int(n),
          "timestamps": [i / fps for i in range(n)], "camera_intrinsic": K.tolist() if K is not None else None,
          "camera_poses": [np.eye(4).tolist()] * int(n), "hands": hands,
          "annotation": {"task": {"name": str(attrs.get("task_name", "")), "instruction": str(attrs.get("task_description", "") or "")}},
          "coverage": {}, "hand_pose": {"backend": "egoverse_keypoints", "frame": "camera", "metric_translation": True, "label_rate_hz": fps},
          "egoverse": {"head_const": head_const, "n30": int(n30), "intrinsics_present": K is not None},
          "license_note": "EgoVerse/Mecka: catalog license blank for freeform rows (flagship CC BY-SA 4.0); private research use only"}
    ep.update(empty_interaction(n))
    FR.tag(ep, FR.CAMERA, gravity_aligned=False, source_frame="egoverse_world_zup")
    return ep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="含 *.zarr 的目录（sync_s3 同步下来的 episode）")
    ap.add_argument("--out", required=True, help="统一 JSON 输出目录")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stride", type=int, default=3, help="30 fps 导出只取每第 k 帧（原生标签 10 Hz，默认 3）")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    paths = sorted(Path(a.input).glob("*.zarr"))[: a.limit]
    n = 0
    for p in paths:
        ep = convert_one(p, a.stride)
        (out / (p.name.replace(".zarr", "") + ".json")).write_text(json.dumps(ep))
        n += 1; print(p.name, ep["num_frames"], "head_const" if ep["egoverse"]["head_const"] else "", flush=True)
    print("wrote %d episodes to %s" % (n, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
