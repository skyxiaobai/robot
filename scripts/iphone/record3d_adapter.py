#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Record3D 导出 → 单目 RGB-D 会话目录（docs/iphone_capture.md §会话格式）。

支持两种导出：

1. ``.r3d``（Record3D 默认的 “Export → .r3d”）：zip，内含
   ``metadata``（JSON：``K`` 列主序 3x3、``w``/``h`` RGB 尺寸、``dw``/``dh`` 深度尺寸、
   ``fps``、``poses`` 每帧 ``[qx,qy,qz,qw,tx,ty,tz]``、可选 ``frameTimestamps``）、
   ``rgbd/<i>.jpg``、``rgbd/<i>.depth``（LZFSE 压缩 float32 米，偶见 float16）、
   ``rgbd/<i>.conf``（LZFSE 压缩 uint8，0/1/2 = 低/中/高置信度）。
   解压过的同结构目录也可以。
2. “EXR + JPG” 导出：目录里有 ``metadata``/``metadata.json``、``<i>.jpg`` 和 ``<i>.exr``（深度，米），
   可能没有置信度（此时置信度全记为 1=中）。

ARKit 位姿是 OpenGL 相机约定（x 右、y 上、z 朝后）；这里乘 ``diag(1,-1,-1)`` 转成 OpenCV 约定
（x 右、y 下、z 朝前），和双目管线的 ``slam.tum``（T_world_cam）一致。世界系是 ARKit 的重力对齐世界系（y 朝上）。

输出会话目录::

    rgb.mp4            RGB（原始分辨率）
    depth.npz          depth (N,dh,dw) float16 米；conf (N,dh,dw) uint8
    calib.yaml         sensor: iphone_lidar；K_rgb、RGB 尺寸、深度尺寸
    slam.tum           ARKit 位姿（OpenCV 相机约定，T_world_cam）
    timestamps.csv     frame,timestamp_s
    metadata.json      fps、来源、设备信息

用法::

    python scripts/iphone/record3d_adapter.py capture.r3d --out /data/iphone/sess01
"""
import argparse
import json
import os
import re
import sys
import zipfile
from pathlib import Path

import numpy as np

GL_TO_CV = np.diag([1.0, -1.0, -1.0])


def quat_xyzw_to_matrix(q):
    x, y, z, w = [float(v) for v in q]
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ])


def matrix_to_quat_xyzw(r):
    r = np.asarray(r, dtype=float)
    t = np.trace(r)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        w, x, y, z = 0.25 / s, (r[2, 1] - r[1, 2]) * s, (r[0, 2] - r[2, 0]) * s, (r[1, 0] - r[0, 1]) * s
    else:
        i = int(np.argmax(np.diag(r)))
        if i == 0:
            s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
            w, x, y, z = (r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s
        elif i == 1:
            s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
            w, x, y, z = (r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
            w, x, y, z = (r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def arkit_pose_to_cv(pose7):
    """Record3D ``[qx,qy,qz,qw,tx,ty,tz]``（ARKit/OpenGL 相机）→ 4x4 T_world_cam（OpenCV 相机）。"""
    t = np.eye(4)
    t[:3, :3] = quat_xyzw_to_matrix(pose7[:4]) @ GL_TO_CV
    t[:3, 3] = [float(v) for v in pose7[4:7]]
    return t


def intrinsics_from_metadata(meta):
    """``K`` 是列主序的 9 个数（fx 在 [0]，fy 在 [4]，cx、cy 在 [6]、[7]）。"""
    k = np.asarray(meta["K"], dtype=float).reshape(3, 3).T
    if not (k[0, 0] > 0 and k[1, 1] > 0 and k[2, 2] == 1.0):
        raise ValueError("metadata.K 不像列主序内参：%s" % meta["K"])
    return k


def _decompress(blob):
    try:
        import liblzfse
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("读 .r3d 深度需要 pyliblzfse：pip install pyliblzfse") from exc
    return liblzfse.decompress(blob)


def decode_depth(blob, dh, dw):
    raw = _decompress(blob)
    n = dh * dw
    if len(raw) == 4 * n:
        arr = np.frombuffer(raw, dtype=np.float32)
    elif len(raw) == 2 * n:
        arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    else:
        raise ValueError("深度字节数 %d 与 %dx%d 对不上" % (len(raw), dw, dh))
    return arr.reshape(dh, dw).copy()


def decode_conf(blob, dh, dw):
    raw = _decompress(blob)
    if len(raw) != dh * dw:
        raise ValueError("置信度字节数 %d 与 %dx%d 对不上" % (len(raw), dw, dh))
    return np.frombuffer(raw, dtype=np.uint8).reshape(dh, dw).copy()


class _Source(object):
    """统一 zip 和目录两种来源的文件访问。"""

    def __init__(self, path):
        self.path = Path(path)
        self.zip = zipfile.ZipFile(str(path)) if self.path.is_file() else None
        names = self.zip.namelist() if self.zip else [
            str(p.relative_to(self.path)).replace(os.sep, "/") for p in self.path.rglob("*") if p.is_file()]
        self.names = [n for n in names if not n.startswith("__MACOSX")]

    def read(self, name):
        if self.zip:
            return self.zip.read(name)
        return (self.path / name).read_bytes()

    def find(self, pattern):
        rx = re.compile(pattern)
        return sorted([n for n in self.names if rx.search(n)], key=_frame_key)


def _frame_key(name):
    m = re.search(r"(\d+)\.[A-Za-z]+$", name)
    return (int(m.group(1)) if m else -1, name)


def _frame_id(name):
    return _frame_key(name)[0]


def load_record3d(path):
    """读 Record3D 导出。返回 dict：meta、K、rgb_names、depth_names、conf_names、kind、source。"""
    src = _Source(path)
    meta_names = [n for n in src.names if Path(n).name in ("metadata", "metadata.json")]
    if not meta_names:
        raise FileNotFoundError("Record3D 导出里找不到 metadata：%s" % path)
    meta = json.loads(src.read(sorted(meta_names, key=len)[0]).decode("utf-8"))
    rgb = src.find(r"\.(jpg|jpeg|png)$")
    rgb = [n for n in rgb if "conf" not in Path(n).parent.name.lower() and _frame_id(n) >= 0]
    depth_bin = {_frame_id(n): n for n in src.find(r"\.depth$")}
    depth_exr = {_frame_id(n): n for n in src.find(r"\.exr$")}
    conf = {_frame_id(n): n for n in src.find(r"\.conf$")}
    kind = "r3d" if depth_bin else ("exr" if depth_exr else None)
    if kind is None:
        raise FileNotFoundError("Record3D 导出里没有深度（*.depth 或 *.exr）：%s" % path)
    depth_map = depth_bin if kind == "r3d" else depth_exr
    rgb = [n for n in rgb if _frame_id(n) in depth_map]
    if not rgb:
        raise FileNotFoundError("没有和深度配对的 RGB 帧：%s" % path)
    ids = [_frame_id(n) for n in rgb]
    return {"meta": meta, "K": intrinsics_from_metadata(meta), "kind": kind, "source": src, "ids": ids,
            "rgb_names": rgb, "depth_names": [depth_map[i] for i in ids],
            "conf_names": [conf.get(i) for i in ids]}


def _read_exr(blob):
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    import cv2
    arr = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError("OpenCV 读不了 EXR（需要 OPENCV_IO_ENABLE_OPENEXR=1 且编译时带 OpenEXR）")
    if arr.ndim == 3:
        arr = arr[..., -1] if arr.shape[2] in (1, 2) else arr[..., 2]  # BGR 次序时 R 通道=深度
    return arr.astype(np.float32)


def convert(path, out_dir, max_frames=None, stride=1, log=print):
    """Record3D 导出 → 会话目录。返回摘要 dict。"""
    import cv2
    rec = load_record3d(path)
    meta, src = rec["meta"], rec["source"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sel = list(range(0, len(rec["ids"]), max(1, int(stride))))
    if max_frames:
        sel = sel[:int(max_frames)]
    fps = float(meta.get("fps", 30.0)) / max(1, int(stride))
    poses_raw = meta.get("poses") or []
    stamps_raw = meta.get("frameTimestamps") or []
    first = cv2.imdecode(np.frombuffer(src.read(rec["rgb_names"][sel[0]]), np.uint8), cv2.IMREAD_COLOR)
    h, w = first.shape[:2]
    K = rec["K"].copy()
    mw, mh = int(meta.get("w", w)), int(meta.get("h", h))
    if (mw, mh) != (w, h):  # K 对应 metadata 的 w/h；图像尺寸不同就按比例缩放
        K[0] *= w / float(mw)
        K[1] *= h / float(mh)
    writer = cv2.VideoWriter(str(out / "rgb.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    depths, confs, poses, stamps = [], [], [], []
    dh, dw = int(meta.get("dh", 0) or 0), int(meta.get("dw", 0) or 0)
    for k, i in enumerate(sel):
        img = first if k == 0 else cv2.imdecode(np.frombuffer(src.read(rec["rgb_names"][i]), np.uint8), cv2.IMREAD_COLOR)
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        writer.write(img)
        if rec["kind"] == "r3d":
            if not dh:
                raise ValueError("metadata 缺 dh/dw，无法解析 .depth")
            d = decode_depth(src.read(rec["depth_names"][i]), dh, dw)
            c = decode_conf(src.read(rec["conf_names"][i]), dh, dw) if rec["conf_names"][i] else np.ones((dh, dw), np.uint8)
        else:
            d = _read_exr(src.read(rec["depth_names"][i]))
            c = np.ones(d.shape, np.uint8)
        depths.append(d.astype(np.float16))
        confs.append(c)
        fid = rec["ids"][i]
        poses.append(arkit_pose_to_cv(poses_raw[fid]) if fid < len(poses_raw) else None)
        stamps.append(float(stamps_raw[fid]) if fid < len(stamps_raw) else fid / float(meta.get("fps", 30.0)))
        if k % 200 == 0:
            log("  第 %d / %d 帧" % (k, len(sel)))
    writer.release()
    t0 = stamps[0]
    stamps = [s - t0 for s in stamps]
    np.savez_compressed(out / "depth.npz", depth=np.stack(depths), conf=np.stack(confs))
    with open(out / "timestamps.csv", "w", encoding="utf-8") as fh:
        fh.write("frame,timestamp_s\n")
        for k, s in enumerate(stamps):
            fh.write("%d,%.6f\n" % (k, s))
    n_pose = sum(p is not None for p in poses)
    if n_pose:
        with open(out / "slam.tum", "w", encoding="utf-8") as fh:
            fh.write("# timestamp tx ty tz qx qy qz qw  (ARKit 世界系，OpenCV 相机约定，T_world_cam)\n")
            for s, p in zip(stamps, poses):
                if p is None:
                    continue
                q = matrix_to_quat_xyzw(p[:3, :3])
                fh.write("%.6f %.6f %.6f %.6f %.8f %.8f %.8f %.8f\n" % ((s,) + tuple(p[:3, 3]) + tuple(q)))
    dshape = depths[0].shape
    calib = {"sensor": "iphone_lidar", "image_width": w, "image_height": h,
             "K_rgb": K.tolist(), "depth_width": int(dshape[1]), "depth_height": int(dshape[0]),
             "depth_units": "m", "pose_convention": "T_world_cam, OpenCV camera (x right, y down, z forward)"}
    (out / "calib.yaml").write_text(json.dumps(calib, indent=1), encoding="utf-8")  # JSON 是合法 YAML
    md_path = out / "metadata.json"
    md = json.loads(md_path.read_text(encoding="utf-8")) if md_path.is_file() else {}
    md.update({"fps": fps, "device": "iphone_pro_lidar", "capture_app": "Record3D",
               "record3d_export": rec["kind"], "source_path": str(path),
               "record3d_camera_type": meta.get("cameraType"), "frames": len(sel)})
    md.setdefault("episode_id", "iphone/%s" % out.name)
    md_path.write_text(json.dumps(md, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"session": str(out), "frames": len(sel), "kind": rec["kind"], "rgb": [w, h],
            "depth": [int(dshape[1]), int(dshape[0])], "poses": n_pose, "fps": fps}


def is_iphone_session(path):
    p = Path(path)
    return (p / "depth.npz").is_file() and (p / "rgb.mp4").is_file()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Record3D 导出（.r3d 或 EXR+JPG 目录）→ iPhone 会话目录")
    ap.add_argument("capture")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args(argv)
    print(json.dumps(convert(a.capture, a.out, a.max_frames, a.stride), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
