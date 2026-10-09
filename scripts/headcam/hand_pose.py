# -*- coding: utf-8 -*-
"""头戴双目的手部姿态：可替换后端、三角化、SLAM 位姿。

主后端是 HaMeR（https://github.com/geopavlakos/hamer）和 WiLoR
（https://github.com/rolpotamias/WiLoR）。两者都要 MANO 的右手模型。
MANO 是马普所的非商业学术许可，权重不能放进这个仓库。拿文件的步骤见
``MANO_LICENSE``。没有 GPU、也没有这些权重时，用 MediaPipe Hands：CPU
能跑，不需要 MANO，给出的三维不是公制相机系，要靠双目三角化把尺度补上。

相机系关节再乘 SLAM 的 ``T_world_cam`` 才是规格 §7.1 的世界系标签。
没有轨迹文件时位姿是单位阵，此时坐标仍在相机系，不能当成已经完成的世界系。
"""
import math
import os
from pathlib import Path

import numpy as np

from egodata.schema import make_quaternions_continuous, rotmat_to_quat_xyzw

MANO_LICENSE = """
MANO 人体手模型（MANO_RIGHT.pkl / MANO_LEFT.pkl）使用马普所非商业学术许可，
不能再分发，也不能提交进本仓库。
1. 打开 https://mano.is.tue.mpg.de 注册，并同意 non-commercial license。
2. 下载 mano_v1_2.zip，解压出 MANO_RIGHT.pkl。HaMeR 和 WiLoR 用右手模型，左手由镜像得到。
3. export MANO_MODEL_DIR=/绝对路径/mano   （目录里放 MANO_RIGHT.pkl）

HaMeR 权重按 https://github.com/geopavlakos/hamer 的 README 下载到仓库外，然后
export HAMER_CHECKPOINT=/绝对路径/hamer.ckpt
本模块不会自动 download_models，避免把权重写进工作区。

WiLoR 权重是 CC-BY-NC-ND。从 https://huggingface.co/spaces/rolpotamias/WiLoR
取出 detector.pt、wilor_final.ckpt 和 model_config.yaml，放到仓库外：
export WILOR_CHECKPOINT=/绝对路径/wilor_final.ckpt
export WILOR_DETECTOR=/绝对路径/detector.pt
export WILOR_CONFIG=/绝对路径/model_config.yaml
安装可以是官方仓库，或较轻的 pip install "git+https://github.com/warmshao/WiLoR-mini"。
官方 demo 跑完后，把每帧 21 点写成 hands.json，交给 scripts/convert_headcam.py，
同样不需要把权重放进仓库。
""".strip()

JOINTS = 21
_SIGMA_PX = 2.0


def hamer_missing():
    """返回 HaMeR 还缺的东西。空列表表示包、权重和 MANO 目录都在。"""
    missing = []
    try:
        __import__("hamer")
    except ImportError:
        missing.append("python 包 hamer（https://github.com/geopavlakos/hamer）")
    checkpoint = os.environ.get("HAMER_CHECKPOINT")
    if not checkpoint or not Path(checkpoint).is_file():
        missing.append("环境变量 HAMER_CHECKPOINT 指向的权重文件")
    mano = os.environ.get("MANO_MODEL_DIR")
    if not mano or not Path(mano).is_dir() or not (Path(mano) / "MANO_RIGHT.pkl").is_file():
        missing.append("环境变量 MANO_MODEL_DIR（目录内要有 MANO_RIGHT.pkl）")
    return missing


def wilor_missing():
    """返回 WiLoR 还缺的东西。空列表表示包、权重、检测器和 MANO 都在。"""
    missing = []
    try:
        __import__("wilor")
    except ImportError:
        missing.append("python 包 wilor（https://github.com/rolpotamias/WiLoR）")
    checkpoint = os.environ.get("WILOR_CHECKPOINT")
    if not checkpoint or not Path(checkpoint).is_file():
        missing.append("环境变量 WILOR_CHECKPOINT 指向的 wilor_final.ckpt")
    detector = os.environ.get("WILOR_DETECTOR")
    if not detector or not Path(detector).is_file():
        missing.append("环境变量 WILOR_DETECTOR 指向的 detector.pt")
    config = os.environ.get("WILOR_CONFIG")
    if not config or not Path(config).is_file():
        missing.append("环境变量 WILOR_CONFIG 指向的 model_config.yaml")
    mano = os.environ.get("MANO_MODEL_DIR")
    if not mano or not Path(mano).is_dir() or not (Path(mano) / "MANO_RIGHT.pkl").is_file():
        missing.append("环境变量 MANO_MODEL_DIR（目录内要有 MANO_RIGHT.pkl）")
    return missing


def mediapipe_available():
    try:
        __import__("mediapipe")
        return True
    except ImportError:
        return False


def hamer_available():
    return not hamer_missing()


def wilor_available():
    return not wilor_missing()


def project_pinhole(points, K, R=None, t=None):
    """把相机系或左相机系的点投到像素。``R, t`` 把点变到该相机再投影。"""
    pts = np.asarray(points, dtype=np.float64)
    single = pts.ndim == 1
    if single:
        pts = pts.reshape(1, 3)
    matrix = np.eye(3) if R is None else np.asarray(R, dtype=np.float64)
    translation = np.zeros(3) if t is None else np.asarray(t, dtype=np.float64).reshape(3)
    camera = pts @ matrix.T + translation
    uv = np.full((pts.shape[0], 2), np.nan, dtype=np.float64)
    depth = camera[:, 2]
    valid = depth > 1e-8
    intrinsic = np.asarray(K, dtype=np.float64)
    uv[valid, 0] = intrinsic[0, 0] * camera[valid, 0] / depth[valid] + intrinsic[0, 2]
    uv[valid, 1] = intrinsic[1, 1] * camera[valid, 1] / depth[valid] + intrinsic[1, 2]
    if single:
        return uv[0]
    return uv


def _intrinsics_matrix(value):
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (3, 3):
        return array
    flat = array.reshape(-1)
    if flat.size != 4:
        raise ValueError("内参必须是 [fx, fy, cx, cy] 或 3x3")
    fx, fy, cx, cy = [float(item) for item in flat]
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _pack_calibration(K_left, K_right, dist_left, dist_right, rotation, translation, width, height):
    dist_left = np.zeros(5) if dist_left is None else np.asarray(dist_left, dtype=np.float64).reshape(-1)
    dist_right = np.zeros(5) if dist_right is None else np.asarray(dist_right, dtype=np.float64).reshape(-1)
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(translation, dtype=np.float64).reshape(-1)[:3]
    return {
        "image_width": int(width),
        "image_height": int(height),
        "K_left": np.asarray(K_left, dtype=np.float64).reshape(3, 3),
        "K_right": np.asarray(K_right, dtype=np.float64).reshape(3, 3),
        "dist_left": dist_left,
        "dist_right": dist_right,
        "R": rotation,
        "T": translation,
    }


def write_calibration_yaml(path, calib):
    """写成 PyYAML 能读的简单双目标定，测试和笔记本用。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    left = calib["K_left"]
    right = calib["K_right"]
    pose = np.eye(4)
    pose[:3, :3] = np.asarray(calib["R"], dtype=float)
    pose[:3, 3] = np.asarray(calib["T"], dtype=float).reshape(3)
    lines = [
        "image_width: %d" % int(calib["image_width"]),
        "image_height: %d" % int(calib["image_height"]),
        "left:",
        "  intrinsics: [%s]" % ", ".join("%.10g" % float(value) for value in (left[0, 0], left[1, 1], left[0, 2], left[1, 2])),
        "  dist_coeffs: [%s]" % ", ".join("%.10g" % float(value) for value in np.asarray(calib["dist_left"]).reshape(-1)),
        "right:",
        "  intrinsics: [%s]" % ", ".join("%.10g" % float(value) for value in (right[0, 0], right[1, 1], right[0, 2], right[1, 2])),
        "  dist_coeffs: [%s]" % ", ".join("%.10g" % float(value) for value in np.asarray(calib["dist_right"]).reshape(-1)),
        "T_right_left:",
    ]
    for row in pose:
        lines.append("  - [%s]" % ", ".join("%.10g" % float(value) for value in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def load_calibration(path):
    """读取 Kalibr camchain、简单 YAML，或 OpenCV FileStorage（``!!opencv-matrix``）。

    外参约定与 OpenCV ``stereoCalibrate`` 相同：``p_right = R @ p_left + T``。
    Kalibr 的 ``T_cn_cnm1`` 把 cam0 的点变到 cam1，这里 cam0 是左目、cam1 是右目。
    整机标定过程不在这个仓库里，只读已经算好的文件。
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if "opencv-matrix" in text or "cameraMatrix1:" in text:
        return _load_opencv_calibration(path)
    stripped = []
    for line in text.splitlines():
        token = line.strip()
        if token.startswith("%YAML") or token == "---":
            continue
        stripped.append(line)
    import yaml
    data = yaml.safe_load("\n".join(stripped))
    if not isinstance(data, dict):
        raise ValueError("标定文件不是映射：%s" % path)
    if "cam0" in data and "cam1" in data:
        return _load_kalibr(data)
    return _load_simple(data)


def _load_opencv_calibration(path):
    import cv2
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise ValueError("打不开 OpenCV 标定 %s" % path)

    def matrix(name):
        node = storage.getNode(name)
        if node.empty():
            return None
        return np.asarray(node.mat(), dtype=np.float64)

    def optional_int(name, fallback):
        node = storage.getNode(name)
        if node.empty():
            return fallback
        return int(round(float(node.real())))

    K_left = matrix("cameraMatrix1")
    K_right = matrix("cameraMatrix2")
    rotation = matrix("R")
    translation = matrix("T")
    if K_left is None or K_right is None or rotation is None or translation is None:
        storage.release()
        raise ValueError("OpenCV 标定缺少 cameraMatrix1、cameraMatrix2、R 或 T")
    width = optional_int("image_width", int(round(float(K_left[0, 2]) * 2)))
    height = optional_int("image_height", int(round(float(K_left[1, 2]) * 2)))
    packed = _pack_calibration(
        K_left, K_right, matrix("distCoeffs1"), matrix("distCoeffs2"),
        rotation, translation, width, height,
    )
    storage.release()
    return packed


def _load_kalibr(data):
    left = data["cam0"]
    right = data["cam1"]
    if "T_cn_cnm1" not in right:
        raise ValueError("Kalibr cam1 缺少 T_cn_cnm1")
    pose = np.asarray(right["T_cn_cnm1"], dtype=np.float64)
    resolution = left.get("resolution") or [0, 0]
    return _pack_calibration(
        _intrinsics_matrix(left["intrinsics"]),
        _intrinsics_matrix(right["intrinsics"]),
        left.get("distortion_coeffs"),
        right.get("distortion_coeffs"),
        pose[:3, :3],
        pose[:3, 3],
        int(resolution[0]),
        int(resolution[1]),
    )


def _load_simple(data):
    left = data["left"]
    right = data["right"]
    if "T_right_left" in data:
        pose = np.asarray(data["T_right_left"], dtype=np.float64)
        rotation, translation = pose[:3, :3], pose[:3, 3]
    else:
        rotation = np.eye(3)
        baseline = float(data["baseline"])
        translation = np.array([-baseline, 0.0, 0.0])
    K_left = _intrinsics_matrix(left.get("intrinsics", left.get("camera_matrix")))
    width = int(data.get("image_width", round(float(K_left[0, 2]) * 2)))
    height = int(data.get("image_height", round(float(K_left[1, 2]) * 2)))
    return _pack_calibration(
        K_left,
        _intrinsics_matrix(right.get("intrinsics", right.get("camera_matrix"))),
        left.get("dist_coeffs"),
        right.get("dist_coeffs"),
        rotation,
        translation,
        width,
        height,
    )


def _undistort(uv, K, dist):
    uv = np.asarray(uv, dtype=np.float64)
    if dist is None or np.allclose(dist, 0):
        return uv
    import cv2
    finite = np.isfinite(uv).all(axis=1)
    out = uv.copy()
    if not np.any(finite):
        return out
    points = uv[finite].reshape(-1, 1, 2)
    refined = cv2.undistortPoints(points, np.asarray(K, dtype=np.float64), np.asarray(dist, dtype=np.float64).reshape(-1), P=np.asarray(K, dtype=np.float64))
    out[finite] = refined.reshape(-1, 2)
    return out


def _dlt(P_left, P_right, uv_left, uv_right):
    design = np.zeros((4, 4), dtype=np.float64)
    design[0] = uv_left[0] * P_left[2] - P_left[0]
    design[1] = uv_left[1] * P_left[2] - P_left[1]
    design[2] = uv_right[0] * P_right[2] - P_right[0]
    design[3] = uv_right[1] * P_right[2] - P_right[1]
    _, _, vh = np.linalg.svd(design)
    homogeneous = vh[-1]
    if abs(float(homogeneous[3])) < 1e-12:
        return None
    return homogeneous[:3] / homogeneous[3]


def _bad_disparity(uv_left, uv_right, rotation, translation):
    """已校正的左右目，视差符号和基线相反时点在相机后面或无穷远。"""
    if not np.allclose(rotation, np.eye(3), atol=1e-3):
        return False
    if abs(float(translation[1])) > 1e-3 or abs(float(translation[2])) > 1e-3:
        return False
    baseline_sign = -float(translation[0])
    disparity = float(uv_left[0] - uv_right[0])
    return baseline_sign * disparity <= 0.0


def _reprojection_px(point, uv_left, uv_right, K_left, K_right, rotation, translation):
    left = project_pinhole(point, K_left)
    right = project_pinhole(point, K_right, rotation, translation)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return None
    return 0.5 * (float(np.linalg.norm(left - uv_left)) + float(np.linalg.norm(right - uv_right)))


def triangulate_pair(uv_left, uv_right, calib):
    """左右目 2D 点三角化到左相机系。返回 ``(points, confidence)``，各 (N, 3) 与 (N,)。

    置信度是重投影误差的高斯：``exp(-0.5 * (像素误差 / 2)^2)``。
    点在任一相机后面，或已校正双目的视差符号不对时，置信度为 0，坐标为 NaN。
    """
    left = np.asarray(uv_left, dtype=np.float64).reshape(-1, 2)
    right = np.asarray(uv_right, dtype=np.float64).reshape(-1, 2)
    K_left = np.asarray(calib["K_left"], dtype=np.float64)
    K_right = np.asarray(calib["K_right"], dtype=np.float64)
    rotation = np.asarray(calib["R"], dtype=np.float64)
    translation = np.asarray(calib["T"], dtype=np.float64).reshape(3)
    left = _undistort(left, K_left, calib.get("dist_left"))
    right = _undistort(right, K_right, calib.get("dist_right"))
    P_left = K_left @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P_right = K_right @ np.hstack([rotation, translation.reshape(3, 1)])
    points = np.full((left.shape[0], 3), np.nan, dtype=np.float64)
    confidence = np.zeros(left.shape[0], dtype=np.float64)
    for index in range(left.shape[0]):
        if not np.isfinite(left[index]).all() or not np.isfinite(right[index]).all():
            continue
        point = _dlt(P_left, P_right, left[index], right[index])
        if point is None:
            continue
        other = rotation @ point + translation
        if point[2] <= 1e-6 or other[2] <= 1e-6 or _bad_disparity(left[index], right[index], rotation, translation):
            continue
        error = _reprojection_px(point, left[index], right[index], K_left, K_right, rotation, translation)
        if error is None:
            continue
        points[index] = point
        confidence[index] = math.exp(-0.5 * (error / _SIGMA_PX) ** 2)
    return points, confidence


def estimate_scale(mono, metric, weights):
    """绕手腕求尺度，使 ``metric_wrist + s * (mono - mono_wrist)`` 贴近三角化结果。

    单目整体放大两倍时，返回值约为 0.5。手腕本身不参与求和。
    """
    mono = np.asarray(mono, dtype=np.float64)
    metric = np.asarray(metric, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if not np.isfinite(mono[0]).all() or not np.isfinite(metric[0]).all():
        return 1.0
    delta_mono = mono - mono[0]
    delta_metric = metric - metric[0]
    mask = (weights > 1e-3) & np.isfinite(delta_mono).all(axis=1) & np.isfinite(delta_metric).all(axis=1)
    mask[0] = False
    if not np.any(mask):
        return 1.0
    weight = weights[mask, None]
    numer = float(np.sum(weight * delta_mono[mask] * delta_metric[mask]))
    denom = float(np.sum(weight * delta_mono[mask] * delta_mono[mask]))
    if denom < 1e-12:
        return 1.0
    return numer / denom


def apply_scale(mono, metric, scale):
    mono = np.asarray(mono, dtype=np.float64)
    metric = np.asarray(metric, dtype=np.float64)
    aligned = np.full(mono.shape, np.nan, dtype=np.float64)
    if not np.isfinite(mono[0]).all() or not np.isfinite(metric[0]).all():
        return mono.copy()
    aligned = metric[0] + float(scale) * (mono - mono[0])
    return aligned


def fuse_metric_joints(mono, triangulated, confidence, threshold=0.5):
    """高置信度关节用三角化的公制坐标，其余关节用绕手腕对齐后的单目点。"""
    scale = estimate_scale(mono, triangulated, confidence)
    fused = apply_scale(mono, triangulated, scale)
    for index in range(fused.shape[0]):
        if confidence[index] >= threshold and np.isfinite(triangulated[index]).all():
            fused[index] = triangulated[index]
    return fused, scale


def correct_monocular_with_stereo(joints_mono, keypoints_left, keypoints_right, calib, threshold=0.5):
    """用左右目 2D 把单目三维收到左相机系的米制坐标。

    返回 ``(joints, confidence, scale)``。没有单目三维时，结果就是三角化点，尺度记 1。
    """
    triangulated, confidence = triangulate_pair(keypoints_left, keypoints_right, calib)
    if joints_mono is None:
        return triangulated, confidence, 1.0
    fused, scale = fuse_metric_joints(joints_mono, triangulated, confidence, threshold=threshold)
    return fused, confidence, scale


def quat_xyzw_to_rotmat(quat):
    x, y, z, w = [float(value) for value in quat]
    norm = x * x + y * y + z * z + w * w
    if norm < 1e-12:
        return np.eye(3)
    scale = 2.0 / norm
    xx, yy, zz = x * x * scale, y * y * scale, z * z * scale
    xy, xz, yz = x * y * scale, x * z * scale, y * z * scale
    wx, wy, wz = w * x * scale, w * y * scale, w * z * scale
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ], dtype=np.float64)


def load_tum_trajectory(path):
    """TUM：``timestamp tx ty tz qx qy qz qw``。返回时间数组和 4x4 列表。"""
    timestamps = []
    poses = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        timestamps.append(float(parts[0]))
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = quat_xyzw_to_rotmat(parts[4:8])
        matrix[:3, 3] = [float(parts[1]), float(parts[2]), float(parts[3])]
        poses.append(matrix)
    if not timestamps:
        raise ValueError("TUM 轨迹是空的：%s" % path)
    return np.asarray(timestamps, dtype=np.float64), poses


def associate_camera_poses(frame_timestamps, tum_path):
    """每帧取最近的 TUM 位姿。没有文件时返回单位阵，时间差为 0。

    单位阵表示坐标仍在相机系。头戴 IMU 的 csv 不在这里积分。
    """
    count = len(frame_timestamps)
    if tum_path is None or not Path(tum_path).is_file():
        return [np.eye(4, dtype=np.float64) for _ in range(count)], [0.0] * count
    times, poses = load_tum_trajectory(tum_path)
    chosen = []
    gaps = []
    for stamp in frame_timestamps:
        index = int(np.argmin(np.abs(times - float(stamp))))
        chosen.append(poses[index])
        gaps.append(abs(float(times[index]) - float(stamp)))
    return chosen, gaps


def transform_points(points, T_world_cam):
    """``p_world = T_world_cam @ p_cam``。非有限坐标保持 NaN。"""
    pts = np.asarray(points, dtype=np.float64)
    single = pts.ndim == 1
    if single:
        pts = pts.reshape(1, 3)
    pose = np.asarray(T_world_cam, dtype=np.float64)
    rotation = pose[:3, :3]
    translation = pose[:3, 3]
    out = np.full(pts.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(pts).all(axis=1)
    if np.any(finite):
        out[finite] = pts[finite] @ rotation.T + translation
    if single:
        return out[0]
    return out


def world_to_camera(points, T_world_cam):
    """``p_cam = R.T @ (p_world - t)``，与 QC 投影用的相机位姿互逆。"""
    pts = np.asarray(points, dtype=np.float64)
    single = pts.ndim == 1
    if single:
        pts = pts.reshape(1, 3)
    pose = np.asarray(T_world_cam, dtype=np.float64)
    rotation = pose[:3, :3]
    translation = pose[:3, 3]
    out = np.full(pts.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(pts).all(axis=1)
    if np.any(finite):
        out[finite] = (pts[finite] - translation) @ rotation
    if single:
        return out[0]
    return out


def wrist_rotation(joints):
    """由世界系（或相机系）21 点构造手腕旋转。列是 x、y、z 轴。

    x 从手腕指向食指 MCP（第 5 点）。掌面法向 z = (食指 MCP − 手腕) × (小指 MCP − 手腕)，
    小指 MCP 是第 17 点。y = z × x。点缺失或共线时返回 None。
    """
    joints = np.asarray(joints, dtype=np.float64)
    if joints.shape != (JOINTS, 3) or not np.isfinite(joints[[0, 5, 17]]).all():
        return None
    toward_index = joints[5] - joints[0]
    toward_pinky = joints[17] - joints[0]
    if np.linalg.norm(toward_index) < 1e-6:
        return None
    normal = np.cross(toward_index, toward_pinky)
    if np.linalg.norm(normal) < 1e-8:
        return None
    z_axis = normal / np.linalg.norm(normal)
    x_axis = toward_index / np.linalg.norm(toward_index)
    y_axis = np.cross(z_axis, x_axis)
    if np.linalg.norm(y_axis) < 1e-8:
        return None
    y_axis = y_axis / np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def wrist_poses_from_joints(joints_sequence):
    """每帧 21 点 → 手腕 xyz + xyzw。缺测帧里对应分量是 None。四元数在片段内同半球。"""
    quats = np.full((len(joints_sequence), 4), np.nan, dtype=np.float64)
    positions = []
    for index, joints in enumerate(joints_sequence):
        joints = np.asarray(joints, dtype=np.float64)
        positions.append(joints[0])
        rotation = wrist_rotation(joints)
        if rotation is not None:
            quats[index] = rotmat_to_quat_xyzw(rotation)
    quats = make_quaternions_continuous(quats)
    poses = []
    for wrist, quat in zip(positions, quats):
        if not np.isfinite(wrist).all():
            poses.append([None] * 7)
            continue
        pose = [float(wrist[0]), float(wrist[1]), float(wrist[2])]
        if not np.isfinite(quat).all():
            pose.extend([None, None, None, None])
        else:
            pose.extend(float(value) for value in quat)
        poses.append(pose)
    return poses


def mean_wrist_error_m(joints_a, joints_b):
    """两段序列第 0 个关节（手腕）的平均 L2，单位米。只统计两边都有限的帧。"""
    left = np.asarray(joints_a, dtype=np.float64)
    right = np.asarray(joints_b, dtype=np.float64)
    if left.ndim == 3:
        left = left[:, 0, :]
    if right.ndim == 3:
        right = right[:, 0, :]
    if left.shape != right.shape or left.shape[-1] != 3:
        raise ValueError("手腕序列形状不一致：%s vs %s" % (left.shape, right.shape))
    finite = np.isfinite(left).all(axis=-1) & np.isfinite(right).all(axis=-1)
    if not np.any(finite):
        return float("nan")
    return float(np.mean(np.linalg.norm(left[finite] - right[finite], axis=-1)))


def xyz_to_json(points):
    rows = []
    for point in np.asarray(points, dtype=np.float64).reshape(-1, 3):
        if not np.isfinite(point).all():
            rows.append([None, None, None])
        else:
            rows.append([float(point[0]), float(point[1]), float(point[2])])
    return rows


def empty_hand():
    return {"joints_cam": None, "keypoints_2d": None, "confidence": [0.0] * JOINTS}


def empty_prediction():
    return {"left": empty_hand(), "right": empty_hand()}


def _joints_array(value):
    if value is None:
        return None
    array = np.full((JOINTS, 3), np.nan, dtype=np.float64)
    if len(value) != JOINTS:
        raise ValueError("joints_cam 必须有 21 个点")
    for index, point in enumerate(value):
        if point is None or any(coord is None for coord in point):
            continue
        array[index] = [float(point[0]), float(point[1]), float(point[2])]
    return array


def _uv_array(value):
    if value is None:
        return None
    array = np.full((JOINTS, 2), np.nan, dtype=np.float64)
    if len(value) != JOINTS:
        raise ValueError("keypoints_2d 必须有 21 个点")
    for index, point in enumerate(value):
        if point is None or point[0] is None or point[1] is None:
            continue
        array[index] = [float(point[0]), float(point[1])]
    return array


def _confidence_vector(value):
    if value is None:
        return np.ones(JOINTS, dtype=np.float64)
    if isinstance(value, (int, float)):
        return np.full(JOINTS, float(value))
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 1:
        return np.full(JOINTS, float(array[0]))
    if array.size != JOINTS:
        raise ValueError("confidence 必须是 1 个数或 21 个数")
    return array


def _keep_hand(slot, joints, keypoints, confidence):
    score = float(np.nanmean(confidence))
    previous = slot.get("_score", -1.0)
    if score < previous:
        return
    slot["joints_cam"] = None if joints is None else np.asarray(joints, dtype=np.float64)
    slot["keypoints_2d"] = None if keypoints is None else np.asarray(keypoints, dtype=np.float64)
    slot["confidence"] = [float(value) for value in np.asarray(confidence, dtype=float).reshape(-1)]
    slot["_score"] = score


class HandPoseBackend(object):
    """``available()`` 为假时 ``predict`` 要说明缺什么，而不是返回空手当成成功。"""

    name = "base"

    def available(self):
        return False

    def predict(self, image_rgb, calib=None):
        raise NotImplementedError


class MediaPipeHandsBackend(HandPoseBackend):
    """CPU 回退。``world_landmarks`` 相对手心，不是公制相机系。

    MediaPipe 默认把输入当成自拍镜像。头戴相机朝前、画面不镜像，所以左右对调。
    手腕放在主点射线上的名义深度，供没有右目时仍能输出 21 点；有右目时由三角化替换。
    """

    name = "mediapipe"

    def __init__(self):
        self._hands = None

    def available(self):
        return mediapipe_available()

    def predict(self, image_rgb, calib=None):
        if not self.available():
            raise RuntimeError("未安装 MediaPipe。CPU 回退可执行 pip install mediapipe，不需要 MANO 许可。")
        import mediapipe as mp
        if self._hands is None:
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                model_complexity=1,
                min_detection_confidence=0.5,
            )
        image = np.asarray(image_rgb)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("MediaPipe 需要 RGB 图像")
        results = self._hands.process(image)
        prediction = empty_prediction()
        if not results.multi_hand_landmarks:
            return prediction
        height, width = image.shape[:2]
        world = results.multi_hand_world_landmarks or [None] * len(results.multi_hand_landmarks)
        handed = results.multi_handedness or []
        for landmarks, world_landmarks, hand_label in zip(results.multi_hand_landmarks, world, handed):
            label = hand_label.classification[0].label
            score = float(hand_label.classification[0].score)
            side = "right" if label == "Left" else "left"
            joints, keypoints = _lift_mediapipe(landmarks.landmark, world_landmarks, width, height, calib)
            confidence = np.full(JOINTS, score)
            for index, point in enumerate(landmarks.landmark):
                visibility = getattr(point, "visibility", None)
                if visibility is not None:
                    confidence[index] = float(visibility) * score
            _keep_hand(prediction[side], joints, keypoints, confidence)
        return prediction


def _lift_mediapipe(landmarks, world_landmarks, width, height, calib, depth=0.55):
    if calib is not None:
        intrinsic = np.asarray(calib["K_left"], dtype=np.float64)
        fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
    else:
        fx = fy = float(max(width, height))
        cx, cy = width / 2.0, height / 2.0
    keypoints = np.array([[point.x * width, point.y * height] for point in landmarks], dtype=np.float64)
    wrist_uv = keypoints[0]
    wrist = np.array([
        (wrist_uv[0] - cx) * depth / fx,
        (wrist_uv[1] - cy) * depth / fy,
        depth,
    ])
    if world_landmarks is not None:
        relative = np.array([[point.x, point.y, point.z] for point in world_landmarks.landmark], dtype=np.float64)
        relative = relative - relative[0]
    else:
        relative = project_pinhole_inverse(keypoints, fx, fy, cx, cy, depth) - wrist
    return wrist + relative, keypoints


def project_pinhole_inverse(keypoints, fx, fy, cx, cy, depth):
    keypoints = np.asarray(keypoints, dtype=np.float64)
    points = np.zeros((keypoints.shape[0], 3), dtype=np.float64)
    points[:, 0] = (keypoints[:, 0] - cx) * depth / fx
    points[:, 1] = (keypoints[:, 1] - cy) * depth / fy
    points[:, 2] = depth
    return points


class _ManoFamilyBackend(HandPoseBackend):
    """HaMeR 与 WiLoR 共用：crop 相机里的 21 点，加全图平移，得到相机系关节。"""

    def __init__(self):
        self._model = None
        self._cfg = None
        self._detector = None
        self._device = None

    def _camera_joints(self, joints, is_right, cam_translation, focal, image_shape):
        joints = np.asarray(joints, dtype=np.float64)
        if joints.shape[0] < JOINTS or joints.shape[-1] != 3:
            raise RuntimeError("模型没有给出 21 个三维关节")
        joints = joints[:JOINTS].copy()
        joints[:, 0] *= (2.0 * float(is_right) - 1.0)
        joints_cam = joints + np.asarray(cam_translation, dtype=np.float64).reshape(1, 3)
        height, width = image_shape[:2]
        intrinsic = np.array([
            [float(focal), 0.0, width / 2.0],
            [0.0, float(focal), height / 2.0],
            [0.0, 0.0, 1.0],
        ])
        keypoints = project_pinhole(joints_cam, intrinsic)
        return joints_cam, keypoints


def _scaled_focal(model_cfg, img_size):
    if hasattr(img_size, "detach"):
        values = img_size.detach().cpu().numpy()
    else:
        values = np.asarray(img_size)
    return float(model_cfg.EXTRA.FOCAL_LENGTH) / float(model_cfg.MODEL.IMAGE_SIZE) * float(np.max(values))


class HaMeRBackend(_ManoFamilyBackend):
    """官方 demo 的检测 + ViTDetDataset 路径。缺权重时说明 MANO 许可，不下载文件。"""

    name = "hamer"

    def __init__(self):
        _ManoFamilyBackend.__init__(self)
        self._pose = None

    def available(self):
        return hamer_available()

    def predict(self, image_rgb, calib=None):
        missing = hamer_missing()
        if missing:
            raise RuntimeError(MANO_LICENSE + "\n\nHaMeR 现在跑不了，缺少：\n- " + "\n- ".join(missing))
        try:
            return self._predict(image_rgb)
        except Exception as exc:
            if isinstance(exc, RuntimeError) and str(exc).startswith(MANO_LICENSE[:12]):
                raise
            raise RuntimeError(
                "HaMeR 推理失败（%s）。请用官方 demo.py 导出每帧 21 点，写成 hands.json。"
                "\n%s" % (exc, MANO_LICENSE)
            )

    def _predict(self, image_rgb):
        self._ensure()
        import cv2
        import torch
        from hamer.datasets.vitdet_dataset import ViTDetDataset
        from hamer.utils import recursive_to
        from hamer.utils.renderer import cam_crop_to_full

        image = np.asarray(image_rgb)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        boxes, rights, scores = self._detect(image, bgr)
        prediction = empty_prediction()
        if len(boxes) == 0:
            return prediction
        dataset = ViTDetDataset(self._cfg, bgr, np.stack(boxes), np.stack(rights), rescale_factor=2.0)
        loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
        cursor = 0
        for batch in loader:
            batch = recursive_to(batch, self._device)
            with torch.no_grad():
                out = self._model(batch)
            if "pred_keypoints_3d" not in out:
                raise RuntimeError("HaMeR 输出没有 pred_keypoints_3d。请用官方 demo 导出 hands.json。")
            multiplier = (2 * batch["right"] - 1)
            pred_cam = out["pred_cam"]
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            img_size = batch["img_size"].float()
            focal = _scaled_focal(self._cfg, img_size)
            cam_full = cam_crop_to_full(
                pred_cam, batch["box_center"].float(), batch["box_size"].float(), img_size, focal,
            ).detach().cpu().numpy()
            count = cam_full.shape[0]
            _fill_prediction(prediction, self, out, batch, cam_full, focal, scores[cursor:cursor + count])
            cursor += count
        return prediction

    def _ensure(self):
        if self._model is not None:
            return
        import torch
        from hamer.models import load_hamer
        from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
        from vitpose_model import ViTPoseModel

        checkpoint = os.environ["HAMER_CHECKPOINT"]
        self._model, self._cfg = load_hamer(checkpoint)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._model.to(self._device)
        self._model.eval()
        self._detector = _hamer_detector(DefaultPredictor_Lazy)
        self._pose = ViTPoseModel(self._device)

    def _detect(self, rgb, bgr):
        det_out = self._detector(bgr)
        instances = det_out["instances"]
        valid = (instances.pred_classes == 0) & (instances.scores > 0.5)
        pred_bboxes = instances.pred_boxes.tensor[valid].cpu().numpy()
        pred_scores = instances.scores[valid].cpu().numpy()
        boxes, rights, scores = [], [], []
        if pred_bboxes.shape[0] == 0:
            return boxes, rights, scores
        people = self._pose.predict_pose(rgb, [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)])
        for person in people:
            for keypoint, is_right in ((person["keypoints"][-42:-21], 0), (person["keypoints"][-21:], 1)):
                valid_joint = keypoint[:, 2] > 0.5
                if int(np.sum(valid_joint)) <= 3:
                    continue
                chosen = keypoint[valid_joint]
                boxes.append([chosen[:, 0].min(), chosen[:, 1].min(), chosen[:, 0].max(), chosen[:, 1].max()])
                rights.append(is_right)
                scores.append(float(np.mean(chosen[:, 2])))
        return boxes, rights, np.asarray(scores, dtype=np.float64)


def _hamer_detector(predictor_cls):
    from detectron2.config import LazyConfig
    import hamer
    cfg_path = Path(hamer.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
    detectron_cfg = LazyConfig.load(str(cfg_path))
    detectron_cfg.train.init_checkpoint = (
        "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    )
    for index in range(3):
        detectron_cfg.model.roi_heads.box_predictors[index].test_score_thresh = 0.25
    return predictor_cls(detectron_cfg)


def _fill_prediction(prediction, backend, out, batch, cam_full, focal, scores):
    joints_batch = out["pred_keypoints_3d"].detach().cpu().numpy()
    rights = batch["right"].detach().cpu().numpy()
    width_height = batch["img_size"].detach().cpu().numpy()
    for index in range(joints_batch.shape[0]):
        image_shape = (int(width_height[index][1]), int(width_height[index][0]))
        joints_cam, keypoints = backend._camera_joints(
            joints_batch[index], rights[index], cam_full[index], focal if np.ndim(focal) == 0 else focal[index],
            image_shape,
        )
        side = "right" if float(rights[index]) >= 0.5 else "left"
        score = 1.0 if scores is None or len(scores) <= index else float(scores[index])
        _keep_hand(prediction[side], joints_cam, keypoints, np.full(JOINTS, score))


class WiLoRBackend(_ManoFamilyBackend):
    """WiLoR demo：YOLO 手框 + ``load_wilor``。权重是 CC-BY-NC-ND，同样不入库。"""

    name = "wilor"

    def available(self):
        return wilor_available()

    def predict(self, image_rgb, calib=None):
        missing = wilor_missing()
        if missing:
            raise RuntimeError(MANO_LICENSE + "\n\nWiLoR 现在跑不了，缺少：\n- " + "\n- ".join(missing))
        try:
            return self._predict(image_rgb)
        except Exception as exc:
            if isinstance(exc, RuntimeError) and "MANO" in str(exc) and "缺少" in str(exc):
                raise
            raise RuntimeError(
                "WiLoR 推理失败（%s）。请用官方 demo.py 导出每帧 21 点，写成 hands.json。"
                "\n%s" % (exc, MANO_LICENSE)
            )

    def _predict(self, image_rgb):
        self._ensure()
        import cv2
        import torch
        from wilor.datasets.vitdet_dataset import ViTDetDataset
        from wilor.utils import recursive_to
        from wilor.utils.renderer import cam_crop_to_full

        image = np.asarray(image_rgb)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        detections = self._detector(bgr, conf=0.3, verbose=False)[0]
        boxes, rights, scores = [], [], []
        for detection in detections:
            box = detection.boxes.data.cpu().detach().squeeze().numpy()
            boxes.append(box[:4].tolist())
            rights.append(float(detection.boxes.cls.cpu().detach().squeeze().item()))
            scores.append(float(detection.boxes.conf.cpu().detach().squeeze().item()))
        prediction = empty_prediction()
        if not boxes:
            return prediction
        dataset = ViTDetDataset(self._cfg, bgr, np.stack(boxes), np.asarray(rights), rescale_factor=2.0)
        loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
        cursor = 0
        for batch in loader:
            batch = recursive_to(batch, self._device)
            with torch.no_grad():
                out = self._model(batch)
            if "pred_keypoints_3d" not in out:
                raise RuntimeError("WiLoR 输出没有 pred_keypoints_3d。")
            multiplier = (2 * batch["right"] - 1)
            pred_cam = out["pred_cam"]
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            img_size = batch["img_size"].float()
            focal = _scaled_focal(self._cfg, img_size)
            cam_full = cam_crop_to_full(
                pred_cam, batch["box_center"].float(), batch["box_size"].float(), img_size, focal,
            ).detach().cpu().numpy()
            count = cam_full.shape[0]
            _fill_prediction(prediction, self, out, batch, cam_full, focal, scores[cursor:cursor + count])
            cursor += count
        return prediction

    def _ensure(self):
        if self._model is not None:
            return
        import torch
        from ultralytics import YOLO
        from wilor.models import load_wilor
        self._model, self._cfg = load_wilor(
            checkpoint_path=os.environ["WILOR_CHECKPOINT"],
            cfg_path=os.environ["WILOR_CONFIG"],
        )
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._model.to(self._device)
        self._model.eval()
        self._detector = YOLO(os.environ["WILOR_DETECTOR"])
        self._detector.to(self._device)


def get_backend(name):
    key = (name or "mediapipe").lower()
    if key == "mediapipe":
        return MediaPipeHandsBackend()
    if key == "hamer":
        return HaMeRBackend()
    if key == "wilor":
        return WiLoRBackend()
    raise ValueError("未知手部后端 %s，可选 mediapipe、hamer、wilor" % name)

