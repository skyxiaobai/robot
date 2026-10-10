# -*- coding: utf-8 -*-
"""头戴双目的手部姿态：可替换后端、三角化、SLAM 位姿。

主后端是 HaMeR（https://github.com/geopavlakos/hamer）和 WiLoR
（https://github.com/rolpotamias/WiLoR）。两者都要 MANO 的右手模型。
MANO 是马普所的非商业学术许可，权重不能放进这个仓库。拿文件的步骤见
``MANO_LICENSE``。没有 GPU、也没有这些权重时，用 MediaPipe Hands：CPU
能跑，不需要 MANO。单目三维不是公制相机系，默认只把手腕放在 0.55 m 的
深度先验上；公制尺度要靠双目三角化。

HaMeR / WiLoR 的三维手是在虚拟焦距下预测的。公制平移用 ``calib['K_left']``
的 fx、fy 和主点；二维点仍用虚拟焦距
``EXTRA.FOCAL_LENGTH / MODEL.IMAGE_SIZE * max(宽, 高)``（1920 宽时约 37500 px）
投到全图。用短焦去投这组三维点会把手形拉歪。不传内参时平移也用虚拟焦距，
手腕会落到几十米，不是米制深度。``cam_crop_to_full``
在本文件实现。不要导入 ``hamer.utils.renderer`` / ``wilor.utils.renderer``：
它们会 ``import pyrender`` 并设置 EGL。模型默认 ``init_renderer=True``
（``wilor/models/wilor.py``、``hamer/models/hamer.py``）会构造
``MeshRenderer``，无头环境在这里失败；加载时传 ``init_renderer=False``。

mediapipe>=0.10.30 去掉了 ``mp.solutions``。后端改用 Tasks ``HandLandmarker``
（下载 ``hand_landmarker.task``）。无头环境可能缺 libEGL / libGLESv2：原生库
在导入时就会加载它们，CPU delegate 只决定推理不走 GPU。旧的
``mp.solutions.hands`` 还在时作为退路。

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
WiLoR 还要：
    pip install ultralytics==8.1.34
    pip install --no-build-isolation chumpy
pyrender 只用于网格可视化，推理不需要。无头环境不要导入 renderer（本模块已本地实现
cam_crop_to_full，并传 init_renderer=False）。没有 pyrender 时会放一个空模块，避免导入失败。
请在 Colab GPU 上跑 WiLoR。CPU 大约一秒一帧。

HaMeR 还要编译 detectron2，以及 ViTPose 用的 mmcv（常见是 mmcv==1.3.9）。这两步需要 C++ 编译器，
CPU 上更慢，不作为默认。

WiLoR 的 load_wilor 会把 MANO 路径改成相对当前目录的 ./mano_data/。本模块再用绝对路径覆盖
DATA_DIR、MODEL_PATH，不会在当前目录建 mano_data 链接。mano_mean_params.npz 不在官方 MANO
压缩包里，随 WiLoR / HaMeR 发布。查找顺序是 MANO_MODEL_DIR、wilor 或 hamer 仓库里的
mano_data/（HaMeR 还有 _DATA/data/），以及 checkpoint 旁边。找不到时 wilor_missing() 会说明，
而不是退回 ./mano_data 再抛 FileNotFoundError。

PyTorch>=2.6 的 torch.load 默认 weights_only=True，官方 checkpoint 和 YOLO detector 会加载失败。
本模块只对 WILOR_CHECKPOINT、WILOR_DETECTOR、HAMER_CHECKPOINT 这三个环境变量指向的官方文件
使用 weights_only=False，不放宽其他路径。

传入 predict(..., calib={"K_left": K}) 时，公制平移用 K 的 fx、fy 和主点 (cx, cy)。
二维关键点始终按虚拟焦距投影，传不传 K 都一样。不传 K 时平移也用约 37500 px（1920 宽），
手腕深度不是米。
官方 demo 跑完后，把每帧 21 点写成 hands.json，交给 scripts/convert_headcam.py，
同样不需要把权重放进仓库。
""".strip()

JOINTS = 21
_SIGMA_PX = 2.0
DEFAULT_WRIST_DEPTH_M = 0.55
HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
_X_TIE_PX = 1e-3


def _importable(module_name):
    try:
        __import__(module_name)
    except ImportError:
        return False
    return True


def hamer_missing():
    """返回 HaMeR 还缺的东西。空列表表示包、torch、detectron2、权重和 MANO 都在。"""
    missing = []
    try:
        __import__("hamer")
    except ImportError:
        missing.append("python 包 hamer（https://github.com/geopavlakos/hamer）")
    if not _importable("torch"):
        missing.append("python 包 torch")
    if not _importable("detectron2"):
        missing.append("python 包 detectron2（HaMeR 的人体检测，需要编译）")
    checkpoint = os.environ.get("HAMER_CHECKPOINT")
    if not checkpoint or not Path(checkpoint).is_file():
        missing.append("环境变量 HAMER_CHECKPOINT 指向的权重文件")
    mano = os.environ.get("MANO_MODEL_DIR")
    if not mano or not Path(mano).is_dir() or not (Path(mano) / "MANO_RIGHT.pkl").is_file():
        missing.append("环境变量 MANO_MODEL_DIR（目录内要有 MANO_RIGHT.pkl）")
    if mano_mean_params_path() is None:
        missing.append(_mean_params_missing_message())
    return missing


def wilor_missing():
    """返回 WiLoR 还缺的东西。空列表表示包、torch、ultralytics、权重、检测器和 MANO 都在。"""
    missing = []
    try:
        __import__("wilor")
    except ImportError:
        missing.append("python 包 wilor（https://github.com/rolpotamias/WiLoR）")
    if not _importable("torch"):
        missing.append("python 包 torch")
    if not _importable("ultralytics"):
        missing.append("python 包 ultralytics（WiLoR 的 YOLO 检测器，建议 ultralytics==8.1.34）")
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
    if mano_mean_params_path() is None:
        missing.append(_mean_params_missing_message())
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
    """同一侧只留更高分的一只。两只手被标成同一侧时，先用 ``_place_detections`` 按图像 x 拆开。"""
    score = float(np.nanmean(confidence))
    previous = slot.get("_score", -1.0)
    if score < previous:
        return
    slot["joints_cam"] = None if joints is None else np.asarray(joints, dtype=np.float64)
    slot["keypoints_2d"] = None if keypoints is None else np.asarray(keypoints, dtype=np.float64)
    slot["confidence"] = [float(value) for value in np.asarray(confidence, dtype=float).reshape(-1)]
    slot["_score"] = score


def _split_same_side(side, group):
    """同一标签的两只手：图像 x 小的是左手，大的是右手。x 分不开时高分留在原标签。"""
    ordered = sorted(group, key=lambda item: (item["wrist_x"], -item["score"]))
    span = float(ordered[-1]["wrist_x"] - ordered[0]["wrist_x"])
    if span <= _X_TIE_PX:
        ranked = sorted(enumerate(group), key=lambda pair: (-pair[1]["score"], pair[0]))
        other = "left" if side == "right" else "right"
        assigned = []
        for index, (_, item) in enumerate(ranked):
            assigned.append((side if index == 0 else other, item))
        return assigned
    return [("left", ordered[0]), ("right", ordered[-1])]


def _place_detections(prediction, detections):
    groups = {}
    for item in detections:
        groups.setdefault(item["side"], []).append(item)
    assigned = []
    for side, group in groups.items():
        if len(group) == 1:
            assigned.append((side, group[0]))
        else:
            assigned.extend(_split_same_side(side, group))
    for side, item in assigned:
        _keep_hand(prediction[side], item["joints"], item["keypoints"], item["confidence"])
    return prediction


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
    单目没有米制深度：手腕放在主点射线上的深度先验 ``wrist_depth_m``（默认 0.55 m），
    只为没有右目时仍能输出 21 点。公制深度要靠双目三角化，不能把这个先验当成真值。
    EgoDex 手腕深度中位大约 0.29 m，沿用 0.55 m 会把相机系手腕拉开大约一倍。

    mediapipe>=0.10.30（Python 3.13 / 当前 Colab 只能装到这些版本）已经去掉
    ``mp.solutions``。这里优先用 Tasks ``HandLandmarker``，并下载
    ``hand_landmarker.task``。无头机器加载原生库时可能缺 ``libEGL.so.1``
    或 ``libGLESv2.so.2``（CPU delegate 也避不开这次加载）：安装
    libgl1、libegl1、libgles2、libglib2.0-0。推理默认
    ``delegate='cpu'``（``BaseOptions.Delegate.CPU``）。Tasks 初始化失败
    且旧的 ``mp.solutions.hands`` 还在时，退回旧接口。
    """

    name = "mediapipe"

    def __init__(self, wrist_depth_m=DEFAULT_WRIST_DEPTH_M, model_path=None, delegate=None):
        if float(wrist_depth_m) <= 0.0:
            raise ValueError("手腕深度先验必须为正，单位米")
        if delegate is None:
            delegate = os.environ.get("MEDIAPIPE_HAND_DELEGATE", "cpu")
        delegate = str(delegate).lower()
        if delegate not in ("cpu", "gpu"):
            raise ValueError("MediaPipe delegate 只能是 cpu 或 gpu")
        self.wrist_depth_m = float(wrist_depth_m)
        self.model_path = model_path
        self.delegate = delegate
        self._runner = None

    def available(self):
        return mediapipe_available()

    def predict(self, image_rgb, calib=None):
        if not self.available():
            raise RuntimeError("未安装 MediaPipe。CPU 回退可执行 pip install mediapipe，不需要 MANO 许可。")
        import mediapipe as mp
        image = np.asarray(image_rgb)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("MediaPipe 需要 RGB 图像")
        if self._runner is None:
            self._runner = open_mediapipe_detector(
                mp, model_path=self.model_path, delegate=self.delegate,
            )
        kind, detector = self._runner
        height, width = image.shape[:2]
        if kind == "tasks":
            result = detector.detect(_tasks_image(mp, image))
            return prediction_from_tasks_result(result, width, height, calib, self.wrist_depth_m)
        results = detector.process(image)
        return prediction_from_solutions_result(results, width, height, calib, self.wrist_depth_m)


def _tasks_image(mp, image):
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image))


def _has_tasks_api(mp):
    tasks = getattr(mp, "tasks", None)
    vision = getattr(tasks, "vision", None)
    return (
        getattr(vision, "HandLandmarker", None) is not None
        and getattr(vision, "HandLandmarkerOptions", None) is not None
        and getattr(tasks, "BaseOptions", None) is not None
    )


def _has_solutions_api(mp):
    solutions = getattr(mp, "solutions", None)
    hands = getattr(solutions, "hands", None)
    return getattr(hands, "Hands", None) is not None


def choose_mediapipe_api(has_tasks, has_solutions):
    """有 Tasks 就用 Tasks。只有旧的 ``mp.solutions.hands`` 时才退回去。"""
    if has_tasks:
        return "tasks"
    if has_solutions:
        return "solutions"
    raise RuntimeError(
        mediapipe_tasks_failure_message(RuntimeError("没有 HandLandmarker，也没有 mp.solutions.hands"))
    )


def mediapipe_gl_libraries_missing(exc):
    """Tasks 初始化是否因为缺少 libEGL / libGLESv2 而失败。

    无头机器没有这两份动态库时，空白帧测试应跳过，不当成手部后端的回归。
    """
    text = "%s\n%s" % (exc, getattr(exc, "__cause__", "") or "")
    return "libGLESv2" in text or "libEGL" in text


def mediapipe_tasks_failure_message(exc):
    return (
        "MediaPipe Tasks HandLandmarker 初始化失败（%s）。"
        "mediapipe>=0.10.30 已去掉 mp.solutions，需要 Tasks API。"
        "无头环境经常缺少 libEGL.so.1 或 libGLESv2.so.2：安装 libgl1、libegl1、libgles2、libglib2.0-0。"
        "CPU delegate（默认 delegate='cpu'，或环境变量 MEDIAPIPE_HAND_DELEGATE=cpu）"
        "只让推理不走 GPU，不能代替这两份动态库。"
        "模型可放到 MEDIAPIPE_HAND_LANDMARKER。若本机仍有 mp.solutions.hands，会自动退回旧接口。"
        % exc
    )


def ensure_hand_landmarker_model(path=None):
    """返回本地 ``hand_landmarker.task``。没有文件时从官方地址下载到缓存。"""
    if path is not None:
        candidate = Path(path)
        if not candidate.is_file():
            raise FileNotFoundError("找不到 HandLandmarker 模型：%s" % candidate)
        return candidate
    env = os.environ.get("MEDIAPIPE_HAND_LANDMARKER")
    if env:
        candidate = Path(env)
        if candidate.is_file():
            return candidate
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    cache = cache_root / "robot" / "hand_landmarker.task"
    if cache.is_file() and cache.stat().st_size > 0:
        return cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request
    request = urllib.request.Request(HAND_LANDMARKER_URL, headers={"User-Agent": "robot-hand-pose"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    temporary = cache.with_suffix(".task.partial")
    temporary.write_bytes(payload)
    temporary.replace(cache)
    return cache


def _create_tasks_landmarker(mp, model_path, delegate):
    base = mp.tasks.BaseOptions
    chosen = base.Delegate.CPU if delegate == "cpu" else base.Delegate.GPU
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base(model_asset_path=str(model_path), delegate=chosen),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def _create_solutions_hands(mp):
    return mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
    )


def open_mediapipe_detector(mp, model_path=None, delegate="cpu"):
    """返回 ``(kind, detector)``。Tasks 创建失败且旧接口还在时退回 solutions。"""
    delegate_name = str(delegate or "cpu").lower()
    if delegate_name not in ("cpu", "gpu"):
        raise ValueError("MediaPipe delegate 只能是 cpu 或 gpu")
    has_tasks = _has_tasks_api(mp)
    has_solutions = _has_solutions_api(mp)
    kind = choose_mediapipe_api(has_tasks, has_solutions)
    if kind == "tasks":
        try:
            resolved = ensure_hand_landmarker_model(model_path)
            return "tasks", _create_tasks_landmarker(mp, resolved, delegate_name)
        except Exception as exc:
            if not has_solutions:
                raise RuntimeError(mediapipe_tasks_failure_message(exc)) from exc
    return "solutions", _create_solutions_hands(mp)


def _flipped_side(label):
    text = str(label or "").strip().lower()
    if text.startswith("left"):
        return "right"
    if text.startswith("right"):
        return "left"
    return "left"


def _landmark_confidence(landmarks, score, trust_visibility):
    confidence = np.full(JOINTS, float(score))
    if not trust_visibility:
        return confidence
    for index, point in enumerate(landmarks):
        if index >= JOINTS:
            break
        visibility = getattr(point, "visibility", None)
        if visibility is not None:
            confidence[index] = float(visibility) * float(score)
    return confidence


def _world_points(world_landmarks):
    if world_landmarks is None:
        return None
    points = world_landmarks.landmark if hasattr(world_landmarks, "landmark") else world_landmarks
    if points is None:
        return None
    return np.array([[point.x, point.y, point.z] for point in points], dtype=np.float64)


def prediction_from_mediapipe_hands(labeled_hands, width, height, calib, wrist_depth_m, trust_visibility=False):
    detections = []
    for label, score, landmarks, world in labeled_hands:
        joints, keypoints = _lift_mediapipe(
            landmarks, world, width, height, calib, depth=wrist_depth_m,
        )
        detections.append({
            "side": _flipped_side(label),
            "score": float(score),
            "wrist_x": float(keypoints[0, 0]),
            "joints": joints,
            "keypoints": keypoints,
            "confidence": _landmark_confidence(landmarks, score, trust_visibility),
        })
    return _place_detections(empty_prediction(), detections)


def _iter_solutions_hands(results):
    landmarks_list = getattr(results, "multi_hand_landmarks", None) or []
    if not landmarks_list:
        return
    world_list = getattr(results, "multi_hand_world_landmarks", None) or [None] * len(landmarks_list)
    handed = getattr(results, "multi_handedness", None) or []
    for index, landmarks in enumerate(landmarks_list):
        label, score = "Left", 0.0
        if index < len(handed) and handed[index] is not None:
            classification = handed[index].classification[0]
            label = classification.label
            score = float(classification.score)
        world = world_list[index] if index < len(world_list) else None
        yield label, score, landmarks.landmark, world


def _iter_tasks_hands(result):
    landmarks_list = getattr(result, "hand_landmarks", None) or []
    if not landmarks_list:
        return
    world_list = getattr(result, "hand_world_landmarks", None) or [None] * len(landmarks_list)
    handed_list = getattr(result, "handedness", None) or []
    for index, landmarks in enumerate(landmarks_list):
        label, score = "Left", 0.0
        if index < len(handed_list) and handed_list[index]:
            category = handed_list[index][0]
            label = getattr(category, "category_name", None) or getattr(category, "display_name", "Left")
            score = float(getattr(category, "score", 0.0))
        world = world_list[index] if index < len(world_list) else None
        yield label, score, landmarks, world


def prediction_from_solutions_result(results, width, height, calib, wrist_depth_m):
    return prediction_from_mediapipe_hands(
        list(_iter_solutions_hands(results)), width, height, calib, wrist_depth_m, trust_visibility=True,
    )


def prediction_from_tasks_result(result, width, height, calib, wrist_depth_m):
    return prediction_from_mediapipe_hands(
        list(_iter_tasks_hands(result)), width, height, calib, wrist_depth_m, trust_visibility=False,
    )


def _lift_mediapipe(landmarks, world_landmarks, width, height, calib, depth=DEFAULT_WRIST_DEPTH_M):
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
    relative_world = _world_points(world_landmarks)
    if relative_world is not None:
        relative = relative_world - relative_world[0]
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

    def _camera_joints(
        self, joints, is_right, cam_translation, image_shape,
        project_translation=None, project_focal=None, project_focal_y=None, project_principal=None,
    ):
        """公制关节用 ``cam_translation``。二维点用虚拟相机，不用真实短焦。

        WiLoR / HaMeR 的手形是在约 37500 px 的虚拟焦距下预测的。把公制关节
        （深度已经按真实 fx 缩到几十厘米）再用短焦投影，会把手形拉歪。
        ``project_translation`` 与 ``project_focal`` 来自虚拟焦距的 crop→全图，
        和有没有传入 K 无关。
        """
        joints = np.asarray(joints, dtype=np.float64)
        if joints.shape[0] < JOINTS or joints.shape[-1] != 3:
            raise RuntimeError("模型没有给出 21 个三维关节")
        if project_focal is None:
            raise ValueError("二维投影必须给出虚拟焦距，不能用真实 K 的短焦")
        joints = joints[:JOINTS].copy()
        joints[:, 0] *= (2.0 * float(is_right) - 1.0)
        joints_cam = joints + np.asarray(cam_translation, dtype=np.float64).reshape(1, 3)
        if project_translation is None:
            project_translation = cam_translation
        pixels = joints + np.asarray(project_translation, dtype=np.float64).reshape(1, 3)
        height, width = image_shape[:2]
        fx = float(project_focal)
        fy = fx if project_focal_y is None else float(project_focal_y)
        if project_principal is None:
            cx, cy = width / 2.0, height / 2.0
        else:
            cx, cy = float(project_principal[0]), float(project_principal[1])
        intrinsic = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ])
        keypoints = project_pinhole(pixels, intrinsic)
        return joints_cam, keypoints


def _scaled_focal(model_cfg, img_size):
    if hasattr(img_size, "detach"):
        values = img_size.detach().cpu().numpy()
    else:
        values = np.asarray(img_size)
    return float(model_cfg.EXTRA.FOCAL_LENGTH) / float(model_cfg.MODEL.IMAGE_SIZE) * float(np.max(values))


def _as_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def cam_crop_to_full(
    cam_bbox, box_center, box_size, img_size, focal_length=5000.0, principal_point=None, focal_y=None,
):
    """crop 相机 ``(scale, tx, ty)`` 变到全图相机平移。

    ``principal_point is None`` 且 ``focal_y is None`` 时与 HaMeR / WiLoR
    ``utils.renderer.cam_crop_to_full`` 一致：主点在每行图像中心，单一焦距。
    传入标定时 ``focal_length`` 是 fx，``focal_y`` 是 fy，主点是 K 的 ``(cx, cy)``。
    ty 按 fx/fy 缩放；两者相等时与上游公式相同。不导入 renderer，避免 pyrender / EGL。
    """
    cam = np.asarray(_as_numpy(cam_bbox), dtype=np.float64)
    center = np.asarray(_as_numpy(box_center), dtype=np.float64)
    size = np.asarray(_as_numpy(box_size), dtype=np.float64).reshape(-1)
    image = np.asarray(_as_numpy(img_size), dtype=np.float64)
    if cam.ndim == 1:
        cam = cam.reshape(1, -1)
    if center.ndim == 1:
        center = center.reshape(1, -1)
    if image.ndim == 1:
        image = image.reshape(1, -1)
    img_w, img_h = image[:, 0], image[:, 1]
    cx, cy = center[:, 0], center[:, 1]
    if principal_point is None:
        px = img_w / 2.0
        py = img_h / 2.0
    else:
        principal = np.asarray(_as_numpy(principal_point), dtype=np.float64).reshape(-1)
        px = np.full(cam.shape[0], float(principal[0]))
        py = np.full(cam.shape[0], float(principal[1]))
    fx = float(focal_length)
    fy = fx if focal_y is None else float(focal_y)
    bs = size * cam[:, 0] + 1e-9
    tz = 2.0 * fx / bs
    tx = (2.0 * (cx - px) / bs) + cam[:, 1]
    ty = (2.0 * (fx / fy) * (cy - py) / bs) + cam[:, 2]
    return np.stack([tx, ty, tz], axis=-1)


def intrinsics_for_crop(calib, img_size, model_cfg):
    """有 ``K_left`` 时返回 fx、fy 和主点；否则用虚拟焦距，主点留给图像中心。"""
    if calib is not None and calib.get("K_left") is not None:
        matrix = np.asarray(calib["K_left"], dtype=np.float64)
        return float(matrix[0, 0]), float(matrix[1, 1]), (float(matrix[0, 2]), float(matrix[1, 2]))
    focal = _scaled_focal(model_cfg, img_size)
    return focal, focal, None


def _full_camera_from_batch(pred_cam, batch, model_cfg, calib):
    """返回公制平移、虚拟焦距下的平移，以及二维投影用的虚拟焦距。

    有 ``K_left`` 时公制平移用真实 fx、fy 和主点。二维相机始终用虚拟焦距和图像中心，
    因此传不传 K，二维点相同。
    """
    img_size = batch["img_size"].float() if hasattr(batch["img_size"], "float") else batch["img_size"]
    box_center = batch["box_center"].float() if hasattr(batch["box_center"], "float") else batch["box_center"]
    box_size = batch["box_size"].float() if hasattr(batch["box_size"], "float") else batch["box_size"]
    project_focal = _scaled_focal(model_cfg, img_size)
    cam_pixels = cam_crop_to_full(
        pred_cam, box_center, box_size, img_size, focal_length=project_focal,
    )
    if calib is not None and calib.get("K_left") is not None:
        fx, fy, principal = intrinsics_for_crop(calib, img_size, model_cfg)
        cam_metric = cam_crop_to_full(
            pred_cam, box_center, box_size, img_size,
            focal_length=fx, principal_point=principal, focal_y=fy,
        )
    else:
        cam_metric = cam_pixels
    return cam_metric, cam_pixels, project_focal


def recursive_to(value, target):
    """把 batch 里的张量送到设备。不从 ``wilor.utils`` / ``hamer.utils`` 导入，以免带上 renderer。"""
    if isinstance(value, dict):
        return {key: recursive_to(item, target) for key, item in value.items()}
    if isinstance(value, list):
        return [recursive_to(item, target) for item in value]
    if hasattr(value, "to") and hasattr(value, "detach"):
        return value.to(target)
    return value


_OFFICIAL_WEIGHT_ENVS = ("WILOR_CHECKPOINT", "WILOR_DETECTOR", "HAMER_CHECKPOINT")


def official_weight_paths():
    found = set()
    for name in _OFFICIAL_WEIGHT_ENVS:
        value = os.environ.get(name)
        if value:
            found.add(os.path.realpath(value))
    return found


def _torch_load_path(source):
    if isinstance(source, (str, os.PathLike)):
        return os.fspath(source)
    name = getattr(source, "name", None)
    if isinstance(name, str) and name and not name.startswith("<"):
        return name
    return None


class _OfficialTorchLoad:
    def __init__(self, torch_module, allowed):
        self._torch = torch_module
        self._allowed = allowed
        self._original = torch_module.load

    def __enter__(self):
        allowed = self._allowed
        original = self._original

        def load(source, *args, **kwargs):
            path = _torch_load_path(source)
            if path is not None and os.path.realpath(path) in allowed:
                kwargs = dict(kwargs)
                kwargs["weights_only"] = False
            return original(source, *args, **kwargs)

        self._torch.load = load
        return self

    def __exit__(self, exc_type, exc, tb):
        self._torch.load = self._original
        return False


def allow_official_torch_load(paths=None, torch_module=None):
    """只对官方权重文件调用 ``torch.load(..., weights_only=False)``。

    PyTorch>=2.6 默认 ``weights_only=True``。Lightning 读 checkpoint、ultralytics
    读 YOLO ``.pt`` 会因此失败。这里不改全局默认，也不放宽名单以外的文件。
    默认名单是 ``WILOR_CHECKPOINT``、``WILOR_DETECTOR``、``HAMER_CHECKPOINT``
    的真实路径。这些文件来自 WiLoR / HaMeR 的官方发布。
    """
    if torch_module is None:
        import torch
        torch_module = torch
    if paths is None:
        allowed = official_weight_paths()
    else:
        allowed = {os.path.realpath(path) for path in paths}
    return _OfficialTorchLoad(torch_module, allowed)


_MEAN_PARAMS_NAME = "mano_mean_params.npz"


def _mean_params_missing_message():
    return (
        "mano_mean_params.npz（官方 MANO 压缩包里没有这个文件，它随 WiLoR / HaMeR 发布。"
        "放到 MANO_MODEL_DIR，或留在 WiLoR 仓库的 mano_data/、HaMeR 的 _DATA/data/，"
        "也可以放在 checkpoint 旁边）"
    )


def _mean_param_candidates():
    """官方 MANO zip 不含均值参数。先看 MANO_MODEL_DIR，再看 WiLoR / HaMeR 仓库。"""
    candidates = []
    mano_dir = os.environ.get("MANO_MODEL_DIR")
    if mano_dir:
        candidates.append(Path(mano_dir) / _MEAN_PARAMS_NAME)
    for package in ("wilor", "hamer"):
        try:
            module = __import__(package)
        except ImportError:
            continue
        origin = getattr(module, "__file__", None)
        if not origin:
            continue
        root = Path(origin).resolve().parent
        candidates.extend([
            root / "mano_data" / _MEAN_PARAMS_NAME,
            root.parent / "mano_data" / _MEAN_PARAMS_NAME,
            root.parent / "_DATA" / "data" / _MEAN_PARAMS_NAME,
            root.parent / "data" / _MEAN_PARAMS_NAME,
        ])
    for env_name in ("WILOR_CONFIG", "WILOR_CHECKPOINT", "HAMER_CHECKPOINT"):
        value = os.environ.get(env_name)
        if not value:
            continue
        parent = Path(value).resolve().parent
        candidates.extend([
            parent / _MEAN_PARAMS_NAME,
            parent / "mano_data" / _MEAN_PARAMS_NAME,
            parent.parent / "mano_data" / _MEAN_PARAMS_NAME,
            parent.parent / "_DATA" / "data" / _MEAN_PARAMS_NAME,
        ])
    return candidates


def mano_mean_params_path():
    """返回 ``mano_mean_params.npz`` 的绝对路径。找不到则是 None。"""
    seen = set()
    for path in _mean_param_candidates():
        key = os.path.normpath(str(path))
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            return path.resolve()
    return None


def apply_mano_model_dir(cfg):
    """把配置里的 MANO 路径改成绝对路径，不再依赖当前目录的 ``./mano_data``。"""
    if not hasattr(cfg, "MANO"):
        return cfg
    mano_dir = os.environ.get("MANO_MODEL_DIR")
    mean = mano_mean_params_path()
    if not mano_dir and mean is None:
        return cfg
    defrost = getattr(cfg, "defrost", None)
    freeze = getattr(cfg, "freeze", None)
    if defrost is not None:
        defrost()
    mano = cfg.MANO
    if mano_dir:
        root = str(Path(mano_dir).resolve())
        mano.DATA_DIR = root
        mano.MODEL_PATH = root
    if hasattr(mano, "MEAN_PARAMS"):
        if mean is not None:
            mano.MEAN_PARAMS = str(mean)
        elif mano_dir:
            mano.MEAN_PARAMS = str(Path(mano_dir).resolve() / _MEAN_PARAMS_NAME)
    if freeze is not None:
        freeze()
    return cfg


def load_checkpoint_without_renderer(load_from_checkpoint, *args, **kwargs):
    """传 ``init_renderer=False``，并在构造模型前写上 MANO 绝对路径。"""
    kwargs = dict(kwargs)
    kwargs["init_renderer"] = False
    cfg = kwargs.get("cfg")
    if cfg is not None:
        apply_mano_model_dir(cfg)
    return load_from_checkpoint(*args, **kwargs)


def _call_without_renderer(model_cls, loader, *args, **kwargs):
    original = model_cls.load_from_checkpoint

    def wrapped(*w_args, **w_kwargs):
        return load_checkpoint_without_renderer(original, *w_args, **w_kwargs)

    model_cls.load_from_checkpoint = wrapped
    try:
        return loader(*args, **kwargs)
    finally:
        model_cls.load_from_checkpoint = original


def install_pyrender_stub():
    """放入最小的 pyrender，让 wilor/hamer 能导入 renderer 模块。

    ``wilor/models/wilor.py`` 和 ``hamer/models/hamer.py`` 在导入时执行
    ``from ..utils import SkeletonRenderer, MeshRenderer``，``utils/__init__.py``
    再导入 renderer。推理不实例化它们：``init_renderer=False`` 跳过
    ``MeshRenderer`` 里的 ``OffscreenRenderer``（那个构造需要 EGL）。
    本桩不能画网格。需要可视化时再安装真正的 pyrender。
    """
    import sys
    import types
    stub = types.ModuleType("pyrender")

    class _OffscreenRenderer:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("pyrender 未安装。手部推理不需要渲染器；需要网格可视化时再安装 pyrender。")

        def render(self, *args, **kwargs):
            raise RuntimeError("pyrender 未安装")

        def delete(self):
            return None

    class _RenderFlags:
        RGBA = 1
        SHADOWS_ALL = 2

    def _factory(*args, **kwargs):
        return types.SimpleNamespace()

    stub.OffscreenRenderer = _OffscreenRenderer
    stub.MetallicRoughnessMaterial = _factory
    stub.Mesh = types.SimpleNamespace(from_trimesh=_factory)
    stub.Node = _factory
    stub.Scene = _factory
    stub.IntrinsicsCamera = _factory
    stub.DirectionalLight = _factory
    stub.PointLight = _factory
    stub.RenderFlags = _RenderFlags
    stub._hand_pose_stub = True
    sys.modules["pyrender"] = stub
    return stub


def ensure_pyrender_importable():
    """pyrender 缺失或无头 EGL 导入失败时安装空模块。已成功导入则不动。"""
    import sys
    existing = sys.modules.get("pyrender")
    if existing is not None and not getattr(existing, "_hand_pose_stub", False):
        return False
    try:
        __import__("pyrender")
    except Exception:
        for name in list(sys.modules):
            if name == "pyrender" or name.startswith("pyrender."):
                sys.modules.pop(name, None)
        install_pyrender_stub()
        return True
    return False


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
            return self._predict(image_rgb, calib)
        except Exception as exc:
            if isinstance(exc, RuntimeError) and str(exc).startswith(MANO_LICENSE[:12]):
                raise
            raise RuntimeError(
                "HaMeR 推理失败（%s）。请用官方 demo.py 导出每帧 21 点，写成 hands.json。"
                "\n%s" % (exc, MANO_LICENSE)
            )

    def _predict(self, image_rgb, calib=None):
        self._ensure()
        import cv2
        import torch
        from hamer.datasets.vitdet_dataset import ViTDetDataset

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
            cam_metric, cam_pixels, project_focal = _full_camera_from_batch(pred_cam, batch, self._cfg, calib)
            count = cam_metric.shape[0]
            _fill_prediction(
                prediction, self, out, batch, cam_metric, cam_pixels, project_focal,
                scores[cursor:cursor + count],
            )
            cursor += count
        return prediction

    def _ensure(self):
        if self._model is not None:
            return
        import torch
        ensure_pyrender_importable()
        from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
        from vitpose_model import ViTPoseModel
        from hamer.models import load_hamer
        from hamer.models.hamer import HAMER

        checkpoint = os.environ["HAMER_CHECKPOINT"]
        with allow_official_torch_load(torch_module=torch):
            self._model, self._cfg = _call_without_renderer(HAMER, load_hamer, checkpoint)
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


def _fill_prediction(prediction, backend, out, batch, cam_metric, cam_pixels, project_focal, scores):
    joints_batch = out["pred_keypoints_3d"].detach().cpu().numpy()
    rights = batch["right"].detach().cpu().numpy()
    width_height = batch["img_size"].detach().cpu().numpy()
    detections = []
    for index in range(joints_batch.shape[0]):
        image_shape = (int(width_height[index][1]), int(width_height[index][0]))
        joints_cam, keypoints = backend._camera_joints(
            joints_batch[index], rights[index], cam_metric[index], image_shape,
            project_translation=cam_pixels[index], project_focal=project_focal,
        )
        side = "right" if float(rights[index]) >= 0.5 else "left"
        score = 1.0 if scores is None or len(scores) <= index else float(scores[index])
        detections.append({
            "side": side,
            "score": score,
            "wrist_x": float(keypoints[0, 0]),
            "joints": joints_cam,
            "keypoints": keypoints,
            "confidence": np.full(JOINTS, score),
        })
    _place_detections(prediction, detections)


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
            return self._predict(image_rgb, calib)
        except Exception as exc:
            if isinstance(exc, RuntimeError) and "MANO" in str(exc) and "缺少" in str(exc):
                raise
            raise RuntimeError(
                "WiLoR 推理失败（%s）。请用官方 demo.py 导出每帧 21 点，写成 hands.json。"
                "\n%s" % (exc, MANO_LICENSE)
            )

    def _predict(self, image_rgb, calib=None):
        self._ensure()
        import cv2
        import torch
        from wilor.datasets.vitdet_dataset import ViTDetDataset

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
            cam_metric, cam_pixels, project_focal = _full_camera_from_batch(pred_cam, batch, self._cfg, calib)
            count = cam_metric.shape[0]
            _fill_prediction(
                prediction, self, out, batch, cam_metric, cam_pixels, project_focal,
                scores[cursor:cursor + count],
            )
            cursor += count
        return prediction

    def _ensure(self):
        if self._model is not None:
            return
        import torch
        ensure_pyrender_importable()
        from ultralytics import YOLO
        from wilor.models import load_wilor
        from wilor.models.wilor import WiLoR

        with allow_official_torch_load(torch_module=torch):
            self._model, self._cfg = _call_without_renderer(
                WiLoR, load_wilor,
                checkpoint_path=os.environ["WILOR_CHECKPOINT"],
                cfg_path=os.environ["WILOR_CONFIG"],
            )
            self._detector = YOLO(os.environ["WILOR_DETECTOR"])
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._model.to(self._device)
        self._model.eval()
        self._detector.to(self._device)


def refine_tracked_hands(frames, params=None, timestamps=None, camera_poses=None):
    """对一整段检测做可选的时序精修。默认关闭，要调用才会改结果。

    实现在 ``headcam.hand_track_refine``。不传 ``params`` 时用 One Euro、最多补 5 帧、
    固定骨长，并做左右手轨迹一致性。``frames`` 与 ``predict`` 的返回值相同。
    """
    from headcam.hand_track_refine import RefineParams, refine_prediction_sequence

    if params is None:
        params = RefineParams(
            smooth="one_euro",
            gap_fill=True,
            max_gap=5,
            fixed_shape=True,
            lr_consistency=True,
        )
    return refine_prediction_sequence(
        frames, params, timestamps=timestamps, camera_poses=camera_poses,
    )


def get_backend(name, wrist_depth_m=DEFAULT_WRIST_DEPTH_M):
    key = (name or "mediapipe").lower()
    if key == "mediapipe":
        return MediaPipeHandsBackend(wrist_depth_m=wrist_depth_m)
    if key == "hamer":
        return HaMeRBackend()
    if key == "wilor":
        return WiLoRBackend()
    raise ValueError("未知手部后端 %s，可选 mediapipe、hamer、wilor" % name)


def _coerce_joints(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray) and value.shape == (JOINTS, 3) and value.dtype != object:
        return np.asarray(value, dtype=np.float64)
    array = np.full((JOINTS, 3), np.nan, dtype=np.float64)
    for index, point in enumerate(value):
        if point is None:
            continue
        coords = list(point)
        if len(coords) != 3 or any(coord is None for coord in coords):
            continue
        array[index] = coords
    return array


def _inside_image(uv, width, height):
    return bool(np.isfinite(uv).all() and 0.0 <= float(uv[0]) < width and 0.0 <= float(uv[1]) < height)


def _gt_hand_visible(joints, uv, width, height, confidence, min_visible_fraction):
    if joints is None or not np.isfinite(joints[0]).all():
        return False
    if confidence is not None and float(confidence) < 0.5:
        return False
    if not _inside_image(uv[0], width, height):
        return False
    inside = 0
    for index in range(JOINTS):
        if _inside_image(uv[index], width, height):
            inside += 1
    return inside / float(JOINTS) >= float(min_visible_fraction)


def _pred_arrays(pred, intrinsic):
    if pred is None:
        return None, None
    joints = _coerce_joints(pred.get("joints_cam"))
    keypoints = pred.get("keypoints_2d")
    uv = None
    if keypoints is not None:
        array = np.asarray(keypoints, dtype=np.float64)
        if array.shape == (JOINTS, 2):
            uv = array
    if uv is None and joints is not None:
        uv = project_pinhole(joints, intrinsic)
    if joints is None or uv is None or not np.isfinite(joints[0]).all() or not np.isfinite(uv[0]).all():
        return None, None
    return joints, uv


def _best_assignment(cost):
    from itertools import permutations
    gt_count, pred_count = cost.shape
    if gt_count == 0 or pred_count == 0:
        return []
    best = None
    if gt_count <= pred_count:
        for choice in permutations(range(pred_count), gt_count):
            total = float(sum(cost[index, choice[index]] for index in range(gt_count)))
            pairs = [(index, choice[index]) for index in range(gt_count)]
            if best is None or total < best[0]:
                best = (total, pairs)
    else:
        for choice in permutations(range(gt_count), pred_count):
            total = float(sum(cost[choice[index], index] for index in range(pred_count)))
            pairs = [(choice[index], index) for index in range(pred_count)]
            if best is None or total < best[0]:
                best = (total, pairs)
    return best[1]


def _mean_joint_distance(left, right, excluded):
    errors = []
    for index in range(JOINTS):
        if index in excluded:
            continue
        if not np.isfinite(left[index]).all() or not np.isfinite(right[index]).all():
            continue
        errors.append(float(np.linalg.norm(left[index] - right[index])))
    if not errors:
        return float("nan")
    return float(np.mean(errors))


def _root_relative_error(gt, pred, excluded):
    if not np.isfinite(gt[0]).all() or not np.isfinite(pred[0]).all():
        return float("nan")
    errors = []
    for index in range(1, JOINTS):
        if index in excluded:
            continue
        if not np.isfinite(gt[index]).all() or not np.isfinite(pred[index]).all():
            continue
        delta = (gt[index] - gt[0]) - (pred[index] - pred[0])
        errors.append(float(np.linalg.norm(delta)))
    if not errors:
        return float("nan")
    return float(np.mean(errors))


def _finite_values(values):
    return [float(value) for value in values if np.isfinite(value)]


def _median(values):
    finite = _finite_values(values)
    if not finite:
        return float("nan")
    return float(np.median(finite))


def _mean(values):
    finite = _finite_values(values)
    if not finite:
        return float("nan")
    return float(np.mean(finite))


def evaluate_hand_frames(samples, exclude_joints=(), match_px=250.0, min_visible_fraction=0.75):
    """把预测和相机系真值按手腕 2D 匹配，汇总检出、左右交换和误差。

    ``samples`` 里每一帧要有 ``K``、``width``、``height``、``gt``、``pred``。
    GT 可见：置信度未知或 ≥0.5，手腕在画面内，且至少 75% 的关节在画面内。
    匹配距离默认 250 px。``exclude_joints`` 从 2D 和相对根关节 3D 的平均里拿掉
    （EgoDex 的 Hand / ThumbKnuckle 用 ``(0, 1)``）。手腕位置误差仍用第 0 点。
    尺度是每个 ``episode_id``、每只真值手一个最小二乘系数，作用在相机系手腕上。
    """
    excluded = tuple(int(index) for index in exclude_joints)
    gt_visible = 0
    matched = 0
    swaps = 0
    px_errors = []
    relative_errors = []
    wrist_errors = []
    groups = {}
    matched_wrists = []
    for sample in samples:
        intrinsic = np.asarray(sample["K"], dtype=np.float64)
        width = int(sample["width"])
        height = int(sample["height"])
        episode_id = sample.get("episode_id", "")
        confidence_map = sample.get("gt_confidence") or {}
        gt_items = []
        pred_items = []
        for side in ("left", "right"):
            gt = _coerce_joints(sample["gt"].get(side))
            if gt is not None:
                uv = project_pinhole(gt, intrinsic)
                if _gt_hand_visible(gt, uv, width, height, confidence_map.get(side), min_visible_fraction):
                    gt_items.append({"side": side, "joints": gt, "uv": uv})
            joints, uv = _pred_arrays(sample["pred"].get(side), intrinsic)
            if joints is not None:
                pred_items.append({"side": side, "joints": joints, "uv": uv})
        gt_visible += len(gt_items)
        if not gt_items or not pred_items:
            continue
        cost = np.zeros((len(gt_items), len(pred_items)), dtype=np.float64)
        for gt_index, gt_item in enumerate(gt_items):
            for pred_index, pred_item in enumerate(pred_items):
                cost[gt_index, pred_index] = float(np.linalg.norm(gt_item["uv"][0] - pred_item["uv"][0]))
        for gt_index, pred_index in _best_assignment(cost):
            if cost[gt_index, pred_index] > float(match_px):
                continue
            gt_item = gt_items[gt_index]
            pred_item = pred_items[pred_index]
            matched += 1
            if gt_item["side"] != pred_item["side"]:
                swaps += 1
            px_errors.append(_mean_joint_distance(gt_item["uv"], pred_item["uv"], excluded))
            relative_errors.append(_root_relative_error(gt_item["joints"], pred_item["joints"], excluded))
            wrist_errors.append(float(np.linalg.norm(gt_item["joints"][0] - pred_item["joints"][0])))
            key = (episode_id, gt_item["side"])
            groups.setdefault(key, []).append((pred_item["joints"][0], gt_item["joints"][0]))
            matched_wrists.append((key, pred_item["joints"][0], gt_item["joints"][0]))
    scales = {}
    for key, pairs in groups.items():
        pred = np.stack([item[0] for item in pairs])
        truth = np.stack([item[1] for item in pairs])
        denom = float(np.sum(pred * pred))
        numer = float(np.sum(pred * truth))
        scales[key] = 1.0 if denom < 1e-12 else numer / denom
    scaled_errors = []
    for key, pred_wrist, gt_wrist in matched_wrists:
        scaled_errors.append(float(np.linalg.norm(scales[key] * pred_wrist - gt_wrist)))
    return {
        "gt_visible": gt_visible,
        "matched": matched,
        "detection_rate": (float(matched) / float(gt_visible)) if gt_visible else float("nan"),
        "swap_rate": (float(swaps) / float(matched)) if matched else float("nan"),
        "px_error_median": _median(px_errors),
        "px_error_mean": _mean(px_errors),
        "root_relative_m_median": _median(relative_errors),
        "root_relative_m_mean": _mean(relative_errors),
        "wrist_error_m_median": _median(wrist_errors),
        "wrist_error_m_mean": _mean(wrist_errors),
        "wrist_error_scaled_m_median": _median(scaled_errors),
        "wrist_error_scaled_m_mean": _mean(scaled_errors),
        "scale_median": _median(list(scales.values())),
        "exclude_joints": excluded,
    }


def _format_number(value, pattern):
    if value is None or not np.isfinite(value):
        return "n/a"
    return pattern % float(value)


def _format_percent(value):
    if value is None or not np.isfinite(value):
        return "n/a"
    return "%.1f%%" % (100.0 * float(value))


def format_hand_eval_report(report):
    """把 ``evaluate_hand_frames`` 的结果写成可直接打印的中文摘要。"""
    excluded = tuple(report.get("exclude_joints") or ())
    if excluded:
        header = "排除关节 %s" % ", ".join(str(index) for index in excluded)
    else:
        header = "全部关节"
    return "\n".join([
        header,
        "检出率 %s（GT 可见 %d，匹配 %d）" % (
            _format_percent(report["detection_rate"]),
            int(report["gt_visible"]),
            int(report["matched"]),
        ),
        "左右交换率 %s" % _format_percent(report["swap_rate"]),
        "2D 误差 中位 %s px，均值 %s px" % (
            _format_number(report["px_error_median"], "%.1f"),
            _format_number(report["px_error_mean"], "%.1f"),
        ),
        "相对根关节 3D 误差 中位 %s cm" % _format_number(
            None if not np.isfinite(report["root_relative_m_median"]) else report["root_relative_m_median"] * 100.0,
            "%.2f",
        ),
        "手腕误差 无尺度对齐 中位 %s cm；每段每手尺度对齐 中位 %s cm（尺度中位 %s）" % (
            _format_number(
                None if not np.isfinite(report["wrist_error_m_median"]) else report["wrist_error_m_median"] * 100.0,
                "%.2f",
            ),
            _format_number(
                None if not np.isfinite(report["wrist_error_scaled_m_median"]) else report["wrist_error_scaled_m_median"] * 100.0,
                "%.2f",
            ),
            _format_number(report["scale_median"], "%.2f"),
        ),
    ])

