# -*- coding: utf-8 -*-
"""双目外参约定，以及给 ORB-SLAM3 用的校正。

OpenCV 的 ``stereoRectify`` 要 ``p_right = R @ p_left + t``。
ORB-SLAM3 的 ``Stereo.T_c1_c2`` 存的是反过来的那一截：cam2 在 cam1 里的位姿，
也就是 ``p_left = R @ p_right + t``，平移大约是右相机光心在左相机系的位置。
它在 ``Settings::precomputeRectificationMaps`` 里会再取逆，然后才交给 ``stereoRectify``。

把 OpenCV 的 ``(R, t)`` 原样写进 ``T_c1_c2``，校正用的就是反的外参。
基线沿 X、只有大约 2 度相对旋转时，反了之后同一行的竖向误差会到几十像素，
ORB-SLAM3 的双目匹配只在大约 2 像素的行带里找，跟不上。
"""
from pathlib import Path

import numpy as np


def T_right_from_left(rotation, translation):
    """OpenCV：``p_right = R @ p_left + t``。返回 4×4。"""
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = np.asarray(rotation, dtype=float)
    pose[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return pose


def T_c1_c2_for_orbslam3(rotation, translation):
    """OpenCV ``(R, t)`` → ORB-SLAM3 ``Stereo.T_c1_c2``。

    返回的平移是右相机光心在左相机坐标系里的位置。
    右相机在左边相机的 +X 时，这一项的 x 是正的基线。
    """
    return np.linalg.inv(T_right_from_left(rotation, translation))


def rectify_pinhole_pair(k_left, k_right, rotation, translation, image_size, alpha=0.0):
    """已经是针孔、畸变当作 0。``rotation, translation`` 是 OpenCV 约定。

    返回校正旋转、投影矩阵、基线（米，取绝对值）和从针孔图到校正图的 remap。
    校正绕光心转，左右光心的世界坐标不变。
    """
    import cv2

    size = (int(image_size[0]), int(image_size[1]))
    left = np.asarray(k_left, dtype=float)
    right = np.asarray(k_right, dtype=float)
    distortion = np.zeros(4, dtype=float)
    r1, r2, p1, p2, q, roi1, roi2 = cv2.stereoRectify(
        left, distortion, right, distortion, size,
        np.asarray(rotation, dtype=float),
        np.asarray(translation, dtype=float).reshape(3, 1),
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=float(alpha),
    )
    map1x, map1y = cv2.initUndistortRectifyMap(left, distortion, r1, p1, size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(right, distortion, r2, p2, size, cv2.CV_32FC1)
    focal = float(p2[0, 0])
    baseline_m = abs(float(p2[0, 3]) / focal) if focal else None
    return {
        "R1": r1,
        "R2": r2,
        "P1": p1,
        "P2": p2,
        "Q": q,
        "roi1": roi1,
        "roi2": roi2,
        "baseline_m": baseline_m,
        "map_left": (map1x, map1y),
        "map_right": (map2x, map2y),
        "image_size": size,
    }


def vertical_epipolar_pixels(k_left, k_right, rotation, translation, image_size, depths=(0.4, 1.0, 2.0)):
    """真几何下，右图 v 减左图 v 的绝对值。校正正确时接近 0。"""
    width, height = int(image_size[0]), int(image_size[1])
    left = np.asarray(k_left, dtype=float)
    right = np.asarray(k_right, dtype=float)
    rot = np.asarray(rotation, dtype=float)
    trans = np.asarray(translation, dtype=float).reshape(3)
    errors = []
    for depth in depths:
        for u in np.linspace(width * 0.1, width * 0.9, 8):
            for v in np.linspace(height * 0.1, height * 0.9, 8):
                point_left = np.array([
                    (u - left[0, 2]) / left[0, 0] * depth,
                    (v - left[1, 2]) / left[1, 1] * depth,
                    depth,
                ])
                point_right = rot @ point_left + trans
                if point_right[2] <= 0.05:
                    continue
                u_right = right[0, 0] * point_right[0] / point_right[2] + right[0, 2]
                v_right = right[1, 1] * point_right[1] / point_right[2] + right[1, 2]
                if 0 <= u_right < width and 0 <= v_right < height:
                    errors.append(abs(float(v_right - v)))
    if not errors:
        return {"n": 0}
    values = np.asarray(errors, dtype=float)
    return {
        "n": int(values.size),
        "median_px": float(np.median(values)),
        "p90_px": float(np.percentile(values, 90)),
        "max_px": float(np.max(values)),
    }


def write_rectified_orbslam3_yaml(path, k_rect, image_size, fps, baseline_m, n_features=2000):
    """图像已经校成行对齐时用。``Camera.type`` 是 Rectified，只写 ``Stereo.b``。

    不再写 ``T_c1_c2``，避免 ORB-SLAM3 再校正一次。
    """
    k_rect = np.asarray(k_rect, dtype=float)
    text = "\n".join([
        "%YAML:1.0",
        "",
        "File.version: \"1.0\"",
        "Camera.type: \"Rectified\"",
        "",
        "# 图已经按 OpenCV stereoRectify 校成行对齐。畸变是 0，不是把鱼眼系数填 0。",
        "Camera1.fx: %.10g" % k_rect[0, 0],
        "Camera1.fy: %.10g" % k_rect[1, 1],
        "Camera1.cx: %.10g" % k_rect[0, 2],
        "Camera1.cy: %.10g" % k_rect[1, 2],
        "",
        "Camera.width: %d" % int(image_size[0]),
        "Camera.height: %d" % int(image_size[1]),
        "Camera.fps: %d" % int(fps),
        "Camera.RGB: 0",
        "",
        "Stereo.ThDepth: 40.0",
        "Stereo.b: %.10g" % float(baseline_m),
        "",
        "ORBextractor.nFeatures: %d" % int(n_features),
        "ORBextractor.scaleFactor: 1.2",
        "ORBextractor.nLevels: 8",
        "ORBextractor.iniThFAST: 12",
        "ORBextractor.minThFAST: 5",
        "",
        "Viewer.KeyFrameSize: 0.05",
        "Viewer.KeyFrameLineWidth: 1.0",
        "Viewer.GraphLineWidth: 0.9",
        "Viewer.PointSize: 2.0",
        "Viewer.CameraSize: 0.08",
        "Viewer.CameraLineWidth: 3.0",
        "Viewer.ViewpointX: 0.0",
        "Viewer.ViewpointY: -0.7",
        "Viewer.ViewpointZ: -1.8",
        "Viewer.ViewpointF: 500.0",
        "Viewer.imageViewScale: 1.0",
        "",
    ])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
