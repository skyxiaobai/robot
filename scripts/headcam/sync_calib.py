# -*- coding: utf-8 -*-
"""读写 ``calib.json``，并记录左右目 / IMU 的同步误差。

相机约定与 ``headcam.hand_pose.load_calibration`` 相同：OpenCV
``stereoCalibrate`` 的 ``p_right = R @ p_left + T``。Kalibr 的
``T_cn_cnm1`` 把左目（cam0）变到右目（cam1）。简单 YAML 用
``T_right_left``。

``scripts/retarget/calibrate.py`` 的 ``AxisCalibration`` 是人手腕到机械臂的
轴缩放（``robot = scale * human + offset``），不是相机外参。本模块拒绝把
那种 JSON 当成标定。

``sync_error_ms`` 只在有测量来源时填写。没有 ``stereo/timestamps_lr.csv``
时左右同步是 null，不写成 0。IMU–相机时间偏移只记录调用者或 metadata 里
已经给出的 ``imu_cam_time_offset_s``，这里不估计。
"""
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import numpy as np

from headcam.hand_pose import load_calibration, write_calibration_yaml

SCHEMA = "headcam_calib_v1"


def _metadata(session):
    path = Path(session) / "metadata.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def measure_left_right_sync(session):
    """``stereo/timestamps_lr.csv`` 的 |left_s - right_s|，单位毫秒。

    与 ``validate_session`` 的 ``sync_ms`` 同一算法。文件不存在时返回 None。
    """
    path = Path(session) / "stereo" / "timestamps_lr.csv"
    if not path.is_file():
        return None
    left, right = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part for part in line.replace(",", " ").split() if part]
        if len(parts) < 3:
            continue
        try:
            left.append(float(parts[-2]))
            right.append(float(parts[-1]))
        except ValueError:
            continue
    if not left:
        return {
            "left_right_max": None,
            "left_right_median": None,
            "n": 0,
            "source": "stereo/timestamps_lr.csv",
        }
    delta_ms = np.abs(np.asarray(left) - np.asarray(right)) * 1e3
    return {
        "left_right_max": float(np.max(delta_ms)),
        "left_right_median": float(np.median(delta_ms)),
        "n": int(delta_ms.size),
        "source": "stereo/timestamps_lr.csv",
    }


def sync_error_ms(session, imu_cam_time_offset_s=None):
    """同步误差对象。左右同步没测到就是 null，不是 0。"""
    measured = measure_left_right_sync(session)
    if measured is None:
        payload = {
            "left_right_max": None,
            "left_right_median": None,
            "n": 0,
            "source": None,
        }
    else:
        payload = dict(measured)
    if imu_cam_time_offset_s is None:
        meta = _metadata(session)
        if meta.get("imu_cam_time_offset_s") is not None:
            imu_cam_time_offset_s = float(meta["imu_cam_time_offset_s"])
    if imu_cam_time_offset_s is None:
        payload["imu_cam"] = None
    else:
        payload["imu_cam"] = abs(float(imu_cam_time_offset_s)) * 1e3
    return payload


def _intrinsics(matrix):
    array = np.asarray(matrix, dtype=float)
    return [float(array[0, 0]), float(array[1, 1]), float(array[0, 2]), float(array[1, 2])]


def _distortion(values):
    if values is None:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    flat = np.asarray(values, dtype=float).reshape(-1)
    return [float(value) for value in flat]


def _as_pose(matrix):
    if matrix is None:
        return None
    array = np.asarray(matrix, dtype=float)
    if array.shape != (4, 4):
        raise ValueError("T_cam_imu 必须是 4x4")
    return [[float(value) for value in row] for row in array.tolist()]


def assert_camera_calib(payload):
    """拒绝 retarget 的轴缩放 JSON。"""
    if not isinstance(payload, dict):
        raise ValueError("calib.json 必须是对象")
    keys = set(payload)
    if {"scale", "offset"} <= keys and "T_right_left" not in payload and "left" not in payload:
        raise ValueError(
            "这是 retarget.calibrate.AxisCalibration（robot = scale * human + offset），"
            "不是相机标定。相机外参沿用 OpenCV：p_right = R @ p_left + T。"
        )


def calibration_document(calib, sync, T_cam_imu=None, imu_cam_time_offset_s=None):
    """``load_calibration`` 的字典 → ``calib.json`` 对象。``calib`` 可以是 None。"""
    document = {
        "schema": SCHEMA,
        "convention": "opencv_stereo",
        "convention_note": (
            "p_right = R @ p_left + T，与 headcam.hand_pose.load_calibration 相同。"
            "不是 scripts/retarget/calibrate.py 的 AxisCalibration。"
        ),
        "sync_error_ms": sync,
        "T_cam_imu": _as_pose(T_cam_imu),
        "imu_cam_time_offset_s": None if imu_cam_time_offset_s is None else float(imu_cam_time_offset_s),
    }
    if calib is None:
        document["calibration"] = None
        return document
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = np.asarray(calib["R"], dtype=float)
    pose[:3, 3] = np.asarray(calib["T"], dtype=float).reshape(3)
    document.update({
        "calibration": "stereo",
        "image_width": int(calib["image_width"]),
        "image_height": int(calib["image_height"]),
        "left": {
            "intrinsics": _intrinsics(calib["K_left"]),
            "dist_coeffs": _distortion(calib.get("dist_left")),
        },
        "right": {
            "intrinsics": _intrinsics(calib["K_right"]),
            "dist_coeffs": _distortion(calib.get("dist_right")),
        },
        "T_right_left": _as_pose(pose),
        "baseline_m": float(np.linalg.norm(np.asarray(calib["T"], dtype=float).reshape(-1)[:3])),
    })
    return document


def _matrix_from_intrinsics(values):
    fx, fy, cx, cy = [float(value) for value in values]
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)


def calibration_from_document(payload):
    """``calib.json`` → ``load_calibration`` 同结构的字典。没有相机块时返回 None。"""
    assert_camera_calib(payload)
    if payload.get("calibration") is None or "T_right_left" not in payload:
        return None
    pose = np.asarray(payload["T_right_left"], dtype=float)
    if pose.shape != (4, 4):
        raise ValueError("T_right_left 必须是 4x4")
    return {
        "image_width": int(payload["image_width"]),
        "image_height": int(payload["image_height"]),
        "K_left": _matrix_from_intrinsics(payload["left"]["intrinsics"]),
        "K_right": _matrix_from_intrinsics(payload["right"]["intrinsics"]),
        "dist_left": np.asarray(payload["left"].get("dist_coeffs") or [0, 0, 0, 0, 0], dtype=float),
        "dist_right": np.asarray(payload["right"].get("dist_coeffs") or [0, 0, 0, 0, 0], dtype=float),
        "R": pose[:3, :3].copy(),
        "T": pose[:3, 3].copy(),
    }


def find_calib_yaml(session):
    root = Path(session)
    for relative in ("calib.yaml", "calib.yml", "stereo/calib.yaml"):
        path = root / relative
        if path.is_file():
            return path
    return None


def load_session_calibration(session):
    """优先 ``calib.yaml``（现有管线的入口），否则 ``calib.json``。都没有则 (None, None)。"""
    yaml_path = find_calib_yaml(session)
    if yaml_path is not None:
        return load_calibration(yaml_path), yaml_path
    json_path = Path(session) / "calib.json"
    if json_path.is_file():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        return calibration_from_document(payload), json_path
    return None, None


def write_calib_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def read_calib_json(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assert_camera_calib(payload)
    return payload


def write_session_calibration(session, calib, sync=None, T_cam_imu=None, imu_cam_time_offset_s=None, json_name="calib.json"):
    """同时写 ``calib.yaml`` 和 ``calib.json``，方便现有 ``load_calibration`` 继续读 YAML。"""
    root = Path(session)
    write_calibration_yaml(root / "calib.yaml", calib)
    if sync is None:
        sync = sync_error_ms(root, imu_cam_time_offset_s=imu_cam_time_offset_s)
    document = calibration_document(
        calib, sync, T_cam_imu=T_cam_imu, imu_cam_time_offset_s=imu_cam_time_offset_s,
    )
    write_calib_json(root / json_name, document)
    return document


def refresh_calib_json(session, out_path=None, imu_cam_time_offset_s=None, T_cam_imu=None):
    """从会话里已有的标定和 ``timestamps_lr.csv`` 生成 ``calib.json``。不估算没写下来的量。"""
    root = Path(session)
    calib, _source = load_session_calibration(root)
    meta = _metadata(root)
    if imu_cam_time_offset_s is None and meta.get("imu_cam_time_offset_s") is not None:
        imu_cam_time_offset_s = float(meta["imu_cam_time_offset_s"])
    if T_cam_imu is None and meta.get("T_cam_imu") is not None:
        T_cam_imu = meta["T_cam_imu"]
    document = calibration_document(
        calib,
        sync_error_ms(root, imu_cam_time_offset_s=imu_cam_time_offset_s),
        T_cam_imu=T_cam_imu,
        imu_cam_time_offset_s=imu_cam_time_offset_s,
    )
    destination = Path(out_path) if out_path else root / "calib.json"
    write_calib_json(destination, document)
    return document


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="从会话写 calib.json，并记录 sync_error_ms")
    parser.add_argument("--session", required=True)
    parser.add_argument("--out", default=None, help="默认写到会话目录的 calib.json")
    parser.add_argument("--imu-cam-time-offset-s", type=float, default=None,
                        help="已经测得的 IMU–相机时间偏移（秒）。不传就不填，不在这里估计")
    args = parser.parse_args(argv)
    document = refresh_calib_json(
        args.session, out_path=args.out, imu_cam_time_offset_s=args.imu_cam_time_offset_s,
    )
    sync = document["sync_error_ms"]
    print(
        "sync_error_ms left_right_max=%s left_right_median=%s imu_cam=%s calibration=%s"
        % (sync.get("left_right_max"), sync.get("left_right_median"), sync.get("imu_cam"), document.get("calibration"))
    )
    return 0


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
