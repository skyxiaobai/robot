# -*- coding: utf-8 -*-
"""把一节 HOT3D-Clips（Quest3）收成 ``record_session`` 的会话。

图像按该帧自己的 FISHEYE624 模型去畸变成针孔。Quest3 的 SLAM 相机是侧放的，
去畸变时图像和外参一起转 -90°，与 ``eval_hot3d_stereo.pinhole_of`` 相同。
转完之后的图和 ``T_world_cam`` 是同一套虚拟相机。

不写 ``slam.tum``。头戴定位写到 ``gt/headset_tum.txt``，文件头标明
``pose_source=hot3d_gt_mocap``。那是地面真值，不是 ORB-SLAM3 或其他 SLAM。

IMU、曝光、任务句子没有来源时不编造。原始鱼眼参数原样记在 metadata 里，
不改写成径向切向畸变。
"""
import json
import sys
import tempfile
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import numpy as np

from headcam.euroc_session import _write_video, ns_to_second_text
from headcam.hand_pose import write_calibration_yaml
from headcam.hot3d_adapter import STREAM_LEFT, STREAM_RIGHT, relative_extrinsics, rotmat_to_quat_xyzw
from headcam.record_session import init_session
from headcam.stereo_geometry import T_c1_c2_for_orbslam3
from headcam.sync_calib import refresh_calib_json

POSE_SOURCE_GT = "hot3d_gt_mocap"


def integer_fps_ns(nanoseconds):
    """纳秒时间戳的中位间隔 → 整数 fps。偏差超过 2% 就不四舍五入。"""
    if len(nanoseconds) < 2:
        return None
    steps = [int(b) - int(a) for a, b in zip(nanoseconds, nanoseconds[1:]) if int(b) > int(a)]
    if not steps:
        return None
    steps.sort()
    median = steps[len(steps) // 2]
    if median <= 0:
        return None
    hz = 1e9 / float(median)
    rounded = int(round(hz))
    if rounded < 1 or abs(hz - rounded) / float(rounded) > 0.02:
        return None
    return rounded


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _timestamp_ns(info, stream):
    raw = info.get("image_timestamps_ns")
    if not isinstance(raw, dict) or stream not in raw:
        raise ValueError("info 缺少 image_timestamps_ns.%s" % stream)
    value = raw[stream]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("时间戳不是整数纳秒：%s" % value)
    if isinstance(value, float) and not value.is_integer():
        raise ValueError("时间戳不是整数纳秒：%s" % value)
    return int(value)


def _calibration_note(camera_json):
    """原始相机模型，原样摘录。不把 FISHEYE624 换成别的畸变模型。"""
    calibration = camera_json.get("calibration") if isinstance(camera_json, dict) else None
    if not isinstance(calibration, dict):
        return {"present": False}
    params = calibration.get("projection_params")
    return {
        "present": True,
        "projection_model_type": calibration.get("projection_model_type"),
        "image_width": calibration.get("image_width"),
        "image_height": calibration.get("image_height"),
        "projection_params": params,
        "label": calibration.get("label"),
        "stream_id": calibration.get("stream_id"),
    }


def write_headset_tum(path, timestamps_text, poses):
    """秒文本和虚拟左目的 4×4。时间戳用和 ``timestamps.csv`` 相同的十进制，不经第二次 float 格式化。"""
    lines = [
        "# pose_source=%s" % POSE_SOURCE_GT,
        "# HOT3D headset ground truth for the undistorted left pinhole. Not ORB-SLAM3 or any other SLAM output.",
        "# timestamp tx ty tz qx qy qz qw",
    ]
    for stamp, pose in zip(timestamps_text, poses):
        pose = np.asarray(pose, dtype=float)
        quat = rotmat_to_quat_xyzw(pose[:3, :3])
        values = list(pose[:3, 3]) + list(quat)
        lines.append("%s %s" % (stamp, " ".join("%.9f" % float(value) for value in values)))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_orbslam3_yaml(path, calib, fps):
    """给这份去畸变针孔写 ORB-SLAM3 stereo yaml。畸变四项是 0，因为图已经是针孔。

    ``calib["R"]``、``calib["T"]`` 仍是 OpenCV 约定 ``p_right = R @ p_left + T``。
    写进 ``Stereo.T_c1_c2`` 的是它的逆。ORB-SLAM3 会再取逆才做 ``stereoRectify``。
    图本身还没校正；校正发生在 ORB-SLAM3 里面。已经校好的图用
    ``stereo_geometry.write_rectified_orbslam3_yaml``，不要再走这条。
    """
    left = np.asarray(calib["K_left"], dtype=float)
    right = np.asarray(calib["K_right"], dtype=float)
    pose = T_c1_c2_for_orbslam3(calib["R"], calib["T"])
    flat = ", ".join("%.10g" % float(value) for value in pose.reshape(-1))
    text = "\n".join([
        "%YAML:1.0",
        "",
        "File.version: \"1.0\"",
        "Camera.type: \"PinHole\"",
        "",
        "# 虚拟针孔。k1 k2 p1 p2 是 0，因为图像已按原 FISHEYE624 去畸变，不是把鱼眼系数填 0。",
        "Camera1.fx: %.10g" % left[0, 0],
        "Camera1.fy: %.10g" % left[1, 1],
        "Camera1.cx: %.10g" % left[0, 2],
        "Camera1.cy: %.10g" % left[1, 2],
        "Camera1.k1: 0.0",
        "Camera1.k2: 0.0",
        "Camera1.p1: 0.0",
        "Camera1.p2: 0.0",
        "",
        "Camera2.fx: %.10g" % right[0, 0],
        "Camera2.fy: %.10g" % right[1, 1],
        "Camera2.cx: %.10g" % right[0, 2],
        "Camera2.cy: %.10g" % right[1, 2],
        "Camera2.k1: 0.0",
        "Camera2.k2: 0.0",
        "Camera2.p1: 0.0",
        "Camera2.p2: 0.0",
        "",
        "Camera.width: %d" % int(calib["image_width"]),
        "Camera.height: %d" % int(calib["image_height"]),
        "Camera.fps: %d" % int(fps),
        "Camera.RGB: 0",
        "",
        "Stereo.ThDepth: 40.0",
        "# T_c1_c2 是 cam2 在 cam1 里的位姿（右光心在左相机系，基线沿 +X 时 x 为正）。",
        "# 不是 OpenCV 的 p_right = R p_left + t。ORB-SLAM3 会再取逆后做 stereoRectify。",
        "Stereo.T_c1_c2: !!opencv-matrix",
        "  rows: 4",
        "  cols: 4",
        "  dt: f",
        "  data: [%s]" % flat,
        "",
        "ORBextractor.nFeatures: 1500",
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


def _save_png(path, image):
    import cv2
    frame = np.asarray(image)
    if frame.ndim == 2:
        frame = np.stack([frame, frame, frame], axis=-1)
    frame = np.ascontiguousarray(frame[..., :3])
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError("写 png 失败：%s" % path)
    return path


def hot3d_clip_to_session(clip, out, episode_id=None, max_frames=None, euroc_out=None, orb_yaml=None):
    """Quest3 clip 目录 → §7.7 会话。返回摘要。不写 ``slam.tum``。"""
    import imageio.v2 as imageio
    from eval_hot3d_stereo import K_of, load_frame_cameras, pinhole_of, undistort
    from headcam.hot3d_adapter import clip_keys

    clip = Path(clip)
    keys = clip_keys(str(clip))
    if max_frames:
        keys = keys[: int(max_frames)]
    if not keys:
        raise ValueError("HOT3D clip 里没有帧：%s" % clip)
    first_info = _load_json(clip / ("%s.info.json" % keys[0]))
    device = first_info.get("device")
    if device != "Quest3":
        raise ValueError("只转换 Quest3。这个 clip 的 device=%s，不猜测旋转。" % device)

    stamps_ns = []
    stamps_ns_right = []
    poses = []
    relatives = []
    pin0 = None
    source_models = None
    euroc_root = Path(euroc_out) if euroc_out else None
    left_pngs = []
    right_pngs = []
    temp_holder = None
    if euroc_root is not None:
        (euroc_root / "mav0" / "cam0" / "data").mkdir(parents=True, exist_ok=True)
        (euroc_root / "mav0" / "cam1" / "data").mkdir(parents=True, exist_ok=True)
        png_root = euroc_root
    else:
        temp_holder = tempfile.TemporaryDirectory(prefix="hot3d_png_")
        png_root = Path(temp_holder.name)

    try:
        for key in keys:
            cams = load_frame_cameras(str(clip), key)
            pins = {stream: pinhole_of(cams[stream], 1.0, -90) for stream in (STREAM_LEFT, STREAM_RIGHT)}
            if pin0 is None:
                pin0 = pins
                cameras = _load_json(clip / ("%s.cameras.json" % key))
                source_models = {
                    "left": _calibration_note(cameras.get(STREAM_LEFT)),
                    "right": _calibration_note(cameras.get(STREAM_RIGHT)),
                }
            info = _load_json(clip / ("%s.info.json" % key))
            left_ns = _timestamp_ns(info, STREAM_LEFT)
            right_ns = _timestamp_ns(info, STREAM_RIGHT)
            stamps_ns.append(left_ns)
            stamps_ns_right.append(right_ns)
            poses.append(np.asarray(pins[STREAM_LEFT].T_world_from_eye, dtype=float))
            relatives.append(relative_extrinsics(
                pins[STREAM_LEFT].T_world_from_eye, pins[STREAM_RIGHT].T_world_from_eye,
            ))
            for stream, bucket, folder in (
                (STREAM_LEFT, left_pngs, "cam0"),
                (STREAM_RIGHT, right_pngs, "cam1"),
            ):
                image = imageio.imread(clip / ("%s.image_%s.jpg" % (key, stream)))
                if getattr(image, "ndim", 0) == 2:
                    image = np.stack([image, image, image], axis=-1)
                undistorted = undistort(image, cams[stream], pins[stream])
                if euroc_root is None:
                    written = _save_png(png_root / stream / ("%06d.png" % len(bucket)), undistorted)
                else:
                    stamp = left_ns if stream == STREAM_LEFT else right_ns
                    written = _save_png(png_root / "mav0" / folder / "data" / ("%d.png" % stamp), undistorted)
                bucket.append(written)

        fps = integer_fps_ns(stamps_ns)
        if fps is None:
            raise ValueError("图像时间间隔不是整数 fps，没有四舍五入")
        root = Path(out)
        name = clip.name
        init_session(
            root,
            fps=float(fps),
            episode_id=episode_id or "hot3d/quest3/%s" % name,
            image_width=int(pin0[STREAM_LEFT].width),
            image_height=int(pin0[STREAM_LEFT].height),
        )
        _write_video(left_pngs, root / "stereo" / "left.mp4", fps)
        _write_video(right_pngs, root / "stereo" / "right.mp4", fps)
    finally:
        if temp_holder is not None:
            temp_holder.cleanup()

    left_pin, right_pin = pin0[STREAM_LEFT], pin0[STREAM_RIGHT]
    rotation, translation = relatives[0]
    drift = max(float(np.linalg.norm(item[1] - translation)) for item in relatives)
    kl, kr = K_of(left_pin), K_of(right_pin)
    calib = {
        "image_width": int(left_pin.width),
        "image_height": int(left_pin.height),
        "K_left": kl,
        "K_right": kr,
        "dist_left": np.zeros(5),
        "dist_right": np.zeros(5),
        "R": rotation,
        "T": translation,
    }
    root = Path(out)
    name = clip.name
    stamp_text = [ns_to_second_text(value) for value in stamps_ns]
    right_text = [ns_to_second_text(value) for value in stamps_ns_right]
    (root / "timestamps.csv").write_text(
        "frame_index,timestamp_s\n" + "".join("%d,%s\n" % (index, text) for index, text in enumerate(stamp_text)),
        encoding="utf-8",
    )
    (root / "stereo" / "timestamps_lr.csv").write_text(
        "frame_index,left_s,right_s\n" + "".join(
            "%d,%s,%s\n" % (index, left, right) for index, (left, right) in enumerate(zip(stamp_text, right_text))
        ),
        encoding="utf-8",
    )
    write_calibration_yaml(root / "calib.yaml", calib)
    calib_path = root / "calib.yaml"
    calib_path.write_text(
        "# 虚拟针孔。dist_coeffs 是 0，因为图像已按原 FISHEYE624 去畸变，不是把鱼眼系数改成 0。\n"
        + calib_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    write_headset_tum(root / "gt" / "headset_tum.txt", stamp_text, poses)
    (root / "gt" / "pose_source.json").write_text(json.dumps({
        "pose_source": POSE_SOURCE_GT,
        "file": "gt/headset_tum.txt",
        "time_unit": "seconds",
        "quaternion": "qx qy qz qw",
        "meaning": (
            "去畸变并转正之后的左目虚拟针孔在世界系的位姿。"
            "来自 HOT3D 头戴定位，不是 ORB-SLAM3 或其他 SLAM。"
        ),
        "rotation_deg": -90,
        "rotation_note": "Quest3 SLAM 图侧放。图像和 T_world_from_eye 一起转 -90 度，与 eval_hot3d_stereo.pinhole_of 相同。",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if (root / "slam.tum").exists():
        raise RuntimeError("转换不应写出 slam.tum")
    meta_path = root / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["task"] = ""
    meta["instruction"] = ""
    meta["environment"] = "unknown"
    meta["timestamps_source"] = "hot3d_image_timestamps_ns"
    meta["timestamps_are_hardware"] = False
    meta["hardware_capture"] = False
    meta["hot3d"] = {
        "dataset": "HOT3D-Clips",
        "clip": name,
        "device": device,
        "sequence_id": first_info.get("sequence_id"),
        "participant_id": first_info.get("participant_id"),
        "frames": len(keys),
        "frame_keys": keys,
        "fps_source": "image_timestamps_ns 中位间隔换成整数 Hz",
        "pose_source": POSE_SOURCE_GT,
        "pose_file": "gt/headset_tum.txt",
        "slam_tum_written": False,
        "baseline_m": float(np.linalg.norm(translation)),
        "extrinsic_drift_m": drift,
        "extrinsic_note": "calib.yaml 用第一帧的 T_right_left。extrinsic_drift_m 是其余帧平移相对第一帧的最大差。",
        "undistort": "hand_tracking_toolkit warp_image，虚拟针孔，Quest3 旋转 -90 度",
        "source_cameras": source_models,
        "imu_note": "HOT3D-Clips 的这个 tar 没有 IMU 样本。不写测量，也不跑 stereo-inertial。",
        "unknown": ["exposure", "task", "instruction", "imu_samples", "imu_cam_time_offset_s", "accuracy"],
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    refresh_calib_json(root)
    if euroc_root is not None:
        times = euroc_root / "times.txt"
        times.write_text("".join("%d\n" % value for value in stamps_ns), encoding="utf-8")
    if orb_yaml is not None:
        write_orbslam3_yaml(orb_yaml, calib, fps)
    return {
        "session": str(root),
        "frames": len(keys),
        "fps": fps,
        "device": device,
        "baseline_m": float(np.linalg.norm(translation)),
        "extrinsic_drift_m": drift,
        "image_width": int(left_pin.width),
        "image_height": int(left_pin.height),
        "pose_source": POSE_SOURCE_GT,
        "slam_tum_written": False,
    }


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="HOT3D Quest3 clip → 头戴会话。不把真值写成 slam.tum。")
    parser.add_argument("--clip", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--episode-id", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--euroc-out", default=None, help="可选。写成 ORB-SLAM3 stereo_euroc 要的 mav0 目录")
    parser.add_argument("--orb-yaml", default=None, help="可选。写虚拟针孔的 ORB-SLAM3 yaml")
    args = parser.parse_args(argv)
    try:
        summary = hot3d_clip_to_session(
            args.clip, args.out, episode_id=args.episode_id, max_frames=args.max_frames,
            euroc_out=args.euroc_out, orb_yaml=args.orb_yaml,
        )
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
