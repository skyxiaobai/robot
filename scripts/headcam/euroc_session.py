# -*- coding: utf-8 -*-
"""把 EuRoC mav0 目录收成 ``record_session`` 的会话。只做对得上的换算。

图像文件名是纳秒，写成秒时用整数除法保留 9 位小数，不先转成 float64。
左右外参来自两份 ``sensor.yaml`` 的 ``T_BS``：

    p_B = T_BS @ p_C
    T_right_left = inv(T_BS_cam1) @ T_BS_cam0

``T_cam_imu`` 用左目：``inv(T_BS_cam0)``。机体在 EuRoC 里就是 IMU
（``imu0`` 的 ``T_BS`` 是单位阵时才这样写）。

曝光、任务、手部、精度都没有来源，不填 0，也不编造。畸变只有 4 个数时不补 k3。
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import numpy as np

from headcam.record_session import init_session
from headcam.sync_calib import refresh_calib_json
from headcam.hand_pose import write_calibration_yaml


def ns_to_second_text(nanoseconds):
    """纳秒整数 → 秒的十进制文本。不经过 float64。"""
    value = int(nanoseconds)
    sign = "-" if value < 0 else ""
    value = abs(value)
    return "%s%d.%09d" % (sign, value // 1000000000, value % 1000000000)


def tum_nanosecond_stamps(path):
    """轨迹第一列的整数纳秒。``123.000000`` 取点号前，不用 float。"""
    stamps = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        token = line.split()[0]
        if "." in token:
            token = token.split(".", 1)[0]
        stamps.append(int(token))
    if not stamps:
        raise ValueError("轨迹里没有时间戳：%s" % path)
    return stamps


def _load_yaml(path):
    import yaml
    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("不是映射：%s" % path)
    return data


def _T_BS(sensor, path):
    block = sensor.get("T_BS")
    if not isinstance(block, dict) or "data" not in block:
        raise ValueError("%s 没有 T_BS.data" % path)
    matrix = np.asarray(block["data"], dtype=np.float64).reshape(4, 4)
    if not np.isfinite(matrix).all():
        raise ValueError("%s 的 T_BS 不是有限数" % path)
    return matrix


def _intrinsics_matrix(values):
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError("intrinsics 需要 fu, fv, cu, cv 四个数")
    fu, fv, cu, cv = [float(value) for value in values]
    return np.array([[fu, 0.0, cu], [0.0, fv, cv], [0.0, 0.0, 1.0]], dtype=np.float64)


def stereo_from_euroc(cam0, cam1):
    """两份 EuRoC 相机 yaml → ``load_calibration`` 用的字典，外加说明。

    相机模型不是 pinhole，或畸变不是 radial-tangential 时直接拒绝。
    """
    for sensor, path in (cam0, cam1):
        if sensor.get("camera_model") != "pinhole":
            raise ValueError("%s 的 camera_model 不是 pinhole，不猜测投影" % path)
        if sensor.get("distortion_model") != "radial-tangential":
            raise ValueError("%s 的 distortion_model 不是 radial-tangential" % path)
    left, left_path = cam0
    right, right_path = cam1
    resolution = left.get("resolution")
    if not isinstance(resolution, (list, tuple)) or len(resolution) != 2:
        raise ValueError("%s 没有 resolution" % left_path)
    T_left = _T_BS(left, left_path)
    T_right = _T_BS(right, right_path)
    pose = np.linalg.inv(T_right) @ T_left
    dist_left = [float(value) for value in left["distortion_coefficients"]]
    dist_right = [float(value) for value in right["distortion_coefficients"]]
    calib = {
        "image_width": int(resolution[0]),
        "image_height": int(resolution[1]),
        "K_left": _intrinsics_matrix(left["intrinsics"]),
        "K_right": _intrinsics_matrix(right["intrinsics"]),
        "dist_left": np.asarray(dist_left, dtype=np.float64),
        "dist_right": np.asarray(dist_right, dtype=np.float64),
        "R": pose[:3, :3],
        "T": pose[:3, 3],
    }
    notes = {
        "distortion_model": "radial-tangential",
        "distortion_note": "系数按文件原样写入，不把缺少的 k3 补成 0。",
        "T_right_left": "inv(T_BS_cam1) @ T_BS_cam0。EuRoC 的 T_BS 满足 p_B = T_BS @ p_C。",
        "T_cam_imu": "inv(T_BS_cam0)。只在 imu0 的 T_BS 是单位阵时，机体才是 IMU。",
    }
    return calib, np.linalg.inv(T_left), notes


def _mav_root(euroc):
    root = Path(euroc)
    if (root / "mav0").is_dir():
        return root / "mav0"
    if (root / "cam0").is_dir():
        return root
    raise FileNotFoundError("找不到 mav0/cam0：%s" % root)


def _integer_fps(nanoseconds):
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


def _write_video(frames, dest, fps):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    work = dest.parent / (dest.stem + "_frames")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    try:
        for index, source in enumerate(frames):
            link = work / ("%06d.png" % index)
            link.symlink_to(Path(source).resolve())
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-framerate", str(int(fps)),
                "-start_number", "0",
                "-i", str(work / "%06d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
                str(dest),
            ],
            check=True,
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return dest


def _imu_rows(imu_csv, first_ns, last_ns):
    rows = []
    if not Path(imu_csv).is_file():
        return rows, "没有 imu0/data.csv。不编造样本。"
    for line in Path(imu_csv).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 7:
            continue
        try:
            stamp = int(parts[0])
            values = [float(part) for part in parts[1:7]]
        except ValueError:
            continue
        if stamp < first_ns or stamp > last_ns:
            continue
        rows.append((stamp, values))
    return rows, "列按 imu0/data.csv 的顺序原样抄：w_RS_S_x,y,z 然后 a_RS_S_x,y,z。没有做轴旋转。"


def euroc_to_session(euroc, tum_path, out, episode_id=None):
    """用轨迹里的纳秒文件名选左右图像，写成 §7.7 会话。"""
    mav = _mav_root(euroc)
    stamps = tum_nanosecond_stamps(tum_path)
    fps = _integer_fps(stamps)
    if fps is None:
        raise ValueError("图像时间间隔不是整数 fps，没有四舍五入")
    cam0_yaml = mav / "cam0" / "sensor.yaml"
    cam1_yaml = mav / "cam1" / "sensor.yaml"
    calib, T_cam_imu, notes = stereo_from_euroc(
        (_load_yaml(cam0_yaml), cam0_yaml),
        (_load_yaml(cam1_yaml), cam1_yaml),
    )
    left_frames = []
    right_frames = []
    for stamp in stamps:
        left = mav / "cam0" / "data" / ("%d.png" % stamp)
        right = mav / "cam1" / "data" / ("%d.png" % stamp)
        if not left.is_file() or not right.is_file():
            raise FileNotFoundError("轨迹时间 %d 没有成对的左右图像" % stamp)
        left_frames.append(left)
        right_frames.append(right)
    imu_rows, imu_note = _imu_rows(mav / "imu0" / "data.csv", stamps[0], stamps[-1])
    imu_sensor = mav / "imu0" / "sensor.yaml"
    imu_is_body = False
    if imu_sensor.is_file():
        imu_T = _T_BS(_load_yaml(imu_sensor), imu_sensor)
        imu_is_body = bool(np.allclose(imu_T, np.eye(4), atol=1e-8))
    if not imu_is_body:
        T_cam_imu = None
        notes["T_cam_imu"] = "imu0 的 T_BS 不是单位阵，或不存在。不把相机外参写成 T_cam_imu。"

    root = Path(out)
    init_session(
        root,
        fps=float(fps),
        episode_id=episode_id or "euroc/%s" % root.name,
        image_width=int(calib["image_width"]),
        image_height=int(calib["image_height"]),
    )
    _write_video(left_frames, root / "stereo" / "left.mp4", fps)
    _write_video(right_frames, root / "stereo" / "right.mp4", fps)
    stamp_lines = ["frame_index,timestamp_s"]
    lr_lines = ["frame_index,left_s,right_s"]
    for index, stamp in enumerate(stamps):
        text = ns_to_second_text(stamp)
        stamp_lines.append("%d,%s" % (index, text))
        lr_lines.append("%d,%s,%s" % (index, text, text))
    (root / "timestamps.csv").write_text("\n".join(stamp_lines) + "\n", encoding="utf-8")
    (root / "stereo" / "timestamps_lr.csv").write_text("\n".join(lr_lines) + "\n", encoding="utf-8")
    imu_lines = ["timestamp_s,gx,gy,gz,ax,ay,az"]
    for stamp, values in imu_rows:
        imu_lines.append(
            "%s,%s" % (ns_to_second_text(stamp), ",".join("%.10g" % value for value in values))
        )
    (root / "imu.csv").write_text("\n".join(imu_lines) + "\n", encoding="utf-8")
    write_calibration_yaml(root / "calib.yaml", calib)
    meta_path = root / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["timestamps_source"] = "euroc_mav_filename_nanoseconds"
    meta["timestamps_are_hardware"] = False
    meta["hardware_capture"] = False
    meta["euroc"] = {
        "tum": str(Path(tum_path)),
        "frames": len(stamps),
        "fps_source": "图像文件名纳秒间隔的中位数，换成整数 Hz",
        "sensor_rate_hz": {
            "cam0": _load_yaml(cam0_yaml).get("rate_hz"),
            "cam1": _load_yaml(cam1_yaml).get("rate_hz"),
        },
        "imu_note": imu_note,
        "imu_rows": len(imu_rows),
        "calib_notes": notes,
        "unknown": ["exposure", "task", "instruction", "hands", "accuracy", "imu_cam_time_offset_s"],
    }
    if T_cam_imu is not None:
        meta["T_cam_imu"] = [[float(value) for value in row] for row in T_cam_imu.tolist()]
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    refresh_calib_json(root)
    return root


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="EuRoC mav0 → 头戴会话目录。对不上的字段保持未知。")
    parser.add_argument("--euroc", required=True, help="含 mav0 的目录，或 mav0 本身")
    parser.add_argument("--tum", required=True, help="SaveTrajectoryEuRoC 的轨迹。第一列是图像文件名的纳秒")
    parser.add_argument("--out", required=True)
    parser.add_argument("--episode-id", default=None)
    args = parser.parse_args(argv)
    try:
        root = euroc_to_session(args.euroc, args.tum, args.out, episode_id=args.episode_id)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(str(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
