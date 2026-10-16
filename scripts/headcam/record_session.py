# -*- coding: utf-8 -*-
"""头戴会话录制的落盘布局和命令行。

目录对齐 ``docs/headcam_data_spec.md`` §7.7，后面的 ``slam_vio`` / ``export_mvp``
直接读这个目录。硬件采集只生成 ffmpeg 计划：两个 v4l2 节点不能代替硬件同步，
本模块也不编造 IMU 样本或硬件时间戳。

``init`` 只写说明性的 metadata 和 IMU 表头，不写视频，不写 ``slam.tum``。
``import`` 把已经录好的文件拷进目录。没有时间戳文件时不按帧率补时间。
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

EXPECTED = (
    "metadata.json",
    "timestamps.csv",
    "imu.csv",
    "calib.yaml",
    "calib.json",
    "stereo/left.mp4",
    "stereo/right.mp4",
    "stereo/timestamps_lr.csv",
    "stereo/exposure.csv",
)

OPTIONAL = ("slam.tum", "trajectory.tum", "rgb.mp4")
IMU_HEADER = "timestamp_s,gx,gy,gz,ax,ay,az"


def _numeric_rows(path):
    if not path.is_file():
        return 0
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        token = line.replace(",", " ").split()
        if not token:
            continue
        try:
            float(token[0])
        except ValueError:
            continue
        count += 1
    return count


def layout_report(session):
    root = Path(session)
    files = {}
    for relative in EXPECTED + OPTIONAL:
        files[relative] = (root / relative).is_file()
    present_videos = files["stereo/left.mp4"] and files["stereo/right.mp4"]
    imu_rows = _numeric_rows(root / "imu.csv")
    return {
        "session": str(root),
        "files": files,
        "videos_present": present_videos,
        "timestamps_present": files["timestamps.csv"],
        "imu_file": files["imu.csv"],
        "imu_rows": imu_rows,
        "imu_present": imu_rows > 0,
        "calib_present": files["calib.yaml"] or files["calib.json"],
        "ready_for_export_hands": present_videos and files["timestamps.csv"] and files["calib.yaml"],
        "slam_tum_present": files["slam.tum"] or files["trajectory.tum"],
    }


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def init_session(out, fps=30.0, episode_id=None, task="", instruction="", environment="",
                 exposure_time_s=None, image_width=None, image_height=None):
    """创建空会话。不写视频，不写位姿，不写假的逐帧时间戳。"""
    root = Path(out)
    (root / "stereo").mkdir(parents=True, exist_ok=True)
    meta = {
        "episode_id": episode_id or "headcam/%s" % root.name,
        "fps": float(fps),
        "task": task,
        "instruction": instruction,
        "environment": environment,
        "objects": [],
        "verbs": [],
        "image_width": image_width,
        "image_height": image_height,
        "auto_exposure": False,
        "exposure_locked": True,
        "recorder": "headcam.record_session",
        "hardware_capture": False,
        "timestamps_source": None,
        "timestamps_are_hardware": None,
    }
    if exposure_time_s is not None:
        meta["exposure_time_s"] = float(exposure_time_s)
    _write_json(root / "metadata.json", meta)
    imu_path = root / "imu.csv"
    if not imu_path.exists():
        imu_path.write_text(IMU_HEADER + "\n# 还没有 IMU 样本。不要用空文件积分出位姿。\n", encoding="utf-8")
    manifest = layout_report(root)
    manifest["note"] = "init 只建立目录。视频、硬件时间戳和 slam.tum 都还没有。"
    _write_json(root / "record_manifest.json", manifest)
    return manifest


def _copy_if(src, dest):
    if src is None:
        return False
    source = Path(src)
    if not source.is_file():
        raise FileNotFoundError("找不到 %s" % source)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    return True


def import_session(out, left, right, imu=None, timestamps=None, timestamps_lr=None, exposure=None,
                   calib=None, fps=30.0, episode_id=None, hardware_timestamps=False):
    """把已有文件放进 §7.7 目录。缺时间戳就不补。"""
    root = Path(out)
    manifest = init_session(root, fps=fps, episode_id=episode_id)
    _copy_if(left, root / "stereo" / "left.mp4")
    _copy_if(right, root / "stereo" / "right.mp4")
    _copy_if(imu, root / "imu.csv")
    copied_timestamps = _copy_if(timestamps, root / "timestamps.csv")
    _copy_if(timestamps_lr, root / "stereo" / "timestamps_lr.csv")
    _copy_if(exposure, root / "stereo" / "exposure.csv")
    if calib is not None:
        suffix = Path(calib).suffix.lower()
        if suffix in (".yaml", ".yml"):
            _copy_if(calib, root / "calib.yaml")
        elif suffix == ".json":
            _copy_if(calib, root / "calib.json")
        else:
            raise ValueError("标定文件只接受 .yaml / .yml / .json：%s" % calib)
    meta_path = root / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["hardware_capture"] = False
    if copied_timestamps:
        meta["timestamps_source"] = "imported"
        meta["timestamps_are_hardware"] = bool(hardware_timestamps)
    else:
        meta["timestamps_source"] = None
        meta["timestamps_are_hardware"] = None
    _write_json(meta_path, meta)
    if (root / "calib.yaml").is_file() or (root / "calib.json").is_file() or (root / "stereo" / "timestamps_lr.csv").is_file():
        from headcam.sync_calib import refresh_calib_json
        refresh_calib_json(root)
    manifest = layout_report(root)
    manifest["note"] = "import 只拷贝文件。没有 timestamps 时不会按 fps 编造。"
    _write_json(root / "record_manifest.json", manifest)
    return manifest


def ffmpeg_commands(out, left_device, right_device, fps, seconds, size=None):
    """两个独立的 v4l2 命令。它们不同时保证左右曝光对齐。"""
    root = Path(out)
    def one(device, relative):
        command = ["ffmpeg", "-y", "-f", "v4l2", "-framerate", "%g" % float(fps)]
        if size:
            command.extend(["-video_size", str(size)])
        command.extend(["-i", str(device), "-t", "%g" % float(seconds), str(root / relative)])
        return command
    return {
        "left": one(left_device, "stereo/left.mp4"),
        "right": one(right_device, "stereo/right.mp4"),
        "imu": None,
        "note": (
            "两个 ffmpeg 各自打开一个 v4l2 节点，不能保证硬件同步。"
            "左右同步以 stereo/timestamps_lr.csv 的实测差为准。这里不录 IMU，也不写 slam.tum。"
        ),
    }


def capture_session(out, left_device="/dev/video0", right_device="/dev/video2", fps=30.0, seconds=10.0,
                    size=None, dry_run=True):
    """默认 dry-run：只写计划和空目录。设备或 ffmpeg 不在时不创建 mp4。"""
    root = Path(out)
    init_session(root, fps=fps)
    commands = ffmpeg_commands(root, left_device, right_device, fps, seconds, size=size)
    ffmpeg = shutil.which("ffmpeg")
    left_ok = Path(left_device).exists()
    right_ok = Path(right_device).exists()
    plan = {
        "recorded": False,
        "dry_run": bool(dry_run),
        "ffmpeg": ffmpeg,
        "left_device": str(left_device),
        "right_device": str(right_device),
        "left_device_exists": left_ok,
        "right_device_exists": right_ok,
        "commands": commands,
        "timestamps_are_hardware": False,
        "reason": None,
    }
    if dry_run:
        plan["reason"] = "dry-run：没有打开相机。"
    elif ffmpeg is None:
        plan["reason"] = "找不到 ffmpeg。"
    elif not (left_ok and right_ok):
        plan["reason"] = "视频设备不在：%s %s" % (left_device, right_device)
    else:
        (root / "stereo").mkdir(parents=True, exist_ok=True)
        processes = []
        for key in ("left", "right"):
            processes.append(subprocess.Popen(commands[key], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
        errors = []
        for proc in processes:
            _stdout, stderr = proc.communicate()
            if proc.returncode != 0:
                errors.append((stderr or "")[-1500:])
        if errors:
            plan["reason"] = "ffmpeg 失败：" + " | ".join(errors)
            plan["recorded"] = False
        else:
            plan["recorded"] = True
            plan["reason"] = (
                "视频已写入。时间戳不是硬件时钟；在补上 timestamps.csv 和 timestamps_lr.csv 之前，"
                "不要把这段当成已同步的会话。"
            )
    _write_json(root / "capture_plan.json", plan)
    meta_path = root / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["hardware_capture"] = bool(plan["recorded"])
    meta["timestamps_are_hardware"] = False
    _write_json(meta_path, meta)
    manifest = layout_report(root)
    manifest["capture"] = {"recorded": plan["recorded"], "reason": plan["reason"]}
    _write_json(root / "record_manifest.json", manifest)
    return plan


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="头戴会话目录：init / import / capture")
    sub = parser.add_subparsers(dest="cmd", required=True)

    init = sub.add_parser("init", help="建立空的 §7.7 目录")
    init.add_argument("--out", required=True)
    init.add_argument("--fps", type=float, default=30.0)
    init.add_argument("--episode-id", default=None)
    init.add_argument("--task", default="")
    init.add_argument("--exposure-time-s", type=float, default=None)

    imported = sub.add_parser("import", help="把已有视频和 CSV 放进会话目录")
    imported.add_argument("--out", required=True)
    imported.add_argument("--left", required=True)
    imported.add_argument("--right", required=True)
    imported.add_argument("--imu", default=None)
    imported.add_argument("--timestamps", default=None)
    imported.add_argument("--timestamps-lr", default=None)
    imported.add_argument("--exposure", default=None)
    imported.add_argument("--calib", default=None)
    imported.add_argument("--fps", type=float, default=30.0)
    imported.add_argument("--episode-id", default=None)
    imported.add_argument("--hardware-timestamps", action="store_true")

    capture = sub.add_parser("capture", help="v4l2 采集计划。默认不打开设备")
    capture.add_argument("--out", required=True)
    capture.add_argument("--left-device", default="/dev/video0")
    capture.add_argument("--right-device", default="/dev/video2")
    capture.add_argument("--fps", type=float, default=30.0)
    capture.add_argument("--seconds", type=float, default=10.0)
    capture.add_argument("--size", default=None, help="例如 1280x800")
    capture.add_argument("--run", action="store_true", help="设备都在时才真正调用 ffmpeg")

    args = parser.parse_args(argv)
    if args.cmd == "init":
        report = init_session(
            args.out, fps=args.fps, episode_id=args.episode_id, task=args.task,
            exposure_time_s=args.exposure_time_s,
        )
        print("videos_present=%s timestamps_present=%s" % (report["videos_present"], report["timestamps_present"]))
        return 0
    if args.cmd == "import":
        report = import_session(
            args.out, args.left, args.right, imu=args.imu, timestamps=args.timestamps,
            timestamps_lr=args.timestamps_lr, exposure=args.exposure, calib=args.calib,
            fps=args.fps, episode_id=args.episode_id, hardware_timestamps=args.hardware_timestamps,
        )
        print("ready_for_export_hands=%s" % str(report["ready_for_export_hands"]).lower())
        return 0
    plan = capture_session(
        args.out, left_device=args.left_device, right_device=args.right_device,
        fps=args.fps, seconds=args.seconds, size=args.size, dry_run=not args.run,
    )
    print("recorded=%s reason=%s" % (str(plan["recorded"]).lower(), plan["reason"]))
    if args.run and not plan["recorded"]:
        return 2
    return 0


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
