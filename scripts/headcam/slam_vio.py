# -*- coding: utf-8 -*-
"""双目 / 单目 + IMU 会话的世界系相机位姿接口。

输入是 ``docs/headcam_data_spec.md`` §7.7 的会话目录。输出是逐帧
``T_world_cam``，或者明确的 ``pose_missing`` / ``tracking_lost``。

没有轨迹时不写 ``slam.tum``，也不把单位阵当成世界位姿。单位阵只在下游
episode 里、并且 ``coordinate_frame=camera`` 时表示「还在相机系」
（见 ``hand_pose.associate_camera_poses``）。本模块不测量轨迹误差。

ORB-SLAM3 不放进仓库。装好之后用环境变量指向可执行文件，或把自己跑出来的
TUM 交给 ``--backend external_tum``。失败、超时、没写出轨迹，都记成
``pose_missing``。
"""
import json
import os
import subprocess
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import numpy as np

from egodata.schema import rotmat_to_quat_xyzw
from headcam.hand_pose import load_tum_trajectory
from headcam.time_units import describe_clocks

STATUS_OK = "ok"
STATUS_POSE_MISSING = "pose_missing"
STATUS_TRACKING_LOST = "tracking_lost"
STATUSES = (STATUS_OK, STATUS_POSE_MISSING, STATUS_TRACKING_LOST)

EUROC_STEPS = (
    "安装 ORB-SLAM3。本仓库不内置源码，也不在这里编译。",
    "按 timestamps.csv 把 stereo/left.mp4 和 stereo/right.mp4 抽成 EuRoC 的 mav0/cam0/data 与 mav0/cam1/data。抽帧不在本模块里做。",
    "把 imu.csv（timestamp_s,gx,gy,gz,ax,ay,az）换成 EuRoC imu0/data.csv 的纳秒时间戳。不要对 IMU 做双重积分来冒充世界位置。",
    "准备 Vocabulary/ORBvoc.txt 和一套双目惯性 yaml。设置 ORB_SLAM3_BIN、ORB_SLAM3_VOCAB、ORB_SLAM3_SETTINGS。",
    "自行运行 stereo_euroc 或 stereo_inertial_euroc，留下 CameraTrajectory.txt（tx ty tz qx qy qz qw）。UZ-SLAMLab 的 SaveTrajectoryEuRoC 把时间写成纳秒（内部秒乘 1e9），不是秒。KeyFrameTrajectory.txt 若来自 SaveKeyFrameTrajectoryEuRoC，也是同一套纳秒。关键帧轨迹盖不住的帧会标成 tracking_lost。",
    "python scripts/headcam/slam_vio.py --session SESSION --backend external_tum --tum CameraTrajectory.txt --tum-time-scale 1e-9 --out OUT",
    "本仓库没有测过这条轨迹的误差。不要把论文或别人的闭环数字写进报告。",
)


def read_timestamps(session):
    """读 ``timestamps.csv`` 的最后一列，单位秒。没有文件或没有数据行就报错。"""
    path = Path(session) / "timestamps.csv"
    if not path.is_file():
        raise FileNotFoundError("没有 timestamps.csv，无法逐帧标记位姿：%s" % path)
    stamps = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part for part in line.replace(",", " ").split() if part]
        if not parts:
            continue
        try:
            stamps.append(float(parts[-1]))
        except ValueError:
            continue
    if not stamps:
        raise ValueError("timestamps.csv 没有数据行：%s" % path)
    return stamps


def _metadata(session):
    path = Path(session) / "metadata.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def half_frame_seconds(timestamps, fps=None):
    """关联门限：半帧。与 ``validate_session`` 的「超过半帧算对不齐」一致。"""
    values = np.asarray(list(timestamps), dtype=float)
    if values.size >= 2:
        step = np.diff(values)
        step = step[step > 0]
        if step.size:
            return 0.5 * float(np.median(step))
    if fps:
        return 0.5 / float(fps)
    return 0.5 / 30.0


def _matrix_list(matrix):
    array = np.asarray(matrix, dtype=float)
    return [[float(value) for value in row] for row in array.tolist()]


def _find_tum(session):
    root = Path(session)
    for name in ("slam.tum", "trajectory.tum"):
        path = root / name
        if path.is_file():
            return path
    return None


def _load_poses(path):
    try:
        times, poses = load_tum_trajectory(path)
    except ValueError:
        return np.zeros(0, dtype=float), []
    return times, poses


def associate_poses(frame_times, tum_times, poses, max_gap_s):
    """最近邻关联。空隙大于 ``max_gap_s`` 的帧是 ``tracking_lost``，``T`` 为 None。

    没有轨迹点时每一帧都是 ``pose_missing``。不会填单位阵。
    """
    frames = []
    times = np.asarray(tum_times, dtype=float)
    if times.size == 0 or not poses:
        for stamp in frame_times:
            frames.append({
                "timestamp_s": float(stamp),
                "status": STATUS_POSE_MISSING,
                "gap_s": None,
                "T_world_cam": None,
            })
        return frames
    for stamp in frame_times:
        index = int(np.argmin(np.abs(times - float(stamp))))
        gap = abs(float(times[index]) - float(stamp))
        if gap > float(max_gap_s):
            frames.append({
                "timestamp_s": float(stamp),
                "status": STATUS_TRACKING_LOST,
                "gap_s": gap,
                "T_world_cam": None,
            })
        else:
            frames.append({
                "timestamp_s": float(stamp),
                "status": STATUS_OK,
                "gap_s": gap,
                "T_world_cam": _matrix_list(poses[index]),
            })
    return frames


def _coverage(frames):
    ok = sum(frame["status"] == STATUS_OK for frame in frames)
    missing = sum(frame["status"] == STATUS_POSE_MISSING for frame in frames)
    lost = sum(frame["status"] == STATUS_TRACKING_LOST for frame in frames)
    count = len(frames)
    if count and ok == count:
        coverage = "complete"
    elif ok == 0:
        coverage = "none"
    else:
        coverage = "partial"
    world_pose = coverage == "complete"
    coordinate_frame = "slam_world" if world_pose else "camera"
    return {
        "frame_count": count,
        "ok_frames": ok,
        "pose_missing_frames": missing,
        "tracking_lost_frames": lost,
        "pose_coverage": coverage,
        "world_pose": world_pose,
        "coordinate_frame": coordinate_frame,
    }


PARTIAL_HANDS_WARNING = (
    "world_hands=refused。不是每一帧都 ok，不能把双手标成世界系。"
    " --allow-partial 只把退出码改成 0，并不把 tracking_lost 或 pose_missing 配成世界位姿。"
)


def world_hands_gate(result, allow_partial=False):
    """世界系双手是否可以声称。

    ``allowed`` 只在每一帧都是 ``ok`` 时出现。``--allow-partial`` 不把残缺轨迹改成 allowed，
    只决定调用方要不要用非 0 退出码停住。本函数不测量精度。
    """
    complete = bool(result.get("world_pose"))
    return {
        "world_hands": "allowed" if complete else "refused",
        "allow_partial": bool(allow_partial) and not complete,
        "warning": None if complete else PARTIAL_HANDS_WARNING,
        "exit_code": 0 if complete or allow_partial else 1,
    }


def format_slam_counts(result, gate):
    """机器可读的帧数。精度行固定写成 unmeasured。"""
    lines = [
        "frames=%d" % int(result.get("frame_count") or 0),
        "ok=%d" % int(result.get("ok_frames") or 0),
        "tracking_lost=%d" % int(result.get("tracking_lost_frames") or 0),
        "pose_missing=%d" % int(result.get("pose_missing_frames") or 0),
        "world_pose=%s" % str(bool(result.get("world_pose"))).lower(),
        "world_hands=%s" % gate["world_hands"],
        "accuracy=unmeasured",
        "frame_time_unit=%s" % result.get("frame_time_unit", "unknown"),
        "tum_time_unit=%s" % result.get("tum_time_unit", "unknown"),
        "tum_time_scale=%s" % _format_scale(result.get("tum_time_scale", 1.0)),
        "time_unit_mismatch=%s" % str(bool(result.get("time_unit_mismatch"))).lower(),
    ]
    if result.get("time_unit_mismatch"):
        lines.append("time_unit_note=轨迹与会话时间单位不一致，未自动缩放")
    elif result.get("frame_time_unit") == "nanoseconds":
        lines.append("time_unit_note=会话时间戳像纳秒，未自动除以 1e9")
    if gate.get("warning"):
        lines.append("warning=%s" % gate["warning"])
    return "\n".join(lines) + "\n"


def _format_scale(scale):
    value = float(scale if scale is not None else 1.0)
    if value == 1.0:
        return "1"
    return "%.12g" % value


def orbslam3_plan(session, binary=None, vocab=None, settings=None):
    """描述怎么调用 ORB-SLAM3。文件不全时 ``available`` 为 false，``argv`` 为 None。"""
    binary = binary if binary is not None else os.environ.get("ORB_SLAM3_BIN", "")
    vocab = vocab if vocab is not None else os.environ.get("ORB_SLAM3_VOCAB", "")
    settings = settings if settings is not None else os.environ.get("ORB_SLAM3_SETTINGS", "")
    missing = []
    if not binary or not Path(binary).is_file():
        missing.append("ORB_SLAM3_BIN")
    if not vocab or not Path(vocab).is_file():
        missing.append("ORB_SLAM3_VOCAB")
    if not settings or not Path(settings).is_file():
        missing.append("ORB_SLAM3_SETTINGS")
    argv = None
    if not missing:
        root = Path(session).resolve()
        argv = [str(binary), str(vocab), str(settings), str(root), str(root / "timestamps.csv")]
    return {
        "available": not missing,
        "ran": False,
        "missing": missing,
        "argv": argv,
        "expected_output": ["CameraTrajectory.txt", "KeyFrameTrajectory.txt"],
        "euroc_steps": list(EUROC_STEPS),
        "notes": (
            "库存的 stereo_inertial_euroc 吃的是 EuRoC 序列，不是 §7.7 会话目录。"
            "直接把会话目录塞给它通常会失败；失败记成 pose_missing，不生成位姿。"
            "推荐自己跑完，再用 --tum 读 CameraTrajectory.txt。"
        ),
    }


def _result(session, frames, max_gap_s, backend, notes, orbslam3=None, imu_present=False, ran=False,
           time_fields=None):
    summary = _coverage(frames)
    plan = orbslam3 if orbslam3 is not None else orbslam3_plan(session)
    if ran:
        plan = dict(plan)
        plan["ran"] = True
    clocks = time_fields or {}
    return {
        "schema": "headcam_slam_status_v1",
        "session": str(session),
        "backend": backend,
        "max_gap_s": float(max_gap_s),
        "imu_present": bool(imu_present),
        "imu_used_for_pose": False,
        "accuracy": None,
        "frame_time_unit": clocks.get("frame_time_unit", "unknown"),
        "tum_time_unit": clocks.get("tum_time_unit", "unknown"),
        "tum_time_scale": float(clocks.get("tum_time_scale", 1.0)),
        "time_unit_mismatch": bool(clocks.get("time_unit_mismatch", False)),
        "time_unit_note": clocks.get("note") or "",
        "notes": notes,
        "orbslam3": plan,
        "frames": frames,
        **summary,
    }


def _notes_for_coverage(coverage, backend, detail):
    if coverage == "complete":
        return "每一帧都有 max_gap_s 以内的 T_world_cam。精度未测量。%s" % detail
    if coverage == "partial":
        return (
            "只有一部分帧有位姿。现有手部管线会用最近位姿填空隙，所以不要把这份残缺轨迹当成整段世界系。"
            "缺的帧是 tracking_lost，T_world_cam 为 null。%s" % detail
        )
    return "没有可用的世界位姿。不写单位阵 slam.tum。%s" % detail


def _join_note(text, extra):
    extra = (extra or "").strip()
    text = (text or "").strip()
    if extra and text:
        return text + " " + extra
    return text or extra


def _drop_cross_unit_poses(frames):
    """单位对不上时不保留最近邻位姿。状态改成 tracking_lost，矩阵保持 null。"""
    dropped = []
    for frame in frames:
        if frame.get("status") == STATUS_OK:
            dropped.append({
                "timestamp_s": frame["timestamp_s"],
                "status": STATUS_TRACKING_LOST,
                "gap_s": frame.get("gap_s"),
                "T_world_cam": None,
            })
        else:
            dropped.append(frame)
    return dropped


def result_from_tum(session, tum_path, frame_times, max_gap_s, backend, note="", orbslam3=None, ran=False,
                    tum_time_scale=None):
    times, poses = _load_poses(tum_path)
    clocks = describe_clocks(frame_times, times, tum_time_scale)
    extra = _join_note(note, clocks["note"])
    imu_present = (Path(session) / "imu.csv").is_file()
    if len(poses) == 0:
        text = "轨迹文件是空的（%s），按 pose_missing 处理，不发明位姿。%s" % (tum_path, extra)
        return _result(
            session, associate_poses(frame_times, [], [], max_gap_s), max_gap_s, backend,
            text.strip(), orbslam3=orbslam3, imu_present=imu_present, ran=ran, time_fields=clocks,
        )
    frames = associate_poses(frame_times, clocks["times_for_association"], poses, max_gap_s)
    if clocks["time_unit_mismatch"]:
        frames = _drop_cross_unit_poses(frames)
    coverage = _coverage(frames)["pose_coverage"]
    detail = _join_note("来源 %s，backend=%s。" % (tum_path, backend), extra)
    return _result(
        session, frames, max_gap_s, backend, _notes_for_coverage(coverage, backend, detail),
        orbslam3=orbslam3, imu_present=imu_present, ran=ran, time_fields=clocks,
    )


def _missing_result(session, frame_times, max_gap_s, backend, notes, orbslam3=None, tum_time_scale=None):
    clocks = describe_clocks(frame_times, [], tum_time_scale)
    frames = associate_poses(frame_times, [], [], max_gap_s)
    return _result(
        session, frames, max_gap_s, backend, _join_note(notes, clocks["note"]), orbslam3=orbslam3,
        imu_present=(Path(session) / "imu.csv").is_file(), ran=bool(orbslam3 and orbslam3.get("ran")),
        time_fields=clocks,
    )


def run_orbslam3_process(plan, run_dir, timeout_s=None):
    """执行计划里的 argv。返回码非 0 时不返回轨迹路径，避免把半成品当成成功。"""
    if not plan.get("available") or not plan.get("argv"):
        raise RuntimeError("ORB-SLAM3 计划不完整，不能执行")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(
            plan["argv"], cwd=str(run_dir), capture_output=True, text=True, timeout=timeout_s, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        err = exc.stderr or ""
        if isinstance(err, bytes):
            err = err.decode("utf-8", "replace")
        return {"returncode": None, "timed_out": True, "stderr": err[-2000:], "tum_path": None}
    stderr = (proc.stderr or "")[-2000:]
    if proc.returncode != 0:
        return {"returncode": proc.returncode, "timed_out": False, "stderr": stderr, "tum_path": None}
    tum_path = None
    for name in ("CameraTrajectory.txt", "slam.tum", "KeyFrameTrajectory.txt"):
        candidate = run_dir / name
        if candidate.is_file() and candidate.stat().st_size > 0:
            tum_path = candidate
            break
    return {"returncode": proc.returncode, "timed_out": False, "stderr": stderr, "tum_path": tum_path}


def run_slam(session, backend="auto", tum_path=None, max_gap_s=None, out_dir=None,
            run_orbslam3=None, timeout_s=None, tum_time_scale=None):
    """读会话并返回可 JSON 化的位姿结果。``auto`` 只读已有 TUM，不启动 ORB-SLAM3。

    ``backend=none`` 即使目录里已有 ``slam.tum`` 也标记 ``pose_missing``。
    ``backend=orbslam3`` 默认不执行二进制；设置 ``run_orbslam3=True`` 或环境变量
    ``ORB_SLAM3_RUN=1`` 才会执行。执行失败不发明位姿。

    ``tum_time_scale`` 只在调用者显式给出时缩放轨迹时间戳（纳秒轨迹用 ``1e-9``）。
    会话时间戳和轨迹单位不一致时不自动换算，也不采纳跨单位的位姿。
    """
    session = Path(session)
    frame_times = read_timestamps(session)
    meta = _metadata(session)
    if max_gap_s is None:
        max_gap_s = half_frame_seconds(frame_times, meta.get("fps"))
    backend = backend or "auto"
    imu_note = "IMU 文件%s，本模块不对它积分。" % ("在" if (session / "imu.csv").is_file() else "不在")
    explicit = Path(tum_path) if tum_path else None
    if explicit is None and os.environ.get("ORB_SLAM3_TUM"):
        explicit = Path(os.environ["ORB_SLAM3_TUM"])

    if backend == "none":
        result = _missing_result(
            session, frame_times, max_gap_s, "none",
            "backend=none：不读轨迹，每一帧 pose_missing。%s" % imu_note,
            tum_time_scale=tum_time_scale,
        )
    elif backend == "external_tum":
        path = explicit or _find_tum(session)
        if path is None or not path.is_file():
            result = _missing_result(
                session, frame_times, max_gap_s, "external_tum",
                "没有可读取的 TUM 轨迹，pose_missing。%s" % imu_note,
                tum_time_scale=tum_time_scale,
            )
        else:
            result = result_from_tum(
                session, path, frame_times, max_gap_s, "external_tum", "",
                tum_time_scale=tum_time_scale,
            )
    elif backend == "orbslam3":
        result = _run_orbslam3_backend(
            session, frame_times, max_gap_s, explicit, run_orbslam3, timeout_s, out_dir, imu_note,
            tum_time_scale=tum_time_scale,
        )
    elif backend == "auto":
        path = explicit or _find_tum(session)
        if path is not None and path.is_file():
            result = result_from_tum(
                session, path, frame_times, max_gap_s, "external_tum", "",
                tum_time_scale=tum_time_scale,
            )
        else:
            plan = orbslam3_plan(session)
            result = _missing_result(
                session, frame_times, max_gap_s, "none",
                "没有 slam.tum / trajectory.tum，也没有执行 ORB-SLAM3，每一帧 pose_missing。%s" % imu_note,
                orbslam3=plan, tum_time_scale=tum_time_scale,
            )
    else:
        raise ValueError("未知 backend %s（可用 auto、none、external_tum、orbslam3）" % backend)

    if out_dir is not None:
        out = Path(out_dir)
        status_path = write_slam_status(out / "slam_status.json", result)
        tum_path_out = out / "slam.tum"
        if result["world_pose"]:
            tum_written = write_slam_tum(tum_path_out, result)
        else:
            _remove_generated_tum(tum_path_out)
            tum_written = None
        result["slam_status"] = str(status_path)
        result["slam_tum"] = None if tum_written is None else str(tum_written)
    return result


def _run_orbslam3_backend(session, frame_times, max_gap_s, explicit, run_orbslam3, timeout_s, out_dir, imu_note,
                          tum_time_scale=None):
    should_run = run_orbslam3 if run_orbslam3 is not None else os.environ.get("ORB_SLAM3_RUN") == "1"
    plan = orbslam3_plan(session)
    if explicit is not None and explicit.is_file() and not should_run:
        result = result_from_tum(
            session, explicit, frame_times, max_gap_s, "orbslam3", "", orbslam3=plan, ran=False,
            tum_time_scale=tum_time_scale,
        )
        result["notes"] = _join_note(
            "读入了外部 TUM（%s），本进程没有执行 ORB-SLAM3，也没有测量精度。%s" % (explicit, imu_note),
            result.get("time_unit_note"),
        )
        result["orbslam3"] = dict(plan)
        result["orbslam3"]["ran"] = False
        result["orbslam3"]["source"] = str(explicit)
        return result
    if not should_run:
        return _missing_result(
            session, frame_times, max_gap_s, "orbslam3",
            "没有执行 ORB-SLAM3（未设置 ORB_SLAM3_RUN=1）。会话里已有的 slam.tum 不会被改称为 ORB-SLAM3 的输出。%s" % imu_note,
            orbslam3=plan, tum_time_scale=tum_time_scale,
        )
    if not plan["available"]:
        plan = dict(plan)
        plan["ran"] = False
        return _missing_result(
            session, frame_times, max_gap_s, "orbslam3",
            "ORB-SLAM3 还不能跑，缺少 %s。pose_missing。%s" % (", ".join(plan["missing"]), imu_note),
            orbslam3=plan, tum_time_scale=tum_time_scale,
        )
    run_dir = Path(out_dir or Path(session) / "orbslam3_run")
    if out_dir is not None:
        run_dir = Path(out_dir) / "orbslam3_run"
    executed = run_orbslam3_process(plan, run_dir, timeout_s=timeout_s)
    plan = dict(plan)
    plan["ran"] = True
    plan["returncode"] = executed["returncode"]
    plan["timed_out"] = executed["timed_out"]
    plan["stderr"] = executed["stderr"]
    if executed["tum_path"] is None:
        reason = "超时" if executed["timed_out"] else "退出码 %s，没有留下轨迹" % executed["returncode"]
        return _missing_result(
            session, frame_times, max_gap_s, "orbslam3",
            "ORB-SLAM3 %s。不使用残缺输出，pose_missing。%s" % (reason, imu_note),
            orbslam3=plan, tum_time_scale=tum_time_scale,
        )
    result = result_from_tum(
        session, executed["tum_path"], frame_times, max_gap_s, "orbslam3", "", orbslam3=plan, ran=True,
        tum_time_scale=tum_time_scale,
    )
    result["notes"] = _join_note(
        _notes_for_coverage(result["pose_coverage"], "orbslam3", "进程已执行，精度未测量。%s" % imu_note),
        result.get("time_unit_note"),
    )
    result["orbslam3"] = plan
    return result


def write_slam_status(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in result.items() if key not in ("slam_status", "slam_tum")}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _remove_generated_tum(path):
    """只删除本模块写过的轨迹，避免把用户原来的 slam.tum 清掉。"""
    path = Path(path)
    if not path.is_file():
        return
    first = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            first = line.strip()
            break
    if first.startswith("# headcam_slam_status_v1"):
        path.unlink()


def write_slam_tum(path, result):
    """只写 ``status=ok`` 的行。一行都没有时不创建文件。"""
    ok_frames = [
        frame for frame in result.get("frames") or []
        if frame.get("status") == STATUS_OK and frame.get("T_world_cam") is not None
    ]
    if not ok_frames:
        return None
    lines = ["# headcam_slam_status_v1 backend=%s accuracy=unmeasured" % result.get("backend")]
    for frame in ok_frames:
        matrix = np.asarray(frame["T_world_cam"], dtype=float)
        quat = rotmat_to_quat_xyzw(matrix[:3, :3])
        translation = matrix[:3, 3]
        lines.append("%.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f" % (
            float(frame["timestamp_s"]),
            float(translation[0]), float(translation[1]), float(translation[2]),
            float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]),
        ))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="会话目录 → 逐帧 T_world_cam 或 pose_missing / tracking_lost")
    parser.add_argument("--session", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--backend", default="auto", choices=("auto", "none", "external_tum", "orbslam3"))
    parser.add_argument("--tum", default=None, help="已有 TUM 轨迹。不执行 SLAM")
    parser.add_argument("--tum-time-scale", type=float, default=None,
                        help="显式缩放轨迹时间戳。纳秒轨迹用 1e-9。不自动换算")
    parser.add_argument("--max-gap-s", type=float, default=None)
    parser.add_argument("--run-orbslam3", action="store_true", help="backend=orbslam3 时真的执行二进制")
    parser.add_argument("--strict", action="store_true", help="没有完整世界位姿时退出码 1")
    args = parser.parse_args(argv)
    try:
        result = run_slam(
            args.session, backend=args.backend, tum_path=args.tum, max_gap_s=args.max_gap_s,
            out_dir=args.out, run_orbslam3=True if args.run_orbslam3 else None,
            tum_time_scale=args.tum_time_scale,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(
        "world_pose=%s coordinate_frame=%s ok=%d pose_missing=%d tracking_lost=%d "
        "frame_time_unit=%s tum_time_unit=%s time_unit_mismatch=%s accuracy=unmeasured"
        % (
            str(result["world_pose"]).lower(), result["coordinate_frame"],
            result["ok_frames"], result["pose_missing_frames"], result["tracking_lost_frames"],
            result["frame_time_unit"], result["tum_time_unit"],
            str(result["time_unit_mismatch"]).lower(),
        )
    )
    if args.strict and not result["world_pose"]:
        return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
