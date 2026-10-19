# -*- coding: utf-8 -*-
"""一条入口：会话 → 同步记录 → SLAM/VIO 状态 → 现有双目手部管线 → QC → LeRobot。

手部、QC、导出都调用已经在仓库里的模块（``stereo_pipeline.run_pipeline``、
``egodata.stereo_qc``、``egodata.qc``、``ego_to_lerobot``）。这里不重写
C 系列实验。

世界位姿只在 ``slam_vio`` 判定每一帧都 ``ok`` 时才交给手部管线。否则工作副本里
不放 ``slam.tum``，episode 保持 ``coordinate_frame=camera``。残缺轨迹的真实
``T_world_cam`` 留在 ``slam_status.json``，不用最近邻把丢失帧填上。
"""
import json
import shutil
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from egodata.qc import write_yield_reports, yield_report
from egodata.stereo_qc import qc_stereo_episode
from headcam.slam_frames import apply_ok_frame_world
from headcam.slam_vio import run_slam, world_hands_gate, write_slam_tum
from headcam.sync_calib import refresh_calib_json
from headcam.time_units import median_positive_step

SKIP_LINKS = {"slam.tum", "trajectory.tum", "slam_status.json"}


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if type(value).__module__ == "numpy":
        if hasattr(value, "tolist"):
            return _jsonable(value.tolist())
        if hasattr(value, "item"):
            return value.item()
    return value


def _dump(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _link_or_copy(src, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        dest.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dest)


def prepare_hand_session(session, dest, slam):
    """给手部管线的会话副本。只有完整世界轨迹才放入 ``slam.tum``。"""
    session = Path(session)
    dest = Path(dest)
    dest_resolved, session_resolved = dest.resolve(), session.resolve()
    if dest_resolved == session_resolved or dest_resolved in session_resolved.parents or session_resolved in dest_resolved.parents:
        raise ValueError("手部会话副本必须和源会话分开：%s" % dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for path in session.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(session)
        if relative.as_posix() in SKIP_LINKS or relative.name in SKIP_LINKS:
            continue
        _link_or_copy(path, dest / relative)
    tum_path = None
    if slam.get("world_pose"):
        tum_path = write_slam_tum(dest / "slam.tum", slam)
    return {"session": str(dest), "slam_tum": None if tum_path is None else str(tum_path)}


def slam_public_summary(slam):
    """给报告用的摘要。逐帧矩阵留在 slam_status.json，不放进这段摘要。"""
    return {
        "backend": slam.get("backend"),
        "world_pose": bool(slam.get("world_pose")),
        "coordinate_frame": slam.get("coordinate_frame"),
        "pose_coverage": slam.get("pose_coverage"),
        "frames": slam.get("frame_count"),
        "ok_frames": slam.get("ok_frames"),
        "pose_missing_frames": slam.get("pose_missing_frames"),
        "tracking_lost_frames": slam.get("tracking_lost_frames"),
        "max_gap_s": slam.get("max_gap_s"),
        "frame_time_unit": slam.get("frame_time_unit"),
        "tum_time_unit": slam.get("tum_time_unit"),
        "tum_time_scale": slam.get("tum_time_scale"),
        "time_unit_mismatch": bool(slam.get("time_unit_mismatch")),
        "accuracy": None,
        "pose_source": slam.get("pose_source"),
        "imu_used_for_pose": False,
        "notes": slam.get("notes"),
        "frame_convention": "world" if slam.get("world_pose") else "camera",
        "camera_pose_meaning": (
            "T_world_cam" if slam.get("world_pose")
            else "ok_frames_only" if int(slam.get("ok_frames") or 0) else "identity_placeholder_camera_frame"
        ),
    }


def annotate_episode(episode, slam):
    """在 episode 上写明位姿状态。不改关节，也不把丢失帧填成单位阵以外的世界位姿。"""
    summary = slam_public_summary(slam)
    frames = slam.get("frames")
    aligned = isinstance(frames, list) and len(frames) == int(episode.get("num_frames") or -1)
    summary["pose_status_aligned"] = bool(aligned)
    if aligned:
        summary["pose_status"] = [frame["status"] for frame in frames]
    hand_pose = episode.setdefault("hand_pose", {})
    if not isinstance(hand_pose, dict):
        hand_pose = {}
        episode["hand_pose"] = hand_pose
    hand_pose["slam"] = summary
    episode["world_pose"] = bool(slam.get("world_pose"))
    apply_ok_frame_world(episode, slam)
    return episode


def annotate_episode_dir(episodes_dir, slam):
    root = Path(episodes_dir)
    if not root.is_dir():
        return []
    written = []
    for path in sorted(root.glob("*.json")):
        episode = json.loads(path.read_text(encoding="utf-8"))
        annotate_episode(episode, slam)
        _dump(path, episode)
        written.append(str(path))
    return written


def qc_episode_dir(episodes_dir, out_dir):
    """对已有统一 episode 跑现有双目 QC，写出 yield.csv / yield.html。"""
    root = Path(episodes_dir)
    results = []
    paths = sorted(root.glob("*.json"))
    for path in paths:
        episode = json.loads(path.read_text(encoding="utf-8"))
        qc = qc_stereo_episode(episode)
        results.append({key: value for key, value in qc.items() if key != "good_frame_mask"})
    report = yield_report(results)
    qc_dir = Path(out_dir) / "qc"
    html_path, csv_path = write_yield_reports(report, qc_dir / "yield.html", qc_dir / "yield.csv")
    return report, html_path, csv_path


def export_lerobot_dir(episodes_dir, csv_path, lerobot_dir, repo_id):
    """调用现有导出。缺 pyarrow 时返回说明，不写一份假的数据集。"""
    try:
        from headcam.stereo_pipeline import export_accepted
    except ImportError as exc:
        return {"dir": None, "error": "导入导出模块失败：%s" % exc, "dependency_missing": True}
    try:
        return export_accepted(episodes_dir, csv_path, lerobot_dir, repo_id)
    except RuntimeError as exc:
        text = str(exc)
        missing = "pyarrow" in text or "pandas" in text
        script = Path(__file__).resolve().parents[1] / "ego_to_lerobot.py"
        return {
            "dir": None if missing else str(lerobot_dir),
            "error": text,
            "dependency_missing": missing,
            "command": [
                "python", str(script),
                "--episodes", str(episodes_dir),
                "--yield-csv", str(csv_path),
                "--out", str(lerobot_dir),
                "--repo-id", repo_id,
            ],
        }


def _default_hand_pipeline(sessions, out_dir, **kwargs):
    from headcam.stereo_pipeline import run_pipeline
    return run_pipeline(sessions, out_dir, **kwargs)


def _integer_fps(stamps, metadata_fps):
    """用时间戳中位间隔得到整数 fps。对不上就不四舍五入冒充。"""
    step = median_positive_step(stamps)
    if step is None or step <= 0:
        if metadata_fps is None:
            return None
        value = float(metadata_fps)
        rounded = int(round(value))
        if rounded < 1 or abs(value - rounded) > 1e-3:
            return None
        return rounded
    hz = 1.0 / float(step)
    rounded = int(round(hz))
    if rounded < 1 or abs(hz - rounded) / float(rounded) > 0.02:
        return None
    return rounded


def _intrinsic_from_calib(document):
    if not isinstance(document, dict) or document.get("calibration") != "stereo":
        return None
    left = document.get("left") or {}
    intrinsics = left.get("intrinsics")
    if not isinstance(intrinsics, list) or len(intrinsics) != 4:
        return None
    width = document.get("image_width")
    height = document.get("image_height")
    if width is None or height is None:
        return None
    fx, fy, cx, cy = [float(value) for value in intrinsics]
    return {
        "image_width": int(width),
        "image_height": int(height),
        "camera_intrinsic": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
    }


def _unmeasured_hands(num_frames):
    """21 个点都是 null。不是零坐标。"""
    return {
        "joints": [[[None, None, None] for _ in range(21)] for _ in range(num_frames)],
        "confidence": [None] * num_frames,
        "wrist_pose": [[None, None, None, None, None, None, None] for _ in range(num_frames)],
        "filled": [False] * num_frames,
    }


def prepare_unmeasured_episode(session, slam, calib_document, episode_dir):
    """没跑手部时写一条 episode，交给现有 QC 和 LeRobot。

    手部标成 ``unmeasured``，不填 0。标定或整数 fps 对不上时不写文件。
    返回 ``(path or None, reason)``。
    """
    frames = slam.get("frames") if isinstance(slam, dict) else None
    if not isinstance(frames, list) or not frames:
        return None, "没有逐帧位姿状态，未导出"
    camera = _intrinsic_from_calib(calib_document)
    if camera is None:
        return None, "没有可用的双目标定，未把缺测手写成数据集"
    stamps = [frame.get("timestamp_s") for frame in frames]
    if any(stamp is None for stamp in stamps):
        return None, "有的帧没有 timestamp_s，未导出"
    meta_path = Path(session) / "metadata.json"
    meta = {}
    if meta_path.is_file():
        loaded = json.loads(meta_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            meta = loaded
    fps = _integer_fps(stamps, meta.get("fps"))
    if fps is None:
        return None, "帧间隔不是整数 fps，没有四舍五入，未导出"
    num_frames = len(frames)
    poses = []
    for frame in frames:
        if frame.get("status") == "ok" and frame.get("T_world_cam") is not None:
            poses.append(frame["T_world_cam"])
        else:
            poses.append(None)
    video = Path(session) / "stereo" / "left.mp4"
    episode = {
        "episode_id": meta.get("episode_id") or "headcam/%s" % Path(session).name,
        "num_frames": num_frames,
        "fps": float(fps),
        "timestamps": [float(stamp) for stamp in stamps],
        "image_width": camera["image_width"],
        "image_height": camera["image_height"],
        "camera_intrinsic": camera["camera_intrinsic"],
        "camera_poses": poses,
        "coordinate_frame": slam.get("coordinate_frame") or "camera",
        "hands_measured": False,
        "hands": {"left": _unmeasured_hands(num_frames), "right": _unmeasured_hands(num_frames)},
        "stereo": {
            "per_frame": {
                "left": ["unmeasured"] * num_frames,
                "right": ["unmeasured"] * num_frames,
            },
        },
    }
    instruction = str(meta.get("instruction") or "").strip()
    if instruction:
        episode["annotation"] = {"task": {"instruction": instruction}}
    if video.is_file() and video.stat().st_size > 0:
        episode["video_path"] = str(video.resolve())
    annotate_episode(episode, slam)
    episode_dir = Path(episode_dir)
    episode_dir.mkdir(parents=True, exist_ok=True)
    for old in episode_dir.glob("*.json"):
        old.unlink()
    path = _dump(episode_dir / "session.json", episode)
    return path, None


def export_mvp(session, out, slam_backend="auto", tum_path=None, max_gap_s=None, run_hands=True,
               episodes=None, export=True, hand_pipeline=None, repo_id="local/headcam_mvp",
               run_orbslam3=None, allow_partial=False, tum_time_scale=None, pose_source=None, log=print):
    """跑 MVP 前半段并接上现有 QC / LeRobot。返回报告字典，同时写 ``out/report.json``。"""
    session = Path(session)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if not (session / "timestamps.csv").is_file():
        raise FileNotFoundError("export_mvp 需要 timestamps.csv：%s" % session)

    calib_document = refresh_calib_json(session, out_path=out / "calib.json")
    slam = run_slam(
        session, backend=slam_backend, tum_path=tum_path, max_gap_s=max_gap_s,
        out_dir=out, run_orbslam3=run_orbslam3, tum_time_scale=tum_time_scale,
        pose_source=pose_source,
    )
    gate = world_hands_gate(slam, allow_partial=allow_partial)
    hand_session = prepare_hand_session(session, out / "hand_session", slam)
    use_existing = episodes is not None
    hands = {"status": "skipped", "reason": "run_hands 关闭", "world_hands": gate["world_hands"]}
    pipeline_summary = None
    if use_existing:
        hands = {"status": "external_episodes", "episodes": str(episodes), "world_hands": gate["world_hands"]}
        run_hands = False
    elif run_hands and gate["world_hands"] != "allowed" and not allow_partial:
        hands = {
            "status": "refused",
            "world_hands": "refused",
            "reason": gate["warning"],
        }
        run_hands = False
    elif run_hands:
        pipeline = hand_pipeline or _default_hand_pipeline
        try:
            pipeline_summary = pipeline(
                [hand_session["session"]], out, export=False, repo_id=repo_id, log=log,
            )
            hands = {
                "status": "ran",
                "session": hand_session["session"],
                "world_hands": gate["world_hands"],
            }
            if gate.get("warning"):
                hands["warning"] = gate["warning"]
        except Exception as exc:  # noqa: BLE001 — 失败写进报告，由 CLI 用退出码表示
            hands = {
                "status": "failed",
                "world_hands": gate["world_hands"],
                "error": "%s: %s" % (type(exc).__name__, exc),
                "session": hand_session["session"],
            }
            pipeline_summary = None

    episode_dir = Path(episodes) if episodes else out / "episodes"
    if hands["status"] in ("ran", "external_episodes") and episode_dir.is_dir():
        annotate_episode_dir(episode_dir, slam)

    unmeasured_reason = None
    if hands["status"] == "skipped" and export and episodes is None:
        _episode_path, unmeasured_reason = prepare_unmeasured_episode(
            session, slam, calib_document, out / "episodes",
        )
        if _episode_path is not None:
            episode_dir = out / "episodes"

    qc_report = None
    csv_path = None
    lerobot = None
    pose_only = hands["status"] == "skipped" and export and episode_dir.is_dir() and any(episode_dir.glob("*.json"))
    should_qc = hands["status"] in ("ran", "external_episodes") and episode_dir.is_dir() and any(episode_dir.glob("*.json"))
    if hands["status"] == "ran" and isinstance(pipeline_summary, dict):
        qc_report = {
            "yield": pipeline_summary.get("yield"),
            "raw_frames": pipeline_summary.get("raw_frames"),
            "usable_frames": pipeline_summary.get("usable_frames"),
            "episodes": pipeline_summary.get("episodes"),
            "accepted_episodes": pipeline_summary.get("accepted_episodes"),
            "source": "stereo_pipeline.run_pipeline",
        }
        csv_candidate = out / "qc" / "yield.csv"
        if csv_candidate.is_file():
            csv_path = csv_candidate
    elif should_qc or hands["status"] == "external_episodes" or pose_only:
        if any(episode_dir.glob("*.json")):
            qc_report, _html, csv_path = qc_episode_dir(episode_dir, out)
            qc_report = {
                "yield": qc_report.get("yield"),
                "raw_frames": qc_report.get("raw_frames"),
                "usable_frames": qc_report.get("usable_frames"),
                "episodes": qc_report.get("episodes"),
                "accepted_episodes": qc_report.get("accepted_episodes"),
                "source": "egodata.stereo_qc.qc_stereo_episode",
            }

    accepted = 0 if not qc_report else int(qc_report.get("accepted_episodes") or 0)
    if hands["status"] == "refused":
        lerobot = {"dir": None, "result": "世界系双手被拒绝，未跑手部，未导出"}
    elif not export:
        lerobot = {"dir": None, "result": "未请求导出"}
    elif hands["status"] == "skipped" and not pose_only:
        lerobot = {
            "dir": None,
            "result": unmeasured_reason or "未跑手部，也没有已有 episode，未导出",
        }
    elif export and accepted and csv_path is not None:
        lerobot = export_lerobot_dir(episode_dir, csv_path, out / "lerobot", repo_id)
    else:
        lerobot = {"dir": None, "result": "没有通过 QC 的片段，未导出"}

    report = {
        "session": str(session),
        "world_pose": bool(slam.get("world_pose")),
        "world_hands": hands.get("world_hands"),
        "coordinate_frame": slam.get("coordinate_frame"),
        "sync_error_ms": calib_document.get("sync_error_ms"),
        "calibration": calib_document.get("calibration"),
        "calib_json": str(out / "calib.json"),
        "slam": slam_public_summary(slam),
        "slam_status": slam.get("slam_status"),
        "slam_tum": slam.get("slam_tum"),
        "hand_session": hand_session,
        "hands": hands,
        "pipeline": pipeline_summary,
        "qc": qc_report,
        "lerobot": lerobot,
        "pose_source": slam.get("pose_source"),
        "accuracy": None,
        "limits": [
            "没有完整 SLAM 轨迹时 world_pose=false，不写单位阵 slam.tum。",
            "覆盖不全的轨迹不会交给手部管线，避免最近邻把 tracking_lost 填成世界位姿。",
            "已有相机系 episode 上，只有 status=ok 的帧写 p_world = T_world_cam @ p_cam。覆盖不全时这些点在 hand_pose.slam.world_hands，丢失帧为 null，手腕主数组仍在相机系。",
            "accuracy 始终是 null：本入口不测量 SLAM 误差。",
            "没跑手部时，通过 QC 的片段仍可导出。手部列不写，也不用 0 填充。",
            "轨迹时间单位和会话不一致时不自动换算，也不把跨单位的最近邻写成 T_world_cam。",
        ],
    }
    if slam.get("pose_source") and str(slam.get("pose_source")).startswith("hot3d_gt"):
        report["limits"].append(
            "pose_source=%s。轨迹是 HOT3D 头戴地面真值，不是 SLAM 输出。accuracy 仍是 null。"
            % slam.get("pose_source")
        )
    if slam.get("time_unit_mismatch"):
        report["limits"].append(
            "time_unit_mismatch=true。确认轨迹是纳秒后，用 --tum-time-scale 1e-9 再跑，不要放宽 max_gap_s。"
        )
    if slam.get("pose_coverage") == "partial":
        report["limits"].append("ok 帧的 T_world_cam 在 slam_status.json，episode 仍按相机系处理。")
    if gate.get("warning"):
        report["limits"].append(gate["warning"])
    _dump(out / "report.json", report)
    if log:
        sync = report["sync_error_ms"] or {}
        log(
            "world_pose=%s ok=%s pose_missing=%s tracking_lost=%s sync_left_right_max_ms=%s "
            "time_unit_mismatch=%s hands=%s"
            % (
                str(report["world_pose"]).lower(),
                slam.get("ok_frames"), slam.get("pose_missing_frames"), slam.get("tracking_lost_frames"),
                sync.get("left_right_max"),
                str(bool(slam.get("time_unit_mismatch"))).lower(),
                hands.get("status"),
            )
        )
    return report


def main(argv=None):
    import argparse
    import sys
    parser = argparse.ArgumentParser(description="头戴 MVP：同步 + 位姿状态 + 现有 QC/LeRobot")
    parser.add_argument("--session", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--slam-backend", default="auto", choices=("auto", "none", "external_tum", "orbslam3"))
    parser.add_argument("--tum", default=None)
    parser.add_argument("--tum-time-scale", type=float, default=None,
                        help="显式缩放轨迹时间戳。纳秒轨迹用 1e-9。不自动换算")
    parser.add_argument("--max-gap-s", type=float, default=None)
    parser.add_argument("--skip-hands", action="store_true", help="不跑 WiLoR / 手部管线，只写同步和位姿状态")
    parser.add_argument("--allow-partial", action="store_true",
                        help="轨迹不完整时仍在相机系跑手部。世界系双手仍然拒绝，并打印警告")
    parser.add_argument("--episodes", default=None, help="已有统一 episode 目录：跳过手部，做 QC 和 LeRobot")
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--run-orbslam3", action="store_true")
    parser.add_argument("--repo-id", default="local/headcam_mvp")
    parser.add_argument("--pose-source", default=None,
                        help="轨迹来源。HOT3D 头戴真值用 hot3d_gt_mocap，不要把它叫成 SLAM")
    args = parser.parse_args(argv)
    try:
        report = export_mvp(
            args.session, args.out, slam_backend=args.slam_backend, tum_path=args.tum,
            max_gap_s=args.max_gap_s, run_hands=not args.skip_hands, episodes=args.episodes,
            export=not args.no_export, repo_id=args.repo_id,
            run_orbslam3=True if args.run_orbslam3 else None,
            allow_partial=args.allow_partial, tum_time_scale=args.tum_time_scale,
            pose_source=args.pose_source,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if report["hands"].get("warning"):
        print(report["hands"]["warning"], file=sys.stderr)
    if report["hands"].get("status") == "refused":
        print(report["hands"].get("reason"), file=sys.stderr)
        return 1
    if report["hands"].get("status") == "failed":
        print(report["hands"].get("error"), file=sys.stderr)
        return 1
    lerobot = report.get("lerobot") or {}
    if lerobot.get("error") and not lerobot.get("dependency_missing"):
        print(lerobot["error"], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
