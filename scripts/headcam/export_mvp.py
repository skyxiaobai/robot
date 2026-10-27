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
from headcam.hot3d_check import shared_measured_accuracy
from headcam.slam_frames import apply_ok_frame_world
from headcam.slam_vio import run_slam, world_hands_gate, write_slam_status, write_slam_tum
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


def _published_accuracy(slam):
    """只公布测过的 Umeyama 结构。手写数字、缺字段、头戴真值来源都是 null。"""
    from headcam.hot3d_check import is_measured_accuracy
    if not isinstance(slam, dict):
        return None
    value = slam.get("accuracy")
    if is_measured_accuracy(value, slam.get("pose_source")):
        return value
    return None


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
        "accuracy": _published_accuracy(slam),
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


def _append_note(text, extra):
    extra = (extra or "").strip()
    text = (text or "").strip()
    if extra and text:
        return text + " " + extra
    return text or extra


def _apply_accuracy_publication(slam, publication):
    """用调用方已经核定的测量替换 accuracy。手写数字在这里拒绝。"""
    from headcam.hot3d_check import is_measured_accuracy

    value = publication.get("value")
    source = slam.get("pose_source")
    if value is None:
        slam["accuracy"] = None
    elif is_measured_accuracy(value, source):
        slam["accuracy"] = value
        notes = slam.get("notes")
        if isinstance(notes, str):
            slam["notes"] = notes.replace(
                "没有另给真值轨迹时 accuracy 仍是 null。",
                "accuracy 用的是这次给出的测量。",
            )
    else:
        raise ValueError("要写入的 accuracy 不是测过的 Umeyama，已拒绝。")
    note = publication.get("note")
    if note:
        slam["notes"] = _append_note(slam.get("notes"), note)
    status_path = slam.get("slam_status")
    if status_path:
        write_slam_status(status_path, slam)
    return slam


def export_mvp(session, out, slam_backend="auto", tum_path=None, max_gap_s=None, run_hands=True,
               episodes=None, export=True, hand_pipeline=None, repo_id="local/headcam_mvp",
               run_orbslam3=None, allow_partial=False, tum_time_scale=None, pose_source=None,
               gt_tum_path=None, gt_time_scale=None, accuracy_publication=None, log=print):
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
        pose_source=pose_source, gt_tum_path=gt_tum_path, gt_time_scale=gt_time_scale,
    )
    if accuracy_publication is not None:
        slam = _apply_accuracy_publication(slam, accuracy_publication)
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
        "accuracy": _published_accuracy(slam),
        "limits": [
            "没有完整 SLAM 轨迹时 world_pose=false，不写单位阵 slam.tum。",
            "覆盖不全的轨迹不会交给手部管线，避免最近邻把 tracking_lost 填成世界位姿。",
            "已有相机系 episode 上，只有 status=ok 的帧写 p_world = T_world_cam @ p_cam。覆盖不全时这些点在 hand_pose.slam.world_hands，丢失帧为 null，手腕主数组仍在相机系。",
            "accuracy 只在给了另一份真值轨迹、并且来源不是头戴真值时写入 Umeyama ATE/RPE。否则是 null。",
            "没跑手部时，通过 QC 的片段仍可导出。手部列不写，也不用 0 填充。",
            "轨迹时间单位和会话不一致时不自动换算，也不把跨单位的最近邻写成 T_world_cam。",
        ],
    }
    if accuracy_publication is not None:
        report["accuracy_source"] = accuracy_publication.get("source")
        report["accuracy_note"] = accuracy_publication.get("note")
        report["batch_sequence"] = accuracy_publication.get("sequence")
        report["limits"] = [
            (
                "accuracy 只接受 batch 行里的 trajectory_error_v1，或这次对真值测到的同一类结构。"
                "汇总的 min/median/max、手写数字、对不上的两份误差都不写。"
            )
            if "accuracy 只在给了另一份真值轨迹" in item else item
            for item in report["limits"]
        ]
        if accuracy_publication.get("note"):
            report["limits"].append(accuracy_publication["note"])
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


def _require_annotated_source(episode):
    """已标好的世界片段必须自带逐帧位姿和同一个来源名。不猜测。"""
    episode_id = episode.get("episode_id")
    slam = (episode.get("hand_pose") or {}).get("slam") if isinstance(episode.get("hand_pose"), dict) else None
    if not isinstance(slam, dict):
        raise ValueError("%s 没有 hand_pose.slam，不能当成已经标好的世界片段" % episode_id)
    if slam.get("timestamp_alignment") == "refused":
        raise ValueError("%s 的时间对齐被拒绝，不打包" % episode_id)
    if slam.get("world_points_scope") != "all_frames":
        raise ValueError(
            "%s 的 world_points_scope=%s，不是 all_frames。不把相机系手腕和世界系打在一个包里。"
            % (episode_id, slam.get("world_points_scope"))
        )
    source = slam.get("pose_source")
    if not isinstance(source, str) or not source:
        raise ValueError("%s 没有 pose_source" % episode_id)
    num_frames = int(episode.get("num_frames") or 0)
    statuses = slam.get("pose_status")
    matrices = slam.get("T_world_cam")
    if not isinstance(statuses, list) or not isinstance(matrices, list):
        raise ValueError("%s 的逐帧位姿不完整" % episode_id)
    if len(statuses) < num_frames or len(matrices) < num_frames:
        raise ValueError("%s 的逐帧位姿短于帧数" % episode_id)
    return source


def _qc_brief(item):
    return {
        "episode_id": item.get("episode_id"),
        "num_frames": item.get("num_frames"),
        "accepted": bool(item.get("accepted")),
        "bad_frames": item.get("bad_frames"),
        "bad_fraction": item.get("bad_fraction"),
        "labeled_frames": item.get("labeled_frames"),
        "dropped_label_frames": item.get("dropped_label_frames"),
        "label_coverage": item.get("label_coverage"),
        "reasons": list(item.get("reasons") or []),
        "flags": item.get("flags"),
    }


def package_annotated_episodes(episodes_dir, out, repo_id="local/headcam_mvp", log=print):
    """把各自已经写好世界位姿的 episode 打成一个 LeRobot 数据集。

    不读取某一个会话的轨迹，也不调用 ``annotate_episode``。
    每一份的 ``hand_pose.slam.T_world_cam`` 原样进入 ``observation.pose``。
    ``pose_source`` 必须相同。``accuracy`` 只在每一份被接受的 episode 上都有
    同一份测过的 Umeyama 结构、并且来源不是头戴真值时写入。手写数字不抄。
    QC 拒绝的片段留在报告里，不写进包。
    """
    root = Path(episodes_dir)
    if not root.is_dir():
        raise FileNotFoundError("没有 episode 目录：%s" % root)
    paths = sorted(path for path in root.glob("*.json") if path.is_file())
    if not paths:
        raise ValueError("没有可打包的 episode：%s" % root)
    sources = []
    loaded = []
    for path in paths:
        episode = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(episode, dict):
            raise ValueError("episode 必须是对象：%s" % path)
        sources.append(_require_annotated_source(episode))
        loaded.append(episode)
    unique = sorted(set(sources))
    if len(unique) != 1:
        raise ValueError("同一次导出里的 pose_source 不一致：%s" % unique)
    pose_source = unique[0]

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    qc_report, _html, csv_path = qc_episode_dir(root, out)
    packaged = []
    rejected = []
    for item in qc_report.get("results") or []:
        brief = _qc_brief(item)
        if brief["accepted"]:
            packaged.append(brief)
        else:
            rejected.append(brief)
    accepted = int(qc_report.get("accepted_episodes") or 0)
    accepted_ids = {item["episode_id"] for item in packaged}
    accuracy = shared_measured_accuracy([
        episode for episode in loaded if episode.get("episode_id") in accepted_ids
    ])
    if accepted and csv_path is not None:
        lerobot = export_lerobot_dir(root, csv_path, out / "lerobot", repo_id)
    else:
        lerobot = {"dir": None, "result": "没有通过 QC 的片段，未导出"}
    report = {
        "package": "annotated_episodes",
        "episodes_dir": str(root),
        "pose_source": pose_source,
        "accuracy": accuracy,
        "episodes": len(paths),
        "packaged_episodes": packaged,
        "rejected_episodes": rejected,
        "qc": {
            "yield": qc_report.get("yield"),
            "raw_frames": qc_report.get("raw_frames"),
            "usable_frames": qc_report.get("usable_frames"),
            "episodes": qc_report.get("episodes"),
            "accepted_episodes": qc_report.get("accepted_episodes"),
            "rejected_count": qc_report.get("rejected_episodes"),
            "source": "egodata.stereo_qc.qc_stereo_episode",
        },
        "lerobot": lerobot,
        "limits": [
            "每一份 episode 用自己的 hand_pose.slam.T_world_cam。没有用某一个会话的轨迹覆盖其他片段。",
            "pose_source=%s。这个字段只标明轨迹从哪来。" % pose_source,
            "accuracy 只接受测过的 Umeyama 结构。手写数字、头戴真值来源、各段不一致时都是 null。",
            "QC 拒绝的片段留在 rejected_episodes，不写进 LeRobot 包。",
        ],
    }
    _dump(out / "report.json", _jsonable(report))
    if log:
        log(
            "package=annotated episodes=%d accepted=%d rejected=%d pose_source=%s accuracy=%s"
            % (len(paths), accepted, len(rejected), pose_source, "null" if accuracy is None else "measured")
        )
    return report


def _finite_pose_matrix(matrix):
    import numpy as np

    pose = np.asarray(matrix, dtype=float)
    return pose.shape == (4, 4) and bool(np.isfinite(pose).all())


def _pose_is_identity(matrix, atol=1e-8):
    import numpy as np

    if not _finite_pose_matrix(matrix):
        return False
    return bool(np.allclose(np.asarray(matrix, dtype=float), np.eye(4), atol=atol))


def _unmeasured_pose_counts(episode):
    slam = (episode.get("hand_pose") or {}).get("slam") or {}
    statuses = slam.get("pose_status") or []
    return {
        "ok_frames": sum(1 for status in statuses if status == "ok"),
        "tracking_lost_frames": sum(1 for status in statuses if status == "tracking_lost"),
        "pose_missing_frames": sum(1 for status in statuses if status == "pose_missing"),
    }


def _validate_unmeasured_pose_episode(episode):
    """缺测手的外部轨迹。丢失帧必须是 null，不能是单位阵。"""
    if not isinstance(episode, dict):
        raise ValueError("episode 必须是对象")
    episode_id = episode.get("episode_id")
    if episode.get("hands_measured") is not False:
        raise ValueError("%s 不是缺测手。这条打包不把手放进世界系。" % episode_id)
    slam = (episode.get("hand_pose") or {}).get("slam") if isinstance(episode.get("hand_pose"), dict) else None
    if not isinstance(slam, dict):
        raise ValueError("%s 没有 hand_pose.slam" % episode_id)
    if slam.get("timestamp_alignment") == "refused":
        raise ValueError("%s 的时间对齐被拒绝，不打包" % episode_id)
    scope = slam.get("world_points_scope")
    if scope not in ("all_frames", "ok_frames_only", "already_world"):
        raise ValueError(
            "%s 的 world_points_scope=%s。没有逐帧外部位姿，不打包。"
            % (episode_id, scope)
        )
    source = slam.get("pose_source")
    if not isinstance(source, str) or not source:
        raise ValueError("%s 没有 pose_source" % episode_id)
    if source.startswith("hot3d_gt"):
        raise ValueError("%s 的 pose_source=%s。这是头戴真值，不是测过的外部轨迹。" % (episode_id, source))
    num_frames = int(episode.get("num_frames") or 0)
    statuses = slam.get("pose_status")
    matrices = slam.get("T_world_cam")
    if not isinstance(statuses, list) or not isinstance(matrices, list):
        raise ValueError("%s 的逐帧位姿不完整" % episode_id)
    if len(statuses) != num_frames or len(matrices) != num_frames:
        raise ValueError("%s 的逐帧位姿长度和帧数不一致" % episode_id)
    ok_poses = []
    for index, (status, matrix) in enumerate(zip(statuses, matrices)):
        if status not in ("ok", "tracking_lost", "pose_missing"):
            raise ValueError("%s 第 %d 帧的 slam_status 无法识别：%s" % (episode_id, index, status))
        if status == "ok":
            if not _finite_pose_matrix(matrix):
                raise ValueError("%s 第 %d 帧是 ok，但没有有限的 T_world_cam" % (episode_id, index))
            ok_poses.append(matrix)
        elif matrix is not None:
            raise ValueError("%s 第 %d 帧是 %s，位姿必须是 null，不能写成单位阵。" % (episode_id, index, status))
    if not ok_poses or all(_pose_is_identity(matrix) for matrix in ok_poses):
        raise ValueError("%s 的 ok 位姿全是单位阵，或者没有 ok 帧。不把单位阵当成世界轨迹，也不写出包。" % episode_id)
    return source


def package_unmeasured_pose_episodes(episodes_dir, out, repo_id="local/headcam_mvp", log=print):
    """把各自已经写好外部轨迹的缺测手 episode 打成一个 LeRobot 包。

    不读取某一个会话的轨迹，也不跑手。``pose_source`` 必须相同，而且不能是头戴真值。
    丢失帧的 ``T_world_cam`` 必须是 null。ok 位姿全是单位阵时不写 ``out``。
    各段测到的 accuracy 不一致时，包级 accuracy 保持 null，不取中位数。
    """
    from headcam.hot3d_check import is_measured_accuracy

    root = Path(episodes_dir)
    if not root.is_dir():
        raise FileNotFoundError("没有 episode 目录：%s" % root)
    paths = sorted(path for path in root.glob("*.json") if path.is_file())
    if not paths:
        raise ValueError("没有可打包的 episode：%s" % root)
    loaded = []
    sources = []
    for path in paths:
        episode = json.loads(path.read_text(encoding="utf-8"))
        sources.append(_validate_unmeasured_pose_episode(episode))
        loaded.append(episode)
    ids = [episode.get("episode_id") for episode in loaded]
    if len(ids) != len(set(ids)):
        raise ValueError("episode_id 重复，不能打进同一个包：%s" % ids)
    unique = sorted(set(sources))
    if len(unique) != 1:
        raise ValueError("同一次导出里的 pose_source 不一致：%s" % unique)
    pose_source = unique[0]

    out = Path(out)
    if out.exists():
        raise FileExistsError("输出目录已经存在，不覆盖：%s" % out)
    out.mkdir(parents=True)
    qc_report, _html, csv_path = qc_episode_dir(root, out)
    by_id = {episode.get("episode_id"): episode for episode in loaded}
    packaged = []
    rejected = []
    for item in qc_report.get("results") or []:
        brief = _qc_brief(item)
        episode = by_id.get(brief["episode_id"])
        counts = _unmeasured_pose_counts(episode) if episode is not None else {}
        brief.update(counts)
        slam = ((episode or {}).get("hand_pose") or {}).get("slam") or {}
        value = slam.get("accuracy")
        brief["accuracy"] = value if is_measured_accuracy(value, slam.get("pose_source")) else None
        brief["accuracy_note"] = slam.get("notes")
        if brief["accepted"]:
            packaged.append(brief)
        else:
            rejected.append(brief)
    accepted = int(qc_report.get("accepted_episodes") or 0)
    accepted_ids = {item["episode_id"] for item in packaged}
    accuracy = shared_measured_accuracy([
        episode for episode in loaded if episode.get("episode_id") in accepted_ids
    ])
    if accepted and csv_path is not None:
        lerobot = export_lerobot_dir(root, csv_path, out / "lerobot", repo_id)
    else:
        lerobot = {"dir": None, "result": "没有通过 QC 的片段，未导出"}

    def _total(key):
        return int(sum(int(item.get(key) or 0) for item in packaged))

    ok_frames = _total("ok_frames")
    tracking_lost = _total("tracking_lost_frames")
    pose_missing = _total("pose_missing_frames")
    report = {
        "package": "unmeasured_pose_episodes",
        "episodes_dir": str(root),
        "pose_source": pose_source,
        "hands": "absent",
        "accuracy": accuracy,
        "ok_frames": ok_frames,
        "tracking_lost_frames": tracking_lost,
        "pose_missing_frames": pose_missing,
        "episodes": len(paths),
        "packaged_episodes": packaged,
        "rejected_episodes": rejected,
        "qc": {
            "yield": qc_report.get("yield"),
            "raw_frames": qc_report.get("raw_frames"),
            "usable_frames": qc_report.get("usable_frames"),
            "episodes": qc_report.get("episodes"),
            "accepted_episodes": qc_report.get("accepted_episodes"),
            "rejected_count": qc_report.get("rejected_episodes"),
            "source": "egodata.stereo_qc.qc_stereo_episode",
        },
        "lerobot": lerobot,
        "limits": [
            "每一份 episode 用自己的 hand_pose.slam.T_world_cam。没有用某一个会话的轨迹覆盖其他片段。",
            "手部没有测量。不把关节或手腕放进 SLAM 世界系。",
            "pose_source=%s。丢失帧的 observation.pose 是 null，不补单位阵。" % pose_source,
            "包级 accuracy 只在每一份被接受的 episode 都有同一份测过的 Umeyama 时写入。各段不同、手写数字、头戴真值来源都是 null。",
            "QC 拒绝的片段留在 rejected_episodes，不写进 LeRobot 包。",
        ],
    }
    _dump(out / "report.json", _jsonable(report))
    if log:
        log(
            "package=pose_only episodes=%d accepted=%d rejected=%d pose_source=%s "
            "ok=%d tracking_lost=%d pose_missing=%d accuracy=%s"
            % (
                len(paths), accepted, len(rejected), pose_source,
                ok_frames, tracking_lost, pose_missing,
                "null" if accuracy is None else "measured",
            )
        )
    return report


def main(argv=None):
    import argparse
    import sys
    parser = argparse.ArgumentParser(description="头戴 MVP：同步 + 位姿状态 + 现有 QC/LeRobot")
    parser.add_argument("--session", default=None)
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
    parser.add_argument("--package-annotated", action="store_true",
                        help="episodes 里每份已经有自己的世界位姿。不再用一个会话的轨迹覆盖它们")
    parser.add_argument("--package-pose-only", action="store_true",
                        help="缺测手、各自已有外部轨迹的 episode 打成一个包。不跑手")
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--run-orbslam3", action="store_true")
    parser.add_argument("--repo-id", default="local/headcam_mvp")
    parser.add_argument("--pose-source", default=None,
                        help="轨迹来源。HOT3D 头戴真值用 hot3d_gt_mocap，不要把它叫成 SLAM")
    parser.add_argument("--gt-tum", default=None, help="另一份真值轨迹，只用来算 ATE/RPE")
    parser.add_argument("--gt-tum-time-scale", type=float, default=None)
    parser.add_argument("--batch", default=None, help="orbslam3 batch json。和 --sequence 一起用")
    parser.add_argument("--sequence", default=None, help="batch 里的序列名。accuracy 只取这一行")
    args = parser.parse_args(argv)
    if args.batch or args.sequence:
        if not args.batch or not args.sequence:
            print("需要同时给出 --batch 和 --sequence", file=sys.stderr)
            return 1
        if args.package_annotated or args.package_pose_only:
            print("--batch 不和打包入口一起用。", file=sys.stderr)
            return 1
        if not args.skip_hands:
            print("这条导出不跑手。手进 SLAM 世界系还停在 MANO 许可上。请加 --skip-hands。", file=sys.stderr)
            return 1
        if not args.session or not args.tum:
            print("需要 --session 和 --tum", file=sys.stderr)
            return 1
        from headcam.batch_export import ESTIMATE_POSE_SOURCE, export_from_batch
        if args.pose_source and args.pose_source != ESTIMATE_POSE_SOURCE:
            print(
                "--pose-source 必须是 %s，不能改成别的来源。" % ESTIMATE_POSE_SOURCE,
                file=sys.stderr,
            )
            return 1
        try:
            report = export_from_batch(
                args.batch, args.sequence, args.session, args.tum, args.out,
                gt_tum_path=args.gt_tum, tum_time_scale=args.tum_time_scale,
                gt_time_scale=args.gt_tum_time_scale, repo_id=args.repo_id,
                run_hands=False,
            )
        except (FileNotFoundError, ValueError, OSError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        lerobot = report.get("lerobot") or {}
        if lerobot.get("error") and not lerobot.get("dependency_missing"):
            print(lerobot["error"], file=sys.stderr)
            return 1
        if not lerobot.get("dir") and not lerobot.get("dependency_missing"):
            print(lerobot.get("result") or "未导出", file=sys.stderr)
            return 1
        return 0
    if args.package_pose_only and args.package_annotated:
        print("--package-pose-only 和 --package-annotated 不能一起用。", file=sys.stderr)
        return 1
    if args.package_pose_only:
        if not args.episodes:
            print("需要 --episodes", file=sys.stderr)
            return 1
        if args.session or args.tum or args.pose_source:
            print(
                "--package-pose-only 不再读取会话或另一条轨迹。不要同时传 --session、--tum、--pose-source。",
                file=sys.stderr,
            )
            return 1
        try:
            report = package_unmeasured_pose_episodes(args.episodes, args.out, repo_id=args.repo_id)
        except (FileNotFoundError, FileExistsError, ValueError, OSError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        lerobot = report.get("lerobot") or {}
        if lerobot.get("error") and not lerobot.get("dependency_missing"):
            print(lerobot["error"], file=sys.stderr)
            return 1
        if not lerobot.get("dir"):
            print(lerobot.get("result") or "未导出", file=sys.stderr)
            return 1
        return 0
    if args.package_annotated:
        if not args.episodes:
            print("需要 --episodes", file=sys.stderr)
            return 1
        if args.session or args.tum or args.pose_source:
            print(
                "--package-annotated 不再读取会话或另一条轨迹。不要同时传 --session、--tum、--pose-source。",
                file=sys.stderr,
            )
            return 1
        try:
            report = package_annotated_episodes(args.episodes, args.out, repo_id=args.repo_id)
        except (FileNotFoundError, ValueError, OSError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        lerobot = report.get("lerobot") or {}
        if lerobot.get("error") and not lerobot.get("dependency_missing"):
            print(lerobot["error"], file=sys.stderr)
            return 1
        if not lerobot.get("dir"):
            print(lerobot.get("result") or "未导出", file=sys.stderr)
            return 1
        return 0
    if not args.session:
        print("需要 --session", file=sys.stderr)
        return 1
    try:
        report = export_mvp(
            args.session, args.out, slam_backend=args.slam_backend, tum_path=args.tum,
            max_gap_s=args.max_gap_s, run_hands=not args.skip_hands, episodes=args.episodes,
            export=not args.no_export, repo_id=args.repo_id,
            run_orbslam3=True if args.run_orbslam3 else None,
            allow_partial=args.allow_partial, tum_time_scale=args.tum_time_scale,
            pose_source=args.pose_source, gt_tum_path=args.gt_tum,
            gt_time_scale=args.gt_tum_time_scale,
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
