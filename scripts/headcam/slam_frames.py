# -*- coding: utf-8 -*-
"""把相机系点变到世界系：只乘 status=ok 的 T_world_cam。

``p_world = T_world_cam @ p_cam``（与 ``hand_pose.transform_points`` 相同）。
``tracking_lost`` 和 ``pose_missing`` 的世界坐标是 null，不抄相邻帧。

坐标名用 ``common.frames``：完整轨迹标 ``world``（公制 SLAM，上轴还没核实），
不是 ``world_yup``。覆盖不全时 episode 仍是 ``camera``，世界点只放在
``hand_pose.slam.world_hands``，不改手腕主数组，避免成绩单把缺测帧当成世界坐标。

不写精度。没有逐帧激光深度时不编 ``iphone.per_frame``。

帧数相同还不够。episode 和 slam 帧都有时间戳时，同一序号必须落在半帧门限内，
秒和纳秒对不上就拒绝，不按序号把两套时钟配成同一帧，也不自动换算。
缺一边的时间戳时仍按帧数对齐，并在 ``timestamp_alignment`` 里写明。
"""
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import numpy as np

from common.frames import CAMERA, CAMERA_YUP, WORLD, WORLD_YUP, tag
from egodata.schema import rotmat_to_quat_xyzw
from headcam.hand_pose import quat_xyzw_to_rotmat, transform_points, xyz_to_json
from headcam.slam_vio import STATUS_OK, half_frame_seconds
from headcam.time_units import UNIT_NANOSECONDS, UNIT_SECONDS, clock_unit

_SIDES = ("left", "right")


def _matrix(value):
    if value is None:
        return None
    pose = np.asarray(value, dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        return None
    return pose


def _ok_pose(frame):
    if not isinstance(frame, dict) or frame.get("status") != STATUS_OK:
        return None
    return _matrix(frame.get("T_world_cam"))


def _blank_joints():
    return [[None, None, None] for _ in range(21)]


def _joints_world(joints, pose):
    if not isinstance(joints, list) or len(joints) != 21:
        return None
    raw = []
    any_finite = False
    for point in joints:
        if not isinstance(point, (list, tuple)) or len(point) < 3 or any(coord is None for coord in point[:3]):
            raw.append([np.nan, np.nan, np.nan])
            continue
        coords = [float(coord) for coord in point[:3]]
        if not np.isfinite(coords).all():
            raw.append([np.nan, np.nan, np.nan])
            continue
        raw.append(coords)
        any_finite = True
    if not any_finite:
        return None
    return xyz_to_json(transform_points(raw, pose))


def _wrist_world(wrist, pose):
    if not isinstance(wrist, (list, tuple)) or len(wrist) < 3 or any(coord is None for coord in wrist[:3]):
        return None
    position = np.asarray(wrist[:3], dtype=float)
    if not np.isfinite(position).all():
        return None
    world = transform_points(position, pose)
    out = [float(world[0]), float(world[1]), float(world[2])]
    quat = list(wrist[3:7]) if len(wrist) >= 7 else []
    if len(quat) == 4 and all(value is not None and np.isfinite(value) for value in quat):
        rotation = pose[:3, :3] @ quat_xyzw_to_rotmat(quat)
        out.extend(float(value) for value in rotmat_to_quat_xyzw(rotation))
    else:
        out.extend([None, None, None, None])
    return out


def _per_frame_lists(episode, num_frames):
    stereo = episode.get("stereo") if isinstance(episode.get("stereo"), dict) else {}
    track = stereo.get("per_frame") if isinstance(stereo.get("per_frame"), dict) else None
    track_status = None
    if track is not None:
        copied = {}
        for side in _SIDES:
            values = track.get(side)
            if isinstance(values, list) and len(values) == num_frames:
                copied[side] = list(values)
        if copied:
            track_status = copied
    depth_status = None
    iphone = episode.get("iphone") if isinstance(episode.get("iphone"), dict) else {}
    per_frame = iphone.get("per_frame") if isinstance(iphone.get("per_frame"), dict) else None
    if per_frame is not None:
        copied = {}
        for side in _SIDES:
            values = per_frame.get(side)
            if isinstance(values, list) and len(values) == num_frames:
                copied[side] = list(values)
        if copied:
            depth_status = copied
    return track_status, depth_status


def _tag_camera(episode):
    source = episode.get("coordinate_frame") or CAMERA
    if episode.get("coordinate_frame") not in (CAMERA, CAMERA_YUP):
        episode["coordinate_frame"] = CAMERA
    tag(episode, CAMERA, gravity_aligned=False, source_frame=source)
    episode["frame_convention"]["camera_axes"] = "cv"
    episode["frame_convention"]["up_axis_verified"] = False


def _tag_world(episode):
    source = episode.get("coordinate_frame") or "slam_world"
    episode["coordinate_frame"] = "slam_world"
    tag(episode, WORLD, gravity_aligned=False, source_frame=source)
    episode["frame_convention"]["camera_axes"] = "cv"
    episode["frame_convention"]["up_axis_verified"] = False


def _timestamp_alignment(episode, frames):
    """两边都有时间戳时核对序号。对不上返回 ``refused``，不重排。"""
    stamps = episode.get("timestamps")
    if not isinstance(stamps, list) or len(stamps) != len(frames):
        return "frame_count", (
            "episode 没有和帧数一样长的 timestamps。帧数相同就按序号对齐。"
            "这不会发现错位，也不会发现两套时钟。"
        )
    slam_times = []
    for frame in frames:
        if not isinstance(frame, dict) or frame.get("timestamp_s") is None:
            return "frame_count", (
                "slam 帧没有 timestamp_s。帧数相同就按序号对齐。"
                "两边时间戳都在、但对不上时会拒绝，不按序号硬配。"
            )
        try:
            slam_times.append(float(frame["timestamp_s"]))
        except (TypeError, ValueError):
            return "refused", "slam 帧的 timestamp_s 不是数，拒绝按序号配对。"
    try:
        episode_times = [float(value) for value in stamps]
    except (TypeError, ValueError):
        return "refused", "episode timestamps 不是数，拒绝按序号配对。"
    left = clock_unit(episode_times)
    right = clock_unit(slam_times)
    known = (UNIT_SECONDS, UNIT_NANOSECONDS)
    if left in known and right in known and left != right:
        return "refused", (
            "episode 时间单位是 %s，slam 帧时间单位是 %s。"
            "不按帧数硬配，也不自动换算。"
            % (left, right)
        )
    limit = half_frame_seconds(episode_times, episode.get("fps"))
    gap = max(abs(item - other) for item, other in zip(episode_times, slam_times))
    if gap > limit:
        return "refused", (
            "同一序号的时间最大差 %.6g 秒，超过半帧门限 %.6g 秒。"
            "不按序号把不同时刻配成同一帧。"
            % (gap, limit)
        )
    return "timestamps", "逐帧时间戳落在半帧门限内，按这个顺序使用 T_world_cam。"


def apply_ok_frame_world(episode, slam):
    """按逐帧状态写世界点。返回 episode 上的 ``hand_pose.slam`` 摘要。

    已经是世界系的 episode 不再乘一次 T。``camera_yup`` 不是 OpenCV 光学系，
    这里不变换。覆盖不全时不把主手腕改成世界坐标。
    """
    if not isinstance(episode, dict):
        raise ValueError("episode 必须是对象")
    hand_pose = episode.setdefault("hand_pose", {})
    if not isinstance(hand_pose, dict):
        hand_pose = {}
        episode["hand_pose"] = hand_pose
    summary = hand_pose.get("slam")
    if not isinstance(summary, dict):
        summary = {}
        hand_pose["slam"] = summary
    frames = slam.get("frames") if isinstance(slam, dict) else None
    num_frames = int(episode.get("num_frames") or 0)
    aligned = isinstance(frames, list) and num_frames > 0 and len(frames) == num_frames
    summary["pose_status_aligned"] = bool(aligned)
    summary["accuracy"] = None
    track_status, depth_status = _per_frame_lists(episode, num_frames) if num_frames else (None, None)
    summary["track_status"] = track_status
    summary["depth_status"] = depth_status
    if depth_status is None:
        summary["depth_status_note"] = (
            "没有逐帧深度证据。不写 iphone.per_frame，也不把缺测记成深度通过。"
        )
    if not aligned:
        summary["timestamp_alignment"] = "frame_count"
        summary["timestamp_alignment_note"] = "帧数不同，没有按序号配对，也没有按时间重排。"
        if summary.get("world_points_applied"):
            return summary
        summary["world_points_scope"] = "none"
        summary["T_world_cam"] = None
        return summary
    alignment, alignment_note = _timestamp_alignment(episode, frames)
    summary["timestamp_alignment"] = alignment
    summary["timestamp_alignment_note"] = alignment_note
    if alignment == "refused":
        summary["pose_status_aligned"] = False
        summary["world_points_scope"] = "none"
        summary["T_world_cam"] = None
        summary["world_points_note"] = alignment_note
        return summary
    if summary.get("world_points_applied"):
        return summary

    statuses = []
    poses = []
    for frame in frames:
        pose = _ok_pose(frame)
        statuses.append(frame.get("status") if isinstance(frame, dict) else None)
        poses.append(None if pose is None else pose.tolist())
    summary["pose_status"] = statuses
    summary["T_world_cam"] = poses

    convention = _stored_convention(episode)
    if convention == CAMERA_YUP:
        summary["world_points_scope"] = "none"
        summary["world_points_note"] = "episode 是 camera_yup，T_world_cam 按 OpenCV 光学系定义，这里不变换。"
        _ensure_tag(episode, CAMERA_YUP)
        return summary
    if convention in (WORLD, WORLD_YUP):
        summary["world_points_scope"] = "already_world"
        summary["world_points_applied"] = False
        summary["world_points_note"] = "episode 已经是世界系，不再乘 T_world_cam。"
        if convention == WORLD and not isinstance(episode.get("frame_convention"), dict):
            _tag_world(episode)
        return summary

    complete = all(pose is not None for pose in poses)
    world_hands = {side: [] for side in _SIDES}
    for index, pose_list in enumerate(poses):
        pose = None if pose_list is None else _matrix(pose_list)
        for side in _SIDES:
            hand = (episode.get("hands") or {}).get(side) or {}
            joints = hand.get("joints")
            wrists = hand.get("wrist_pose")
            joint_frame = joints[index] if isinstance(joints, list) and index < len(joints) else None
            wrist_frame = wrists[index] if isinstance(wrists, list) and index < len(wrists) else None
            if pose is None:
                world_hands[side].append(None)
                continue
            packed = {
                "joints": _joints_world(joint_frame, pose),
                "wrist": _wrist_world(wrist_frame, pose),
            }
            if packed["joints"] is None and packed["wrist"] is None:
                world_hands[side].append(None)
            else:
                world_hands[side].append(packed)
    summary["world_hands"] = world_hands
    any_ok = any(pose is not None for pose in poses)
    if not any_ok:
        summary["world_points_scope"] = "none"
        summary["world_points_applied"] = False
        summary["camera_pose_meaning"] = "identity_placeholder_camera_frame"
        summary["world_points_note"] = "没有 ok 位姿。世界点全是 null，手腕仍在相机系。"
        _tag_camera(episode)
        return summary
    if not complete:
        summary["world_points_scope"] = "ok_frames_only"
        summary["world_points_applied"] = False
        summary["camera_pose_meaning"] = "ok_frames_only"
        summary["world_points_note"] = (
            "只有 status=ok 的帧写了 p_world = T_world_cam @ p_cam。"
            "tracking_lost 和 pose_missing 是 null，没有用相邻位姿填。"
            "覆盖不全，手腕主数组仍在相机系。"
        )
        _tag_camera(episode)
        return summary

    for side in _SIDES:
        hand = episode["hands"][side]
        for index, packed in enumerate(world_hands[side]):
            if packed is None:
                hand["wrist_pose"][index] = [None] * 7
                if isinstance(hand.get("joints"), list):
                    hand["joints"][index] = _blank_joints()
                continue
            if packed["wrist"] is not None:
                hand["wrist_pose"][index] = packed["wrist"]
            if packed["joints"] is not None and isinstance(hand.get("joints"), list):
                hand["joints"][index] = packed["joints"]
    episode["camera_poses"] = [pose for pose in poses]
    summary["world_points_scope"] = "all_frames"
    summary["world_points_applied"] = True
    summary["camera_pose_meaning"] = "T_world_cam"
    summary["world_points_note"] = (
        "每一帧都是 ok，手腕和关节已用 p_world = T_world_cam @ p_cam 写入。"
        "frame_convention 是 world：上轴未核实，不是 world_yup。accuracy 仍是 null。"
    )
    _tag_world(episode)
    return summary


def _stored_convention(episode):
    block = episode.get("frame_convention")
    if isinstance(block, dict) and block.get("name") in (CAMERA, CAMERA_YUP, WORLD, WORLD_YUP):
        return block["name"]
    raw = episode.get("coordinate_frame")
    if raw in (None, CAMERA):
        return CAMERA
    if raw == CAMERA_YUP:
        return CAMERA_YUP
    if raw in ("slam_world", WORLD):
        return WORLD
    if raw in ("world_yup", "arkit_world_yup", "arkit_world"):
        return WORLD_YUP
    return CAMERA


def _ensure_tag(episode, convention):
    block = episode.get("frame_convention")
    if isinstance(block, dict) and block.get("name") == convention:
        block.setdefault("camera_axes", "cv")
        block.setdefault("up_axis_verified", False)
        return
    source = episode.get("coordinate_frame") or convention
    tag(episode, convention, gravity_aligned=convention == WORLD_YUP, source_frame=source)
    episode["frame_convention"]["camera_axes"] = "cv"
    episode["frame_convention"]["up_axis_verified"] = False


def _world_fill_counts(summary):
    filled = missing = 0
    hands = summary.get("world_hands") if isinstance(summary, dict) else None
    left = hands.get("left") if isinstance(hands, dict) else None
    if not isinstance(left, list):
        return filled, missing
    for item in left:
        if item is None:
            missing += 1
        else:
            filled += 1
    return filled, missing


def check_camera_episode(episode, slam):
    """对一份相机系 episode 做 ok 帧拼接，并跑 ``frames.sanity_check``。

    返回 ``(text, exit_code, episode)``。``accuracy`` 不从 slam 状态里抄，
    打印行固定是 ``accuracy=unmeasured``。覆盖不全时退出码仍看轴向检查，
    不看有没有世界点。
    """
    from common.frames import convention_of, sanity_check

    if not isinstance(episode, dict):
        raise ValueError("episode 必须是对象")
    apply_ok_frame_world(episode, slam)
    sanity = sanity_check(episode)
    block = episode.get("hand_pose", {}).get("slam") or {}
    statuses = block.get("pose_status") or []
    if not isinstance(statuses, list):
        statuses = []
    filled, missing = _world_fill_counts(block)
    lines = [
        "frames=%d" % len(statuses),
        "ok=%d" % statuses.count(STATUS_OK),
        "tracking_lost=%d" % statuses.count("tracking_lost"),
        "pose_missing=%d" % statuses.count("pose_missing"),
        "world_points_scope=%s" % block.get("world_points_scope"),
        "frame_convention=%s" % convention_of(episode),
        "world_hands_filled=%d" % filled,
        "world_hands_missing=%d" % missing,
        "sanity_ok=%s" % str(bool(sanity.get("ok"))).lower(),
        "timestamp_alignment=%s" % block.get("timestamp_alignment"),
        "accuracy=unmeasured",
    ]
    flags = sanity.get("flags") or []
    if flags:
        lines.append("sanity_flags=%s" % "; ".join(str(flag) for flag in flags))
    refused = block.get("timestamp_alignment") == "refused"
    return "\n".join(lines) + "\n", (0 if sanity.get("ok") and not refused else 1), episode


def _load_json(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("需要 JSON 对象：%s" % path)
    return data


def _slam_from_args(args):
    if args.slam_status:
        slam = _load_json(args.slam_status)
        if not isinstance(slam.get("frames"), list):
            raise ValueError("slam_status 缺少 frames：%s" % args.slam_status)
        return slam
    if not args.session:
        raise ValueError("需要 --slam-status，或 --session 加上可选的 --tum")
    from headcam.slam_vio import run_slam
    backend = args.backend
    if args.tum and backend == "auto":
        backend = "external_tum"
    return run_slam(
        args.session, backend=backend, tum_path=args.tum, tum_time_scale=args.tum_time_scale,
    )


def main(argv=None):
    """检查一份已有的相机系 episode。不跑 WiLoR，不写精度。"""
    import argparse

    parser = argparse.ArgumentParser(
        description="相机系 episode 只在 ok 帧乘 T_world_cam，再用 frames.sanity_check。不写精度。",
    )
    parser.add_argument("--episode", required=True, help="WiLoR 或其他模块写出的相机系 episode JSON")
    parser.add_argument("--slam-status", default=None, help="slam_vio 写出的 slam_status.json")
    parser.add_argument("--session", default=None, help="没有 slam_status 时，用会话目录现算逐帧状态")
    parser.add_argument("--tum", default=None)
    parser.add_argument("--tum-time-scale", type=float, default=None)
    parser.add_argument("--backend", default="auto", choices=("auto", "none", "external_tum", "orbslam3"))
    parser.add_argument("--out", default=None, help="写出检查后的 episode。再跑一次不会把世界点乘第二次")
    args = parser.parse_args(argv)
    try:
        episode = _load_json(args.episode)
        slam = _slam_from_args(args)
        text, code, checked = check_camera_episode(episode, slam)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(checked, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(text)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
