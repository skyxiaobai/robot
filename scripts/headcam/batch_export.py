# -*- coding: utf-8 -*-
"""把 batch 里已经测过的 ORB-SLAM3 轨迹接到缺测手的 LeRobot 导出。

位姿只来自外部 TUM。``accuracy`` 只接受两种来源：batch 行里的
``trajectory_error_v1``，或这次用同一份 TUM 和真值重新测到的同一类结构。
汇总里的 min / median / max 不写进包。手写数字不写进包。

全是单位阵的轨迹、和头戴真值逐帧相同的轨迹、文件头不是
``orbslam3_stereo_euroc`` 的轨迹，都拒绝，不写出包。
clip 集合和 batch 行不一致时也不套用那一行的 accuracy。

不跑手部。手进 SLAM 世界系还停在 MANO 许可上。
"""
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import numpy as np

from headcam.hand_pose import load_tum_trajectory
from headcam.hot3d_check import (
    _load_scaled_tum,
    _unique_trajectory,
    is_measured_accuracy,
    trajectory_accuracy,
)
from headcam.slam_vio import read_pose_source, read_timestamps

ESTIMATE_POSE_SOURCE = "orbslam3_stereo_euroc"
REFERENCE_POSE_SOURCE = "hot3d_gt_mocap"
RMSE_MATCH_ATOL_M = 1e-6
ASSOCIATION_MAX_DT_S = 0.05


def load_batch(path):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("sequences"), list):
        raise ValueError("batch 文件没有 sequences：%s" % path)
    return document


def find_sequence(batch, name):
    """按名字找一行。名字唯一时用那一行，即使别的行共用 ``sequence_id``。

    没有这个名字、只靠 ``sequence_id`` 对上多行时不猜测，返回前抛 ``ValueError``。
    一行都没有则返回 None。
    """
    named = []
    identified = []
    for record in batch.get("sequences") or []:
        if not isinstance(record, dict):
            continue
        if record.get("name") == name:
            named.append(record)
        if record.get("sequence_id") == name or record.get("name") == name:
            identified.append(record)
    if len(named) > 1:
        raise ValueError("batch 里名字 %s 有 %d 条，不猜测用哪一条。" % (name, len(named)))
    if len(named) == 1:
        return named[0]
    if not identified:
        return None
    if len(identified) > 1:
        raise ValueError("batch 里 %s 有 %d 条，不猜测用哪一条。" % (name, len(identified)))
    return identified[0]


def match_sequence(batch, sequence_id, clip_ids):
    """clip 集合必须和某一行完全一致。只同名、clip 对不上，返回 None。"""
    wanted = [int(value) for value in clip_ids]
    found = None
    for record in batch.get("sequences") or []:
        if not isinstance(record, dict):
            continue
        if record.get("sequence_id") != sequence_id and record.get("name") != sequence_id:
            continue
        clips = record.get("clip_ids")
        if not isinstance(clips, list) or [int(value) for value in clips] != wanted:
            continue
        if found is not None:
            raise ValueError("batch 里 %s clips %s 有多条。" % (sequence_id, wanted))
        found = record
    return found


def poses_all_identity(poses, atol=1e-8):
    """每一帧都是单位阵才算。第一帧在原点、后面在动，不算。"""
    if not poses:
        return False
    eye = np.eye(4)
    for pose in poses:
        matrix = np.asarray(pose, dtype=float)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            return False
        if not np.allclose(matrix, eye, atol=atol):
            return False
    return True


def measured_accuracy_from_record(record):
    """返回 ``(accuracy or None, note)``。不读汇总的 min / median / max。

    SE3 的 n 和 ``tracked_frames`` 对不上时抛 ``ValueError``，避免把另一段的
    误差套到这一段上。
    """
    if not isinstance(record, dict):
        return None, "batch 行不是对象。accuracy 保持 null。"
    source = record.get("pose_source")
    if source != ESTIMATE_POSE_SOURCE:
        return None, (
            "batch 行的 pose_source=%s，不是 %s。accuracy 保持 null。"
            % (source, ESTIMATE_POSE_SOURCE)
        )
    accuracy = record.get("accuracy")
    label = record.get("name") or record.get("sequence_id")
    if record.get("status") != "ok" or accuracy is None:
        failure = record.get("failure")
        if failure:
            return None, "batch 里 %s 没有测到 ATE（%s）。accuracy 保持 null。" % (label, failure)
        return None, "batch 里 %s 没有测到 ATE。accuracy 保持 null。" % label
    if not is_measured_accuracy(accuracy, source):
        return None, (
            "batch 里 %s 的 accuracy 不是 trajectory_error_v1。手写数字不抄。accuracy 保持 null。"
            % label
        )
    reference = accuracy.get("reference_pose_source")
    if reference != REFERENCE_POSE_SOURCE:
        return None, (
            "batch 里 %s 的 reference_pose_source=%s，不是 %s。accuracy 保持 null。"
            % (label, reference, REFERENCE_POSE_SOURCE)
        )
    tracked = record.get("tracked_frames")
    se3_n = int(accuracy["ate"]["se3"]["n"])
    if tracked is not None and se3_n != int(tracked):
        raise ValueError(
            "batch 里 %s 的 SE3 n=%s，tracked_frames=%s。两份对不上，不导出。"
            % (label, se3_n, tracked)
        )
    return accuracy, "accuracy 来自 batch 行 %s 的 trajectory_error_v1，没有改数字。" % label


def _scaled_unique(path, scale):
    times, poses = _load_scaled_tum(path, scale)
    unique_t, unique_p, _duplicates = _unique_trajectory(times, poses)
    raw_count = int(len(times))
    return unique_t, unique_p, raw_count


def _nearest_gap(stamp, times):
    if len(times) == 0:
        return None
    return float(np.min(np.abs(np.asarray(times, dtype=float) - float(stamp))))


def _same_trajectory(est_times, est_poses, gt_times, gt_poses):
    if len(est_poses) == 0 or len(est_poses) != len(gt_poses):
        return False
    for est_t, gt_t, est_pose, gt_pose in zip(est_times, gt_times, est_poses, gt_poses):
        if abs(float(est_t) - float(gt_t)) > 1e-6:
            return False
        if not np.allclose(est_pose, gt_pose, atol=1e-6):
            return False
    return True


def _accuracies_agree(batch_accuracy, fresh):
    if not is_measured_accuracy(batch_accuracy, ESTIMATE_POSE_SOURCE):
        return False
    if not is_measured_accuracy(fresh, ESTIMATE_POSE_SOURCE):
        return False
    batch_se3 = batch_accuracy["ate"]["se3"]
    fresh_se3 = fresh["ate"]["se3"]
    if int(batch_se3["n"]) != int(fresh_se3["n"]):
        return False
    return abs(float(batch_se3["rmse_m"]) - float(fresh_se3["rmse_m"])) <= RMSE_MATCH_ATOL_M


def assert_trajectory_usable(tum_path, record, session, time_scale):
    """拒绝单位阵、真值冒充、来源不对、帧数或时间对不上。通过则不返回。"""
    tum_path = Path(tum_path)
    if not tum_path.is_file():
        raise FileNotFoundError("没有估计轨迹：%s" % tum_path)
    header = read_pose_source(tum_path)
    if header != ESTIMATE_POSE_SOURCE:
        raise ValueError(
            "估计轨迹的 pose_source=%s，不是 %s。不把别的轨迹标成 ORB-SLAM3。"
            % (header, ESTIMATE_POSE_SOURCE)
        )
    if time_scale is None:
        raise ValueError("没有轨迹时间尺度。纳秒轨迹要 1e-9，不自动猜。")
    _raw_times, raw_poses = load_tum_trajectory(tum_path)
    if poses_all_identity(raw_poses):
        raise ValueError("估计轨迹每一帧都是单位阵，不能当成 ORB-SLAM3 世界位姿，也不写出包。")
    unique_t, _unique_p, raw_count = _scaled_unique(tum_path, time_scale)
    label = record.get("name") or record.get("sequence_id")
    tracked = record.get("tracked_frames")
    if tracked is not None and int(len(unique_t)) != int(tracked):
        raise ValueError(
            "估计轨迹不同时间戳 %d 个，batch 里 %s 的 tracked_frames 是 %s。不把这条 accuracy 套到另一份轨迹上。"
            % (len(unique_t), label, tracked)
        )
    rows = record.get("trajectory_rows")
    if rows is not None and raw_count != int(rows):
        raise ValueError(
            "估计轨迹 %d 行，batch 里 %s 的 trajectory_rows 是 %s。不套用这一行。"
            % (raw_count, label, rows)
        )
    stamps = read_timestamps(session)
    input_frames = record.get("input_frames")
    if input_frames is not None and len(stamps) != int(input_frames):
        raise ValueError(
            "会话 %d 帧，batch 里 %s 的 input_frames 是 %s。clip 范围对不上，不导出。"
            % (len(stamps), label, input_frames)
        )
    matched = 0
    for stamp in stamps:
        gap = _nearest_gap(stamp, unique_t)
        if gap is not None and gap <= ASSOCIATION_MAX_DT_S:
            matched += 1
    if tracked is not None and matched != int(tracked):
        raise ValueError(
            "会话里有 %d 帧对上估计轨迹，batch 里 %s 的 tracked_frames 是 %s。时间对不上，不导出。"
            % (matched, label, tracked)
        )
    return unique_t


def resolve_published_accuracy(record, tum_path, gt_path, estimate_scale, gt_scale):
    """决定写进包的 accuracy。对不上时返回 None，并说明原因。"""
    batch_accuracy, batch_note = measured_accuracy_from_record(record)
    if gt_path is None:
        source = "batch" if batch_accuracy is not None else "unmeasured"
        return batch_accuracy, source, batch_note
    gt_path = Path(gt_path)
    if not gt_path.is_file():
        raise FileNotFoundError("没有真值轨迹：%s" % gt_path)
    reference = read_pose_source(gt_path)
    if reference != REFERENCE_POSE_SOURCE:
        return None, "unmeasured", (
            "真值文件头的 pose_source=%s，不是 %s。不猜测来源。accuracy 保持 null。"
            % (reference, REFERENCE_POSE_SOURCE)
        )
    est_times, est_poses, _raw = _scaled_unique(tum_path, estimate_scale)
    gt_times, gt_poses, _gt_raw = _scaled_unique(gt_path, gt_scale)
    if _same_trajectory(est_times, est_poses, gt_times, gt_poses):
        raise ValueError("估计轨迹和头戴真值逐帧相同，不把真值当成 ORB-SLAM3，也不写出包。")
    fresh = trajectory_accuracy(
        tum_path, gt_path,
        estimate_scale=estimate_scale, gt_scale=gt_scale,
        estimate_pose_source=ESTIMATE_POSE_SOURCE,
        reference_pose_source=reference,
    )
    fresh_ok = is_measured_accuracy(fresh, ESTIMATE_POSE_SOURCE)
    if batch_accuracy is not None and fresh_ok and _accuracies_agree(batch_accuracy, fresh):
        return batch_accuracy, "batch", (
            batch_note + " 这次用真值重算的 SE3 RMSE 与之相同。"
        )
    if batch_accuracy is not None and fresh_ok and not _accuracies_agree(batch_accuracy, fresh):
        return None, "unmeasured", (
            "batch 的 SE3 RMSE 是 %s，这次测到 %s。两份不一致，accuracy 保持 null。"
            % (batch_accuracy["ate"]["se3"]["rmse_m"], fresh["ate"]["se3"]["rmse_m"])
        )
    if batch_accuracy is not None and not fresh_ok:
        return None, "unmeasured", (
            "batch 里有 ATE，但这份真值对不上 3 个时间戳。accuracy 保持 null。"
        )
    if batch_accuracy is None and fresh_ok:
        return fresh, "measured_this_run", (
            "batch 里这条 accuracy 是 null。这次用真值测到了 Umeyama，写入的是这次的结果。"
        )
    return None, "unmeasured", batch_note


def export_from_batch(
    batch_path, sequence, session, tum_path, out,
    gt_tum_path=None, tum_time_scale=None, gt_time_scale=None,
    repo_id="local/headcam_mvp", run_hands=False, log=print,
):
    """用 batch 里的一行和外部 TUM 导出缺测手的包。

    校验失败时抛 ``ValueError`` 或 ``FileNotFoundError``，不写 ``out``。
    """
    if run_hands:
        raise ValueError("手部进 SLAM 世界系还停着，这条导出不跑手。")
    batch = load_batch(batch_path)
    record = find_sequence(batch, sequence)
    if record is None:
        raise ValueError("batch 里没有序列 %s。不编造 accuracy。" % sequence)
    estimate_scale = record.get("estimate_time_scale") if tum_time_scale is None else tum_time_scale
    if estimate_scale is None:
        raise ValueError("batch 行没有 estimate_time_scale，调用方也没有给。不自动猜。")
    estimate_scale = float(estimate_scale)
    assert_trajectory_usable(tum_path, record, session, estimate_scale)
    gt_scale = 1.0
    if gt_tum_path is not None:
        gt_scale = record.get("gt_time_scale") if gt_time_scale is None else gt_time_scale
        if gt_scale is None:
            raise ValueError("batch 行没有 gt_time_scale，调用方也没有给。不自动猜。")
        gt_scale = float(gt_scale)
    accuracy, source, note = resolve_published_accuracy(
        record, tum_path, gt_tum_path, estimate_scale, gt_scale,
    )
    tracked = record.get("tracked_frames")
    input_frames = record.get("input_frames")
    if tracked is not None and input_frames is not None and int(tracked) < int(input_frames):
        note += " 跟踪 %s/%s 帧，丢失帧不补单位阵。" % (tracked, input_frames)
    from headcam.export_mvp import export_mvp
    report = export_mvp(
        session, out,
        slam_backend="external_tum",
        tum_path=tum_path,
        tum_time_scale=estimate_scale,
        pose_source=ESTIMATE_POSE_SOURCE,
        run_hands=False,
        repo_id=repo_id,
        accuracy_publication={
            "value": accuracy,
            "note": note,
            "source": source,
            "sequence": record.get("name") or record.get("sequence_id"),
        },
        log=log,
    )
    return report
