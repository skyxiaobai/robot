# -*- coding: utf-8 -*-
"""从已经提交的时间线 JSON 区分两种残差形状。

不读轨迹文件，不跑 ORB-SLAM3，不新算 Umeyama。
时间线里的 rmse 原样抄出。由 1 秒箱的 RMSE 再汇总，只是把已有的箱重新加总，
不是一次新的对齐。短 clip 包的 RMSE 只在 sequence_id 和 clip_ids 都对得上时
才算同一条录制；对不上就不抄。包级 accuracy 不在这里改，输出里的 accuracy 是 null。
"""
import argparse
import json
import math
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

REPORT_KIND = "residual_shape_diagnosis_v1"
TIMELINE_KIND = "trajectory_error_timeline_v1"
ESTIMATE_POSE_SOURCE = "orbslam3_stereo_euroc"
REFERENCE_POSE_SOURCE = "hot3d_gt_mocap"

# 这些门限只用于给已有时间线归类。合成测试锁住门限两侧。
OFFSET_OVER_SPREAD_MIN = 5.0
STEP_OVER_POSITION_MAX = 0.1
OUTSIDE_OVER_TAIL_MAX = 0.2
OPENING_SSE_SHARE_MIN = 0.5
FULLY_INSIDE_OVER_TAIL_MIN = 0.5
WHOLE_WORST_SHARE_MAX = 0.5
WHOLE_STEP_OVER_POSITION_MAX = 0.15
WHOLE_MIN_BIN_OVER_RMSE = 0.5
WHOLE_MAX_BIN_OVER_RMSE = 2.0
SIM3_STILL_THERE = 0.8
RMSE_IDENTITY_ABS_M = 1e-12
HOST_SAME_ABS_M = 1e-6

NOTE = (
    "形状诊断只读已提交的时间线和 batch 行。"
    "rmse 从那些文件原样抄出，没有新的 Umeyama，也没有新的 ATE。"
    "1 秒箱的 pooled_rmse_m 是各箱 n * rmse^2 的加总，不是重新对齐。"
    "accuracy 是 null。短 clip 的 RMSE 在录制对不上时不抄。"
    "归类不是原因。"
)

CASES = (
    {
        "name": "P0002_273c2819",
        "timeline": "P0002_273c2819_timeline.json",
        "record": "P0002_273c2819.json",
        "rerun": "P0002_273c2819_rerun.json",
        "prevhost": "P0002_273c2819_prevhost.json",
    },
    {
        "name": "P0003_b153ade2",
        "timeline": "P0003_b153ade2_timeline.json",
        "record": "P0003_b153ade2.json",
        "rerun": None,
        "prevhost": "P0003_b153ade2_prevhost.json",
    },
)
SHORT_EXPORT = "quest3_four_short_clips_orbslam3_export.json"


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _se3(record):
    accuracy = record.get("accuracy") if isinstance(record, dict) else None
    if not isinstance(accuracy, dict):
        return None
    ate = accuracy.get("ate")
    if not isinstance(ate, dict):
        return None
    se3 = ate.get("se3")
    if not isinstance(se3, dict) or not isinstance(se3.get("rmse_m"), (int, float)):
        return None
    return se3


def _sim3(record):
    accuracy = record.get("accuracy") if isinstance(record, dict) else None
    if not isinstance(accuracy, dict):
        return None
    ate = accuracy.get("ate")
    if not isinstance(ate, dict):
        return None
    sim3 = ate.get("sim3")
    if not isinstance(sim3, dict) or not isinstance(sim3.get("rmse_m"), (int, float)):
        return None
    return sim3


def _keyframes(record):
    rows = record.get("map_keyframes") if isinstance(record, dict) else None
    if not isinstance(rows, list) or len(rows) != 1:
        return None
    value = rows[0].get("keyframes")
    if not isinstance(value, int):
        return None
    return value


def _pooled(bins):
    if not bins:
        return None
    count = 0
    sse = 0.0
    for item in bins:
        n = int(item["n"])
        count += n
        sse += n * float(item["rmse_m"]) ** 2
    if count <= 0:
        return None
    return {"n": count, "pooled_rmse_m": math.sqrt(sse / count), "sse": sse}


def _finite_rmse(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _measurement(timeline, record, rerun):
    """时间线、表行、重跑记录必须是同一份 SE3 RMSE。对不上就拒绝，不挑一个来用。"""
    if not isinstance(timeline, dict):
        return {
            "same_measurement": False,
            "timeline_rmse_m": None,
            "record_rmse_m": None,
            "rerun_rmse_m": None,
            "refuse_reasons": ["timeline_kind"],
        }
    reasons = []
    timeline_rmse = timeline.get("rmse_m")
    if timeline.get("kind") != TIMELINE_KIND:
        reasons.append("timeline_kind")
    if timeline.get("status") != "ok":
        reasons.append("timeline_not_ok")
    if not _finite_rmse(timeline_rmse):
        reasons.append("rmse_missing")
    if timeline.get("estimate_pose_source") != ESTIMATE_POSE_SOURCE:
        reasons.append("estimate_pose_source")
    if timeline.get("reference_pose_source") != REFERENCE_POSE_SOURCE:
        reasons.append("reference_pose_source")
    record_rmse = None
    rerun_rmse = None
    if record is not None:
        if record.get("pose_source") != ESTIMATE_POSE_SOURCE:
            reasons.append("record_pose_source")
        if record.get("reference_pose_source") != REFERENCE_POSE_SOURCE:
            reasons.append("record_reference_pose_source")
        se3 = _se3(record)
        if se3 is None:
            reasons.append("record_rmse_missing")
        else:
            record_rmse = float(se3["rmse_m"])
            if _finite_rmse(timeline_rmse) and abs(record_rmse - float(timeline_rmse)) > RMSE_IDENTITY_ABS_M:
                reasons.append("rmse_mismatch")
            if se3.get("n") != timeline.get("n"):
                reasons.append("n_mismatch")
    if rerun is not None:
        se3 = _se3(rerun)
        if se3 is None:
            reasons.append("rerun_rmse_missing")
        else:
            rerun_rmse = float(se3["rmse_m"])
            if record_rmse is not None and abs(rerun_rmse - record_rmse) > RMSE_IDENTITY_ABS_M:
                reasons.append("rerun_mismatch")
    return {
        "same_measurement": not reasons,
        "timeline_rmse_m": None if not _finite_rmse(timeline_rmse) else float(timeline_rmse),
        "record_rmse_m": record_rmse,
        "rerun_rmse_m": rerun_rmse,
        "refuse_reasons": reasons,
    }


def _opening_assessment(timeline):
    """开头一段固定平移：尾巴从 t=0 开始，误差向量挤在一个均值附近，帧间运动仍小。"""
    reasons = []
    derived = None
    runs = timeline.get("tail_runs")
    geometry = timeline.get("tail_geometry")
    if not isinstance(runs, list) or len(runs) != 1:
        return ["tail_count"], None
    run = runs[0]
    if float(run.get("t0_s", 1.0)) > 1e-9:
        reasons.append("tail_not_prefix")
    if int(run.get("n") or 0) < 3 or not _finite_rmse(run.get("rmse_m")):
        reasons.append("tail_rmse")
    if not isinstance(geometry, list) or len(geometry) != 1:
        reasons.append("tail_geometry")
        geo = None
    else:
        geo = geometry[0]
        if abs(float(geo.get("t0_s", -1.0)) - float(run["t0_s"])) > 1e-9:
            reasons.append("tail_geometry")
        if abs(float(geo.get("t1_s", -1.0)) - float(run["t1_s"])) > 1e-9:
            reasons.append("tail_geometry")
    offset = None if geo is None else geo.get("offset")
    step = None if geo is None else geo.get("step")
    outside = timeline.get("outside_tail") if isinstance(timeline.get("outside_tail"), dict) else {}
    outside_step = timeline.get("outside_tail_step") if isinstance(timeline.get("outside_tail_step"), dict) else None
    tail_rmse = float(run["rmse_m"]) if _finite_rmse(run.get("rmse_m")) else None
    if not isinstance(offset, dict) or not _finite_rmse(offset.get("mean_norm_m")) or not _finite_rmse(offset.get("rms_component_std_m")):
        reasons.append("offset_spread")
        offset_over = None
    else:
        spread = float(offset["rms_component_std_m"])
        if spread <= 0.0:
            reasons.append("offset_spread")
            offset_over = None
        else:
            offset_over = float(offset["mean_norm_m"]) / spread
            if offset_over < OFFSET_OVER_SPREAD_MIN:
                reasons.append("offset_spread")
        if offset.get("n") != run.get("n"):
            reasons.append("offset_n")
    if not isinstance(step, dict) or not _finite_rmse(step.get("trans_rmse_m")) or tail_rmse is None:
        reasons.append("step_not_small")
        step_over = None
    else:
        step_over = float(step["trans_rmse_m"]) / tail_rmse
        if step_over > STEP_OVER_POSITION_MAX:
            reasons.append("step_not_small")
        if step.get("n_steps") != int(run["n"]) - 1:
            reasons.append("step_count")
    if not _finite_rmse(outside.get("rmse_m")) or tail_rmse is None:
        reasons.append("outside_not_small")
        outside_over = None
    else:
        outside_over = float(outside["rmse_m"]) / tail_rmse
        if outside_over > OUTSIDE_OVER_TAIL_MAX:
            reasons.append("outside_not_small")
        if int(outside.get("n") or -1) != int(timeline.get("n") or -2) - int(run["n"]):
            reasons.append("outside_n")
    if isinstance(outside_step, dict) and outside_step.get("n_steps") != int(outside.get("n") or 0) - 1:
        reasons.append("outside_step_count")
    max_error = timeline.get("max_error") if isinstance(timeline.get("max_error"), dict) else {}
    max_t = max_error.get("t_s")
    if not _finite_rmse(max_t) or float(max_t) < float(run["t0_s"]) - 1e-9 or float(max_t) > float(run["t1_s"]) + 1e-9:
        reasons.append("max_error_outside_tail")
    bins = timeline.get("bins_s")
    if not isinstance(bins, list) or sum(int(item["n"]) for item in bins) != int(timeline.get("n") or -1):
        reasons.append("bins_do_not_cover")
        before = after = inside = None
        share = None
        extends_past = None
    else:
        t1 = float(run["t1_s"])
        before_bins = [item for item in bins if float(item["t0_s"]) < t1 - 1e-9]
        after_bins = [item for item in bins if float(item["t0_s"]) >= t1 - 1e-9]
        inside_bins = [item for item in bins if float(item["t1_s"]) <= t1 + 1e-9]
        before = _pooled(before_bins)
        after = _pooled(after_bins)
        inside = _pooled(inside_bins)
        total = _pooled(bins)
        share = None if before is None or total is None or total["sse"] <= 0.0 else before["sse"] / total["sse"]
        extends_past = any(float(item["t1_s"]) > t1 + 1e-9 for item in before_bins)
        if share is None or share < OPENING_SSE_SHARE_MIN:
            reasons.append("opening_sse_share")
        if after is None or after["n"] <= int(run["n"]):
            reasons.append("after_tail_not_longer")
        elif tail_rmse is not None and after["pooled_rmse_m"] > OUTSIDE_OVER_TAIL_MAX * tail_rmse:
            reasons.append("after_tail_not_quiet")
        if inside is not None and tail_rmse is not None and inside["pooled_rmse_m"] < FULLY_INSIDE_OVER_TAIL_MIN * tail_rmse:
            reasons.append("fully_inside_not_loud")
    motion_in = (timeline.get("motion_in_tail") or {}).get("linear_speed_m_s") if isinstance(timeline.get("motion_in_tail"), dict) else None
    motion_out = (timeline.get("motion_outside_tail") or {}).get("linear_speed_m_s") if isinstance(timeline.get("motion_outside_tail"), dict) else None
    in_max = motion_in.get("max") if isinstance(motion_in, dict) else None
    out_max = motion_out.get("max") if isinstance(motion_out, dict) else None
    fastest_outside = None
    if _finite_rmse(in_max) and _finite_rmse(out_max):
        fastest_outside = float(out_max) > float(in_max)
    derived = {
        "offset_over_spread": offset_over,
        "step_over_tail_rmse": step_over,
        "outside_over_tail_rmse": outside_over,
        "bins_starting_before_tail_end": None if before is None else {
            "n": before["n"],
            "pooled_rmse_m": before["pooled_rmse_m"],
            "sse_share": share,
            "includes_bins_that_extend_past_the_tail": extends_past if before is not None else None,
        },
        "bins_starting_at_or_after_tail_end": None if after is None else {
            "n": after["n"],
            "pooled_rmse_m": after["pooled_rmse_m"],
        },
        "bins_ending_at_or_before_tail_end": None if inside is None else {
            "n": inside["n"],
            "pooled_rmse_m": inside["pooled_rmse_m"],
        },
        "fastest_linear_speed_m_s_in_tail": None if not _finite_rmse(in_max) else float(in_max),
        "fastest_linear_speed_m_s_outside_tail": None if not _finite_rmse(out_max) else float(out_max),
        "fastest_linear_speed_is_outside_tail": fastest_outside,
    }
    return reasons, derived


def _whole_assessment(timeline, record):
    """整段残差：没有超过门限的尾巴，每一秒都在同一量级，帧间运动仍小，放开尺度也收不掉。"""
    reasons = []
    runs = timeline.get("tail_runs")
    if not isinstance(runs, list) or len(runs) != 0:
        reasons.append("has_tail")
    if timeline.get("tail_geometry") not in ([], None) and timeline.get("tail_geometry") != []:
        reasons.append("has_tail_geometry")
    outside = timeline.get("outside_tail") if isinstance(timeline.get("outside_tail"), dict) else {}
    rmse = timeline.get("rmse_m")
    if not _finite_rmse(rmse):
        reasons.append("rmse_missing")
        rmse = None
    else:
        rmse = float(rmse)
    if int(outside.get("n") or -1) != int(timeline.get("n") or -2):
        reasons.append("outside_n")
    if rmse is None or not _finite_rmse(outside.get("rmse_m")) or abs(float(outside.get("rmse_m")) - rmse) > RMSE_IDENTITY_ABS_M:
        reasons.append("outside_rmse")
    share = timeline.get("sse_share_worst_10pct") if isinstance(timeline.get("sse_share_worst_10pct"), dict) else {}
    worst = share.get("share")
    if not _finite_rmse(worst) or float(worst) >= WHOLE_WORST_SHARE_MAX:
        reasons.append("worst_share")
    step = timeline.get("outside_tail_step") if isinstance(timeline.get("outside_tail_step"), dict) else {}
    step_over = None
    if rmse is None or not _finite_rmse(step.get("trans_rmse_m")):
        reasons.append("step_large")
    else:
        step_over = float(step["trans_rmse_m"]) / rmse
        if step_over > WHOLE_STEP_OVER_POSITION_MAX:
            reasons.append("step_large")
        if step.get("n_steps") != int(timeline.get("n") or 0) - 1:
            reasons.append("step_count")
    bins = timeline.get("bins_s")
    min_bin = max_bin = min_over = max_over = None
    if not isinstance(bins, list) or len(bins) < 3:
        reasons.append("too_few_bins")
    elif sum(int(item["n"]) for item in bins) != int(timeline.get("n") or -1):
        reasons.append("bins_do_not_cover")
    else:
        values = [float(item["rmse_m"]) for item in bins]
        min_bin = min(values)
        max_bin = max(values)
        if rmse is None or rmse <= 0.0:
            reasons.append("bin_span")
        else:
            min_over = min_bin / rmse
            max_over = max_bin / rmse
            if min_over < WHOLE_MIN_BIN_OVER_RMSE or max_over > WHOLE_MAX_BIN_OVER_RMSE:
                reasons.append("bin_span")
    sim3 = _sim3(record) if record is not None else None
    sim3_over = None
    scale_removes = None
    if record is not None:
        if sim3 is None or rmse is None or rmse <= 0.0:
            reasons.append("sim3_missing")
        else:
            sim3_over = float(sim3["rmse_m"]) / rmse
            scale_removes = sim3_over < SIM3_STILL_THERE
            if scale_removes:
                reasons.append("sim3_reduces_rmse")
    if record is not None and record.get("lost_frames") not in (0, None):
        reasons.append("lost_frames")
    derived = {
        "step_over_position_rmse": step_over,
        "worst_10pct_share": None if not _finite_rmse(worst) else float(worst),
        "min_bin_rmse_m": min_bin,
        "max_bin_rmse_m": max_bin,
        "min_bin_over_rmse": min_over,
        "max_bin_over_rmse": max_over,
        "sim3_over_se3": sim3_over,
        "scale_removes_residual": scale_removes,
    }
    return reasons, derived


def _proven(timeline, record):
    se3 = _se3(record) if record is not None else None
    sim3 = _sim3(record) if record is not None else None
    seams = timeline.get("seams") if isinstance(timeline.get("seams"), dict) else None
    near_max = None if seams is None or not isinstance(seams.get("near"), dict) else seams["near"].get("max_m")
    max_m = timeline.get("max_m")
    return {
        "n": timeline.get("n"),
        "rmse_m": timeline.get("rmse_m"),
        "median_m": timeline.get("median_m"),
        "max_m": max_m,
        "max_error": timeline.get("max_error"),
        "sse_share_worst_10pct": timeline.get("sse_share_worst_10pct"),
        "tail_threshold_m": timeline.get("tail_threshold_m"),
        "tail_runs": timeline.get("tail_runs"),
        "tail_geometry": timeline.get("tail_geometry"),
        "outside_tail": timeline.get("outside_tail"),
        "outside_tail_step": timeline.get("outside_tail_step"),
        "seams": seams,
        "max_error_exceeds_seam_neighborhood": (
            None if not _finite_rmse(near_max) or not _finite_rmse(max_m) else float(near_max) < float(max_m)
        ),
        "tracked_features": timeline.get("tracked_features"),
        "record_se3_rmse_m": None if se3 is None else se3.get("rmse_m"),
        "record_sim3_rmse_m": None if sim3 is None else sim3.get("rmse_m"),
        "record_similarity_scale": None if sim3 is None else sim3.get("similarity_scale"),
        "lost_frames": None if record is None else record.get("lost_frames"),
        "fail_to_track_local_map": None if record is None else record.get("fail_to_track_local_map"),
        "loop_closures": None if record is None else record.get("loop_closures"),
        "reinits": None if record is None else record.get("reinits"),
        "keyframes": None if record is None else _keyframes(record),
    }


def _host_comparison(record, prevhost):
    out = {
        "prevhost_timeline_in_repo": False,
        "shape_on_prevhost": "unknown",
        "prevhost_rmse_used_as_this_run": False,
    }
    if not isinstance(prevhost, dict):
        out["compared"] = False
        return out
    current = _se3(record) if isinstance(record, dict) else None
    previous = _se3(prevhost)
    if current is None or previous is None:
        out["compared"] = False
        out["refuse_reasons"] = ["host_rmse_missing"]
        return out
    delta = abs(float(current["rmse_m"]) - float(previous["rmse_m"]))
    out.update({
        "compared": True,
        "current_se3_rmse_m": current["rmse_m"],
        "prevhost_se3_rmse_m": previous["rmse_m"],
        "abs_delta_m": delta,
        "rmse_within_1e-6_m": delta <= HOST_SAME_ABS_M,
        "current_keyframes": _keyframes(record),
        "prevhost_keyframes": _keyframes(prevhost),
        "current_lost_frames": record.get("lost_frames"),
        "prevhost_lost_frames": prevhost.get("lost_frames"),
        "current_fail_to_track_local_map": record.get("fail_to_track_local_map"),
        "prevhost_fail_to_track_local_map": prevhost.get("fail_to_track_local_map"),
    })
    return out


def _short_clip_relation(record, short_export):
    out = {
        "checked": isinstance(short_export, dict),
        "same_recording": False,
        "overlaps": [],
        "used_as_remeasurement": False,
        "copied_rmse_m": None,
        "observed_package_accuracy": None if not isinstance(short_export, dict) else short_export.get("package_accuracy"),
    }
    if not isinstance(short_export, dict) or not isinstance(record, dict):
        return out
    clips = set(record.get("clip_ids") or [])
    sequence_id = record.get("sequence_id")
    for seq in short_export.get("sequences") or []:
        if not isinstance(seq, dict):
            continue
        shared = sorted(set(seq.get("clip_ids") or []) & clips)
        if seq.get("sequence_id") == sequence_id or shared:
            out["overlaps"].append({
                "name": seq.get("name"),
                "sequence_id": seq.get("sequence_id"),
                "shared_clip_ids": shared,
            })
    out["same_recording"] = bool(out["overlaps"])
    return out


def _unknowns(timeline, host, short_rel, pattern):
    brightness = "unmeasured"
    detail = None
    if isinstance(timeline.get("images"), dict):
        brightness = "aggregate_only"
        detail = timeline.get("images_note") or "时间线只有整段 Pearson，没有逐帧亮度，开头一段没有单独拿掉再算。"
    elif timeline.get("images_note"):
        brightness = "aggregate_only"
        detail = timeline.get("images_note")
    items = [
        {"item": "cause", "status": "unknown", "detail": "形状归类没有给出原因。日志里没有带时间戳的重定位。"},
        {"item": "per_frame_error_vectors", "status": "absent_from_json", "detail": "时间线没有逐帧误差向量，不能把均值平移减掉再算一次 RMSE。"},
        {"item": "relocalization_timestamp", "status": "absent_from_json"},
        {"item": "tracked_features", "status": "null" if timeline.get("tracked_features") is None else "present"},
        {"item": "imu", "status": "not_in_these_records"},
        {"item": "brightness_without_opening", "status": brightness, "detail": detail},
        {"item": "prevhost_shape", "status": host.get("shape_on_prevhost", "unknown")},
        {
            "item": "short_clip_package",
            "status": "same_recording" if short_rel.get("same_recording") else "different_recording",
            "detail": "短 clip 的 RMSE 没有抄进这份诊断。",
        },
        {"item": "hands_in_slam_world", "status": "waiting_on_mano_license"},
        {"item": "new_ate", "status": "not_computed"},
        {"item": "pattern", "status": pattern},
    ]
    return items


def diagnose_case(name, timeline, record=None, prevhost=None, short_export=None, rerun=None, sources=None):
    """给一份时间线归类。accuracy 固定是 null，不接收外部传入的误差数字。"""
    measurement = _measurement(timeline if isinstance(timeline, dict) else {}, record, rerun)
    host = _host_comparison(record, prevhost)
    short_rel = _short_clip_relation(record, short_export)
    opening_reasons, opening_derived = ([], None)
    whole_reasons, whole_derived = ([], None)
    if isinstance(timeline, dict) and measurement["same_measurement"]:
        opening_reasons, opening_derived = _opening_assessment(timeline)
        whole_reasons, whole_derived = _whole_assessment(timeline, record)
        if record is not None and record.get("lost_frames") not in (0, None):
            if "lost_frames" not in opening_reasons:
                opening_reasons = list(opening_reasons) + ["lost_frames"]
            if "lost_frames" not in whole_reasons:
                whole_reasons = list(whole_reasons) + ["lost_frames"]
    pattern = "refused"
    derived = None
    proven = None
    if measurement["same_measurement"]:
        proven = _proven(timeline, record)
        if not opening_reasons:
            pattern = "opening_fixed_translation"
            derived = opening_derived
        elif not whole_reasons:
            pattern = "whole_sequence_residual"
            derived = whole_derived
        else:
            pattern = "unclassified"
            derived = {"opening": opening_derived, "whole_sequence": whole_derived}
    return {
        "name": name,
        "pattern": pattern,
        "accuracy": None,
        "new_ate": False,
        "measurement": measurement,
        "proven": proven,
        "derived": derived,
        "opening_reasons": opening_reasons,
        "whole_reasons": whole_reasons,
        "host_comparison": host,
        "short_clip_package": short_rel,
        "unknown": _unknowns(timeline if isinstance(timeline, dict) else {}, host, short_rel, pattern),
        "sources": sources or {},
    }


def diagnose_repo(docs_dir):
    """读 docs/headcam/orbslam3_quest3 里已经提交的两份时间线。"""
    root = Path(docs_dir)
    short_path = root / SHORT_EXPORT
    short_export = _load(short_path) if short_path.is_file() else None
    cases = []
    for spec in CASES:
        timeline_path = root / spec["timeline"]
        record_path = root / spec["record"]
        rerun_path = root / spec["rerun"] if spec["rerun"] else None
        prevhost_path = root / spec["prevhost"] if spec["prevhost"] else None
        cases.append(diagnose_case(
            spec["name"],
            _load(timeline_path),
            record=_load(record_path),
            prevhost=_load(prevhost_path) if prevhost_path is not None else None,
            short_export=short_export,
            rerun=_load(rerun_path) if rerun_path is not None else None,
            sources={
                "timeline": spec["timeline"],
                "record": spec["record"],
                "rerun": spec["rerun"],
                "prevhost": spec["prevhost"],
                "short_clip_export": SHORT_EXPORT if short_export is not None else None,
            },
        ))
    return {
        "kind": REPORT_KIND,
        "accuracy": None,
        "new_ate": False,
        "package_accuracy_rule": "null_when_episode_measurements_differ",
        "note": NOTE,
        "cases": cases,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="从已提交的时间线 JSON 归类 P0002 开头平移和 P0003 整段残差。")
    parser.add_argument("--docs", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        report = diagnose_repo(args.docs)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for case in report["cases"]:
        print(case["name"], case["pattern"], "accuracy", case["accuracy"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
