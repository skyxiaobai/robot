# -*- coding: utf-8 -*-
"""把多条 ORB-SLAM3 轨迹收成一张表。

失败的序列 accuracy 是 null，仍留在表里并计入序列数。
RMSE 汇总只使用 accuracy 不是 null 的序列，并在结果里写明这一点。
不把头戴真值当成 SLAM 轨迹，也不用手写数字充当 accuracy。
"""
import argparse
import json
import re
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from headcam.hot3d_check import (  # noqa: E402
    _load_scaled_tum,
    _unique_trajectory,
    is_measured_accuracy,
    trajectory_accuracy,
)
from headcam.hot3d_sequence import contiguous_runs, parse_orbslam3_stdout  # noqa: E402

ESTIMATE_POSE_SOURCE = "orbslam3_stereo_euroc"
REFERENCE_POSE_SOURCE = "hot3d_gt_mocap"
AGGREGATE_NOTE = (
    "失败和对不上 3 个时间戳的序列留在表里，不进 RMSE 汇总。"
    "n_sequences 计入全部序列。"
)
COUNT_NOTE = (
    "tracked_frames 是保存轨迹里不同时间戳的个数。"
    "丢失帧是输入帧数减去它。ORB-SLAM3 不把跟踪失败的帧写进 CameraTrajectory.txt。"
    "reinits 是日志里 Creation of new map with id 的次数减 1，第一次是启动时的地图。"
    "日志没打印就是 null，不当成 0。"
    "maps 是保存轨迹时打印的 atlas 地图数。"
    "frames_set_to_lost_at_reset 是重置地图时打印的整数，不是丢失帧。"
)

_LOST_AT_RESET = re.compile(r"(\d+) Frames set to lost")


def _header_pose_source(path):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("# pose_source="):
            return stripped.split("=", 1)[1].strip()
        if not stripped.startswith("#"):
            return None
    return None


def _log_counts(text):
    text = text or ""
    parsed = parse_orbslam3_stdout(text)
    creations = text.count("Creation of new map with id:")
    return {
        "fail_to_track_local_map": parsed["fail_to_track_local_map"],
        "loop_closures": text.count("*Loop detected"),
        "map_creations": creations,
        "reinits": (creations - 1) if creations >= 1 else None,
        "maps": parsed["atlas_maps"],
        "map_keyframes": parsed["map_keyframes"],
        "frames_set_to_lost_at_reset": [int(value) for value in _LOST_AT_RESET.findall(text)],
    }


def _blank_counts():
    return {
        "tracked_frames": None,
        "lost_frames": None,
        "trajectory_rows": None,
        "duplicate_trajectory_rows": None,
        "fail_to_track_local_map": None,
        "loop_closures": None,
        "map_creations": None,
        "reinits": None,
        "maps": None,
        "map_keyframes": None,
        "frames_set_to_lost_at_reset": None,
    }


def _identity(name, sequence_id, participant_id, clip_ids, input_frames, device):
    return {
        "name": name,
        "sequence_id": sequence_id,
        "participant_id": participant_id,
        "device": device,
        "clip_ids": [int(value) for value in clip_ids],
        "input_frames": int(input_frames),
        "pose_source": ESTIMATE_POSE_SOURCE,
        "reference_pose_source": REFERENCE_POSE_SOURCE,
        "count_note": COUNT_NOTE,
    }


def _failed(identity, cause, counts=None):
    record = dict(identity)
    record.update(counts or _blank_counts())
    record["status"] = "failed"
    record["failure"] = cause
    record["accuracy"] = None
    return record


def score_sequence(
    name, sequence_id, participant_id, clip_ids, input_frames,
    trajectory_path=None, gt_path=None, log_text=None, failure=None,
    estimate_scale=1e-9, gt_scale=1.0, device="Quest3",
    baseline_m=None, extrinsic_drift_m=None,
):
    """一条序列的跟踪计数和 trajectory_error_v1。失败时 accuracy 是 null。"""
    identity = _identity(name, sequence_id, participant_id, clip_ids, input_frames, device)
    identity["baseline_m"] = None if baseline_m is None else float(baseline_m)
    identity["extrinsic_drift_m"] = None if extrinsic_drift_m is None else float(extrinsic_drift_m)
    identity["estimate_time_scale"] = float(estimate_scale)
    identity["gt_time_scale"] = float(gt_scale)
    if failure:
        return _failed(identity, str(failure))
    if trajectory_path is None or not Path(trajectory_path).is_file():
        return _failed(identity, "没有轨迹文件")
    if gt_path is None or not Path(gt_path).is_file():
        return _failed(identity, "没有头戴真值轨迹")

    counts = _log_counts(log_text)
    try:
        times, _poses = _load_scaled_tum(trajectory_path, estimate_scale)
    except (OSError, ValueError) as exc:
        return _failed(identity, "轨迹读不了：%s" % exc, counts)
    _unique_t, _unique_p, duplicates = _unique_trajectory(times, _poses)
    tracked = int(len(_unique_t))
    counts["tracked_frames"] = tracked
    counts["trajectory_rows"] = int(len(times))
    counts["duplicate_trajectory_rows"] = int(duplicates)
    counts["lost_frames"] = int(input_frames) - tracked if tracked <= int(input_frames) else None

    estimate_source = _header_pose_source(trajectory_path)
    if isinstance(estimate_source, str) and estimate_source.startswith("hot3d_gt"):
        return _failed(identity, "估计轨迹的文件头是头戴真值，不能当作 SLAM 位姿。", counts)
    if isinstance(estimate_source, str) and estimate_source != ESTIMATE_POSE_SOURCE:
        return _failed(
            identity,
            "估计轨迹的 pose_source=%s，不是 %s。" % (estimate_source, ESTIMATE_POSE_SOURCE),
            counts,
        )
    reference_source = _header_pose_source(gt_path)
    if reference_source is None:
        return _failed(identity, "真值文件头没有 pose_source", counts)
    if reference_source != REFERENCE_POSE_SOURCE:
        return _failed(
            identity,
            "真值文件头的 pose_source=%s，不是 %s。" % (reference_source, REFERENCE_POSE_SOURCE),
            counts,
        )

    accuracy = trajectory_accuracy(
        trajectory_path, gt_path,
        estimate_scale=estimate_scale, gt_scale=gt_scale,
        estimate_pose_source=ESTIMATE_POSE_SOURCE,
        reference_pose_source=REFERENCE_POSE_SOURCE,
    )
    record = dict(identity)
    record.update(counts)
    if accuracy is None:
        record["status"] = "unmeasured"
        record["failure"] = "对上的不同时间戳少于 3，或者尺度没有定义，不算 ATE。"
        record["accuracy"] = None
        return record
    record["status"] = "ok"
    record["failure"] = None
    record["accuracy"] = accuracy
    return record


def _span(values):
    ordered = sorted(float(value) for value in values)
    return {
        "n": len(ordered),
        "min": ordered[0],
        "median": ordered[len(ordered) // 2],
        "max": ordered[-1],
    }


def _normalize(record):
    """失败段不把塞进来的 accuracy 留在汇总里。"""
    out = dict(record)
    if out.get("status") == "failed" or not is_measured_accuracy(
        out.get("accuracy"), out.get("pose_source") or ESTIMATE_POSE_SOURCE,
    ):
        if out.get("status") != "failed":
            out["status"] = "unmeasured"
        out["accuracy"] = None
    return out


def _rpe_row(record, delta_s):
    accuracy = record.get("accuracy")
    if not isinstance(accuracy, dict):
        return None
    for row in accuracy.get("rpe") or []:
        if abs(float(row.get("delta_s", -1)) - float(delta_s)) < 1e-9:
            return row
    return None


def _rpe_aggregate(records, delta_s):
    trans = []
    rot = []
    for record in records:
        row = _rpe_row(record, delta_s)
        if not isinstance(row, dict) or "trans_rmse_m" not in row or "rot_rmse_deg" not in row:
            continue
        trans.append(row["trans_rmse_m"])
        rot.append(row["rot_rmse_deg"])
    report = {"delta_s": float(delta_s), "n": len(trans)}
    if trans:
        report["trans_rmse_m"] = _span(trans)
        report["rot_rmse_deg"] = _span(rot)
    return report


def summarize_sequences(records):
    """按输入顺序保留每一条。accuracy 为 null 的不进 RMSE。"""
    if not isinstance(records, list):
        raise ValueError("records 要是列表")
    sequences = [_normalize(record) for record in records]
    measured = [record for record in sequences if record.get("status") == "ok" and record.get("accuracy")]
    se3 = [record["accuracy"]["ate"]["se3"]["rmse_m"] for record in measured]
    sim3 = [record["accuracy"]["ate"]["sim3"]["rmse_m"] for record in measured]
    return {
        "n_sequences": len(sequences),
        "n_failed": sum(1 for record in sequences if record.get("status") == "failed"),
        "n_unmeasured": sum(1 for record in sequences if record.get("status") == "unmeasured"),
        "n_measured": len(measured),
        "aggregate_excludes_null_accuracy": True,
        "aggregate_note": AGGREGATE_NOTE,
        "ate_se3_rmse_m": _span(se3) if se3 else None,
        "ate_sim3_rmse_m": _span(sim3) if sim3 else None,
        "rpe_1s": _rpe_aggregate(measured, 1.0),
        "rpe_5s": _rpe_aggregate(measured, 5.0),
        "sequences": sequences,
    }


def _fmt(value):
    if value is None:
        return "null"
    if isinstance(value, float):
        return "%.6g" % value
    return str(value)


def _ate_field(record, alignment, key):
    accuracy = record.get("accuracy")
    if not isinstance(accuracy, dict):
        return None
    return accuracy["ate"][alignment][key]


def _rpe_field(record, delta_s, key):
    row = _rpe_row(record, delta_s)
    if not isinstance(row, dict):
        return None
    return row.get(key)


def render_markdown(summary):
    """中文表在前，英文说明在后。null 就是 null，不写成 0。"""
    lines = [
        "## 中文",
        "",
        summary["aggregate_note"],
        "",
        "序列数 %d。失败 %d。对上的时间戳不够、accuracy 为 null 的 %d。进入 RMSE 汇总的 %d。" % (
            summary["n_sequences"], summary["n_failed"], summary["n_unmeasured"], summary["n_measured"],
        ),
        "",
        "| 序列 | 参与者 | clips | 输入帧 | 跟踪 | 丢失 | 重建 | 地图 | 状态 | SE3 RMSE | SE3 中位 | SE3 最大 | Sim3 RMSE | Sim3 中位 | Sim3 最大 | 尺度 | RPE 1s 平移 RMSE | RPE 1s 旋转 RMSE | RPE 5s 平移 RMSE | RPE 5s 旋转 RMSE | 原因 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in summary["sequences"]:
        clips = record.get("clip_ids") or []
        clip_text = "%s-%s" % (clips[0], clips[-1]) if clips else "null"
        if len(clips) == 1:
            clip_text = str(clips[0])
        scale = None
        accuracy = record.get("accuracy")
        if isinstance(accuracy, dict):
            scale = accuracy["ate"]["sim3"].get("similarity_scale")
        lines.append("| %s |" % " | ".join(_fmt(value) for value in (
            record.get("name"),
            record.get("participant_id"),
            clip_text,
            record.get("input_frames"),
            record.get("tracked_frames"),
            record.get("lost_frames"),
            record.get("reinits"),
            record.get("maps"),
            record.get("status"),
            _ate_field(record, "se3", "rmse_m"),
            _ate_field(record, "se3", "median_m"),
            _ate_field(record, "se3", "max_m"),
            _ate_field(record, "sim3", "rmse_m"),
            _ate_field(record, "sim3", "median_m"),
            _ate_field(record, "sim3", "max_m"),
            scale,
            _rpe_field(record, 1.0, "trans_rmse_m"),
            _rpe_field(record, 1.0, "rot_rmse_deg"),
            _rpe_field(record, 5.0, "trans_rmse_m"),
            _rpe_field(record, 5.0, "rot_rmse_deg"),
            record.get("failure"),
        )))
    lines.extend([
        "",
        "RMSE 汇总（不含 accuracy 为 null 的序列）：",
        "",
        "- SE3 RMSE：%s" % (json.dumps(summary["ate_se3_rmse_m"], ensure_ascii=False) if summary["ate_se3_rmse_m"] else "null"),
        "- Sim3 RMSE：%s" % (json.dumps(summary["ate_sim3_rmse_m"], ensure_ascii=False) if summary["ate_sim3_rmse_m"] else "null"),
        "- RPE 1 s：%s" % json.dumps(summary["rpe_1s"], ensure_ascii=False),
        "- RPE 5 s：%s" % json.dumps(summary["rpe_5s"], ensure_ascii=False),
        "",
        "## English",
        "",
        "Failed sequences and sequences with fewer than 3 matched timestamps stay in the table and are left out of the RMSE aggregate. n_sequences counts every sequence.",
        "",
        "%d sequences, %d failed, %d unmeasured, %d in the RMSE aggregate." % (
            summary["n_sequences"], summary["n_failed"], summary["n_unmeasured"], summary["n_measured"],
        ),
        "",
    ])
    return "\n".join(lines) + "\n"


def write_reports(summary, json_path, markdown_path):
    json_path = Path(json_path)
    markdown_path = Path(markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")
    return json_path, markdown_path


def quest3_split_runs(definitions, splits, split="train", device="Quest3"):
    """训练集（或指定 split）里 Quest3 的连续段。复用 contiguous_runs，不下载 tar。"""
    records = []
    for clip_id in splits[split][device]:
        item = definitions[str(clip_id)]
        if item.get("device") != device:
            continue
        stamps = []
        for frame in item["per_frame_timestamps_ns"]:
            if "1201-1" not in frame:
                raise ValueError("clip %s 缺少 1201-1 时间戳" % clip_id)
            stamps.append(int(frame["1201-1"]))
        records.append({
            "clip_id": int(clip_id),
            "sequence_id": item["sequence_id"],
            "device": item["device"],
            "timestamps_ns": stamps,
        })
    return contiguous_runs(records)


def participant_id_of(sequence_id):
    """HOT3D 的 sequence_id 以参与者编号开头，例如 P0003_b153ade2。"""
    return str(sequence_id).split("_", 1)[0]


def run_span(run):
    """一条连续段的身份：(sequence_id, 第一段 clip, 最后一段 clip)。"""
    return (str(run["sequence_id"]), int(run["clip_ids"][0]), int(run["clip_ids"][-1]))


def next_unused_runs(runs, used_spans, per_participant=1):
    """每位参与者还没进表的最长连续段。

    used_spans 是已经入表的 run_span。同一 sequence_id 里更短的另一段不算已用。
    参与者按编号排序。同一人按帧数从长到短，帧数相同则按第一段 clip。
    """
    if per_participant < 1:
        raise ValueError("per_participant 至少是 1")
    used = {tuple(span) for span in used_spans}
    grouped = {}
    for run in runs:
        grouped.setdefault(participant_id_of(run["sequence_id"]), []).append(run)
    chosen = []
    for participant in sorted(grouped):
        ordered = sorted(
            grouped[participant],
            key=lambda run: (-int(run["n_frames"]), int(run["clip_ids"][0]), str(run["sequence_id"])),
        )
        kept = 0
        for run in ordered:
            if run_span(run) in used:
                continue
            chosen.append(run)
            kept += 1
            if kept >= per_participant:
                break
    return chosen


def _load_record(path):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("记录不是对象：%s" % path)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description="ORB-SLAM3 多序列误差表。失败序列保留，accuracy 为 null。")
    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("score", help="从一条轨迹和日志写出一份记录")
    score.add_argument("--name", required=True)
    score.add_argument("--sequence-id", required=True)
    score.add_argument("--participant", required=True)
    score.add_argument("--clips", required=True, help="逗号分隔的 clip 编号")
    score.add_argument("--input-frames", type=int, required=True)
    score.add_argument("--trajectory", default=None)
    score.add_argument("--gt", default=None)
    score.add_argument("--log", default=None, help="stereo_euroc 的标准输出")
    score.add_argument("--failure", default=None)
    score.add_argument("--estimate-scale", type=float, default=1e-9)
    score.add_argument("--gt-scale", type=float, default=1.0)
    score.add_argument("--baseline-m", type=float, default=None)
    score.add_argument("--extrinsic-drift-m", type=float, default=None)
    score.add_argument("--out", required=True)

    summary = sub.add_parser("summarize", help="把多份记录写成 JSON 和 markdown")
    summary.add_argument("--records", nargs="+", required=True)
    summary.add_argument("--json", required=True)
    summary.add_argument("--md", required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "score":
            log_text = Path(args.log).read_text(encoding="utf-8", errors="replace") if args.log else ""
            record = score_sequence(
                name=args.name,
                sequence_id=args.sequence_id,
                participant_id=args.participant,
                clip_ids=[int(part) for part in args.clips.split(",") if part != ""],
                input_frames=args.input_frames,
                trajectory_path=args.trajectory,
                gt_path=args.gt,
                log_text=log_text,
                failure=args.failure,
                estimate_scale=args.estimate_scale,
                gt_scale=args.gt_scale,
                baseline_m=args.baseline_m,
                extrinsic_drift_m=args.extrinsic_drift_m,
            )
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(record["status"], record["name"], "accuracy", "null" if record["accuracy"] is None else "measured")
            return 0
        report = summarize_sequences([_load_record(path) for path in args.records])
        write_reports(report, args.json, args.md)
        print(
            "sequences", report["n_sequences"],
            "failed", report["n_failed"],
            "unmeasured", report["n_unmeasured"],
            "measured", report["n_measured"],
        )
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
