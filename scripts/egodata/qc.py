# -*- coding: utf-8 -*-
"""用已有的手部位姿和相机位姿做自动 QC，并计算训练产出率。

产出率 = 通过片段级 QC 的帧数 / 原始帧数。
一条片段被拒绝的条件：坏帧比例超过 ``max_bad_fraction``。
坏帧来自四类信号（手出画、视线飘走、运动模糊、摆拍/静止）。
手腕置信度缺失（``None``）表示源数据没有该通道，不当成 0，也不因此判出画。
"""
import csv
import html
import math
from pathlib import Path

import numpy as np

from egodata.action_valid import DEFAULT_CHUNK_LENGTHS, validity_fields

DEFAULTS = {
    "confidence_min": 0.5,
    "blur_ang_speed": 1.5,
    "static_speed": 0.015,
    "static_min_seconds": 1.0,
    "view_angle_deg": 50.0,
    "max_bad_fraction": 0.2,
}

_FLAG_LABELS = {
    "hands_out_of_frame": "手出画",
    "view_drift": "视线飘移",
    "blur": "运动模糊",
    "staged_static": "摆拍或静止",
}


# 双目丢掉的标注（scripts/egodata/stereo_qc.py）。不计入坏帧，写报告时单独列出。
STEREO_FLAG_LABELS = {
    "stereo_one_view": "只有一目认到手",
    "stereo_inconsistent": "左右目对不上",
    "stereo_filled": "补出来的帧",
}

_EXTRA_LABELS = {
    "low_label_coverage": "标注覆盖不足",
}


def _label(name):
    return _FLAG_LABELS.get(name) or STEREO_FLAG_LABELS.get(name) or _EXTRA_LABELS.get(name) or name


def _project(intrinsic, camera_pose, point):
    """OpenCV 约定：相机 +Z 向前、+Y 向下。返回 (u, v)，在相机后方时返回 None。"""
    matrix = np.asarray(camera_pose, dtype=float)
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    point = np.asarray(point, dtype=float)
    camera_point = rotation.T @ (point - translation)
    depth = camera_point[2]
    if depth <= 1e-6:
        return None
    intrinsic = np.asarray(intrinsic, dtype=float)
    u = intrinsic[0, 0] * camera_point[0] / depth + intrinsic[0, 2]
    v = intrinsic[1, 1] * camera_point[1] / depth + intrinsic[1, 2]
    return float(u), float(v)


def _rotation_angle(before, after):
    relative = np.asarray(before, dtype=float).T @ np.asarray(after, dtype=float)
    cosine = (float(np.trace(relative)) - 1.0) / 2.0
    cosine = max(-1.0, min(1.0, cosine))
    return math.acos(cosine)


def _wrist_positions(episode, side):
    frames = episode["hands"][side]["joints"]
    positions = []
    for frame in frames:
        point = frame[0] if frame else None
        if point is None or any(coord is None for coord in point):
            positions.append(None)
        else:
            positions.append(np.asarray(point, dtype=float))
    return positions


def _speeds(positions, dt):
    speeds = [0.0] * len(positions)
    for index in range(1, len(positions)):
        prev, curr = positions[index - 1], positions[index]
        if prev is None or curr is None:
            speeds[index] = 0.0
        else:
            speeds[index] = float(np.linalg.norm(curr - prev) / dt)
    if len(speeds) > 1:
        speeds[0] = speeds[1]
    return speeds


def frame_qc_flags(episode, **overrides):
    """每帧的四类 QC 标记。布尔数组，规则与 ``qc_episode`` 相同。"""
    options = dict(DEFAULTS)
    options.update(overrides)
    num_frames = episode["num_frames"]
    fps = float(episode["fps"])
    dt = 1.0 / fps
    width = int(episode["image_width"])
    height = int(episode["image_height"])
    intrinsic = episode["camera_intrinsic"]
    camera_poses = episode["camera_poses"]
    flags = {name: np.zeros(num_frames, dtype=bool) for name in _FLAG_LABELS}

    positions = {side: _wrist_positions(episode, side) for side in ("left", "right")}
    confidences = {
        side: episode["hands"][side]["confidence"] for side in ("left", "right")
    }
    speeds = {side: _speeds(positions[side], dt) for side in ("left", "right")}

    for index in range(num_frames):
        pose = np.asarray(camera_poses[index], dtype=float)
        forward = pose[:3, 2]
        forward_norm = np.linalg.norm(forward)
        if forward_norm > 0:
            forward = forward / forward_norm
        camera_origin = pose[:3, 3]
        visible = []
        outside = []
        for side in ("left", "right"):
            point = positions[side][index]
            confidence = confidences[side][index]
            filled_flags = episode["hands"][side].get("filled")
            filled = (
                isinstance(filled_flags, list)
                and index < len(filled_flags)
                and bool(filled_flags[index])
            )
            # 置信度缺失（源 HDF5 没有 confidences）表示未知，不当成 0。
            # 未知时只靠投影判断出画；只有读到了低于阈值的数才算跟踪失败。
            # ``filled`` 是时序补洞标出来的帧：即使置信度被写成很高，QC 也不当跟踪成功。
            if filled:
                tracked = False
            elif confidence is None:
                tracked = point is not None
            else:
                tracked = point is not None and float(confidence) >= options["confidence_min"]
            if not tracked:
                outside.append(side)
                continue
            projected = _project(intrinsic, pose, point)
            if projected is None:
                outside.append(side)
                continue
            u, v = projected
            if u < 0 or v < 0 or u >= width or v >= height:
                outside.append(side)
                continue
            visible.append(point)
        if len(outside) == 2:
            flags["hands_out_of_frame"][index] = True
        if visible:
            target = np.mean(np.stack(visible), axis=0)
            direction = target - camera_origin
            norm = np.linalg.norm(direction)
            if norm > 1e-8 and forward_norm > 0:
                cosine = float(np.dot(forward, direction / norm))
                cosine = max(-1.0, min(1.0, cosine))
                angle = math.degrees(math.acos(cosine))
                if angle > options["view_angle_deg"]:
                    flags["view_drift"][index] = True
        moving = False
        for side in ("left", "right"):
            if positions[side][index] is None:
                continue
            if speeds[side][index] >= options["static_speed"]:
                moving = True
        if not moving:
            flags["staged_static"][index] = True

    for index in range(num_frames - 1):
        before = np.asarray(camera_poses[index], dtype=float)[:3, :3]
        after = np.asarray(camera_poses[index + 1], dtype=float)[:3, :3]
        ang_speed = _rotation_angle(before, after) / dt
        if ang_speed > options["blur_ang_speed"]:
            flags["blur"][index + 1] = True

    # 短暂停顿保留；只有连续静止达到阈值才算摆拍/静止。
    static = flags["staged_static"]
    run_start = None
    for index in range(num_frames + 1):
        if index < num_frames and static[index]:
            if run_start is None:
                run_start = index
        elif run_start is not None:
            duration = (index - run_start) * dt
            if duration < options["static_min_seconds"]:
                for cursor in range(run_start, index):
                    static[cursor] = False
            run_start = None
    return flags


def qc_episode(episode, **overrides):
    """对一条统一 episode 打 QC 标记。返回片段级结论。"""
    options = dict(DEFAULTS)
    options.update(overrides)
    flags = frame_qc_flags(episode, **overrides)
    num_frames = episode["num_frames"]
    fps = float(episode["fps"])
    counts = {name: int(np.sum(values)) for name, values in flags.items()}
    bad = int(np.sum(np.any(np.stack([flags[name] for name in flags]), axis=0)))
    bad_fraction = bad / float(num_frames)
    accepted = bad_fraction <= options["max_bad_fraction"]
    reasons = []
    if not accepted:
        reasons = [name for name in _FLAG_LABELS if counts[name] > 0]
    result = {
        "episode_id": episode.get("episode_id", ""),
        "num_frames": num_frames,
        "fps": fps,
        "duration_s": num_frames / fps,
        "accepted": accepted,
        "bad_frames": bad,
        "bad_fraction": bad_fraction,
        "flags": counts,
        "reasons": reasons,
    }
    result.update(validity_fields(episode))
    return result


def _aggregate_validity(results):
    """全部片段的动作步，不只是通过质检的片段。不够长的块长保持 None。"""
    action_count = 0
    valid_count = 0
    chunks = {int(length): [0, 0] for length in DEFAULT_CHUNK_LENGTHS}
    for item in results:
        action_count += int(item.get("action_count") or 0)
        valid_count += int(item.get("valid_action_count") or 0)
        for length in chunks:
            chunks[length][0] += int(item.get("chunk_count_%d" % length) or 0)
            chunks[length][1] += int(item.get("full_chunk_count_%d" % length) or 0)
    ratio = None if action_count == 0 else valid_count / float(action_count)
    full = {}
    for length, (count, full_count) in chunks.items():
        full[length] = None if count == 0 else full_count / float(count)
    return ratio, full


def yield_report(results):
    raw_frames = sum(item["num_frames"] for item in results)
    usable_frames = sum(item["num_frames"] for item in results if item["accepted"])
    rejected = sum(1 for item in results if not item["accepted"])
    ratio = 0.0 if raw_frames == 0 else usable_frames / float(raw_frames)
    valid_ratio, full_chunks = _aggregate_validity(results)
    return {
        "yield": ratio,
        "raw_frames": raw_frames,
        "usable_frames": usable_frames,
        "episodes": len(results),
        "accepted_episodes": len(results) - rejected,
        "rejected_episodes": rejected,
        "valid_action_ratio": valid_ratio,
        "valid_full_chunk_ratio": full_chunks,
        "results": results,
    }


def write_yield_reports(report, html_path, csv_path):
    html_path = Path(html_path)
    csv_path = Path(csv_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    has_coverage = any("label_coverage" in item for item in report["results"])
    rows = []
    for item in report["results"]:
        coverage_cell = ""
        if has_coverage:
            coverage = item.get("label_coverage")
            coverage_cell = "<td>%s</td>" % ("—" if coverage is None else "%.1f%%" % (100.0 * coverage))
        rows.append(
            "<tr><td>%s</td><td>%d</td><td>%s</td><td>%.1f%%</td>%s<td>%s</td></tr>"
            % (
                html.escape(str(item["episode_id"])),
                item["num_frames"],
                "通过" if item["accepted"] else "拒绝",
                100.0 * item["bad_fraction"],
                coverage_cell,
                html.escape(", ".join(_label(name) for name in item["reasons"]) or "—"),
            )
        )
    coverage_head = "<th>标注覆盖</th>" if has_coverage else ""

    def _fmt_ratio(value):
        if value is None:
            return "—"
        return "%.1f%%" % (100.0 * float(value))

    full = report.get("valid_full_chunk_ratio") or {}
    chunk_text = "，".join(
        "块长 %d %s" % (length, _fmt_ratio(full.get(length)))
        for length in DEFAULT_CHUNK_LENGTHS
    )
    validity_note = (
        "<p>有效动作比例（每只手的每一步；全部片段，不只是通过质检的）= <strong>%s</strong>。"
        "整段都有效的动作块比例：%s。没有够长的块时记为 —。</p>"
        % (_fmt_ratio(report.get("valid_action_ratio")), chunk_text)
    )
    document = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>QC 产出率</title>
<style>
body { font-family: sans-serif; margin: 2rem; }
table { border-collapse: collapse; }
td, th { border: 1px solid #ccc; padding: 0.4rem 0.6rem; }
</style></head><body>
<h1>训练产出率</h1>
<p>产出率 = 通过片段 QC 的帧数 / 原始帧数 = %d / %d = <strong>%.1f%%</strong></p>
<p>片段 %d 条，拒绝 %d 条。</p>
%s
<table>
<tr><th>episode</th><th>帧数</th><th>结论</th><th>坏帧比例</th>%s<th>原因</th></tr>
%s
</table>
</body></html>
""" % (
        report["usable_frames"],
        report["raw_frames"],
        100.0 * report["yield"],
        report["episodes"],
        report["rejected_episodes"],
        validity_note,
        coverage_head,
        "\n".join(rows),
    )
    html_path.write_text(document, encoding="utf-8")
    base = ["hands_out_of_frame", "view_drift", "blur", "staged_static"]
    extra = []
    for item in report["results"]:
        for name in item["flags"]:
            if name not in base and name not in extra:
                extra.append(name)
    coverage_cols = ["label_coverage", "labeled_frames", "dropped_label_frames"] if has_coverage else []
    validity_cols = ["valid_action_ratio"] + [
        "valid_full_chunk_ratio_%d" % length for length in DEFAULT_CHUNK_LENGTHS
    ]

    def _cell(item, key):
        if key not in item or item[key] is None:
            return ""
        return "%.6f" % float(item[key])

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "episode_id", "num_frames", "duration_s", "accepted", "bad_fraction",
            "hands_out_of_frame", "view_drift", "blur", "staged_static", "reasons",
        ] + extra + coverage_cols + validity_cols)
        for item in report["results"]:
            coverage_cells = []
            if has_coverage:
                coverage = item.get("label_coverage")
                coverage_cells = [
                    "" if coverage is None else "%.4f" % coverage,
                    item.get("labeled_frames", ""),
                    item.get("dropped_label_frames", ""),
                ]
            writer.writerow([
                item["episode_id"],
                item["num_frames"],
                "%.4f" % item["duration_s"],
                "yes" if item["accepted"] else "no",
                "%.4f" % item["bad_fraction"],
                item["flags"]["hands_out_of_frame"],
                item["flags"]["view_drift"],
                item["flags"]["blur"],
                item["flags"]["staged_static"],
                "|".join(item["reasons"]),
            ] + [item["flags"].get(name, 0) for name in extra] + coverage_cells + [
                _cell(item, key) for key in validity_cols
            ])
    return html_path, csv_path
