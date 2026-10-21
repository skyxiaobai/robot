# -*- coding: utf-8 -*-
"""把同一条 HOT3D 录制里时间接得上的 clip 拼成一条序列。

clip 编号相邻不等于时间相邻。拼起来的条件是：同一 ``sequence_id``、同一设备，
并且上一段最后一帧到下一段第一帧的间隔等于这一段自己的帧间隔（允许 2%）。
间隔更大，或时间重叠，就分成两段，不把缺口当成连续跟踪。
"""
import re


def _median(values):
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def frame_step_ns(timestamps_ns):
    """中位帧间隔。少于两帧，或间隔不是正数，返回 None。"""
    stamps = [int(value) for value in timestamps_ns]
    steps = [b - a for a, b in zip(stamps, stamps[1:]) if b > a]
    if not steps:
        return None
    return int(_median(steps))


def _record(item):
    stamps = [int(value) for value in item["timestamps_ns"]]
    if len(stamps) < 2:
        raise ValueError("clip %s 少于 2 帧，不能判断是不是接得上" % item.get("clip_id"))
    if stamps != sorted(stamps) or len(set(stamps)) != len(stamps):
        raise ValueError("clip %s 的时间戳没有严格递增" % item.get("clip_id"))
    return {
        "clip_id": item["clip_id"],
        "sequence_id": item["sequence_id"],
        "device": item["device"],
        "timestamps_ns": stamps,
    }


def _gap_matches_step(previous, nxt):
    """下一段起点比上一段终点正好晚一帧。差一个数量级的缺口返回 False。"""
    step = frame_step_ns(previous["timestamps_ns"])
    if not step:
        return False
    gap = int(nxt["timestamps_ns"][0]) - int(previous["timestamps_ns"][-1])
    return abs(gap - step) <= max(2, int(round(0.02 * step)))


def contiguous_runs(records):
    """按设备和序列分组，再按时间把接得上的 clip 合成 run。"""
    grouped = {}
    for item in records:
        record = _record(item)
        key = (record["device"], record["sequence_id"])
        grouped.setdefault(key, []).append(record)
    runs = []
    for key in sorted(grouped, key=lambda item: (str(item[0]), str(item[1]))):
        ordered = sorted(grouped[key], key=lambda record: record["timestamps_ns"][0])
        current = [ordered[0]]
        for record in ordered[1:]:
            if _gap_matches_step(current[-1], record):
                current.append(record)
            else:
                runs.append(_finish_run(current))
                current = [record]
        runs.append(_finish_run(current))
    runs.sort(key=lambda run: (-run["n_frames"], str(run["sequence_id"]), run["timestamps_ns"][0]))
    return runs


def _finish_run(records):
    stamps = []
    gaps = []
    for index, record in enumerate(records):
        if index:
            gaps.append(int(record["timestamps_ns"][0]) - int(records[index - 1]["timestamps_ns"][-1]))
        stamps.extend(record["timestamps_ns"])
    if len(set(stamps)) != len(stamps) or stamps != sorted(stamps):
        raise ValueError("拼起来的时间戳不连续：%s" % [record["clip_id"] for record in records])
    return {
        "sequence_id": records[0]["sequence_id"],
        "device": records[0]["device"],
        "clip_ids": [record["clip_id"] for record in records],
        "timestamps_ns": stamps,
        "gaps_ns": gaps,
        "n_frames": len(stamps),
        "frame_step_ns": frame_step_ns(stamps),
    }


def concat_clip_records(records):
    """这些 clip 必须正好是一条接得上的 run。否则抛出 ValueError，不返回半截。"""
    if not records:
        raise ValueError("没有 clip")
    runs = contiguous_runs(records)
    wanted = {_record(item)["clip_id"] for item in records}
    matches = [run for run in runs if set(run["clip_ids"]) == wanted and len(run["clip_ids"]) == len(records)]
    if len(matches) != 1:
        raise ValueError(
            "这些 clip 不是同一条连续录制。按时间接得上的是：%s"
            % [run["clip_ids"] for run in runs]
        )
    return matches[0]


_MAP_LINE = re.compile(r"Map (\d+) has (\d+) KFs")
_ATLAS_LINE = re.compile(r"There are (\d+) maps in the atlas")


def parse_orbslam3_stdout(text):
    """从 ORB-SLAM3 的标准输出里数跟踪失败和地图。不把没打印的状态猜成 0 以外的原因。"""
    text = text or ""
    atlas = _ATLAS_LINE.search(text)
    return {
        "fail_to_track_local_map": text.count("Fail to track local map!"),
        "new_maps": text.count("New Map created with"),
        "few_reference_matches": text.count("Less than 15 matches"),
        "atlas_maps": int(atlas.group(1)) if atlas else None,
        "map_keyframes": [
            {"map": int(map_id), "keyframes": int(count)}
            for map_id, count in _MAP_LINE.findall(text)
        ],
    }
