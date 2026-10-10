#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""在固定的 EgoDex 测试片段上比较手部时序精修。

片段在看误差之前就定好了，参数也没有在这些片段上搜索。

1. ``test/open_close_insert_remove_case/8``。仓库文档里先前完整读过的那一条。
2. test.zip 里未压缩 hdf5 在 1.6MB 到 3.2MB 之间、同名 mp4 压缩后小于 8MB 的文件，
   按路径排序取前 3 条：``add_remove_lid/14``、``add_remove_lid/23``、``add_remove_lid/8``。

手部笔记本没有写死 episode 列表。上面第 1 条是仓库里唯一被点名核对过的测试片段。
每条都用全部帧。

WiLoR 需要 MANO，权重不能放进仓库。没有 MANO、也没有缓存的 WiLoR 预测时，
用 MediaPipe，并在结果里写明。指标仍是 ``evaluate_hand_frames``。

示例::

    python scripts/eval_hand_refine.py --root data/egodex_hand_eval \\
        --out docs/hand_refine_egodex.md
"""
import argparse
import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from egodata.egodex import EGODEX_NONCORRESPONDING_JOINTS, load_episode_hdf5  # noqa: E402
from headcam.hand_pose import (  # noqa: E402
    JOINTS,
    evaluate_hand_frames,
    wilor_missing,
)
from headcam.hand_pose import transform_points  # noqa: E402
from headcam.hand_track_refine import (  # noqa: E402
    ablation_presets,
    accel_samples,
    format_ablation_table,
    metrics_row,
    refine_hands,
)

EGODEX_ZIP_URL = "https://ml-site.cdn-apple.com/datasets/egodex/test.zip"

# 选择规则写在模块说明里。不要按误差增删。
EVAL_EPISODES = (
    "test/open_close_insert_remove_case/8.hdf5",
    "test/add_remove_lid/14.hdf5",
    "test/add_remove_lid/23.hdf5",
    "test/add_remove_lid/8.hdf5",
)

LABELS = {
    "baseline": "基线",
    "smoothing": "只平滑",
    "gap_fill": "只补洞",
    "fixed_shape": "只固定手型",
    "all": "全部打开",
}

_SIG_LOCAL = 0x04034B50
_SIG_CENTRAL = 0x02014B50
_SIG_EOCD = 0x06054B50
_SIG_ZIP64_EOCD = 0x06064B50
_SIG_ZIP64_LOCATOR = 0x07064B50


class BytesRangeSource(object):
    """测试用：从一个内存里的 zip 按偏移读，不走网络。"""

    def __init__(self, data):
        self.data = data

    def content_length(self):
        return len(self.data)

    def read(self, start, length):
        start = int(start)
        length = int(length)
        if start < 0 or length < 0 or start + length > len(self.data):
            raise ValueError("读取范围超出压缩包：%d+%d，总长 %d" % (start, length, len(self.data)))
        return self.data[start:start + length]


class HttpRangeSource(object):
    """只按 HTTP Range 取字节。服务器若忽略 Range 并返回 200，直接失败，避免拉下整个 test.zip。"""

    def __init__(self, url, timeout=120):
        self.url = url
        self.timeout = timeout
        self._length = None

    def content_length(self):
        if self._length is None:
            self._length = self._probe_length()
        return self._length

    def _probe_length(self):
        import urllib.request

        request = urllib.request.Request(self.url, method="HEAD")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                value = response.headers.get("Content-Length")
                if value and value.isdigit():
                    return int(value)
        except Exception:
            pass
        request = urllib.request.Request(self.url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            status = getattr(response, "status", None)
            if status == 200:
                raise RuntimeError("服务器没有按 Range 返回，拒绝下载整个压缩包")
            content_range = response.headers.get("Content-Range", "")
            total = content_range.rsplit("/", 1)[-1] if "/" in content_range else ""
            if not total.isdigit():
                raise RuntimeError("无法从 Content-Range 读出 test.zip 的长度：%s" % content_range)
            return int(total)

    def read(self, start, length):
        import urllib.request

        start = int(start)
        length = int(length)
        if length == 0:
            return b""
        end = start + length - 1
        request = urllib.request.Request(
            self.url,
            headers={"Range": "bytes=%d-%d" % (start, end)},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            status = getattr(response, "status", None)
            if status == 200:
                raise RuntimeError("服务器没有按 Range 返回，拒绝下载整个压缩包")
            if status not in (206, None):
                raise RuntimeError("Range 请求失败，HTTP %s" % status)
            payload = response.read()
        if len(payload) != length:
            raise ValueError("Range 返回 %d 字节，期望 %d（起点 %d）" % (len(payload), length, start))
        return payload


def _zip64_sizes(compressed, uncompressed, local_offset, disk, extra):
    need_uncompressed = uncompressed == 0xFFFFFFFF
    need_compressed = compressed == 0xFFFFFFFF
    need_offset = local_offset == 0xFFFFFFFF
    need_disk = disk == 0xFFFF
    if not (need_uncompressed or need_compressed or need_offset or need_disk):
        return compressed, uncompressed, local_offset
    cursor = 0
    while cursor + 4 <= len(extra):
        header_id, size = struct.unpack_from("<HH", extra, cursor)
        payload = extra[cursor + 4:cursor + 4 + size]
        cursor += 4 + size
        if header_id != 0x0001:
            continue
        inner = 0
        if need_uncompressed:
            uncompressed = struct.unpack_from("<Q", payload, inner)[0]
            inner += 8
        if need_compressed:
            compressed = struct.unpack_from("<Q", payload, inner)[0]
            inner += 8
        if need_offset:
            local_offset = struct.unpack_from("<Q", payload, inner)[0]
            inner += 8
        return compressed, uncompressed, local_offset
    raise ValueError("ZIP64 扩展字段缺少 64 位大小或偏移")


def _parse_central_directory(blob):
    index = {}
    offset = 0
    while offset + 46 <= len(blob):
        signature = struct.unpack_from("<I", blob, offset)[0]
        if signature != _SIG_CENTRAL:
            break
        (
            _made, _need, flags, method, _time, _date, crc,
            compressed, uncompressed, name_len, extra_len, comment_len,
            disk, _internal, _external, local_offset,
        ) = struct.unpack_from("<HHHHHHIIIHHHHHII", blob, offset + 4)
        name_at = offset + 46
        name = blob[name_at:name_at + name_len].decode("utf-8")
        extra = blob[name_at + name_len:name_at + name_len + extra_len]
        compressed, uncompressed, local_offset = _zip64_sizes(
            compressed, uncompressed, local_offset, disk, extra,
        )
        index[name] = {
            "method": int(method),
            "crc": int(crc) & 0xFFFFFFFF,
            "compressed_size": int(compressed),
            "uncompressed_size": int(uncompressed),
            "local_offset": int(local_offset),
            "flags": int(flags),
        }
        offset = name_at + name_len + extra_len + comment_len
    if not index:
        raise ValueError("中央目录里没有文件")
    return index


def _locate_eocd(tail):
    signature = struct.pack("<I", _SIG_EOCD)
    position = len(tail)
    while True:
        position = tail.rfind(signature, 0, position)
        if position < 0:
            raise ValueError("找不到 ZIP 结束记录")
        if position + 22 > len(tail):
            continue
        comment_len = struct.unpack_from("<H", tail, position + 20)[0]
        if position + 22 + comment_len == len(tail):
            return position


def read_zip_index(source):
    """读 ZIP 或 ZIP64 的中央目录。只取目录本身和文件尾，不读成员数据。"""
    length = int(source.content_length())
    tail_len = min(length, 22 + 65535 + 76)
    tail = source.read(length - tail_len, tail_len)
    eocd_at = _locate_eocd(tail)
    (
        _disk, _cd_disk, _entries_here, entries, cd_size, cd_offset, _comment_len,
    ) = struct.unpack_from("<HHHHIIH", tail, eocd_at + 4)
    if cd_offset == 0xFFFFFFFF or cd_size == 0xFFFFFFFF or entries == 0xFFFF:
        locator_at = eocd_at - 20
        if locator_at < 0:
            raise ValueError("ZIP64 定位记录不在已读取的文件尾里")
        loc_sig, _loc_disk, zip64_offset, _disks = struct.unpack_from("<IIQI", tail, locator_at)
        if loc_sig != _SIG_ZIP64_LOCATOR:
            raise ValueError("找不到 ZIP64 中央目录定位记录")
        record = source.read(zip64_offset, 56)
        (
            zsig, _zsize, _made, _need, _zdisk, _zcd_disk,
            _entries_here64, entries, cd_size, cd_offset,
        ) = struct.unpack("<IQHHIIQQQQ", record)
        if zsig != _SIG_ZIP64_EOCD:
            raise ValueError("ZIP64 结束记录签名不对")
    directory = source.read(int(cd_offset), int(cd_size))
    return _parse_central_directory(directory)


def extract_member(source, entry):
    """按中央目录里的偏移和压缩大小取出一个成员并校验 CRC。"""
    header = source.read(entry["local_offset"], 30)
    signature, _ver, _flags, _method, _time, _date, _crc, _comp, _uncomp, name_len, extra_len = struct.unpack(
        "<IHHHHHIIIHH", header,
    )
    if signature != _SIG_LOCAL:
        raise ValueError("本地文件头签名不对，偏移 %d" % entry["local_offset"])
    data_offset = entry["local_offset"] + 30 + name_len + extra_len
    compressed = source.read(data_offset, entry["compressed_size"])
    method = entry["method"]
    if method == 0:
        data = compressed
    elif method == 8:
        data = zlib.decompress(compressed, -15)
    else:
        raise ValueError("不支持的压缩方法 %d" % method)
    if len(data) != entry["uncompressed_size"]:
        raise ValueError("解压后 %d 字节，目录记录 %d" % (len(data), entry["uncompressed_size"]))
    actual = zlib.crc32(data) & 0xFFFFFFFF
    if actual != entry["crc"]:
        raise ValueError("CRC 不一致")
    return data


def missing_episode_names(root, episodes):
    root = Path(root)
    missing = []
    for relative in episodes:
        hdf5 = root / relative
        mp4_rel = str(Path(relative).with_suffix(".mp4")).replace("\\", "/")
        if not hdf5.is_file():
            missing.append(str(relative).replace("\\", "/"))
        if not (root / mp4_rel).is_file():
            missing.append(mp4_rel)
    return missing


def ensure_episodes(root, episodes=EVAL_EPISODES, url=EGODEX_ZIP_URL, source=None):
    """缺的 hdf5/mp4 才从 test.zip 按成员取出。文件已在时不访问网络。"""
    root = Path(root)
    missing = missing_episode_names(root, episodes)
    if not missing:
        return []
    if source is None:
        source = HttpRangeSource(url)
    index = read_zip_index(source)
    written = []
    for name in missing:
        if name not in index:
            raise FileNotFoundError("test.zip 里没有 %s" % name)
        payload = extract_member(source, index[name])
        dest = root / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)
        written.append(name)
    return written


def _empty_joints(count):
    return np.full((count, JOINTS, 3), np.nan, dtype=np.float64)


def _empty_conf(count):
    return np.full((count, JOINTS), np.nan, dtype=np.float64)


def _parse_hand(hand, joints, confidence, index):
    if not isinstance(hand, dict) or hand.get("joints_cam") is None:
        confidence[index] = 0.0
        return
    value = np.asarray(hand["joints_cam"], dtype=np.float64)
    if value.shape == (JOINTS, 3):
        finite = np.isfinite(value)
        joints[index][finite] = value[finite]
    raw = hand.get("confidence")
    if raw is None:
        return
    array = np.asarray(raw, dtype=np.float64).reshape(-1)
    if array.size == 1:
        confidence[index] = float(array[0])
    elif array.size == JOINTS:
        finite = np.isfinite(array)
        confidence[index][finite] = array[finite]


def predict_episode(mp4, intrinsic, cache_path, wrist_depth_m, backend_name="mediapipe"):
    cache_path = Path(cache_path)
    if cache_path.is_file():
        stored = np.load(cache_path)
        return {
            "left": stored["left"],
            "right": stored["right"],
            "left_conf": stored["left_conf"],
            "right_conf": stored["right_conf"],
            "width": int(stored["width"]),
            "height": int(stored["height"]),
        }
    import cv2
    from headcam.hand_pose import MediaPipeHandsBackend, get_backend

    if backend_name == "mediapipe":
        backend = MediaPipeHandsBackend(wrist_depth_m=wrist_depth_m)
    else:
        backend = get_backend(backend_name, wrist_depth_m=wrist_depth_m)
    calib = {"K_left": np.asarray(intrinsic, dtype=np.float64)}
    capture = cv2.VideoCapture(str(mp4))
    if not capture.isOpened():
        raise RuntimeError("打不开视频 %s" % mp4)
    left_rows = []
    right_rows = []
    left_conf_rows = []
    right_conf_rows = []
    width = height = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if width is None:
                height, width = frame.shape[:2]
            prediction = backend.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), calib=calib)
            if len(left_rows) % 100 == 0:
                print("predict", mp4, "frame", len(left_rows), flush=True)
            left = _empty_joints(1)[0]
            right = _empty_joints(1)[0]
            left_conf = _empty_conf(1)[0]
            right_conf = _empty_conf(1)[0]
            _parse_hand(prediction.get("left"), left.reshape(1, JOINTS, 3), left_conf.reshape(1, JOINTS), 0)
            _parse_hand(prediction.get("right"), right.reshape(1, JOINTS, 3), right_conf.reshape(1, JOINTS), 0)
            # _parse_hand expects arrays with an index. Call it on length-1 buffers.
            left_rows.append(left)
            right_rows.append(right)
            left_conf_rows.append(left_conf)
            right_conf_rows.append(right_conf)
    finally:
        capture.release()
    if width is None:
        raise RuntimeError("视频里没有帧：%s" % mp4)
    payload = {
        "left": np.stack(left_rows),
        "right": np.stack(right_rows),
        "left_conf": np.stack(left_conf_rows),
        "right_conf": np.stack(right_conf_rows),
        "width": int(width),
        "height": int(height),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **payload)
    return payload


def _world_to_camera(points, pose):
    from headcam.hand_pose import world_to_camera
    return world_to_camera(points, pose)


def build_samples(episode, prediction, width, height, drop_unconfident):
    intrinsic = np.asarray(episode["camera_intrinsic"], dtype=np.float64)
    count = min(int(episode["num_frames"]), int(prediction["left"].shape[0]))
    samples = []
    for index in range(count):
        pose = np.asarray(episode["camera_poses"][index], dtype=np.float64)
        sample = {
            "episode_id": episode["episode_id"],
            "width": int(width),
            "height": int(height),
            "K": intrinsic,
            "gt": {},
            "gt_confidence": {},
            "pred": {},
        }
        for side in ("left", "right"):
            raw = episode["hands"][side]["joints"][index]
            numeric = np.full((JOINTS, 3), np.nan, dtype=np.float64)
            for joint_index, point in enumerate(raw):
                if point is not None and point[0] is not None:
                    numeric[joint_index] = point
            if np.isfinite(numeric[0]).all():
                sample["gt"][side] = _world_to_camera(numeric, pose)
            else:
                sample["gt"][side] = None
            sample["gt_confidence"][side] = episode["hands"][side]["confidence"][index]
            joints = prediction[side][index]
            score = prediction[side + "_conf"][index, 0]
            filled = bool(prediction[side + "_filled"][index])
            low = np.isfinite(score) and float(score) < 0.5
            if drop_unconfident and (filled or low):
                sample["pred"][side] = None
            elif not np.isfinite(joints[0]).all():
                sample["pred"][side] = None
            else:
                sample["pred"][side] = {"joints_cam": joints, "keypoints_2d": None}
        samples.append(sample)
    return samples


def _baseline_mask(joints):
    return np.isfinite(joints[:, 0, :]).all(axis=1)


def _world_sequence(joints, poses):
    out = np.full_like(joints, np.nan)
    count = min(len(poses), joints.shape[0])
    for index in range(count):
        out[index] = transform_points(joints[index], poses[index])
    return out


def pooled_jitter(episodes):
    chunks = []
    for item in episodes:
        chunks.append(accel_samples(item["points"], item["timestamps"], item["mask"]))
    if not chunks:
        return float("nan")
    values = np.concatenate(chunks)
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def refine_episode(prediction, episode, params):
    count = min(int(episode["num_frames"]), int(prediction["left"].shape[0]))
    poses = np.asarray(episode["camera_poses"][:count], dtype=np.float64)
    times = np.asarray(episode["timestamps"][:count], dtype=np.float64)
    if not params.enabled():
        refined = {
            "left": prediction["left"][:count],
            "right": prediction["right"][:count],
            "left_conf": prediction["left_conf"][:count],
            "right_conf": prediction["right_conf"][:count],
            "left_filled": np.zeros(count, dtype=bool),
            "right_filled": np.zeros(count, dtype=bool),
        }
        return refined, poses, times
    result = refine_hands(
        prediction["left"][:count],
        prediction["right"][:count],
        prediction["left_conf"][:count],
        prediction["right_conf"][:count],
        times,
        params,
        camera_poses=poses,
    )
    refined = {
        "left": result["left"]["joints"],
        "right": result["right"]["joints"],
        "left_conf": result["left"]["confidence"],
        "right_conf": result["right"]["confidence"],
        "left_filled": result["left"]["filled"],
        "right_filled": result["right"]["filled"],
    }
    return refined, poses, times


def _fmt_delta(before, after, digits, suffix=""):
    if before is None or after is None or not np.isfinite(before) or not np.isfinite(after):
        return "n/a"
    pattern = "%." + str(int(digits)) + "f"
    return ((pattern + " → " + pattern) % (float(before), float(after))) + suffix


def _reading_notes(rows, wrist_depth_m):
    """只用已经算出来的数写比较，不另造指标。"""
    base = rows[("baseline", "excluded")]
    smooth = rows[("smoothing", "excluded")]
    gap = rows[("gap_fill", "excluded")]
    shape = rows[("fixed_shape", "excluded")]
    whole = rows[("all", "excluded")]
    lines = [
        "下面这几句只对照「排除关节 0 和 1」那张表，每一项都和基线比。",
        "",
        "- 只平滑：抖动 %.3f → %.3f m/s²，检出率 %.1f%% → %.1f%%，MPJPE 均值 %.2f → %.2f cm，对齐后手腕中位 %.2f → %.2f cm。滤波会压掉帧间跳动；手真的在动时也会滞后，所以位置误差不一定一起变好。" % (
            base["jitter_mps2"], smooth["jitter_mps2"],
            100.0 * base["detection_rate"], 100.0 * smooth["detection_rate"],
            base["mpjpe_cm"], smooth["mpjpe_cm"],
            base["wrist_scaled_cm_median"], smooth["wrist_scaled_cm_median"],
        ),
        "- 只补洞（含左右轨迹）：几何检出率 %.1f%% → %.1f%%，高置信检出率 %.1f%% → %.1f%%，左右交换率 %.1f%% → %.1f%%，MPJPE 均值 %.2f → %.2f cm。补上的帧置信度是 0，所以高置信检出率不靠它们上升；QC 也会拒绝这些帧。这一档的抖动如果下降，多半是左右标签不再中途对调：对调会让同一侧轨迹跳到另一只手上，加速度会很大。抖动只统计基线里本来就有手腕的帧，不是把插值点算进去。" % (
            100.0 * base["detection_rate"], 100.0 * gap["detection_rate"],
            100.0 * base["confident_detection_rate"], 100.0 * gap["confident_detection_rate"],
            100.0 * base["swap_rate"], 100.0 * gap["swap_rate"],
            base["mpjpe_cm"], gap["mpjpe_cm"],
        ),
        "- 只固定手型：MPJPE 均值 %.2f → %.2f cm，手腕无尺度中位 %.2f → %.2f cm，抖动 %.3f → %.3f m/s²。重摆时手腕留在原地，所以手腕位置误差基本不动。" % (
            base["mpjpe_cm"], shape["mpjpe_cm"],
            base["wrist_cm_median"], shape["wrist_cm_median"],
            base["jitter_mps2"], shape["jitter_mps2"],
        ),
        "- 全部打开：检出率 %.1f%% → %.1f%%，MPJPE 均值 %.2f → %.2f cm，抖动 %.3f → %.3f m/s²，左右交换率 %.1f%% → %.1f%%，高置信检出率 %.1f%% → %.1f%%。" % (
            100.0 * base["detection_rate"], 100.0 * whole["detection_rate"],
            base["mpjpe_cm"], whole["mpjpe_cm"],
            base["jitter_mps2"], whole["jitter_mps2"],
            100.0 * base["swap_rate"], 100.0 * whole["swap_rate"],
            100.0 * base["confident_detection_rate"], 100.0 * whole["confident_detection_rate"],
        ),
    ]
    if np.isfinite(whole["swap_rate"]) and whole["swap_rate"] > 0.2:
        lines.extend([
            "",
            "全部打开之后左右交换率仍有 %.1f%%（基线 %.1f%%）。轨迹一致性只把中途对调的标签换回来，第一帧定下来的左右会一直跟着走。它不会把整段标签翻成 EgoDex 的定义。"
            % (100.0 * whole["swap_rate"], 100.0 * base["swap_rate"]),
            "",
        ])
    lines.append(
        "抖动的绝对值很大。MediaPipe 的深度是 %.2f m 的先验，再乘上头的转动，世界系里的加速度会被放大。五档用的是同一次相机位姿，所以可以比谁更稳，这个数本身不是手的真实加速度。"
        % float(wrist_depth_m)
    )
    return "\n".join(lines)


def write_report(path, context, tables, rows_by_key):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    excluded = rows_by_key[("all", "excluded")]
    baseline = rows_by_key[("baseline", "excluded")]
    lines = [
        "# 手部时序精修：EgoDex 测试片段",
        "",
        "这份结果由 `python scripts/eval_hand_refine.py` 写出。表里的数是同一次运行的输出，没有手填，也没有在这些片段上改参数。",
        "",
        "## 后端",
        "",
        context["backend_note"],
        "",
        "单目 MediaPipe 的手腕深度是先验 **%.2f m**，不是双目测出来的米。手腕「无尺度」误差里有一大截是这个深度差。对齐后的手腕误差会按每条片段、每只手乘一个尺度，用来看形状和方向，不把深度先验的整体缩放算进去。" % context["wrist_depth_m"],
        "",
        "## 片段",
        "",
        "手部笔记本（`notebooks/headcam_hand_pose_colab.ipynb`）没有写死 episode，只认环境变量 `EGODEX_HDF5`。仓库里先前完整核对过结构的测试片段是 `test/open_close_insert_remove_case/8`。这次把它留下，再按文件大小事先选定另外 3 条，凑够看时序的帧数。规则写在 `scripts/eval_hand_refine.py` 顶部：未压缩 hdf5 在 1.6–3.2MB、mp4 压缩后小于 8MB，按路径排序取前 3 条。没有按误差换片段，也没有抽帧。",
        "",
        "| 片段 | 帧数 |",
        "| --- | --- |",
    ]
    for name, frames in context["episodes"]:
        lines.append("| `%s` | %d |" % (name, frames))
    lines.extend([
        "",
        "合计 **%d** 帧。骨长按每条片段单独取中位数，再把各条的误差合在一起算。" % context["total_frames"],
        "",
        "## 开关",
        "",
        "默认参数，没有在这 4 条上搜索：",
        "",
        "- 平滑：One Euro，`min_cutoff=%.2f` Hz，`beta=%.2f` 1/(米/秒)，`d_cutoff=%.2f` Hz。有相机位姿时在世界系里滤，再变回相机系。" % (
            context["min_cutoff"], context["beta"], context["d_cutoff"],
        ),
        "- 补洞：最多 **%d** 帧。补上的帧 `filled=true`，置信度 **0**。QC 看到 `filled` 就不当成跟踪成功。" % context["max_gap"],
        "- 只补洞这一档同时打开左右轨迹一致性。两种左右分配的代价差不到 2 cm 时，保留检测器原来的标签。",
        "- 固定手型：20 段骨头在高置信帧上的稳健中位数，手腕不动。没有 MANO 文件，所以没有 betas。",
        "- 常速度卡尔曼在代码里可选（`--smooth kalman`），这张表用的是 One Euro。",
        "",
        "## 指标",
        "",
        "和笔记本同一个函数 `evaluate_hand_frames`：手腕 2D 距离 250 px 以内算检出，左右标签不一致算交换。MPJPE 是相对手腕的关节误差的**均值**（厘米）；表里也给了中位数。手腕误差给对齐前和对齐后的中位数和均值。",
        "",
        "抖动是世界系里关节加速度的模长，再对关节和帧取平均，单位米/秒²。只统计基线里手腕有数、并且前后帧也有数的那些帧，五档用同一批帧，避免补洞多出来的点把抖动拉低。",
        "",
        "高置信检出率把 `filled` 或置信度低于 0.5 的预测拿掉再匹配。补洞不该靠这些帧把「QC 能留下的检出」刷高。",
        "",
        "## 排除关节 0 和 1",
        "",
        "EgoDex 的 `Hand` 在前臂上，`ThumbKnuckle` 不是拇指 CMC。和 MediaPipe 比的时候，笔记本会排除这两点。下面这张表同样排除。",
        "",
        tables["excluded"],
        "",
        "基线匹配 %d / 可见 %d。全部打开匹配 %d / 可见 %d。" % (
            baseline["matched"], baseline["gt_visible"], excluded["matched"], excluded["gt_visible"],
        ),
        "",
        "全部打开相对基线（排除关节 0 和 1）：检出率 %s，MPJPE 均值 %s cm，手腕无尺度中位 %s cm，手腕对齐后中位 %s cm，抖动 %s m/s²，左右交换率 %s，高置信检出率 %s。" % (
            _fmt_delta(baseline["detection_rate"] * 100.0, excluded["detection_rate"] * 100.0, 1, "%"),
            _fmt_delta(baseline["mpjpe_cm"], excluded["mpjpe_cm"], 2),
            _fmt_delta(baseline["wrist_cm_median"], excluded["wrist_cm_median"], 2),
            _fmt_delta(baseline["wrist_scaled_cm_median"], excluded["wrist_scaled_cm_median"], 2),
            _fmt_delta(baseline["jitter_mps2"], excluded["jitter_mps2"], 3),
            _fmt_delta(baseline["swap_rate"] * 100.0, excluded["swap_rate"] * 100.0, 1, "%"),
            _fmt_delta(
                baseline["confident_detection_rate"] * 100.0,
                excluded["confident_detection_rate"] * 100.0,
                1,
                "%",
            ),
        ),
        "",
        "## 全部关节",
        "",
        "不排除任何关节时，相对手腕的误差会把第 1 点也算进去。手腕位置误差和交换率、抖动与上一张表相同，因为它们不用这套排除列表。",
        "",
        tables["all"],
        "",
        "## 这些数在说什么",
        "",
        _reading_notes(rows_by_key, context["wrist_depth_m"]),
        "",
        "## 复现",
        "",
        "```bash",
        "python scripts/eval_hand_refine.py --root data/egodex_hand_eval --out docs/hand_refine_egodex.md",
        "```",
        "",
        "数据放在 `data/egodex_hand_eval/`（`/data/` 已在 gitignore 里，不提交）。缺 hdf5 或 mp4 时，脚本用 HTTP Range 从 EgoDex 的 `test.zip` 只取出上面 4 对文件，不下载整个约 16GB 的压缩包。检测缓存在该目录的 `cache/`。",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run(root, out_path, wrist_depth_m, backend_name="mediapipe"):
    root = Path(root)
    downloaded = ensure_episodes(root)
    if downloaded:
        print("downloaded", len(downloaded), "zip members", flush=True)
    missing = wilor_missing()
    cache_hits = list(root.rglob("*wilor*")) if root.exists() else []
    repo_hits = list(Path(".").glob("**/*wilor*.npz"))
    if missing:
        backend_note = (
            "WiLoR 在这台机器上跑不了，缺少：%s。"
            "仓库和 `data/` 里也没有缓存的 WiLoR 预测（找到 %d 个文件名里带 wilor 的缓存）。"
            "下面的数是 **MediaPipe Hands**（Tasks `HandLandmarker`，CPU）对这 4 条视频的检测，再做时序精修。"
            "它说明精修对这个检测器有没有用，不能当成 WiLoR 的误差。"
            % ("；".join(missing), len(cache_hits) + len(repo_hits))
        )
    else:
        backend_note = "WiLoR 可用。"
    if backend_name == "wilor":
        if missing:
            raise RuntimeError("要求 --backend wilor，但缺少：%s" % "；".join(missing))
        backend_note = (
            "下面的数是 **WiLoR**（官方 wilor_final.ckpt + detector.pt，经 `headcam.hand_pose.get_backend(\"wilor\")`，CPU）"
            "对这 4 条视频逐帧检测，再做时序精修。单目 WiLoR 的手腕深度来自它自己的相机平移估计，不是双目测出来的米。"
        )
    predictions = []
    episode_rows = []
    for relative in EVAL_EPISODES:
        hdf5 = root / relative
        mp4 = hdf5.with_suffix(".mp4")
        if not hdf5.is_file() or not mp4.is_file():
            raise FileNotFoundError("缺少 %s 或同名 mp4。放到 %s 后再跑。" % (relative, root))
        episode = load_episode_hdf5(hdf5)
        cache = root / "cache" / (relative.replace("/", "__") + "_%s_d%03d.npz" % (backend_name, int(round(wrist_depth_m * 100))))
        prediction = predict_episode(mp4, episode["camera_intrinsic"], cache, wrist_depth_m, backend_name)
        count = min(int(episode["num_frames"]), int(prediction["left"].shape[0]))
        episode_rows.append((relative, count))
        predictions.append((episode, prediction, count))
        print("loaded", relative, "frames", count, flush=True)
    grouped = {}
    jitter = {}
    for key, params in ablation_presets():
        pooled = []
        confident = []
        jitter_items = []
        for episode, prediction, count in predictions:
            refined, poses, times = refine_episode(prediction, episode, params)
            width = int(prediction["width"])
            height = int(prediction["height"])
            pooled.extend(build_samples(episode, refined, width, height, drop_unconfident=False))
            confident.extend(build_samples(episode, refined, width, height, drop_unconfident=True))
            if key == "baseline":
                prediction["_baseline_mask"] = {
                    "left": _baseline_mask(refined["left"]),
                    "right": _baseline_mask(refined["right"]),
                }
            masks = prediction["_baseline_mask"]
            for side in ("left", "right"):
                world = _world_sequence(refined[side], poses)
                jitter_items.append({
                    "points": world,
                    "timestamps": times,
                    "mask": masks[side][:count],
                })
        grouped[key] = {
            "samples": pooled,
            "confident": confident,
            "jitter": pooled_jitter(jitter_items),
        }
        print("variant", key, "samples", len(pooled), "jitter", grouped[key]["jitter"], flush=True)
    if "baseline" not in grouped:
        raise RuntimeError("消融缺少 baseline，抖动掩码没有参照")
    rows_by_key = {}
    tables = {}
    serializable = {}
    for label, exclude in (("excluded", EGODEX_NONCORRESPONDING_JOINTS), ("all", ())):
        table_rows = []
        for key, _params in ablation_presets():
            report = evaluate_hand_frames(grouped[key]["samples"], exclude_joints=exclude)
            confident_report = evaluate_hand_frames(grouped[key]["confident"], exclude_joints=exclude)
            row = metrics_row(LABELS[key], report, grouped[key]["jitter"], confident_report)
            table_rows.append(row)
            rows_by_key[(key, label)] = row
            serializable["%s/%s" % (key, label)] = row
        tables[label] = format_ablation_table(table_rows)
        print(label)
        print(tables[label])
    preset = dict(ablation_presets())["all"]
    context = {
        "backend_note": backend_note,
        "wrist_depth_m": float(wrist_depth_m),
        "episodes": episode_rows,
        "total_frames": int(sum(frames for _name, frames in episode_rows)),
        "min_cutoff": preset.min_cutoff,
        "beta": preset.beta,
        "d_cutoff": preset.d_cutoff,
        "max_gap": preset.max_gap,
    }
    if out_path:
        write_report(out_path, context, tables, rows_by_key)
        sidecar = Path(out_path).with_suffix(".json")
        sidecar.write_text(json.dumps({
            "backend": backend_name,
            "wilor_missing": missing,
            "episodes": [{"path": name, "frames": frames} for name, frames in episode_rows],
            "metrics": serializable,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print("wrote", out_path)
    return rows_by_key


def main(argv=None):
    parser = argparse.ArgumentParser(description="EgoDex 上的手部时序精修消融")
    parser.add_argument("--root", default="data/egodex_hand_eval")
    parser.add_argument("--out", default="docs/hand_refine_egodex.md")
    parser.add_argument("--wrist-depth-m", type=float, default=0.55)
    parser.add_argument("--backend", choices=("mediapipe", "wilor"), default="mediapipe")
    args = parser.parse_args(argv)
    run(args.root, args.out, args.wrist_depth_m, args.backend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
