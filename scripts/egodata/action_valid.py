# -*- coding: utf-8 -*-
"""每只手的动作是否来自实测手腕。

缺测、补帧、立体或逐帧标注丢掉的帧、以及读到了低于 0.5 的置信度，
都不能当成「保持不动」。占位增量可以留下，训练损失必须把这些步标成无效。

``good_frame_mask`` 若写在 episode 上（或 ``qc`` / ``stereo`` / ``iphone`` 里），
该帧为 false 时两只手都无效。没有这份掩码时，按每只手自己的标注状态判断，
不把模糊、视线飘移、摆拍这类「帧是坏的但手腕测到了」的情况算成缺测。
"""
import numpy as np

WRIST_DIM = 7
JOINTS = 21
CONFIDENCE_MIN = 0.5
DEFAULT_CHUNK_LENGTHS = (16, 50, 100)
_OK_STATUS = {"", "none", "ok"}


def _finite_wrist(value):
    if not isinstance(value, (list, tuple)) or len(value) != WRIST_DIM:
        return False
    if any(item is None for item in value):
        return False
    array = np.asarray(value, dtype=np.float64)
    return bool(np.isfinite(array).all())


def _confidence_ok(confidence):
    """缺失表示未知，通过。只有读到了低于阈值的数才失败。"""
    if confidence is None:
        return True
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(value):
        return False
    return value >= CONFIDENCE_MIN


def _is_filled(hand, frame_index):
    filled = hand.get("filled")
    if not isinstance(filled, (list, tuple)) or frame_index >= len(filled):
        return False
    return bool(filled[frame_index])


def _status_ok(status):
    if status is None:
        return True
    if isinstance(status, str):
        return status.strip().lower() in _OK_STATUS
    return False


def _label_statuses(episode, side, frame_index):
    hand = episode["hands"][side]
    for key in ("label_status", "status"):
        column = hand.get(key)
        if isinstance(column, (list, tuple)) and frame_index < len(column):
            yield column[frame_index]
    for parent_key in ("stereo", "iphone"):
        parent = episode.get(parent_key)
        if not isinstance(parent, dict):
            continue
        per_frame = parent.get("per_frame") or {}
        if not isinstance(per_frame, dict):
            continue
        column = per_frame.get(side)
        if isinstance(column, (list, tuple)) and frame_index < len(column):
            yield column[frame_index]


def _mask_rejects(episode, frame_index):
    """显式 good_frame_mask 里的 false 帧，两只手都不当成实测。"""
    masks = []
    direct = episode.get("good_frame_mask")
    if isinstance(direct, (list, tuple)):
        masks.append(direct)
    for parent_key in ("qc", "stereo", "iphone"):
        parent = episode.get(parent_key)
        if isinstance(parent, dict) and isinstance(parent.get("good_frame_mask"), (list, tuple)):
            masks.append(parent["good_frame_mask"])
    for mask in masks:
        if frame_index < len(mask) and not bool(mask[frame_index]):
            return True
    return False


def wrist_measured(episode, side, frame_index):
    """这一帧这只手的手腕是不是测出来的。"""
    hand = episode["hands"][side]
    if frame_index < 0 or frame_index >= len(hand["wrist_pose"]):
        return False
    if not _finite_wrist(hand["wrist_pose"][frame_index]):
        return False
    if _is_filled(hand, frame_index):
        return False
    confidence = hand["confidence"][frame_index] if frame_index < len(hand["confidence"]) else None
    if not _confidence_ok(confidence):
        return False
    for status in _label_statuses(episode, side, frame_index):
        if not _status_ok(status):
            return False
    if _mask_rejects(episode, frame_index):
        return False
    return True


def action_valid_flags(episode, frame_index):
    """当前帧到下一帧，左手、右手的增量是否可监督。float32，形状 (2,)。"""
    nxt = int(frame_index) + 1
    flags = np.zeros(2, dtype=np.float32)
    if nxt >= int(episode["num_frames"]):
        return flags
    for index, side in enumerate(("left", "right")):
        if wrist_measured(episode, side, frame_index) and wrist_measured(episode, side, nxt):
            flags[index] = 1.0
    return flags


def episode_action_valid(episode):
    """去掉最后一帧之后的 (T, 2)。没有下一步时 T 为 0。"""
    count = int(episode["num_frames"]) - 1
    if count < 1:
        return np.zeros((0, 2), dtype=np.float32)
    rows = [action_valid_flags(episode, index) for index in range(count)]
    return np.stack(rows, axis=0)


def validity_summary(action_valid, chunk_lengths=DEFAULT_CHUNK_LENGTHS):
    """有效动作比例，以及给定块长下「整段每只手都有效」的比例。

    块长比动作步还长时，比例是 None，计数是 0。不要把这种情况写成 0。
    """
    valid = np.asarray(action_valid, dtype=np.float64).reshape(-1, 2)
    action_count = int(valid.shape[0] * 2)
    valid_count = int(np.sum(valid >= 0.5))
    summary = {
        "valid_action_ratio": None if action_count == 0 else valid_count / float(action_count),
        "valid_action_count": valid_count,
        "action_count": action_count,
    }
    both = np.all(valid >= 0.5, axis=1) if valid.shape[0] else np.zeros(0, dtype=bool)
    flags = both.astype(np.int32)
    cumulative = np.cumsum(flags) if flags.size else flags
    for length in chunk_lengths:
        length = int(length)
        count = int(valid.shape[0] - length + 1)
        ratio_key = "valid_full_chunk_ratio_%d" % length
        if count <= 0:
            summary[ratio_key] = None
            summary["chunk_count_%d" % length] = 0
            summary["full_chunk_count_%d" % length] = 0
            continue
        ends = cumulative[length - 1:length - 1 + count]
        previous = np.zeros(count, dtype=np.int32)
        if count > 1:
            previous[1:] = cumulative[: count - 1]
        full = int(np.sum((ends - previous) == length))
        summary["chunk_count_%d" % length] = count
        summary["full_chunk_count_%d" % length] = full
        summary[ratio_key] = full / float(count)
    return summary


def validity_fields(episode, chunk_lengths=DEFAULT_CHUNK_LENGTHS):
    return validity_summary(episode_action_valid(episode), chunk_lengths)


def chunk_ratio_map(summary, chunk_lengths=DEFAULT_CHUNK_LENGTHS):
    return {int(length): summary.get("valid_full_chunk_ratio_%d" % int(length)) for length in chunk_lengths}


def action_dim_mask(action_valid, step):
    """(T, 2) → (T, step)。左手 7 维手腕，再右手；有关节增量时同样按手接在后面。"""
    valid = np.asarray(action_valid, dtype=np.float64).reshape(-1, 2)
    step = int(step)
    mask = np.zeros((valid.shape[0], step), dtype=np.float64)
    left = (valid[:, 0] >= 0.5).astype(np.float64)
    right = (valid[:, 1] >= 0.5).astype(np.float64)
    if step >= WRIST_DIM:
        mask[:, 0:WRIST_DIM] = left[:, None]
    if step >= 2 * WRIST_DIM:
        mask[:, WRIST_DIM:2 * WRIST_DIM] = right[:, None]
    joint = JOINTS * 3
    base = 2 * WRIST_DIM
    if step >= base + joint:
        mask[:, base:base + joint] = left[:, None]
    if step >= base + 2 * joint:
        mask[:, base + joint:base + 2 * joint] = right[:, None]
    return mask


def stack_chunk_mask(action_valid, indices, horizon, step):
    """把连续 horizon 步的按手掩码拼成和多步目标一样的一行。"""
    indices = np.asarray(indices, dtype=np.int64)
    horizon = int(horizon)
    step = int(step)
    mask = np.empty((len(indices), horizon * step), dtype=np.float64)
    for offset in range(horizon):
        block = action_dim_mask(np.asarray(action_valid)[indices + offset], step)
        mask[:, offset * step:(offset + 1) * step] = block
    return mask


def element_mask(episode_is_pad, action_valid, action_dim):
    """(T, action_dim)。片段末尾的 pad 清掉整步，无效手只清掉那只手的维。"""
    mask = action_dim_mask(action_valid, action_dim)
    if episode_is_pad is None:
        return mask
    pad = np.asarray(episode_is_pad, dtype=bool).reshape(-1)
    if pad.shape[0] != mask.shape[0]:
        raise ValueError("action_is_pad 长度 %d 和动作步 %d 不一致" % (pad.shape[0], mask.shape[0]))
    return mask * (~pad).astype(np.float64)[:, None]


def action_is_pad_from_valid(action_valid, episode_is_pad=None):
    """两只手都无效，或本来就是片段外的 pad，这一步才整步丢掉。

    只有一只手无效时不能整步 pad，否则另一只手的实测增量也不进损失。
    """
    valid = np.asarray(action_valid, dtype=np.float64).reshape(-1, 2)
    dropped = np.sum(valid >= 0.5, axis=1) == 0
    if episode_is_pad is not None:
        dropped = dropped | np.asarray(episode_is_pad, dtype=bool).reshape(-1)
    return dropped


def masked_mse(prediction, target, mask):
    """mask 为 0 的位置不进分子，也不进分母。全部无效时损失是 0。"""
    error = (np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)) ** 2
    if mask is None:
        return float(np.mean(error))
    weight = np.asarray(mask, dtype=np.float64)
    total = float(weight.sum())
    if total <= 0.0:
        return 0.0
    return float(np.sum(error * weight) / total)


def masked_l1(prediction, target, mask):
    error = np.abs(np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64))
    if mask is None:
        return float(np.mean(error))
    weight = np.asarray(mask, dtype=np.float64)
    total = float(weight.sum())
    if total <= 0.0:
        return 0.0
    return float(np.sum(error * weight) / total)


def chunk_valid_fraction(action_valid, start, horizon):
    window = np.asarray(action_valid, dtype=np.float64)[int(start):int(start) + int(horizon)]
    if window.size == 0:
        return 0.0
    return float(np.mean(window >= 0.5))


def filter_chunk_starts(starts, action_valid, horizon, min_fraction):
    """有效手-步比例低于 ``min_fraction`` 的动作块不采样。0 表示不过滤。"""
    starts = np.asarray(starts, dtype=np.int64)
    if action_valid is None or float(min_fraction) <= 0.0 or len(starts) == 0:
        return starts
    keep = [
        int(start)
        for start in starts
        if chunk_valid_fraction(action_valid, int(start), horizon) + 1e-12 >= float(min_fraction)
    ]
    return np.asarray(keep, dtype=np.int64)


def fully_valid(mask):
    if mask is None:
        return True
    array = np.asarray(mask)
    if array.size == 0:
        return True
    return bool(np.min(array) >= 1.0 - 1e-8)
