# -*- coding: utf-8 -*-
"""双目 episode 的 QC。

片段会不会被拒绝，只看真正的片段问题，也就是 ``qc.py`` 的四类坏帧
（手出画、视线飘移、运动模糊、摆拍或静止）。坏帧比例超过 ``max_bad_fraction``
（默认 20%）就拒绝整段。

立体门限丢掉的帧不是坏帧，而是丢掉的标注，排除在坏帧之外，单独用标注覆盖率报告：

- ``stereo_one_view``：这一帧有手只在一目里被认到，没法三角化。
- ``stereo_inconsistent``：左右对不上，或被速度门限、严格门限丢掉。
  状态包括 ``reproj`` / ``depth`` / ``palm`` / ``few_joints`` / ``jump`` / ``strict``。
- ``stereo_filled``：这一帧的手是时序补出来的（``filled=true``），不是测出来的。

只要这一帧里任何一只手落进上面三类，这一帧就算丢掉标注。门限把手腕拿掉之后，
基础 QC 会把「两只手都没有手腕」看成手出画，也会把长段缺测看成摆拍；这种帧已经算丢掉的标注，
不再记成手出画或静止。

标注覆盖率 = 仍有实测标注的帧数 / 总帧数。``min_label_coverage`` 是单独的片段门限，
默认 ``0``：只把覆盖率写进报告，不因此拒绝片段。这个数还没有真实头戴数据，是临时值，
等自采录像再定，不要当成已经标定过的门限。
"""
import numpy as np

from egodata.qc import DEFAULTS, _FLAG_LABELS, STEREO_FLAG_LABELS, frame_qc_flags

# 临时值。0 = 报告标注覆盖率，但不因此拒绝片段。等真实设备数据再改。
DEFAULT_MIN_LABEL_COVERAGE = 0.0


def stereo_frame_flags(episode):
    num_frames = int(episode["num_frames"])
    flags = {name: np.zeros(num_frames, dtype=bool) for name in STEREO_FLAG_LABELS}
    per_frame = (episode.get("stereo") or {}).get("per_frame") or {}
    for side in ("left", "right"):
        status = per_frame.get(side) or []
        for index, value in enumerate(status[:num_frames]):
            if value == "one_view":
                flags["stereo_one_view"][index] = True
            elif value not in (None, "none", "ok"):
                flags["stereo_inconsistent"][index] = True
        filled = episode["hands"][side].get("filled") or []
        for index, value in enumerate(filled[:num_frames]):
            if value:
                flags["stereo_filled"][index] = True
    return flags


def _dropped_label_mask(stereo_flags, num_frames):
    dropped = np.zeros(num_frames, dtype=bool)
    for name in STEREO_FLAG_LABELS:
        dropped |= stereo_flags[name]
    return dropped


def qc_stereo_episode(episode, **overrides):
    """返回与 ``qc.qc_episode`` 相同结构的结果，并另报标注覆盖率。

    ``flags`` 仍含三类双目原因，但它们不进入 ``bad_frames``。
    ``min_label_coverage`` 可覆盖默认的临时下限。
    """
    options = dict(DEFAULTS)
    options["min_label_coverage"] = DEFAULT_MIN_LABEL_COVERAGE
    options.update(overrides)
    base = frame_qc_flags(episode, **overrides)
    stereo = stereo_frame_flags(episode)
    num_frames = int(episode["num_frames"])
    dropped = _dropped_label_mask(stereo, num_frames)
    # 丢掉标注后手腕没了。不能据此再说手出画，也不能把这段缺测当成摆拍或静止。
    base["hands_out_of_frame"] = np.asarray(base["hands_out_of_frame"]) & ~dropped
    base["staged_static"] = np.asarray(base["staged_static"]) & ~dropped
    flags = dict(base)
    flags.update(stereo)
    fps = float(episode["fps"])
    counts = {name: int(np.sum(values)) for name, values in flags.items()}
    bad_mask = np.any(np.stack([base[name] for name in _FLAG_LABELS]), axis=0)
    bad = int(np.sum(bad_mask))
    bad_fraction = bad / float(num_frames)
    labeled = int(np.sum(~dropped))
    label_coverage = labeled / float(num_frames)
    min_label_coverage = float(options["min_label_coverage"])
    fraction_ok = bad_fraction <= options["max_bad_fraction"]
    coverage_ok = label_coverage + 1e-12 >= min_label_coverage
    accepted = bool(fraction_ok and coverage_ok)
    only = {}
    for name in flags:
        others = [flags[o] for o in flags if o != name]
        rest = np.any(np.stack(others), axis=0) if others else np.zeros(num_frames, dtype=bool)
        only[name] = int(np.sum(flags[name] & ~rest))
    reasons = []
    if not fraction_ok:
        reasons.extend(name for name in _FLAG_LABELS if counts.get(name))
    if not coverage_ok:
        reasons.append("low_label_coverage")
    kept = ~dropped & ~bad_mask
    return {
        "episode_id": episode.get("episode_id", ""),
        "num_frames": num_frames,
        "fps": fps,
        "duration_s": num_frames / fps,
        "accepted": accepted,
        "bad_frames": bad,
        "bad_fraction": bad_fraction,
        "flags": counts,
        "flags_only_reason": only,
        "reasons": reasons,
        "good_frame_mask": [bool(v) for v in kept],
        "labeled_frames": labeled,
        "dropped_label_frames": int(np.sum(dropped)),
        "label_coverage": label_coverage,
        "min_label_coverage": min_label_coverage,
    }
