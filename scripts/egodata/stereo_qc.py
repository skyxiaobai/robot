# -*- coding: utf-8 -*-
"""双目 episode 的 QC：在 ``qc.py`` 的四类坏帧之外，再加三类双目专用的坏帧。

- ``stereo_one_view``：这一帧有手只在一目里被认到（另一目漏认或出画），没法三角化。
- ``stereo_inconsistent``：两目都认到了，但左右对不上（重投影误差大、深度或手掌尺寸不合理），
  被一致性检查拒掉。
- ``stereo_filled``：这一帧的手是时序补出来的（``filled=true``），不是测出来的。

逐帧信息来自 ``episode["stereo"]["per_frame"]``（``headcam/stereo_pipeline.py`` 写入）。
默认是“严格”口径：只要这一帧里**任何一只**被认到的手不合格，就算坏帧。
片段级规则和 ``qc.py`` 一样：坏帧比例超过 ``max_bad_fraction`` 就拒绝整段。
"""
import numpy as np

from egodata.qc import DEFAULTS, _FLAG_LABELS, STEREO_FLAG_LABELS, frame_qc_flags


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


def qc_stereo_episode(episode, **overrides):
    """返回与 ``qc.qc_episode`` 相同结构的结果，``flags`` 里多三类双目原因。"""
    options = dict(DEFAULTS)
    options.update(overrides)
    base = frame_qc_flags(episode, **overrides)
    stereo = stereo_frame_flags(episode)
    flags = dict(base)
    flags.update(stereo)
    num_frames = int(episode["num_frames"])
    fps = float(episode["fps"])
    counts = {name: int(np.sum(values)) for name, values in flags.items()}
    bad_mask = np.any(np.stack([flags[name] for name in flags]), axis=0)
    bad = int(np.sum(bad_mask))
    bad_fraction = bad / float(num_frames)
    accepted = bad_fraction <= options["max_bad_fraction"]
    # “只因这一条原因而坏”的帧数，用来看哪条规则最伤产出率
    only = {}
    for name in flags:
        others = [flags[o] for o in flags if o != name]
        rest = np.any(np.stack(others), axis=0) if others else np.zeros(num_frames, dtype=bool)
        only[name] = int(np.sum(flags[name] & ~rest))
    reasons = [] if accepted else [n for n in list(_FLAG_LABELS) + list(STEREO_FLAG_LABELS) if counts.get(n)]
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
        "good_frame_mask": [bool(not v) for v in bad_mask],
    }
