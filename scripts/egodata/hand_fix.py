# -*- coding: utf-8 -*-
"""WiLoR 认手之后的「左右手 + 遮挡」修正（逐 episode，离线）。

1. 左右手互换：WiLoR 偶尔把左手认成右手。按手腕连续性判断：这一帧「左→上次左、右→上次右」
   的距离和，比「交叉」的距离和大很多（且超过跳变阈值）时，认为认反了，交换这一帧两只手的全部字段。
2. 遮挡 / 低置信：置信度 < min_conf，或标注状态不是 ok（lowconf / fit / depth / tracking …）的帧当成缺测。
3. 补帧：两端都是好帧、长度 ≤ max_gap_s（默认 0.3 s）的缺口，手腕位置线性插值、朝向 nlerp，
   标 ``filled=True``、``label_status="interp"``（输入里有合理的值，但 action_valid 仍为 0，不当监督）。
   更长的缺口：``valid=False``、``label_status="gap"`` → hand_valid / action_valid 都为 0。

结果摘要写在 ``episode["hand_fix"]``，QC 成绩单直接读它。
"""
import numpy as np

SIDES = ("left", "right")
OK_STATUS = {"", "ok", "none"}


def _wrist(hand, n):
    W = np.full((n, 3), np.nan)
    for i, p in enumerate(hand["wrist_pose"][:n]):
        if p and len(p) >= 3 and all(v is not None for v in p[:3]):
            W[i] = p[:3]
    return W


def _good(hand, n, min_conf):
    W = _wrist(hand, n)
    ok = np.isfinite(W).all(1)
    conf = hand.get("confidence") or [None] * n
    st = hand.get("label_status") or [None] * n
    filled = hand.get("filled") or [False] * n
    for i in range(n):
        if not ok[i]:
            continue
        c = conf[i] if i < len(conf) else None
        s = st[i] if i < len(st) else None
        if (c is not None and float(c) < min_conf) or (isinstance(s, str) and s.strip().lower() not in OK_STATUS) \
                or (i < len(filled) and filled[i]):
            ok[i] = False
    return ok, W


def _swap_frame(ep, n, i):
    L, R = ep["hands"]["left"], ep["hands"]["right"]
    for k in set(L) & set(R):
        if isinstance(L[k], list) and isinstance(R[k], list) and len(L[k]) == n == len(R[k]):
            L[k][i], R[k][i] = R[k][i], L[k][i]
    for parent in ("stereo", "iphone"):
        pf = (ep.get(parent) or {}).get("per_frame") if isinstance(ep.get(parent), dict) else None
        if isinstance(pf, dict) and all(isinstance(pf.get(s), list) and len(pf[s]) == n for s in SIDES):
            pf["left"][i], pf["right"][i] = pf["right"][i], pf["left"][i]


def detect_swaps(WL, WR, fps, jump_m=0.08, ratio=0.5, max_age_s=0.5):
    """返回需要交换的帧号。WL/WR: (n,3)，缺测为 nan。只用前面的帧（因果），交换后再更新记忆。"""
    n = len(WL); max_age = max(1, int(round(max_age_s * fps)))
    last = {0: None, 1: None}; age = {0: 10 ** 9, 1: 10 ** 9}; out = []
    for i in range(n):
        cur = [WL[i] if np.isfinite(WL[i]).all() else None, WR[i] if np.isfinite(WR[i]).all() else None]
        ref = [last[k] if age[k] <= max_age else None for k in (0, 1)]

        def d(a, b):
            return float(np.linalg.norm(a - b)) if a is not None and b is not None else None
        keep = [d(cur[0], ref[0]), d(cur[1], ref[1])]
        cross = [d(cur[0], ref[1]), d(cur[1], ref[0])]
        swap = False
        if all(v is not None for v in keep + cross):
            k, c = sum(keep), sum(cross)
            swap = c < ratio * k and max(keep) > jump_m
        elif cur[0] is not None and cur[1] is None and keep[0] is not None and cross[0] is not None:
            swap = cross[0] < ratio * keep[0] and keep[0] > jump_m
        elif cur[1] is not None and cur[0] is None and keep[1] is not None and cross[1] is not None:
            swap = cross[1] < ratio * keep[1] and keep[1] > jump_m
        if swap:
            out.append(i); cur = cur[::-1]
        for k in (0, 1):
            if cur[k] is not None:
                last[k], age[k] = cur[k], 0
            else:
                age[k] += 1
    return out


def _runs(mask):
    """mask 为 True 的连续段 [(s, e)) 。"""
    runs, s = [], None
    for i, m in enumerate(list(mask) + [False]):
        if m and s is None:
            s = i
        elif not m and s is not None:
            runs.append((s, i)); s = None
    return runs


def _nlerp(q0, q1, a):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    if q0 @ q1 < 0:
        q1 = -q1
    q = (1 - a) * q0 + a * q1
    return q / (np.linalg.norm(q) or 1.0)


def fix_episode_hands(ep, fps=None, max_gap_s=0.3, min_conf=0.5, fix_swaps=True, jump_m=0.08):
    """原地修正 episode，返回摘要 dict（也写进 ep["hand_fix"]）。"""
    n = int(ep["num_frames"]); fps = float(fps or ep.get("fps") or 15.0)
    swaps = []
    if fix_swaps:
        okL, WL = _good(ep["hands"]["left"], n, min_conf)
        okR, WR = _good(ep["hands"]["right"], n, min_conf)
        swaps = detect_swaps(np.where(okL[:, None], WL, np.nan), np.where(okR[:, None], WR, np.nan), fps, jump_m)
        for i in swaps:
            _swap_frame(ep, n, i)
    max_gap = max(1, int(round(max_gap_s * fps)))
    summary = {"swaps": len(swaps), "swap_frames": swaps[:200], "max_gap_frames": max_gap, "max_gap_s": max_gap_s,
               "min_conf": min_conf}
    for side in SIDES:
        h = ep["hands"][side]
        for k, default in (("filled", False), ("valid", False), ("label_status", "missing"), ("confidence", None)):
            if not isinstance(h.get(k), list) or len(h[k]) != n:
                h[k] = [default] * n
        ok, W = _good(h, n, min_conf)
        detected = np.isfinite(W).all(1)
        occluded = int((detected & ~ok).sum())
        interp = gap = 0
        idx = np.flatnonzero(ok)
        for s, e in _runs(~ok):
            bounded = s > 0 and e < n and ok[s - 1] and ok[e]
            if bounded and e - s <= max_gap:
                a, b = s - 1, e
                pa, pb = h["wrist_pose"][a], h["wrist_pose"][b]
                ja = np.asarray(h["joints"][a], float) if h.get("joints") else None
                jb = np.asarray(h["joints"][b], float) if h.get("joints") else None
                for i in range(s, e):
                    t = (i - a) / float(b - a)
                    pos = ((1 - t) * np.asarray(pa[:3], float) + t * np.asarray(pb[:3], float)).tolist()
                    q = pa[3:7]
                    if len(pa) >= 7 and len(pb) >= 7 and None not in pa[3:7] and None not in pb[3:7]:
                        q = _nlerp(pa[3:7], pb[3:7], t).tolist()
                    h["wrist_pose"][i] = pos + list(q)
                    if ja is not None and jb is not None and ja.shape == jb.shape and np.isfinite(ja).all() and np.isfinite(jb).all():
                        h["joints"][i] = ((1 - t) * ja + t * jb).tolist()
                    h["filled"][i] = True; h["valid"][i] = True; h["label_status"][i] = "interp"
                    interp += 1
            else:
                for i in range(s, e):
                    h["valid"][i] = False; h["label_status"][i] = "gap"
                    gap += 1
        for i in idx:
            h["valid"][i] = True
        summary[side] = {"good": int(ok.sum()), "detected": int(detected.sum()), "occluded_or_lowconf": occluded,
                         "interp_frames": interp, "gap_frames": gap}
    for parent in ("stereo", "iphone"):  # per_frame 状态跟 label_status 对齐，避免旧 lowconf 状态把补帧又打掉
        pf = (ep.get(parent) or {}).get("per_frame") if isinstance(ep.get(parent), dict) else None
        if isinstance(pf, dict):
            for side in SIDES:
                if isinstance(pf.get(side), list) and len(pf[side]) == n:
                    for i, s in enumerate(ep["hands"][side]["label_status"]):
                        if s in ("interp", "gap"):
                            pf[side][i] = s
    ep["hand_fix"] = summary
    return summary
