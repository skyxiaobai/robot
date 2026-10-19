#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""每段录像一张 QC 成绩单（JSON + 一行摘要 + 小图），不合格的段训练时用 --qc-filter 排除。

指标（每段）：
  move_frac     手在动的帧占比：任一只手 1 秒内手腕位移 > --move-cm（默认 5 cm）
  depth_valid   有逐帧深度状态时：检测到手的帧里，深度检查通过（状态 ok）的比例。
                没有逐帧证据记 null，并报 unknown（警告，不判失败）；不再把缺失证据当成 100%。
                只有 lowconf / fit / depth 这类实测坏深度才会拉低比例并判失败。
  detect_rate   至少一只手被认到的帧占比
  jitter_cm     手腕逐帧位移的中位数（cm，原生帧率；越小越稳）
  swaps         左右手互换次数（egodata.hand_fix 按手腕连续性检出）
  gap_ratio     超过 0.3 s 的长缺口占的帧比例（两只手平均）
  frame_ok      坐标单位 / 轴向检查（common.frames.sanity_check）

合格规则（默认，可改）：detect_rate ≥ 0.5、gap_ratio ≤ 0.5、jitter_cm ≤ 2.5、move_frac ≥ 0.05、
swaps ≤ 每分钟 10 次、depth_valid ≥ 0.5（有实测深度时）、frame_ok。
stera / dynamic_intelligence / zeno_lidar / iphone_lidar / hoi4d 若转换器没写逐帧深度状态，记 unknown 警告，仍可合格。

用法：
  python scripts/qc_scorecard.py --episodes data/stera/unified --out outputs/qc/stera
  python scripts/train_xhey.py ... --qc-filter outputs/qc/stera/scorecards.json
"""
import argparse, copy, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import frames as F  # noqa: E402
from egodata.hand_fix import SIDES, _good, fix_episode_hands  # noqa: E402

DEPTH_BAD = {"lowconf", "fit", "depth"}
DEPTH_UNKNOWN = {"none", "missing", "unknown"}
# 这些源的转换器经常不写逐帧深度状态；缺失证据报 unknown，不当作失败，也不再记成 100%。
DEPTH_UNKNOWN_SOURCES = ("stera", "dynamic_intelligence", "zeno_lidar", "iphone_lidar", "hoi4d")
# 只标注交互手的数据集（HOI4D）：另一只手整段缺失不是质量问题。
SINGLE_HAND_SOURCES = ("hoi4d",)
RULES = {"detect_rate_min": 0.5, "gap_ratio_max": 0.5, "jitter_cm_max": 2.5, "move_frac_min": 0.05,
         "swaps_per_min_max": 10.0, "depth_valid_min": 0.5}


def _has_depth_status(ep):
    pose = ep.get("hand_pose") or {}
    return "lidar" in str(pose.get("backend", "")) and isinstance((ep.get("iphone") or {}).get("per_frame"), dict)


def score_episode(ep, name="", move_cm=5.0, max_gap_s=0.3, rules=RULES):
    n = int(ep["num_frames"]); fps = float(ep.get("fps") or 15.0)
    had_fix = isinstance(ep.get("hand_fix"), dict)
    e2 = ep if had_fix else copy.deepcopy(ep)
    fixed = e2["hand_fix"] if had_fix else fix_episode_hands(e2, fps, max_gap_s=max_gap_s)
    det = np.zeros(n, bool); residuals = []; moving = np.zeros(n, bool); one_s = max(1, int(round(fps)))
    depth_ok = depth_all = 0
    for side in SIDES:
        h = e2["hands"][side]
        ok, W = _good(h, n, 0.5)
        st = h.get("label_status") or [""] * n
        detected = np.isfinite(W).all(1) & np.array([str(s) not in ("missing", "none_detected") for s in st[:n]])
        det |= ok   # QC detect_rate requires genuinely measured, unfilled, trusted wrist
        Wg = np.where(ok[:, None], W, np.nan)
        if n >= 3:
            residual = np.linalg.norm(Wg[2:] - 2 * Wg[1:-1] + Wg[:-2], axis=1) * (15.0 / fps) ** 2
            residuals.append(residual[np.isfinite(residual)])
        if n > one_s:
            mv = np.linalg.norm(Wg[one_s:] - Wg[:-one_s], axis=1)
            moving[:-one_s] |= np.nan_to_num(mv, nan=0.0) > move_cm / 100.0
    if _has_depth_status(ep):
        for side in SIDES:
            for s in ep["iphone"]["per_frame"][side]:
                token = "" if s is None else str(s).lower()
                if token in DEPTH_UNKNOWN:
                    continue
                # 只有 ok 和明确的坏深度才算证据；unknown / 没写状态不算失败，也不记成 100%。
                if token == "ok" or token in DEPTH_BAD:
                    depth_all += 1
                    depth_ok += int(token == "ok")
    st = np.concatenate(residuals) if residuals else np.array([])
    gsides = list(SIDES)
    if str(ep.get("source")) in SINGLE_HAND_SOURCES:  # 只标注了交互手：整段没标的那只手不算缺口
        gsides = [s for s in SIDES if fixed[s]["gap_frames"] < n] or gsides
    gap = np.mean([fixed[s]["gap_frames"] / max(n, 1) for s in gsides])
    sanity = F.sanity_check(ep)
    card = {"clip": name or ep.get("episode_id"), "source": ep.get("source"), "frames": n, "fps": round(fps, 2),
            "seconds": round(n / fps, 1), "convention": F.convention_of(ep),
            "move_frac": round(float(moving.mean()), 3),
            "depth_valid": round(depth_ok / depth_all, 3) if depth_all else None,
            "detect_rate": round(float(det.mean()), 3),
            "jitter_cm": round(float(np.median(st)) * 100, 3) if len(st) else None,
            "swaps": int(fixed["swaps"]), "gap_ratio": round(float(gap), 3),
            "interp_frames": int(sum(fixed[s]["interp_frames"] for s in SIDES)),
            "frame_ok": bool(sanity["ok"]), "frame_flags": sanity["flags"], "median_hand_dist_m": sanity["median_dist_m"]}
    card["fails"] = check(card, rules)
    card["warnings"] = depth_warnings(card)
    card["pass"] = not card["fails"]
    return card


def check(c, r=RULES):
    f = []
    if c["detect_rate"] < r["detect_rate_min"]:
        f.append("认手率低")
    if c["gap_ratio"] > r["gap_ratio_max"]:
        f.append("长缺口多")
    if c["jitter_cm"] is None or c["jitter_cm"] > r["jitter_cm_max"]:
        f.append("抖动大")
    if c["move_frac"] < r["move_frac_min"]:
        f.append("手几乎不动")
    if c["swaps"] / max(c["seconds"] / 60.0, 1e-6) > r["swaps_per_min_max"] and c["swaps"] > 2:
        f.append("左右手常认反")
    if c["depth_valid"] is not None and c["depth_valid"] < r["depth_valid_min"]:
        f.append("深度有效率低")
    if not c["frame_ok"]:
        f.append("坐标轴/单位不对")
    return f


def _depth_source(c):
    src = str(c.get("source") or "").lower()
    return any(k in src for k in DEPTH_UNKNOWN_SOURCES)


def depth_warnings(c):
    """没有实测深度证据时报告 unknown。警告不进入 fails，合格与否只看实测坏深度。"""
    if c.get("depth_valid") is None and _depth_source(c):
        return ["深度状态未知"]
    return []


def one_line(c):
    warns = c.get("warnings")
    if warns is None:
        warns = depth_warnings(c)
    if c["depth_valid"] is None and "深度状态未知" in warns:
        dv = "unknown"
    elif c["depth_valid"] is None:
        dv = "—"
    else:
        dv = "%.0f%%" % (100 * c["depth_valid"])
    tail = "合格" if c["pass"] else "、".join(c["fails"])
    if warns:
        tail += "（警告：" + "、".join(warns) + "）"
    return "%s %s | 动 %.0f%% 深度 %s 认手 %.0f%% 抖 %.2fcm 换手 %d 缺口 %.0f%% | %s" % (
        "✅" if c["pass"] else "❌", c["clip"], 100 * c["move_frac"], dv, 100 * c["detect_rate"],
        c["jitter_cm"] if c["jitter_cm"] is not None else float("nan"), c["swaps"], 100 * c["gap_ratio"],
        tail)


def chart(cards, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for fname in ("Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei"):
        try:
            from matplotlib import font_manager
            font_manager.findfont(fname, fallback_to_default=False)
            plt.rcParams["font.family"] = fname; break
        except Exception:  # noqa: BLE001
            continue
    # 颜色 = 「好不好」：绿好红差。动/深度/认手越大越好；抖动、缺口越小越好（按合格线归一）
    keys = [("move_frac", "move", lambda v: min(1.0, v / 0.3)), ("depth_valid", "depth", lambda v: v),
            ("detect_rate", "detect", lambda v: v), ("jitter_cm", "jitter cm", lambda v: max(0.0, 1 - v / 5.0)),
            ("gap_ratio", "gap", lambda v: max(0.0, 1 - v)), ("swaps", "swaps", lambda v: max(0.0, 1 - v / 50.0))]
    M = np.array([[np.nan if c[k] is None else f(c[k]) for k, _, f in keys] for c in cards])
    h = max(2.5, 0.22 * len(cards) + 1.2)
    fig, ax = plt.subplots(figsize=(7, h))
    ax.imshow(M, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    for j, (k, _, _) in enumerate(keys):
        for i, c in enumerate(cards):
            v = c[k]
            ax.text(j, i, "-" if v is None else (str(v) if k == "swaps" else "%.2f" % v), ha="center", va="center", fontsize=6)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels([l for _, l, _ in keys], fontsize=7)
    ax.set_yticks(range(len(cards)))
    ax.set_yticklabels([("+ " if c["pass"] else "x ") + str(c["clip"])[-28:] for c in cards], fontsize=6)
    ax.set_title(title + "  pass %d/%d" % (sum(c["pass"] for c in cards), len(cards)), fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def load_scored(path):
    data = json.loads(Path(path).read_text())
    cards = data["cards"] if isinstance(data, dict) else data
    return {str(c["clip"]) for c in cards}


def load_filter(path):
    """--qc-filter：scorecards.json → 不合格段的名字集合（clip 名 = episode 文件名去掉 .json）。"""
    data = json.loads(Path(path).read_text())
    cards = data["cards"] if isinstance(data, dict) else data
    return {c["clip"] for c in cards if not c["pass"]}


# ---------------- 分段模式（--segments）：按小窗口打分，剪掉坏的那几截，好的留下 ----------------
SEG_S = 3.0  # 默认 3 s 一窗：15 fps 下 45 帧 ≈ 2.8 个 16 步预测块，块能完整落在一两个窗口里


def _frame_signals(ep, move_cm=5.0, max_gap_s=0.3):
    """整段一次性算好逐帧信号，窗口打分只做切片统计（和整段规则同源）。"""
    n = int(ep["num_frames"]); fps = float(ep.get("fps") or 15.0)
    e2 = copy.deepcopy(ep)
    fixed = e2["hand_fix"] if isinstance(e2.get("hand_fix"), dict) else fix_episode_hands(e2, fps, max_gap_s=max_gap_s)
    det = np.zeros(n, bool); resid = np.full((2, n), np.nan); moving = np.zeros(n, bool)
    gap = np.zeros((2, n), bool); one_s = max(1, int(round(fps))); gmin = max(1, int(round(max_gap_s * fps)))
    for k, side in enumerate(SIDES):
        ok, W = _good(e2["hands"][side], n, 0.5)
        det |= ok
        Wg = np.where(ok[:, None], W, np.nan)
        if n >= 3:
            resid[k, 1:-1] = np.linalg.norm(Wg[2:] - 2 * Wg[1:-1] + Wg[:-2], axis=1) * (15.0 / fps) ** 2
        if n > one_s:
            mv = np.linalg.norm(Wg[one_s:] - Wg[:-one_s], axis=1)
            moving[:-one_s] |= np.nan_to_num(mv, nan=0.0) > move_cm / 100.0
        i = 0
        while i < n:  # 长缺口：连续丢手超过 max_gap_s
            if not ok[i]:
                j = i
                while j < n and not ok[j]:
                    j += 1
                if j - i > gmin:
                    gap[k, i:j] = True
                i = j
            else:
                i += 1
    swap = np.zeros(n, int)
    for f in fixed.get("swap_frames") or []:
        if 0 <= int(f) < n:
            swap[int(f)] += 1
    dep_ok = np.zeros(n, int); dep_all = np.zeros(n, int)
    if _has_depth_status(ep):
        for side in SIDES:
            for i, s in enumerate(ep["iphone"]["per_frame"][side][:n]):
                t = "" if s is None else str(s).lower()
                if t == "ok" or t in DEPTH_BAD:
                    dep_all[i] += 1; dep_ok[i] += int(t == "ok")
    return {"n": n, "fps": fps, "det": det, "resid": resid, "moving": moving, "gap": gap, "swap": swap,
            "dep_ok": dep_ok, "dep_all": dep_all}


def window_fails(sig, a, b, rules=RULES, clip_still=False, frame_ok=True):
    """窗口 [a, b) 的 7 条规则（窗口版）。手不动只在整段都不动时才判失败；换手按窗口内次数：
    超过 max(1, 每分钟上限×窗口分钟) 才判失败（3 s 窗即 ≥2 次；单次已被修正的认反放过）。"""
    f = []
    det = sig["det"][a:b]
    if det.mean() < rules["detect_rate_min"]:
        f.append("认手率低")
    if sig["gap"][:, a:b].mean() > rules["gap_ratio_max"]:
        f.append("长缺口多")
    r = sig["resid"][:, a:b]; r = r[np.isfinite(r)]
    if len(r) < 3 or float(np.median(r)) * 100 > rules["jitter_cm_max"]:
        if "认手率低" not in f:
            f.append("抖动大")
    if clip_still:
        f.append("手几乎不动")
    win_min = (b - a) / sig["fps"] / 60.0
    if sig["swap"][a:b].sum() > max(1.0, rules["swaps_per_min_max"] * win_min):
        f.append("左右手常认反")
    da = sig["dep_all"][a:b].sum()
    if da and sig["dep_ok"][a:b].sum() / da < rules["depth_valid_min"]:
        f.append("深度有效率低")
    if not frame_ok:
        f.append("坐标轴/单位不对")
    return f


def segment_episode(ep, name="", seg_s=SEG_S, move_cm=5.0, max_gap_s=0.3, rules=RULES):
    """整段切成 seg_s 秒的窗口逐个打分；连续合格的窗口合并成保留区间 spans=[[起, 止), ...]（原生帧号）。"""
    sig = _frame_signals(ep, move_cm, max_gap_s)
    n, fps = sig["n"], sig["fps"]
    w = max(2, int(round(seg_s * fps)))
    edges = list(range(0, n, w)) + [n]
    if len(edges) > 2 and edges[-1] - edges[-2] < w // 2:  # 尾巴太短并进上一窗
        edges.pop(-2)
    clip_still = float(sig["moving"].mean()) < rules["move_frac_min"]
    frame_ok = bool(F.sanity_check(ep)["ok"])
    windows, spans = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        fails = window_fails(sig, a, b, rules, clip_still, frame_ok)
        windows.append({"start": a, "end": b, "fails": fails})
        if not fails:
            if spans and spans[-1][1] == a:
                spans[-1][1] = b
            else:
                spans.append([a, b])
    kept = sum(b - a for a, b in spans)
    return {"clip": name or ep.get("episode_id"), "source": ep.get("source"), "frames": n, "fps": round(fps, 3),
            "seconds": round(n / fps, 2), "window_frames": w, "spans": spans, "windows": windows,
            "kept_frames": int(kept), "kept_seconds": round(kept / fps, 2), "kept_frac": round(kept / max(n, 1), 4)}


def load_segments(paths):
    """--qc-segments：segments.json（可多个）→ {clip: {"fps", "frames", "spans"}}。"""
    out = {}
    for p in paths:
        data = json.loads(Path(p).read_text())
        for c in (data["clips"] if isinstance(data, dict) else data):
            out[str(c["clip"])] = {"fps": float(c["fps"]), "frames": int(c["frames"]), "spans": c["spans"]}
    return out


def segment_frame_mask(entry, n_frames=None, step=1):
    """保留区间 → 逐帧布尔掩码；step>1 时对应下采样后的帧（第 i 帧 = 原第 i*step 帧）。"""
    m = np.zeros(int(entry["frames"]), bool)
    for a, b in entry["spans"]:
        m[int(a):int(b)] = True
    m = m[::step]
    if n_frames is not None:
        m = np.concatenate([m, np.zeros(max(0, n_frames - len(m)), bool)])[:n_frames]
    return m


def main():
    from egodata.schema import iter_episode_paths
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", nargs="+", required=True, help="episode JSON 目录或文件")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="QC")
    ap.add_argument("--move-cm", type=float, default=5.0)
    ap.add_argument("--max-gap-s", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--segments", action="store_true",
                    help="另外按小窗口打分，输出 segments.json（每段保留区间），训练用 --qc-segments")
    ap.add_argument("--seg-s", type=float, default=SEG_S, help="分段模式窗口长度（秒），默认 3")
    for k, v in RULES.items():
        ap.add_argument("--" + k.replace("_", "-"), type=float, default=v)
    a = ap.parse_args()
    rules = {k: getattr(a, k) for k in RULES}
    out = Path(a.out); (out / "clips").mkdir(parents=True, exist_ok=True)
    paths = [p for root in a.episodes for p in iter_episode_paths(root)][: a.limit]
    cards, segs = [], []
    for p in paths:
        ep = json.loads(p.read_text())
        if "hands" not in ep:
            continue
        c = score_episode(ep, p.stem, a.move_cm, a.max_gap_s, rules)
        (out / "clips" / (p.stem + ".json")).write_text(json.dumps(c, ensure_ascii=False, indent=1))
        cards.append(c); print(one_line(c), flush=True)
        if a.segments:
            sg = segment_episode(ep, p.stem, a.seg_s, a.move_cm, a.max_gap_s, rules)
            sg["whole_clip_pass"] = c["pass"]; segs.append(sg)
            print("   分段：保留 %.0f%%（%d 个区间）" % (100 * sg["kept_frac"], len(sg["spans"])), flush=True)
    summ = {"rules": rules, "n": len(cards), "pass": sum(c["pass"] for c in cards), "cards": cards}
    for k in ("move_frac", "depth_valid", "detect_rate", "jitter_cm", "gap_ratio", "swaps"):
        v = [c[k] for c in cards if c[k] is not None]
        summ["median_" + k] = float(np.median(v)) if v else None
    fails = {}
    for c in cards:
        for f in c["fails"]:
            fails[f] = fails.get(f, 0) + 1
    summ["fail_reasons"] = fails
    (out / "scorecards.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1))
    (out / "summary.txt").write_text("\n".join(one_line(c) for c in cards) + "\n", encoding="utf-8")
    if cards:
        show = cards if len(cards) <= 40 else sorted(cards, key=lambda c: c["pass"])[:20] + [c for c in cards if c["pass"]][:20]
        chart(show, out / "scorecard.png", a.title)
    if a.segments:
        tot = sum(g["seconds"] for g in segs)
        seg_summ = {"seg_s": a.seg_s, "rules": rules, "n": len(segs), "total_hours": round(tot / 3600, 4),
                    "segment_hours": round(sum(g["kept_seconds"] for g in segs) / 3600, 4),
                    "whole_clip_hours": round(sum(g["seconds"] for g in segs if g["whole_clip_pass"]) / 3600, 4),
                    "clips": segs}
        (out / "segments.json").write_text(json.dumps(seg_summ, ensure_ascii=False, indent=1))
        print("分段保留 %.3f h / 整段过滤 %.3f h / 不过滤 %.3f h" % (
            seg_summ["segment_hours"], seg_summ["whole_clip_hours"], seg_summ["total_hours"]), flush=True)
    print("pass %d/%d" % (summ["pass"], len(cards)), json.dumps(fails, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
