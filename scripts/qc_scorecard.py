#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""每段录像一张 QC 成绩单（JSON + 一行摘要 + 小图），不合格的段训练时用 --qc-filter 排除。

指标（每段）：
  move_frac     手在动的帧占比：任一只手 1 秒内手腕位移 > --move-cm（默认 5 cm）
  depth_valid   有深度源时：检测到手的帧里，深度检查通过（状态 ok）的比例；没有深度信息记 null
  detect_rate   至少一只手被认到的帧占比
  jitter_cm     手腕逐帧位移的中位数（cm，原生帧率；越小越稳）
  swaps         左右手互换次数（egodata.hand_fix 按手腕连续性检出）
  gap_ratio     超过 0.3 s 的长缺口占的帧比例（两只手平均）
  frame_ok      坐标单位 / 轴向检查（common.frames.sanity_check）

合格规则（默认，可改）：detect_rate ≥ 0.5、gap_ratio ≤ 0.5、jitter_cm ≤ 2.5、move_frac ≥ 0.05、
swaps ≤ 每分钟 10 次、depth_valid ≥ 0.5（有深度时）、frame_ok。

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
        det |= detected
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
                if s in ("none", "missing", None):
                    continue
                depth_all += 1; depth_ok += int(s == "ok")
    st = np.concatenate(residuals) if residuals else np.array([])
    gap = np.mean([fixed[s]["gap_frames"] / max(n, 1) for s in SIDES])
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
    if c["depth_valid"] is None and any(k in str(c.get("source") or "").lower() for k in ("stera", "iphone_lidar", "dynamic_intelligence", "zeno_lidar")):
        f.append("深度状态未知")
    elif c["depth_valid"] is not None and c["depth_valid"] < r["depth_valid_min"]:
        f.append("深度有效率低")
    if not c["frame_ok"]:
        f.append("坐标轴/单位不对")
    return f


def one_line(c):
    dv = "—" if c["depth_valid"] is None else "%.0f%%" % (100 * c["depth_valid"])
    return "%s %s | 动 %.0f%% 深度 %s 认手 %.0f%% 抖 %.2fcm 换手 %d 缺口 %.0f%% | %s" % (
        "✅" if c["pass"] else "❌", c["clip"], 100 * c["move_frac"], dv, 100 * c["detect_rate"],
        c["jitter_cm"] if c["jitter_cm"] is not None else float("nan"), c["swaps"], 100 * c["gap_ratio"],
        "合格" if c["pass"] else "、".join(c["fails"]))


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


def load_filter(path):
    """--qc-filter：scorecards.json → 不合格段的名字集合（clip 名 = episode 文件名去掉 .json）。"""
    data = json.loads(Path(path).read_text())
    cards = data["cards"] if isinstance(data, dict) else data
    return {c["clip"] for c in cards if not c["pass"]}


def main():
    from egodata.schema import iter_episode_paths
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", nargs="+", required=True, help="episode JSON 目录或文件")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="QC")
    ap.add_argument("--move-cm", type=float, default=5.0)
    ap.add_argument("--max-gap-s", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=None)
    for k, v in RULES.items():
        ap.add_argument("--" + k.replace("_", "-"), type=float, default=v)
    a = ap.parse_args()
    rules = {k: getattr(a, k) for k in RULES}
    out = Path(a.out); (out / "clips").mkdir(parents=True, exist_ok=True)
    paths = [p for root in a.episodes for p in iter_episode_paths(root)][: a.limit]
    cards = []
    for p in paths:
        ep = json.loads(p.read_text())
        if "hands" not in ep:
            continue
        c = score_episode(ep, p.stem, a.move_cm, a.max_gap_s, rules)
        (out / "clips" / (p.stem + ".json")).write_text(json.dumps(c, ensure_ascii=False, indent=1))
        cards.append(c); print(one_line(c), flush=True)
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
    print("pass %d/%d" % (summ["pass"], len(cards)), json.dumps(fails, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
