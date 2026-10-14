"""XHEY 训练报告：每次跑完出一组中文图表 + 全部实验的汇总表。

  python scripts/xhey_report.py run --metrics path/metrics.json --name C8   # 单次实验
  python scripts/xhey_report.py summary                                    # 汇总（读 experiments/ledger.jsonl）
  python scripts/xhey_report.py all                                        # experiments/runs/** + docs/xhey/*metrics.json + 汇总

输出：experiments/report/<run>/*.png 和 experiments/report/summary.png / summary.csv
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "experiments" / "report"
LEDGER = ROOT / "experiments" / "ledger.jsonl"
BASE = ("zero", "linear", "history_revert")
NAME = {"zero": "猜手不动", "linear": "线性外推", "history_revert": "历史回归"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Serif CJK SC", "PingFang SC", "WenQuanYi Zen Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def label(g):
    return NAME.get(g, g.replace("iphone", "iPhone").replace("+", "+\n"))


def _err(v, ci):
    if len(v) == 0:
        return None
    ci = np.array([c if c else (x, x) for c, x in zip(ci, v)], float)
    return [np.maximum(v - ci[:, 0], 0), np.maximum(ci[:, 1] - v, 0)]


def _models(res):
    return [g for g in res if g not in BASE]


def chart_ade(res, title, path):
    plt = _plt()
    gs = [g for g in ("zero", "linear") if g in res] + _models(res)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    v = np.array([res[g]["ade_cm_pooled"] for g in gs])
    col = ["#999" if g in BASE else "#6a9fd8" for g in gs]
    axes[0].bar(range(len(gs)), v, yerr=_err(v, [res[g].get("ade_ci95") for g in gs]), capsize=4, color=col)
    axes[0].axhline(res["zero"]["ade_cm_pooled"], ls="--", c="k", lw=.8)
    for i, x in enumerate(v):
        axes[0].text(i, x, "%.2f" % x, ha="center", va="bottom", fontsize=8)
    axes[0].set_ylim(v.min() * .95, v.max() * 1.03)
    axes[0].set_title("整体 ADE（cm，越低越好）")
    ms = [g for g in gs if g != "zero" and res[g].get("moving_minus_zero_cm") is not None]
    d = np.array([res[g]["moving_minus_zero_cm"] for g in ms])
    axes[1].bar(range(len(ms)), d, yerr=_err(d, [res[g].get("moving_minus_zero_ci95") for g in ms]), capsize=4,
                color=["#3a3" if x < 0 else "#e4572e" for x in d])
    axes[1].axhline(0, c="k", lw=.8)
    for i, x in enumerate(d):
        axes[1].text(i, x, "%+.3f" % x, ha="center", va="bottom" if x >= 0 else "top", fontsize=8)
    axes[1].set_title("手在动时：比「猜手不动」好多少（cm，<0 更好，带 95% 置信区间）")
    for ax, L in ((axes[0], gs), (axes[1], ms)):
        ax.set_xticks(range(len(L))); ax.set_xticklabels([label(g) for g in L], fontsize=8)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def chart_clips(res, clips, title, path):
    plt = _plt()
    gs = [g for g in ("zero",) if g in res] + _models(res)
    clips = clips or list(res["zero"].get("per_clip_ade_cm", {}))
    if not clips:
        return False
    fig, ax = plt.subplots(figsize=(11, 4.6)); w = .8 / len(gs)
    for j, g in enumerate(gs):
        v = [res[g].get("per_clip_ade_cm", {}).get(c, np.nan) for c in clips]
        ax.bar(np.arange(len(clips)) + j * w, v, w, label=label(g).replace("\n", ""), color="#999" if g == "zero" else None)
    ax.set_xticks(np.arange(len(clips)) + w * (len(gs) - 1) / 2); ax.set_xticklabels(["测试片段\n" + c for c in clips])
    ax.set_ylabel("ADE（cm）"); ax.set_title(title + "：每段留出录像的误差（留一段交叉验证）")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return True


def chart_horizon(res, title, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for g in [g for g in ("zero", "linear") if g in res] + _models(res):
        r = res[g]
        if r.get("horizon_ade_cm"):
            y = np.array(r["horizon_ade_cm"]); x = np.arange(1, len(y) + 1)
            sd = np.array(r.get("horizon_ade_cm_seed_std") or np.zeros_like(y))
        else:  # 旧结果只有第 1/8/16 步
            ks = sorted(int(k.split("@")[1]) for k in r["mean"] if k.startswith("cum_pos_cm@"))
            x = np.array(ks); y = np.array([r["mean"]["cum_pos_cm@%d" % k] for k in ks]); sd = np.zeros_like(y)
        ls = "--" if g in BASE else "-"
        ax.plot(x, y, ls, marker="o", ms=3, label=label(g).replace("\n", ""), c="k" if g == "zero" else None)
        if sd.any():
            ax.fill_between(x, y - sd, y + sd, alpha=.2)
    ax.set_xlabel("预测第几步（15 帧/秒）"); ax.set_ylabel("手腕位置误差（cm）")
    ax.set_title(title + "：越往后预测，误差越大"); ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)


def chart_curves(curves, title, path):
    if not curves:
        return False
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    mixes = list(dict.fromkeys(c["mix"] for c in curves))
    for mi, mix in enumerate(mixes):
        col = "C%d" % mi
        for c in [c for c in curves if c["mix"] == mix]:
            off = 0
            for stage in ("pretrain", "finetune"):
                pts = [p for p in c.get(stage) or [] if "step" in p]
                if not pts:
                    continue
                s = np.array([p["step"] for p in pts]) + off
                axes[0].plot(s, [p["loss"] for p in pts], c=col, alpha=.35, lw=.8)
                axes[1].plot(s, [p["val_ade_cm"] if p["val_ade_cm"] is not None else np.nan for p in pts], c=col, alpha=.35, lw=.8)
                off = s[-1]
        axes[0].plot([], [], c=col, label=label(mix).replace("\n", ""))
    axes[0].set_yscale("log"); axes[0].set_title("训练损失"); axes[1].set_title("验证 ADE（cm）")
    for ax in axes:
        ax.set_xlabel("训练步数（先预训练再微调，每条线=一个种子×一个折）")
    axes[0].legend(fontsize=7); fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return True


def chart_traj(examples, title, path):
    gs = [g for g in examples if g not in BASE]
    if not gs:
        return False
    plt = _plt()
    ex = examples[gs[0]]; ch = ex["chunks"]
    if not ch:
        return False
    fig, axes = plt.subplots(1, len(ch), figsize=(4.4 * len(ch), 4.4), squeeze=False)
    for ax, c in zip(axes[0], ch):
        gt = np.array(c["gt_cm"]); ax.plot(gt[:, 0], gt[:, 1], "-o", c="k", ms=3, label="真值")
        for g in gs + ["linear"]:
            hit = next((e for e in examples.get(g, {}).get("chunks", []) if e["chunk"] == c["chunk"] and e["hand"] == c["hand"]), None)
            if hit:
                p = np.array(hit["pred_cm"]); ax.plot(p[:, 0], p[:, 1], "-o", ms=2, alpha=.8, label=label(g).replace("\n", ""))
        ax.plot(0, 0, "s", c="#999", label="猜手不动")
        ax.set_aspect("equal", "datalim"); ax.set_xlabel("x（cm）"); ax.set_ylabel("y（cm）")
        ax.set_title("%s手 · 第 %d 段" % ("左右"[c["hand"]] if c["hand"] < 2 else "", c["chunk"]))
    axes[0][0].legend(fontsize=7)
    fig.suptitle("%s：手腕轨迹 预测 vs 真值（片段 %s，未来 16 步）" % (title, ex["clip"])); fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)
    return True


def run_report(metrics_path, name, out_root=REPORT):
    m = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    res = m["results"]; out = Path(out_root) / name; out.mkdir(parents=True, exist_ok=True)
    made = []
    chart_ade(res, name, out / "ade.png"); made.append(out / "ade.png")
    if "zero" not in res:
        raise ValueError("metrics 里没有 zero 基线")
    if chart_clips(res, m.get("clips"), name, out / "per_clip.png"):
        made.append(out / "per_clip.png")
    chart_horizon(res, name, out / "horizon.png"); made.append(out / "horizon.png")
    if chart_curves(m.get("curves"), name, out / "loss_curves.png"):
        made.append(out / "loss_curves.png")
    if chart_traj(m.get("examples") or {}, name, out / "trajectory.png"):
        made.append(out / "trajectory.png")
    rows = [{"组": g, "整体ADE_cm": round(r["ade_cm_pooled"], 3), "比不动_cm": round(r.get("ade_minus_zero_cm", 0), 3),
             "手在动ADE_cm": round(r.get("moving_ade_cm", np.nan), 3),
             "手在动比不动_cm": round(r.get("moving_minus_zero_cm", np.nan), 3),
             "95%CI": r.get("moving_minus_zero_ci95"), "种子数": r.get("n_seeds")} for g, r in res.items()]
    with open(out / "table.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / "README.md").write_text("# %s\n\n" % name + "\n".join("![](%s)" % p.name for p in made) + "\n", encoding="utf-8")
    print("report", name, *[str(p) for p in made])
    return made


def load_ledger(path=LEDGER):
    if not Path(path).exists():
        return []
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def summary(ledger_path=LEDGER, out_root=REPORT):
    L = [r for r in load_ledger(ledger_path) if r.get("status") == "ok" and r.get("overall_ade_cm") is not None]
    if not L:
        return []
    out = Path(out_root); out.mkdir(parents=True, exist_ok=True)
    cols = ["id", "数据配比", "整体ADE", "比不动", "手在动ADE", "手在动比不动", "95%CI"]
    rows = []
    for r in L:
        ci = r.get("delta_moving_ci95")
        rows.append([r["id"], (r.get("config") or {}).get("mix", ""), "%.3f" % r["overall_ade_cm"],
                     "%+.3f" % r["delta_overall_cm"] if r.get("delta_overall_cm") is not None else "-",
                     "%.3f" % r["moving_ade_cm"] if r.get("moving_ade_cm") is not None else "-",
                     "%+.3f" % r["delta_moving_cm"] if r.get("delta_moving_cm") is not None else "-",
                     "[%+.3f, %+.3f]" % tuple(ci) if ci else "-"])
    with open(out / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(cols); w.writerows(rows)
    plt = _plt()
    M = [r for r in L if r.get("delta_moving_cm") is not None]
    fig = plt.figure(figsize=(14, 5 + .32 * len(rows)))
    ax = fig.add_axes([.06, .58 if M else .05, .9, .38 if M else .9]) if M else None
    if M:
        d = np.array([r["delta_moving_cm"] for r in M])
        ax.bar(range(len(M)), d, yerr=_err(d, [r.get("delta_moving_ci95") for r in M]), capsize=3,
               color=["#3a3" if x < 0 else "#e4572e" for x in d])
        ax.axhline(0, c="k", lw=.8); ax.set_xticks(range(len(M))); ax.set_xticklabels([r["id"] for r in M], rotation=30, ha="right", fontsize=8)
        ax.set_title("所有实验：手在动时比「猜手不动」好多少（cm，<0 更好，绿=赢）")
    tax = fig.add_axes([.02, .02, .96, .5 if M else .9]); tax.axis("off")
    best = min(range(len(rows)), key=lambda i: L[i].get("delta_moving_cm") if L[i].get("delta_moving_cm") is not None else 9)
    t = tax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center"); t.auto_set_font_size(False); t.set_fontsize(8)
    t.auto_set_column_width(list(range(len(cols))))
    for j in range(len(cols)):
        t[best + 1, j].set_facecolor("#d8f5d0")
    fig.savefig(out / "summary.png", dpi=130); plt.close(fig)
    print("summary", out / "summary.png")
    return [out / "summary.png", out / "summary.csv"]


def all_runs(out_root=REPORT):
    made = []
    for mf in sorted((ROOT / "experiments" / "runs").rglob("metrics.json")):
        rel = mf.parent.relative_to(ROOT / "experiments" / "runs")
        made += run_report(mf, "_".join(rel.parts), out_root)
    for mf in sorted((ROOT / "docs" / "xhey").glob("*metrics.json")):
        made += run_report(mf, mf.stem.replace("_metrics", ""), out_root)
    return made + summary(out_root=out_root)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "summary", "all"])
    ap.add_argument("--metrics"); ap.add_argument("--name")
    ap.add_argument("--keep-json", action="store_true", help="把 metrics.json 也拷进报告目录")
    a = ap.parse_args()
    if a.cmd == "run":
        run_report(a.metrics, a.name or Path(a.metrics).parent.name)
        if a.keep_json:
            shutil.copy(a.metrics, REPORT / (a.name or Path(a.metrics).parent.name) / "metrics.json")
    elif a.cmd == "summary":
        summary()
    else:
        all_runs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
