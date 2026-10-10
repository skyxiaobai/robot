#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""画 hot3d_train_compare.py 的结果：左图 ADE 及 95% CI，右图 MLP 测试 ADE 学习曲线。"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ORDER = ["zero", "linear_stereo", "linear_mono", "linear_gt", "linear_egodex",
         "mlp_stereo", "mlp_mono", "mlp_gt", "mlp_egodex", "mlp_egodex_ft_stereo"]


def main():
    p = argparse.ArgumentParser(); p.add_argument("--dir", required=True); p.add_argument("--out", required=True)
    p.add_argument("--curves", default=None, help="学习曲线取自哪个结果目录（默认同 --dir）")
    a = p.parse_args(); d = Path(a.dir)
    res = json.loads((d / "metrics.json").read_text())["results"]
    curves = json.loads((Path(a.curves or d) / "curves.json").read_text())
    names = [n for n in ORDER if n in res]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    y = [res[n]["ade_cm_pooled"] for n in names]
    lo = [y[i] - res[n]["ade_ci95"][0] for i, n in enumerate(names)]
    hi = [res[n]["ade_ci95"][1] - y[i] for i, n in enumerate(names)]
    col = ["gray" if n == "zero" else ("tab:blue" if "stereo" in n else "tab:green" if "gt" in n
           else "tab:orange" if "mono" in n else "tab:purple") for n in names]
    ax[0].bar(range(len(names)), y, yerr=[lo, hi], color=col, capsize=4)
    ax[0].axhline(res["zero"]["ade_cm_pooled"], color="gray", ls="--", lw=1)
    ax[0].set_xticks(range(len(names))); ax[0].set_xticklabels(names, rotation=40, ha="right")
    ax[0].set_ylabel("ADE (cm), lower is better"); ax[0].set_title("HOT3D held-out clip, target = GT wrist delta (95% bootstrap CI)")
    for n, c in [("mlp_stereo", "tab:blue"), ("mlp_mono", "tab:orange"), ("mlp_gt", "tab:green"), ("mlp_egodex_ft_stereo", "tab:purple")]:
        if n not in curves:
            continue
        L = min(len(run["curve"]) for run in curves[n])
        steps = [r["step"] for r in curves[n][0]["curve"][:L]]
        m = np.array([[r["test_ade_cm"] if r["test_ade_cm"] is not None else np.nan for r in run["curve"][:L]] for run in curves[n]])
        tl = np.array([[r["train_loss"] for r in run["curve"][:L]] for run in curves[n]])
        ax[1].plot(steps, np.nanmean(m, 0), color=c, label=n + " test ADE")
        ax[1].plot(steps, np.nanmean(tl, 0) * res["zero"]["ade_cm_pooled"] / max(np.nanmean(tl, 0)[0], 1e-9),
                   color=c, ls=":", lw=1)
    ax[1].axhline(res["zero"]["ade_cm_pooled"], color="gray", ls="--", lw=1, label="zero motion")
    ax[1].set_xlabel("training step"); ax[1].set_ylabel("ADE (cm)")
    ax[1].set_title("MLP curves (solid: held-out ADE; dotted: train loss, rescaled)"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(a.out, dpi=120); print(a.out)


if __name__ == "__main__":
    main()
