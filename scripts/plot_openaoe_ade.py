#!/usr/bin/env python
"""Regenerate docs/xhey/openaoe_ade.png from docs/xhey/openaoe_metrics.json."""
import json, os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
m = json.loads((root / "docs/xhey/openaoe_metrics.json").read_text())
nano, tiny = m["nano"]["variants"], m["tiny"]["variants"]
rows = [
    ("zero", nano["zero"]),
    ("iPhone-only", nano["iphone"]),
    ("iPhone+xhey_rgb+aug", nano["iphone+xhey_rgb+aug"]),
    ("+OpenAoE nano", nano["iphone+openaoe+aug"]),
    ("+OpenAoE nano +xhey_rgb", nano["iphone+openaoe+xhey_rgb+aug"]),
    ("+OpenAoE tiny", tiny["iphone+openaoe+aug"]),
    ("+OpenAoE tiny +xhey_rgb", tiny["iphone+openaoe+xhey_rgb+aug"]),
]
labels = [r[0] for r in rows]
ades = [r[1]["ade_cm_pooled"] for r in rows]
moves = [r[1]["moving_ade_cm"] for r in rows]
fig, ax = plt.subplots(figsize=(10, 5.2))
x = range(len(labels)); w = 0.38
b1 = ax.bar([i - w/2 for i in x], ades, w, label="Overall ADE", color="#4C78A8")
b2 = ax.bar([i + w/2 for i in x], moves, w, label="Hand-moving ADE", color="#F58518")
ax.axhline(ades[0], color="#4C78A8", ls="--", lw=1, alpha=0.7)
ax.axhline(moves[0], color="#F58518", ls="--", lw=1, alpha=0.7)
ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
ax.set_ylabel("ADE (cm)  ↓ better")
ax.set_title("XHEY + Open-AoE pretrain (Kaggle)  |  dashed = zero-motion baseline")
ax.legend(loc="upper right"); ax.set_ylim(9.5, 15.5); ax.grid(axis="y", alpha=0.3)
for bars in (b1, b2):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.05, f"{h:.2f}", ha="center", va="bottom", fontsize=7)
fig.tight_layout()
out = root / "docs/xhey/openaoe_ade.png"
fig.savefig(out, dpi=140)
print("wrote", out)
