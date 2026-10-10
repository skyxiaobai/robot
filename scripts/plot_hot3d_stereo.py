"""画 HOT3D 双目评测图：左=实测（误差 vs 手距离），右=仿真（硬件选型）。"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for name in ("Noto Sans CJK SC", "WenQuanYi Zen Hei", "Noto Sans CJK JP", "SimHei"):
    if any(name in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [name]
        break
plt.rcParams["axes.unicode_minus"] = False

report, sim, out = sys.argv[1:4]
R = json.load(open(report))
S = json.load(open(sim))
fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))

# 1 实测：误差 vs 距离
bins = [b for b in R["summary"]["by_depth"] if b["stereo"].get("n", 0) >= 5]
x = range(len(bins))
lab = ["%d-%dcm\n(n=%d)" % (b["bin"][0] * 100, min(b["bin"][1], 1) * 100, b["stereo"]["n"]) for b in bins]
ax[0].bar([i - 0.2 for i in x], [b["mono"]["median"] for b in bins], 0.4, label="单目 WiLoR（中位数）", color="#f39c12")
ax[0].bar([i + 0.2 for i in x], [b["stereo"]["median"] for b in bins], 0.4, label="双目三角化（中位数）", color="#2e86de")
ax[0].scatter([i + 0.2 for i in x], [b["stereo"]["p90"] for b in bins], marker="_", s=400, c="k", label="双目 p90")
ax[0].axhline(2, ls="--", c="r", label="2 cm 目标")
ax[0].set_xticks(list(x)); ax[0].set_xticklabels(lab)
ax[0].set_ylabel("手腕三维误差 (cm)"); ax[0].set_title("【实测】HOT3D Quest3 真实图像\n手腕误差 vs 手离相机距离"); ax[0].legend(fontsize=8)

# 2 仿真：基线 & 同步
runs = S["runs"]
def pick(g):
    return [r for r in runs if r["group"] == g]
for g, c in (("基线", "#2e86de"), ("双目同步偏差", "#c0392b")):
    rs = pick(g)
    ax[1].errorbar([r["label"] for r in rs], [r["wrist"]["median"] for r in rs],
                   yerr=[[r["wrist"]["median"] - r["wrist_median_ci"][0] for r in rs],
                         [r["wrist_median_ci"][1] - r["wrist"]["median"] for r in rs]],
                   fmt="o-", c=c, capsize=4, label=g + "（中位数）")
    ax[1].plot([r["label"] for r in rs], [r["wrist"]["p90"] for r in rs], "x:", c=c, label=g + "（p90）")
ax[1].axhline(2, ls="--", c="r")
ax[1].set_ylabel("手腕三维误差 (cm)"); ax[1].set_title("【仿真】基线 / 双目同步偏差\n（默认 75°, 1280x800, 全局快门）")
ax[1].legend(fontsize=8); ax[1].tick_params(axis="x", rotation=30)

# 3 仿真：分辨率 × 视场角，及 <2cm 比例 + 覆盖率
rs = pick("分辨率") + pick("视场角")
labels = [r["label"].replace(" (OV9281)", "\nOV9281").replace(" (AR0234)", "\nAR0234") for r in rs]
ax[2].bar(range(len(rs)), [100 * r["within_2cm"] for r in rs], color="#27ae60", label="达到 2cm 的帧比例 %")
ax[2].plot(range(len(rs)), [100 * r["coverage"] for r in rs], "ko-", label="手在两目视野内的比例 %")
ax[2].set_xticks(range(len(rs))); ax[2].set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
ax[2].set_ylim(0, 105); ax[2].set_title("【仿真】分辨率 / 视场角：精度 vs 覆盖率\n(alpha=%.2f，由真实图像降采样实测拟合)" % S["alpha_used"])
ax[2].legend(fontsize=8)
plt.tight_layout()
plt.savefig(out, dpi=130)
print("wrote", out)
