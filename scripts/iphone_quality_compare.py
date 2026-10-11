#!/usr/bin/env python3
"""Path B: iPhone vs EgoDex vs HOT3D quality comparison charts + JSON.

Numbers are taken from already-committed docs / local pipeline outputs and the
local detailed QC xlsx (not committed). Re-run:

  python scripts/iphone_quality_compare.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DOC = ROOT / "docs" / "iphone_mcap"
OUT_DOC.mkdir(parents=True, exist_ok=True)

# Prefer Chinese-capable fonts already on the Ubuntu box.
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC",
    "WenQuanYi Micro Hei",
    "Noto Sans CJK JP",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def build_metrics() -> dict:
    """Assemble comparison metrics from known repo numbers + local runs."""
    # --- HOT3D (Quest3 stereo WiLoR), docs/hot3d_stereo/results.md + scale ---
    hot3d = {
        "source": "HOT3D Quest3 + WiLoR stereo",
        "refs": [
            "docs/hot3d_stereo/results.md",
            "docs/hot3d_stereo/train_compare_scale.md",
        ],
        "detect_left": 0.916,
        "detect_right": 0.901,
        "detect_both": 0.860,
        "wrist_median_cm_stereo": 1.22,
        "wrist_mean_cm_stereo": 3.20,
        "wrist_p90_cm_stereo": 3.41,
        "wrist_median_cm_mono": 2.71,
        "mpjpe_median_cm_stereo": 2.12,
        "mpjpe_median_cm_mono": 4.09,
        "wrist_le2cm_stereo": 0.75,
        "wrist_le2cm_mono": 0.33,
        "label_vs_gt_wrist_median_cm_24clips": 1.24,
        "label_coverage_24clips": 0.79,
        "has_metric_depth": True,
        "has_gt": True,
        "fps_nominal": None,  # SLAM cams; subsampled in eval
        "notes": "真值 MANO；双目三角化。24 clips 扩大后手腕 vs GT 中位 1.24 cm。",
    }

    # --- EgoDex WiLoR mono, docs/reviv_compare.md ---
    egodex = {
        "source": "EgoDex + WiLoR mono",
        "refs": ["docs/reviv_compare.md", "docs/egodex_act_results.md"],
        "detect_rate": 0.976,
        "detect_rate_loose": 0.984,
        "mpjpe_root_rel_median_cm": 3.97,  # ~3.5–4 cm band beginners quote
        "wrist_median_cm_unscaled": 22.3,
        "wrist_median_cm_scaled": 4.03,
        "wrist_p90_cm_unscaled": 29.5,
        "wrist_le2cm_unscaled": 0.0,
        "has_metric_depth": False,
        "has_gt": True,  # ARKit world poses
        "fps_nominal": 30.0,
        "notes": "单目无米制深度：绝对手腕误差大；相对根/缩放后约 4 cm。",
    }

    # --- iPhone Pro Max LiDAR (3 clips) ---
    # Capture QC from local 详细分析 xlsx (IDs 266/312/597); pipeline from outputs.
    clips = [
        {
            "id": 266,
            "name": "2026-08-22_18-21",
            "task": "在家做饭 · 备菜到收拾",
            "duration_s": 856.1,
            "rgb_fps": 29.988,
            "depth_fps": 30.0,
            "rgb_period_med_ms": 33.343,
            "rgb_period_p95_ms": 33.349,
            "rgb_period_p99_ms": 33.354,
            "rgb_max_gap_ms": 66.688,
            "rgb_frames": 25675,
            "depth_frames": 25675,
            "onboard_both_hands": 0.915,
            "hand_msg_over_rgb": 0.986,
            "depth_conf_ge1": 0.810,
            "depth_conf_eq2": 0.699,
            "imu_hz": 99.85,
            "pose_hz": 36.34,
            # WiLoR+LiDAR fused valid (either hand), short 180 @30fps vs full stride-2 @15fps
            "wilor_either_short180": 0.867,
            "wilor_left_short180": 0.800,
            "wilor_right_short180": 0.633,
            "wilor_either_full_s2": 0.156,
            "wilor_left_full_s2": 0.080,
            "wilor_right_full_s2": 0.140,
            "label_coverage_s2": 0.9535,
            "depth_valid_frac_session": 0.792,
        },
        {
            "id": 312,
            "name": "2026-08-24_21-18",
            "task": "新手首拍 · 随便拍点啥",
            "duration_s": 928.5,
            "rgb_fps": 29.985,
            "depth_fps": 30.0,
            "rgb_period_med_ms": 33.343,
            "rgb_period_p95_ms": 33.348,
            "rgb_period_p99_ms": 33.353,
            "rgb_max_gap_ms": 100.031,
            "rgb_frames": 27843,
            "depth_frames": 27843,
            "onboard_both_hands": 0.854,
            "hand_msg_over_rgb": 0.962,
            "depth_conf_ge1": 0.909,
            "depth_conf_eq2": 0.846,
            "imu_hz": 99.86,
            "pose_hz": 46.96,
            "wilor_either_short180": 0.906,
            "wilor_left_short180": 0.844,
            "wilor_right_short180": 0.550,
            "wilor_either_full_s2": 0.079,
            "wilor_left_full_s2": 0.071,
            "wilor_right_full_s2": 0.070,
            "label_coverage_s2": 0.9959,
            "depth_valid_frac_session": 0.908,
        },
        {
            "id": 597,
            "name": "2026-09-05_17-16",
            "task": "居家整理 · 叠衣收纳",
            "duration_s": 1327.7,
            "rgb_fps": 29.916,
            "depth_fps": 29.9,
            "rgb_period_med_ms": 33.343,
            "rgb_period_p95_ms": 33.348,
            "rgb_period_p99_ms": 33.353,
            "rgb_max_gap_ms": 1167.002,
            "rgb_frames": 39720,
            "depth_frames": 39720,
            "onboard_both_hands": 0.810,
            "hand_msg_over_rgb": 0.979,
            "depth_conf_ge1": 0.891,
            "depth_conf_eq2": 0.794,
            "imu_hz": 99.86,
            "pose_hz": 35.61,
            "wilor_either_short180": 0.528,
            "wilor_left_short180": 0.456,
            "wilor_right_short180": 0.128,
            "wilor_either_full_s2": 0.280,
            "wilor_left_full_s2": 0.090,
            "wilor_right_full_s2": 0.251,
            "label_coverage_s2": 0.9086,
            "depth_valid_frac_session": 0.898,
        },
    ]

    # DexYCB proxy for iPhone-like RGB-D (no GT on real mcap)
    dexycb_iphone_sim = {
        "ref": "docs/iphone_eval/dexycb_summary.json / docs/iphone_capture_eval.md",
        "setting": "DexYCB 真值手上，用 iPhone 深度分辨率模拟 LiDAR 反投影",
        "wilor_detect_of_gt": 515 / 519,
        "wrist_median_cm_rgbd": 1.511,
        "wrist_p90_cm_rgbd": 2.607,
        "wrist_le2cm_rgbd": 0.718,
        "mpjpe_cm_rgbd": 1.809,
        "wrist_median_cm_mono_same": 2.429,
        "note": "第三视角桌面相机，不是头戴；只作「RGB-D 融合能到多准」的代理。",
    }

    iphone = {
        "source": "iPhone 13 Pro Max + LiDAR (xhey mcap)",
        "refs": [
            "outputs/iphone_mcap/run_s2/report.json",
            "outputs/iphone_mcap/sessions_s2/*/metadata.json",
            "本地 视频审核_收藏_20260907-1623_详细分析.xlsx（未入库）",
            "docs/iphone_mcap/results.md",
        ],
        "n_clips": 3,
        "clips": clips,
        "onboard_both_hands_mean": float(np.mean([c["onboard_both_hands"] for c in clips])),
        "depth_conf_ge1_mean": float(np.mean([c["depth_conf_ge1"] for c in clips])),
        "rgb_fps_mean": float(np.mean([c["rgb_fps"] for c in clips])),
        "rgb_depth_frame_match": all(c["rgb_frames"] == c["depth_frames"] for c in clips),
        "wilor_either_short180_mean": float(np.mean([c["wilor_either_short180"] for c in clips])),
        "wilor_either_full_s2_mean": float(np.mean([c["wilor_either_full_s2"] for c in clips])),
        "label_coverage_s2_mean": float(np.mean([c["label_coverage_s2"] for c in clips])),
        "has_metric_depth": True,
        "has_gt": False,
        "proxy_gt": dexycb_iphone_sim,
        "notes": (
            "采集层：RGB≈30fps 很稳、深度帧数=彩色、ARKit 双手 2D 检出 81–91%。"
            "训练层：整段 WiLoR+LiDAR 有效三维手偏低（出画/静止），短窗口更高。"
            "无头戴真值；DexYCB iPhone-sim RGB-D 手腕中位 1.51 cm 作代理。"
        ),
    }

    # Summary row for charts (comparable fields)
    summary = {
        "wrist_error_median_cm": {
            "HOT3D 双目": hot3d["wrist_median_cm_stereo"],
            "HOT3D 单目": hot3d["wrist_median_cm_mono"],
            "iPhone RGB-D\n(DexYCB 代理)": dexycb_iphone_sim["wrist_median_cm_rgbd"],
            "EgoDex WiLoR\n(缩放后)": egodex["wrist_median_cm_scaled"],
            "EgoDex WiLoR\n(绝对/未缩放)": egodex["wrist_median_cm_unscaled"],
        },
        "joint_mpjpe_median_cm": {
            "HOT3D 双目": hot3d["mpjpe_median_cm_stereo"],
            "iPhone RGB-D\n(DexYCB 代理)": dexycb_iphone_sim["mpjpe_cm_rgbd"],
            "EgoDex WiLoR\n(相对根)": egodex["mpjpe_root_rel_median_cm"],
            "HOT3D 单目": hot3d["mpjpe_median_cm_mono"],
        },
        "hand_detect": {
            "HOT3D 双目都检出": hot3d["detect_both"],
            "EgoDex WiLoR": egodex["detect_rate"],
            "iPhone ARKit 双手2D\n(三段均值)": iphone["onboard_both_hands_mean"],
            "iPhone WiLoR 任一手\n(前180帧均值)": iphone["wilor_either_short180_mean"],
            "iPhone WiLoR 任一手\n(整段 stride2 均值)": iphone["wilor_either_full_s2_mean"],
        },
        "depth_coverage": {
            "iPhone LiDAR conf≥1\n(三段均值)": iphone["depth_conf_ge1_mean"],
            "HOT3D 双目标签覆盖\n(24 clips)": hot3d["label_coverage_24clips"],
            "EgoDex 米制深度": 0.0,  # none
        },
        "fps_stability_iphone": {
            c["name"]: {
                "rgb_fps": c["rgb_fps"],
                "period_med_ms": c["rgb_period_med_ms"],
                "period_p99_ms": c["rgb_period_p99_ms"],
                "max_gap_ms": c["rgb_max_gap_ms"],
            }
            for c in clips
        },
    }

    return {
        "title": "Path B · iPhone vs EgoDex vs HOT3D 数据质量对比",
        "date": "2026-10-11",
        "hot3d": hot3d,
        "egodex": egodex,
        "iphone": iphone,
        "summary": summary,
    }


def _barh(ax, mapping: dict, title: str, xlabel: str, color: str = "#4C78A8"):
    labels = list(mapping.keys())
    vals = [mapping[k] for k in labels]
    y = np.arange(len(labels))
    bars = ax.barh(y, vals, color=color, height=0.65)
    ax.set_yticks(y, labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.invert_yaxis()
    xmax = max(vals) * 1.18 if vals else 1
    ax.set_xlim(0, xmax)
    for b, v in zip(bars, vals):
        ax.text(
            v + xmax * 0.01,
            b.get_y() + b.get_height() / 2,
            f"{v:.2f}" if v < 10 else f"{v:.1f}",
            va="center",
            fontsize=9,
        )
    ax.grid(axis="x", linestyle=":", alpha=0.5)


def make_charts(metrics: dict) -> list[Path]:
    paths = []
    s = metrics["summary"]

    # 1) Wrist error
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    _barh(ax, s["wrist_error_median_cm"], "手腕误差中位数（有真值或代理真值）", "厘米 (cm)", "#4C78A8")
    ax.axvline(2.0, color="#E45756", ls="--", lw=1.2, label="2 cm 目标")
    ax.legend(loc="lower right")
    fig.tight_layout()
    p1 = OUT_DOC / "quality_wrist_error.png"
    fig.savefig(p1, dpi=140)
    plt.close(fig)
    paths.append(p1)

    # 2) Detect rates
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    detect_pct = {k: v * 100 for k, v in s["hand_detect"].items()}
    _barh(ax, detect_pct, "手部检出率（口径不同，见图注）", "百分比 (%)", "#54A24B")
    fig.tight_layout()
    p2 = OUT_DOC / "quality_hand_detect.png"
    fig.savefig(p2, dpi=140)
    plt.close(fig)
    paths.append(p2)

    # 3) Depth / coverage
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    depth_pct = {k: v * 100 for k, v in s["depth_coverage"].items()}
    _barh(ax, depth_pct, "深度 / 三维标签覆盖", "百分比 (%)", "#F58518")
    fig.tight_layout()
    p3 = OUT_DOC / "quality_depth_coverage.png"
    fig.savefig(p3, dpi=140)
    plt.close(fig)
    paths.append(p3)

    # 4) iPhone FPS stability
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    names = [c["name"] for c in metrics["iphone"]["clips"]]
    fps = [c["rgb_fps"] for c in metrics["iphone"]["clips"]]
    gaps = [c["rgb_max_gap_ms"] for c in metrics["iphone"]["clips"]]
    x = np.arange(len(names))
    axes[0].bar(x, fps, color="#4C78A8")
    axes[0].axhline(30, color="#E45756", ls="--", lw=1, label="标称 30 fps")
    axes[0].set_xticks(x, [n.replace("2026-", "") for n in names], rotation=15)
    axes[0].set_ylim(28, 31)
    axes[0].set_ylabel("实测 RGB fps")
    axes[0].set_title("iPhone 三段 RGB 帧率")
    axes[0].legend()
    axes[0].grid(axis="y", ls=":", alpha=0.5)

    axes[1].bar(x, gaps, color="#E45756")
    axes[1].set_xticks(x, [n.replace("2026-", "") for n in names], rotation=15)
    axes[1].set_ylabel("最大帧间隔 (ms)")
    axes[1].set_title("iPhone 三段 RGB 最大间隔")
    axes[1].grid(axis="y", ls=":", alpha=0.5)
    # annotate the big gap
    for i, g in enumerate(gaps):
        axes[1].text(i, g + max(gaps) * 0.02, f"{g:.0f}", ha="center", fontsize=8)
    fig.tight_layout()
    p4 = OUT_DOC / "quality_fps_stability.png"
    fig.savefig(p4, dpi=140)
    plt.close(fig)
    paths.append(p4)

    # 5) Combined overview panel
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    _barh(
        axes[0, 0],
        {
            "HOT3D 双目": s["wrist_error_median_cm"]["HOT3D 双目"],
            "iPhone RGB-D 代理": s["wrist_error_median_cm"]["iPhone RGB-D\n(DexYCB 代理)"],
            "HOT3D 单目": s["wrist_error_median_cm"]["HOT3D 单目"],
            "EgoDex 缩放后": s["wrist_error_median_cm"]["EgoDex WiLoR\n(缩放后)"],
        },
        "手腕误差中位 (cm)",
        "cm",
        "#4C78A8",
    )
    axes[0, 0].axvline(2.0, color="#E45756", ls="--", lw=1)

    _barh(
        axes[0, 1],
        {
            "HOT3D 双目 MPJPE": s["joint_mpjpe_median_cm"]["HOT3D 双目"],
            "iPhone RGB-D 代理": s["joint_mpjpe_median_cm"]["iPhone RGB-D\n(DexYCB 代理)"],
            "EgoDex 相对根": s["joint_mpjpe_median_cm"]["EgoDex WiLoR\n(相对根)"],
        },
        "关节误差 MPJPE (cm)",
        "cm",
        "#72B7B2",
    )

    _barh(
        axes[1, 0],
        {
            "EgoDex WiLoR": detect_pct["EgoDex WiLoR"],
            "HOT3D 双目都检出": detect_pct["HOT3D 双目都检出"],
            "iPhone ARKit 双手2D": detect_pct["iPhone ARKit 双手2D\n(三段均值)"],
            "iPhone WiLoR 短窗": detect_pct["iPhone WiLoR 任一手\n(前180帧均值)"],
            "iPhone WiLoR 整段": detect_pct["iPhone WiLoR 任一手\n(整段 stride2 均值)"],
        },
        "检出率 (%)",
        "%",
        "#54A24B",
    )

    _barh(
        axes[1, 1],
        {
            "iPhone LiDAR conf≥1": depth_pct["iPhone LiDAR conf≥1\n(三段均值)"],
            "HOT3D 双目标签覆盖": depth_pct["HOT3D 双目标签覆盖\n(24 clips)"],
            "EgoDex 米制深度": 0.0,
        },
        "深度 / 标签覆盖 (%)",
        "%",
        "#F58518",
    )
    fig.suptitle("Path B · iPhone / EgoDex / HOT3D 质量一览", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p5 = OUT_DOC / "quality_compare.png"
    fig.savefig(p5, dpi=140)
    plt.close(fig)
    paths.append(p5)

    return paths


def write_markdown(metrics: dict, chart_paths: list[Path]) -> Path:
    ip = metrics["iphone"]
    hot = metrics["hot3d"]
    ego = metrics["egodex"]
    proxy = ip["proxy_gt"]
    clips = ip["clips"]

    md = OUT_DOC / "quality_compare.md"
    lines = []
    a = lines.append
    a("# Path B：iPhone vs EgoDex vs HOT3D 数据质量对比")
    a("")
    a(f"**日期**：{metrics['date']} · **机器**：Ubuntu RTX 4090  ")
    a("**目的**：用初学者也关心的几项指标，说明我们现在的 iPhone 采集数据，相对公开集（EgoDex、HOT3D）好在哪、短在哪。  ")
    a("**图**：本目录 `quality_*.png`；数字汇总 `quality_metrics.json`。")
    a("")
    a("## 一句话结论")
    a("")
    a("1. **采集层（帧率 / 深度 / 机载手 2D）**：3 段 Pro Max 很稳——RGB ≈ **30.0 fps**，深度帧数与彩色 **一一对应**，LiDAR 有效深度约 **81–91%**，机载双手 2D 检出 **81–91%**。")
    a(f"2. **三维标签精度（有真值时）**：HOT3D 双目手腕中位 **{hot['wrist_median_cm_stereo']:.2f} cm** 仍是上限参照；用 DexYCB 模拟 iPhone 深度分辨率时，RGB-D 融合手腕中位 **{proxy['wrist_median_cm_rgbd']:.2f} cm**，接近双目。EgoDex 上 WiLoR 相对根 MPJPE ≈ **{ego['mpjpe_root_rel_median_cm']:.1f} cm**，但**绝对手腕**未缩放约 **{ego['wrist_median_cm_unscaled']:.0f} cm**（单目没米制深度）。")
    a(f"3. **训练可用的三维手**：整段 WiLoR+LiDAR「任一手有效」均值只有约 **{ip['wilor_either_full_s2_mean']*100:.0f}%**（出画/静止多）；前 180 帧活跃窗口约 **{ip['wilor_either_short180_mean']*100:.0f}%**。质量故事要写清「采集很稳 ≠ 每帧都有可训三维手」。")
    a("")
    a("![总览](quality_compare.png)")
    a("")
    a("---")
    a("")
    a("## 1. 手腕 / 关节误差（越小越好）")
    a("")
    a("| 来源 | 手腕误差中位 | 关节 MPJPE | ≤2 cm 比例 | 真值？ |")
    a("|---|---|---|---|---|")
    a(f"| **HOT3D 双目三角化** | **{hot['wrist_median_cm_stereo']:.2f} cm** | {hot['mpjpe_median_cm_stereo']:.2f} cm | {hot['wrist_le2cm_stereo']*100:.0f}% | 有 MANO |")
    a(f"| HOT3D 单目 WiLoR | {hot['wrist_median_cm_mono']:.2f} cm | {hot['mpjpe_median_cm_mono']:.2f} cm | {hot['wrist_le2cm_mono']*100:.0f}% | 有 |")
    a(f"| HOT3D 24 clips 双目 vs GT | **{hot['label_vs_gt_wrist_median_cm_24clips']:.2f} cm** | — | — | 有 |")
    a(f"| **iPhone RGB-D（DexYCB 代理）** | **{proxy['wrist_median_cm_rgbd']:.2f} cm** | {proxy['mpjpe_cm_rgbd']:.2f} cm | {proxy['wrist_le2cm_rgbd']*100:.0f}% | 代理真值 |")
    a(f"| EgoDex WiLoR 相对根 / 缩放后手腕 | 缩放后 {ego['wrist_median_cm_scaled']:.2f} cm | {ego['mpjpe_root_rel_median_cm']:.2f} cm | 绝对 0% | 有 ARKit |")
    a(f"| EgoDex WiLoR 绝对手腕（未缩放） | {ego['wrist_median_cm_unscaled']:.1f} cm | — | 0% | 有 |")
    a("")
    a("![手腕误差](quality_wrist_error.png)")
    a("")
    a("> 真实 iPhone mcap **没有**头戴动捕真值，所以「iPhone 能到多准」用 DexYCB 上模拟 iPhone 深度分辨率的 RGB-D 结果作代理（见 `docs/iphone_capture_eval.md`）。那是第三视角桌面相机，不能直接当成头戴误差。")
    a("")
    a("---")
    a("")
    a("## 2. 手部检出率")
    a("")
    a("| 来源 | 口径 | 检出率 |")
    a("|---|---|---|")
    a(f"| HOT3D | 左右目都检出（WiLoR） | **{hot['detect_both']*100:.1f}%**（左 {hot['detect_left']*100:.1f}% / 右 {hot['detect_right']*100:.1f}%） |")
    a(f"| EgoDex | WiLoR 单目检出 | **{ego['detect_rate']*100:.1f}%** |")
    a(f"| iPhone 机载 | ARKit 双手 2D（索引） | 三段 **{clips[0]['onboard_both_hands']*100:.1f}% / {clips[1]['onboard_both_hands']*100:.1f}% / {clips[2]['onboard_both_hands']*100:.1f}%**（均 {ip['onboard_both_hands_mean']*100:.1f}%） |")
    a(f"| iPhone WiLoR | 任一手有效 · 前 180 帧 | {clips[0]['wilor_either_short180']*100:.0f}% / {clips[1]['wilor_either_short180']*100:.0f}% / {clips[2]['wilor_either_short180']*100:.0f}%（均 {ip['wilor_either_short180_mean']*100:.0f}%） |")
    a(f"| iPhone WiLoR | 任一手有效 · 整段 stride=2 | {clips[0]['wilor_either_full_s2']*100:.0f}% / {clips[1]['wilor_either_full_s2']*100:.0f}% / {clips[2]['wilor_either_full_s2']*100:.0f}%（均 {ip['wilor_either_full_s2_mean']*100:.0f}%） |")
    a("")
    a("![检出率](quality_hand_detect.png)")
    a("")
    a("机载 2D 高、整段 WiLoR 三维低：QC 报告里「手出画」「摆拍/静止」占大多数坏帧（见 `outputs/iphone_mcap/run_s2/report.md`）。论文 B 部分应把这两层分开写。")
    a("")
    a("---")
    a("")
    a("## 3. 深度覆盖与同步")
    a("")
    a("| 片段 | 深度 conf≥1 | conf=2 | 会话 depth_valid | RGB 帧 = 深度帧 | IMU Hz |")
    a("|---|---|---|---|---|---|")
    for c in clips:
        a(
            f"| {c['name']} | {c['depth_conf_ge1']*100:.1f}% | {c['depth_conf_eq2']*100:.1f}% | "
            f"{c['depth_valid_frac_session']*100:.1f}% | "
            f"{'是' if c['rgb_frames']==c['depth_frames'] else '否'} "
            f"({c['rgb_frames']}) | {c['imu_hz']:.1f} |"
        )
    a(f"| **均值** | **{ip['depth_conf_ge1_mean']*100:.1f}%** | — | — | 三段均匹配 | ≈99.9 |")
    a("")
    a("| 对照 | 深度 / 三维标签 |")
    a("|---|---|")
    a(f"| HOT3D 24 clips 双目标签覆盖 | {hot['label_coverage_24clips']*100:.0f}% |")
    a("| EgoDex | **无**主动深度；靠 ARKit 位姿 |")
    a("")
    a("![深度覆盖](quality_depth_coverage.png)")
    a("")
    a("---")
    a("")
    a("## 4. 帧率稳定性（iPhone 三段）")
    a("")
    a("| 片段 | 实测 RGB fps | 周期中位 / P99 (ms) | 最大间隔 (ms) | 深度 fps |")
    a("|---|---|---|---|---|")
    for c in clips:
        a(
            f"| {c['name']} | **{c['rgb_fps']:.3f}** | "
            f"{c['rgb_period_med_ms']:.3f} / {c['rgb_period_p99_ms']:.3f} | "
            f"{c['rgb_max_gap_ms']:.1f} | {c['depth_fps']} |"
        )
    a("")
    a("![帧率](quality_fps_stability.png)")
    a("")
    a("- 前两段最大间隔 ≤100 ms；**09-05** 出现过 **1167 ms** 的大间隔（详细分析已标「待复核」），写论文时要当例外披露。")
    a("- 周期 P99 仍在 33.35 ms 附近 → 绝大多数时间锁在 30 fps。")
    a("")
    a("---")
    a("")
    a("## 5. 对论文 Path B 意味着什么")
    a("")
    a("| 主张 | 证据够不够 | 还要补什么 |")
    a("|---|---|---|---|")
    a("| 「手机采集管线稳、深度齐」 | **够**：fps / RGB=Depth / LiDAR 覆盖 | 深度字节序已在适配器约定；继续抽样交叉验证 |")
    a("| 「三维手精度接近双目」 | **代理够、真机不够**：DexYCB 1.51 cm；头戴无 GT | 双目硬件回来后在同场景重测；或桌面标定板/OptiTrack |")
    a("| 「比 EgoDex 更有米制深度」 | **够**：EgoDex 绝对手腕 ~22 cm vs 我们有 LiDAR | 在相同任务上并排训策略（Path C） |")
    a("| 「每帧都适合训练」 | **不够**：整段 WiLoR 有效手偏低 | 采集时双手入画、少静止；QC 后有效动作比例写进表 |")
    a("")
    a("---")
    a("")
    a("## 6. 数据来源与复现")
    a("")
    a("- HOT3D：`docs/hot3d_stereo/results.md`，`docs/hot3d_stereo/train_compare_scale.md`")
    a("- EgoDex WiLoR：`docs/reviv_compare.md`（相对根 MPJPE ≈ 3.97 cm）")
    a("- iPhone 管线：`outputs/iphone_mcap/run_s2/`，`sessions_s2/*/metadata.json`")
    a("- iPhone 采集质检：本地 `视频审核_收藏_20260907-1623_详细分析.xlsx`（按用户要求不入库，只作参考）")
    a("- DexYCB 代理：`docs/iphone_eval/dexycb_summary.json`")
    a("- 本页图表：`python scripts/iphone_quality_compare.py`")
    a("")
    a("指标 JSON：[`quality_metrics.json`](quality_metrics.json)")
    a("")

    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md


def main():
    metrics = build_metrics()
    charts = make_charts(metrics)
    json_path = OUT_DOC / "quality_metrics.json"
    json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    md = write_markdown(metrics, charts)
    print("wrote", md)
    print("wrote", json_path)
    for p in charts:
        print("wrote", p)


if __name__ == "__main__":
    main()
