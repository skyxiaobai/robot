#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一条命令跑完双目手部管线（自有设备和 HOT3D 共用这个入口）。

自有设备（会话目录格式见 docs/headcam_data_spec.md §7.7）::

    python scripts/run_stereo_pipeline.py --session /data/sess01 /data/sess02 --out outputs/stereo

HOT3D-Clips（先自动转成会话目录，再跑同一条管线）::

    python scripts/run_stereo_pipeline.py --hot3d data/train_quest3/clip-000000 ... --out outputs/hot3d

iPhone Pro（Record3D 导出 .r3d / EXR+JPG 目录，或已转好的 iPhone 会话目录；用激光雷达深度代替三角化）::

    python scripts/run_stereo_pipeline.py --iphone capture1.r3d capture2.r3d --out outputs/iphone

需要 WiLoR 环境变量（WILOR_CHECKPOINT / WILOR_CONFIG / WILOR_DETECTOR / MANO_MODEL_DIR），
见 scripts/headcam/hand_pose.py。HOT3D 适配还需要 hand_tracking_toolkit 和 smplx（真值）。
输出：episodes/*.json、qc/yield.{csv,html}、lerobot/（v3.0）、report.json、report.md。
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from headcam.stereo_pipeline import StereoParams, run_pipeline  # noqa: E402

REASON_CN = {
    "hands_out_of_frame": "手出画", "view_drift": "视线飘移", "blur": "运动模糊", "staged_static": "摆拍或静止",
    "stereo_one_view": "只有一目认到手", "stereo_inconsistent": "左右目对不上", "stereo_filled": "补出来的帧",
    "low_label_coverage": "标注覆盖不足",
}
DROPPED_LABEL_REASONS = ("stereo_one_view", "stereo_inconsistent", "stereo_filled")
STATUS_CN = {"ok": "通过", "one_view": "只有一目认到", "few_joints": "可三角化的关节太少",
             "reproj": "重投影误差大", "depth": "深度不合理", "palm": "手掌尺寸不合理", "none": "两目都没认到",
             "jump": "速度门限剔除的跳点", "strict": "严格门限剔除",
             "lowconf": "手区域激光雷达高置信深度太少", "fit": "深度与 WiLoR 手型对不上", "tracking": "ARKit 位姿跳变"}


def _fmt(s):
    if not s or not s.get("n"):
        return "—（0）"
    return "%.2f / %.2f / %.0f%%（n=%d）" % (s["median_cm"], s["p90_cm"], 100 * s["within_2cm"], s["n"])


def write_markdown(summary, path):
    lines = ["# %s管线运行报告" % ("iPhone 激光雷达" if summary.get("iphone") else "双目"), "",
             "产出率 %.1f%%（%d / %d 帧），片段 %d 条通过 %d 条。" % (
                 100 * summary["yield"], summary["usable_frames"], summary["raw_frames"],
                 summary["episodes"], summary["accepted_episodes"])]
    if "label_coverage" in summary:
        lines.append("标注覆盖 %.1f%%（%d / %d 帧）。丢掉的立体标注不计入坏帧。覆盖率下限 %.0f%%（临时值，等真实设备数据再定；0 表示不因此拒绝）。" % (
            100 * summary["label_coverage"], summary["labeled_frames"], summary["raw_frames"],
            100 * summary.get("min_label_coverage", 0.0)))
    lines += ["", "## QC 坏帧原因（帧数；同一帧可有多个原因。只含手出画、视线、模糊、静止）", "",
              "| 原因 | 帧数 | 只因这一条 |", "|---|---|---|"]
    for k, v in summary["bad_frames_by_reason"].items():
        lines.append("| %s | %d | %d |" % (REASON_CN.get(k, k), v, summary["bad_frames_only_this_reason"].get(k, 0)))
    dropped = summary.get("dropped_labels_by_reason") or {}
    if dropped:
        lines += ["", "## 丢掉的标注（不计入坏帧）", "", "| 原因 | 帧数 | 只因这一条 |", "|---|---|---|"]
        only_dropped = summary.get("dropped_labels_only_this_reason") or {}
        for k, v in dropped.items():
            lines.append("| %s | %d | %d |" % (REASON_CN.get(k, k), v, only_dropped.get(k, 0)))
    lines += ["", "## 每段", "", "| 片段 | 帧 | 结论 | 坏帧比例 | 标注覆盖 |", "|---|---|---|---|---|"]
    for s in summary["sessions"]:
        coverage = s["qc"].get("label_coverage")
        coverage_txt = "—" if coverage is None else "%.1f%%" % (100 * coverage)
        lines.append("| %s | %d | %s | %.1f%% | %s |" % (
            Path(s["session"]).name, s["frames"],
            "通过" if s["qc"]["accepted"] else "拒绝", 100 * s["qc"]["bad_fraction"], coverage_txt))
    ev = summary.get("eval_all")
    if ev:
        lines += ["", "## 手腕世界坐标误差（对真值）：中位数 cm / p90 cm / ≤2cm 比例", "",
                  "| 组 | 结果 |", "|---|---|",
                  "| 检查前：%s | %s |" % ("激光雷达深度直接反投影的手腕" if summary.get("iphone") else "两目都认到、三角化",
                                         _fmt(ev["before_check"])),
                  "| 同一批手的 WiLoR 单目（参照） | %s |" % _fmt(ev["mono_same_hands"]),
                  "| 检查后：通过一致性检查 | %s |" % _fmt(ev["after_check"]),
                  "| 最终：平滑后、测到的帧 | %s |" % _fmt(ev["final_measured"]),
                  "| 最终：补出来的帧 | %s |" % _fmt(ev["final_filled"])]
        for k, v in ev["rejected_by_reason"].items():
            lines.append("| 被拒：%s | %s |" % (STATUS_CN.get(k, k), _fmt(v)))
        lines += ["", "逐手状态：" + "，".join("%s %d" % (STATUS_CN.get(k, k), v) for k, v in ev["status_counts"].items())]
    t = summary["timing"]
    lines += ["", "## 耗时（秒）", "",
              "手部模型 %.1f，三角化+检查+平滑 %.1f，QC %.2f，导出 %.1f，总计 %.1f" % (
                  t["backend_s"], t["stereo_refine_s"], t["qc_s"], t["export_s"], t.get("total_s", 0.0))]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description="双目手部管线：会话目录 → episode → QC → LeRobot")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--session", nargs="+", help="自有设备会话目录（§7.7）")
    src.add_argument("--hot3d", nargs="+", help="HOT3D-Clips 的 clip 目录，先转成会话目录")
    src.add_argument("--iphone", nargs="+", help="Record3D 导出（.r3d 或 EXR+JPG 目录）或已转好的 iPhone 会话目录")
    ap.add_argument("--stride", type=int, default=1, help="iPhone：每几帧取一帧（60 fps 录制可用 2）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="wilor", choices=["wilor", "hamer", "mediapipe"])
    ap.add_argument("--max-frames", type=int, default=None, help="HOT3D：每段最多转多少帧（调试用）")
    ap.add_argument("--no-gt", action="store_true", help="HOT3D：不生成真值（不需要 MANO / smplx）")
    ap.add_argument("--no-consistency", action="store_true", help="关闭左右一致性检查（对比用）")
    ap.add_argument("--smooth", default="rts", choices=["none", "rts", "one_euro", "kalman"],
                    help="rts：世界系手腕离线 RTS 平滑（默认）；one_euro 为 PR #22 行为")
    ap.add_argument("--wrist-mode", default="rigid_fit", choices=["rigid_fit", "tri"],
                    help="rigid_fit：单目手型稳健对齐三角化关节后取手腕（默认）；tri：直接用三角化手腕")
    ap.add_argument("--velocity-gate-m", type=float, default=0.02, help="世界系手腕跳点门限（米），0 关闭")
    ap.add_argument("--strict-reproj-px", type=float, default=None, help="可选严格门限：中位重投影误差上限（像素）")
    ap.add_argument("--strict-offaxis-deg", type=float, default=None, help="可选严格门限：手腕偏离光轴角度上限（度）")
    ap.add_argument("--min-cutoff", type=float, default=3.0, help="One Euro 静止截止频率 Hz")
    ap.add_argument("--beta", type=float, default=50.0, help="One Euro 速度系数 1/(m/s)")
    ap.add_argument("--max-gap", type=int, default=5, help="最多补几帧，0 表示不补")
    ap.add_argument("--min-label-coverage", type=float, default=0.0,
                    help="标注覆盖率下限。默认 0：只报告，不因此拒绝片段。临时值，等真实设备数据再定")
    ap.add_argument("--fixed-shape", action="store_true", help="固定手型（默认关）")
    ap.add_argument("--max-reproj-px", type=float, default=10.0)
    ap.add_argument("--no-assoc", action="store_true", help="关闭手部关联（翻转 TTA + 时序/双目关联），恢复 PR #23 行为")
    ap.add_argument("--no-recrop", action="store_true", help="关闭另一目重新裁剪")
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--export-python", default=os.environ.get("STEREO_EXPORT_PYTHON"),
                    help="用另一个装了 pyarrow 的 Python 做 LeRobot 导出（WiLoR 环境 numpy 太老时用）")
    ap.add_argument("--repo-id", default="local/headcam_stereo")
    a = ap.parse_args(argv)
    t0 = time.time()
    out = Path(a.out)
    sessions = a.session
    adapter = None
    if a.hot3d:
        from headcam.hot3d_adapter import convert_clip
        sessions, adapter = [], []
        mano = None if a.no_gt else os.environ.get("MANO_MODEL_DIR")
        for clip in a.hot3d:
            target = out / "sessions" / Path(clip.rstrip("/")).name
            if (target / "calib.yaml").is_file() and (target / "stereo" / "right.mp4").is_file():
                print("复用已转好的会话 %s" % target)
            else:
                ts = time.time()
                info = convert_clip(clip, target, mano_dir=mano, max_frames=a.max_frames)
                info["seconds"] = time.time() - ts
                adapter.append(info)
                print("HOT3D → 会话 %s（%d 帧，基线 %.1f cm）" % (target, info["frames"], 100 * info["baseline_m"]))
            sessions.append(str(target))
    builder = None
    if a.iphone:
        from iphone.record3d_adapter import convert, is_iphone_session
        from headcam.rgbd_pipeline import make_session_builder
        sessions, adapter = [], []
        for cap in a.iphone:
            if is_iphone_session(cap):
                sessions.append(cap)
                continue
            target = out / "sessions" / Path(cap.rstrip("/")).stem
            if is_iphone_session(target):
                print("复用已转好的会话 %s" % target)
            else:
                ts = time.time()
                info = convert(cap, target, max_frames=a.max_frames, stride=a.stride)
                info["seconds"] = time.time() - ts
                adapter.append(info)
                print("Record3D → 会话 %s（%d 帧，RGB %dx%d，深度 %dx%d）" % (
                    target, info["frames"], info["rgb"][0], info["rgb"][1], info["depth"][0], info["depth"][1]))
            sessions.append(str(target))
        builder = make_session_builder()
        if a.repo_id == "local/headcam_stereo":
            a.repo_id = "local/iphone_lidar"
    params = StereoParams(max_reproj_px=a.max_reproj_px, wrist_mode=a.wrist_mode,
                          velocity_gate_m=a.velocity_gate_m, max_median_reproj_px=a.strict_reproj_px,
                          max_offaxis_deg=a.strict_offaxis_deg, smooth=a.smooth, min_cutoff=a.min_cutoff, beta=a.beta, gap_fill=a.max_gap > 0,
                          max_gap=max(a.max_gap, 0), fixed_shape=a.fixed_shape, consistency=not a.no_consistency,
                          assoc=not a.no_assoc, recrop=not a.no_recrop)
    summary = run_pipeline(sessions, out, params, backend_name=a.backend, repo_id=a.repo_id, export=not a.no_export,
                           export_python=a.export_python, min_label_coverage=a.min_label_coverage,
                           session_builder=builder)
    summary["iphone"] = bool(a.iphone)
    summary["timing"]["adapter_s"] = sum(i.get("seconds", 0.0) for i in adapter or [])
    summary["timing"]["total_s"] = time.time() - t0
    if adapter:
        summary["adapter"] = adapter
    (out / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    write_markdown(summary, out / "report.md")
    print((out / "report.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
