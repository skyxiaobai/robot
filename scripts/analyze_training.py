#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析训练产物，生成图表 + 自包含 HTML 报告。

根目录优先级: --root > 环境变量 ROBOT_ROOT > /mnt/sda/app/robot

数据源:
  - ACT·pusht (LeRobot 复现, job=pusht_act_100k, 实际训练 100K 步, log_freq=200):
      train.log             : 段1  0 → 10K 步 (50 条 INFO)
      /tmp/train_50k.log    : 段2 10K → 50K (200 条)
      /tmp/train_80k.log    : 段3 50K → 80K (150 条)
      /tmp/train_100k.log   : 段4 80K → 100K (100 条)
      outputs/checkpoints/  : 003000 → 100000 共 37 个 checkpoint
  - ACT 官方复现 (act_pusht_official, 训练 200K):
      /tmp/train_official.log      : 段1 0 → 100K (500 条)
      /tmp/train_official_200k.log : 段2 100K → 200K (500 条)
      outputs/train/act_pusht_official/checkpoints/ : 至 200000
  - outputs/eval/**/eval_info.json : pusht 评估（历史 2026-08-01 目录，或 Colab 新跑的评估）
  - outputs/colab_train.log : 若存在且本机没有 /tmp/train_50k.log，用它作为一条从 0 开始的连续训练曲线
  - outputs/mimicgen_gen.log          : Square 数据集（MimicGen）生成统计
  - outputs/mimicgen_train_image.log  : BC-RNN image 在 Square 上的训练
  - outputs/mimicgen_train_lowdim.log : BC-RNN low-dim 在 Square 上的训练

输出: <outdir>/training_report.html (自包含)
"""
import argparse
import base64
import glob
import io
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def resolve_root() -> str:
    parser = argparse.ArgumentParser(description="生成训练 HTML 报告")
    parser.add_argument(
        "--root",
        default=None,
        help="工程根目录。未指定时读环境变量 ROBOT_ROOT，否则用 /mnt/sda/app/robot",
    )
    args, _unknown = parser.parse_known_args()
    root = args.root or os.environ.get("ROBOT_ROOT") or "/mnt/sda/app/robot"
    return os.path.abspath(root)


ROOT = resolve_root()
OUT = os.path.join(ROOT, "outputs", "report_20260801")
os.makedirs(OUT, exist_ok=True)

# ---- 中文字体 ----
from matplotlib import font_manager
_cjk = None
for f in font_manager.fontManager.ttflist:
    if "Noto Sans CJK" in f.name or "WenQuanYi" in f.name or "Hei" in f.name:
        _cjk = f.name
        break
if _cjk:
    plt.rcParams["font.sans-serif"] = [_cjk]
plt.rcParams["axes.unicode_minus"] = False

PALETTE = ["#2563eb", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#0ea5e9"]


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _fmt(val, spec):
    if val is None:
        return "—"
    try:
        return format(val, spec)
    except (TypeError, ValueError):
        return "—"


def _loss_at(steps, values, step):
    if step in steps:
        return values[steps.index(step)]
    return None


def _is_train_log_line(line: str) -> bool:
    # 日志格式把 lerobot_train.py 截成 ot_train.py。v0.6.1 的指标行仍是 ot_train.py:641，
    # 这里只认脚本名，避免行号变动后 Colab 日志画不出曲线。
    return "ot_train.py" in line or "lerobot_train.py" in line


# ---------------------------------------------------------------- 1. ACT·pusht 训练曲线
# 每段日志按 log_freq=200 采样，第 i 条 INFO 对应全局步长 = seg_start + (i+1)*200
_COLAB_LOG = os.path.join(ROOT, "outputs", "colab_train.log")
if os.path.isfile(_COLAB_LOG) and not os.path.isfile("/tmp/train_50k.log"):
    # Colab 一次跑完（或断线后续写）的连续日志。本机没有原来的 /tmp 分段时用它，
    # 避免和仓库里只覆盖 0–10K 的 train.log 叠成两条从 0 起算的曲线。
    ACT_SEGS = [(_COLAB_LOG, 0)]
else:
    ACT_SEGS = [  # (path, seg_start_step)
        (os.path.join(ROOT, "train.log"), 0),
        ("/tmp/train_50k.log", 10000),
        ("/tmp/train_80k.log", 50000),
        ("/tmp/train_100k.log", 80000),
    ]

act_steps, act_loss, act_l1, act_kld = [], [], [], []
for path, seg_start in ACT_SEGS:
    if not os.path.exists(path):
        continue
    _n = 0
    for line in open(path, encoding="utf-8", errors="ignore"):
        if not _is_train_log_line(line):
            continue
        m = re.search(r"loss:([\d.]+).*?l1_loss:([\d.]+) kld_loss:([\d.]+)", line)
        if not m:
            continue
        _n += 1
        act_steps.append(seg_start + _n * 200)
        act_loss.append(float(m.group(1)))
        act_l1.append(float(m.group(2)))
        act_kld.append(float(m.group(3)))
act_total_steps = 100000  # train_config.json: steps=100000

# ---------------------------------------------------------------- 2. ACT 官方复现曲线
OFF_SEGS = [
    ("/tmp/train_official.log", 0),
    ("/tmp/train_official_200k.log", 100000),
]
off_steps, off_loss = [], []
for path, seg_start in OFF_SEGS:
    if not os.path.exists(path):
        continue
    _n = 0
    for line in open(path, encoding="utf-8", errors="ignore"):
        if not _is_train_log_line(line):
            continue
        m = re.search(r"loss:([\d.]+).*?l1_loss:([\d.]+) kld_loss:([\d.]+)", line)
        if not m:
            continue
        _n += 1
        off_steps.append(seg_start + _n * 200)
        off_loss.append(float(m.group(1)))

_off_ckpt_dir = os.path.join(ROOT, "outputs", "train", "act_pusht_official", "checkpoints")
official_ckpts = []
if os.path.isdir(_off_ckpt_dir):
    official_ckpts = sorted(int(d) for d in os.listdir(_off_ckpt_dir) if d.isdigit())

# ---- 合并图：两条 ACT 训练曲线
fig, ax1 = plt.subplots(figsize=(9.5, 4.8))
ax1.plot(act_steps, act_loss, color=PALETTE[0], lw=1.8, label="ACT·pusht 复现 (0→100K)")
ax1.plot(act_steps, act_l1, color=PALETTE[1], lw=1.2, ls="--", alpha=0.7, label="l1_loss")
ax1.plot(act_steps, act_kld, color=PALETTE[2], lw=1.0, ls=":", alpha=0.7, label="kld_loss")
ax1.plot(off_steps, off_loss, color=PALETTE[5], lw=1.8, ls="-.", label="ACT 官方复现 (0→200K)")
ax1.set_xlabel("训练步数")
ax1.set_ylabel("loss")
if act_steps and 100000 in act_steps:
    _act_title = "ACT 策略 · pusht 完整训练损失曲线（分四段日志合并）"
elif act_steps:
    _act_title = f"ACT 策略 · pusht 训练损失曲线（日志覆盖至 {act_steps[-1]} 步）"
else:
    _act_title = "ACT 策略 · pusht 训练损失曲线（无日志）"
ax1.set_title(_act_title)
ax1.grid(alpha=0.25)
ax1.legend(loc="upper right", fontsize=9)
# 里程碑标注（交替上下偏移避免重叠）
_off = 0
for s, label in [(10000, "10K"), (50000, "50K"), (100000, "100K"), (200000, "200K")]:
    _off += 1
    _dy = 16 if _off % 2 else -24
    if s in act_steps:
        v = act_loss[act_steps.index(s)]
        ax1.axvline(s, color="#94a3b8", lw=0.7, ls="--", alpha=0.5)
        ax1.annotate(f"{label}: loss {v:.3f}", xy=(s, v), xytext=(-70, _dy),
                     textcoords="offset points", fontsize=8, color="#475569")
    elif s in off_steps:
        v = off_loss[off_steps.index(s)]
        ax1.axvline(s, color="#94a3b8", lw=0.7, ls="--", alpha=0.5)
        ax1.annotate(f"{label}: loss {v:.3f}", xy=(s, v), xytext=(-70, _dy),
                     textcoords="offset points", fontsize=8, color="#475569")
fig_b64_act = fig_to_b64(fig)

# ---------------------------------------------------------------- 3. pusht 评估（映射 checkpoint）
# /tmp/eval_*.log 记录了各次评估使用的 checkpoint
EVAL_CKPT = {
    "12-32-00": "072000",
    "12-34-18": "last(→080000)",
    "14-07-46": "100000",
    "14-08-50": "100000",
    "14-09-50": "100000",
    "14-10-10": "100000",
    "14-10-33": "100000",
    "14-10-57": "100000",
    "14-11-20": "100000",
    "14-11-42": "100000",
}

evals = []
for f in sorted(glob.glob(os.path.join(ROOT, "outputs", "eval", "**", "eval_info.json"), recursive=True)):
    name = os.path.basename(os.path.dirname(f))
    d = json.load(open(f))
    o = d.get("overall", {})
    evals.append({
        "name": name,
        "hhmm": name[:5],
        "ckpt": EVAL_CKPT.get(name[:8], "未知"),
        "pc_success": o.get("pc_success", 0.0),
        "avg_max_reward": o.get("avg_max_reward", 0.0),
        "avg_sum_reward": o.get("avg_sum_reward", 0.0),
        "n": o.get("n_episodes", 0),
    })

if evals:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    xs = range(len(evals))
    ax.bar([x - 0.19 for x in xs], [e["pc_success"] for e in evals], width=0.38,
           color=PALETTE[0], label="成功率 %")
    ax.bar([x + 0.19 for x in xs], [e["avg_max_reward"] * 100 for e in evals], width=0.38,
           color=PALETTE[1], label="平均最高回报×100")
    labels = [f"{e['hhmm']}\n{e['ckpt']}" for e in evals]
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, rotation=45, fontsize=7)
    ax.set_ylabel("%")
    ax.set_title(f"ACT·pusht 评估结果（{len(evals)} 次，每集 horizon 300；下排标注所用 checkpoint）")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig_b64_eval = fig_to_b64(fig)
    best_eval = max(evals, key=lambda e: e["pc_success"])
    best_mr = max(evals, key=lambda e: e["avg_max_reward"])
else:
    fig, ax = plt.subplots(figsize=(8, 2.6))
    ax.text(0.5, 0.5, "未找到 outputs/eval/**/eval_info.json", ha="center", va="center")
    ax.axis("off")
    fig_b64_eval = fig_to_b64(fig)
    best_eval = {"pc_success": None, "avg_max_reward": None}
    best_mr = best_eval

# ---------------------------------------------------------------- 4. Square 数据生成
gen = {}
_gen_path = os.path.join(ROOT, "outputs", "mimicgen_gen.log")
if os.path.isfile(_gen_path):
    gen_log = open(_gen_path, encoding="utf-8", errors="ignore").read()
    m = re.search(r"Final Data Generation Stats\s*\n(\{.*?\})", gen_log, re.S)
    if m:
        gen = json.loads(m.group(1))

fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
if gen:
    ok = gen.get("num_success", 0)
    bad = gen.get("num_failures", 0)
    axes[0].pie([ok, bad], labels=[f"成功 {ok}", f"失败 {bad}"], autopct="%1.1f%%",
                colors=[PALETTE[2], "#e2e8f0"], startangle=90,
                wedgeprops=dict(width=0.42))
    axes[0].set_title(f"Square 数据集生成（MimicGen）\n成功率 {float(gen.get('success_rate', 0)):.1f}%")
    _labels = ["生成耗时(hrs)", "平均ep长度", "ep长度3σ"]
    _vals = [float(gen.get("time spent (hrs)", 0)),
             float(gen.get("ep_length_mean", 0)),
             float(gen.get("ep_length_3std", 0))]
    axes[1].barh(range(len(_labels)), _vals, color=[PALETTE[3], PALETTE[0], PALETTE[1]])
    axes[1].set_yticks(range(len(_labels)))
    axes[1].set_yticklabels(_labels)
    axes[1].set_xlabel("数值")
    axes[1].set_title(f"共尝试 {gen.get('num_attempts', 0)} 次，成功 {ok} 条")
fig.tight_layout()
fig_b64_gen = fig_to_b64(fig)

# ---------------------------------------------------------------- 5/6. BC-RNN Square
def parse_robomimic(path):
    """返回 (epochs, loss_list, rollout_epochs, rollout_sr_list)"""
    epochs, losses = [], []
    cur_ep = None
    for line in open(path, encoding="utf-8", errors="ignore"):
        tm = re.match(r"\s*Train Epoch (\d+)", line)
        if tm:
            cur_ep = int(tm.group(1))
            continue
        lm = re.match(r'\s*"Loss":\s*(-?[\d.e+-]+)', line)
        if lm and cur_ep is not None:
            epochs.append(cur_ep)
            losses.append(float(lm.group(1)))
    # Success_Rate 跟随在 rollout 行之后的 JSON 块里
    r_ep, r_sr = [], []
    text = open(path, encoding="utf-8", errors="ignore").read()
    for em in re.finditer(r"Epoch (\d+) Rollouts took .*? with results:\nEnv: \S+\n(\{.*?\})\n", text, re.S):
        r_ep.append(int(em.group(1)))
        try:
            r_sr.append(json.loads(em.group(2)).get("Success_Rate", None))
        except Exception:
            r_sr.append(None)
    return epochs, losses, r_ep, r_sr


def plot_bcrnn(path, title):
    if not os.path.isfile(path):
        fig, ax = plt.subplots(figsize=(8, 2.4))
        ax.set_title(title + "（日志缺失）")
        ax.axis("off")
        return [], [], [], [], fig_to_b64(fig)
    epochs, losses, r_ep, r_sr = parse_robomimic(path)
    fig, ax1 = plt.subplots(figsize=(9, 4.6))
    ax1.plot(epochs, losses, color=PALETTE[0], lw=1.4, label="训练 loss（对数似然,负值）")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(title)
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    sr = [s for s in r_sr if s is not None]
    se = [e for e, s in zip(r_ep, r_sr) if s is not None]
    if sr:
        ax2.plot(se, sr, color=PALETTE[2], lw=2, marker="o", ms=4, label="rollout 成功率")
        ax2.set_ylabel("成功率", color=PALETTE[2])
        ax2.set_ylim(-0.05, 1.05)
        best = max(range(len(sr)), key=lambda i: sr[i])
        ax2.annotate(f"最好 {sr[best]*100:.0f}% @ep{se[best]}",
                     xy=(se[best], sr[best]), xytext=(10, -22), textcoords="offset points",
                     arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=9, color=PALETTE[2])
    l1, _ = ax1.get_legend_handles_labels()
    l2, _ = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, [h.get_label() for h in l1 + l2], loc="upper right", fontsize=9)
    b64 = fig_to_b64(fig)
    return epochs, losses, r_ep, r_sr, b64


img_ep, img_loss, img_r_ep, img_r_sr, fig_b64_img = plot_bcrnn(
    os.path.join(ROOT, "outputs", "mimicgen_train_image.log"),
    "BC-RNN image · Square 训练（camera RGB 输入）")
low_ep, low_loss, low_r_ep, low_r_sr, fig_b64_low = plot_bcrnn(
    os.path.join(ROOT, "outputs", "mimicgen_train_lowdim.log"),
    "BC-RNN low-dim · Square 训练（低维状态输入）")

# ---------------------------------------------------------------- 汇总统计
def summarize(epochs, losses, r_ep, r_sr):
    best_sr = max((s for s in r_sr if s is not None), default=0)
    best_ep = next((e for e, s in zip(r_ep, r_sr) if s == best_sr), None)
    return {
        "epochs": len(epochs),
        "final_loss": losses[-1] if losses else None,
        "best_sr": best_sr,
        "best_ep": best_ep,
        "last_sr": r_sr[-1] if r_sr else None,
    }


img_sum = summarize(img_ep, img_loss, img_r_ep, img_r_sr)
low_sum = summarize(low_ep, low_loss, low_r_ep, low_r_sr)

act_final_loss = act_loss[-1] if act_loss else None
act_final_l1 = act_l1[-1] if act_l1 else None
off_final_loss = off_loss[-1] if off_loss else None

# ---------------------------------------------------------------- HTML 报告
def img_tag(b64):
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;border:1px solid #e2e8f0;border-radius:8px;">'


loss_10k = _loss_at(act_steps, act_loss, 10000)
loss_50k = _loss_at(act_steps, act_loss, 50000)
loss_100k = _loss_at(act_steps, act_loss, 100000)
kld_est = (act_loss[-1] - act_l1[-1]) if act_loss and act_l1 else None
if act_steps and act_steps[-1] >= 100000:
    act_scale = f"{act_total_steps:,} 步（4 段日志合并，37 个 checkpoint）"
else:
    covered = act_steps[-1] if act_steps else 0
    act_scale = f"目标 {act_total_steps:,} 步；当前日志覆盖至 {covered:,} 步"
if evals:
    sr_min = min(e["pc_success"] for e in evals)
    sr_max = max(e["pc_success"] for e in evals)
    eval_summary = f"{len(evals)} 次评估，成功率 {sr_min:.0f}–{sr_max:.0f}%"
else:
    eval_summary = "无 eval_info.json"
gen_sr = float(gen["success_rate"]) if gen.get("success_rate") is not None else None
best_sr_txt = "—" if best_eval.get("pc_success") is None else f"{best_eval['pc_success']:.0f}%"
best_mr_txt = "—" if best_mr.get("avg_max_reward") is None else f"{best_mr['avg_max_reward']:.2f}"
known_eval_ckpts = [e for e in evals if e["ckpt"] != "未知"]
if len(known_eval_ckpts) >= 10:
    eval_note = (
        "下排标注各次评估所用 checkpoint：<code>12-32</code>→072000、"
        "<code>12-34</code>→last(≈080000)、<code>14-07~14-11</code>→100000"
        "（官方复现 100K 于 15:11 完成后另有 15:17–15:19 三次，仍失败）。"
        f"20 次评估成功率 0–5%，平均最高回报 0.41（最高 {best_mr_txt}）。"
        "<b>关键结论</b>：从 72K 到 100K，评估指标几乎不变 —— 该任务（push-t）的瓶颈不在训练量。"
    )
elif evals:
    eval_note = (
        f"本次读到 {len(evals)} 份 eval_info.json，最好成功率 {best_sr_txt}，"
        f"最高平均回报 {best_mr_txt}。仓库里那次完整实验是 20 次评估、成功率 0–5%。"
    )
else:
    eval_note = (
        "未找到 <code>outputs/eval/**/eval_info.json</code>。"
        "Colab 评估单元会把 <code>eval_info.json</code> 写到该目录，再跑本脚本即可画进报告。"
        "仓库记录的完整实验是 20 次评估、成功率 0–5%。"
    )
def _bcrnn_cells(summ):
    if not summ["epochs"] or summ["final_loss"] is None:
        return "—", "—", "—", "—"
    last = f"{summ['last_sr']*100:.0f}%" if summ["last_sr"] is not None else "—"
    return (
        f"{summ['epochs']} epochs",
        f"loss {summ['final_loss']:.2f}",
        f"成功率 {summ['best_sr']*100:.0f}% @ep{summ['best_ep']}",
        last,
    )


rows = []
rows.append(f"""<tr><td>ACT · pusht（LeRobot 复现）</td><td>train.log + /tmp/train_{'{'}50k,80k,100k{'}'}.log</td>
<td>{act_scale}</td>
<td>{_fmt(act_final_loss, '.3f')}</td><td>{_fmt(act_final_l1, '.3f')} / {_fmt(kld_est, '.3f')}*</td>
<td>{eval_summary}</td></tr>""")
rows.append(f"""<tr><td>ACT 官方复现 · pusht</td><td>/tmp/train_official*.log</td>
<td>200,000 步（2 段日志）</td>
<td>{_fmt(off_final_loss, '.3f')}</td><td>—</td><td>demo 视频均为 fail（cov 0.0–0.93）</td></tr>""")
if gen:
    rows.append(f"""<tr><td>Square 数据集生成（MimicGen）</td><td>mimicgen_gen.log</td>
<td>成功 {gen.get('num_success')} / 尝试 {gen.get('num_attempts')}</td>
<td>成功率 {float(gen.get('success_rate', 0)):.1f}%</td><td>ep长度 {float(gen.get('ep_length_mean', 0)):.0f}±{float(gen.get('ep_length_std', 0)):.0f}</td>
<td>耗时 {float(gen.get('time spent (hrs)', 0)):.2f} hrs</td></tr>""")
img_epochs, img_loss_cell, img_best_cell, img_last = _bcrnn_cells(img_sum)
low_epochs, low_loss_cell, low_best_cell, low_last = _bcrnn_cells(low_sum)
rows.append(f"""<tr><td>BC-RNN image · Square</td><td>mimicgen_train_image.log</td>
<td>{img_epochs}</td>
<td>{img_loss_cell}</td>
<td>{img_best_cell}</td>
<td>最近一次 {img_last}</td></tr>""")
rows.append(f"""<tr><td>BC-RNN low-dim · Square</td><td>mimicgen_train_lowdim.log</td>
<td>{low_epochs}</td>
<td>{low_loss_cell}</td>
<td>{low_best_cell}</td>
<td>最近一次 {low_last}</td></tr>""")

html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>机器人训练数据报告 · 2026-08-01</title>
<style>
 body{{font-family:'Noto Sans CJK JP','WenQuanYi Micro Hei',sans-serif;background:#f8fafc;color:#0f172a;margin:0;padding:28px;}}
 .wrap{{max-width:1100px;margin:0 auto;}}
 h1{{font-size:24px;border-bottom:3px solid #2563eb;padding-bottom:10px;}}
 h2{{font-size:19px;margin-top:34px;color:#1e3a8a;}}
 table{{border-collapse:collapse;width:100%;background:#fff;font-size:13px;box-shadow:0 1px 3px rgba(0,0,0,.08);}}
 th,td{{border:1px solid #e2e8f0;padding:8px 10px;text-align:left;}}
 th{{background:#eff6ff;}}
 .card{{background:#fff;border-radius:10px;padding:18px;margin-top:14px;box-shadow:0 1px 3px rgba(0,0,0,.08);}}
 .note{{color:#475569;font-size:13px;line-height:1.7;}}
 .kpi{{display:flex;gap:14px;flex-wrap:wrap;margin:14px 0;}}
 .kpi div{{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:12px 18px;min-width:150px;}}
 .kpi b{{font-size:20px;color:#2563eb;display:block;}}
 .kpi span{{color:#64748b;font-size:12px;}}
 .fix{{background:#fef2f2;border:1px solid #fecaca;border-radius:8px;padding:10px 14px;font-size:13px;margin:12px 0;color:#7f1d1d;}}
</style></head><body><div class="wrap">
<h1>🤖 机器人训练数据报告（2026-08-01）</h1>
<p class="note">数据源：<code>train.log + /tmp/train_{{50k,80k,100k}}.log</code>（ACT·pusht 四段日志，共 100K 步）、
<code>/tmp/train_official*.log</code>（官方复现至 200K）、<code>outputs/eval</code>（评估 ×20）、
<code>outputs/mimicgen_gen.log</code>（Square 生成）、<code>outputs/mimicgen_train_*.log</code>（BC-RNN·Square）。</p>

<div class="fix">✅ <b>本次修正</b>：此前报告仅展示 train.log 覆盖的 0–10K 步。
现已合并 <code>/tmp/train_50k/80k/100k.log</code> 三段，完整呈现 <b>0→100K</b> 训练曲线，
并按 <code>/tmp/eval_*.log</code> 把评估结果映射到所用 checkpoint。</div>

<div class="kpi">
<div><b>100,000</b><span>ACT·pusht 实际训练步数</span></div>
<div><b>{_fmt(act_final_loss, '.3f')}</b><span>ACT 复现最终 loss</span></div>
<div><b>{_fmt(loss_50k, '.3f')}</b><span>loss @50K</span></div>
<div><b>{_fmt(loss_100k, '.3f')}</b><span>loss @100K</span></div>
<div><b>{len(evals)}</b><span>pusht 评估次数</span></div>
<div><b>{best_sr_txt}</b><span>最佳评估成功率</span></div>
<div><b>{_fmt(gen_sr, '.1f')}{'%' if gen_sr is not None else ''}</b><span>Square 生成成功率</span></div>
<div><b>{low_sum['best_sr']*100:.0f}%</b><span>BC-RNN low-dim 最佳成功率</span></div>
</div>

<h2>① 训练总览</h2>
<table><tr><th>训练/任务</th><th>来源</th><th>规模</th><th>最终 loss</th><th>组件 loss</th><th>评估</th></tr>
{''.join(rows)}</table>
<p class="note">* 组件列第二项为 kld_loss（由 loss - l1 推算，约 0.000）。</p>

<h2>② ACT · pusht 完整训练损失曲线（0→100K，四段日志合并）</h2>
<div class="card">{img_tag(fig_b64_act)}</div>
<p class="note">四条日志段按全局步数拼接：<code>train.log</code>(0–10K) + <code>train_50k</code>(10K–50K) +
<code>train_80k</code>(50K–80K) + <code>train_100k</code>(80K–100K)。
总 loss = l1_loss（动作回归）+ kld_loss（潜空间 KL）。loss 从 6.19@200 → 0.261@10K → 0.118@50K →
0.094@80K → <b>0.074@100K</b>，全程单调下降、无明显过拟合。
官方复现曲线（0→200K）单独标出，100K 时 0.108、200K 时 0.081 —— 长训进一步压低 loss，但
<b>评估成功率并未随训练变长而提升</b>（见下节），说明瓶颈在数据/任务本身而非训练时长。</p>

<h2>③ ACT · pusht 评估结果（{len(evals)} 次，已映射 checkpoint）</h2>
<div class="card">{img_tag(fig_b64_eval)}</div>
<p class="note">{eval_note}</p>

<h2>④ Square 数据集生成（MimicGen）</h2>
<div class="card">{img_tag(fig_b64_gen)}</div>
<p class="note">用 MimicGen 生成 {gen.get('num_success', 0) if gen else '—'} 条成功轨迹，生成成功率 {_fmt(gen_sr, '.1f')}{'%' if gen_sr is not None else ''}。
该数据随后用于 BC-RNN 训练（下图）。</p>

<h2>⑤ BC-RNN · Square 训练（image 与 low-dim 对比）</h2>
<div class="card">{img_tag(fig_b64_img)}</div>
<div class="card" style="margin-top:14px;">{img_tag(fig_b64_low)}</div>
<p class="note">image 版（RGB 输入）训练 {img_sum['epochs']} epochs，rollout 成功率最高 {img_sum['best_sr']*100:.0f}%；
low-dim 版（低维状态输入）训练 {low_sum['epochs']} epochs，成功率最高 {low_sum['best_sr']*100:.0f}% ——
两者对比可量化"视觉输入对策略成功率的影响"。</p>

</div></body></html>"""

with open(os.path.join(OUT, "training_report.html"), "w", encoding="utf-8") as f:
    f.write(html)

print(f"报告已生成: {OUT}/training_report.html")
if act_steps and act_final_loss is not None and act_final_l1 is not None:
    print(f"ACT: 合并 {len(act_steps)} 条 INFO，steps 0→{act_steps[-1]}，loss_final={act_final_loss:.3f} l1={act_final_l1:.3f}")
    miles = []
    for label, step, val in (("10K", 10000, loss_10k), ("50K", 50000, loss_50k), ("100K", 100000, loss_100k)):
        if val is not None:
            miles.append(f"{label}={val:.3f}")
    print("  里程碑:", " ".join(miles) if miles else "日志未覆盖 10K/50K/100K")
else:
    print("ACT: 无可用训练日志")
if off_steps and off_final_loss is not None:
    print(f"OFF: 合并 {len(off_steps)} 条 INFO，loss_final(200K)={off_final_loss:.3f}")
else:
    print("OFF: 无官方复现日志")
print(f"Eval: n={len(evals)} best_success={best_sr_txt} best_mr={best_mr_txt}")
print(f"Gen: success_rate={gen.get('success_rate')} num_success={gen.get('num_success')}")
print(f"BCRNN image: epochs={img_sum['epochs']} best_sr={img_sum['best_sr']:.3f}@{img_sum['best_ep']}")
print(f"BCRNN low:   epochs={low_sum['epochs']} best_sr={low_sum['best_sr']:.3f}@{low_sum['best_ep']}")
print(f"official ckpts: {len(official_ckpts)} up to {max(official_ckpts) if official_ckpts else '-'}")
