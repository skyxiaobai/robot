#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""拟合「最优验证损失 ~ a + b·ln(数据量)」并写入 HTML 报告。

输入是一份 YAML 或 CSV：每行一次训练的名字、数据量（小时或条数）、日志路径。
从日志里取验证损失。同一数据量的多种子先取均值，并画标准差。
同时拟合对数直线和饱和幂律 ``L = L∞ + A·N^(-α)``（不依赖 scipy），
两条都写入报告；R² 更高的那条作为图上的默认曲线，打平则用对数直线。
若日志里有 ``copy_current_wrist``、``trans_mse``、``rot_mse``，同时画出保持不动
基线、平移/旋转拆分，并对验证损失与基线的比值做同样的两种拟合。
没有 --runs、配置不存在、或有效 run 不足 2 个时直接跳过，退出码为 0。

示例:
  python scripts/scaling_law.py --runs examples/scaling_law_runs.yaml
"""
import argparse
import base64
import csv
import html
import io
import math
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

_cjk = None
for _font in font_manager.fontManager.ttflist:
    if "Noto Sans CJK" in _font.name or "WenQuanYi" in _font.name or "Hei" in _font.name:
        _cjk = _font.name
        break
if _cjk:
    plt.rcParams["font.sans-serif"] = [_cjk]
plt.rcParams["axes.unicode_minus"] = False

_NUMBER = r'(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)'
_VAL_RE = re.compile(
    r'(?:'
    r'val(?:idation)?[\s_-]*loss'
    r'|"(?:val_loss|validation_loss|Validation_Loss)"'
    r')\s*[:=]\s*'
    + _NUMBER,
    re.IGNORECASE,
)
# copy_trans_mse 不能被当成 trans_mse。前导下划线或字母都排除。
_EXTRA_RES = (
    ("copy_current_wrist", re.compile(r'(?<![A-Za-z_])copy_current_wrist\s*[:=]\s*' + _NUMBER)),
    ("trans_mse", re.compile(r'(?<![A-Za-z_])trans_mse\s*[:=]\s*' + _NUMBER)),
    ("rot_mse", re.compile(r'(?<![A-Za-z_])rot_mse\s*[:=]\s*' + _NUMBER)),
)

_SIZE_KINDS = ("hours", "episodes", "size")
_KIND_LABEL = {"hours": "小时", "episodes": "条", "size": "数据量"}


def size_axis_label(kind):
    """横轴文字。单位本身已经是「数据量」时不再写成「数据量 / 数据量」。"""
    return "ln(N)（%s）" % kind


def format_metric(value):
    """损失和拟合系数。小于 0.01 时用科学计数法，避免 1e-4 被写成 0.0000。"""
    number = float(value)
    if number != 0.0 and abs(number) < 1e-2:
        return "%.6e" % number
    return "%.6f" % number


def _load_text(text):
    if not isinstance(text, str) or os.path.isfile(text):
        with open(text, encoding="utf-8", errors="ignore") as handle:
            return handle.read()
    return text


_AGG_FIELDS = (
    ("val_loss", "val_loss_mean"),
    ("val_loss_std", "val_loss_std"),
    ("copy_current_wrist", "copy_current_wrist_mean"),
    ("copy_current_wrist_std", "copy_current_wrist_std"),
    ("trans_mse", "trans_mse_mean"),
    ("trans_mse_std", "trans_mse_std"),
    ("rot_mse", "rot_mse_mean"),
    ("rot_mse_std", "rot_mse_std"),
    ("val_baseline_ratio", "val_baseline_ratio_mean"),
    ("val_baseline_ratio_std", "val_baseline_ratio_std"),
)


def _last_named_number(name, text):
    pattern = re.compile(r"(?<![A-Za-z_])" + re.escape(name) + r"\s*[:=]\s*" + _NUMBER)
    found = pattern.findall(text)
    if not found:
        return None
    return float(found[-1])


def _metrics_from_seed_blocks(text):
    matches = list(_VAL_RE.finditer(text))
    if not matches:
        return None
    best = None
    for index, match in enumerate(matches):
        start = 0 if index == 0 else match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        window = text[start:end]
        record = {"val_loss": float(match.group(1))}
        for key, pattern in _EXTRA_RES:
            found = pattern.findall(window)
            if found:
                record[key] = float(found[-1])
        if best is None or record["val_loss"] < best["val_loss"]:
            best = record
    return best


def extract_run_metrics(text):
    """读取验证损失，以及同一段里的基线、平移和旋转。

    传入已存在的文件路径时读取文件；否则把参数当作日志正文。
    日志里若有 ``val_loss_mean``，用多种子均值，不用其中最小的那一次。
    否则取最小的 ``val_loss``。两条 ``val_loss`` 之间的指标算在前一条上。
    ``copy_trans_mse`` 不会被当成 ``trans_mse``。没有验证损失时返回 None。
    """
    text = _load_text(text)
    if _last_named_number("val_loss_mean", text) is not None:
        record = {}
        for key, name in _AGG_FIELDS:
            value = _last_named_number(name, text)
            if value is not None:
                record[key] = value
        if "val_loss" not in record:
            return None
    else:
        record = _metrics_from_seed_blocks(text)
        if record is None:
            return None
    copy_loss = record.get("copy_current_wrist")
    if copy_loss and "val_baseline_ratio" not in record:
        record["val_baseline_ratio"] = record["val_loss"] / copy_loss
    return record


def extract_best_val_loss(text):
    """返回日志中的最小验证损失；没有验证损失时返回 None。

    传入已存在的文件路径时读取文件；否则把参数当作日志正文。
    小数和科学计数法都可以，例如 ``val_loss: 1.085100e-04``。
    """
    metrics = extract_run_metrics(text)
    if not metrics:
        return None
    return metrics["val_loss"]


def fit_log_linear(sizes, losses):
    """最小二乘拟合 loss = intercept + slope * ln(size)。"""
    xs = [math.log(float(s)) for s in sizes]
    ys = [float(y) for y in losses]
    n = len(xs)
    if n < 2:
        raise ValueError("至少需要 2 个点才能拟合缩放律")
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    var = sum((x - x_mean) ** 2 for x in xs)
    if var == 0:
        raise ValueError("数据量全部相同，无法对 log(数据量) 拟合")
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / var
    intercept = y_mean - slope * x_mean
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return {"form": "log", "intercept": intercept, "slope": slope, "r2": r2}


def _r_squared(ys, predicted):
    mean = sum(ys) / len(ys)
    ss_tot = sum((y - mean) ** 2 for y in ys)
    ss_res = sum((y - pred) ** 2 for y, pred in zip(ys, predicted))
    if ss_tot == 0:
        return 1.0
    return 1.0 - ss_res / ss_tot


def _ols_line(xs, ys):
    n = len(xs)
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    var = sum((x - x_mean) ** 2 for x in xs)
    if var == 0:
        return None
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / var
    return y_mean - slope * x_mean, slope


def predict_loss(fit, size):
    """按拟合形式计算某个数据量上的损失。"""
    size = float(size)
    if fit.get("form") == "power":
        return fit["linf"] + fit["A"] * (size ** (-fit["alpha"]))
    return fit["intercept"] + fit["slope"] * math.log(size)


def fit_power_saturating(sizes, losses):
    """拟合 ``L = L∞ + A·N^(-α)``，α>0 且 A>0。找不到下降曲线时返回 None。

    对每个 α，A 和 L∞ 有闭式最小二乘。α 在对数网格上搜索后再局部加密。
    不使用 scipy。
    """
    sizes = [float(s) for s in sizes]
    ys = [float(y) for y in losses]
    if len(sizes) < 3 or any(s <= 0 for s in sizes):
        return None
    logs = [math.log(s) for s in sizes]
    best = None

    def consider(alpha):
        nonlocal best
        if alpha <= 0:
            return
        xs = [math.exp(-alpha * log_s) for log_s in logs]
        coef = _ols_line(xs, ys)
        if coef is None:
            return
        linf, amp = coef
        if amp <= 0:
            return
        predicted = [linf + amp * x for x in xs]
        score = _r_squared(ys, predicted)
        if best is None or score > best["r2"]:
            best = {"form": "power", "linf": linf, "A": amp, "alpha": alpha, "r2": score}

    for index in range(64):
        alpha = math.exp(math.log(0.02) + (math.log(4.0) - math.log(0.02)) * index / 63.0)
        consider(alpha)
    if best is None:
        return None
    center = best["alpha"]
    for index in range(41):
        consider(center * math.exp((index - 20) * 0.06))
    return best


def select_fit(log_fit, power_fit):
    """R² 更高的作为默认。打平或没有幂律时用对数直线。"""
    if power_fit is not None and power_fit["r2"] > log_fit["r2"]:
        return power_fit
    return log_fit


def describe_fit(fit):
    """一行公式，给报告和终端用。"""
    if fit.get("form") == "power":
        return "L = %s + %s·N^(-%s)" % (
            format_metric(fit["linf"]), format_metric(fit["A"]), format_metric(fit["alpha"]),
        )
    return "L = %s + (%s)·ln(N)" % (format_metric(fit["intercept"]), format_metric(fit["slope"]))


def label_offsets(xs, ys):
    """给靠得很近的点错开标注，避免 256k 和全量叠在一起。

    返回与输入等长的 ``(dx, dy)``，单位是点。
    """
    count = len(xs)
    offsets = [(8, 8)] * count
    if count < 2:
        return offsets
    xspan = max(xs) - min(xs) or 1.0
    yspan = max(ys) - min(ys) or 1.0
    candidates = [(8, 10), (8, -16), (-72, 10), (-72, -16), (8, 24), (8, -30)]
    placed = []
    for index in sorted(range(count), key=lambda i: (xs[i], ys[i])):
        chosen = candidates[-1]
        for cand in candidates:
            clear = True
            for other, pdx, pdy in placed:
                far_point = (
                    abs((xs[index] - xs[other]) / xspan) > 0.08
                    or abs((ys[index] - ys[other]) / yspan) > 0.08
                )
                far_label = abs(cand[0] - pdx) > 24 or abs(cand[1] - pdy) > 12
                if not far_point and not far_label:
                    clear = False
                    break
            if clear:
                chosen = cand
                break
        offsets[index] = chosen
        placed.append((index, chosen[0], chosen[1]))
    return offsets


def _sample_std(values):
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _group_by_size(points):
    """同一数据量的多次运行合成一个点：均值和样本标准差。"""
    order = []
    buckets = {}
    for point in points:
        key = point["size"]
        if key not in buckets:
            order.append(key)
            buckets[key] = []
        buckets[key].append(point)
    grouped = []
    for size in order:
        group = buckets[size]
        if len(group) == 1:
            grouped.append(group[0])
            continue
        losses = [point["loss"] for point in group]
        merged = {
            "name": "、".join(point["name"] for point in group),
            "size": size,
            "size_kind": group[0]["size_kind"],
            "loss": sum(losses) / len(losses),
            "loss_std": _sample_std(losses),
            "n_seeds": len(group),
            "log": " ; ".join(point["log"] for point in group),
        }
        for key in ("copy_current_wrist", "trans_mse", "rot_mse", "val_baseline_ratio"):
            values = [point[key] for point in group if point.get(key) is not None]
            if not values:
                continue
            merged[key] = sum(values) / len(values)
            if len(values) >= 2:
                merged[key + "_std"] = _sample_std(values)
        grouped.append(merged)
    return grouped


def _unquote(value):
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def _resolve_log(log_path, config_path):
    if os.path.isabs(log_path) and os.path.isfile(log_path):
        return log_path
    config_dir = os.path.dirname(os.path.abspath(config_path))
    candidates = [
        log_path,
        os.path.join(config_dir, log_path),
        os.path.join(os.path.dirname(config_dir), log_path),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    if os.path.isabs(log_path):
        return log_path
    return os.path.abspath(os.path.join(config_dir, log_path))


def _normalize_run(raw, config_path):
    name = str(raw.get("name", "")).strip()
    log_path = str(raw.get("log") or raw.get("log_path") or "").strip()
    if not name or not log_path:
        raise ValueError("每条 run 都需要 name 和 log")
    present = [k for k in _SIZE_KINDS if str(raw.get(k, "")).strip() != ""]
    if len(present) != 1:
        raise ValueError("%s 需要且只能填写 hours、episodes、size 之一" % name)
    kind = present[0]
    size = float(raw[kind])
    if size <= 0:
        raise ValueError("%s 的数据量必须为正数" % name)
    return {
        "name": name,
        "size": size,
        "size_kind": kind,
        "log": _resolve_log(log_path, config_path),
    }


def _load_yaml(path):
    runs = []
    current = None
    with open(path, encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, 1):
            line = raw.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            if re.match(r"^runs:\s*(\[\])?\s*$", line.strip()):
                continue
            named = re.match(r"^\s*-\s+name:\s*(.+?)\s*$", line)
            if named:
                current = {"name": _unquote(named.group(1))}
                runs.append(current)
                continue
            field = re.match(r"^\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*?)\s*$", line)
            if field and current is not None:
                current[field.group(1)] = _unquote(field.group(2))
                continue
            raise ValueError("%s:%d 无法解析: %s" % (path, lineno, raw.rstrip()))
    return [_normalize_run(item, path) for item in runs]


def _load_csv(path):
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return []
        rows = []
        for row in reader:
            cleaned = {k.strip(): (v or "").strip() for k, v in row.items() if k}
            if not any(cleaned.values()):
                continue
            rows.append(cleaned)
    return [_normalize_run(item, path) for item in rows]


def load_runs(path):
    """读取 YAML 或 CSV。相对日志路径先相对配置文件目录解析。"""
    if path.lower().endswith(".csv"):
        return _load_csv(path)
    return _load_yaml(path)


def analyze_config(path):
    """返回拟合结果；输入不足时返回 None（调用方跳过）。"""
    if not path or not os.path.isfile(path):
        return None
    runs = load_runs(path)
    if len(runs) < 2:
        return None
    points = []
    skipped = []
    for run in runs:
        if not os.path.isfile(run["log"]):
            skipped.append(run["name"])
            continue
        metrics = extract_run_metrics(run["log"])
        if not metrics:
            skipped.append(run["name"])
            continue
        points.append({
            "name": run["name"],
            "size": run["size"],
            "size_kind": run["size_kind"],
            "loss": metrics["val_loss"],
            "loss_std": metrics.get("val_loss_std"),
            "copy_current_wrist": metrics.get("copy_current_wrist"),
            "copy_current_wrist_std": metrics.get("copy_current_wrist_std"),
            "trans_mse": metrics.get("trans_mse"),
            "trans_mse_std": metrics.get("trans_mse_std"),
            "rot_mse": metrics.get("rot_mse"),
            "rot_mse_std": metrics.get("rot_mse_std"),
            "val_baseline_ratio": metrics.get("val_baseline_ratio"),
            "val_baseline_ratio_std": metrics.get("val_baseline_ratio_std"),
            "log": run["log"],
        })
    kinds = {p["size_kind"] for p in points}
    if len(points) < 2 or len(kinds) != 1:
        return None
    points.sort(key=lambda p: p["size"])
    points = _group_by_size(points)
    if len(points) < 2:
        return None
    log_fit = fit_log_linear([p["size"] for p in points], [p["loss"] for p in points])
    power_fit = fit_power_saturating([p["size"] for p in points], [p["loss"] for p in points])
    fit = select_fit(log_fit, power_fit)
    ratio_points = [p for p in points if p.get("val_baseline_ratio") is not None]
    ratio_log_fit = None
    ratio_power_fit = None
    ratio_fit = None
    if len(ratio_points) >= 2:
        ratio_log_fit = fit_log_linear(
            [p["size"] for p in ratio_points],
            [p["val_baseline_ratio"] for p in ratio_points],
        )
        ratio_power_fit = fit_power_saturating(
            [p["size"] for p in ratio_points],
            [p["val_baseline_ratio"] for p in ratio_points],
        )
        ratio_fit = select_fit(ratio_log_fit, ratio_power_fit)
    return {
        "points": points,
        "fit": fit,
        "log_fit": log_fit,
        "power_fit": power_fit,
        "ratio_fit": ratio_fit,
        "ratio_log_fit": ratio_log_fit,
        "ratio_power_fit": ratio_power_fit,
        "skipped": skipped,
        "size_kind": points[0]["size_kind"],
    }


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _series(points, key):
    xs = []
    ys = []
    for point in points:
        value = point.get(key)
        if value is None:
            continue
        xs.append(math.log(point["size"]))
        ys.append(value)
    return xs, ys


def _draw_fit(ax, sizes, fit, color, primary):
    if not sizes or not fit:
        return
    logs = [math.log(float(size)) for size in sizes]
    x0, x1 = min(logs), max(logs)
    pad = (x1 - x0) * 0.08 or 0.2
    line_x = [x0 - pad + (x1 - x0 + 2 * pad) * i / 40.0 for i in range(41)]
    line_y = [predict_loss(fit, math.exp(x)) for x in line_x]
    style = "-" if primary else "--"
    width = 1.8 if primary else 1.2
    prefix = "默认 " if primary else ""
    ax.plot(line_x, line_y, color=color, lw=width, ls=style, label=prefix + describe_fit(fit))


def plot_b64(result):
    points = result["points"]
    fit = result["fit"]
    kind = _KIND_LABEL[result["size_kind"]]
    xs = [math.log(p["size"]) for p in points]
    has_baseline = any(p.get("copy_current_wrist") is not None for p in points)
    has_split = any(p.get("trans_mse") is not None or p.get("rot_mse") is not None for p in points)
    has_ratio = result.get("ratio_fit") is not None
    panels = 1 + int(has_split) + int(has_ratio)
    fig, axes = plt.subplots(1, panels, figsize=(8.2 * panels, 4.6), squeeze=False)
    ax = axes[0, 0]
    losses = [p["loss"] for p in points]
    loss_err = [p.get("loss_std") or 0.0 for p in points]
    if any(loss_err):
        ax.errorbar(xs, losses, yerr=loss_err, fmt="o", ms=6, color="#2563eb",
                    ecolor="#93c5fd", capsize=3, zorder=3, label="验证损失")
    else:
        ax.scatter(xs, losses, s=46, color="#2563eb", zorder=3, label="验证损失")
    offsets = label_offsets(xs, losses)
    for point, x, y, (dx, dy) in zip(points, xs, losses, offsets):
        ax.annotate(point["name"], xy=(x, y), xytext=(dx, dy),
                    textcoords="offset points", fontsize=8, color="#334155")
    _draw_fit(ax, [p["size"] for p in points], fit, "#f59e0b", True)
    other = result.get("power_fit") if fit.get("form") != "power" else result.get("log_fit")
    _draw_fit(ax, [p["size"] for p in points], other, "#94a3b8", False)
    if has_baseline:
        base_x, base_y = _series(points, "copy_current_wrist")
        ax.plot(base_x, base_y, color="#64748b", lw=1.6, marker="s", ms=5, label="保持不动基线")
    ax.set_xlabel(size_axis_label(kind))
    ax.set_ylabel("最优验证损失")
    ax.set_title("缩放律：最优验证损失 vs ln(数据量)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    panel = 1
    if has_split:
        ax_split = axes[0, panel]
        panel += 1
        trans_x, trans_y = _series(points, "trans_mse")
        rot_x, rot_y = _series(points, "rot_mse")
        if trans_x:
            ax_split.plot(trans_x, trans_y, color="#0f766e", lw=1.6, marker="o", label="平移 trans_mse")
        if rot_x:
            ax_split.plot(rot_x, rot_y, color="#b45309", lw=1.6, marker="o", label="旋转 rot_mse")
        ax_split.set_xlabel(size_axis_label(kind))
        ax_split.set_ylabel("分量 MSE")
        ax_split.set_title("平移 / 旋转")
        ax_split.grid(alpha=0.25)
        ax_split.legend(loc="best", fontsize=8)
    if has_ratio:
        ax_ratio = axes[0, panel]
        ratio_x, ratio_y = _series(points, "val_baseline_ratio")
        ratio_err = [p.get("val_baseline_ratio_std") or 0.0 for p in points if p.get("val_baseline_ratio") is not None]
        if any(ratio_err):
            ax_ratio.errorbar(ratio_x, ratio_y, yerr=ratio_err, fmt="o", ms=5, color="#2563eb",
                              ecolor="#93c5fd", capsize=3, label="val / 基线")
        else:
            ax_ratio.plot(ratio_x, ratio_y, color="#2563eb", lw=1.6, marker="o", label="val / 基线")
        ax_ratio.axhline(1.0, color="#94a3b8", lw=1.0, label="与基线持平")
        ratio_sizes = [p["size"] for p in points if p.get("val_baseline_ratio") is not None]
        _draw_fit(ax_ratio, ratio_sizes, result.get("ratio_fit"), "#f59e0b", True)
        ratio_other = result.get("ratio_power_fit") if result.get("ratio_fit", {}).get("form") != "power" else result.get("ratio_log_fit")
        _draw_fit(ax_ratio, ratio_sizes, ratio_other, "#94a3b8", False)
        ax_ratio.set_xlabel(size_axis_label(kind))
        ax_ratio.set_ylabel("验证损失 / 基线")
        ax_ratio.set_title("相对基线")
        ax_ratio.grid(alpha=0.25)
        ax_ratio.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _fmt_with_std(value, std):
    text = _fmt_optional(value)
    if value is None or not std:
        return text
    return text + " ± " + format_metric(std)


def render_section(result):
    """返回可嵌进训练报告的 HTML 片段。"""
    fit = result["fit"]
    kind = _KIND_LABEL[result["size_kind"]]
    has_extra = any(
        point.get("copy_current_wrist") is not None
        or point.get("trans_mse") is not None
        or point.get("rot_mse") is not None
        for point in result["points"]
    )
    if has_extra:
        header = (
            "<tr><th>run</th><th>数据量</th><th>最优验证损失</th>"
            "<th>保持不动基线</th><th>val/基线</th><th>平移</th><th>旋转</th><th>日志</th></tr>"
        )
    else:
        header = "<tr><th>run</th><th>数据量</th><th>最优验证损失</th><th>日志</th></tr>"
    rows = []
    for point in result["points"]:
        if has_extra:
            rows.append(
                "<tr><td>%s</td><td>%.4g %s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td><code>%s</code></td></tr>"
                % (
                    html.escape(point["name"]), point["size"], kind,
                    _fmt_with_std(point["loss"], point.get("loss_std")),
                    _fmt_optional(point.get("copy_current_wrist")),
                    _fmt_optional(point.get("val_baseline_ratio")),
                    _fmt_optional(point.get("trans_mse")),
                    _fmt_optional(point.get("rot_mse")),
                    html.escape(point["log"]),
                )
            )
        else:
            rows.append(
                "<tr><td>%s</td><td>%.4g %s</td><td>%s</td><td><code>%s</code></td></tr>"
                % (html.escape(point["name"]), point["size"], kind,
                   _fmt_with_std(point["loss"], point.get("loss_std")),
                   html.escape(point["log"]))
            )
    skipped = ""
    if result["skipped"]:
        skipped = "<p class=\"note\">已跳过缺少日志或没有验证损失的 run：%s。</p>" % html.escape(
            "、".join(result["skipped"])
        )
    log_fit = result["log_fit"]
    power_fit = result.get("power_fit")
    power_text = describe_fit(power_fit) + "，R² = %.4f" % power_fit["r2"] if power_fit else "未找到 A&gt;0 的下降曲线"
    default_name = "饱和幂律" if fit.get("form") == "power" else "对数直线"
    ratio_note = ""
    ratio_fit = result.get("ratio_fit")
    if ratio_fit:
        ratio_note = (
            "<p class=\"note\">相对基线（验证损失 / copy_current_wrist）默认曲线是%s：<b>%s</b>，R² = %.4f。"
            "比值小于 1 表示好于保持不动。</p>"
            % (
                "饱和幂律" if ratio_fit.get("form") == "power" else "对数直线",
                describe_fit(ratio_fit), ratio_fit["r2"],
            )
        )
    img = plot_b64(result)
    return """<h2>数据缩放律（最优验证损失 vs ln(数据量)）</h2>
<div class="card"><img src="data:image/png;base64,%s" style="max-width:100%%;border:1px solid #e2e8f0;border-radius:8px;"></div>
<table>%s
%s</table>
<p class="note">默认曲线是<b>%s</b>：<b>%s</b>，R² = %.4f。
对数直线：%s，R² = %.4f。
饱和幂律 L = L∞ + A·N^(-α)：%s。
同一数据量有多次运行时，点是均值，误差线是样本标准差。
每个 run 取日志中的最小验证损失；若日志写了 val_loss_mean，则用多种子均值。不使用训练损失。</p>
%s%s""" % (
        img, header, "".join(rows),
        default_name, describe_fit(fit), fit["r2"],
        describe_fit(log_fit), log_fit["r2"],
        power_text, ratio_note, skipped,
    )


def _fmt_optional(value):
    if value is None:
        return "—"
    return format_metric(value)


def section_html(config_path):
    """供训练报告调用。无输入或有效点不足时返回空字符串。"""
    try:
        result = analyze_config(config_path)
    except (OSError, ValueError) as exc:
        print("缩放律章节跳过：%s" % exc)
        return ""
    if not result:
        return ""
    return render_section(result)


def render_report(result):
    section = render_section(result)
    return (
        "<!DOCTYPE html>\n"
        "<html lang=\"zh-CN\"><head><meta charset=\"utf-8\">\n"
        "<title>数据缩放律</title>\n"
        "<style>\n"
        " body{font-family:'Noto Sans CJK JP','WenQuanYi Micro Hei',sans-serif;"
        "background:#f8fafc;color:#0f172a;margin:0;padding:28px;}\n"
        " .wrap{max-width:1100px;margin:0 auto;}\n"
        " h1{font-size:24px;border-bottom:3px solid #2563eb;padding-bottom:10px;}\n"
        " h2{font-size:19px;margin-top:34px;color:#1e3a8a;}\n"
        " table{border-collapse:collapse;width:100%;background:#fff;font-size:13px;"
        "box-shadow:0 1px 3px rgba(0,0,0,.08);}\n"
        " th,td{border:1px solid #e2e8f0;padding:8px 10px;text-align:left;}\n"
        " th{background:#eff6ff;}\n"
        " .card{background:#fff;border-radius:10px;padding:18px;margin-top:14px;"
        "box-shadow:0 1px 3px rgba(0,0,0,.08);}\n"
        " .note{color:#475569;font-size:13px;line-height:1.7;}\n"
        "</style></head><body><div class=\"wrap\">\n"
        "<h1>数据缩放律</h1>\n"
        "<p class=\"note\">多次不同数据量的训练，取各自最优验证损失，拟合对 ln(数据量) 的直线。</p>\n"
        + section +
        "\n</div></body></html>"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="拟合验证损失对 log(数据量) 的缩放律")
    parser.add_argument("--runs", help="YAML 或 CSV：name, hours|episodes|size, log")
    parser.add_argument("--out", help="HTML 报告路径")
    args = parser.parse_args(argv)
    if not args.runs:
        print("未提供 --runs，缩放律分析已跳过。")
        return 0
    if not os.path.isfile(args.runs):
        print("缩放律配置不存在：%s，已跳过。" % args.runs)
        return 0
    try:
        result = analyze_config(args.runs)
    except (OSError, ValueError) as exc:
        print("缩放律配置无法读取：%s" % exc)
        return 1
    if not result:
        print("有效 run 不足（需要至少 2 个含验证损失、且数据量单位一致的 run），已跳过。")
        return 0
    out = args.out or os.path.join("outputs", "scaling_law_report.html")
    parent = os.path.dirname(os.path.abspath(out))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        handle.write(render_report(result))
    fit = result["fit"]
    print("缩放律报告已生成: %s" % out)
    print("默认 %s  R2=%.4f  n=%d" % (describe_fit(fit), fit["r2"], len(result["points"])))
    print("对数直线 %s  R2=%.4f" % (describe_fit(result["log_fit"]), result["log_fit"]["r2"]))
    if result.get("power_fit"):
        print("饱和幂律 %s  R2=%.4f" % (describe_fit(result["power_fit"]), result["power_fit"]["r2"]))
    ratio_fit = result.get("ratio_fit")
    if ratio_fit:
        print("val/baseline %s  R2=%.4f" % (describe_fit(ratio_fit), ratio_fit["r2"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
