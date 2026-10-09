#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""拟合「最优验证损失 ~ a + b·ln(数据量)」并写入 HTML 报告。

输入是一份 YAML 或 CSV：每行一次训练的名字、数据量（小时或条数）、日志路径。
从日志里取验证损失的最小值，对 ln(数据量) 做线性拟合，把散点与拟合线画进报告。
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

_VAL_RE = re.compile(
    r'(?:'
    r'val(?:idation)?[\s_-]*loss'
    r'|"(?:val_loss|validation_loss|Validation_Loss)"'
    r')\s*[:=]\s*'
    r'(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)',
    re.IGNORECASE,
)

_SIZE_KINDS = ("hours", "episodes", "size")
_KIND_LABEL = {"hours": "小时", "episodes": "条", "size": "数据量"}


def extract_best_val_loss(text):
    """返回日志中的最小验证损失；没有验证损失时返回 None。

    传入已存在的文件路径时读取文件；否则把参数当作日志正文。
    """
    if not isinstance(text, str) or os.path.isfile(text):
        with open(text, encoding="utf-8", errors="ignore") as handle:
            text = handle.read()
    vals = [float(m.group(1)) for m in _VAL_RE.finditer(text)]
    if not vals:
        return None
    return min(vals)


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
    return {"intercept": intercept, "slope": slope, "r2": r2}


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
        loss = extract_best_val_loss(run["log"])
        if loss is None:
            skipped.append(run["name"])
            continue
        points.append({
            "name": run["name"],
            "size": run["size"],
            "size_kind": run["size_kind"],
            "loss": loss,
            "log": run["log"],
        })
    kinds = {p["size_kind"] for p in points}
    if len(points) < 2 or len(kinds) != 1:
        return None
    points.sort(key=lambda p: p["size"])
    fit = fit_log_linear([p["size"] for p in points], [p["loss"] for p in points])
    return {
        "points": points,
        "fit": fit,
        "skipped": skipped,
        "size_kind": points[0]["size_kind"],
    }


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def plot_b64(result):
    points = result["points"]
    fit = result["fit"]
    kind = _KIND_LABEL[result["size_kind"]]
    xs = [math.log(p["size"]) for p in points]
    ys = [p["loss"] for p in points]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.scatter(xs, ys, s=46, color="#2563eb", zorder=3)
    for point, x, y in zip(points, xs, ys):
        ax.annotate(point["name"], xy=(x, y), xytext=(6, 6),
                    textcoords="offset points", fontsize=8, color="#334155")
    x0, x1 = min(xs), max(xs)
    pad = (x1 - x0) * 0.08 or 0.2
    line_x = [x0 - pad, x1 + pad]
    line_y = [fit["intercept"] + fit["slope"] * x for x in line_x]
    ax.plot(line_x, line_y, color="#f59e0b", lw=1.8,
            label="L = %.4f + (%.4f)·ln(N)" % (fit["intercept"], fit["slope"]))
    ax.set_xlabel("ln(数据量 / %s)" % kind)
    ax.set_ylabel("最优验证损失")
    ax.set_title("缩放律：最优验证损失 vs ln(数据量)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return _fig_to_b64(fig)


def render_section(result):
    """返回可嵌进训练报告的 HTML 片段。"""
    fit = result["fit"]
    kind = _KIND_LABEL[result["size_kind"]]
    per_decade = fit["slope"] * math.log(10)
    rows = []
    for point in result["points"]:
        rows.append(
            "<tr><td>%s</td><td>%.4g %s</td><td>%.6f</td><td><code>%s</code></td></tr>"
            % (html.escape(point["name"]), point["size"], kind, point["loss"],
               html.escape(point["log"]))
        )
    skipped = ""
    if result["skipped"]:
        skipped = "<p class=\"note\">已跳过缺少日志或没有验证损失的 run：%s。</p>" % html.escape(
            "、".join(result["skipped"])
        )
    img = plot_b64(result)
    return """<h2>数据缩放律（最优验证损失 vs ln(数据量)）</h2>
<div class="card"><img src="data:image/png;base64,%s" style="max-width:100%%;border:1px solid #e2e8f0;border-radius:8px;"></div>
<table><tr><th>run</th><th>数据量</th><th>最优验证损失</th><th>日志</th></tr>
%s</table>
<p class="note">拟合（自然对数）：<b>L = %.4f + (%.4f)·ln(N)</b>，R² = %.4f。
数据量每增加 10 倍，拟合损失变化 %.4f。
这是 EgoScale 报告的 log-linear 关系：最优验证损失对预训练小时数（或条数）的对数近似线性。
每个 run 取日志中的最小验证损失，不使用训练损失。</p>
%s""" % (img, "".join(rows), fit["intercept"], fit["slope"], fit["r2"], per_decade, skipped)


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
    print("L = %.4f + (%.4f)·ln(N)  R2=%.4f  n=%d" % (
        fit["intercept"], fit["slope"], fit["r2"], len(result["points"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
