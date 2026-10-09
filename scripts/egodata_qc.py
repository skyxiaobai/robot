#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对统一 episode 做 QC，写出产出率 HTML 和 CSV。

示例:
    python scripts/egodata_qc.py --episodes outputs/egodex_unified \\
        --html outputs/yield_report.html --csv outputs/yield_episodes.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from egodata.qc import qc_episode, write_yield_reports, yield_report  # noqa: E402
from egodata.schema import iter_episode_paths, load_episode  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description="统一 episode 的 QC 与产出率")
    parser.add_argument("--episodes", required=True, help="episode JSON 文件或目录")
    parser.add_argument("--html", required=True, help="HTML 报告路径")
    parser.add_argument("--csv", required=True, help="逐条 CSV 路径")
    args = parser.parse_args(argv)
    paths = iter_episode_paths(args.episodes)
    if not paths:
        print("没有找到 episode JSON", file=sys.stderr)
        return 1
    results = [qc_episode(load_episode(path)) for path in paths]
    report = yield_report(results)
    write_yield_reports(report, args.html, args.csv)
    print("yield %.3f  (%d/%d frames, rejected %d/%d)" % (
        report["yield"],
        report["usable_frames"],
        report["raw_frames"],
        report["rejected_episodes"],
        report["episodes"],
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
