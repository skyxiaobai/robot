#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""按环境 / 物体 / 任务 / 动作类型统计覆盖并标出空档。

示例:
    python scripts/egodata_coverage.py --episodes outputs/egodex_unified \\
        --html outputs/coverage_report.html --csv outputs/coverage_counts.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from egodata.coverage import coverage_report, write_coverage_reports  # noqa: E402
from egodata.schema import iter_episode_paths, load_episode  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description="采集覆盖度报告")
    parser.add_argument("--episodes", required=True, help="episode JSON 文件或目录")
    parser.add_argument("--html", required=True, help="HTML 报告路径")
    parser.add_argument("--csv", required=True, help="计数 CSV 路径")
    args = parser.parse_args(argv)
    paths = iter_episode_paths(args.episodes)
    if not paths:
        print("没有找到 episode JSON", file=sys.stderr)
        return 1
    episodes = [load_episode(path) for path in paths]
    report = coverage_report(episodes)
    write_coverage_reports(report, args.html, args.csv)
    print("episodes %d, gaps %d, csv %s" % (report["episodes"], len(report["gaps"]), args.csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
