#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""QC 通过的统一 episode → LeRobot v3.0（对应 lerobot 0.6.1）。

示例:
    python scripts/ego_to_lerobot.py \\
        --episodes outputs/open_data_demo/unified \\
        --yield-csv outputs/open_data_demo/yield_episodes.csv \\
        --out outputs/egodex_lerobot
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from egodata.lerobot_export import export_lerobot  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description="统一 episode 导出为 LeRobot v3.0")
    parser.add_argument("--episodes", required=True, help="统一 episode JSON 目录")
    parser.add_argument("--yield-csv", required=True, help="egodata_qc 写出的 CSV")
    parser.add_argument("--out", required=True, help="LeRobot 数据集根目录")
    parser.add_argument("--repo-id", default="local/egodex", help="写进导出说明的数据集 id")
    args = parser.parse_args(argv)
    summary = export_lerobot(args.episodes, args.yield_csv, args.out, repo_id=args.repo_id)
    print("episodes %d frames %d -> %s" % (summary["episodes"], summary["frames"], summary["out"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
