#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 EgoDex 目录（task/*.hdf5）转成统一 episode JSON。

示例:
    python scripts/convert_egodex.py --input /path/to/egodex/test --out outputs/egodex_unified --limit 2
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from egodata.egodex import convert_tree  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description="EgoDex HDF5 → 统一 episode JSON")
    parser.add_argument("--input", required=True, help="含 *.hdf5 的目录，例如解压后的 test/")
    parser.add_argument("--out", required=True, help="统一 JSON 输出目录")
    parser.add_argument("--limit", type=int, default=None, help="只转换前 N 条，便于抽样")
    args = parser.parse_args(argv)
    written = convert_tree(args.input, args.out, limit=args.limit)
    print("wrote %d episodes to %s" % (len(written), args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
