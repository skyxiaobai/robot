#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""评测接触和抓取：精确率、召回率，以及事件时间差。

合成夹具不需要 HOT3D，用来确认指标代码：

    python scripts/eval_contact_grasp.py --synthetic

真片段要有统一 episode，以及一份同结构的真值 JSON（``contact`` / ``grasp`` / ``events``）。
本仓库环境里没有 HOT3D clip、物体网格和 MANO，所以真实片段的数字是待补，脚本不会编一个。

    python scripts/eval_contact_grasp.py --episode pred.json --gt gt.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from egodata.interaction import evaluate_interaction, synthetic_disagreement  # noqa: E402
from egodata.schema import load_episode  # noqa: E402


def _load_labels(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "contact" in payload and "grasp" in payload:
        return payload
    raise ValueError("%s 里没有 contact/grasp" % path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="接触和抓取的精确率、召回率、事件时间差")
    parser.add_argument("--synthetic", action="store_true", help="跑 3 帧合成小球，不读 HOT3D")
    parser.add_argument("--episode", help="启发式或模型写出的统一 episode JSON")
    parser.add_argument("--gt", help="真值 JSON，含 contact、grasp、events")
    parser.add_argument("--out", help="把指标写成 JSON。不传则打印到标准输出")
    args = parser.parse_args(argv)
    if args.synthetic:
        gt, pred, timestamps = synthetic_disagreement()
        metrics = evaluate_interaction(gt, pred, timestamps)
        metrics["source"] = "synthetic_sphere_3frames"
        metrics["hot3d"] = "待补"
    elif args.episode and args.gt:
        episode = load_episode(args.episode)
        gt = _load_labels(args.gt)
        metrics = evaluate_interaction(gt, episode, episode["timestamps"])
        metrics["source"] = str(args.episode)
    else:
        print("需要 --synthetic，或同时给 --episode 和 --gt。HOT3D 真实片段的指标是待补。", file=sys.stderr)
        return 2
    text = json.dumps(metrics, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
