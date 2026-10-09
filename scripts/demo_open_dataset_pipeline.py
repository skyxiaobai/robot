#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""在合成的 EgoDex 式 HDF5 上跑通：转换 → QC/产出率 → 覆盖度 → 标注校验。

不下载 16GB 的 test.zip。真实数据把 --input 指到解压后的 test/ 即可，见 README。
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from egodata.coverage import coverage_report, write_coverage_reports  # noqa: E402
from egodata.egodex import convert_tree  # noqa: E402
from egodata.labels import validate_annotation  # noqa: E402
from egodata.qc import qc_episode, write_yield_reports, yield_report  # noqa: E402
from egodata.schema import iter_episode_paths, load_episode  # noqa: E402


def _pose(translation):
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float32)
    return matrix


def _write_clip(path, wrists, confidence=0.99):
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    num_frames = len(wrists)
    intrinsic = np.array([[736.6339, 0, 960], [0, 736.6339, 540], [0, 0, 1]], dtype=np.float32)
    camera = np.stack([_pose((0, 0, 0)) for _ in range(num_frames)])
    with h5py.File(path, "w") as handle:
        handle.create_dataset("camera/intrinsic", data=intrinsic)
        handle.create_dataset("transforms/camera", data=camera)
        for prefix in ("left", "right"):
            series = np.stack([_pose(point) for point in wrists])
            handle.create_dataset("transforms/%sHand" % prefix, data=series)
            handle.create_dataset(
                "confidences/%sHand" % prefix,
                data=np.full((num_frames,), confidence, dtype=np.float32),
            )
        handle.attrs["llm_type"] = "reversible"
        handle.attrs["which_llm_description"] = "1"
        handle.attrs["llm_description"] = "Pick up the cup and place it on the tray."
        handle.attrs["llm_description2"] = "Pick up the cup and place it back."
        handle.attrs["llm_objects"] = np.array(["cup"], dtype=object)
        handle.attrs["llm_verbs"] = np.array(["pick", "place"], dtype=object)
        handle.attrs["environment"] = "table:wood, position:sitting"
        handle.attrs["task"] = path.parent.name


def build_sample(root):
    """两条短片段：一条手在动，一条手停在画面外。"""
    moving = [(0.002 * i, 0.0, 1.0) for i in range(30)]
    outside = [(10.0, 0.0, 1.0) for _ in range(30)]
    _write_clip(Path(root) / "basic_pick_place" / "0.hdf5", moving)
    _write_clip(Path(root) / "pour" / "0.hdf5", outside, confidence=0.2)


def main(argv=None):
    parser = argparse.ArgumentParser(description="开放数据集流水线的小样本演示")
    parser.add_argument("--input", default=None, help="已解压的 EgoDex 目录；默认用合成样本")
    parser.add_argument("--out", default="outputs/open_data_demo", help="报告与统一 JSON 的目录")
    parser.add_argument("--limit", type=int, default=2)
    args = parser.parse_args(argv)
    out = Path(args.out)
    if args.input:
        source = args.input
    else:
        sample_dir = out / "synthetic_egodex"
        build_sample(sample_dir)
        source = sample_dir
    unified = out / "unified"
    written = convert_tree(source, unified, limit=args.limit)
    episodes = [load_episode(path) for path in iter_episode_paths(unified)]
    qc = yield_report([qc_episode(episode) for episode in episodes])
    write_yield_reports(qc, out / "yield_report.html", out / "yield_episodes.csv")
    coverage = coverage_report(episodes)
    write_coverage_reports(coverage, out / "coverage_report.html", out / "coverage_counts.csv")
    incomplete = 0
    for episode in episodes:
        result = validate_annotation(episode["annotation"], duration_s=episode["num_frames"] / episode["fps"])
        if result["missing_levels"]:
            incomplete += 1
    print("episodes %d" % len(written))
    print("yield %.3f" % qc["yield"])
    print("coverage gaps %d" % len(coverage["gaps"]))
    print("episodes missing subtask/instruction %d" % incomplete)
    print("reports %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
