#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""双目接触：帧标签不滤波，事件按停留时长去抖。

不改 scripts/egodata/interaction.py 的默认（滤波开、dwell_s=0.10，帧标签一起滤）。
本脚本只扫参。说明在 docs/contact_grasp.md，已有结果在
docs/contact_eval/dwell_results.jsonl。缓存、网格和双目 episode 不在仓库里。

    python scripts/dwell.py \\
        --cache /path/out_on/cache \\
        --models /path/models \\
        --stereo-episodes /path/episodes \\
        --out docs/contact_eval/dwell_results.jsonl
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "headcam"))

import eval_hot3d_contact as E  # noqa: E402

CLIPS = ("clip-000000", "clip-000300", "clip-000700", "clip-001100")
DWELLS = (0.0, 0.03, 0.05, 0.10)


def load_meshes(models_dir):
    import trimesh

    meshes = {}
    for path in sorted(glob.glob(os.path.join(models_dir, "obj_*.glb"))):
        mesh = trimesh.load(path, force="mesh", process=False)
        key = str(int(Path(path).stem.split("_")[1]))
        meshes[key] = {
            "vertices": np.asarray(mesh.vertices),
            "faces": np.asarray(mesh.faces),
        }
    return meshes


def done_keys(path):
    found = set()
    if not os.path.exists(path):
        return found
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            found.add((row["clip"], float(row["dwell"])))
    return found


def sweep(cache_dir, models_dir, stereo_dir, out_path):
    meshes = load_meshes(models_dir)
    finished = done_keys(out_path)
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    for clip in CLIPS:
        with open(os.path.join(cache_dir, clip + ".json")) as handle:
            data = json.load(handle)
        gt = E.gt_labels(data, 0.005, 2)
        hands = E.stereo_joints(
            os.path.join(stereo_dir, clip + ".json"),
            len(data["timestamps"]),
        )
        surfaces = E.mesh_surfaces(data, meshes)
        raw = None
        for dwell in DWELLS:
            if (clip, float(dwell)) in finished:
                continue
            if raw is None:
                raw = E.estimate_interaction(
                    hands,
                    data["objects"],
                    data["timestamps"],
                    surfaces,
                    temporal_filter=False,
                )
            filtered = E.estimate_interaction(
                hands,
                data["objects"],
                data["timestamps"],
                surfaces,
                params={"dwell_s": dwell},
                temporal_filter=True,
            )
            event_only = dict(raw)
            event_only["events"] = filtered["events"]
            record = {
                "clip": clip,
                "dwell": dwell,
                "frame_filter": E.summarize(
                    E.evaluate_interaction(gt, filtered, data["timestamps"])
                ),
                "event_only": E.summarize(
                    E.evaluate_interaction(gt, event_only, data["timestamps"])
                ),
                "gt_events": len(gt["events"]),
            }
            with open(out_path, "a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(clip, dwell, flush=True)


def main():
    parser = argparse.ArgumentParser(description="帧标签不滤波，按 dwell_s 只对事件去抖")
    parser.add_argument("--cache", required=True, help="eval_hot3d_contact 写出的 cache 目录")
    parser.add_argument("--models", required=True, help="object_models_eval，里面是 obj_*.glb")
    parser.add_argument("--stereo-episodes", required=True, help="双目 episode 的目录")
    parser.add_argument("--out", required=True, help="jsonl 输出；已有的 (clip, dwell) 会跳过")
    args = parser.parse_args()
    sweep(args.cache, args.models, args.stereo_episodes, args.out)


if __name__ == "__main__":
    main()
