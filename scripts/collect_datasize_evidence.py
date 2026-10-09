#!/usr/bin/env python3
"""Collect local evidence for the data-volume section of headcam_data_spec.md."""
import json
import os
import glob

from repo_root import dataset_fps, resolve_root

ROOT = resolve_root()

print("=== pusht dataset (LeRobot) ===")
try:
    with open(os.path.join(ROOT, "data/pusht/meta/info.json"), encoding="utf-8") as f:
        info = json.load(f)
    print("total_episodes:", info.get("total_episodes"))
    print("fps:", dataset_fps(info))
    feats = info.get("features", {})
    for k in ("observation.image", "observation.state", "action"):
        f = feats.get(k, {})
        print(f"  {k}: shape={f.get('shape')} dtype={f.get('dtype')}")
    # episodes.jsonl for per-episode frame counts
    with open(os.path.join(ROOT, "data/pusht/meta/episodes.jsonl"), encoding="utf-8") as f:
        ep = json.loads(f.read().strip().split("\n")[0])
    print("episode keys:", sorted(ep.keys()))
    n_frames = ep.get("length") or ep.get("num_frames") or "?"
    print("first episode length:", n_frames)
except Exception as e:
    print("ERR", e)

print()
print("=== Square dataset (MimicGen demo.hdf5) ===")
try:
    import h5py
    p = "/tmp/core_datasets/square/demo_src_square_task_D1/demo.hdf5"
    f = h5py.File(p, "r")
    data = f["data"]
    names = list(data.keys())
    print("total demos:", len(names))
    d0 = data[names[0]]
    acts = d0["actions"]
    print("first demo actions shape:", acts.shape)
    print("env timestep (s):", f.attrs.get("env_timestep", "?"))
    lengths = []
    for n in names:
        lengths.append(data[n]["actions"].shape[0])
    lengths.sort()
    print(f"episode lengths: min={lengths[0]} median={lengths[len(lengths)//2]} max={lengths[-1]}")
    print("action keys:", list(d0.keys()))
    obs = d0["obs"]
    print("obs keys:", sorted(obs.keys()))
    f.close()
except Exception as e:
    print("ERR", e)

print()
print("=== ACT train config (pusht) ===")
try:
    cfg = os.path.join(ROOT, "outputs/checkpoints/100000/pretrained_model/train_config.json")
    with open(cfg, encoding="utf-8") as f:
        c = json.load(f)
    print("steps:", c.get("steps"))
    print("batch_size:", c.get("batch_size"))
    print("log_freq:", c.get("log_freq"))
    print("n_action_steps:", c.get("policy", {}).get("n_action_steps"))
    print("chunk_size:", c.get("policy", {}).get("chunk_size"))
    print("dataset.repo_id:", c.get("dataset", {}).get("repo_id"))
except Exception as e:
    print("ERR", e)

print()
print("=== BC-RNN config (Square) ===")
try:
    cs = sorted(glob.glob("/tmp/core_train_configs/bc_rnn_*_ds_*_seed_101.json"))
    if cs:
        with open(cs[0], encoding="utf-8") as f:
            c = json.load(f)
        print("config file:", cs[0])
        print("seq_length:", c.get("seq_length"))
        print("rollout:", c.get("rollout"))
        algo = c.get("algo", {})
        print("train steps/epochs:", algo.get("num_epochs") or algo.get("num_steps"))
    else:
        print("no configs found")
except Exception as e:
    print("ERR", e)
