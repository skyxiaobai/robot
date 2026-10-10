#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EgoDex 图像 ACT 对比：同一份按 episode 固定的 test 划分上评估
保持不动基线、线性 BC（有/无掩码）和 ACT（有/无掩码）。

子命令:
  split     按 episode 写 train/val/test 划分（json）
  linear    岭回归线性 BC，写预测 npz
  baseline  保持不动（每步 dxyz=0、四元数单位）预测 npz
  act       用训练好的 lerobot ACT 在 test 上预测 npz（需要 lerobot）
  metrics   读若干预测 npz，按同一套掩码算指标，写 json/markdown

指标只在 action_valid 为 1 的手上算：
  step_pos_cm@h    第 h 步单步手腕位移误差（cm）
  cum_pos_cm@h     前 h 步累加位移的终点误差（cm），要求这只手 1..h 步都有效
  cum_rot_deg@h    前 h 步相对旋转复合后的角度误差（度），同样要求连续有效
  mse_cm2@h        第 h 步单步位移 MSE（cm^2）
  ade_cm           开环轨迹误差：cum_pos_cm 在 1..H 上的平均（只取整段有效的手）
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

H = 16
STEP = 14


def load(dataset):
    files = sorted((Path(dataset) / "data").glob("chunk-*/file-*.parquet"))
    table = pa.concat_tables([pq.read_table(f) for f in files])
    col = lambda k: np.stack(table.column(k).to_pylist()).astype(np.float64)
    ep = np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64)
    idx = np.asarray(table.column("index").to_pylist(), dtype=np.int64)
    return {"state": col("observation.state"), "action": col("action")[:, :STEP],
            "valid": col("action_valid"), "episode": ep, "index": idx}


def chunk_starts(episode, horizon=H, stride=1):
    starts = []
    for e in np.unique(episode):
        rows = np.nonzero(episode == e)[0]
        if len(rows) >= horizon:
            starts.extend(rows[: len(rows) - horizon + 1 : stride].tolist())
    return np.asarray(starts, dtype=np.int64)


def gather(d, starts, horizon=H):
    off = starts[:, None] + np.arange(horizon)[None]
    return d["action"][off], d["valid"][off]


def cmd_split(a):
    d = load(a.dataset)
    eps = np.unique(d["episode"])
    rng = np.random.default_rng(a.seed)
    eps = rng.permutation(eps)
    n = len(eps)
    nt, nv = int(round(n * a.test)), int(round(n * a.val))
    info = json.loads((Path(a.dataset) / "meta" / "egodata_export.json").read_text())
    split = {"seed": a.seed, "test": sorted(int(x) for x in eps[:nt]),
             "val": sorted(int(x) for x in eps[nt:nt + nv]),
             "train": sorted(int(x) for x in eps[nt + nv:])}
    names = info.get("episode_ids") or info.get("episodes")
    if isinstance(names, list) and len(names) == n:
        split["episode_names"] = {k: [names[i] for i in v] for k, v in split.items() if k in ("train", "val", "test")}
    Path(a.out).write_text(json.dumps(split, ensure_ascii=False, indent=1))
    print({k: len(v) for k, v in split.items() if isinstance(v, list)})


def _test_starts(d, split, stride):
    keep = np.isin(d["episode"], split["test"])
    starts = chunk_starts(d["episode"], H, stride)
    return starts[keep[starts]]


def cmd_baseline(a):
    d = load(a.dataset); split = json.loads(Path(a.split).read_text())
    starts = _test_starts(d, split, a.stride)
    pred = np.zeros((len(starts), H, STEP)); pred[:, :, 6] = 1; pred[:, :, 13] = 1
    np.savez(a.out, starts=starts, pred=pred)


def cmd_linear(a):
    d = load(a.dataset); split = json.loads(Path(a.split).read_text())
    starts = chunk_starts(d["episode"], H, 1)
    tr = starts[np.isin(d["episode"][starts], split["train"])]
    te = _test_starts(d, split, a.stride)
    feat = lambda s: d["state"][s]
    X = feat(tr); mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    Xt = np.hstack([(feat(te) - mu) / sd, np.ones((len(te), 1))])
    Y, V = gather(d, tr)
    pred = np.zeros((len(te), H, STEP))
    reg = a.l2 * np.eye(Xn.shape[1]); reg[-1, -1] = 0
    for h in range(H):
        for hand in range(2):
            rows = V[:, h, hand] >= 0.5 if a.mask else np.ones(len(Xn), bool)
            A = Xn[rows]; y = Y[rows, h, hand * 7:(hand + 1) * 7]
            W = np.linalg.solve(A.T @ A + reg * len(A), A.T @ y)
            pred[:, h, hand * 7:(hand + 1) * 7] = Xt @ W
    np.savez(a.out, starts=te, pred=pred)
    print("linear train_chunks %d test_chunks %d mask %s" % (len(tr), len(te), a.mask))


def cmd_act(a):
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from ego_act_scaling import _find_policy_dir
    d = load(a.dataset); split = json.loads(Path(a.split).read_text())
    starts = _test_starts(d, split, a.stride)
    ds = LeRobotDataset("local/egodex", root=str(Path(a.dataset).resolve()))
    pdir = _find_policy_dir(a.policy)
    policy = ACTPolicy.from_pretrained(str(pdir)).to(a.device).eval()
    pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=str(pdir))
    pos = {int(g): i for i, g in enumerate(d["index"])}
    preds = []
    for b in range(0, len(starts), a.batch):
        items = [ds[int(d["index"][s])] for s in starts[b:b + a.batch]]
        batch = {}
        for k in ("observation.image", "observation.state"):
            v = torch.stack([torch.as_tensor(it[k]) for it in items]).to(a.device)
            if k == "observation.image" and v.dtype == torch.uint8:
                v = v.float() / 255.0
            batch[k] = v
        batch = pre(batch)
        with torch.no_grad():
            out = post(policy.predict_action_chunk(batch))
        preds.append(out.float().cpu().numpy()[:, :H, :STEP])
    pred = np.concatenate(preds)
    np.savez(a.out, starts=starts, pred=pred)
    print("act test_chunks %d" % len(starts))


def _qnorm(q):
    return q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-9, None)


def _qmul(a, b):  # xyzw
    ax, ay, az, aw = np.moveaxis(a, -1, 0); bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([aw*bx+ax*bw+ay*bz-az*by, aw*by-ax*bz+ay*bw+az*bx,
                     aw*bz+ax*by-ay*bx+az*bw, aw*bw-ax*bx-ay*by-az*bz], -1)


def _cumq(q):
    out = np.empty_like(q); cur = _qnorm(q[:, 0])
    out[:, 0] = cur
    for h in range(1, q.shape[1]):
        cur = _qnorm(_qmul(_qnorm(q[:, h]), cur)); out[:, h] = cur
    return out


def metrics_for(d, starts, pred, horizons):
    tgt, val = gather(d, starts)
    res = {"chunks": int(len(starts))}
    step_pos, cum_pos, cum_rot, ok_all = [], [], [], []
    for hand in range(2):
        s = slice(hand * 7, hand * 7 + 3); r = slice(hand * 7 + 3, hand * 7 + 7)
        v = val[:, :, hand] >= 0.5
        cont = np.cumprod(v, axis=1).astype(bool)
        step_pos.append((np.linalg.norm(pred[:, :, s] - tgt[:, :, s], axis=-1) * 100, v))
        cp = np.linalg.norm(np.cumsum(pred[:, :, s], 1) - np.cumsum(tgt[:, :, s], 1), axis=-1) * 100
        qa, qb = _cumq(pred[:, :, r]), _cumq(tgt[:, :, r])
        dot = np.clip(np.abs(np.sum(qa * qb, -1)), 0, 1)
        cum_pos.append((cp, cont)); cum_rot.append((np.degrees(2 * np.arccos(dot)), cont))
    def avg(pairs, h, sq=False):
        num = sum(float(np.sum((e[:, h] ** 2 if sq else e[:, h]) * m[:, h])) for e, m in pairs)
        den = sum(float(np.sum(m[:, h])) for e, m in pairs)
        return num / den if den else None
    for h in horizons:
        res["step_pos_cm@%d" % h] = avg(step_pos, h - 1)
        res["mse_cm2@%d" % h] = avg(step_pos, h - 1, sq=True)
        res["cum_pos_cm@%d" % h] = avg(cum_pos, h - 1)
        res["cum_rot_deg@%d" % h] = avg(cum_rot, h - 1)
        res["n_hands@%d" % h] = int(sum(np.sum(m[:, h - 1]) for _, m in cum_pos))
    full = [(e.mean(1), m[:, -1]) for e, m in cum_pos]
    num = sum(float(np.sum(e * m)) for e, m in full); den = sum(float(np.sum(m)) for _, m in full)
    res["ade_cm"] = num / den if den else None
    return res


def cmd_metrics(a):
    d = load(a.dataset)
    hs = [int(x) for x in a.horizons.split(",")]
    out = {}
    for item in a.pred:
        name, path = item.split("=", 1)
        z = np.load(path)
        out[name] = metrics_for(d, z["starts"], z["pred"], hs)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split"); s.add_argument("--dataset", required=True); s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=0); s.add_argument("--test", type=float, default=0.2); s.add_argument("--val", type=float, default=0.1)
    for name in ("baseline", "linear", "act"):
        s = sub.add_parser(name); s.add_argument("--dataset", required=True); s.add_argument("--split", required=True)
        s.add_argument("--out", required=True); s.add_argument("--stride", type=int, default=1)
        if name == "linear":
            s.add_argument("--l2", type=float, default=1e-3); s.add_argument("--mask", type=int, default=1)
        if name == "act":
            s.add_argument("--policy", required=True); s.add_argument("--device", default="cuda"); s.add_argument("--batch", type=int, default=64)
    s = sub.add_parser("metrics"); s.add_argument("--dataset", required=True); s.add_argument("--pred", nargs="+", required=True)
    s.add_argument("--out", required=True); s.add_argument("--horizons", default="1,8,16")
    a = p.parse_args()
    globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    main()
