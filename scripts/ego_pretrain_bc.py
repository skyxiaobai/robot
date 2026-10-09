#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""在导出的 LeRobot 数据集上拟合线性行为克隆，并写下验证损失。

每一行 ``action`` 是下一步手腕增量。本脚本把同一片段里连续 ``horizon`` 行
拼成多步目标（默认读 ``meta/egodata_export.json`` 里的 horizon，否则 16）。
保持不动的预测是每一步 dxyz=0、相对四元数 0,0,0,1，日志里记成
``copy_current_wrist:``，不会被 ``scaling_law.py`` 当成 val_loss。

验证集是按 episode 留出的固定子集（默认 10%，种子固定），所有数据量共用。
训练集按同一套随机顺序整段累加，直到 ``--max-frames`` 个训练样本；更小的
预算是更大预算的前缀，不会按字母序切掉后面的任务，也不会把同一条切进
训练和验证。只有一条片段时（合成冒烟）改为留出该条末尾固定比例的样本。

拟合是标准化特征上的岭回归，闭式解，相当于线性模型已经收敛。
读 parquet 用 pyarrow 的扁平数组，不把整张表转成 Python 列表。

示例:
    python scripts/ego_pretrain_bc.py --dataset outputs/egodex_lerobot \\
        --max-frames 1000 --log outputs/ego_pretrain_1k.log
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

from egodata.lerobot_export import DEFAULT_HORIZON, WRIST_STEP_DIM  # noqa: E402


def _flatten_fixed(column):
    values = column.combine_chunks().flatten()
    width = column.type.list_size
    flat = values.to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
    return np.asarray(flat).reshape(-1, width)


def _load_arrays(dataset_dir):
    path = Path(dataset_dir) / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(path, columns=["observation.state", "action", "episode_index"])
    state = _flatten_fixed(table.column("observation.state"))
    action = _flatten_fixed(table.column("action"))
    episodes = np.asarray(table.column("episode_index").to_numpy(), dtype=np.int64)
    return state, action, episodes


def _read_horizon(dataset_dir, horizon):
    if horizon is not None:
        return int(horizon)
    note = Path(dataset_dir) / "meta" / "egodata_export.json"
    if note.is_file():
        payload = json.loads(note.read_text(encoding="utf-8"))
        if payload.get("horizon"):
            return int(payload["horizon"])
    return DEFAULT_HORIZON


def _spans(episodes):
    if len(episodes) == 0:
        return []
    cuts = np.flatnonzero(np.diff(episodes)) + 1
    bounds = np.concatenate(([0], cuts, [len(episodes)]))
    spans = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        spans.append((int(episodes[start]), int(start), int(end - start)))
    return spans


def _chunk_starts(span_start, length, horizon):
    count = length - horizon + 1
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    return span_start + np.arange(count, dtype=np.int64)


def _select_indices(episodes, horizon, max_frames, seed, val_fraction):
    """返回训练样本下标、验证样本下标、训练 episode id、验证 episode id。"""
    spans = _spans(episodes)
    usable = []
    for episode_id, start, length in spans:
        starts = _chunk_starts(start, length, horizon)
        if len(starts):
            usable.append((episode_id, starts))
    if not usable:
        raise ValueError("没有足够长的片段来组成 %d 步动作" % horizon)
    if len(usable) == 1:
        episode_id, starts = usable[0]
        n_val = max(1, len(starts) // 5)
        if n_val >= len(starts):
            n_val = len(starts) - 1
        if n_val < 1:
            raise ValueError("单条片段太短，无法同时留下训练和验证样本")
        val_index = starts[-n_val:]
        train_pool = starts[:-n_val]
        if max_frames is not None:
            train_pool = train_pool[: int(max_frames)]
        if len(train_pool) < 1:
            raise ValueError("训练样本为空")
        return train_pool, val_index, [episode_id], [episode_id]

    rng = np.random.default_rng(seed)
    order = np.array([episode_id for episode_id, _starts in usable], dtype=np.int64)
    rng.shuffle(order)
    n_val = max(1, int(round(float(val_fraction) * len(order))))
    if n_val >= len(order):
        n_val = len(order) - 1
    val_ids = [int(item) for item in order[:n_val]]
    train_order = [int(item) for item in order[n_val:]]
    by_id = {episode_id: starts for episode_id, starts in usable}
    chosen = []
    total = 0
    for episode_id in train_order:
        count = len(by_id[episode_id])
        if max_frames is not None and chosen and total + count > int(max_frames):
            break
        chosen.append(episode_id)
        total += count
        if max_frames is not None and total >= int(max_frames):
            break
    if not chosen:
        raise ValueError("训练 episode 为空")
    train_index = np.concatenate([by_id[episode_id] for episode_id in chosen])
    val_index = np.concatenate([by_id[episode_id] for episode_id in val_ids])
    return train_index, val_index, chosen, val_ids


def _stack_targets(action, indices, horizon):
    step = action.shape[1]
    target = np.empty((len(indices), horizon * step), dtype=np.float64)
    for offset in range(horizon):
        target[:, offset * step:(offset + 1) * step] = action[indices + offset]
    return target


def _copy_current_vector(step, horizon):
    """保持当前手腕：每步 dxyz 为 0，两只手的相对四元数都是单位四元数。"""
    one = np.zeros(step, dtype=np.float64)
    one[6] = 1.0
    if step >= WRIST_STEP_DIM:
        one[WRIST_STEP_DIM - 1] = 1.0
    return np.tile(one, horizon)


def _ridge(train_x, train_y, val_x, l2=1.0):
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)

    def augment(values):
        scaled = (values - mean) / std
        return np.concatenate([scaled, np.ones((len(values), 1))], axis=1)

    design = augment(train_x)
    width = design.shape[1]
    gram = design.T @ design
    gram.flat[:: width + 1] += float(l2)
    weight = np.linalg.solve(gram, design.T @ train_y)
    train_pred = design @ weight
    val_pred = augment(val_x) @ weight
    return train_pred, val_pred


def train_bc(
    dataset_dir,
    steps=1,
    log_path=None,
    max_frames=None,
    lr=0.01,
    seed=0,
    horizon=None,
    val_fraction=0.1,
):
    """岭回归拟合多步手腕增量。``steps`` 和 ``lr`` 保留是为了兼容旧命令，不参与拟合。"""
    del steps, lr
    state, action, episodes = _load_arrays(dataset_dir)
    horizon = _read_horizon(dataset_dir, horizon)
    if horizon < 1:
        raise ValueError("horizon 至少为 1")
    if action.shape[1] < WRIST_STEP_DIM:
        raise ValueError("action 宽度至少要有双手手腕增量")
    train_index, val_index, train_ids, val_ids = _select_indices(
        episodes, horizon, max_frames, seed, val_fraction,
    )
    train_x = state[train_index]
    val_x = state[val_index]
    train_y = _stack_targets(action, train_index, horizon)
    val_y = _stack_targets(action, val_index, horizon)
    train_pred, val_pred = _ridge(train_x, train_y, val_x)
    train_loss = float(np.mean((train_pred - train_y) ** 2))
    val_loss = float(np.mean((val_pred - val_y) ** 2))
    baseline = _copy_current_vector(action.shape[1], horizon)
    copy_loss = float(np.mean((val_y - baseline) ** 2))
    lines = [
        "policy=linear_bc learner=ridge device=cpu frames=%d episodes=%d val_episodes=%d val_frames=%d horizon=%d"
        % (len(train_index), len(set(train_ids)), len(set(val_ids)), len(val_index), horizon),
        "copy_current_wrist: %.8f" % copy_loss,
        "step=1 train_loss: %.6f val_loss: %.6f" % (train_loss, val_loss),
    ]
    text = "\n".join(lines) + "\n"
    if log_path is not None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "copy_current_wrist": copy_loss,
        "frames": int(len(train_index)),
        "episodes": int(len(set(train_ids))),
        "val_episodes": int(len(set(val_ids))),
        "val_frames": int(len(val_index)),
        "train_episode_ids": sorted(int(item) for item in train_ids),
        "val_episode_ids": sorted(int(item) for item in val_ids),
        "horizon": int(horizon),
        "log": text,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="EgoDex LeRobot 样本上的线性 BC（岭回归）")
    parser.add_argument("--dataset", required=True, help="ego_to_lerobot 的输出目录")
    parser.add_argument("--steps", type=int, default=1, help="兼容旧命令。拟合是闭式岭回归，不按这个步数迭代")
    parser.add_argument("--max-frames", type=int, default=None, help="训练样本上限。按固定随机顺序整段累加")
    parser.add_argument("--lr", type=float, default=0.01, help="兼容旧命令，岭回归不使用")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=None, help="多步增量长度。默认读导出说明")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="按 episode 留出的验证比例")
    parser.add_argument("--log", required=True, help="写入 val_loss 的日志路径")
    args = parser.parse_args(argv)
    result = train_bc(
        args.dataset,
        steps=args.steps,
        log_path=args.log,
        max_frames=args.max_frames,
        lr=args.lr,
        seed=args.seed,
        horizon=args.horizon,
        val_fraction=args.val_fraction,
    )
    print(
        "frames %d val_loss %.6f copy_current_wrist %.6f log %s"
        % (result["frames"], result["val_loss"], result["copy_current_wrist"], args.log)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
