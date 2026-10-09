#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""在导出的 LeRobot 数据集上拟合线性行为克隆，并写下验证损失。

每一行 ``action`` 是下一步手腕增量。本脚本把同一片段里连续 ``horizon`` 行
拼成多步目标（默认读 ``meta/egodata_export.json`` 里的 horizon，否则 16）。
保持不动的预测是每一步 dxyz=0、相对四元数 0,0,0,1，日志里记成
``copy_current_wrist:``，不会被 ``scaling_law.py`` 当成 val_loss。

验证集是按 episode 留出的固定子集（默认 10%，种子固定），所有数据量共用。
训练集先在每条 episode 内按同一种子打乱动作块，再在固定的 episode 顺序上
轮转抽取，直到 ``--max-frames`` 个样本。打乱之后小预算不再只看见片段开头，
更小预算仍是这条轮转序列的前缀。预算会取满；只要训练 episode 不少于预算、
且每条至少有一个动作块，这一档就会用到这么多条 episode。
只有一条片段时（合成冒烟）改为留出该条末尾固定比例，训练块同样先打乱再截断。

动作和特征的逐维均值、标准差都只从全部训练池算一次，所有 ``--max-frames``
共用，并同样作用在验证目标和「保持不动」基线上。这样各档的 ``val_loss``
单位相同，基线也不随数据量变。日志另写 ``val_baseline_ratio``（验证损失 / 基线）。
平移和旋转分开报告。损失用科学计数法。

岭回归系数默认在该档训练样本内部再划出一份选择集上扫描，选完后用全部
训练样本重拟合。``--l2`` 可以钉死系数。``--seeds`` 用多个种子各跑一档
（每个种子自己的验证 episode），日志写均值和样本标准差。
读 parquet 用 pyarrow 的扁平数组。

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
    """返回训练下标、验证下标、训练 episode、验证 episode、全训练池下标。

    全训练池不随 ``max_frames`` 变，用来算各档共用的动作标准化统计量。
    """
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
        pool = np.asarray(starts[:-n_val], dtype=np.int64)
        rng = np.random.default_rng(seed)
        train_pool = _shuffled_copy(pool, rng)
        if max_frames is not None:
            train_pool = train_pool[: int(max_frames)]
        if len(train_pool) < 1:
            raise ValueError("训练样本为空")
        return train_pool, val_index, [episode_id], [episode_id], pool

    rng = np.random.default_rng(seed)
    order = np.array([episode_id for episode_id, _starts in usable], dtype=np.int64)
    rng.shuffle(order)
    n_val = max(1, int(round(float(val_fraction) * len(order))))
    if n_val >= len(order):
        n_val = len(order) - 1
    val_ids = [int(item) for item in order[:n_val]]
    train_order = [int(item) for item in order[n_val:]]
    by_id = {episode_id: starts for episode_id, starts in usable}
    # 统计量用时间顺序的全训练池，和打乱后的抽样顺序无关，各档完全相同。
    pool_index = np.concatenate([by_id[episode_id] for episode_id in train_order])
    lists = [_shuffled_copy(by_id[episode_id], rng) for episode_id in train_order]
    budget = int(sum(len(item) for item in lists))
    if max_frames is not None:
        budget = int(max_frames)
    train_index, chosen = _round_robin(train_order, lists, budget)
    if len(train_index) < 1:
        raise ValueError("训练样本为空")
    val_index = np.concatenate([by_id[episode_id] for episode_id in val_ids])
    return train_index, val_index, chosen, val_ids, pool_index


def _shuffled_copy(starts, rng):
    """打乱副本。原数组留给全训练池的统计量，顺序保持时间先后。"""
    copied = np.asarray(starts, dtype=np.int64).copy()
    rng.shuffle(copied)
    return copied


def _round_robin(episode_ids, start_lists, budget):
    """按 episode 轮转取动作块，直到 ``budget`` 个或全部取完。

    ``start_lists`` 应已按种子在每条 episode 内打乱，这样小预算不会只抽到
    片段开头。返回的样本是这条固定序列的前缀，所以更小的预算是更大预算的子集。
    预算为 N 且每条都有块时，会用到 min(N, episode 数) 条。
    """
    cursors = [0] * len(start_lists)
    picked = []
    used = []
    seen = set()
    remaining = int(budget)
    while remaining > 0:
        progressed = False
        for index, starts in enumerate(start_lists):
            if cursors[index] >= len(starts):
                continue
            picked.append(int(starts[cursors[index]]))
            cursors[index] += 1
            episode_id = int(episode_ids[index])
            if episode_id not in seen:
                seen.add(episode_id)
                used.append(episode_id)
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    if not picked:
        return np.empty(0, dtype=np.int64), []
    return np.asarray(picked, dtype=np.int64), used


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


def _pool_target_stats(action, indices, horizon):
    """全训练池的逐维均值和标准差。各档共用这一次的结果。"""
    pooled = _stack_targets(action, indices, horizon)
    mean = pooled.mean(axis=0)
    std = pooled.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    del pooled
    return mean, std


def _apply_target_stats(values, mean, std):
    return (np.asarray(values, dtype=np.float64) - mean) / std


def wrist_component_masks(step, horizon):
    """平移（左右手 xyz）和旋转（左右手四元数）在展平动作里的位置。"""
    width = int(horizon) * int(step)
    translation = np.zeros(width, dtype=bool)
    rotation = np.zeros(width, dtype=bool)
    if step < WRIST_STEP_DIM:
        return translation, rotation
    for offset in range(int(horizon)):
        base = offset * int(step)
        translation[base:base + 3] = True
        translation[base + 7:base + 10] = True
        rotation[base + 3:base + 7] = True
        rotation[base + 10:base + 14] = True
    return translation, rotation


def _masked_mse(prediction, target, mask):
    if not np.any(mask):
        return 0.0
    return float(np.mean((prediction[:, mask] - target[:, mask]) ** 2))


# 训练内部扫描用的岭回归系数。1.0 留在网格里，和以前的固定值可比。
DEFAULT_L2_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3)


def _column_stats(values):
    mean = np.asarray(values.mean(axis=0), dtype=np.float64)
    std = np.asarray(values.std(axis=0), dtype=np.float64)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def _ridge(train_x, train_y, val_x, l2, mean, std):
    """用调用方给的特征均值和标准差。不要在这里按当前子集重算。"""

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


def _select_l2(train_x, train_y, seed, mean, std, l2, lambdas):
    """``l2`` 给定时直接用。否则在训练样本内部留出 20% 选系数，不看真正的验证集。"""
    if l2 is not None:
        return float(l2)
    count = len(train_x)
    if count < 4:
        return 1.0
    rng = np.random.default_rng(int(seed) + 10007)
    order = rng.permutation(count)
    n_hold = max(1, int(round(0.2 * count)))
    if n_hold >= count:
        return 1.0
    hold = order[:n_hold]
    fit = order[n_hold:]
    best_l2 = float(lambdas[0])
    best_loss = float("inf")
    for candidate in lambdas:
        _, pred = _ridge(train_x[fit], train_y[fit], train_x[hold], float(candidate), mean, std)
        loss = float(np.mean((pred - train_y[hold]) ** 2))
        if loss < best_loss:
            best_loss = loss
            best_l2 = float(candidate)
    return best_l2


def train_bc(
    dataset_dir,
    steps=1,
    log_path=None,
    max_frames=None,
    lr=0.01,
    seed=0,
    horizon=None,
    val_fraction=0.1,
    l2=None,
    l2_grid=None,
):
    """岭回归拟合多步手腕增量。``steps`` 和 ``lr`` 保留是为了兼容旧命令，不参与拟合。"""
    del steps, lr
    state, action, episodes = _load_arrays(dataset_dir)
    horizon = _read_horizon(dataset_dir, horizon)
    if horizon < 1:
        raise ValueError("horizon 至少为 1")
    if action.shape[1] < WRIST_STEP_DIM:
        raise ValueError("action 宽度至少要有双手手腕增量")
    train_index, val_index, train_ids, val_ids, pool_index = _select_indices(
        episodes, horizon, max_frames, seed, val_fraction,
    )
    pool_rows = np.asarray(pool_index, dtype=np.int64)
    feat_mean, feat_std = _column_stats(state[pool_rows])
    train_x = state[train_index]
    val_x = state[val_index]
    mean, std = _pool_target_stats(action, pool_index, horizon)
    train_y = _apply_target_stats(_stack_targets(action, train_index, horizon), mean, std)
    val_y = _apply_target_stats(_stack_targets(action, val_index, horizon), mean, std)
    baseline = _copy_current_vector(action.shape[1], horizon)
    baseline_n = _apply_target_stats(baseline, mean, std)
    chosen_l2 = _select_l2(
        train_x, train_y, seed, feat_mean, feat_std, l2, l2_grid or DEFAULT_L2_GRID,
    )
    train_pred, val_pred = _ridge(train_x, train_y, val_x, chosen_l2, feat_mean, feat_std)
    train_loss = float(np.mean((train_pred - train_y) ** 2))
    val_loss = float(np.mean((val_pred - val_y) ** 2))
    copy_pred = np.broadcast_to(baseline_n, val_y.shape)
    copy_loss = float(np.mean((copy_pred - val_y) ** 2))
    translation, rotation = wrist_component_masks(action.shape[1], horizon)
    trans_mse = _masked_mse(val_pred, val_y, translation)
    rot_mse = _masked_mse(val_pred, val_y, rotation)
    copy_trans = _masked_mse(copy_pred, val_y, translation)
    copy_rot = _masked_mse(copy_pred, val_y, rotation)
    ratio = float(val_loss / copy_loss) if copy_loss > 0.0 else float("nan")
    lines = [
        "policy=linear_bc learner=ridge device=cpu frames=%d episodes=%d val_episodes=%d val_frames=%d horizon=%d ridge_l2=%.6e"
        % (len(train_index), len(set(train_ids)), len(set(val_ids)), len(val_index), horizon, chosen_l2),
        "ridge_l2: %.6e" % chosen_l2,
        "copy_current_wrist: %.6e" % copy_loss,
        "copy_trans_mse: %.6e" % copy_trans,
        "copy_rot_mse: %.6e" % copy_rot,
        "step=1 train_loss: %.6e val_loss: %.6e" % (train_loss, val_loss),
        "val_baseline_ratio: %.6e" % ratio,
        "trans_mse: %.6e" % trans_mse,
        "rot_mse: %.6e" % rot_mse,
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
        "trans_mse": trans_mse,
        "rot_mse": rot_mse,
        "copy_trans_mse": copy_trans,
        "copy_rot_mse": copy_rot,
        "val_baseline_ratio": ratio,
        "ridge_l2": chosen_l2,
        "feature_mean_sum": float(feat_mean.sum()),
        "feature_std_sum": float(feat_std.sum()),
        "train_indices": [int(item) for item in train_index],
        "frames": int(len(train_index)),
        "episodes": int(len(set(train_ids))),
        "val_episodes": int(len(set(val_ids))),
        "val_frames": int(len(val_index)),
        "train_episode_ids": sorted(int(item) for item in train_ids),
        "val_episode_ids": sorted(int(item) for item in val_ids),
        "horizon": int(horizon),
        "log": text,
    }


def _sample_std(values):
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < 2:
        return 0.0
    return float(arr.std(ddof=1))


def _parse_seeds(seed, seeds_text):
    text = (seeds_text or "").strip()
    if not text:
        return [int(seed)]
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError("--seeds 是空的")
    return values


def train_bc_seeds(
    dataset_dir,
    seeds,
    log_path=None,
    max_frames=None,
    seed=0,
    horizon=None,
    val_fraction=0.1,
    l2=None,
):
    """每个种子各拟合一次。多个种子时日志用均值，供缩放律画误差线。

    种子会重抽验证 episode。同一个种子下，不同 ``max_frames`` 仍共用验证集。
    """
    del seed
    results = [
        train_bc(
            dataset_dir,
            log_path=None,
            max_frames=max_frames,
            seed=int(one),
            horizon=horizon,
            val_fraction=val_fraction,
            l2=l2,
        )
        for one in seeds
    ]
    if len(results) == 1:
        text = results[0]["log"]
        if log_path is not None:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        results[0]["log"] = text
        results[0]["seeds"] = [int(seeds[0])]
        return results[0]
    blocks = []
    for one, result in zip(seeds, results):
        blocks.append("seed=%d\n%s" % (int(one), result["log"].rstrip("\n")))
    summary = []
    aggregate = {}
    for key in ("val_loss", "copy_current_wrist", "trans_mse", "rot_mse", "val_baseline_ratio"):
        values = [float(result[key]) for result in results]
        mean = float(np.mean(values))
        std = _sample_std(values)
        aggregate[key] = mean
        aggregate[key + "_std"] = std
        summary.append("%s_mean: %.6e" % (key, mean))
        summary.append("%s_std: %.6e" % (key, std))
    text = "\n".join(blocks + summary) + "\n"
    if log_path is not None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    merged = dict(results[0])
    merged.update(aggregate)
    merged["log"] = text
    merged["seeds"] = [int(one) for one in seeds]
    merged["n_seeds"] = len(seeds)
    return merged


def main(argv=None):
    parser = argparse.ArgumentParser(description="EgoDex LeRobot 样本上的线性 BC（岭回归）")
    parser.add_argument("--dataset", required=True, help="ego_to_lerobot 的输出目录")
    parser.add_argument("--steps", type=int, default=1, help="兼容旧命令。拟合是闭式岭回归，不按这个步数迭代")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="训练样本上限。每条 episode 内先打乱动作块，再轮转取满这个数量",
    )
    parser.add_argument("--lr", type=float, default=0.01, help="兼容旧命令，岭回归不使用")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", default="", help="逗号分隔的多个种子。日志写 val_loss_mean 和 val_loss_std")
    parser.add_argument("--l2", type=float, default=None, help="固定岭回归系数。默认在训练内部划分上扫描")
    parser.add_argument("--horizon", type=int, default=None, help="多步增量长度。默认读导出说明")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="按 episode 留出的验证比例")
    parser.add_argument("--log", required=True, help="写入 val_loss 的日志路径")
    args = parser.parse_args(argv)
    result = train_bc_seeds(
        args.dataset,
        seeds=_parse_seeds(args.seed, args.seeds),
        log_path=args.log,
        max_frames=args.max_frames,
        horizon=args.horizon,
        val_fraction=args.val_fraction,
        l2=args.l2,
    )
    print(
        "frames %d episodes %d val_loss %.6e copy_current_wrist %.6e ratio %.6e log %s"
        % (
            result["frames"],
            result["episodes"],
            result["val_loss"],
            result["copy_current_wrist"],
            result["val_baseline_ratio"],
            args.log,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
