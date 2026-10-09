#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""在导出的 LeRobot 数据集上做几步线性行为克隆，并写下验证损失。

策略是 ``action ≈ state @ W``：状态是双手关节加当前手腕，动作是下一帧手腕。
不依赖 torch / lerobot。日志行是 ``val_loss:``，``scripts/scaling_law.py`` 能直接读。

用 ``--max-frames`` 截断帧数，就可以对同一数据集跑不同数据量。

示例:
    python scripts/ego_pretrain_bc.py --dataset outputs/egodex_lerobot --steps 20 \\
        --log outputs/ego_pretrain.log
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _load_arrays(dataset_dir, max_frames):
    path = Path(dataset_dir) / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(path)
    state = np.stack([np.asarray(row, dtype=np.float64) for row in table.column("observation.state").to_pylist()])
    action = np.stack([np.asarray(row, dtype=np.float64) for row in table.column("action").to_pylist()])
    episodes = np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64)
    if max_frames is not None:
        state = state[: int(max_frames)]
        action = action[: int(max_frames)]
        episodes = episodes[: int(max_frames)]
    if len(state) < 2:
        raise ValueError("至少需要 2 帧才能分成训练和验证")
    return state, action, episodes


def train_bc(dataset_dir, steps=5, log_path=None, max_frames=None, lr=0.01, seed=0):
    """CPU 上做 ``steps`` 步全批量梯度下降。返回最后一轮的损失和用到的帧数。"""
    state, action, episodes = _load_arrays(dataset_dir, max_frames)
    split = max(1, len(state) // 2)
    if split >= len(state):
        split = len(state) - 1
    train_x, val_x = state[:split], state[split:]
    train_y, val_y = action[:split], action[split:]
    rng = np.random.default_rng(seed)
    weight = rng.normal(0.0, 0.01, size=(train_x.shape[1], train_y.shape[1]))
    lines = [
        "policy=linear_bc device=cpu frames=%d episodes=%d" % (len(state), len(set(episodes.tolist()))),
    ]
    last_train = last_val = None
    for step in range(1, int(steps) + 1):
        prediction = train_x @ weight
        residual = prediction - train_y
        train_loss = float(np.mean(residual ** 2))
        grad = train_x.T @ residual / len(train_x)
        weight = weight - float(lr) * grad
        val_loss = float(np.mean((val_x @ weight - val_y) ** 2))
        last_train, last_val = train_loss, val_loss
        lines.append("step=%d train_loss: %.6f val_loss: %.6f" % (step, train_loss, val_loss))
    text = "\n".join(lines) + "\n"
    if log_path is not None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return {
        "train_loss": last_train,
        "val_loss": last_val,
        "frames": int(len(state)),
        "episodes": int(len(set(episodes.tolist()))),
        "log": text,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="EgoDex LeRobot 样本上的线性 BC")
    parser.add_argument("--dataset", required=True, help="ego_to_lerobot 的输出目录")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=None, help="只用前 N 帧，用来做不同数据量")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log", required=True, help="写入 val_loss 的日志路径")
    args = parser.parse_args(argv)
    result = train_bc(
        args.dataset,
        steps=args.steps,
        log_path=args.log,
        max_frames=args.max_frames,
        lr=args.lr,
        seed=args.seed,
    )
    print("frames %d val_loss %.6f log %s" % (result["frames"], result["val_loss"], args.log))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
