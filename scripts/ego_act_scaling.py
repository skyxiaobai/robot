#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""为 lerobot 0.6.1 的 ACT 安排多档数据量，并共用线性 BC 的验证 episode。

ACT 吃 ``observation.image`` 和 ``observation.state``，动作块长度 16。
lerobot 0.6.1 的 ACT 没有语言编码器。任务句子已经写在数据集的 task 字段里，
这里不另造一个语言开关。换到带语言编码器的策略时可以直接用这些句子。

数据量按整条 episode 计：验证 episode 由种子固定，训练 episode 是打乱后
顺序的前缀。更小的档是更大档的前缀。lerobot 的 ``eval_split`` 会按任务
丢掉每个子集末尾的 episode，各档验证集不一样，所以这里不用它。
训练只看训练 episode；验证损失和保持不动基线在训练结束后另算。

T4（16GB）/ L4（24GB）上，224 视频、batch 4、ResNet18，一步大约 0.4–1 秒。
建议四档（训练 episode 数 / 优化步数）：64/300、256/800、1024/1500、全部/2000。
四档合计大约 80 分钟训练。加上 16GB 的 test.zip、转换和 224 导出，
一整次大约 2.5–3.5 小时。只要曲线方向时把转换限制在 80 条，
三档 16/32/64 条、各 40/80/120 步，训练大约 15 分钟。

示例:
    python scripts/ego_act_scaling.py --dataset outputs/egodex_lerobot \\
        --sizes 64,256,1024,all --steps 300,800,1500,2000 --dry-run
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ego_pretrain_bc import (  # noqa: E402
    _chunk_starts,
    _copy_current_vector,
    _load_arrays,
    _read_horizon,
    _select_indices,
    _spans,
    _stack_targets,
)


def read_episode_index(dataset_dir):
    path = Path(dataset_dir) / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(path, columns=["episode_index"])
    return np.asarray(table.column("episode_index").to_numpy(), dtype=np.int64)


def plan_episode_budgets(episodes, horizon, budgets, seed, val_fraction):
    """返回每档的训练 / 验证 episode。验证集不随预算变化，训练是前缀。"""
    _train_index, _val_index, chosen, val_ids, _pool = _select_indices(
        episodes, horizon, None, seed, val_fraction,
    )
    del _train_index, _val_index, _pool
    plans = []
    for budget in budgets:
        if budget is None or int(budget) >= len(chosen):
            train_ids = [int(item) for item in chosen]
        else:
            train_ids = [int(item) for item in chosen[: int(budget)]]
        if not train_ids:
            raise ValueError("这一档没有训练 episode")
        plans.append({
            "train_episode_ids": train_ids,
            "val_episode_ids": [int(item) for item in val_ids],
            "episodes": len(train_ids),
            "frames": count_frames(episodes, train_ids),
            "language_conditioning": "unsupported_by_act_0.6.1",
            "image_key": "observation.image",
        })
    return plans


def count_frames(episodes, episode_ids):
    wanted = {int(item) for item in episode_ids}
    return int(sum(int(episode) in wanted for episode in episodes))


def copy_baseline_l1(dataset_dir, val_episode_ids, horizon=None):
    """验证集上「保持不动」相对真实增量的平均绝对误差，原始动作单位。"""
    _state, action, episodes = _load_arrays(dataset_dir)
    del _state
    horizon = _read_horizon(dataset_dir, horizon)
    wanted = {int(item) for item in val_episode_ids}
    chunks = []
    for episode_id, start, length in _spans(episodes):
        if episode_id not in wanted:
            continue
        chunks.append(_chunk_starts(start, length, horizon))
    if not chunks:
        raise ValueError("验证 episode 里没有足够长的动作块")
    indices = np.concatenate(chunks)
    targets = _stack_targets(action, indices, horizon)
    baseline = _copy_current_vector(action.shape[1], horizon)
    return float(np.mean(np.abs(targets - baseline)))


def _lerobot_train_command():
    """pip 的脚本有时在 ~/.local/bin，不在当前 PATH 里。"""
    found = shutil.which("lerobot-train")
    if found:
        return found
    candidates = [
        Path(sys.executable).resolve().parent / "lerobot-train",
        Path.home() / ".local" / "bin" / "lerobot-train",
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    return "lerobot-train"


def lerobot_train_argv(
    dataset_dir,
    train_episode_ids,
    output_dir,
    steps,
    batch_size,
    device,
    seed,
    chunk_size=16,
):
    """lerobot 0.6.1 ACT。图像由数据集的 observation.image 提供，不传语言参数。"""
    episode_list = ",".join(str(int(item)) for item in train_episode_ids)
    log_freq = 1 if int(steps) < 20 else 50
    workers = 0 if device == "cpu" else 2
    return [
        _lerobot_train_command(),
        "--dataset.repo_id=local/egodex",
        "--dataset.root=%s" % Path(dataset_dir).resolve(),
        "--dataset.episodes=[%s]" % episode_list,
        "--policy.type=act",
        "--policy.device=%s" % device,
        "--policy.push_to_hub=false",
        "--policy.chunk_size=%d" % int(chunk_size),
        "--policy.n_action_steps=%d" % int(chunk_size),
        "--policy.n_obs_steps=1",
        "--batch_size=%d" % int(batch_size),
        "--steps=%d" % int(steps),
        "--eval_steps=0",
        "--save_freq=%d" % max(int(steps), 1),
        "--log_freq=%d" % log_freq,
        "--num_workers=%d" % workers,
        "--seed=%d" % int(seed),
        "--wandb.enable=false",
        "--policy.optimizer_lr=1e-5",
        "--policy.optimizer_lr_backbone=1e-5",
        "--output_dir=%s" % Path(output_dir).resolve(),
        "--job_name=egodex_act_%s" % Path(output_dir).name,
    ]


def format_metrics_log(plan, val_loss, copy_loss, steps, device):
    """缩放律能读的日志。val_loss 与 copy 都是验证集上的动作 L1。"""
    ratio = float(val_loss) / float(copy_loss) if copy_loss else float("nan")
    lines = [
        "policy=act learner=lerobot device=%s frames=%d episodes=%d val_episodes=%d steps=%d"
        % (device, plan["frames"], plan["episodes"], len(plan["val_episode_ids"]), int(steps)),
        "image_key=%s" % plan["image_key"],
        "language_conditioning=%s" % plan["language_conditioning"],
        "copy_current_wrist: %.6e" % float(copy_loss),
        "step=%d val_loss: %.6e" % (int(steps), float(val_loss)),
        "val_baseline_ratio: %.6e" % ratio,
    ]
    return "\n".join(lines) + "\n"


def _parse_list(text, allow_all=False):
    values = []
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        if allow_all and part in ("all", "none"):
            values.append(None)
        else:
            values.append(int(part))
    if not values:
        raise ValueError("列表是空的")
    return values


def _align_steps(budgets, steps):
    if len(steps) == 1 and len(budgets) > 1:
        return steps * len(budgets)
    if len(steps) != len(budgets):
        raise ValueError("steps 的个数要等于 sizes，或只给一个步数")
    return steps


def build_plans(dataset_dir, sizes_text, steps_text, seed, val_fraction, horizon, batch_size, device, out_dir):
    episodes = read_episode_index(dataset_dir)
    horizon = _read_horizon(dataset_dir, horizon)
    budgets = _parse_list(sizes_text, allow_all=True)
    steps = _align_steps(budgets, _parse_list(steps_text))
    plans = plan_episode_budgets(episodes, horizon, budgets, seed, val_fraction)
    out_dir = Path(out_dir)
    runs = []
    for index, (plan, step_count) in enumerate(zip(plans, steps)):
        name = "act_%s" % ("all" if budgets[index] is None else str(budgets[index]))
        run_dir = out_dir / name
        log_path = out_dir / ("%s.log" % name)
        argv = lerobot_train_argv(
            dataset_dir, plan["train_episode_ids"], run_dir, step_count, batch_size, device, seed, horizon,
        )
        plan.update({
            "name": name,
            "steps": int(step_count),
            "output_dir": str(run_dir),
            "log": str(log_path),
            "argv": argv,
            "horizon": int(horizon),
        })
        runs.append(plan)
    return runs


def write_runs_yaml(runs, path):
    lines = ["runs:"]
    for run in runs:
        lines.append("  - name: %s" % run["name"])
        lines.append("    size: %d" % int(run["frames"]))
        lines.append("    log: %s" % Path(run["log"]).name)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _find_policy_dir(output_dir):
    root = Path(output_dir)
    candidates = [root, root / "pretrained_model"]
    candidates.extend(root.glob("checkpoints/**/pretrained_model"))
    for path in candidates:
        if (path / "config.json").is_file():
            return path
    raise FileNotFoundError("在 %s 下没有找到 ACT config.json" % output_dir)


def _batch_from_item(item, device):
    import torch

    batch = {}
    for key, value in item.items():
        if key != "action" and not str(key).startswith("observation."):
            continue
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        if key == "observation.image" and value.dtype == torch.uint8:
            value = value.float() / 255.0
        batch[key] = value.unsqueeze(0).to(device)
    return batch


def evaluate_act_l1(dataset_dir, policy_dir, val_episode_ids, device, horizon, max_batches=4):
    """验证集上的动作 L1，反归一化之后，和保持不动基线同一单位。需要 lerobot 0.6.1。"""
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    note = json.loads((Path(dataset_dir) / "meta" / "egodata_export.json").read_text(encoding="utf-8"))
    info = json.loads((Path(dataset_dir) / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = float(info.get("fps") or 10.0)
    horizon = int(horizon or note.get("horizon") or 16)
    delta = 1.0 / fps
    dataset = LeRobotDataset(
        "local/egodex",
        root=str(Path(dataset_dir).resolve()),
        episodes=[int(item) for item in val_episode_ids],
        delta_timestamps={"action": [index * delta for index in range(horizon)]},
    )
    policy_path = _find_policy_dir(policy_dir)
    policy = ACTPolicy.from_pretrained(str(policy_path))
    policy.eval()
    policy.to(device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(policy_path),
    )
    total = 0.0
    count = 0
    limit = min(len(dataset), max(1, int(max_batches) * 2))
    for index in range(limit):
        item = dataset[index]
        action = item["action"]
        if not torch.is_tensor(action):
            action = torch.as_tensor(action)
        action = action.to(dtype=torch.float32, device="cpu")
        batch = preprocessor(_batch_from_item(item, device))
        with torch.no_grad():
            normalized = policy.predict_action_chunk(batch)
        prediction = postprocessor(normalized)
        prediction = prediction.detach().to(dtype=torch.float32, device="cpu")
        if prediction.ndim == 3:
            prediction = prediction[0]
        width = min(prediction.shape[-1], action.shape[-1])
        steps = min(prediction.shape[0], action.shape[0])
        total += torch.mean(torch.abs(prediction[:steps, :width] - action[:steps, :width])).item()
        count += 1
    if count < 1:
        raise ValueError("验证集是空的")
    return total / count


def main(argv=None):
    parser = argparse.ArgumentParser(description="安排并可选地跑 ACT 缩放实验")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sizes", default="64,256,1024,all", help="训练 episode 数，all 表示全部训练 episode")
    parser.add_argument("--steps", default="300,800,1500,2000")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--out-dir", default="outputs/ego_act_scaling")
    parser.add_argument("--dry-run", action="store_true", help="只打印命令，不训练")
    parser.add_argument("--run", action="store_true", help="调用 lerobot-train，再写验证日志和缩放报告")
    parser.add_argument("--max-eval-batches", type=int, default=4)
    args = parser.parse_args(argv)
    runs = build_plans(
        args.dataset, args.sizes, args.steps, args.seed, args.val_fraction,
        args.horizon, args.batch_size, args.device, args.out_dir,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        print(" ".join(run["argv"]))
        print(
            "episodes %d frames %d val_episodes %d language %s"
            % (run["episodes"], run["frames"], len(run["val_episode_ids"]), run["language_conditioning"])
        )
    if args.dry_run or not args.run:
        return 0
    for run in runs:
        if Path(run["output_dir"]).exists():
            raise SystemExit("%s 已存在。lerobot-train 在 resume=false 时不会覆盖它。" % run["output_dir"])
        subprocess.check_call(run["argv"])
        copy_loss = copy_baseline_l1(args.dataset, run["val_episode_ids"], run["horizon"])
        val_loss = evaluate_act_l1(
            args.dataset, run["output_dir"], run["val_episode_ids"], args.device,
            run["horizon"], args.max_eval_batches,
        )
        Path(run["log"]).write_text(
            format_metrics_log(run, val_loss, copy_loss, run["steps"], args.device),
            encoding="utf-8",
        )
        print("val_loss %.6e copy_current_wrist %.6e log %s" % (val_loss, copy_loss, run["log"]))
    runs_yaml = out_dir / "runs.yaml"
    write_runs_yaml(runs, runs_yaml)
    report = out_dir / "scaling_law.html"
    subprocess.check_call([
        sys.executable, str(Path(__file__).resolve().parent / "scaling_law.py"),
        "--runs", str(runs_yaml),
        "--out", str(report),
    ])
    if report.is_file():
        print("缩放律报告 %s" % report)
    else:
        print("有效档不足 2 个，未写缩放律报告。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
