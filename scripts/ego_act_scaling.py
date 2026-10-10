#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""为 lerobot 0.6.1 的 ACT 安排多档数据量，并共用线性 BC 的验证 episode。

ACT 吃 ``observation.image`` 和 ``observation.state``，动作块长度 16。
lerobot 0.6.1 的 ACT 没有语言编码器。任务句子已经写在数据集的 task 字段里，
这里不另造一个语言开关。换到带语言编码器的策略时可以直接用这些句子。

``action_valid`` 为 0 的手不进验证 L1，也不进「保持不动」基线。
两只手都无效的步并进 ``action_is_pad``；只有一只手无效时用按维掩码，
避免把另一只手的实测增量也丢掉。训练入口是 ``scripts/ego_act_train.py``，
它在调用 lerobot 之前装上这个掩码。没有 ``action_valid`` 列时按全部有效，并警告。

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
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

from egodata.action_valid import (  # noqa: E402
    action_is_pad_from_valid,
    chunk_valid_fraction,
    element_mask,
    masked_l1,
    stack_chunk_mask,
)
from ego_pretrain_bc import (  # noqa: E402
    _chunk_starts,
    _copy_current_vector,
    _load_arrays,
    _read_horizon,
    _select_indices,
    _spans,
    _stack_targets,
    load_action_valid,
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


def copy_baseline_l1(dataset_dir, val_episode_ids, horizon=None, min_valid_fraction=0.0):
    """验证集上「保持不动」相对真实增量的平均绝对误差，原始动作单位。

    无效手不计入。``min_valid_fraction`` 大于 0 时，比例不够的块整块跳过。
    """
    _state, action, episodes = _load_arrays(dataset_dir)
    del _state
    action_valid = load_action_valid(dataset_dir, len(action))
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
    if float(min_valid_fraction) > 0.0:
        kept = [
            int(start)
            for start in indices
            if chunk_valid_fraction(action_valid, int(start), horizon) + 1e-12 >= float(min_valid_fraction)
        ]
        if not kept:
            return 0.0
        indices = np.asarray(kept, dtype=np.int64)
    targets = _stack_targets(action, indices, horizon)
    baseline = _copy_current_vector(action.shape[1], horizon)
    mask = stack_chunk_mask(action_valid, indices, horizon, action.shape[1])
    return masked_l1(targets, baseline, mask)


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
        sys.executable,
        str(Path(__file__).resolve().parent / "ego_act_train.py"),
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


def evaluate_act_l1(
    dataset_dir, policy_dir, val_episode_ids, device, horizon, max_batches=4, min_valid_fraction=0.0,
):
    """验证集上的动作 L1，反归一化之后，和保持不动基线同一单位。需要 lerobot 0.6.1。

    只在有效手上平均。比例低于 ``min_valid_fraction`` 的块不参与。
    """
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
    valid_table = load_action_valid(dataset_dir)
    total = 0.0
    weight = 0.0
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
        width = min(int(prediction.shape[-1]), int(action.shape[-1]))
        steps = min(int(prediction.shape[0]), int(action.shape[0]))
        row = int(np.asarray(item["index"]).reshape(-1)[0]) if "index" in item else None
        window = np.ones((steps, 2), dtype=np.float64)
        if row is not None:
            take = max(0, min(steps, len(valid_table) - row))
            if take:
                window[:take] = valid_table[row:row + take]
            if take < steps:
                window[take:] = 0.0
        pad = None
        if "action_is_pad" in item:
            pad = np.asarray(item["action_is_pad"]).reshape(-1)[:steps]
        if float(min_valid_fraction) > 0.0:
            if chunk_valid_fraction(window, 0, steps) + 1e-12 < float(min_valid_fraction):
                continue
        mask = element_mask(pad, window, width)
        error = torch.abs(prediction[:steps, :width] - action[:steps, :width]).detach().cpu().numpy()
        total += float(np.sum(error * mask))
        weight += float(np.sum(mask))
    if weight <= 0.0:
        raise ValueError("验证集是空的，或有效动作被掩码滤空了")
    return total / weight


def _cached_action_valid(root):
    if root is None:
        return None
    key = str(root)
    cache = getattr(_cached_action_valid, "cache", {})
    if key not in cache:
        try:
            cache[key] = load_action_valid(key)
        except (OSError, ValueError):
            cache[key] = None
        _cached_action_valid.cache = cache
    return cache[key]


def _attach_action_valid_item(dataset, item):
    """把这一块的 action_valid 放进样本，并把两只手都无效的步并进 action_is_pad。"""
    import torch

    action = item.get("action")
    if action is None or "index" not in item:
        return item
    if not torch.is_tensor(action):
        action = torch.as_tensor(action)
    horizon = 1 if action.ndim == 1 else int(action.shape[0])
    table = _cached_action_valid(getattr(dataset, "root", None))
    if table is None:
        return item
    row = int(np.asarray(item["index"]).reshape(-1)[0])
    window = np.zeros((horizon, 2), dtype=np.float32)
    take = max(0, min(horizon, len(table) - row))
    if take:
        window[:take] = table[row:row + take]
    pad = None
    if "action_is_pad" in item:
        pad = np.asarray(item["action_is_pad"], dtype=bool).reshape(-1)[:horizon]
        if pad.shape[0] < horizon:
            pad = np.pad(pad, (0, horizon - pad.shape[0]), constant_values=True)
        window[pad] = 0.0
    minimum = float(os.environ.get("EGO_MIN_VALID_FRACTION", "0") or 0)
    if minimum > 0.0 and chunk_valid_fraction(window, 0, horizon) + 1e-12 < minimum:
        window[:] = 0.0
    item["action_valid"] = torch.as_tensor(window)
    item["action_is_pad"] = torch.as_tensor(action_is_pad_from_valid(window, pad))
    return item


def _torch_hand_mask(valid, action_dim):
    import torch

    left = (valid[..., 0] >= 0.5).to(dtype=torch.float32)
    right = (valid[..., 1] >= 0.5).to(dtype=torch.float32)
    mask = torch.zeros(*valid.shape[:-1], int(action_dim), dtype=left.dtype, device=valid.device)
    if action_dim >= 7:
        mask[..., 0:7] = left.unsqueeze(-1)
    if action_dim >= 14:
        mask[..., 7:14] = right.unsqueeze(-1)
    joint = 63
    if action_dim >= 14 + joint:
        mask[..., 14:14 + joint] = left.unsqueeze(-1)
    if action_dim >= 14 + 2 * joint:
        mask[..., 14 + joint:14 + 2 * joint] = right.unsqueeze(-1)
    return mask


def _act_constants():
    import importlib

    for module_name in ("lerobot.utils.constants", "lerobot.common.constants", "lerobot.constants"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        action = getattr(module, "ACTION", None)
        images = getattr(module, "OBS_IMAGES", None)
        if action is not None:
            return action, images
    return "action", None


def _act_forward_masked(policy, batch):
    """复刻 ACT 的 L1，但分母只数有效手的维度。"""
    import torch
    import torch.nn.functional as F

    batch = dict(batch)
    action_key, images_key = _act_constants()
    image_features = getattr(policy.config, "image_features", None)
    if image_features and images_key is not None:
        batch[images_key] = [batch[key] for key in image_features]
    valid = batch["action_valid"]
    if not torch.is_tensor(valid):
        valid = torch.as_tensor(valid, device=batch[action_key].device)
    if valid.ndim == 2:
        valid = valid.unsqueeze(0)
    pad = batch["action_is_pad"]
    if not torch.is_tensor(pad):
        pad = torch.as_tensor(pad, device=valid.device)
    if pad.ndim == 1:
        pad = pad.unsqueeze(0)
    both_invalid = valid.sum(dim=-1) < 0.5
    pad = pad.bool() | both_invalid.to(device=pad.device)
    batch["action_is_pad"] = pad
    actions_hat, latent = policy.model(batch)
    mu_hat, log_sigma = (None, None)
    if isinstance(latent, tuple) and len(latent) == 2:
        mu_hat, log_sigma = latent
    abs_err = F.l1_loss(batch[action_key], actions_hat, reduction="none")
    element = _torch_hand_mask(valid.to(device=abs_err.device), abs_err.shape[-1])
    element = element * (~pad).unsqueeze(-1).to(dtype=abs_err.dtype, device=abs_err.device)
    l1_loss = (abs_err * element).sum() / element.sum().clamp_min(1)
    loss_dict = {"l1_loss": float(l1_loss.detach().cpu())}
    use_vae = bool(getattr(policy.config, "use_vae", False))
    if use_vae and log_sigma is not None and mu_hat is not None:
        mean_kld = (
            (-0.5 * (1 + log_sigma - mu_hat.pow(2) - log_sigma.exp())).sum(-1).mean()
        )
        loss_dict["kld_loss"] = float(mean_kld.detach().cpu())
        loss = l1_loss + mean_kld * policy.config.kl_weight
    else:
        loss = l1_loss
    return loss, loss_dict


def install_act_hand_mask():
    """在当前进程里给 lerobot 的数据集和 ACT 损失接上 action_valid。

    找不到对应类时只警告。两只手都无效的步写入 action_is_pad；
    只有一只手无效时，损失按维屏蔽。
    """
    import importlib

    dataset_cls = None
    for module_name in (
        "lerobot.datasets.lerobot_dataset",
        "lerobot.common.datasets.lerobot_dataset",
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        dataset_cls = getattr(module, "LeRobotDataset", None)
        if dataset_cls is not None:
            break
    policy_cls = None
    for module_name in (
        "lerobot.policies.act.modeling_act",
        "lerobot.common.policies.act.modeling_act",
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        policy_cls = getattr(module, "ACTPolicy", None)
        if policy_cls is not None:
            break
    if dataset_cls is not None and not getattr(dataset_cls, "_egodata_action_valid", False):
        original_get = dataset_cls.__getitem__

        def getitem(self, idx):
            return _attach_action_valid_item(self, original_get(self, idx))

        dataset_cls.__getitem__ = getitem
        dataset_cls._egodata_action_valid = True
    if policy_cls is not None and not getattr(policy_cls, "_egodata_action_valid", False):
        original_forward = policy_cls.forward

        def forward(self, batch):
            if not isinstance(batch, dict) or "action_valid" not in batch:
                return original_forward(self, batch)
            try:
                return _act_forward_masked(self, batch)
            except Exception as exc:
                warnings.warn(
                    "ACT 按手屏蔽失败，退回原损失：%s" % exc,
                    RuntimeWarning,
                    stacklevel=2,
                )
                return original_forward(self, batch)

        policy_cls.forward = forward
        policy_cls._egodata_action_valid = True
    if dataset_cls is None or policy_cls is None:
        warnings.warn(
            "没有找到 lerobot 的数据集或 ACT，训练损失不会按 action_valid 按手屏蔽。",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    return True


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
    parser.add_argument(
        "--min-valid-fraction",
        type=float,
        default=0.0,
        help="动作块里有效手-步比例低于这个值时，验证不算这块；训练进程里整块标成 pad",
    )
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
        env = os.environ.copy()
        env["EGO_MIN_VALID_FRACTION"] = str(args.min_valid_fraction)
        subprocess.check_call(run["argv"], env=env)
        copy_loss = copy_baseline_l1(
            args.dataset, run["val_episode_ids"], run["horizon"], args.min_valid_fraction,
        )
        val_loss = evaluate_act_l1(
            args.dataset, run["output_dir"], run["val_episode_ids"], args.device,
            run["horizon"], args.max_eval_batches, args.min_valid_fraction,
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
