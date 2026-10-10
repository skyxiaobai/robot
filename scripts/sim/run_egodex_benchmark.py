# -*- coding: utf-8 -*-
"""用真实 EgoDex basic_pick_place 做重定向重放和闭环行为克隆。

数字只写程序跑出来的。下载失败或筛完不够条数时，JSON 里写原因，不补一个成功率。

    python scripts/retarget/fetch_egodex.py --dest data/egodex_basic_pick_place
    python scripts/sim/run_egodex_benchmark.py --out docs/retarget_egodex_results.json
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from egodata.egodex import load_episode_hdf5  # noqa: E402
from retarget.arkit import (  # noqa: E402
    episode_center,
    fit_shared_calibration,
    height_gain,
    map_centered,
    permute_arkit_xyz,
)
from retarget.calibrate import AxisCalibration  # noqa: E402
from retarget.events import (  # noqa: E402
    MIN_ROBOT_SEPARATION_M,
    pickup_index,
    prepare_episode,
)
from retarget.frames import CONTROL_DT, GRIP_OPEN, R_DOWN, SUCCESS_XY_M  # noqa: E402
from retarget.paths import event_segments, path_duration, scripted_segments  # noqa: E402
from retarget.retarget import retarget_episode  # noqa: E402
from sim.arm import PickPlaceEnv  # noqa: E402
from sim.bc import BCPolicy, encode_waypoint, relative_features  # noqa: E402
from sim.rollout import collect_robot_demo, replay_joints, run_oracle, run_policy, segment_waypoint  # noqa: E402

TRAIN_SEEDS = (0, 1, 2)
SCALE_POINTS = (20, 40, 80)


def _summary(rows):
    n = len(rows)
    n_ok = int(sum(1 for row in rows if row["success"]))
    errors = [float(row["xy_error_m"]) for row in rows]
    return {
        "n": n,
        "n_success": n_ok,
        "success_rate": (float(n_ok) / float(n)) if n else None,
        "mean_xy_error_m": float(np.mean(errors)) if errors else None,
        "median_xy_error_m": float(np.median(errors)) if errors else None,
        "episodes": rows,
    }


def _rate_stats(rates):
    values = [float(item) for item in rates]
    if not values:
        return {"n_seeds": 0, "mean": None, "std": None, "rates": []}
    array = np.asarray(values, dtype=float)
    std = float(np.std(array, ddof=1)) if array.shape[0] >= 2 else None
    return {
        "n_seeds": int(array.shape[0]),
        "mean": float(np.mean(array)),
        "std": std,
        "rates": values,
    }


def _episode_row(index, cube, goal, result):
    return {
        "index": int(index),
        "cube_xy": [float(cube[0]), float(cube[1])],
        "goal_xy": [float(goal[0]), float(goal[1])],
        "separation_m": float(np.linalg.norm(np.asarray(goal) - np.asarray(cube))),
        "success": bool(result["success"]),
        "xy_error_m": float(result["xy_error"]),
        "final_cube_xyz": [float(v) for v in result["cube"]],
    }


def _interp_series(times, values, t):
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if t <= times[0]:
        return values[0]
    if t >= times[-1]:
        return values[-1]
    index = int(np.searchsorted(times, t, side="right") - 1)
    index = max(0, min(index, len(times) - 2))
    span = float(times[index + 1] - times[index])
    alpha = 0.0 if span <= 1e-8 else (float(t) - float(times[index])) / span
    return (1.0 - alpha) * values[index] + alpha * values[index + 1]


def _resample(features, actions, count):
    features = np.asarray(features, dtype=float)
    actions = np.asarray(actions, dtype=float)
    if features.shape[0] == 0:
        raise ValueError("演示没有步")
    if features.shape[0] == 1:
        return np.repeat(features, count, axis=0), np.repeat(actions, count, axis=0)
    indices = np.linspace(0, features.shape[0] - 1, int(count))
    picked = np.rint(indices).astype(int)
    return features[picked], actions[picked]


def _identity_calibration():
    return AxisCalibration(scale=np.ones(3, dtype=float), offset=np.zeros(3, dtype=float))


def _load_prepared(root, limit=None):
    root = Path(root)
    files = sorted(root.glob("*.hdf5"), key=lambda path: int(path.stem))
    if limit is not None:
        files = files[: int(limit)]
    prepared = []
    reasons = Counter()
    for path in files:
        episode = load_episode_hdf5(path)
        item, error = prepare_episode(episode)
        if item is None:
            if isinstance(error, dict):
                reasons["+".join(sorted(set(error.values())))] += 1
            else:
                reasons[str(error)] += 1
            continue
        item["index"] = int(path.stem)
        item["episode_id"] = episode.get("episode_id")
        prepared.append(item)
    return prepared, {"n_files": len(files), "n_prepared": len(prepared), "reject_reasons": dict(reasons)}


def _y_up_report(prepared):
    gains = []
    for item in prepared:
        gains.append(height_gain(item["wrist_xyz"], item["grasp_index"], item["release_index"]))
    gains = np.asarray(gains, dtype=float) if gains else np.zeros((0, 3))
    median = np.median(gains, axis=0) if gains.size else np.zeros(3)
    report = {
        "n": int(gains.shape[0]),
        "median_gain_xyz_m": [float(v) for v in median],
        "fraction_y_gain_over_0_02": float(np.mean(gains[:, 1] > 0.02)) if gains.size else None,
        "y_is_up": bool(gains.size and median[1] > 0.02 and median[1] > median[0] and median[1] > median[2]),
    }
    return report


def _accept_mapped(prepared):
    """减去每条录像自己的中位数后拟合一把标定，丢掉映射后起点和目标太近的片段。"""
    if len(prepared) < 2:
        raise RuntimeError("通过抓放筛选的片段少于 2 条，无法拟合标定")
    centered_rows = []
    grasp_heights = []
    carry_heights = []
    for item in prepared:
        permuted = permute_arkit_xyz(item["wrist_xyz"])
        center = episode_center(permuted)
        item["center"] = center
        centered = permuted - center
        item["centered"] = centered
        centered_rows.append(centered)
        grasp = int(item["grasp_index"])
        release = int(item["release_index"])
        grasp_heights.append(float(centered[grasp, 2]))
        carry_heights.append(float(np.max(centered[grasp:release + 1, 2])))
    calibration, fit_info = fit_shared_calibration(np.vstack(centered_rows), grasp_heights, carry_heights)
    accepted = []
    too_close = 0
    for item in prepared:
        mapped = map_centered(calibration, item["wrist_xyz"], item["center"])
        release = int(item["release_index"])
        grasp = pickup_index(item["wrist_xyz"], item["grip"], item["grasp_index"], release)
        item["pickup_index"] = grasp
        separation = float(np.linalg.norm(mapped[grasp, :2] - mapped[release, :2]))
        item["mapped"] = mapped
        item["separation_m"] = separation
        if separation < MIN_ROBOT_SEPARATION_M:
            too_close += 1
            continue
        accepted.append(item)
    meta = {
        "scale": [float(v) for v in calibration.scale],
        "offset": [float(v) for v in calibration.offset],
        "human_min": [float(v) for v in fit_info["human_min"]],
        "human_max": [float(v) for v in fit_info["human_max"]],
        "z_from_grasp_median": bool(fit_info["z_from_grasp_median"]),
        "n_too_close_after_map": int(too_close),
        "min_robot_separation_m": float(MIN_ROBOT_SEPARATION_M),
        "centering": "subtract per-episode median wrist in robot-axis order before the shared scale",
    }
    return calibration, accepted, meta


def _retarget_one(env, item):
    mapped = np.asarray(item["mapped"], dtype=float)
    quat = np.asarray(item["wrist_quat"], dtype=float)
    wrist = np.concatenate([mapped, quat], axis=1)
    side = item["side"]
    episode = {
        "timestamps": np.asarray(item["times"], dtype=float),
        "hands": {side: {"wrist_pose": wrist}},
    }
    # 手腕已经在机器人坐标里。恒等标定不再缩放。朝向固定向下：EgoDex 的 Hand 在前臂上。
    retargeted = retarget_episode(
        episode,
        env,
        _identity_calibration(),
        side=side,
        grip=item["grip"],
        target_rotation=R_DOWN,
    )
    grasp = int(item.get("pickup_index", item["grasp_index"]))
    release = int(item["release_index"])
    cube = retargeted.ee_target[grasp, :2].copy()
    goal = retargeted.ee_target[release, :2].copy()
    return retargeted, cube, goal


def collect_human_demo(env, cube_xy, goal_xy, retargeted, grasp_index, release_index, control_dt=CONTROL_DT):
    """开环重放同一条关节轨迹，并记下路点标签。

    预热和 ``replay_joints`` 一样：0.3 秒内手臂到位、夹爪张开。末端跟踪在这些真实轨迹上
    抓不住方块，所以训练数据走关节回放，而不是再求一次 IK。
    """
    import mujoco

    times = np.asarray(retargeted.times, dtype=float)
    warmup = 0.3
    grasp_t = float(times[int(grasp_index)]) + warmup
    release_t = float(times[int(release_index)]) + warmup
    total = float(times[-1]) + warmup
    segments = event_segments(cube_xy, goal_xy, grasp_t, release_t, total)
    env.reset(cube_xy)
    features = []
    actions = []
    t = 0.0
    next_sample = 0.0
    while t < total - 1e-12:
        if next_sample <= t + 1e-9 and next_sample < total - 1e-9:
            obs = env.observe(goal_xy)
            features.append(relative_features(obs))
            waypoint, grip_label = segment_waypoint(segments, next_sample + 1e-4)
            actions.append(encode_waypoint(waypoint, grip_label, obs["cube"][:2], obs["goal"][:2]))
            next_sample += float(control_dt)
        if t < warmup:
            env.data.ctrl[:6] = retargeted.q[0]
            env.data.ctrl[6:] = GRIP_OPEN
        else:
            q_cmd = _interp_series(times, retargeted.q, t - warmup)
            grip = float(_interp_series(times, retargeted.grip, t - warmup))
            env.data.ctrl[:6] = q_cmd
            env.data.ctrl[6:] = grip
        mujoco.mj_step(env.model, env.data)
        t += env.dt
    return {
        "success": env.success(goal_xy),
        "xy_error": env.xy_error(goal_xy),
        "cube": env.cube_position().copy(),
        "features": np.vstack(features) if features else np.zeros((0, 9)),
        "actions": np.vstack(actions) if actions else np.zeros((0, 4)),
    }


def _fit_policy(features, actions, seed):
    policy = BCPolicy(hidden=128, seed=seed, lr=3e-3, epochs=250, l2=1e-6)
    stats = policy.fit(features, actions)
    return policy, {
        "train_mse_waypoint": stats.train_mse_pose,
        "train_mse_grip": stats.train_mse_grip,
        "epochs": stats.epochs,
        "n_samples": stats.n_samples,
        "seed": int(seed),
    }


def _stack_prefix(demos, count, step_count):
    feats = []
    acts = []
    for demo in demos[: int(count)]:
        feat, act = _resample(demo["features"], demo["actions"], step_count)
        feats.append(feat)
        acts.append(act)
    return np.vstack(feats), np.vstack(acts)


def _held_out_count(n_accepted, n_eval, quick):
    """留出评测条数。够 20 条训练时，评测不超过调用方给的上限，并且至少给训练留 20 条。"""
    n_accepted = int(n_accepted)
    if n_accepted < 5:
        return 0
    if quick:
        return min(2, max(1, n_accepted // 3))
    if n_accepted <= 20:
        return max(1, n_accepted // 5)
    return min(int(n_eval), n_accepted - 20)


def run(root, n_eval=50, seed_count=3, limit=None, quick=False):
    started = time.time()
    if quick:
        limit = 40 if limit is None else limit
        n_eval = min(int(n_eval), 2)
        seed_count = 1
    prepared, load_meta = _load_prepared(root, limit=limit)
    y_up = _y_up_report(prepared)
    report = {
        "task": "basic_pick_place",
        "human_source": "EgoDex test.zip test/basic_pick_place HDF5 via HTTP Range",
        "object_pose": "EgoDex has no object pose. cube_xy/goal_xy are retargeted wrist xy at grasp and release.",
        "frames": "longest wrist_measured run, trimmed to 1s before grasp through 1s after release, stride 3",
        "orientation": "fixed gripper-down. EgoDex Hand is the forearm, not a palm frame.",
        "robot": "mujoco franka-like 6-DoF + parallel gripper, CPU headless",
        "policy": "numpy MLP behavior cloning, state only, not image ACT",
        "action": "u along cube-to-goal, lateral v, height z, gripper; label is the phase waypoint timed by grasp/release",
        "command_step_m": 0.03,
        "success_xy_m": SUCCESS_XY_M,
        "success_z_m": [0.20, 0.30],
        "load": load_meta,
        "y_up": y_up,
    }
    if not y_up["y_is_up"]:
        report["blocker"] = "抓住到松开之间，Y 轴不是升高最多的轴，拒绝把 Y 当成竖直方向后再去报成功率。"
        report["elapsed_s"] = time.time() - started
        return report
    _calibration, accepted, calib_meta = _accept_mapped(prepared)
    report["calibration"] = calib_meta
    accepted = sorted(accepted, key=lambda item: int(item["index"]))
    n_accepted = len(accepted)
    n_eval = _held_out_count(n_accepted, n_eval, quick)
    train_items = accepted[: n_accepted - n_eval]
    eval_items = accepted[n_accepted - n_eval :]
    eval_ids = {int(item["index"]) for item in eval_items}
    report["split"] = {
        "n_accepted": n_accepted,
        "n_train_pool": len(train_items),
        "n_eval": len(eval_items),
        "train_indices": [int(item["index"]) for item in train_items],
        "eval_indices": [int(item["index"]) for item in eval_items],
        "rule": "按 HDF5 序号排序。序号小的进训练池，最后 n_eval 条留作闭环评测，不进训练集。",
    }
    if len(train_items) < 2 or len(eval_items) < 1:
        report["blocker"] = "筛完之后训练池 %d 条、评测 %d 条，不够做闭环对比。" % (len(train_items), len(eval_items))
        report["elapsed_s"] = time.time() - started
        return report

    env = PickPlaceEnv()
    print("retarget", n_accepted, "accepted episodes", flush=True)
    library = []
    for item in accepted:
        retargeted, cube, goal = _retarget_one(env, item)
        library.append({
            "item": item,
            "retargeted": retargeted,
            "cube": cube,
            "goal": goal,
            "held_out": int(item["index"]) in eval_ids,
        })
        print("retarget", item["index"], "sep", round(float(np.linalg.norm(goal - cube)), 3), "ik", round(retargeted.median_pos_err, 4), flush=True)

    by_index = {int(entry["item"]["index"]): entry for entry in library}
    replay_rows = []
    ik_rows = []
    for entry in library:
        item = entry["item"]
        retargeted = entry["retargeted"]
        played = replay_joints(env, entry["cube"], entry["goal"], retargeted.times, retargeted.q, retargeted.grip)
        row = _episode_row(item["index"], entry["cube"], entry["goal"], played)
        row["side"] = item["side"]
        row["side_fallback"] = bool(item["side_fallback"])
        row["held_out"] = bool(entry["held_out"])
        row["ik_median_pos_err_m"] = retargeted.median_pos_err
        row["ik_max_pos_err_m"] = retargeted.max_pos_err
        row["n_lifted"] = int(retargeted.n_lifted)
        row["n_colliding"] = int(retargeted.n_colliding)
        row["duration_s"] = float(retargeted.times[-1])
        replay_rows.append(row)
        ik_rows.append({
            "index": int(item["index"]),
            "median_pos_err_m": retargeted.median_pos_err,
            "max_pos_err_m": retargeted.max_pos_err,
            "n_lifted": int(retargeted.n_lifted),
            "n_colliding": int(retargeted.n_colliding),
        })
        print("replay", item["index"], "success", played["success"], "err", round(played["xy_error"], 4), flush=True)
    held_in = [row for row in replay_rows if not row["held_out"]]
    held_out = [row for row in replay_rows if row["held_out"]]
    report["replay_retargeted_human"] = _summary(replay_rows)
    report["replay_train_pool"] = _summary(held_in)
    report["replay_held_out"] = _summary(held_out)
    report["ik"] = {
        "median_of_median_pos_err_m": float(np.median([row["median_pos_err_m"] for row in ik_rows])),
        "max_of_max_pos_err_m": float(np.max([row["max_pos_err_m"] for row in ik_rows])),
        "n_episodes_with_lift": int(sum(1 for row in ik_rows if row["n_lifted"])),
        "n_episodes_with_collision": int(sum(1 for row in ik_rows if row["n_colliding"])),
        "episodes": ik_rows,
    }

    replay_ok = {int(row["index"]): bool(row["success"]) for row in replay_rows}
    # 开环没把方块放到目标上的片段，不能当成「任务演示」去监督。机器人演示只用同一批成功场景。
    trainable = [item for item in train_items if replay_ok.get(int(item["index"]), False)]
    report["n_train_replay_success"] = len(trainable)
    print("collect demos", len(trainable), "successful replays out of", len(train_items), flush=True)
    human_demos = []
    robot_demos = []
    human_teacher_rows = []
    robot_teacher_rows = []
    for item in trainable:
        entry = by_index[int(item["index"])]
        human = collect_human_demo(
            env,
            entry["cube"],
            entry["goal"],
            entry["retargeted"],
            item.get("pickup_index", item["grasp_index"]),
            item["release_index"],
        )
        robot = collect_robot_demo(env, entry["cube"], entry["goal"])
        human_demos.append(human)
        robot_demos.append(robot)
        human_teacher_rows.append(_episode_row(item["index"], entry["cube"], entry["goal"], human))
        robot_teacher_rows.append(_episode_row(item["index"], entry["cube"], entry["goal"], robot))
        print("demo", item["index"], "human", human["success"], "robot", robot["success"], flush=True)
    report["teacher_while_collecting"] = {
        "human_track": _summary(human_teacher_rows),
        "robot_scripted": _summary(robot_teacher_rows),
        "note": "只收录训练池里开环重放成功的片段。收集时重放同一条关节轨迹。开环重放的全量成功率见 replay_retargeted_human。",
    }
    if len(trainable) < 2:
        report["blocker"] = "训练池里开环重放成功的只有 %d 条，不够训练行为克隆。" % len(trainable)
        report["elapsed_s"] = time.time() - started
        return report

    step_count = int(round(path_duration(scripted_segments(np.zeros(2), np.array([0.12, 0.0]))) / CONTROL_DT))
    horizon = path_duration(scripted_segments(np.zeros(2), np.array([0.12, 0.0])))
    scales = [int(n) for n in SCALE_POINTS if int(n) <= len(trainable)]
    if 80 not in scales and len(trainable) >= 2 and len(trainable) not in scales:
        scales.append(len(trainable))
    if quick:
        scales = [min(4, len(trainable))]
    seeds = list(TRAIN_SEEDS[: int(seed_count)])
    eval_scenes = [(by_index[int(item["index"])]["cube"], by_index[int(item["index"])]["goal"], int(item["index"])) for item in eval_items]
    oracle_rows = []
    for cube, goal, index in eval_scenes:
        oracle = run_oracle(env, cube, goal)
        oracle_rows.append(_episode_row(index, cube, goal, oracle))
        print("oracle", index, oracle["success"], round(oracle["xy_error"], 4), flush=True)
    report["oracle_scripted"] = _summary(oracle_rows)
    report["horizon_s"] = horizon
    report["resampled_steps_per_demo"] = step_count
    report["scales"] = scales
    report["train_seeds"] = seeds

    closed = {}
    for scale in scales:
        h_feat, h_act = _stack_prefix(human_demos, scale, step_count)
        r_feat, r_act = _stack_prefix(robot_demos, scale, step_count)
        half = scale // 2
        m_feat = np.vstack([h_feat[: half * step_count], r_feat[: half * step_count]])
        m_act = np.vstack([h_act[: half * step_count], r_act[: half * step_count]])
        per_seed = {"bc_retargeted_human": [], "bc_robot": [], "bc_mixed": []}
        seed_details = []
        for seed in seeds:
            policy_h, stats_h = _fit_policy(h_feat, h_act, seed)
            policy_r, stats_r = _fit_policy(r_feat, r_act, seed + 100)
            policy_m, stats_m = _fit_policy(m_feat, m_act, seed + 200)
            rows = {name: [] for name in per_seed}
            for cube, goal, index in eval_scenes:
                human = run_policy(env, policy_h, cube, goal, horizon)
                robot = run_policy(env, policy_r, cube, goal, horizon)
                mixed = run_policy(env, policy_m, cube, goal, horizon)
                rows["bc_retargeted_human"].append(_episode_row(index, cube, goal, human))
                rows["bc_robot"].append(_episode_row(index, cube, goal, robot))
                rows["bc_mixed"].append(_episode_row(index, cube, goal, mixed))
            summarized = {name: _summary(value) for name, value in rows.items()}
            for name, summary in summarized.items():
                per_seed[name].append(summary["success_rate"])
            seed_details.append({
                "seed": int(seed),
                "train": {"human": stats_h, "robot": stats_r, "mixed": stats_m},
                "closed_loop": summarized,
            })
            print(
                "scale", scale, "seed", seed,
                "human", summarized["bc_retargeted_human"]["n_success"],
                "robot", summarized["bc_robot"]["n_success"],
                "mixed", summarized["bc_mixed"]["n_success"],
                "of", len(eval_scenes),
                flush=True,
            )
        closed["n_%d" % scale] = {
            "n_demos_per_source": int(scale),
            "mixed_n": int(half * 2),
            "mixed_rule": "训练池前一半的人类演示加前一半的机器人演示，总数与单一来源相同",
            "scenes": "held-out real episodes, same cube/goal for every method and seed",
            "bc_retargeted_human": _rate_stats(per_seed["bc_retargeted_human"]),
            "bc_robot": _rate_stats(per_seed["bc_robot"]),
            "bc_mixed": _rate_stats(per_seed["bc_mixed"]),
            "seeds": seed_details,
        }
    report["closed_loop"] = closed
    report["elapsed_s"] = time.time() - started
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="真实 EgoDex basic_pick_place 的重放和闭环评测")
    parser.add_argument("--root", default="data/egodex_basic_pick_place")
    parser.add_argument("--out", default="docs/retarget_egodex_results.json")
    parser.add_argument("--eval", type=int, default=50, help="留出的闭环评测条数上限")
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="只读序号最小的若干 HDF5，用来试跑")
    parser.add_argument("--quick", action="store_true", help="只跑几条和 1 个种子，不能当成正式成功率")
    args = parser.parse_args(argv)
    report = run(
        args.root,
        n_eval=args.eval,
        seed_count=args.seed_count,
        limit=args.limit,
        quick=args.quick,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out)
    if report.get("blocker"):
        print("blocker", report["blocker"])
        return 2
    replay = report.get("replay_retargeted_human") or {}
    print("replay", replay.get("n_success"), "/", replay.get("n"))
    oracle = report.get("oracle_scripted") or {}
    print("oracle", oracle.get("n_success"), "/", oracle.get("n"))
    for key, block in (report.get("closed_loop") or {}).items():
        human = block["bc_retargeted_human"]
        robot = block["bc_robot"]
        mixed = block["bc_mixed"]
        print(key, "human", human["mean"], "robot", robot["mean"], "mixed", mixed["mean"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
