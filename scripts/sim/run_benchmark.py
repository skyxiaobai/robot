# -*- coding: utf-8 -*-
"""人手重定向回放，以及脚本 / 人类 BC / 机器人 BC / 混合 BC 的闭环成功率。

数字写到 JSON。没有跑完的字段不要手填。

    python scripts/sim/run_benchmark.py --out docs/retarget_sim_results.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retarget.calibrate import default_calibration  # noqa: E402
from retarget.frames import CONTROL_DT, sample_task  # noqa: E402
from retarget.human_demo import make_basic_pick_place  # noqa: E402
from retarget.paths import path_duration, query_path, scripted_segments  # noqa: E402
from retarget.retarget import infer_scene_xy, retarget_episode, task_xy  # noqa: E402
from sim.arm import PickPlaceEnv  # noqa: E402
from sim.bc import BCPolicy, encode_waypoint, relative_features  # noqa: E402
from sim.rollout import (  # noqa: E402
    collect_robot_demo,
    replay_joints,
    run_oracle,
    run_policy,
    segment_waypoint,
)


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


def _episode_row(seed, cube, goal, result):
    return {
        "seed": int(seed),
        "cube_xy": [float(cube[0]), float(cube[1])],
        "goal_xy": [float(goal[0]), float(goal[1])],
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


def collect_retargeted_demo(env, cube_xy, goal_xy, retargeted, control_dt=CONTROL_DT, arc_m=0.03):
    """教师走重定向后的末端（含侧向弧）。标签仍是这一段要去的路点。"""
    segments = scripted_segments(cube_xy, goal_xy, arc_m=arc_m)
    env.reset(cube_xy)
    times = retargeted.times
    duration = float(times[-1])
    features = []
    actions = []
    t = 0.0
    while t < duration - 1e-9:
        step = min(float(control_dt), duration - t)
        obs = env.observe(goal_xy)
        features.append(relative_features(obs))
        waypoint, grip_label = segment_waypoint(segments, t + 1e-4)
        actions.append(encode_waypoint(waypoint, grip_label, obs["cube"][:2], obs["goal"][:2]))
        position = _interp_series(times, retargeted.ee_target, t + step)
        grip = float(_interp_series(times, retargeted.grip, t + step))
        env.track_ee(position, grip, step)
        t += step
    return {
        "success": env.success(goal_xy),
        "xy_error": env.xy_error(goal_xy),
        "cube": env.cube_position().copy(),
        "features": np.vstack(features),
        "actions": np.vstack(actions),
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


def run(n_eval=50, n_train=30, n_replay=20, seed=0, quick=False):
    if quick:
        n_eval = min(n_eval, 4)
        n_train = min(n_train, 4)
        n_replay = min(n_replay, 2)
    started = time.time()
    env = PickPlaceEnv()
    calibration = default_calibration()
    horizon = path_duration(scripted_segments(np.zeros(2), np.array([0.2, 0.0])))
    rng_note = {
        "human_train_seeds": list(range(seed, seed + n_train)),
        "robot_train_seeds": list(range(1000 + seed, 1000 + seed + n_train)),
        "replay_seeds": list(range(2000 + seed, 2000 + seed + n_replay)),
        "eval_seeds": list(range(3000 + seed, 3000 + seed + n_eval)),
        "mixed": "人类训练集的前一半加机器人训练集的前一半，总数与单一来源相同",
    }

    human_features = []
    human_actions = []
    human_demo_rows = []
    for demo_seed in rng_note["human_train_seeds"]:
        rng = np.random.default_rng(demo_seed)
        cube, goal = sample_task(rng)
        episode = make_basic_pick_place(
            cube, goal, calibration, arc_m=0.03, episode_id="synthetic/basic_pick_place/%d" % demo_seed
        )
        retargeted = retarget_episode(episode, env, calibration)
        collected = collect_retargeted_demo(env, cube, goal, retargeted)
        human_features.append(collected["features"])
        human_actions.append(collected["actions"])
        human_demo_rows.append(_episode_row(demo_seed, cube, goal, collected))
        print("human demo", demo_seed, "success", collected["success"], "err", round(collected["xy_error"], 4), flush=True)

    robot_features = []
    robot_actions = []
    robot_demo_rows = []
    for demo_seed in rng_note["robot_train_seeds"]:
        rng = np.random.default_rng(demo_seed)
        cube, goal = sample_task(rng)
        collected = collect_robot_demo(env, cube, goal)
        robot_features.append(collected["features"])
        robot_actions.append(collected["actions"])
        robot_demo_rows.append(_episode_row(demo_seed, cube, goal, collected))
        print("robot demo", demo_seed, "success", collected["success"], "err", round(collected["xy_error"], 4), flush=True)

    replay_rows = []
    ik_err = []
    for demo_seed in rng_note["replay_seeds"]:
        rng = np.random.default_rng(demo_seed)
        cube, goal = sample_task(rng)
        episode = make_basic_pick_place(
            cube, goal, calibration, arc_m=0.03, episode_id="synthetic/basic_pick_place/replay-%d" % demo_seed
        )
        retargeted = retarget_episode(episode, env, calibration)
        ik_err.append(
            {
                "seed": int(demo_seed),
                "median_pos_err_m": retargeted.median_pos_err,
                "max_pos_err_m": retargeted.max_pos_err,
                "n_colliding": int(retargeted.n_colliding),
                "n_lifted": int(retargeted.n_lifted),
            }
        )
        stored = task_xy(episode)
        inferred = infer_scene_xy(retargeted.ee_target, retargeted.grip)
        played = replay_joints(env, cube, goal, retargeted.times, retargeted.q, retargeted.grip)
        row = _episode_row(demo_seed, cube, goal, played)
        row["ik_median_pos_err_m"] = retargeted.median_pos_err
        if inferred is not None:
            row["inferred_cube_xy"] = [float(inferred[0][0]), float(inferred[0][1])]
            row["inferred_goal_xy"] = [float(inferred[1][0]), float(inferred[1][1])]
            row["inferred_vs_task_cube_m"] = float(np.linalg.norm(inferred[0] - cube))
            row["inferred_vs_task_goal_m"] = float(np.linalg.norm(inferred[1] - goal))
        if stored is None:
            row["scene"] = "inferred"
        replay_rows.append(row)
        print("replay", demo_seed, "success", played["success"], "err", round(played["xy_error"], 4), flush=True)

    h_feat = np.vstack(human_features)
    h_act = np.vstack(human_actions)
    r_feat = np.vstack(robot_features)
    r_act = np.vstack(robot_actions)
    half = n_train // 2
    # 每个演示的步数相同。混合集用前一半人类演示和前一半机器人演示，总数与单一来源相同。
    h_per = h_feat.shape[0] // n_train
    r_per = r_feat.shape[0] // n_train
    m_feat = np.vstack([h_feat[: half * h_per], r_feat[: half * r_per]])
    m_act = np.vstack([h_act[: half * h_per], r_act[: half * r_per]])

    policy_h, stats_h = _fit_policy(h_feat, h_act, seed=10)
    policy_r, stats_r = _fit_policy(r_feat, r_act, seed=11)
    policy_m, stats_m = _fit_policy(m_feat, m_act, seed=12)
    print("bc fit", stats_h, stats_r, stats_m, flush=True)

    oracle_rows = []
    human_rows = []
    robot_rows = []
    mixed_rows = []
    for eval_seed in rng_note["eval_seeds"]:
        rng = np.random.default_rng(eval_seed)
        cube, goal = sample_task(rng)
        oracle = run_oracle(env, cube, goal)
        human = run_policy(env, policy_h, cube, goal, horizon)
        robot = run_policy(env, policy_r, cube, goal, horizon)
        mixed = run_policy(env, policy_m, cube, goal, horizon)
        oracle_rows.append(_episode_row(eval_seed, cube, goal, oracle))
        human_rows.append(_episode_row(eval_seed, cube, goal, human))
        robot_rows.append(_episode_row(eval_seed, cube, goal, robot))
        mixed_rows.append(_episode_row(eval_seed, cube, goal, mixed))
        print(
            "eval",
            eval_seed,
            "oracle", int(oracle["success"]),
            "human", int(human["success"]),
            "robot", int(robot["success"]),
            "mixed", int(mixed["success"]),
            flush=True,
        )

    report = {
        "task": "basic_pick_place",
        "human_source": "synthetic EgoDex-schema basic_pick_place (test.zip is not in the repo)",
        "robot": "mujoco franka-like 6-DoF + parallel gripper, CPU headless",
        "policy": "numpy MLP behavior cloning, state only, not image ACT",
        "action": "u along cube-to-goal, lateral offset v in meters, height z, gripper; label is the end of the current demonstration segment",
        "command_step_m": 0.03,
        "success_xy_m": 0.04,
        "success_z_m": [0.20, 0.30],
        "control_dt_s": CONTROL_DT,
        "horizon_s": horizon,
        "n_train_per_source": n_train,
        "mixed_n": half * 2,
        "calibration": {
            "scale": calibration.scale.tolist(),
            "offset": calibration.offset.tolist(),
        },
        "seeds": rng_note,
        "demo_rollout_human": _summary(human_demo_rows),
        "demo_rollout_robot": _summary(robot_demo_rows),
        "replay_retargeted_human": _summary(replay_rows),
        "ik_on_replay": ik_err,
        "train": {"human": stats_h, "robot": stats_r, "mixed": stats_m},
        "closed_loop": {
            "oracle_scripted": _summary(oracle_rows),
            "bc_retargeted_human": _summary(human_rows),
            "bc_robot": _summary(robot_rows),
            "bc_mixed": _summary(mixed_rows),
        },
        "elapsed_s": time.time() - started,
    }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="重定向回放 + 行为克隆闭环评测")
    parser.add_argument("--episodes", type=int, default=50, help="每种闭环策略的评测条数，至少 50 才算正式结果")
    parser.add_argument("--train", type=int, default=40, help="人类演示条数，机器人演示条数与之相同")
    parser.add_argument("--replay", type=int, default=20, help="留出的人类轨迹开环回放条数")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="只跑几条，用来检查命令，不能当成正式成功率")
    parser.add_argument("--out", default="docs/retarget_sim_results.json")
    args = parser.parse_args(argv)
    report = run(
        n_eval=args.episodes,
        n_train=args.train,
        n_replay=args.replay,
        seed=args.seed,
        quick=args.quick,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    closed = report["closed_loop"]
    print("oracle", closed["oracle_scripted"]["success_rate"])
    print("bc_human", closed["bc_retargeted_human"]["success_rate"])
    print("bc_robot", closed["bc_robot"]["success_rate"])
    print("bc_mixed", closed["bc_mixed"]["success_rate"])
    print("replay", report["replay_retargeted_human"]["success_rate"])
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
