# -*- coding: utf-8 -*-
"""开环回放和闭环滚动。数字只来自这里真正步进仿真的结果。"""
import numpy as np

from retarget.frames import CONTROL_DT, GRIP_OPEN, HOME_EE, goal_xyz
from retarget.paths import path_duration, query_path, scripted_segments
from sim.bc import decode_action, encode_waypoint, relative_features

# 每 0.1 秒命令最多前进这么多，避免一下扑到路点上。
COMMAND_STEP_M = 0.03


def segment_waypoint(segments, time_s):
    """当前所在那一段的终点：演示里这一段要去的地方。"""
    covered = 0.0
    for segment in segments:
        covered += segment.duration
        if time_s <= covered + 1e-9:
            return query_path(segments, covered)
    return query_path(segments, covered)


def follow_reference(env, cube_xy, goal_xy, segments, control_dt=CONTROL_DT, record=False):
    """用末端位置伺服跟上一条参考路径。可选地记下相对特征和路点动作。"""
    env.reset(cube_xy)
    duration = path_duration(segments)
    features = []
    actions = []
    solutions = []
    t = 0.0
    while t < duration - 1e-9:
        step = min(float(control_dt), duration - t)
        obs = env.observe(goal_xy)
        if record:
            features.append(relative_features(obs))
            waypoint, grip_label = segment_waypoint(segments, t + 1e-4)
            actions.append(encode_waypoint(waypoint, grip_label, obs["cube"][:2], obs["goal"][:2]))
        position, grip = query_path(segments, t + step)
        solutions.append(env.track_ee(position, grip, step))
        t += step
    return {
        "success": env.success(goal_xy),
        "xy_error": env.xy_error(goal_xy),
        "cube": env.cube_position().copy(),
        "features": np.vstack(features) if features else np.zeros((0, 9)),
        "actions": np.vstack(actions) if actions else np.zeros((0, 4)),
        "median_ik_err": float(np.median([item.pos_err for item in solutions])) if solutions else None,
        "goal_xyz": goal_xyz(goal_xy),
    }


def replay_joints(env, cube_xy, goal_xy, times, q_traj, grip_traj):
    """把已经求好的关节轨迹开环放进仿真。"""
    env.reset(cube_xy)
    # 先在 0.3 秒内从复位姿态走到轨迹第一帧，夹爪保持张开。
    # 轨迹若从「已经夹紧」开始，合着爪冲过去会把方块推走。
    env.data.ctrl[:6] = q_traj[0]
    env.data.ctrl[6:] = GRIP_OPEN
    warmup = max(1, int(round(0.3 / env.dt)))
    import mujoco

    for _ in range(warmup):
        mujoco.mj_step(env.model, env.data)
    env.play_joints(times, q_traj, grip_traj)
    return {
        "success": env.success(goal_xy),
        "xy_error": env.xy_error(goal_xy),
        "cube": env.cube_position().copy(),
    }


def run_oracle(env, cube_xy, goal_xy, control_dt=CONTROL_DT):
    """脚本专家：知道起点和目标，走没有侧向弧的抓放路径。这是闭环成功率的上界。"""
    segments = scripted_segments(cube_xy, goal_xy, arc_m=0.0)
    result = follow_reference(env, cube_xy, goal_xy, segments, control_dt=control_dt, record=False)
    result["kind"] = "oracle"
    return result


def collect_robot_demo(env, cube_xy, goal_xy, control_dt=CONTROL_DT):
    segments = scripted_segments(cube_xy, goal_xy, arc_m=0.0)
    result = follow_reference(env, cube_xy, goal_xy, segments, control_dt=control_dt, record=True)
    result["kind"] = "robot"
    return result


def run_policy(env, policy, cube_xy, goal_xy, horizon_s, control_dt=CONTROL_DT):
    """闭环：预测相对路点，命令以有限步长走向它，再用 IK 跟上。"""
    env.reset(cube_xy)
    command = np.array(HOME_EE, dtype=float)
    t = 0.0
    while t < horizon_s - 1e-9:
        step = min(float(control_dt), horizon_s - t)
        obs = env.observe(goal_xy)
        action = policy.act(relative_features(obs))
        waypoint, grip = decode_action(action, obs["cube"][:2], obs["goal"][:2])
        delta = waypoint - command
        distance = float(np.linalg.norm(delta))
        if distance > COMMAND_STEP_M:
            delta = delta * (COMMAND_STEP_M / distance)
        command = command + delta
        env.track_ee(command, grip, step)
        t += step
    return {
        "success": env.success(goal_xy),
        "xy_error": env.xy_error(goal_xy),
        "cube": env.cube_position().copy(),
    }
