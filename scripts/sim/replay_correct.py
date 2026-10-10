# -*- coding: utf-8 -*-
"""靠近物体时在线修正，而不是把关节轨迹刚性放完。

接近段仍走已经求好的关节轨迹（这条在真实 EgoDex 上比逐帧笛卡尔跟踪更能夹住）。
到达抓住时刻之后，末端每步最多走向「仿真物体 + 记录的相对抓取偏移」1.5 cm，
高度是这一场景的夹取高度。物体没有抬起超过 1 cm 时，只再抓一次：张开、抬高 3 cm、
下降、合上。然后从抬起帧起重新求逆运动学，把剩下的末端目标走完。

这些步长在看成功率之前就定好。
"""
import numpy as np

from retarget.frames import CONTROL_DT, GRIP_CLOSE, GRIP_OPEN, R_DOWN, Z_HOVER
from retarget.paths import event_segments
from sim.bc import encode_waypoint, relative_features
from sim.rollout import segment_waypoint

SERVO_STEP_M = 0.015
SERVO_DT = 0.02
SERVO_WINDOW_S = 0.8
REGRASP_UP_M = 0.03
LIFT_DETECT_M = 0.01
WARMUP_S = 0.3


def _play_until(env, times, q_traj, grip_traj, t_end, on_step=None):
    """从 0 放到 ``t_end``。``on_step(dt)`` 在每个物理步之后调用。"""
    import mujoco

    from sim.arm import _interp

    times = np.asarray(times, dtype=float)
    if times.shape[0] < 2 or float(t_end) <= float(times[0]) + 1e-8:
        return 0.0
    index = int(np.searchsorted(times, t_end, side="right"))
    index = max(2, min(index, times.shape[0]))
    span_t = times[:index]
    span_q = q_traj[:index]
    span_g = grip_traj[:index]
    t = 0.0
    end = min(float(t_end), float(span_t[-1]))
    while t < end - 1e-12:
        q_cmd, g_cmd = _interp(span_t, span_q, span_g, t)
        env.data.ctrl[:6] = q_cmd
        env.data.ctrl[6:] = g_cmd
        mujoco.mj_step(env.model, env.data)
        t += env.dt
        if on_step is not None:
            on_step(env.dt)
    return t


def _servo_to(env, target, grip, step_m=SERVO_STEP_M, dt=SERVO_DT, max_time=1.0, on_step=None):
    command = env.ee_pose()[0].copy()
    elapsed = 0.0
    target = np.asarray(target, dtype=float)
    while elapsed < max_time - 1e-9:
        delta = target - command
        distance = float(np.linalg.norm(delta))
        step = dt if distance > step_m else dt
        if distance <= step_m:
            command = target.copy()
        else:
            command = command + delta * (step_m / distance)
        env.track_ee(command, grip, step)
        elapsed += step
        if on_step is not None:
            on_step(step)
        if distance <= step_m:
            break
    return elapsed


def replay_online(
    env,
    cube_xy,
    goal_xy,
    times,
    q_traj,
    grip_traj,
    ee_target,
    grasp_index,
    release_index,
    offset_xy,
    grasp_z,
    rest_z=None,
    record=False,
):
    """在线修正加最多一次再抓。``offset_xy`` 是抓住时末端相对物体中心的水平偏移。"""
    import mujoco

    times = np.asarray(times, dtype=float)
    q_traj = np.asarray(q_traj, dtype=float)
    grip_traj = np.asarray(grip_traj, dtype=float)
    ee_target = np.asarray(ee_target, dtype=float)
    grasp = int(grasp_index)
    release = int(release_index)
    env.reset(cube_xy, rest_z=rest_z)
    env.data.ctrl[:6] = q_traj[0]
    env.data.ctrl[6:] = GRIP_OPEN
    warmup = max(1, int(round(WARMUP_S / env.dt)))
    clock = 0.0
    samples = []
    next_sample = 0.0

    def note():
        nonlocal next_sample
        if not record:
            return
        if next_sample <= clock + 1e-9:
            samples.append((clock, env.observe(goal_xy)))
            next_sample += float(CONTROL_DT)

    def advance(dt):
        nonlocal clock
        clock += float(dt)
        note()

    for _ in range(warmup):
        note()
        mujoco.mj_step(env.model, env.data)
        clock += env.dt
    note()
    _play_until(env, times, q_traj, grip_traj, float(times[grasp]), on_step=advance)
    start_z = float(env.cube_position()[2])
    offset = np.asarray(offset_xy, dtype=float).reshape(2)
    grip_hold = float(grip_traj[min(grasp, grip_traj.shape[0] - 1)])
    command = env.ee_pose()[0].copy()
    elapsed = 0.0
    while elapsed < SERVO_WINDOW_S - 1e-9:
        cube = env.cube_position()
        desired = np.array([cube[0] + offset[0], cube[1] + offset[1], float(grasp_z)], dtype=float)
        delta = desired - command
        distance = float(np.linalg.norm(delta))
        if distance > SERVO_STEP_M:
            delta = delta * (SERVO_STEP_M / distance)
        command = command + delta
        env.track_ee(command, grip_hold, SERVO_DT)
        elapsed += SERVO_DT
        advance(SERVO_DT)
        if float(np.linalg.norm(env.ee_pose()[0] - desired)) <= SERVO_STEP_M:
            break
    lifted = float(env.cube_position()[2]) > start_z + LIFT_DETECT_M
    regrasped = False
    if not lifted:
        regrasped = True
        ee = env.ee_pose()[0]
        up = ee.copy()
        up[2] = min(float(Z_HOVER), float(ee[2]) + REGRASP_UP_M)
        _servo_to(env, up, GRIP_OPEN, max_time=0.4, on_step=advance)
        cube = env.cube_position()
        hover = np.array([cube[0] + offset[0], cube[1] + offset[1], up[2]], dtype=float)
        _servo_to(env, hover, GRIP_OPEN, max_time=0.8, on_step=advance)
        down = hover.copy()
        down[2] = float(grasp_z)
        _servo_to(env, down, GRIP_OPEN, max_time=0.8, on_step=advance)
        env.track_ee(down, GRIP_CLOSE, 0.4)
        advance(0.4)
    lift_index = grasp
    for index in range(grasp, release + 1):
        if float(ee_target[index, 2]) > float(ee_target[grasp, 2]) + 0.02:
            lift_index = index
            break
    if lift_index < times.shape[0] - 1:
        remain_t = times[lift_index:] - times[lift_index]
        q = env.arm_q().copy()
        qs = []
        for position in ee_target[lift_index:]:
            solved = env.ik(q, position, R_DOWN)
            q = solved.q
            qs.append(q.copy())
        _play_until(env, remain_t, np.stack(qs), grip_traj[lift_index:], float(remain_t[-1]), on_step=advance)
    result = {
        "success": env.success(goal_xy),
        "xy_error": env.xy_error(goal_xy),
        "cube": env.cube_position().copy(),
        "regrasped": bool(regrasped),
        "lifted_before_regrasp": bool(lifted),
    }
    if record:
        total = max(clock, 1e-3)
        grasp_t = min(WARMUP_S + float(times[grasp]), 0.7 * total)
        release_t = min(max(grasp_t + 0.2 * total, WARMUP_S + float(times[release])), 0.92 * total)
        segments = event_segments(cube_xy, goal_xy, grasp_t, release_t, total, z_grasp=grasp_z)
        features = []
        actions = []
        for sample_t, obs in samples:
            if sample_t >= total:
                continue
            features.append(relative_features(obs))
            waypoint, grip_label = segment_waypoint(segments, sample_t + 1e-4)
            actions.append(encode_waypoint(waypoint, grip_label, obs["cube"][:2], obs["goal"][:2]))
        result["features"] = np.vstack(features) if features else np.zeros((0, 9))
        result["actions"] = np.vstack(actions) if actions else np.zeros((0, 4))
    return result
