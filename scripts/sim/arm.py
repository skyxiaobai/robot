# -*- coding: utf-8 -*-
"""MuJoCo 里的 6 轴抓放环境。

选择这条模型而不是 robosuite / ManiSkill / LIBERO / 官方 Franka mesh：
那些要么要 GPU，要么要额外的网格文件。本文件的 XML 随仓库走，
``pip install mujoco`` 之后可以在 CPU 上无头步进。
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from retarget.frames import (
    ARM_JOINTS,
    ARM_LINK_GEOMS,
    GRIP_JOINTS,
    GRIP_OPEN,
    HOME_EE,
    PAD_GEOMS,
    R_DOWN,
    SUCCESS_XY_M,
    SUCCESS_Z,
    Z_REST,
)
from retarget.ik import damped_least_squares, orientation_error

try:
    import mujoco
except ImportError:  # pragma: no cover - 测试里会跳过
    mujoco = None

XML_PATH = Path(__file__).with_name("franka_like_pick.xml")
IK_SEED = np.array([0.3, -0.8, 1.6, -0.6, 0.0, 0.0], dtype=float)


@dataclass
class IKSolution:
    q: np.ndarray
    pos_err: float
    rot_err: float
    collided: bool
    lifted: bool


def require_mujoco():
    if mujoco is None:
        raise ImportError("需要 mujoco。CPU 环境执行：pip install mujoco")
    return mujoco


class PickPlaceEnv:
    """平行夹爪抓起桌面上的方块，再放到另一个 xy。"""

    def __init__(self, xml_path=None):
        mj = require_mujoco()
        path = Path(xml_path) if xml_path is not None else XML_PATH
        self.model = mj.MjModel.from_xml_path(str(path))
        self.data = mj.MjData(self.model)
        self.dt = float(self.model.opt.timestep)
        self.site_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, "grasp")
        self.joint_ids = [
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINTS
        ]
        self.grip_ids = [
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, name) for name in GRIP_JOINTS
        ]
        self.qadr = [int(self.model.jnt_qposadr[jid]) for jid in self.joint_ids]
        self.grip_adr = [int(self.model.jnt_qposadr[jid]) for jid in self.grip_ids]
        self.dofadr = np.array(
            [int(self.model.jnt_dofadr[jid]) for jid in self.joint_ids], dtype=int
        )
        self.lower = np.array([self.model.jnt_range[jid, 0] for jid in self.joint_ids])
        self.upper = np.array([self.model.jnt_range[jid, 1] for jid in self.joint_ids])
        cube_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "cube_free")
        self.cube_adr = int(self.model.jnt_qposadr[cube_id])
        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))
        self.home_q = self.ik(IK_SEED, HOME_EE, R_DOWN, check_collision=False).q.copy()

    def arm_q(self):
        return np.array([self.data.qpos[adr] for adr in self.qadr], dtype=float)

    def grip_q(self):
        return float(np.mean([self.data.qpos[adr] for adr in self.grip_adr]))

    def ee_pose(self):
        require_mujoco().mj_forward(self.model, self.data)
        position = self.data.site_xpos[self.site_id].copy()
        rotation = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        return position, rotation

    def cube_position(self):
        return self.data.qpos[self.cube_adr:self.cube_adr + 3].copy()

    def observe(self, goal_xy):
        ee, _rotation = self.ee_pose()
        cube = self.cube_position()
        goal = np.array([float(goal_xy[0]), float(goal_xy[1]), Z_REST], dtype=float)
        return {
            "ee": ee,
            "grip": self.grip_q(),
            "cube": cube,
            "goal": goal,
        }

    def success(self, goal_xy, tol=SUCCESS_XY_M):
        """方块中心靠近目标 xy，并且还停在桌面上。"""
        cube = self.cube_position()
        near = float(np.linalg.norm(cube[:2] - np.asarray(goal_xy, dtype=float)[:2])) <= float(tol)
        on_table = SUCCESS_Z[0] <= float(cube[2]) <= SUCCESS_Z[1]
        return bool(near and on_table)

    def xy_error(self, goal_xy):
        cube = self.cube_position()
        return float(np.linalg.norm(cube[:2] - np.asarray(goal_xy, dtype=float)[:2]))

    def contact_pairs(self):
        mj = require_mujoco()
        mj.mj_forward(self.model, self.data)
        mj.mj_collision(self.model, self.data)
        pairs = []
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            name1 = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, int(contact.geom1))
            name2 = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, int(contact.geom2))
            pairs.append((name1, name2, float(contact.dist)))
        return pairs

    def collided(self, pairs=None):
        """连杆撞桌子，或指掌深深扎进桌面。抓取高度的轻微贴合不算。"""
        if pairs is None:
            pairs = self.contact_pairs()
        links = set(ARM_LINK_GEOMS)
        pads = set(PAD_GEOMS)
        for name1, name2, dist in pairs:
            pair = {name1, name2}
            if "table" not in pair:
                continue
            if pair & links:
                return True
            if pair & pads and dist < -0.01:
                return True
        return False

    def ik(self, q_seed, target_pos, target_R=None, iters=50, damp=0.08, pos_tol=1e-3, rot_tol=2e-2,
           check_collision=True, orient_weight=0.35):
        """阻尼最小二乘，关节限位，必要时把目标抬高以躲开桌子。"""
        if target_R is None:
            target_R = R_DOWN
        target = np.asarray(target_pos, dtype=float).copy()
        lifted = False
        solution = None
        saved_q = self.data.qpos.copy()
        saved_v = self.data.qvel.copy()
        try:
            for _attempt in range(5):
                solution = self._ik_once(
                    q_seed, target, target_R, iters, damp, pos_tol, rot_tol, orient_weight
                )
                if not check_collision:
                    break
                self._write_arm(solution.q)
                require_mujoco().mj_forward(self.model, self.data)
                if not self.collided():
                    break
                target = target + np.array([0.0, 0.0, 0.02])
                lifted = True
                q_seed = solution.q
            collided = False
            if check_collision:
                self._write_arm(solution.q)
                require_mujoco().mj_forward(self.model, self.data)
                collided = self.collided()
        finally:
            self.data.qpos[:] = saved_q
            self.data.qvel[:] = saved_v
            require_mujoco().mj_forward(self.model, self.data)
        return IKSolution(
            q=solution.q,
            pos_err=solution.pos_err,
            rot_err=solution.rot_err,
            collided=bool(collided),
            lifted=bool(lifted),
        )

    def _ik_once(self, q_seed, target, target_R, iters, damp, pos_tol, rot_tol, orient_weight):
        mj = require_mujoco()
        saved_q = self.data.qpos.copy()
        saved_v = self.data.qvel.copy()
        q = np.asarray(q_seed, dtype=float).copy()
        pos_err = 0.0
        rot_err = 0.0
        try:
            for _ in range(int(iters)):
                self._write_arm(q)
                mj.mj_forward(self.model, self.data)
                position = self.data.site_xpos[self.site_id]
                rotation = self.data.site_xmat[self.site_id].reshape(3, 3)
                err_p = target - position
                err_r = orientation_error(rotation, target_R)
                pos_err = float(np.linalg.norm(err_p))
                rot_err = float(np.linalg.norm(err_r))
                if pos_err < pos_tol and rot_err < rot_tol:
                    break
                self._jacp.fill(0.0)
                self._jacr.fill(0.0)
                mj.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.site_id)
                jacobian = np.vstack([self._jacp[:, self.dofadr], self._jacr[:, self.dofadr]])
                error = np.concatenate([err_p, orient_weight * err_r])
                jacobian[3:] *= orient_weight
                q, _delta = damped_least_squares(
                    q, error, jacobian, self.lower + 1e-4, self.upper - 1e-4, damp=damp, step=0.6
                )
        finally:
            self.data.qpos[:] = saved_q
            self.data.qvel[:] = saved_v
            mj.mj_forward(self.model, self.data)
        return IKSolution(q=q, pos_err=pos_err, rot_err=rot_err, collided=False, lifted=False)

    def _write_arm(self, q, grip=None):
        for adr, value in zip(self.qadr, q):
            self.data.qpos[adr] = value
        if grip is not None:
            for adr in self.grip_adr:
                self.data.qpos[adr] = grip

    def reset(self, cube_xy, settle_s=0.25, rest_z=None):
        """方块放在桌面上，手臂停在复位姿态，夹爪张开。

        ``rest_z`` 是物体中心的初始高度。不传时用方块那一档。
        场景若设了 ``self.rest_z``（不同包围盒），复位会用那个高度。
        """
        mj = require_mujoco()
        mj.mj_resetData(self.model, self.data)
        self._write_arm(self.home_q, GRIP_OPEN)
        self.data.ctrl[:6] = self.home_q
        self.data.ctrl[6:] = GRIP_OPEN
        if rest_z is None:
            rest_z = getattr(self, "rest_z", None)
        z = (Z_REST + 0.004) if rest_z is None else float(rest_z)
        self.data.qpos[self.cube_adr:self.cube_adr + 3] = [
            float(cube_xy[0]),
            float(cube_xy[1]),
            z,
        ]
        self.data.qpos[self.cube_adr + 3] = 1.0
        mj.mj_forward(self.model, self.data)
        steps = int(round(float(settle_s) / self.dt))
        for _ in range(steps):
            mj.mj_step(self.model, self.data)
        return self

    def track_ee(self, target_pos, grip, duration, target_R=None):
        """求一次 IK，然后用位置伺服在 ``duration`` 秒内跟上。"""
        mj = require_mujoco()
        solution = self.ik(self.arm_q(), target_pos, target_R if target_R is not None else R_DOWN)
        self.data.ctrl[:6] = solution.q
        self.data.ctrl[6:] = float(grip)
        steps = max(1, int(round(float(duration) / self.dt)))
        for _ in range(steps):
            mj.mj_step(self.model, self.data)
        return solution

    def play_joints(self, times, q_traj, grip_traj):
        """按时间插值关节目标并开环回放。``times`` 从 0 开始，单位秒。"""
        mj = require_mujoco()
        times = np.asarray(times, dtype=float)
        q_traj = np.asarray(q_traj, dtype=float)
        grip_traj = np.asarray(grip_traj, dtype=float)
        if times[0] > 1e-8:
            raise ValueError("回放时间要从 0 开始")
        t = 0.0
        end = float(times[-1])
        while t < end - 1e-12:
            q_cmd, g_cmd = _interp(times, q_traj, grip_traj, t)
            self.data.ctrl[:6] = q_cmd
            self.data.ctrl[6:] = g_cmd
            mj.mj_step(self.model, self.data)
            t += self.dt
        return self


def _interp(times, q_traj, grip_traj, t):
    index = int(np.searchsorted(times, t, side="right") - 1)
    index = max(0, min(index, len(times) - 2))
    span = float(times[index + 1] - times[index])
    alpha = 0.0 if span <= 1e-8 else (t - float(times[index])) / span
    alpha = min(max(alpha, 0.0), 1.0)
    q_cmd = (1.0 - alpha) * q_traj[index] + alpha * q_traj[index + 1]
    g_cmd = (1.0 - alpha) * grip_traj[index] + alpha * grip_traj[index + 1]
    return q_cmd, float(g_cmd)
