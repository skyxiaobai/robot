# -*- coding: utf-8 -*-
"""人手重定向和桌面抓放仿真。没有 mujoco 时跳过会步进物理的测试。"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from egodata.schema import validate_episode  # noqa: E402
from retarget.aperture import apertures_from_joints, grip_command, grasp_closed_mask  # noqa: E402
from retarget.calibrate import default_calibration, fit_from_correspondences  # noqa: E402
from retarget.frames import (  # noqa: E402
    GRIP_CLOSE,
    GRIP_OPEN,
    HOME_EE,
    R_DOWN,
    identity_wrist_quat,
    matrix_from_quat_xyzw,
    sample_task,
    site_rotation,
)
from retarget.human_demo import hand_joints, make_basic_pick_place  # noqa: E402
from retarget.ik import damped_least_squares, orientation_error  # noqa: E402
from retarget.paths import path_duration, query_path, sample_path, scripted_segments  # noqa: E402
from retarget.retarget import infer_scene_xy, retarget_episode  # noqa: E402
from sim.bc import BCPolicy, decode_action, encode_waypoint, featurize  # noqa: E402

try:
    import mujoco  # noqa: F401
    from sim.arm import PickPlaceEnv  # noqa: E402
    from sim.rollout import replay_joints, run_oracle  # noqa: E402

    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False


class ApertureTest(unittest.TestCase):
    def test_thumb_index_distance(self):
        joints = np.zeros((21, 3))
        joints[4] = (0.04, 0.0, 0.0)
        joints[8] = (-0.04, 0.0, 0.0)
        self.assertAlmostEqual(apertures_from_joints(joints), 0.08)

    def test_grip_open_and_close(self):
        self.assertAlmostEqual(grip_command(0.10), GRIP_OPEN)
        self.assertAlmostEqual(grip_command(0.01), GRIP_CLOSE)
        mid = grip_command(0.0525)
        self.assertGreater(mid, GRIP_CLOSE)
        self.assertLess(mid, GRIP_OPEN)

    def test_grasp_hysteresis(self):
        aperture = np.array([0.09, 0.05, 0.03, 0.05, 0.08])
        closed = grasp_closed_mask(aperture)
        self.assertFalse(closed[0])
        self.assertFalse(closed[1])
        self.assertTrue(closed[2])
        self.assertTrue(closed[3])
        self.assertFalse(closed[4])


class CalibrationTest(unittest.TestCase):
    def test_correspondences_recover_scale_and_offset(self):
        calib = default_calibration()
        human = np.array(
            [
                [-0.2, -0.3, 0.75],
                [0.4, -0.3, 0.75],
                [-0.2, 0.3, 1.10],
                [0.4, 0.3, 1.10],
                [0.1, 0.0, 0.9],
            ]
        )
        robot = calib.map_points(human)
        fitted = fit_from_correspondences(human, robot)
        np.testing.assert_allclose(fitted.scale, calib.scale, atol=1e-8)
        np.testing.assert_allclose(fitted.offset, calib.offset, atol=1e-8)
        np.testing.assert_allclose(fitted.map_points(human), robot, atol=1e-8)

    def test_roundtrip_changes_numbers(self):
        calib = default_calibration()
        human = np.array([0.0, 0.0, 0.9])
        robot = calib.map_points(human)
        self.assertGreater(float(np.linalg.norm(robot - human)), 0.1)
        np.testing.assert_allclose(calib.unmap_points(robot), human, atol=1e-8)


class OrientationTest(unittest.TestCase):
    def test_identity_wrist_points_gripper_down(self):
        rotation = site_rotation(identity_wrist_quat())
        np.testing.assert_allclose(rotation[:, 2], [0.0, 0.0, -1.0], atol=1e-6)

    def test_yaw_keeps_approach_down(self):
        yaw = np.pi / 2
        wrist = np.array(
            [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]],
            dtype=float,
        )
        from egodata.schema import rotmat_to_quat_xyzw

        rotation = site_rotation(rotmat_to_quat_xyzw(wrist))
        np.testing.assert_allclose(rotation[:, 2], [0.0, 0.0, -1.0], atol=1e-6)
        self.assertAlmostEqual(float(rotation[0, 0]), 0.0, places=5)
        self.assertAlmostEqual(float(rotation[1, 0]), 1.0, places=5)

    def test_orientation_error_zero_at_target(self):
        err = orientation_error(R_DOWN, R_DOWN)
        np.testing.assert_allclose(err, 0.0, atol=1e-8)
        quat = identity_wrist_quat()
        np.testing.assert_allclose(matrix_from_quat_xyzw(quat), np.eye(3), atol=1e-6)


class IKLimitTest(unittest.TestCase):
    def test_step_is_clipped_to_joint_limit(self):
        q, _delta = damped_least_squares(
            np.array([0.0]),
            np.array([5.0]),
            np.array([[1.0]]),
            np.array([-0.2]),
            np.array([0.2]),
            damp=1e-8,
            step=1.0,
        )
        self.assertAlmostEqual(float(q[0]), 0.2, places=5)


class HumanDemoTest(unittest.TestCase):
    def test_episode_schema_and_aperture(self):
        calib = default_calibration()
        cube, goal = sample_task(np.random.default_rng(1))
        episode = make_basic_pick_place(cube, goal, calib, episode_id="synthetic/basic_pick_place/1")
        self.assertEqual(validate_episode(episode), [])
        self.assertEqual(episode["annotation"]["task"]["name"], "basic_pick_place")
        joints = np.asarray(episode["hands"]["right"]["joints"])
        aperture = apertures_from_joints(joints)
        self.assertLess(float(aperture.min()), 0.03)
        self.assertGreater(float(aperture.max()), 0.08)
        closed = grasp_closed_mask(aperture)
        self.assertTrue(bool(closed.any()))
        self.assertTrue(bool((~closed).any()))

    def test_hand_joints_match_requested_aperture(self):
        joints = hand_joints(np.zeros(3), 0.06, np.eye(3))
        self.assertEqual(joints.shape, (21, 3))
        self.assertAlmostEqual(apertures_from_joints(joints), 0.06, places=6)


class PathTest(unittest.TestCase):
    def test_path_starts_home_and_ends_open(self):
        segments = scripted_segments([0.45, 0.0], [0.55, 0.05], arc_m=0.03)
        start, grip0 = query_path(segments, 0.0)
        end, grip1 = query_path(segments, path_duration(segments))
        np.testing.assert_allclose(start, HOME_EE, atol=1e-8)
        self.assertAlmostEqual(grip0, GRIP_OPEN)
        self.assertAlmostEqual(grip1, GRIP_OPEN)
        self.assertGreater(end[2], 0.3)
        times, xyz, grip, duration = sample_path([0.45, 0.0], [0.55, 0.05], dt=0.1, arc_m=0.0)
        self.assertAlmostEqual(float(times[0]), 0.0)
        self.assertAlmostEqual(float(times[-1]), duration)
        self.assertEqual(len(times), len(xyz))
        self.assertEqual(len(times), len(grip))


class BCTest(unittest.TestCase):
    def test_waypoint_roundtrip_on_the_line_and_beside_it(self):
        cube = np.array([0.42, -0.05])
        goal = np.array([0.55, 0.08])
        mid = np.array([0.48, 0.01, 0.36])
        coded = encode_waypoint(mid, 0.024, cube, goal)
        restored, grip = decode_action(coded, cube, goal)
        np.testing.assert_allclose(restored[:2], mid[:2], atol=1e-6)
        self.assertAlmostEqual(restored[2], 0.36)
        self.assertAlmostEqual(grip, 0.024)
        bowed = mid.copy()
        bowed[:2] += np.array([0.0, 0.02])
        coded = encode_waypoint(bowed, 0.0, cube, goal)
        restored, grip = decode_action(coded, cube, goal)
        np.testing.assert_allclose(restored[:2], bowed[:2], atol=1e-6)
        self.assertAlmostEqual(grip, 0.0)
    def test_loss_drops_on_a_smooth_map(self):
        rng = np.random.default_rng(0)
        ee = rng.normal(size=(400, 3))
        grip = rng.uniform(0, 0.02, size=400)
        cube = rng.normal(size=(400, 3))
        goal = rng.normal(size=(400, 3))
        features = np.vstack([featurize(ee[i], grip[i], cube[i], goal[i]) for i in range(400)])
        actions = np.column_stack([cube[:, 0], cube[:, 1], np.full(400, 0.3), grip])
        policy = BCPolicy(hidden=64, seed=0, lr=1e-2, epochs=40, l2=0.0)
        before = float(np.mean((actions - actions.mean(0)) ** 2))
        stats = policy.fit(features, actions)
        self.assertLess(stats.train_mse_pose, before * 0.2)


@unittest.skipUnless(HAS_MUJOCO, "需要 pip install mujoco")
class SimTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = PickPlaceEnv()

    def test_ik_reaches_hover_inside_limits(self):
        target = np.array([0.48, 0.02, 0.36])
        solved = self.env.ik(self.env.home_q, target, R_DOWN)
        self.assertLess(solved.pos_err, 5e-3)
        self.assertFalse(solved.collided)
        self.assertTrue(np.all(solved.q >= self.env.lower - 1e-6))
        self.assertTrue(np.all(solved.q <= self.env.upper + 1e-6))
        # IK 不能把正在仿真的关节角改掉。
        self.env.reset([0.46, 0.0])
        before = self.env.arm_q().copy()
        self.env.ik(before, target, R_DOWN)
        np.testing.assert_allclose(self.env.arm_q(), before, atol=1e-8)

    def test_below_table_is_collision(self):
        hover = self.env.ik(self.env.home_q, np.array([0.48, 0.0, 0.36]), R_DOWN)
        buried = self.env.ik(self.env.home_q, np.array([0.48, 0.0, 0.05]), R_DOWN)
        self.assertFalse(hover.collided)
        self.assertTrue(buried.collided or buried.lifted)

    def test_scripted_oracle_places_block(self):
        result = run_oracle(self.env, np.array([0.46, 0.04]), np.array([0.54, -0.06]))
        self.assertTrue(result["success"], msg="xy_error=%s cube=%s" % (result["xy_error"], result["cube"]))
        self.assertLess(result["xy_error"], 0.04)

    def test_retarget_replay_one_episode(self):
        calib = default_calibration()
        cube = np.array([0.45, -0.02])
        goal = np.array([0.55, 0.08])
        episode = make_basic_pick_place(cube, goal, calib, arc_m=0.03, episode_id="synthetic/basic_pick_place/replay")
        retargeted = retarget_episode(episode, self.env, calib)
        self.assertLess(retargeted.median_pos_err, 0.01)
        inferred = infer_scene_xy(retargeted.ee_target, retargeted.grip)
        self.assertIsNotNone(inferred)
        self.assertLess(float(np.linalg.norm(inferred[0] - cube)), 0.02)
        self.assertLess(float(np.linalg.norm(inferred[1] - goal)), 0.02)
        played = replay_joints(self.env, cube, goal, retargeted.times, retargeted.q, retargeted.grip)
        self.assertTrue(played["success"], msg="xy_error=%s cube=%s" % (played["xy_error"], played["cube"]))


if __name__ == "__main__":
    unittest.main()
