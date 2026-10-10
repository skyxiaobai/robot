# -*- coding: utf-8 -*-
"""HOT3D 重定向的纯函数测试。不下载数据，也不看仿真成功率。"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from retarget.frames import Z_GRASP  # noqa: E402
from retarget.grasp_timing import (  # noqa: E402
    canonical_joints,
    coupling_mask,
    grip_from_closed,
    grip_from_states,
)
from retarget.hot3d_clip import FPS, LIFT_M, find_pick_place  # noqa: E402
from retarget.object_centric import align_grasp_height, object_centric_ee, robot_relative  # noqa: E402
from sim.scene import plan_geometry, scene_xml  # noqa: E402

try:
    import mujoco  # noqa: F401
    from sim.arm import PickPlaceEnv  # noqa: E402

    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False


def _lift_and_place(travel=0.15, lift=0.12):
    count = 90
    obj = np.zeros((count, 3))
    obj[:, 1] = 0.70
    obj[20:45, 1] = 0.70 + np.linspace(0.0, lift, 25)
    obj[45:70, 1] = 0.70 + np.linspace(lift, 0.01, 25)
    obj[20:70, 0] = np.linspace(0.0, travel, 50)
    obj[70:, 0] = travel
    obj[70:, 1] = 0.70 + 0.03
    hand = obj + np.array([0.04, 0.05, 0.0])
    other = obj + np.array([0.90, 0.0, 0.0])
    return obj, hand, other


class SegmentTest(unittest.TestCase):
    def test_lift_then_place_is_kept(self):
        obj, hand, other = _lift_and_place()
        picked = find_pick_place(obj, other, hand, fps=FPS)
        self.assertIsNotNone(picked)
        self.assertEqual(picked["side"], "right")
        self.assertGreaterEqual(picked["lift_m"], LIFT_M)
        self.assertGreaterEqual(picked["travel_xz_m"], 0.08)

    def test_static_object_is_rejected(self):
        obj = np.zeros((60, 3))
        obj[:, 1] = 0.7
        hand = obj + np.array([0.05, 0.0, 0.0])
        self.assertIsNone(find_pick_place(obj, hand, hand + 1.0))

    def test_lift_without_place_is_rejected(self):
        obj, hand, other = _lift_and_place()
        obj[45:, 1] = obj[44, 1]
        self.assertIsNone(find_pick_place(obj, hand, other))

    def test_far_hand_is_rejected(self):
        obj, _hand, _other = _lift_and_place()
        far = obj + np.array([0.8, 0.0, 0.0])
        self.assertIsNone(find_pick_place(obj, far, far))


class GeometryTest(unittest.TestCase):
    def test_spoon_is_not_pinchable_after_uniform_scale(self):
        info = {
            "min_x": 0.0, "min_y": 0.0, "min_z": 0.0,
            "size_x": 0.068, "size_y": 0.018, "size_z": 0.309,
        }
        plan = plan_geometry(info)
        self.assertFalse(plan["pinchable"])
        self.assertLessEqual(plan["scale"] * 0.309, 0.080 + 1e-9)
        self.assertLessEqual(plan["scale"], 1.0)
        self.assertEqual(plan["kind"], "box")

    def test_bowl_is_a_pinchable_cylinder(self):
        info = {
            "min_x": -0.14, "min_y": 0.0, "min_z": -0.14,
            "size_x": 0.28, "size_y": 0.13, "size_z": 0.28,
            "symmetries_continuous": [{"axis": [0, -1, 0]}],
        }
        plan = plan_geometry(info)
        self.assertTrue(plan["pinchable"])
        self.assertEqual(plan["kind"], "cylinder")
        self.assertGreaterEqual(2 * plan["half_y"], 0.026)
        self.assertLessEqual(2 * plan["half_y"], 0.070)

    def test_xml_keeps_the_free_joint(self):
        info = {
            "min_x": -0.018, "min_y": -0.018, "min_z": -0.018,
            "size_x": 0.036, "size_y": 0.036, "size_z": 0.036,
        }
        text = scene_xml(plan_geometry(info))
        self.assertIn('joint name="cube_free"', text)
        self.assertIn('name="cube"', text)


class TimingAndRelativeTest(unittest.TestCase):
    def test_coupling_requires_shared_motion(self):
        count = 40
        obj = np.zeros((count, 3))
        obj[:, 0] = np.linspace(0.0, 0.20, count)
        wrist = obj + np.array([0.05, 0.0, 0.0])
        other = obj + np.array([0.80, 0.0, 0.0])
        moving = coupling_mask(wrist, obj, other, fps=30.0)
        self.assertGreater(int(np.sum(moving)), 10)
        still = obj.copy()
        still[:, 0] = 0.0
        self.assertEqual(int(np.sum(coupling_mask(wrist, still, other, fps=30.0))), 0)

    def test_grip_closes_only_on_grasp_and_leads_by_three(self):
        states = ["open", "pre_grasp", "grasp", "grasp", "release"]
        grip = grip_from_states(states, lead_frames=3)
        self.assertEqual(grip[0], 0.0)
        self.assertEqual(grip[2], 0.0)
        self.assertGreater(grip[4], 0.0)
        closed = np.array([False, False, False, False, True, True])
        led = grip_from_closed(closed, lead_frames=3)
        self.assertAlmostEqual(float(led[0]), 0.024)
        self.assertEqual(int(np.sum(led[1:] == 0.0)), 5)

    def test_template_has_21_joints(self):
        self.assertEqual(canonical_joints("right").shape, (21, 3))
        self.assertLess(canonical_joints("left")[4, 1], 0.0)

    def test_object_relative_offset_uses_geometry_scale(self):
        wrist = np.array([[0.0, 1.0, 0.0], [0.2, 1.1, 0.0]])
        obj = np.array([[0.0, 0.9, 0.0], [0.2, 0.9, 0.0]])
        # 机器人轴：(人手 z, 人手 x, 人手 y)。相对 (0, 0, 0.1) 乘 0.5。
        relative = robot_relative(wrist, obj, 0.5)
        self.assertAlmostEqual(relative[0, 2], 0.05)
        obj_robot = np.array([[0.48, 0.0, 0.25], [0.55, 0.05, 0.34]])
        ee = object_centric_ee(obj_robot, relative, grasp_index=0, release_index=1, grasp_z=Z_GRASP)
        self.assertAlmostEqual(ee[0, 2], Z_GRASP, places=5)

    def test_egodex_height_aligns_grasp_frame(self):
        wrist = np.array([[0.48, 0.0, 0.30], [0.50, 0.02, 0.36]])
        ee = align_grasp_height(wrist, 0, Z_GRASP)
        self.assertAlmostEqual(ee[0, 2], Z_GRASP, places=5)
        self.assertAlmostEqual(ee[1, 0], 0.50, places=5)


@unittest.skipUnless(HAS_MUJOCO, "需要 mujoco")
class SceneLoadTest(unittest.TestCase):
    def test_generated_cube_scene_resets(self):
        info = {
            "min_x": -0.018, "min_y": -0.018, "min_z": -0.018,
            "size_x": 0.036, "size_y": 0.036, "size_z": 0.036,
        }
        plan = plan_geometry(info)
        directory = ROOT / "tests" / "_tmp_scene"
        directory.mkdir(exist_ok=True)
        path = directory / "scene.xml"
        path.write_text(scene_xml(plan), encoding="utf-8")
        try:
            env = PickPlaceEnv(path)
            env.rest_z = plan["rest_z"]
            env.reset([0.48, 0.0])
            self.assertTrue(np.isfinite(env.cube_position()).all())
        finally:
            path.unlink(missing_ok=True)
            directory.rmdir()


if __name__ == "__main__":
    unittest.main()
