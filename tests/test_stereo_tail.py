# -*- coding: utf-8 -*-
"""双目尾部误差改进：稳健刚体对齐手腕、速度门限、RTS 平滑、严格门限。"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from headcam.stereo_pipeline import (  # noqa: E402
    StereoParams, rigid_fit_hand, rts_smooth, stereo_hand, velocity_gate)
from test_stereo_pipeline import hand_joints, make_calib, views_of  # noqa: E402


def _rot(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


class RigidFitTest(unittest.TestCase):

    def test_recovers_similarity_and_ignores_outlier_joint(self):
        tri = hand_joints()
        mono = (tri - tri.mean(0)) @ _rot(20).T * 0.7 + 0.3
        bad = tri.copy()
        bad[0] += [0.0, 0.0, 0.06]  # 三角化的手腕深度错了 6 cm
        fit, res = rigid_fit_hand(mono, bad, np.ones(21))
        self.assertLess(np.linalg.norm(fit[0] - tri[0]), 0.01)
        self.assertLess(res, 0.005)

    def test_too_few_joints(self):
        tri = np.full((21, 3), np.nan)
        tri[:4] = hand_joints()[:4]
        self.assertEqual(rigid_fit_hand(hand_joints(), tri, np.ones(21)), (None, None))


class TemporalTest(unittest.TestCase):

    def test_velocity_gate_drops_jump(self):
        f = np.arange(20)
        pts = np.stack([f * 0.002, np.zeros(20), np.zeros(20)], 1)
        pts[10] += [0, 0, 0.3]  # 换手 / 认错导致的 30 cm 跳点
        keep = velocity_gate(f, pts, 0.02)
        self.assertFalse(keep[10])
        self.assertEqual(int(keep.sum()), 19)
        self.assertTrue(velocity_gate(f, pts, 0.0).all())

    def test_rts_reduces_noise_without_lag(self):
        rng = np.random.default_rng(1)
        t = np.arange(90) / 30.0
        truth = np.stack([0.2 * t, 0.05 * np.sin(2 * t), np.zeros_like(t)], 1)
        noisy = truth + rng.normal(0, 0.005, truth.shape)
        out = rts_smooth(t, noisy)
        self.assertLess(np.abs(out - truth).mean(), 0.6 * np.abs(noisy - truth).mean())


class StereoHandModesTest(unittest.TestCase):

    def test_rigid_fit_matches_exact_triangulation(self):
        calib = make_calib()
        j = hand_joints()
        left, right = views_of(j, calib)
        for mode in ("tri", "rigid_fit"):
            out = stereo_hand(left, right, calib, StereoParams(wrist_mode=mode))
            self.assertEqual(out["status"], "ok")
            self.assertLess(np.linalg.norm(out["joints_cam"][0] - j[0]), 0.005, mode)

    def test_strict_offaxis_gate(self):
        calib = make_calib()
        left, right = views_of(hand_joints(center=(0.25, 0.0, 0.3)), calib)
        out = stereo_hand(left, right, calib, StereoParams(max_offaxis_deg=30))
        self.assertEqual(out["status"], "strict")
        self.assertIsNone(out["joints_cam"])


if __name__ == "__main__":
    unittest.main()
