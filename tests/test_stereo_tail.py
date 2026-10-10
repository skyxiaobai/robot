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
    StereoParams, adaptive_rts_q, rigid_fit_hand, rts_smooth, stereo_hand, velocity_gate)
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
        keep = velocity_gate(f, pts, 0.02, mode="median")
        self.assertFalse(keep[10])
        self.assertEqual(int(keep.sum()), 19)
        self.assertTrue(velocity_gate(f, pts, 0.0).all())
        predicted = velocity_gate(f, pts, 0.02)
        self.assertFalse(predicted[10])
        self.assertTrue(predicted[0] and predicted[1] and predicted[-1])

    def _fast_quadratic(self, fps, duration=1.0):
        # 加速度 4 m/s²，末速约 4 m/s，低于 8 m/s 上限。匀加速，预测残差应接近 0。
        count = int(round(duration * fps))
        times = np.arange(count, dtype=float) / float(fps)
        x = 0.5 * 4.0 * times ** 2
        points = np.stack([x, np.zeros(count), np.zeros(count)], 1)
        return times, points

    def test_predicted_gate_is_fps_invariant(self):
        kept_at = {}
        for fps in (30.0, 60.0, 120.0):
            times, points = self._fast_quadratic(fps)
            keep = velocity_gate(np.arange(len(times)), points, 0.02, timestamps=times)
            self.assertTrue(keep.all(), "fps=%s 匀加速不该被剔，剔了 %d 帧" % (fps, int((~keep).sum())))
            spike = points.copy()
            index = int(round(0.5 * fps))
            spike[index, 2] += 0.30
            dropped = ~velocity_gate(np.arange(len(times)), spike, 0.02, timestamps=times)
            self.assertTrue(dropped[index], fps)
            far = np.abs(times - 0.5) > 0.15
            self.assertFalse(dropped[far].any(), "fps=%s 离跳点 0.15 s 以外不该被剔" % fps)
            kept_at[fps] = times[dropped]
        for fps, stamps in kept_at.items():
            self.assertTrue(np.all(np.abs(stamps - 0.5) <= 0.15), (fps, stamps))

    def test_median_gate_changes_with_fps(self):
        """同一条加速轨迹，旧的 2 cm/帧 门限在 30 fps 比 120 fps 剔得更多。"""
        fractions = {}
        for fps in (30.0, 120.0):
            times, points = self._fast_quadratic(fps, duration=0.9)
            # 去掉序列两头，避免窗口不对称单独造成的边缘效应；仍从 0.05 s 之后取样。
            use = (times >= 0.05) & (times <= 0.85)
            frames = np.arange(int(use.sum()))
            keep = velocity_gate(frames, points[use], 0.02, timestamps=times[use], mode="median")
            fractions[fps] = float((~keep).mean())
        self.assertGreater(fractions[30.0], fractions[120.0])

    def test_adaptive_rts_q_follows_speed_not_fps(self):
        def sample(speed, fps):
            times = np.arange(0.0, 2.0, 1.0 / fps)
            points = np.stack([speed * times, np.zeros_like(times), np.zeros_like(times)], 1)
            return adaptive_rts_q(times, points)
        self.assertEqual(sample(0.05, 30), 30.0)
        self.assertEqual(sample(2.0, 30), 300.0)
        self.assertEqual(sample(0.05, 30), sample(0.05, 60))
        self.assertEqual(sample(2.0, 30), sample(2.0, 120))
        mid30, mid60 = sample(0.4, 30), sample(0.4, 60)
        self.assertGreater(mid30, 30.0)
        self.assertLess(mid30, 300.0)
        self.assertAlmostEqual(mid30, mid60, places=6)

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
