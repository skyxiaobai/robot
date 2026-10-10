# -*- coding: utf-8 -*-
"""双目一条龙管线：三角化、一致性检查、双目 QC、HOT3D 适配器的几何，以及用假后端跑完整流程。"""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from egodata.qc import write_yield_reports, yield_report  # noqa: E402
from egodata.schema import validate_episode  # noqa: E402
from egodata.stereo_qc import qc_stereo_episode, stereo_frame_flags  # noqa: E402
from headcam.hand_pose import load_tum_trajectory, project_pinhole, write_calibration_yaml  # noqa: E402
from headcam.hot3d_adapter import relative_extrinsics, write_tum  # noqa: E402
from headcam.stereo_pipeline import (  # noqa: E402
    StereoParams,
    build_stereo_episode,
    load_session,
    run_pipeline,
    stereo_hand,
)

W, H, F, BASE = 640, 480, 400.0, 0.08


def make_calib():
    K = np.array([[F, 0, W / 2.0], [0, F, H / 2.0], [0, 0, 1.0]])
    return {"image_width": W, "image_height": H, "K_left": K, "K_right": K.copy(),
            "dist_left": np.zeros(5), "dist_right": np.zeros(5), "R": np.eye(3), "T": np.array([-BASE, 0, 0])}


def hand_joints(center=(0.0, 0.05, 0.5)):
    """一只张开的手：手腕在 center，中指 MCP 离手腕 9 cm。"""
    rng = np.random.default_rng(0)
    j = np.zeros((21, 3))
    for finger in range(5):
        for k in range(4):
            j[1 + finger * 4 + k] = [(finger - 2) * 0.02, -0.03 - 0.02 * k - (0.04 if k else 0), 0.0]
    j[9] = [0.0, -0.09, 0.0]
    j += rng.normal(0, 0.002, j.shape)
    j[0] = 0
    return j + np.asarray(center)


def views_of(joints, calib, right_shift=(0.0, 0.0)):
    left = project_pinhole(joints, calib["K_left"])
    right = project_pinhole(joints, calib["K_right"], calib["R"], calib["T"]) + np.asarray(right_shift)
    mono = joints * 1.3  # 单目尺度错了 30%
    return ({"keypoints_2d": left.tolist(), "confidence": [0.9] * 21, "joints_cam": mono.tolist()},
            {"keypoints_2d": right.tolist(), "confidence": [0.8] * 21})


class StereoHandTest(unittest.TestCase):
    def test_triangulates_all_joints_in_metres(self):
        calib = make_calib()
        j = hand_joints()
        lv, rv = views_of(j, calib)
        out = stereo_hand(lv, rv, calib, StereoParams())
        self.assertEqual(out["status"], "ok")
        self.assertLess(np.abs(out["joints_cam"] - j).max(), 1e-6)
        self.assertLess(out["reproj_px"], 1e-6)
        self.assertAlmostEqual(out["palm_m"], float(np.linalg.norm(j[9] - j[0])), places=6)
        np.testing.assert_allclose(out["confidence"], 0.8)

    def test_disagreeing_views_rejected(self):
        calib = make_calib()
        lv, rv = views_of(hand_joints(), calib, right_shift=(0.0, 40.0))  # 右目上下错 40 像素：认错了
        out = stereo_hand(lv, rv, calib, StereoParams())
        self.assertIn(out["status"], ("reproj", "few_joints"))
        self.assertIsNone(out["joints_cam"])
        self.assertIsNotNone(out["raw_joints"])
        # 关掉检查时同一只手会被保留（对比用）
        loose = stereo_hand(lv, rv, calib, StereoParams(consistency=False))
        self.assertEqual(loose["status"], "ok")

    def test_one_view_depth_and_palm(self):
        calib = make_calib()
        lv, rv = views_of(hand_joints(), calib)
        self.assertEqual(stereo_hand(lv, None, calib, StereoParams())["status"], "one_view")
        self.assertEqual(stereo_hand(None, None, calib, StereoParams())["status"], "none")
        far = views_of(hand_joints((0, 0, 2.0)), calib)
        self.assertEqual(stereo_hand(far[0], far[1], calib, StereoParams())["status"], "depth")
        big = views_of((hand_joints() - [0, 0.05, 0.5]) * 2.5 + [0, 0.05, 0.5], calib)
        self.assertEqual(stereo_hand(big[0], big[1], calib, StereoParams())["status"], "palm")


def write_session(root, n, calib, poses=None, fps=30.0):
    root = Path(root)
    (root / "stereo").mkdir(parents=True, exist_ok=True)
    write_calibration_yaml(root / "calib.yaml", calib)
    stamps = [100.0 + i / fps for i in range(n)]
    (root / "timestamps.csv").write_text("frame_index,timestamp_s\n" + "".join(
        "%d,%.6f\n" % (i, t) for i, t in enumerate(stamps)), encoding="utf-8")
    poses = poses or [np.eye(4) for _ in range(n)]
    write_tum(root / "slam.tum", stamps, poses)
    (root / "metadata.json").write_text(json.dumps({"episode_id": "test/%s" % root.name, "fps": fps,
                                                    "image_width": W, "image_height": H}), encoding="utf-8")
    return stamps


def moving_views(n, calib, drop=(), bad=()):
    views = {"left": [], "right": []}
    truth = []
    for i in range(n):
        j = hand_joints((0.02 * np.sin(i / 5.0), 0.05, 0.5))
        truth.append(j)
        lv, rv = views_of(j, calib, right_shift=(0, 40.0) if i in bad else (0, 0))
        views["left"].append({"left": lv, "right": None})
        views["right"].append({"left": None if i in drop else rv, "right": None})
    return views, truth


class EpisodeTest(unittest.TestCase):
    def test_episode_world_frame_and_flags(self):
        calib = make_calib()
        n = 30
        with tempfile.TemporaryDirectory() as tmp:
            pose = np.eye(4)
            pose[:3, 3] = [1.0, 2.0, 3.0]
            write_session(Path(tmp) / "s", n, calib, poses=[pose] * n)
            for name in ("left.mp4", "right.mp4"):
                (Path(tmp) / "s" / "stereo" / name).write_bytes(b"")
            info = load_session(Path(tmp) / "s")
            views, truth = moving_views(n, calib, drop=(10, 11), bad=(20,))
            ep, extra = build_stereo_episode(info, views, StereoParams(smooth="none", gap_fill=True, max_gap=5))
            self.assertEqual(validate_episode(ep), [])
            self.assertEqual(ep["coordinate_frame"], "slam_world")
            st = ep["stereo"]["per_frame"]["left"]
            self.assertEqual(st[10], "one_view")
            self.assertIn(st[20], ("reproj", "few_joints"))
            self.assertEqual(st[0], "ok")
            # 世界系 = 相机系 + 平移
            np.testing.assert_allclose(ep["hands"]["left"]["joints"][0][0], truth[0][0] + [1, 2, 3], atol=1e-6)
            self.assertTrue(ep["hands"]["left"]["filled"][10])
            self.assertTrue(ep["hands"]["left"]["filled"][20])
            self.assertEqual(len(ep["hands"]["left"]["wrist_pose"][0]), 7)
            flags = stereo_frame_flags(ep)
            self.assertTrue(flags["stereo_one_view"][10])
            self.assertTrue(flags["stereo_inconsistent"][20])
            self.assertTrue(flags["stereo_filled"][11])
            self.assertFalse(flags["stereo_one_view"][0])
            qc = qc_stereo_episode(ep)
            self.assertEqual(qc["flags"]["stereo_one_view"], 2)
            self.assertEqual(qc["flags"]["stereo_inconsistent"], 1)

    def test_yield_csv_has_stereo_columns(self):
        results = [{"episode_id": "a", "num_frames": 10, "duration_s": 1.0, "accepted": False, "bad_fraction": 0.5,
                    "flags": {"hands_out_of_frame": 0, "view_drift": 0, "blur": 0, "staged_static": 0,
                              "stereo_one_view": 5}, "reasons": ["stereo_one_view"]}]
        with tempfile.TemporaryDirectory() as tmp:
            html_path, csv_path = write_yield_reports(yield_report(results), Path(tmp) / "y.html", Path(tmp) / "y.csv")
            rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
            self.assertEqual(rows[0]["stereo_one_view"], "5")
            self.assertIn("只有一目认到手", Path(html_path).read_text(encoding="utf-8"))


class AdapterGeometryTest(unittest.TestCase):
    def test_relative_extrinsics_matches_opencv_convention(self):
        T_wl = np.eye(4)
        T_wl[:3, 3] = [0.5, 1.0, 0.0]
        T_wr = T_wl.copy()
        T_wr[:3, 3] += [0.064, 0, 0]  # 右目在左目 +x 方向 6.4 cm
        R, T = relative_extrinsics(T_wl, T_wr)
        p_left = np.array([0.1, 0.0, 0.5])
        p_world = T_wl[:3, :3] @ p_left + T_wl[:3, 3]
        p_right = np.linalg.inv(T_wr) @ np.append(p_world, 1)
        np.testing.assert_allclose(R @ p_left + T, p_right[:3], atol=1e-12)
        np.testing.assert_allclose(T, [-0.064, 0, 0], atol=1e-12)

    def test_tum_roundtrip(self):
        a, b = 0.7, -0.4
        rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
        pose = np.eye(4)
        pose[:3, :3] = rz @ rx
        pose[:3, 3] = [1, 2, 3]
        with tempfile.TemporaryDirectory() as tmp:
            write_tum(Path(tmp) / "t.tum", [1.0, 2.0], [pose, pose])
            stamps, poses = load_tum_trajectory(Path(tmp) / "t.tum")
            np.testing.assert_allclose(stamps, [1.0, 2.0])
            np.testing.assert_allclose(poses[0], pose, atol=1e-6)


class FakeBackend(object):
    """按调用顺序回放：先左目 n 帧，再右目 n 帧。"""

    def __init__(self, views, n):
        self.views, self.n, self.calls = views, n, 0

    def predict(self, image, calib=None):
        view = "left" if self.calls < self.n else "right"
        frame = self.views[view][self.calls % self.n]
        self.calls += 1
        return {"left": frame["left"], "right": frame["right"]}


class RunPipelineTest(unittest.TestCase):
    def test_one_command_without_models(self):
        import cv2
        calib = make_calib()
        n = 20
        with tempfile.TemporaryDirectory() as tmp:
            s = Path(tmp) / "sess"
            write_session(s, n, calib)
            for name in ("left.mp4", "right.mp4"):
                vw = cv2.VideoWriter(str(s / "stereo" / name), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
                for _ in range(n):
                    vw.write(np.zeros((H, W, 3), np.uint8))
                vw.release()
            views, _ = moving_views(n, calib)
            gt = {"frames": [{"left": (np.asarray(j)).tolist(), "right": None} for j in moving_views(n, calib)[1]]}
            (s / "gt").mkdir()
            (s / "gt" / "hands_gt.json").write_text(json.dumps(gt), encoding="utf-8")
            summary = run_pipeline([s], Path(tmp) / "out", StereoParams(smooth="none"), backend_name="fake",
                                   backend=FakeBackend(views, n), export=False, log=lambda *_: None)
            self.assertEqual(summary["raw_frames"], n)
            ev = summary["eval_all"]
            self.assertEqual(ev["after_check"]["n"], n)
            self.assertLess(ev["after_check"]["median_cm"], 0.01)
            self.assertTrue((Path(tmp) / "out" / "qc" / "yield.csv").is_file())
            self.assertTrue((Path(tmp) / "out" / "cache" / "sess.fake.json").is_file())


if __name__ == "__main__":
    unittest.main()


class ValidateSessionTest(unittest.TestCase):
    def _session(self, tmp, n=10):
        import cv2
        calib = make_calib()
        s = Path(tmp) / "sess"
        stamps = write_session(s, n, calib)
        for name in ("left.mp4", "right.mp4"):
            vw = cv2.VideoWriter(str(s / "stereo" / name), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
            for _ in range(n):
                vw.write(np.zeros((H, W, 3), np.uint8))
            vw.release()
        (s / "stereo" / "timestamps_lr.csv").write_text("frame_index,left_s,right_s\n" + "".join(
            "%d,%.9f,%.9f\n" % (i, t, t + 0.0002) for i, t in enumerate(stamps)), encoding="utf-8")
        return s

    def test_good_and_faulty_sessions(self):
        from simulate_device_session import inject_fault
        from validate_session import validate
        for fault, needle in ((None, None), ("drop_frame", "timestamps.csv"), ("desync_5ms", "左右曝光"),
                              ("baseline_mm", "基线"), ("no_calib", "calib.yaml")):
            with tempfile.TemporaryDirectory() as tmp:
                s = self._session(tmp)
                if fault:
                    inject_fault(s, fault)
                r = validate(s)
                if fault is None:
                    self.assertTrue(r["ok"], r["errors"])
                    self.assertAlmostEqual(r["facts"]["baseline_m"], BASE)
                    self.assertLess(r["facts"]["sync_ms"]["max"], 1.0)
                else:
                    self.assertFalse(r["ok"], fault)
                    self.assertTrue(any(needle in e for e in r["errors"]), (fault, r["errors"]))
