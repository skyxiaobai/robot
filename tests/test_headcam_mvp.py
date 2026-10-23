# -*- coding: utf-8 -*-
"""头戴 MVP 前半段：位姿不编造、同步误差、会话布局、导出入口。"""
import json
import os
import stat
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from headcam.export_mvp import export_mvp, prepare_hand_session  # noqa: E402
from headcam.hand_pose import load_calibration, load_tum_trajectory, write_calibration_yaml  # noqa: E402
from headcam.record_session import capture_session, import_session, init_session, layout_report  # noqa: E402
from headcam.slam_vio import (  # noqa: E402
    STATUS_OK,
    STATUS_POSE_MISSING,
    STATUS_TRACKING_LOST,
    run_slam,
    write_slam_tum,
)
from headcam.sync_calib import (  # noqa: E402
    calibration_from_document,
    read_calib_json,
    refresh_calib_json,
    sync_error_ms,
)


def _clear_orb_env():
    for name in ("ORB_SLAM3_BIN", "ORB_SLAM3_VOCAB", "ORB_SLAM3_SETTINGS", "ORB_SLAM3_TUM", "ORB_SLAM3_RUN"):
        os.environ.pop(name, None)


def _calib():
    return {
        "image_width": 320,
        "image_height": 240,
        "K_left": np.array([[200.0, 0.0, 160.0], [0.0, 210.0, 120.0], [0.0, 0.0, 1.0]]),
        "K_right": np.array([[201.0, 0.0, 159.0], [0.0, 211.0, 121.0], [0.0, 0.0, 1.0]]),
        "dist_left": np.array([0.01, -0.02, 0.0, 0.0, 0.0]),
        "dist_right": np.zeros(5),
        "R": np.eye(3),
        "T": np.array([-0.08, 0.0, 0.001]),
    }


def _write_session(root, stamps, tum_lines=None, lr_right_offset=0.0004):
    root = Path(root)
    (root / "stereo").mkdir(parents=True)
    (root / "metadata.json").write_text(json.dumps({
        "episode_id": "headcam/mvp-test",
        "fps": 30.0,
        "exposure_locked": True,
        "auto_exposure": False,
    }), encoding="utf-8")
    lines = ["frame_index,timestamp_s"]
    lr = ["frame_index,left_s,right_s"]
    for index, stamp in enumerate(stamps):
        lines.append("%d,%.9f" % (index, stamp))
        lr.append("%d,%.9f,%.9f" % (index, stamp, stamp + lr_right_offset))
    (root / "timestamps.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "stereo" / "timestamps_lr.csv").write_text("\n".join(lr) + "\n", encoding="utf-8")
    write_calibration_yaml(root / "calib.yaml", _calib())
    if tum_lines is not None:
        (root / "slam.tum").write_text("\n".join(tum_lines) + "\n", encoding="utf-8")
    return root


class SlamVioTest(unittest.TestCase):
    def setUp(self):
        _clear_orb_env()

    def tearDown(self):
        _clear_orb_env()

    def test_missing_trajectory_is_pose_missing_and_not_identity(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sess"
            out = Path(tmp) / "out"
            stamps = [i / 30.0 for i in range(4)]
            _write_session(root, stamps, tum_lines=None)
            result = run_slam(root, backend="auto", out_dir=out)
            self.assertFalse(result["world_pose"])
            self.assertEqual(result["coordinate_frame"], "camera")
            self.assertIsNone(result["accuracy"])
            self.assertEqual(result["pose_missing_frames"], 4)
            self.assertTrue(all(frame["status"] == STATUS_POSE_MISSING for frame in result["frames"]))
            self.assertTrue(all(frame["T_world_cam"] is None for frame in result["frames"]))
            self.assertFalse((out / "slam.tum").exists())
            status = json.loads((out / "slam_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["frames"][0]["status"], "pose_missing")
            self.assertIsNone(status["frames"][0]["T_world_cam"])

    def test_backend_none_ignores_existing_tum(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sess"
            _write_session(root, [0.0], tum_lines=["0 0.2 0 0 0 0 0 1"])
            result = run_slam(root, backend="none", out_dir=root)
            self.assertEqual(result["frames"][0]["status"], STATUS_POSE_MISSING)
            self.assertIsNone(result["frames"][0]["T_world_cam"])
            self.assertFalse(result["world_pose"])
            self.assertTrue((root / "slam.tum").is_file())
            self.assertIn("0.2", (root / "slam.tum").read_text(encoding="utf-8"))

    def test_gap_is_tracking_lost_and_partial_tum_is_not_written(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sess"
            out = Path(tmp) / "out"
            stamps = [0.0, 1.0 / 30.0, 2.0 / 30.0]
            _write_session(root, stamps, tum_lines=["0.000000000 0.250000000 0 0 0 0 0 1"])
            result = run_slam(root, backend="external_tum", out_dir=out)
            self.assertEqual(result["frames"][0]["status"], STATUS_OK)
            self.assertAlmostEqual(result["frames"][0]["T_world_cam"][0][3], 0.25)
            self.assertEqual(result["frames"][1]["status"], STATUS_TRACKING_LOST)
            self.assertIsNone(result["frames"][1]["T_world_cam"])
            self.assertEqual(result["pose_coverage"], "partial")
            self.assertFalse(result["world_pose"])
            self.assertEqual(result["coordinate_frame"], "camera")
            self.assertFalse((out / "slam.tum").exists())
            self.assertIsNone(result["accuracy"])

    def test_complete_tum_roundtrip_keeps_translation(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sess"
            out = Path(tmp) / "out"
            stamps = [i / 30.0 for i in range(3)]
            lines = ["%.9f 0.2 0.0 1.0 0 0 0 1" % stamp for stamp in stamps]
            _write_session(root, stamps, tum_lines=lines)
            result = run_slam(root, out_dir=out)
            self.assertTrue(result["world_pose"])
            self.assertEqual(result["coordinate_frame"], "slam_world")
            written = out / "slam.tum"
            self.assertTrue(written.is_file())
            times, poses = load_tum_trajectory(written)
            self.assertEqual(len(poses), 3)
            self.assertAlmostEqual(float(poses[0][0, 3]), 0.2, places=6)
            self.assertAlmostEqual(float(poses[0][2, 3]), 1.0, places=6)
            again = write_slam_tum(Path(tmp) / "again.tum", result)
            self.assertIsNotNone(again)

    def test_recorded_identity_pose_is_kept(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sess"
            _write_session(root, [0.0], tum_lines=["0 0 0 0 0 0 0 1"])
            result = run_slam(root, backend="external_tum")
            self.assertEqual(result["frames"][0]["status"], STATUS_OK)
            self.assertTrue(np.allclose(result["frames"][0]["T_world_cam"], np.eye(4)))
            self.assertTrue(result["world_pose"])

    def test_empty_tum_is_pose_missing(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sess"
            _write_session(root, [0.0, 0.1], tum_lines=["# nothing"])
            result = run_slam(root, backend="external_tum")
            self.assertTrue(all(frame["status"] == STATUS_POSE_MISSING for frame in result["frames"]))
            self.assertIn("空的", result["notes"])

    def test_orbslam3_not_installed_does_not_invent_or_relabel(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sess"
            out = Path(tmp) / "out"
            _write_session(root, [0.0], tum_lines=["0 0.3 0 0 0 0 0 1"])
            result = run_slam(root, backend="orbslam3", out_dir=out, run_orbslam3=True)
            self.assertFalse(result["world_pose"])
            self.assertEqual(result["frames"][0]["status"], STATUS_POSE_MISSING)
            self.assertIn("ORB_SLAM3_BIN", result["orbslam3"]["missing"])
            self.assertFalse(result["orbslam3"]["ran"])
            self.assertFalse((out / "slam.tum").exists())
            self.assertTrue(any("精度" in step or "误差" in step for step in result["orbslam3"]["euroc_steps"]))

    def test_fake_orbslam3_success_and_failure(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            root = tmp / "sess"
            _write_session(root, [0.0])
            vocab, settings = tmp / "voc.txt", tmp / "settings.yaml"
            vocab.write_text("vocab\n", encoding="utf-8")
            settings.write_text("settings\n", encoding="utf-8")
            ok_bin = tmp / "ok_orb.py"
            ok_bin.write_text(
                "#!/usr/bin/env python3\nfrom pathlib import Path\n"
                "Path('CameraTrajectory.txt').write_text('0.000000000 0.100000000 0 0 0 0 0 1\\n')\n",
                encoding="utf-8",
            )
            ok_bin.chmod(ok_bin.stat().st_mode | stat.S_IEXEC)
            os.environ["ORB_SLAM3_BIN"] = str(ok_bin)
            os.environ["ORB_SLAM3_VOCAB"] = str(vocab)
            os.environ["ORB_SLAM3_SETTINGS"] = str(settings)
            out = tmp / "out_ok"
            result = run_slam(root, backend="orbslam3", out_dir=out, run_orbslam3=True)
            self.assertTrue(result["orbslam3"]["ran"])
            self.assertTrue(result["world_pose"])
            self.assertAlmostEqual(result["frames"][0]["T_world_cam"][0][3], 0.1)
            self.assertIsNone(result["accuracy"])

            bad_bin = tmp / "bad_orb.py"
            bad_bin.write_text("#!/usr/bin/env python3\nimport sys\nsys.stderr.write('boom\\n')\nsys.exit(1)\n", encoding="utf-8")
            bad_bin.chmod(bad_bin.stat().st_mode | stat.S_IEXEC)
            os.environ["ORB_SLAM3_BIN"] = str(bad_bin)
            failed = run_slam(root, backend="orbslam3", out_dir=tmp / "out_bad", run_orbslam3=True)
            self.assertTrue(failed["orbslam3"]["ran"])
            self.assertEqual(failed["frames"][0]["status"], STATUS_POSE_MISSING)
            self.assertFalse((tmp / "out_bad" / "slam.tum").exists())
            self.assertIn("boom", failed["orbslam3"]["stderr"])


class SyncCalibTest(unittest.TestCase):
    def test_roundtrip_matches_yaml_loader(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_calibration_yaml(root / "calib.yaml", _calib())
            loaded = load_calibration(root / "calib.yaml")
            document = refresh_calib_json(root, imu_cam_time_offset_s=0.003)
            again = calibration_from_document(read_calib_json(root / "calib.json"))
            np.testing.assert_allclose(again["K_left"], loaded["K_left"])
            np.testing.assert_allclose(again["K_right"], loaded["K_right"])
            np.testing.assert_allclose(again["T"], loaded["T"])
            np.testing.assert_allclose(again["R"], loaded["R"])
            self.assertAlmostEqual(document["baseline_m"], float(np.linalg.norm(loaded["T"])))
            self.assertAlmostEqual(document["sync_error_ms"]["imu_cam"], 3.0)
            self.assertIsNone(document["sync_error_ms"]["left_right_max"])

    def test_sync_error_from_timestamps_and_missing_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sess"
            _write_session(root, [0.0, 1.0 / 30.0], lr_right_offset=0.0004)
            # 第二帧再偏大一点，确认取的是最大差而不是 0。
            lr = (root / "stereo" / "timestamps_lr.csv").read_text(encoding="utf-8").splitlines()
            index, left, _right = lr[2].split(",")
            lr[2] = "%s,%s,%.9f" % (index, left, float(left) + 0.002)
            (root / "stereo" / "timestamps_lr.csv").write_text("\n".join(lr) + "\n", encoding="utf-8")
            measured = sync_error_ms(root)
            self.assertAlmostEqual(measured["left_right_max"], 2.0, places=6)
            self.assertAlmostEqual(measured["left_right_median"], 1.2, places=6)
            self.assertIsNone(measured["imu_cam"])
            (root / "stereo" / "timestamps_lr.csv").unlink()
            missing = sync_error_ms(root)
            self.assertIsNone(missing["left_right_max"])
            self.assertIsNone(missing["source"])

    def test_nanosecond_pair_converts_to_milliseconds_and_mismatch_is_null(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sess"
            root.mkdir()
            (root / "stereo").mkdir()
            # 400000 ns = 0.4 ms。步长 3.3e7，判定为纳秒，而不是把差值乘 1000。
            (root / "stereo" / "timestamps_lr.csv").write_text(
                "frame_index,left_s,right_s\n"
                "0,0,400000\n"
                "1,33000000,33400000\n"
                "2,66000000,66400000\n",
                encoding="utf-8",
            )
            measured = sync_error_ms(root)
            self.assertEqual(measured["time_unit"], "nanoseconds")
            self.assertAlmostEqual(measured["left_right_max"], 0.4, places=6)
            self.assertAlmostEqual(measured["left_right_median"], 0.4, places=6)
            (root / "stereo" / "timestamps_lr.csv").write_text(
                "frame_index,left_s,right_s\n"
                "0,0.000,0\n"
                "1,0.033,33000000\n"
                "2,0.066,66000000\n",
                encoding="utf-8",
            )
            mixed = sync_error_ms(root)
            self.assertIsNone(mixed["left_right_max"])
            self.assertIsNone(mixed["left_right_median"])
            self.assertIn("不换算", mixed["time_unit_note"])

    def test_retarget_axis_json_is_rejected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calib.json"
            path.write_text(json.dumps({"scale": [1, 1, 1], "offset": [0, 0, 0]}), encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                read_calib_json(path)
            self.assertIn("AxisCalibration", str(caught.exception))


class RecordSessionTest(unittest.TestCase):
    def test_init_has_layout_and_no_pose_or_fake_timestamps(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sess"
            report = init_session(root, fps=60, episode_id="headcam/empty")
            self.assertFalse(report["videos_present"])
            self.assertFalse(report["timestamps_present"])
            self.assertFalse(report["imu_present"])
            self.assertTrue((root / "imu.csv").is_file())
            self.assertFalse((root / "slam.tum").exists())
            self.assertFalse((root / "timestamps.csv").exists())
            meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
            self.assertFalse(meta["hardware_capture"])
            self.assertTrue(meta["exposure_locked"])
            self.assertFalse(meta["auto_exposure"])

    def test_import_copies_files_without_inventing_timestamps(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            left, right = tmp / "left.mp4", tmp / "right.mp4"
            left.write_bytes(b"left")
            right.write_bytes(b"right")
            root = tmp / "sess"
            report = import_session(root, left, right, fps=30)
            self.assertTrue(report["videos_present"])
            self.assertFalse(report["timestamps_present"])
            self.assertFalse(report["ready_for_export_hands"])
            stamps = tmp / "timestamps.csv"
            stamps.write_text("frame_index,timestamp_s\n0,0\n1,0.033333333\n", encoding="utf-8")
            calib = tmp / "calib.yaml"
            write_calibration_yaml(calib, _calib())
            lr = tmp / "timestamps_lr.csv"
            lr.write_text("frame_index,left_s,right_s\n0,0,0.0002\n1,0.033333333,0.033533333\n", encoding="utf-8")
            ready = import_session(root, left, right, timestamps=stamps, timestamps_lr=lr, calib=calib)
            self.assertTrue(ready["ready_for_export_hands"])
            document = json.loads((root / "calib.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(document["sync_error_ms"]["left_right_max"], 0.2, places=5)
            meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["timestamps_source"], "imported")
            self.assertFalse(meta["timestamps_are_hardware"])

    def test_capture_dry_run_does_not_open_devices(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sess"
            plan = capture_session(root, left_device="/dev/does-not-exist-left", right_device="/dev/does-not-exist-right", dry_run=True)
            self.assertFalse(plan["recorded"])
            self.assertIn("ffmpeg", plan["commands"]["left"][0])
            self.assertIn("v4l2", plan["commands"]["left"])
            self.assertFalse((root / "stereo" / "left.mp4").exists())
            self.assertFalse((root / "slam.tum").exists())
            self.assertIsNone(plan["commands"]["imu"])
            fresh = layout_report(root)
            self.assertFalse(fresh["slam_tum_present"])


class ExportMvpTest(unittest.TestCase):
    def setUp(self):
        _clear_orb_env()

    def test_skip_hands_reports_pose_missing_and_sync(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            session = _write_session(tmp / "sess", [i / 30.0 for i in range(3)])
            report = export_mvp(session, tmp / "out", run_hands=False, export=False, log=None)
            self.assertFalse(report["world_pose"])
            self.assertEqual(report["coordinate_frame"], "camera")
            self.assertIsNone(report["accuracy"])
            self.assertEqual(report["slam"]["pose_missing_frames"], 3)
            self.assertIsNone(report["slam"]["accuracy"])
            self.assertAlmostEqual(report["sync_error_ms"]["left_right_max"], 0.4, places=5)
            self.assertEqual(report["calibration"], "stereo")
            self.assertFalse((tmp / "out" / "slam.tum").exists())
            self.assertFalse((tmp / "out" / "hand_session" / "slam.tum").exists())
            self.assertEqual(report["hands"]["status"], "skipped")
            self.assertIn("未请求导出", report["lerobot"]["result"])

    def test_partial_track_is_not_passed_to_hand_pipeline(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            stamps = [i / 30.0 for i in range(3)]
            session = _write_session(tmp / "sess", stamps, tum_lines=["0 0.4 0 0 0 0 0 1"])
            seen = {}

            def runner(sessions, out_dir, **kwargs):
                seen["session"] = Path(sessions[0])
                seen["export"] = kwargs.get("export")
                return {"accepted_episodes": 0, "episodes": 0, "yield": 0.0, "raw_frames": 0, "usable_frames": 0}

            report = export_mvp(session, tmp / "out", run_hands=True, hand_pipeline=runner, export=False, log=None)
            self.assertEqual(seen, {})
            self.assertEqual(report["hands"]["status"], "refused")
            self.assertEqual(report["world_hands"], "refused")
            self.assertFalse(report["world_pose"])
            self.assertEqual(report["slam"]["tracking_lost_frames"], 2)
            self.assertEqual(report["slam"]["ok_frames"], 1)
            self.assertIn("未跑手部", report["lerobot"]["result"])

            def runner_partial(sessions, out_dir, **kwargs):
                seen["session"] = Path(sessions[0])
                seen["export"] = kwargs.get("export")
                return {"accepted_episodes": 0, "episodes": 0, "yield": 0.0, "raw_frames": 0, "usable_frames": 0}

            partial = export_mvp(
                session, tmp / "out_partial", run_hands=True, hand_pipeline=runner_partial,
                allow_partial=True, export=False, log=None,
            )
            self.assertFalse((seen["session"] / "slam.tum").exists())
            self.assertFalse(seen["export"])
            self.assertEqual(partial["hands"]["status"], "ran")
            self.assertEqual(partial["world_hands"], "refused")
            self.assertIn("allow-partial", partial["hands"]["warning"])
            status = json.loads((tmp / "out" / "slam_status.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(status["frames"][0]["T_world_cam"][0][3], 0.4)
            self.assertIsNone(status["frames"][1]["T_world_cam"])

    def test_complete_pose_is_visible_to_hand_pipeline(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            stamps = [i / 30.0 for i in range(2)]
            lines = ["%.9f 0.05 0 0 0 0 0 1" % stamp for stamp in stamps]
            session = _write_session(tmp / "sess", stamps, tum_lines=lines)
            seen = {}

            def runner(sessions, out_dir, **kwargs):
                seen["tum"] = (Path(sessions[0]) / "slam.tum").read_text(encoding="utf-8")
                return {"accepted_episodes": 0, "episodes": 0, "yield": 0.0, "raw_frames": 0, "usable_frames": 0}

            report = export_mvp(session, tmp / "out", run_hands=True, hand_pipeline=runner, log=None)
            self.assertTrue(report["world_pose"])
            self.assertIn("0.05", seen["tum"])
            self.assertEqual(report["coordinate_frame"], "slam_world")

    def test_existing_episodes_go_through_stereo_qc(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            session = _write_session(tmp / "sess", [i / 30.0 for i in range(20)])
            episodes = tmp / "episodes"
            episodes.mkdir()
            (episodes / "clip.json").write_text(json.dumps(_moving_episode()), encoding="utf-8")
            report = export_mvp(session, tmp / "out", episodes=episodes, export=False, log=None)
            self.assertEqual(report["hands"]["status"], "external_episodes")
            self.assertEqual(report["qc"]["accepted_episodes"], 1)
            self.assertEqual(report["qc"]["source"], "egodata.stereo_qc.qc_stereo_episode")
            csv_text = (tmp / "out" / "qc" / "yield.csv").read_text(encoding="utf-8")
            self.assertIn("yes", csv_text)
            episode = json.loads((episodes / "clip.json").read_text(encoding="utf-8"))
            self.assertFalse(episode["world_pose"])
            self.assertEqual(episode["hand_pose"]["slam"]["pose_status"], ["pose_missing"] * 20)
            self.assertEqual(episode["hand_pose"]["slam"]["camera_pose_meaning"], "identity_placeholder_camera_frame")
            self.assertIsNone(episode["hand_pose"]["slam"]["accuracy"])


def _moving_episode():
    """与 tests/test_stereo_qc.py 相同的最小可质检片段。"""
    n = 20
    pose = np.eye(4).tolist()
    hands = {}
    for side in ("left", "right"):
        joints, confidence, wrist_pose = [], [], []
        for index in range(n):
            point = [0.002 * index, 0.0, 1.0]
            joints.append([point] + [[point[0], 0.02, 1.0]] * 20)
            confidence.append(0.99)
            wrist_pose.append(point + [0.0, 0.0, 0.0, 1.0])
        hands[side] = {
            "joints": joints,
            "confidence": confidence,
            "wrist_pose": wrist_pose,
            "filled": [False] * n,
        }
    return {
        "episode_id": "clip",
        "num_frames": n,
        "fps": 30.0,
        "image_width": 640,
        "image_height": 480,
        "camera_intrinsic": [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
        "camera_poses": [pose for _ in range(n)],
        "hands": hands,
        "stereo": {"per_frame": {"left": ["ok"] * n, "right": ["ok"] * n}},
    }


class FixtureGapTumTest(unittest.TestCase):
    """读仓库里的短 TUM，不在测试里现写轨迹。"""

    def setUp(self):
        _clear_orb_env()
        self.session = ROOT / "tests" / "fixtures" / "headcam" / "gap_tum"
        self.tum = self.session / "CameraTrajectory.txt"

    def test_gap_file_labels_tracking_lost_and_missing_file_is_pose_missing(self):
        gapped = run_slam(self.session, backend="external_tum", tum_path=self.tum)
        self.assertEqual(
            [frame["status"] for frame in gapped["frames"]],
            [STATUS_OK, STATUS_TRACKING_LOST, STATUS_TRACKING_LOST, STATUS_OK],
        )
        self.assertIsNone(gapped["frames"][1]["T_world_cam"])
        self.assertAlmostEqual(gapped["frames"][0]["T_world_cam"][0][3], 0.1)
        self.assertAlmostEqual(gapped["frames"][3]["T_world_cam"][0][3], 0.2)
        self.assertFalse(gapped["world_pose"])
        self.assertIsNone(gapped["accuracy"])
        self.assertEqual(gapped["pose_missing_frames"], 0)
        missing = run_slam(self.session, backend="auto")
        self.assertEqual(missing["pose_missing_frames"], 4)
        self.assertEqual(missing["tracking_lost_frames"], 0)
        self.assertTrue(all(frame["T_world_cam"] is None for frame in missing["frames"]))

    def test_summarize_refuses_world_hands_unless_allow_partial(self):
        import io
        from contextlib import redirect_stdout

        from headcam.summarize_slam import main as summarize_main

        buf = io.StringIO()
        with redirect_stdout(buf):
            code = summarize_main(["--session", str(self.session), "--tum", str(self.tum)])
        text = buf.getvalue()
        self.assertEqual(code, 1)
        self.assertIn("frames=4\n", text)
        self.assertIn("ok=2\n", text)
        self.assertIn("tracking_lost=2\n", text)
        self.assertIn("pose_missing=0\n", text)
        self.assertIn("world_hands=refused\n", text)
        self.assertIn("accuracy=unmeasured\n", text)
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = summarize_main([
                "--session", str(self.session), "--tum", str(self.tum), "--allow-partial",
            ])
        self.assertEqual(code, 0)
        self.assertIn("world_hands=refused", buf.getvalue())
        self.assertIn("allow-partial", buf.getvalue())
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = summarize_main(["--session", str(self.session)])
        self.assertEqual(code, 1)
        self.assertIn("pose_missing=4\n", buf.getvalue())
        self.assertIn("world_hands=refused", buf.getvalue())


class OfflineSkipHandsFixtureTest(unittest.TestCase):
    """仓库里的合成会话：不跑 WiLoR，检查秒轨迹和纳秒轨迹。"""

    def setUp(self):
        _clear_orb_env()
        self.session = ROOT / "tests" / "fixtures" / "headcam" / "offline_skip_hands"
        self.seconds = self.session / "CameraTrajectory_seconds.txt"
        self.nanoseconds = self.session / "CameraTrajectory_ns.txt"

    def test_cli_skip_hands_seconds_tum_exports_without_wilor(self):
        import tempfile

        from headcam.export_mvp import main as export_main

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            code = export_main([
                "--session", str(self.session),
                "--out", str(out),
                "--slam-backend", "external_tum",
                "--tum", str(self.seconds),
                "--skip-hands",
            ])
            self.assertEqual(code, 0)
            report = json.loads((out / "report.json").read_text(encoding="utf-8"))
            self.assertTrue(report["world_pose"])
            self.assertEqual(report["hands"]["status"], "skipped")
            self.assertEqual(report["world_hands"], "allowed")
            self.assertIsNone(report["accuracy"])
            self.assertFalse(report["slam"]["time_unit_mismatch"])
            self.assertEqual(report["slam"]["frame_time_unit"], "seconds")
            self.assertEqual(report["slam"]["tum_time_unit"], "seconds")
            self.assertAlmostEqual(report["sync_error_ms"]["left_right_max"], 0.4, places=5)
            self.assertEqual(report["calibration"], "stereo")
            self.assertTrue((out / "slam.tum").is_file())
            self.assertIn("0.100000000", (out / "slam.tum").read_text(encoding="utf-8"))
            status = json.loads((out / "slam_status.json").read_text(encoding="utf-8"))
            self.assertTrue(all(frame["status"] == "ok" for frame in status["frames"]))
            self.assertAlmostEqual(status["frames"][3]["T_world_cam"][0][3], 0.13)

    def test_nanosecond_tum_is_mismatch_until_explicit_scale(self):
        import tempfile

        from headcam.export_mvp import main as export_main
        from headcam.summarize_slam import main as summarize_main

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            out = tmp / "ns"
            code = export_main([
                "--session", str(self.session),
                "--out", str(out),
                "--slam-backend", "external_tum",
                "--tum", str(self.nanoseconds),
                "--skip-hands",
                "--max-gap-s", "1e12",
            ])
            self.assertEqual(code, 0)
            report = json.loads((out / "report.json").read_text(encoding="utf-8"))
            self.assertFalse(report["world_pose"])
            self.assertTrue(report["slam"]["time_unit_mismatch"])
            self.assertEqual(report["slam"]["tum_time_unit"], "nanoseconds")
            self.assertEqual(report["slam"]["tracking_lost_frames"], 4)
            self.assertEqual(report["slam"]["ok_frames"], 0)
            self.assertEqual(report["hands"]["status"], "skipped")
            self.assertFalse((out / "slam.tum").exists())
            status = json.loads((out / "slam_status.json").read_text(encoding="utf-8"))
            self.assertTrue(all(frame["T_world_cam"] is None for frame in status["frames"]))
            self.assertNotIn("0.100000000", json.dumps(status["frames"]))

            scaled = tmp / "scaled"
            code = export_main([
                "--session", str(self.session),
                "--out", str(scaled),
                "--slam-backend", "external_tum",
                "--tum", str(self.nanoseconds),
                "--tum-time-scale", "1e-9",
                "--skip-hands",
            ])
            self.assertEqual(code, 0)
            scaled_report = json.loads((scaled / "report.json").read_text(encoding="utf-8"))
            self.assertTrue(scaled_report["world_pose"])
            self.assertFalse(scaled_report["slam"]["time_unit_mismatch"])
            self.assertAlmostEqual(scaled_report["slam"]["tum_time_scale"], 1e-9)
            self.assertEqual(scaled_report["slam"]["tum_time_unit"], "nanoseconds")
            self.assertAlmostEqual(
                json.loads((scaled / "slam_status.json").read_text(encoding="utf-8"))["frames"][0]["T_world_cam"][0][3],
                0.10,
            )

        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = summarize_main([
                "--session", str(self.session), "--tum", str(self.nanoseconds),
            ])
        text = buf.getvalue()
        self.assertEqual(code, 1)
        self.assertIn("time_unit_mismatch=true\n", text)
        self.assertIn("tum_time_unit=nanoseconds\n", text)
        self.assertIn("world_hands=refused\n", text)
        self.assertIn("未自动缩放", text)
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = summarize_main([
                "--session", str(self.session), "--tum", str(self.nanoseconds),
                "--tum-time-scale", "1e-9",
            ])
        self.assertEqual(code, 0)
        self.assertIn("time_unit_mismatch=false\n", buf.getvalue())
        self.assertIn("world_hands=allowed\n", buf.getvalue())
        self.assertIn("tum_time_scale=1e-09\n", buf.getvalue())

    def test_session_nanoseconds_are_not_silently_divided(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_session(Path(tmp) / "sess", [0.0, 33000000.0, 66000000.0])
            result = run_slam(root, backend="none")
            self.assertEqual(result["frame_time_unit"], "nanoseconds")
            self.assertEqual(result["frames"][1]["timestamp_s"], 33000000.0)
            self.assertIn("不把会话时间戳除以 1e9", result["notes"])
            self.assertFalse(result["time_unit_mismatch"])


def _camera_episode(num_frames, wrist_xy=None):
    """OpenCV 相机系片段。手腕在前方偏下，距离大约 0.5 m，给 sanity_check 用。"""
    hands = {}
    for side_index, side in enumerate(("left", "right")):
        joints, confidence, wrist_pose = [], [], []
        for index in range(num_frames):
            x = 0.0 if wrist_xy is None else wrist_xy[index][0]
            point = [x + 0.04 * side_index, 0.12, 0.50]
            joints.append([point] + [[point[0], 0.14, 0.50]] * 20)
            confidence.append(0.9)
            wrist_pose.append(point + [0.0, 0.0, 0.0, 1.0])
        hands[side] = {"joints": joints, "confidence": confidence, "wrist_pose": wrist_pose}
    return {
        "episode_id": "splice",
        "num_frames": num_frames,
        "fps": 30.0,
        "coordinate_frame": "camera",
        "camera_poses": [np.eye(4).tolist() for _ in range(num_frames)],
        "hands": hands,
        "stereo": {"per_frame": {"left": ["ok"] * num_frames, "right": ["ok"] * num_frames}},
    }


class SlamFrameSpliceTest(unittest.TestCase):
    """ok 帧才做 p_world = T @ p_cam。丢失帧保持缺测，并给成绩单看。"""

    def test_gap_fixture_keeps_lost_frames_missing(self):
        from common import frames as F
        from qc_scorecard import score_episode

        from headcam.slam_frames import apply_ok_frame_world

        session = ROOT / "tests" / "fixtures" / "headcam" / "gap_tum"
        slam = run_slam(session, backend="external_tum", tum_path=session / "CameraTrajectory.txt")
        episode = _camera_episode(4)
        original = episode["hands"]["left"]["wrist_pose"][0][:3]
        apply_ok_frame_world(episode, slam)
        block = episode["hand_pose"]["slam"]
        self.assertEqual(block["pose_status"], ["ok", "tracking_lost", "tracking_lost", "ok"])
        self.assertEqual(block["world_points_scope"], "ok_frames_only")
        self.assertEqual(block["camera_pose_meaning"], "ok_frames_only")
        self.assertIsNone(block["accuracy"])
        self.assertIsNone(block["depth_status"])
        self.assertEqual(block["track_status"]["left"], ["ok"] * 4)
        self.assertIsNone(block["T_world_cam"][1])
        self.assertIsNone(block["world_hands"]["left"][1])
        self.assertIsNone(block["world_hands"]["left"][2])
        self.assertEqual(episode["hands"]["left"]["wrist_pose"][0][:3], original)
        self.assertEqual(F.convention_of(episode), F.CAMERA)
        world0 = block["world_hands"]["left"][0]["wrist"]
        world3 = block["world_hands"]["left"][3]["wrist"]
        self.assertAlmostEqual(world0[0], original[0] + 0.1, places=6)
        self.assertAlmostEqual(world3[0], original[0] + 0.2, places=6)
        self.assertNotEqual(world0[0], world3[0])
        check = F.sanity_check(episode)
        self.assertTrue(check["ok"], check["flags"])
        card = score_episode(episode, "gap-splice")
        self.assertEqual(card["convention"], "camera")
        self.assertTrue(card["frame_ok"], card["frame_flags"])
        self.assertIsNone(card["depth_valid"])

    def test_complete_fixture_writes_world_and_passes_sanity(self):
        from common import frames as F
        from qc_scorecard import score_episode

        from headcam.slam_frames import apply_ok_frame_world

        session = ROOT / "tests" / "fixtures" / "headcam" / "offline_skip_hands"
        slam = run_slam(session, backend="external_tum", tum_path=session / "CameraTrajectory_seconds.txt")
        episode = _camera_episode(4, wrist_xy=[(0.0, 0.0), (0.01, 0.0), (0.02, 0.0), (0.03, 0.0)])
        local = [list(episode["hands"]["left"]["wrist_pose"][i][:3]) for i in range(4)]
        apply_ok_frame_world(episode, slam)
        self.assertEqual(F.convention_of(episode), F.WORLD)
        self.assertNotEqual(F.convention_of(episode), F.WORLD_YUP)
        self.assertFalse(episode["frame_convention"]["up_axis_verified"])
        self.assertEqual(episode["frame_convention"]["camera_axes"], "cv")
        self.assertEqual(episode["coordinate_frame"], "slam_world")
        shifts = (0.10, 0.11, 0.12, 0.13)
        for index, shift in enumerate(shifts):
            wrist = episode["hands"]["left"]["wrist_pose"][index]
            self.assertAlmostEqual(wrist[0], local[index][0] + shift, places=6)
            self.assertAlmostEqual(wrist[1], local[index][1], places=6)
            self.assertAlmostEqual(wrist[2], local[index][2], places=6)
            self.assertAlmostEqual(float(np.linalg.norm(wrist[3:7])), 1.0, places=6)
            self.assertAlmostEqual(episode["camera_poses"][index][0][3], shift, places=6)
        again = episode["hands"]["left"]["wrist_pose"][0][0]
        apply_ok_frame_world(episode, slam)
        self.assertAlmostEqual(episode["hands"]["left"]["wrist_pose"][0][0], again, places=6)
        check = F.sanity_check(episode)
        self.assertTrue(check["ok"], check["flags"])
        card = score_episode(episode, "complete-splice")
        self.assertEqual(card["convention"], "world")
        self.assertTrue(card["frame_ok"], card["frame_flags"])
        self.assertIsNone(card["depth_valid"])
        self.assertIsNone(episode["hand_pose"]["slam"]["accuracy"])

    def test_camera_yup_and_existing_world_are_not_multiplied(self):
        from headcam.slam_frames import apply_ok_frame_world

        session = ROOT / "tests" / "fixtures" / "headcam" / "offline_skip_hands"
        slam = run_slam(session, backend="external_tum", tum_path=session / "CameraTrajectory_seconds.txt")
        yup = _camera_episode(4)
        yup["coordinate_frame"] = "camera_yup"
        before = yup["hands"]["right"]["wrist_pose"][1][0]
        apply_ok_frame_world(yup, slam)
        self.assertEqual(yup["hands"]["right"]["wrist_pose"][1][0], before)
        self.assertEqual(yup["frame_convention"]["name"], "camera_yup")
        self.assertIn("不变换", yup["hand_pose"]["slam"]["world_points_note"])

        world = _camera_episode(4)
        world["coordinate_frame"] = "slam_world"
        world["hands"]["left"]["wrist_pose"][0][0] = 1.25
        apply_ok_frame_world(world, slam)
        self.assertEqual(world["hands"]["left"]["wrist_pose"][0][0], 1.25)
        self.assertEqual(world["hand_pose"]["slam"]["world_points_scope"], "already_world")
        self.assertEqual(world["frame_convention"]["name"], "world")


class WilorPartialFixtureTest(unittest.TestCase):
    """仓库里的相机系手腕。不是 WiLoR 输出。只在 ok 帧填世界点。"""

    def setUp(self):
        self.root = ROOT / "tests" / "fixtures" / "headcam" / "wilor_partial"

    def test_cli_fills_only_ok_frames_and_rerun_does_not_multiply(self):
        import io
        import tempfile
        from contextlib import redirect_stdout

        from headcam.slam_frames import main as check_main

        episode = json.loads((self.root / "episode.json").read_text(encoding="utf-8"))
        self.assertEqual(episode["coordinate_frame"], "camera")
        self.assertEqual(episode["hand_pose"]["backend"], "fixture_not_wilor")
        camera_wrist = episode["hands"]["left"]["wrist_pose"][0][0]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            first = tmp / "once.json"
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = check_main([
                    "--episode", str(self.root / "episode.json"),
                    "--slam-status", str(self.root / "slam_status.json"),
                    "--out", str(first),
                ])
            text = buf.getvalue()
            self.assertEqual(code, 0)
            self.assertIn("world_points_scope=ok_frames_only\n", text)
            self.assertIn("frame_convention=camera\n", text)
            self.assertIn("ok=2\n", text)
            self.assertIn("tracking_lost=2\n", text)
            self.assertIn("world_hands_filled=2\n", text)
            self.assertIn("world_hands_missing=2\n", text)
            self.assertIn("sanity_ok=true\n", text)
            self.assertIn("accuracy=unmeasured\n", text)
            self.assertNotIn("0.01", text)
            checked = json.loads(first.read_text(encoding="utf-8"))
            block = checked["hand_pose"]["slam"]
            self.assertIsNone(block["accuracy"])
            self.assertIsNone(block["world_hands"]["left"][1])
            self.assertIsNone(block["world_hands"]["right"][2])
            self.assertAlmostEqual(block["world_hands"]["left"][0]["wrist"][0], camera_wrist + 0.1, places=6)
            self.assertAlmostEqual(block["world_hands"]["left"][3]["wrist"][0], 0.03 + 0.2, places=6)
            self.assertEqual(checked["hands"]["left"]["wrist_pose"][0][0], camera_wrist)
            self.assertEqual(checked["coordinate_frame"], "camera")

            second = tmp / "twice.json"
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = check_main([
                    "--episode", str(first),
                    "--slam-status", str(self.root / "slam_status.json"),
                    "--out", str(second),
                ])
            self.assertEqual(code, 0)
            self.assertIn("sanity_ok=true\n", buf.getvalue())
            again = json.loads(second.read_text(encoding="utf-8"))
            self.assertEqual(again["hands"]["left"]["wrist_pose"][0][0], camera_wrist)
            self.assertAlmostEqual(again["hand_pose"]["slam"]["world_hands"]["left"][0]["wrist"][0], camera_wrist + 0.1, places=6)
            self.assertIsNone(again["hand_pose"]["slam"]["world_hands"]["left"][1])
            self.assertIsNone(again["hand_pose"]["slam"]["accuracy"])

    def test_claimed_accuracy_in_slam_status_is_not_copied(self):
        import io
        import tempfile
        from contextlib import redirect_stdout

        from headcam.slam_frames import main as check_main

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            slam = json.loads((self.root / "slam_status.json").read_text(encoding="utf-8"))
            slam["accuracy"] = 0.01
            status = tmp / "claimed.json"
            status.write_text(json.dumps(slam), encoding="utf-8")
            out = tmp / "episode.json"
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = check_main([
                    "--episode", str(self.root / "episode.json"),
                    "--slam-status", str(status),
                    "--out", str(out),
                ])
            self.assertEqual(code, 0)
            self.assertIn("accuracy=unmeasured\n", buf.getvalue())
            self.assertNotIn("0.01", buf.getvalue())
            saved = json.loads(out.read_text(encoding="utf-8"))
            self.assertIsNone(saved["hand_pose"]["slam"]["accuracy"])


class EurocOrbslam3ExcerptTest(unittest.TestCase):
    """ORB-SLAM3 stereo_euroc 在 EuRoC V1_01_easy 上写出的真实片段。

    只提交了 40 行。这是无人机双目轨迹的解析回归，不是头戴精度。
    """

    def setUp(self):
        _clear_orb_env()
        self.tum = ROOT / "tests" / "fixtures" / "headcam" / "euroc_orbslam3_v101" / "CameraTrajectory_excerpt.txt"

    def _stamps(self):
        stamps = []
        for line in self.tum.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            stamps.append(int(line.split()[0].split(".")[0]) / 1e9)
        return stamps

    def test_nanosecond_euroc_file_associates_only_with_explicit_scale(self):
        import tempfile

        from headcam.summarize_slam import main as summarize_main

        times, poses = load_tum_trajectory(self.tum)
        self.assertEqual(len(poses), 40)
        self.assertGreater(float(np.median(np.diff(times))), 1e5)
        stamps = self._stamps()
        self.assertEqual(len(stamps), 40)
        with tempfile.TemporaryDirectory() as tmp:
            session = _write_session(Path(tmp) / "sess", stamps)
            raw = run_slam(session, backend="external_tum", tum_path=self.tum)
            self.assertEqual(raw["tum_time_unit"], "nanoseconds")
            self.assertTrue(raw["time_unit_mismatch"])
            self.assertEqual(raw["ok_frames"], 0)
            self.assertEqual(raw["tracking_lost_frames"], 40)
            self.assertTrue(all(frame["T_world_cam"] is None for frame in raw["frames"]))

            scaled = run_slam(session, backend="external_tum", tum_path=self.tum, tum_time_scale=1e-9)
            self.assertFalse(scaled["time_unit_mismatch"])
            self.assertEqual(scaled["ok_frames"], 40)
            self.assertEqual(scaled["tracking_lost_frames"], 0)
            self.assertEqual(scaled["pose_missing_frames"], 0)
            self.assertTrue(scaled["world_pose"])
            self.assertIsNone(scaled["accuracy"])
            last = scaled["frames"][-1]["T_world_cam"]
            self.assertAlmostEqual(last[0][3], 1.723316908, places=6)
            self.assertAlmostEqual(last[1][3], -0.190014720, places=6)
            self.assertAlmostEqual(last[2][3], -0.356315374, places=6)
            # qx qy qz qw = 0.003649197 0.778750420 0.298360109 0.551829457
            # 若把第一列四元数当成 qw，R00 会落到约 -0.213。
            self.assertAlmostEqual(last[0][0], -0.390941890, places=6)
            self.assertLess(scaled["frames"][-1]["gap_s"], 1e-5)

        import io
        from contextlib import redirect_stdout
        with tempfile.TemporaryDirectory() as tmp:
            session = _write_session(Path(tmp) / "sess", stamps)
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = summarize_main([
                    "--session", str(session), "--tum", str(self.tum), "--tum-time-scale", "1e-9",
                ])
            text = buf.getvalue()
            self.assertEqual(code, 0)
            self.assertIn("frames=40\n", text)
            self.assertIn("ok=40\n", text)
            self.assertIn("tracking_lost=0\n", text)
            self.assertIn("pose_missing=0\n", text)
            self.assertIn("accuracy=unmeasured\n", text)
            self.assertIn("tum_time_unit=nanoseconds\n", text)
            self.assertIn("time_unit_mismatch=false\n", text)


class TimestampAlignmentTest(unittest.TestCase):
    """帧数相同不够。两边时钟对不上时拒绝按序号配对。"""

    def test_nanosecond_episode_is_not_paired_with_second_slam(self):
        from headcam.slam_frames import apply_ok_frame_world, main as check_main

        slam = json.loads(
            (ROOT / "tests" / "fixtures" / "headcam" / "wilor_partial" / "slam_status.json").read_text(encoding="utf-8")
        )
        episode = json.loads(
            (ROOT / "tests" / "fixtures" / "headcam" / "wilor_partial" / "episode.json").read_text(encoding="utf-8")
        )
        wrist = episode["hands"]["left"]["wrist_pose"][0][0]
        episode["timestamps"] = [0.0, 33000000.0, 66000000.0, 100000000.0]
        apply_ok_frame_world(episode, slam)
        block = episode["hand_pose"]["slam"]
        self.assertEqual(block["timestamp_alignment"], "refused")
        self.assertFalse(block["pose_status_aligned"])
        self.assertEqual(block["world_points_scope"], "none")
        self.assertIsNone(block["T_world_cam"])
        self.assertNotIn("world_hands", block)
        self.assertEqual(episode["hands"]["left"]["wrist_pose"][0][0], wrist)

        import io
        import tempfile
        from contextlib import redirect_stdout
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            episode_path = tmp / "episode.json"
            episode_path.write_text(json.dumps(episode), encoding="utf-8")
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = check_main([
                    "--episode", str(episode_path),
                    "--slam-status", str(ROOT / "tests" / "fixtures" / "headcam" / "wilor_partial" / "slam_status.json"),
                ])
            self.assertEqual(code, 1)
            self.assertIn("timestamp_alignment=refused\n", buf.getvalue())

    def test_same_unit_offset_is_refused(self):
        from headcam.slam_frames import apply_ok_frame_world

        slam = json.loads(
            (ROOT / "tests" / "fixtures" / "headcam" / "wilor_partial" / "slam_status.json").read_text(encoding="utf-8")
        )
        episode = json.loads(
            (ROOT / "tests" / "fixtures" / "headcam" / "wilor_partial" / "episode.json").read_text(encoding="utf-8")
        )
        episode["timestamps"] = [10.0, 10.033, 10.066, 10.1]
        apply_ok_frame_world(episode, slam)
        self.assertEqual(episode["hand_pose"]["slam"]["timestamp_alignment"], "refused")
        self.assertIsNone(episode["hand_pose"]["slam"].get("world_hands"))


class EurocCalibMathTest(unittest.TestCase):
    """公开的 V1_01_easy 相机 yaml 数字。不是头戴标定。"""

    def test_T_right_left_goes_through_the_body_frame(self):
        from headcam.euroc_session import ns_to_second_text, stereo_from_euroc

        left = {
            "camera_model": "pinhole",
            "distortion_model": "radial-tangential",
            "resolution": [752, 480],
            "intrinsics": [458.654, 457.296, 367.215, 248.375],
            "distortion_coefficients": [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05],
            "T_BS": {"data": [
                0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
                0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
                -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
                0.0, 0.0, 0.0, 1.0,
            ]},
        }
        right = {
            "camera_model": "pinhole",
            "distortion_model": "radial-tangential",
            "resolution": [752, 480],
            "intrinsics": [457.587, 456.134, 379.999, 255.238],
            "distortion_coefficients": [-0.28368365, 0.07451284, -0.00010473, -3.55590700e-05],
            "T_BS": {"data": [
                0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556,
                0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024,
                -0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038,
                0.0, 0.0, 0.0, 1.0,
            ]},
        }
        calib, T_cam_imu, notes = stereo_from_euroc((left, "cam0"), (right, "cam1"))
        self.assertEqual(calib["dist_left"].shape, (4,))
        self.assertEqual(calib["dist_right"].shape, (4,))
        T_BS_left = np.asarray(left["T_BS"]["data"], dtype=float).reshape(4, 4)
        T_BS_right = np.asarray(right["T_BS"]["data"], dtype=float).reshape(4, 4)
        point = np.array([0.1, -0.2, 1.5, 1.0])
        pose = np.eye(4)
        pose[:3, :3] = calib["R"]
        pose[:3, 3] = calib["T"]
        by_stereo = pose @ point
        by_body = np.linalg.inv(T_BS_right) @ T_BS_left @ point
        np.testing.assert_allclose(by_stereo, by_body, atol=1e-9)
        np.testing.assert_allclose(T_cam_imu @ T_BS_left, np.eye(4), atol=1e-8)
        self.assertIn("不把缺少的 k3 补成 0", notes["distortion_note"])
        self.assertEqual(ns_to_second_text(1403715291262142976), "1403715291.262142976")


class UnmeasuredLeRobotPackageTest(unittest.TestCase):
    """没装 pyarrow 就跳过。图像是测试里生成的色块，位姿行来自已提交的 EuRoC 摘录。"""

    def test_package_keeps_pose_status_and_omits_hands(self):
        import shutil
        import subprocess
        import tempfile

        try:
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("未安装 pyarrow，跳过 LeRobot 包写入")
        if shutil.which("ffmpeg") is None:
            self.skipTest("没有 ffmpeg，跳过 LeRobot 包写入")

        excerpt = ROOT / "tests" / "fixtures" / "headcam" / "euroc_orbslam3_v101" / "CameraTrajectory_excerpt.txt"
        lines = []
        for line in excerpt.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                lines.append(line.strip())
        lines = lines[:3]
        stamps_ns = [int(line.split()[0].split(".")[0]) for line in lines]
        stamps_ns.append(stamps_ns[-1] + 10000000000)

        from headcam.export_mvp import main as export_main
        from headcam.euroc_session import ns_to_second_text
        from headcam.hand_pose import quat_xyzw_to_rotmat

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            session = tmp / "sess"
            (session / "stereo").mkdir(parents=True)
            (session / "metadata.json").write_text(json.dumps({
                "episode_id": "euroc/excerpt-prefix",
                "fps": 20.0,
            }), encoding="utf-8")
            stamp_lines = ["frame_index,timestamp_s"]
            lr_lines = ["frame_index,left_s,right_s"]
            for index, stamp in enumerate(stamps_ns):
                text = ns_to_second_text(stamp)
                stamp_lines.append("%d,%s" % (index, text))
                lr_lines.append("%d,%s,%s" % (index, text, text))
            (session / "timestamps.csv").write_text("\n".join(stamp_lines) + "\n", encoding="utf-8")
            (session / "stereo" / "timestamps_lr.csv").write_text("\n".join(lr_lines) + "\n", encoding="utf-8")
            write_calibration_yaml(session / "calib.yaml", _calib())
            raw = bytearray()
            for index in range(4):
                raw.extend(bytes((index * 40 % 256, 90, 140)) * (8 * 8))
            for name in ("left.mp4", "right.mp4"):
                subprocess.run(
                    [
                        "ffmpeg", "-y", "-loglevel", "error",
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "8x8", "-r", "20",
                        "-i", "pipe:0", "-frames:v", "4",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
                        str(session / "stereo" / name),
                    ],
                    input=bytes(raw),
                    check=True,
                )
            tum = tmp / "CameraTrajectory.txt"
            tum.write_text(
                "# 摘自 tests/fixtures/headcam/euroc_orbslam3_v101/CameraTrajectory_excerpt.txt 前 3 行。\n"
                + "\n".join(lines) + "\n",
                encoding="utf-8",
            )
            out = tmp / "out"
            code = export_main([
                "--session", str(session),
                "--out", str(out),
                "--slam-backend", "external_tum",
                "--tum", str(tum),
                "--tum-time-scale", "1e-9",
                "--skip-hands",
            ])
            self.assertEqual(code, 0, (out / "report.json").read_text(encoding="utf-8") if (out / "report.json").is_file() else "no report")
            report = json.loads((out / "report.json").read_text(encoding="utf-8"))
            self.assertIsNone(report["accuracy"])
            self.assertEqual(report["slam"]["ok_frames"], 3)
            self.assertEqual(report["slam"]["tracking_lost_frames"], 1)
            self.assertEqual(report["qc"]["accepted_episodes"], 1)
            self.assertEqual(report["qc"]["source"], "egodata.stereo_qc.qc_stereo_episode")
            note = json.loads((out / "lerobot" / "meta" / "egodata_export.json").read_text(encoding="utf-8"))
            self.assertIsNone(note["accuracy"])
            self.assertEqual(note["hands"], "absent")
            self.assertEqual(note["ok_frames"], 3)
            self.assertEqual(note["frames"], 4)
            info = json.loads((out / "lerobot" / "meta" / "info.json").read_text(encoding="utf-8"))
            self.assertNotIn("observation.state", info["features"])
            self.assertNotIn("action", info["features"])
            self.assertEqual(info["features"]["observation.image"]["info"]["egodata.image_source"], "source_mp4")
            table = pq.read_table(out / "lerobot" / "data" / "chunk-000" / "file-000.parquet")
            self.assertEqual(table.num_rows, 4)
            self.assertNotIn("observation.state", table.column_names)
            self.assertNotIn("action", table.column_names)
            self.assertNotIn("observation.hand_joints", table.column_names)
            self.assertNotIn("observation.wrist_pose", table.column_names)
            statuses = table.column("slam_status").to_pylist()
            self.assertEqual(statuses, ["ok", "ok", "ok", "tracking_lost"])
            poses = table.column("observation.pose").to_pylist()
            self.assertIsNone(poses[3])
            self.assertTrue(all(item is not None and any(abs(value) > 1e-6 for value in item) for item in poses[:3]))
            status = json.loads((out / "slam_status.json").read_text(encoding="utf-8"))
            for index in range(3):
                matrix = np.asarray(status["frames"][index]["T_world_cam"], dtype=float)
                pose = poses[index]
                np.testing.assert_allclose(pose[:3], matrix[:3, 3], atol=1e-8)
                np.testing.assert_allclose(quat_xyzw_to_rotmat(pose[3:]), matrix[:3, :3], atol=1e-6)
            read_back = [float(value) for value in table.column("timestamp").to_pylist()]
            for index, got in enumerate(read_back):
                self.assertAlmostEqual(got, index / 20.0, places=5)
            capture_ns = [int(value) for value in table.column("observation.capture_time_ns").to_pylist()]
            self.assertEqual(capture_ns, stamps_ns)


class OfficialLeRobotLoaderTest(unittest.TestCase):
    """官方 LeRobotDataset 读缺测手的包。没装 lerobot 或 torch 就跳过。"""

    def test_loader_reads_pose_video_and_fps_timestamps(self):
        import shutil
        import subprocess
        import tempfile

        try:
            import torch
            from lerobot.datasets.dataset_metadata import CODEBASE_VERSION
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            self.skipTest("未安装 lerobot 或 torch，跳过官方 Dataset 加载：%s" % exc)
        try:
            import pyarrow.parquet  # noqa: F401
        except ImportError:
            self.skipTest("未安装 pyarrow，跳过官方 Dataset 加载")
        if shutil.which("ffmpeg") is None:
            self.skipTest("没有 ffmpeg，跳过官方 Dataset 加载")

        excerpt = ROOT / "tests" / "fixtures" / "headcam" / "euroc_orbslam3_v101" / "CameraTrajectory_excerpt.txt"
        lines = []
        for line in excerpt.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                lines.append(line.strip())
        lines = lines[:3]
        stamps_ns = [int(line.split()[0].split(".")[0]) for line in lines]
        stamps_ns.append(stamps_ns[-1] + 10000000000)

        from headcam.export_mvp import main as export_main
        from headcam.euroc_session import ns_to_second_text
        from headcam.hand_pose import quat_xyzw_to_rotmat

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            session = tmp / "sess"
            (session / "stereo").mkdir(parents=True)
            (session / "metadata.json").write_text(json.dumps({
                "episode_id": "euroc/excerpt-prefix",
                "fps": 20.0,
            }), encoding="utf-8")
            stamp_lines = ["frame_index,timestamp_s"]
            lr_lines = ["frame_index,left_s,right_s"]
            for index, stamp in enumerate(stamps_ns):
                text = ns_to_second_text(stamp)
                stamp_lines.append("%d,%s" % (index, text))
                lr_lines.append("%d,%s,%s" % (index, text, text))
            (session / "timestamps.csv").write_text("\n".join(stamp_lines) + "\n", encoding="utf-8")
            (session / "stereo" / "timestamps_lr.csv").write_text("\n".join(lr_lines) + "\n", encoding="utf-8")
            write_calibration_yaml(session / "calib.yaml", _calib())
            raw = bytearray()
            for index in range(4):
                raw.extend(bytes((index * 40 % 256, 90, 140)) * (8 * 8))
            for name in ("left.mp4", "right.mp4"):
                subprocess.run(
                    [
                        "ffmpeg", "-y", "-loglevel", "error",
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "8x8", "-r", "20",
                        "-i", "pipe:0", "-frames:v", "4",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
                        str(session / "stereo" / name),
                    ],
                    input=bytes(raw),
                    check=True,
                )
            tum = tmp / "CameraTrajectory.txt"
            tum.write_text(
                "# 摘自 tests/fixtures/headcam/euroc_orbslam3_v101/CameraTrajectory_excerpt.txt 前 3 行。\n"
                + "\n".join(lines) + "\n",
                encoding="utf-8",
            )
            out = tmp / "out"
            code = export_main([
                "--session", str(session),
                "--out", str(out),
                "--slam-backend", "external_tum",
                "--tum", str(tum),
                "--tum-time-scale", "1e-9",
                "--skip-hands",
            ])
            self.assertEqual(code, 0)
            os.environ["HF_HUB_OFFLINE"] = "1"
            dataset = LeRobotDataset(repo_id="local/unmeasured_excerpt", root=out / "lerobot")
            self.assertEqual(dataset.meta.info.codebase_version, CODEBASE_VERSION)
            self.assertEqual(CODEBASE_VERSION, "v3.0")
            self.assertEqual(dataset.num_frames, 4)
            self.assertEqual(dataset.fps, 20)
            self.assertNotIn("observation.state", dataset.meta.features)
            self.assertNotIn("action", dataset.meta.features)
            self.assertNotIn("observation.hand_joints", dataset.meta.features)
            self.assertIn("timestamp", dataset.meta.stats)
            self.assertIn("observation.pose", dataset.meta.stats)
            self.assertIn("observation.image", dataset.meta.stats)
            self.assertEqual(len(dataset.meta.episodes), 1)
            self.assertEqual(int(dataset.meta.episodes[0]["length"]), 4)
            self.assertEqual(dataset.meta.tasks.index.tolist(), ["unknown"])
            note = json.loads((out / "lerobot" / "meta" / "egodata_export.json").read_text(encoding="utf-8"))
            self.assertIsNone(note["accuracy"])
            self.assertEqual(note["hands"], "absent")
            status = json.loads((out / "slam_status.json").read_text(encoding="utf-8"))
            frames = [dataset[index] for index in range(4)]
            images = []
            for index, item in enumerate(frames):
                self.assertEqual(item["task"], "unknown")
                self.assertEqual(item["slam_status"], "ok" if index < 3 else "tracking_lost")
                self.assertAlmostEqual(float(item["timestamp"]), index / 20.0, places=5)
                self.assertLess(abs(float(item["timestamp"]) - index / 20.0), 1e-4)
                self.assertEqual(int(item["observation.capture_time_ns"]), stamps_ns[index])
                image = item["observation.image"]
                self.assertEqual(tuple(image.shape), (3, 224, 224))
                self.assertEqual(image.dtype, torch.float32)
                self.assertGreater(float(image.max()), float(image.min()))
                self.assertGreaterEqual(float(image.min()), 0.0)
                self.assertLessEqual(float(image.max()), 1.0)
                images.append(image)
                pose = item["observation.pose"]
                if index < 3:
                    matrix = np.asarray(status["frames"][index]["T_world_cam"], dtype=float)
                    values = pose.detach().cpu().numpy()
                    np.testing.assert_allclose(values[:3], matrix[:3, 3], atol=1e-5)
                    np.testing.assert_allclose(quat_xyzw_to_rotmat(values[3:]), matrix[:3, :3], atol=1e-5)
                else:
                    self.assertTrue(pose is None or not torch.isfinite(pose).any())
            self.assertFalse(torch.equal(images[0], images[1]))


class PrepareSessionGuardTest(unittest.TestCase):
    def test_refuses_to_nest_work_copy_inside_source(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_session(Path(tmp) / "sess", [0.0])
            slam = run_slam(root, backend="none")
            with self.assertRaises(ValueError):
                prepare_hand_session(root, root / "nested", slam)


def _hands_episode(stamps, translation=(1.0, 0.0, 0.0)):
    """4 帧相机系手。数字是夹具，不是 WiLoR，也不是 HOT3D。"""
    num_frames = len(stamps)
    identity = np.eye(4)
    joints = []
    wrists = []
    for index, _stamp in enumerate(stamps):
        wrist = [0.0, 0.0, 0.50 + 0.01 * index]
        frame = []
        for joint in range(21):
            frame.append([wrist[0] + 0.001 * joint, wrist[1], wrist[2]])
        joints.append(frame)
        wrists.append(wrist + [0.0, 0.0, 0.0, 1.0])
    def _side(x_shift):
        shifted = [[[point[0] + x_shift, point[1], point[2]] for point in frame] for frame in joints]
        shifted_wrists = [[wrist[0] + x_shift, wrist[1], wrist[2], 0.0, 0.0, 0.0, 1.0] for wrist in wrists]
        return {
            "joints": shifted,
            "wrist_pose": shifted_wrists,
            "confidence": [1.0] * num_frames,
            "filled": [False] * num_frames,
            "valid": [True] * num_frames,
        }
    return {
        "episode_id": "headcam/hands-fixture",
        "num_frames": num_frames,
        "fps": 30.0,
        "timestamps": [float(stamp) for stamp in stamps],
        "image_width": 320,
        "image_height": 240,
        "camera_intrinsic": _calib()["K_left"].tolist(),
        "camera_poses": [identity.tolist() for _ in range(num_frames)],
        "coordinate_frame": "camera",
        "hands_measured": True,
        "hands": {"left": _side(0.0), "right": _side(0.02)},
        "stereo": {"per_frame": {"left": ["ok"] * num_frames, "right": ["ok"] * num_frames}},
        "annotation": {"task": {"instruction": "fixture-hands", "name": "fixture-hands"}},
        "hand_pose": {"backend": "fixture_not_wilor"},
    }


def _gt_tum_text(stamps):
    lines = [
        "# pose_source=hot3d_gt_mocap",
        "# fixture ground truth, not SLAM output",
    ]
    for stamp in stamps:
        lines.append("%.9f 1.000000000 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000 1.000000000" % float(stamp))
    return "\n".join(lines) + "\n"


class HandsPresentPackageTest(unittest.TestCase):
    """有手的导出。不需要 WiLoR。没装 pyarrow 或 ffmpeg 就跳过包写入。"""

    def _session_and_episode(self, tmp, stamp_shift=0.0):
        from headcam.euroc_session import ns_to_second_text

        origin = 1000000000000
        step = 33333333
        stamps_ns = [origin + index * step for index in range(4)]
        texts = [ns_to_second_text(value) for value in stamps_ns]
        stamps = [float(text) for text in texts]
        session = _write_session(Path(tmp) / "sess", stamps)
        raw = bytearray()
        for index in range(4):
            raw.extend(bytes((40 + index, 90, 140)) * (320 * 240))
        import subprocess
        for name in ("left.mp4", "right.mp4"):
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "320x240", "-r", "30",
                    "-i", "pipe:0", "-frames:v", "4",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
                    str(session / "stereo" / name),
                ],
                input=bytes(raw),
                check=True,
            )
        tum = Path(tmp) / "headset_tum.txt"
        tum.write_text(_gt_tum_text(stamps), encoding="utf-8")
        episode = _hands_episode([stamp + stamp_shift for stamp in stamps])
        episode_path = Path(tmp) / "camera_episode.json"
        episode_path.write_text(json.dumps(episode), encoding="utf-8")
        return session, tum, episode_path, stamps

    def test_mismatched_clock_is_still_refused(self):
        import tempfile

        from headcam.slam_frames import main as slam_frames_main

        with tempfile.TemporaryDirectory() as tmp:
            session, tum, episode_path, _stamps = self._session_and_episode(tmp, stamp_shift=10.0)
            before = json.loads(episode_path.read_text(encoding="utf-8"))
            out = Path(tmp) / "checked.json"
            code = slam_frames_main([
                "--episode", str(episode_path),
                "--session", str(session),
                "--tum", str(tum),
                "--backend", "external_tum",
                "--out", str(out),
            ])
            self.assertEqual(code, 1)
            checked = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(checked["hand_pose"]["slam"]["timestamp_alignment"], "refused")
            self.assertEqual(checked["hands"]["left"]["joints"][0][0], before["hands"]["left"]["joints"][0][0])
            self.assertIsNone(checked["hand_pose"]["slam"]["accuracy"])

    def test_export_labels_ground_truth_and_writes_state(self):
        import shutil
        import tempfile

        try:
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("未安装 pyarrow，跳过带手部的 LeRobot 包写入")
        if shutil.which("ffmpeg") is None:
            self.skipTest("没有 ffmpeg，跳过带手部的 LeRobot 包写入")

        from headcam.export_mvp import main as export_main
        from headcam.slam_frames import main as slam_frames_main
        from headcam.slam_vio import run_slam

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            session, tum, episode_path, stamps = self._session_and_episode(tmp)
            with self.assertRaises(ValueError):
                run_slam(session, backend="external_tum", tum_path=tum, pose_source="orbslam3_stereo_euroc")
            checked = tmp / "episodes" / "session.json"
            code = slam_frames_main([
                "--episode", str(episode_path),
                "--session", str(session),
                "--tum", str(tum),
                "--backend", "external_tum",
                "--out", str(checked),
            ])
            world = json.loads(checked.read_text(encoding="utf-8"))
            block = world["hand_pose"]["slam"]
            self.assertEqual(block["timestamp_alignment"], "timestamps")
            self.assertEqual(block["pose_source"], "hot3d_gt_mocap")
            self.assertIsNone(block["accuracy"])
            self.assertEqual(block.get("world_points_scope"), "all_frames")
            self.assertAlmostEqual(world["hands"]["left"]["joints"][0][0][0], 1.0, places=6)
            out = tmp / "out"
            export_code = export_main([
                "--session", str(session),
                "--out", str(out),
                "--slam-backend", "external_tum",
                "--tum", str(tum),
                "--episodes", str(tmp / "episodes"),
                "--pose-source", "hot3d_gt_mocap",
            ])
            report_path = out / "report.json"
            report_text = report_path.read_text(encoding="utf-8") if report_path.is_file() else ""
            self.assertEqual(export_code, 0, report_text)
            report = json.loads(report_text)
            self.assertEqual(report["pose_source"], "hot3d_gt_mocap")
            self.assertIsNone(report["accuracy"])
            self.assertIsNone(report["slam"]["accuracy"])
            self.assertIn("不是 ORB-SLAM3", report["slam"]["notes"])
            note = json.loads((out / "lerobot" / "meta" / "egodata_export.json").read_text(encoding="utf-8"))
            self.assertEqual(note["pose_source"], "hot3d_gt_mocap")
            self.assertIsNone(note["accuracy"])
            info = json.loads((out / "lerobot" / "meta" / "info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["features"]["observation.state"]["shape"], [140])
            self.assertEqual(info["features"]["action"]["shape"], [14])
            self.assertEqual(info["total_frames"], 3)
            table = pq.read_table(out / "lerobot" / "data" / "chunk-000" / "file-000.parquet")
            self.assertEqual(table.num_rows, 3)
            state = np.asarray(table.column("observation.state").combine_chunks().flatten().to_numpy(), dtype=float).reshape(3, 140)
            action = np.asarray(table.column("action").combine_chunks().flatten().to_numpy(), dtype=float).reshape(3, 14)
            self.assertFalse(np.allclose(state[:, :63], 0.0))
            self.assertAlmostEqual(state[0, 0], 1.0, places=5)
            self.assertAlmostEqual(state[0, 2], 0.50, places=5)
            self.assertFalse(np.allclose(action[:, :3], 0.0))
            poses = table.column("observation.pose").to_pylist()
            self.assertEqual(len(poses[0]), 7)
            self.assertAlmostEqual(poses[0][0], 1.0, places=6)
            self.assertAlmostEqual(poses[0][6], 1.0, places=6)
            capture = table.column("observation.capture_time_ns").to_pylist()
            self.assertEqual(capture[0], int(round(stamps[0] * 1e9)))
            self.assertIn(code, (0, 1))


class HandsPresentOfficialLoaderTest(unittest.TestCase):
    """官方加载器读带手的包。没装 lerobot 或 torch 就跳过。"""

    def test_loader_reads_state_action_and_pose(self):
        import shutil
        import tempfile

        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            self.skipTest("未安装 lerobot 或 torch，跳过官方 Dataset 加载：%s" % exc)
        try:
            import pyarrow.parquet  # noqa: F401
        except ImportError:
            self.skipTest("未安装 pyarrow，跳过官方 Dataset 加载")
        if shutil.which("ffmpeg") is None:
            self.skipTest("没有 ffmpeg，跳过官方 Dataset 加载")
        case = HandsPresentPackageTest()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            session, tum, episode_path, stamps = case._session_and_episode(tmp)
            from headcam.export_mvp import main as export_main
            from headcam.slam_frames import main as slam_frames_main
            checked = tmp / "episodes" / "session.json"
            slam_frames_main([
                "--episode", str(episode_path),
                "--session", str(session),
                "--tum", str(tum),
                "--backend", "external_tum",
                "--out", str(checked),
            ])
            out = tmp / "out"
            code = export_main([
                "--session", str(session),
                "--out", str(out),
                "--slam-backend", "external_tum",
                "--tum", str(tum),
                "--episodes", str(tmp / "episodes"),
                "--pose-source", "hot3d_gt_mocap",
            ])
            self.assertEqual(code, 0)
            os.environ["HF_HUB_OFFLINE"] = "1"
            dataset = LeRobotDataset(repo_id="local/headcam_mvp", root=out / "lerobot")
            self.assertEqual(dataset.num_frames, 3)
            self.assertEqual(tuple(dataset[0]["observation.state"].shape), (140,))
            self.assertEqual(tuple(dataset[0]["action"].shape), (14,))
            self.assertAlmostEqual(float(dataset[0]["timestamp"]), 0.0, places=5)
            pose = dataset[0]["observation.pose"]
            self.assertAlmostEqual(float(pose[0]), 1.0, places=4)


def _gt_tum_at(stamps, translation):
    tx, ty, tz = translation
    lines = [
        "# pose_source=hot3d_gt_mocap",
        "# fixture ground truth, not SLAM output",
    ]
    for stamp in stamps:
        lines.append(
            "%.9f %.9f %.9f %.9f 0.000000000 0.000000000 0.000000000 1.000000000"
            % (float(stamp), float(tx), float(ty), float(tz))
        )
    return "\n".join(lines) + "\n"


def _retarget_wrists(episode, z0, dz):
    """只改夹具手腕的 z，让两段的下一步增量不一样。"""
    for side in ("left", "right"):
        hand = episode["hands"][side]
        for index, frame in enumerate(hand["joints"]):
            z = float(z0) + float(dz) * index
            for point in frame:
                point[2] = z
            hand["wrist_pose"][index][2] = z
    return episode


class MultiEpisodePackageTest(unittest.TestCase):
    """多段已经标好的世界 episode 打成一个包。不用某一段的轨迹覆盖其他段。"""

    def _world_file(self, root, name, stamps_ns, translation, z0, dz, episode_id):
        from headcam.euroc_session import ns_to_second_text
        from headcam.slam_frames import main as slam_frames_main

        texts = [ns_to_second_text(value) for value in stamps_ns]
        stamps = [float(text) for text in texts]
        session = _write_session(root / ("sess-" + name), stamps)
        episode = _hands_episode(stamps)
        episode["episode_id"] = episode_id
        _retarget_wrists(episode, z0, dz)
        camera = root / ("%s_camera.json" % name)
        camera.write_text(json.dumps(episode), encoding="utf-8")
        tum = root / ("%s_tum.txt" % name)
        tum.write_text(_gt_tum_at(stamps, translation), encoding="utf-8")
        out = root / "episodes" / ("%s.json" % name)
        code = slam_frames_main([
            "--episode", str(camera),
            "--session", str(session),
            "--tum", str(tum),
            "--backend", "external_tum",
            "--out", str(out),
        ])
        self.assertEqual(code, 0, out.read_text(encoding="utf-8") if out.is_file() else "no episode")
        return out, stamps

    def test_packaged_episodes_do_not_share_poses_or_actions(self):
        import shutil
        import tempfile

        try:
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("未安装 pyarrow，跳过多 episode 包写入")
        if shutil.which("ffmpeg") is None:
            self.skipTest("没有 ffmpeg，跳过多 episode 包写入")

        from egodata.lerobot_export import pack_action, wrist_delta
        from headcam.export_mvp import main as export_main
        from headcam.hand_pose import quat_xyzw_to_rotmat

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            origin = 1000000000000
            step = 33333333
            path_a, stamps_a = self._world_file(
                root, "a",
                [origin + index * step for index in range(4)],
                (1.0, 0.0, 0.0), 0.50, 0.01,
                "hot3d/quest3/clip-a",
            )
            path_b, stamps_b = self._world_file(
                root, "b",
                [2 * origin + index * step for index in range(5)],
                (3.0, 0.0, 0.2), 0.40, -0.02,
                "hot3d/quest3/clip-b",
            )
            path_bad, _stamps_bad = self._world_file(
                root, "bad",
                [3 * origin + index * step for index in range(4)],
                (5.0, 0.0, 0.0), 0.40, 0.02,
                "hot3d/quest3/clip-bad",
            )
            bad = json.loads(path_bad.read_text(encoding="utf-8"))
            for side in ("left", "right"):
                bad["hands"][side]["confidence"] = [0.0] * bad["num_frames"]
            path_bad.write_text(json.dumps(bad), encoding="utf-8")
            world_a = json.loads(path_a.read_text(encoding="utf-8"))
            world_b = json.loads(path_b.read_text(encoding="utf-8"))
            world_a["hand_pose"]["slam"]["accuracy"] = 0.02
            path_a.write_text(json.dumps(world_a), encoding="utf-8")
            before = {
                path_a: path_a.read_text(encoding="utf-8"),
                path_b: path_b.read_text(encoding="utf-8"),
                path_bad: path_bad.read_text(encoding="utf-8"),
            }
            out = root / "out"
            code = export_main([
                "--package-annotated",
                "--episodes", str(root / "episodes"),
                "--out", str(out),
                "--repo-id", "local/headcam_mvp_multi",
            ])
            report_text = (out / "report.json").read_text(encoding="utf-8") if (out / "report.json").is_file() else ""
            self.assertEqual(code, 0, report_text)
            for path, text in before.items():
                self.assertEqual(path.read_text(encoding="utf-8"), text)
            report = json.loads(report_text)
            self.assertEqual(report["pose_source"], "hot3d_gt_mocap")
            self.assertIsNone(report["accuracy"])
            rejected_ids = [item["episode_id"] for item in report["rejected_episodes"]]
            self.assertEqual(rejected_ids, ["hot3d/quest3/clip-bad"])
            self.assertTrue(report["rejected_episodes"][0]["reasons"])
            note = json.loads((out / "lerobot" / "meta" / "egodata_export.json").read_text(encoding="utf-8"))
            self.assertEqual(note["pose_source"], "hot3d_gt_mocap")
            self.assertIsNone(note["accuracy"])
            info = json.loads((out / "lerobot" / "meta" / "info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["total_episodes"], 2)
            self.assertEqual(info["total_frames"], 7)
            self.assertEqual(info["pose_source"], "hot3d_gt_mocap")
            self.assertNotIn("accuracy", info)
            events = [
                json.loads(line)
                for line in (out / "lerobot" / "meta" / "interaction_events.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            by_id = {item["episode_id"]: item["episode_index"] for item in events}
            self.assertEqual(set(by_id), {"hot3d/quest3/clip-a", "hot3d/quest3/clip-b"})
            episodes = pq.read_table(
                out / "lerobot" / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
            ).to_pylist()
            self.assertEqual(len(episodes), 2)
            lengths = {item["episode_index"]: item["length"] for item in episodes}
            self.assertEqual(lengths[by_id["hot3d/quest3/clip-a"]], 3)
            self.assertEqual(lengths[by_id["hot3d/quest3/clip-b"]], 4)
            for item in episodes:
                self.assertEqual(item["videos/observation.image/from_timestamp"], 0.0)
                self.assertNotEqual(item["dataset_from_index"], item["dataset_to_index"])
            spans = sorted((item["dataset_from_index"], item["dataset_to_index"]) for item in episodes)
            self.assertEqual(spans[0][0], 0)
            self.assertEqual(spans[0][1], spans[1][0])
            self.assertEqual(spans[1][1], 7)
            table = pq.read_table(out / "lerobot" / "data" / "chunk-000" / "file-000.parquet")
            rows = table.to_pylist()
            self.assertEqual(len(rows), 7)
            worlds = {
                "hot3d/quest3/clip-a": world_a,
                "hot3d/quest3/clip-b": world_b,
            }
            stamps = {
                "hot3d/quest3/clip-a": stamps_a,
                "hot3d/quest3/clip-b": stamps_b,
            }
            for episode_id, index in by_id.items():
                world = worlds[episode_id]
                kept = int(world["num_frames"]) - 1
                subset = [row for row in rows if row["episode_index"] == index]
                self.assertEqual(len(subset), kept)
                for frame_index, row in enumerate(subset):
                    self.assertEqual(row["frame_index"], frame_index)
                    self.assertAlmostEqual(float(row["timestamp"]), frame_index / 30.0, places=5)
                    self.assertEqual(int(row["observation.capture_time_ns"]), int(round(stamps[episode_id][frame_index] * 1e9)))
                    matrix = np.asarray(world["hand_pose"]["slam"]["T_world_cam"][frame_index], dtype=float)
                    pose = row["observation.pose"]
                    np.testing.assert_allclose(pose[:3], matrix[:3, 3], atol=1e-6)
                    np.testing.assert_allclose(quat_xyzw_to_rotmat(pose[3:]), matrix[:3, :3], atol=1e-6)
                    expected = pack_action(world, frame_index)
                    np.testing.assert_allclose(row["action"], expected, atol=1e-5)
                last = kept - 1
                stored = np.asarray(subset[last]["action"], dtype=float)
                other_id = "hot3d/quest3/clip-b" if episode_id.endswith("clip-a") else "hot3d/quest3/clip-a"
                other = worlds[other_id]
                crossed = np.concatenate([
                    wrist_delta(
                        world["hands"][side]["wrist_pose"][last],
                        other["hands"][side]["wrist_pose"][0],
                    )
                    for side in ("left", "right")
                ])
                self.assertGreater(float(np.linalg.norm(stored - crossed)), 0.5)

    def test_mismatched_pose_source_is_refused(self):
        import shutil
        import tempfile

        try:
            import pyarrow.parquet  # noqa: F401
        except ImportError:
            self.skipTest("未安装 pyarrow，跳过 pose_source 一致性检查")
        if shutil.which("ffmpeg") is None:
            self.skipTest("没有 ffmpeg，跳过 pose_source 一致性检查")

        from headcam.export_mvp import main as export_main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            origin = 1000000000000
            step = 33333333
            path_a, _ = self._world_file(
                root, "a",
                [origin + index * step for index in range(4)],
                (1.0, 0.0, 0.0), 0.50, 0.01,
                "hot3d/quest3/clip-a",
            )
            self._world_file(
                root, "b",
                [2 * origin + index * step for index in range(4)],
                (2.0, 0.0, 0.0), 0.50, 0.01,
                "hot3d/quest3/clip-b",
            )
            episode = json.loads(path_a.read_text(encoding="utf-8"))
            episode["hand_pose"]["slam"]["pose_source"] = "orbslam3_stereo_euroc"
            path_a.write_text(json.dumps(episode), encoding="utf-8")
            out = root / "out"
            code = export_main([
                "--package-annotated",
                "--episodes", str(root / "episodes"),
                "--out", str(out),
            ])
            self.assertEqual(code, 1)
            self.assertFalse((out / "lerobot").exists())


class DuplicateTimestampAteTest(unittest.TestCase):
    """重复时间戳不能凑成 ATE。不需要 WiLoR、lerobot 或 HOT3D 图像。"""

    def test_two_unique_times_do_not_report_rmse(self):
        import tempfile

        from headcam.hot3d_check import trajectory_ate

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            estimate = root / "estimate.txt"
            truth = root / "gt.txt"
            # 两行不同时间，其余是同一时刻的重复。尺度是秒。
            estimate.write_text(
                "1.0 0 0 0 0 0 0 1\n"
                "1.1 0.01 0 0 0 0 0 1\n"
                + "1.1 0.01 0 0 0 0 0 1\n" * 20,
                encoding="utf-8",
            )
            truth.write_text(
                "".join(
                    "%s 0 0 0 0 0 0 1\n" % (1.0 + index * 0.1) for index in range(5)
                ),
                encoding="utf-8",
            )
            report = trajectory_ate(estimate, truth)
        self.assertLess(report["n"], 3)
        self.assertNotIn("rmse_m", report)
        self.assertGreater(report["duplicate_rows_ignored"], 0)
        self.assertEqual(report["unique_timestamps"], 2)


def _tum_line(stamp, pose):
    from headcam.hot3d_adapter import rotmat_to_quat_xyzw

    quat = rotmat_to_quat_xyzw(pose[:3, :3])
    values = list(pose[:3, 3]) + list(quat)
    return "%s %s" % (stamp, " ".join("%.9f" % float(value) for value in values))


def _write_pose_tum(path, stamps, poses):
    path.write_text("".join(_tum_line(stamp, pose) + "\n" for stamp, pose in zip(stamps, poses)), encoding="utf-8")


class SyntheticTrajectoryErrorTest(unittest.TestCase):
    """ATE / RPE 用已知刚体答案。不读图像，也不把真值叫做 SLAM。"""

    def _square(self):
        return [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        ]

    def _poses_from_points(self, points, rotation=None):
        rotation = np.eye(3) if rotation is None else rotation
        poses = []
        for point in points:
            pose = np.eye(4)
            pose[:3, :3] = rotation
            pose[:3, 3] = point
            poses.append(pose)
        return poses

    def test_se3_and_sim3_on_a_known_similarity(self):
        import tempfile

        from headcam.hot3d_check import trajectory_ate

        points = self._square()
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        shift = np.array([5.0, -2.0, 1.0])
        scaled = [2.0 * (rotation @ point) + shift for point in points]
        rigid = [(rotation @ point) + shift for point in points]
        stamps = ["0.0", "1.0", "2.0", "3.0"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            truth = root / "gt.txt"
            doubled = root / "x2.txt"
            rigid_path = root / "rigid.txt"
            _write_pose_tum(truth, stamps, self._poses_from_points(points))
            _write_pose_tum(doubled, stamps, self._poses_from_points(scaled))
            _write_pose_tum(rigid_path, stamps, self._poses_from_points(rigid))
            se3 = trajectory_ate(rigid_path, truth, alignment="se3")
            sim3 = trajectory_ate(doubled, truth, alignment="sim3")
            se3_on_scaled = trajectory_ate(doubled, truth, alignment="se3")
        self.assertEqual(se3["alignment"], "umeyama_se3")
        self.assertEqual(se3["scale"], "fixed_1")
        self.assertLess(se3["rmse_m"], 1e-8)
        self.assertEqual(se3["n"], 4)
        self.assertEqual(sim3["alignment"], "umeyama_sim3")
        self.assertAlmostEqual(sim3["similarity_scale"], 0.5, places=8)
        self.assertLess(sim3["rmse_m"], 1e-8)
        self.assertGreater(se3_on_scaled["rmse_m"], 0.5)

    def test_rpe_is_the_relative_step_and_ignores_a_fixed_offset(self):
        import tempfile

        from headcam.hot3d_check import trajectory_rpe

        gt_points = self._square()
        stepped = [np.array([0.0, 0.0, 0.0])]
        for previous, current in zip(gt_points, gt_points[1:]):
            stepped.append(stepped[-1] + 1.1 * (current - previous))
        offset = np.eye(4)
        offset[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        offset[:3, 3] = [4.0, 5.0, 6.0]
        gt_poses = self._poses_from_points(gt_points)
        shifted = [offset @ pose for pose in gt_poses]
        stamps = ["0.0", "1.0", "2.0", "3.0"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            truth = root / "gt.txt"
            fast = root / "fast.txt"
            moved = root / "moved.txt"
            _write_pose_tum(truth, stamps, gt_poses)
            _write_pose_tum(fast, stamps, self._poses_from_points(stepped))
            _write_pose_tum(moved, stamps, shifted)
            drifted = trajectory_rpe(fast, truth, delta_s=1.0)
            rigid = trajectory_rpe(moved, truth, delta_s=1.0)
            short = trajectory_rpe(truth, truth, delta_s=10.0)
        self.assertEqual(drifted["n_pairs"], 3)
        self.assertAlmostEqual(drifted["trans_rmse_m"], 0.1, places=8)
        self.assertAlmostEqual(drifted["rot_rmse_deg"], 0.0, places=6)
        self.assertEqual(rigid["alignment"], "none_relative")
        self.assertLess(rigid["trans_rmse_m"], 1e-8)
        self.assertNotIn("trans_rmse_m", short)

    def test_two_poses_do_not_report_rpe(self):
        import tempfile

        from headcam.hot3d_check import trajectory_rpe

        pose = np.eye(4)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "short.txt"
            _write_pose_tum(path, ["0.0", "1.0"], [pose, pose])
            report = trajectory_rpe(path, path, delta_s=1.0)
        self.assertLess(report["n_poses"], 3)
        self.assertNotIn("trans_rmse_m", report)


class Hot3dSequenceContinuityTest(unittest.TestCase):
    """同一条录制、帧间隔接得上才拼接。不下载 clip。"""

    def _clip(self, clip_id, sequence_id, start_ns, count=4, step=33333333):
        return {
            "clip_id": clip_id,
            "sequence_id": sequence_id,
            "device": "Quest3",
            "timestamps_ns": [start_ns + index * step for index in range(count)],
        }

    def test_abutting_clips_become_one_increasing_clock(self):
        from headcam.hot3d_sequence import concat_clip_records, contiguous_runs

        step = 33333333
        first = self._clip(10, "seq-a", 0)
        second = self._clip(11, "seq-a", 4 * step)
        gapped = self._clip(12, "seq-a", 4 * step + 2_000_000_000)
        other = self._clip(13, "seq-b", 8 * step)
        runs = contiguous_runs([other, gapped, second, first])
        by_ids = {tuple(run["clip_ids"]): run for run in runs}
        joined = by_ids[(10, 11)]
        self.assertEqual(joined["n_frames"], 8)
        self.assertEqual(joined["gaps_ns"], [step])
        self.assertEqual(joined["timestamps_ns"], list(range(0, 8 * step, step)))
        self.assertIn((12,), by_ids)
        self.assertIn((13,), by_ids)
        again = concat_clip_records([second, first])
        self.assertEqual(again["timestamps_ns"], joined["timestamps_ns"])
        with self.assertRaises(ValueError):
            concat_clip_records([first, gapped])

    def test_orbslam3_log_counts_are_literal(self):
        from headcam.hot3d_sequence import parse_orbslam3_stdout

        text = "\n".join([
            "Fail to track local map!",
            "Fail to track local map!",
            "New Map created with 20 points",
            "There are 2 maps in the atlas",
            "  Map 0 has 1 KFs",
            "  Map 1 has 40 KFs",
            "TRACK_REF_KF: Less than 15 matches!!",
        ])
        report = parse_orbslam3_stdout(text)
        self.assertEqual(report["fail_to_track_local_map"], 2)
        self.assertEqual(report["new_maps"], 1)
        self.assertEqual(report["atlas_maps"], 2)
        self.assertEqual(report["few_reference_matches"], 1)
        self.assertEqual(report["map_keyframes"], [
            {"map": 0, "keyframes": 1},
            {"map": 1, "keyframes": 40},
        ])


class Orbslam3ExtrinsicConventionTest(unittest.TestCase):
    """OpenCV 的 p_right = R p_left + t 不能原样写进 ORB-SLAM3 的 T_c1_c2。"""

    def test_yaml_stores_the_inverse_baseline(self):
        import tempfile

        from headcam.hot3d_session import write_orbslam3_yaml

        calib = {
            "K_left": np.array([[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]]),
            "K_right": np.array([[400.0, 0.0, 330.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]]),
            "R": np.eye(3),
            "T": np.array([-0.1, 0.0, 0.0]),
            "image_width": 640,
            "image_height": 480,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = write_orbslam3_yaml(Path(tmp) / "cam.yaml", calib, 30)
            text = path.read_text(encoding="utf-8")
        self.assertIn("再取逆", text)
        data = text.split("data: [", 1)[1].split("]", 1)[0]
        values = [float(item) for item in data.split(",")]
        self.assertEqual(len(values), 16)
        self.assertAlmostEqual(values[3], 0.1, places=8)
        self.assertAlmostEqual(values[7], 0.0, places=8)

    def test_rectified_pair_has_no_vertical_error(self):
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("没有 cv2，跳过立体校正")
        from headcam.stereo_geometry import rectify_pinhole_pair, vertical_epipolar_pixels

        left = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
        right = np.array([[500.0, 0.0, 325.0], [0.0, 500.0, 242.0], [0.0, 0.0, 1.0]])
        angle = np.radians(2.0)
        rotation = np.array([
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ])
        translation = np.array([-0.10, 0.004, 0.001])
        raw = vertical_epipolar_pixels(left, right, rotation, translation, (640, 480))
        self.assertGreater(raw["median_px"], 1.0)
        rectified = rectify_pinhole_pair(left, right, rotation, translation, (640, 480))
        errors = []
        width, height = 640, 480
        r1 = np.asarray(rectified["R1"], dtype=float)
        r2 = np.asarray(rectified["R2"], dtype=float)
        p1 = np.asarray(rectified["P1"], dtype=float)
        p2 = np.asarray(rectified["P2"], dtype=float)
        for depth in (0.5, 1.5):
            for u in (120.0, 320.0, 500.0):
                for v in (80.0, 240.0, 400.0):
                    point_left = np.array([
                        (u - left[0, 2]) / left[0, 0] * depth,
                        (v - left[1, 2]) / left[1, 1] * depth,
                        depth,
                    ])
                    point_right = rotation @ point_left + translation
                    left_r = r1 @ point_left
                    right_r = r2 @ point_right
                    if left_r[2] <= 0.05 or right_r[2] <= 0.05:
                        continue
                    v_left = p1[1, 1] * left_r[1] / left_r[2] + p1[1, 2]
                    v_right = p2[1, 1] * right_r[1] / right_r[2] + p2[1, 2]
                    u_left = p1[0, 0] * left_r[0] / left_r[2] + p1[0, 2]
                    u_right = p2[0, 0] * right_r[0] / right_r[2] + p2[0, 2]
                    if min(u_left, u_right, v_left, v_right) < 0 or max(u_left, u_right) >= width or max(v_left, v_right) >= height:
                        continue
                    errors.append(abs(v_left - v_right))
        self.assertGreaterEqual(len(errors), 3)
        self.assertLess(max(errors), 1e-6)
        self.assertAlmostEqual(rectified["baseline_m"], 0.10, places=3)


class SlamAccuracyPackagingTest(unittest.TestCase):
    """测过的 Umeyama 误差可以进 accuracy。头戴真值改称 SLAM、或手写数字，都不进。"""

    def _moving_pair(self, root):
        stamps = [index / 30.0 for index in range(4)]
        points = [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        ]
        shift = np.array([0.4, -0.2, 0.1])
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        gt_poses = []
        est_poses = []
        for point in points:
            gt = np.eye(4)
            gt[:3, 3] = point
            est = np.eye(4)
            est[:3, :3] = rotation
            est[:3, 3] = rotation @ point + shift
            gt_poses.append(gt)
            est_poses.append(est)
        gt_path = root / "gt.txt"
        est_path = root / "slam.txt"
        gt_path.write_text(
            "# pose_source=hot3d_gt_mocap\n" + "".join(_tum_line(stamp, pose) + "\n" for stamp, pose in zip(stamps, gt_poses)),
            encoding="utf-8",
        )
        est_path.write_text(
            "# pose_source=orbslam3_stereo_euroc\n" + "".join(_tum_line(stamp, pose) + "\n" for stamp, pose in zip(stamps, est_poses)),
            encoding="utf-8",
        )
        return stamps, est_path, gt_path, points

    def test_known_rigid_motion_publishes_se3_and_sim3(self):
        import tempfile

        from headcam.hot3d_check import is_measured_accuracy, trajectory_accuracy

        with tempfile.TemporaryDirectory() as tmp:
            _stamps, est_path, gt_path, points = self._moving_pair(Path(tmp))
            measured = trajectory_accuracy(
                est_path, gt_path,
                estimate_pose_source="orbslam3_stereo_euroc",
                reference_pose_source="hot3d_gt_mocap",
            )
            refused = trajectory_accuracy(
                est_path, gt_path,
                estimate_pose_source="hot3d_gt_mocap",
                reference_pose_source="hot3d_gt_mocap",
            )
        self.assertIsNone(refused)
        self.assertTrue(is_measured_accuracy(measured, "orbslam3_stereo_euroc"))
        self.assertLess(measured["ate"]["se3"]["rmse_m"], 1e-8)
        self.assertAlmostEqual(measured["ate"]["sim3"]["similarity_scale"], 1.0, places=6)
        self.assertEqual(measured["ate"]["se3"]["alignment"], "umeyama_se3")
        self.assertEqual(measured["ate"]["sim3"]["alignment"], "umeyama_sim3")
        self.assertEqual(measured["max_dt_s"], 0.05)
        self.assertEqual([row["delta_s"] for row in measured["rpe"]], [1.0, 5.0])
        rotation = np.asarray(measured["ate"]["se3"]["rotation_row_major"], dtype=float).reshape(3, 3)
        translation = np.asarray(measured["ate"]["se3"]["translation"], dtype=float)
        # The stored SE3 is the same Umeyama that produced the RMSE.
        estimate_centers = []
        for point in points:
            pose = np.eye(4)
            pose[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
            pose[:3, 3] = pose[:3, :3] @ point + np.array([0.4, -0.2, 0.1])
            estimate_centers.append(pose[:3, 3])
        aligned = (rotation @ np.asarray(estimate_centers).T).T + translation
        self.assertLess(float(np.max(np.linalg.norm(aligned - np.asarray(points), axis=1))), 1e-8)
        self.assertIsNone(trajectory_accuracy(
            est_path, gt_path,
            estimate_pose_source="hot3d_gt_mocap",
            reference_pose_source="hot3d_gt_mocap",
        ))
        self.assertFalse(is_measured_accuracy(0.01, "orbslam3_stereo_euroc"))
        self.assertFalse(is_measured_accuracy(measured, "hot3d_gt_mocap"))

    def test_two_poses_leave_accuracy_null_and_gt_label_is_refused(self):
        import tempfile

        from headcam.hot3d_check import trajectory_accuracy
        from headcam.slam_vio import run_slam

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stamps, est_path, gt_path, _points = self._moving_pair(root)
            pose = np.eye(4)
            tiny = root / "tiny.txt"
            _write_pose_tum(tiny, ["0.0", "1.0"], [pose, pose])
            self.assertIsNone(trajectory_accuracy(
                tiny, tiny,
                estimate_pose_source="orbslam3_stereo_euroc",
                reference_pose_source="hot3d_gt_mocap",
            ))
            session = _write_session(root / "sess", stamps)
            bare = run_slam(
                session, backend="external_tum", tum_path=est_path, pose_source="orbslam3_stereo_euroc",
            )
            self.assertIsNone(bare["accuracy"])
            self.assertEqual(bare["pose_source"], "orbslam3_stereo_euroc")
            measured = run_slam(
                session, backend="external_tum", tum_path=est_path, pose_source="orbslam3_stereo_euroc",
                gt_tum_path=gt_path,
            )
            self.assertLess(measured["accuracy"]["ate"]["se3"]["rmse_m"], 1e-8)
            gt_only = run_slam(
                session, backend="external_tum", tum_path=gt_path, pose_source="hot3d_gt_mocap",
                gt_tum_path=gt_path,
            )
            self.assertIsNone(gt_only["accuracy"])
            with self.assertRaises(ValueError):
                run_slam(
                    session, backend="external_tum", tum_path=gt_path, pose_source="orbslam3_stereo_euroc",
                    gt_tum_path=gt_path,
                )

    def test_skip_hands_package_keeps_slam_source_and_measured_accuracy(self):
        import shutil
        import tempfile

        try:
            import pyarrow.parquet  # noqa: F401
        except ImportError:
            self.skipTest("未安装 pyarrow，跳过 SLAM 来源的 LeRobot 包")
        if shutil.which("ffmpeg") is None:
            self.skipTest("没有 ffmpeg，跳过 SLAM 来源的 LeRobot 包")

        from headcam.export_mvp import main as export_main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stamps, est_path, gt_path, _points = self._moving_pair(root)
            session = _write_session(root / "sess", stamps)
            out = root / "out"
            code = export_main([
                "--session", str(session),
                "--out", str(out),
                "--skip-hands",
                "--slam-backend", "external_tum",
                "--tum", str(est_path),
                "--pose-source", "orbslam3_stereo_euroc",
                "--gt-tum", str(gt_path),
            ])
            report_text = (out / "report.json").read_text(encoding="utf-8") if (out / "report.json").is_file() else ""
            self.assertEqual(code, 0, report_text)
            report = json.loads(report_text)
            self.assertEqual(report["pose_source"], "orbslam3_stereo_euroc")
            self.assertLess(report["accuracy"]["ate"]["se3"]["rmse_m"], 1e-8)
            self.assertEqual(report["accuracy"]["ate"]["se3"]["scale"], "fixed_1")
            self.assertEqual(report["accuracy"]["reference_pose_source"], "hot3d_gt_mocap")
            note = json.loads((out / "lerobot" / "meta" / "egodata_export.json").read_text(encoding="utf-8"))
            self.assertEqual(note["pose_source"], "orbslam3_stereo_euroc")
            self.assertEqual(note["hands"], "absent")
            self.assertLess(note["accuracy"]["ate"]["se3"]["rmse_m"], 1e-8)
            info = json.loads((out / "lerobot" / "meta" / "info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["pose_source"], "orbslam3_stereo_euroc")
            self.assertNotIn("accuracy", info)

            from headcam.slam_frames import main as slam_frames_main
            hands = _hands_episode(stamps)
            hands["episode_id"] = "headcam/slam-accuracy"
            camera = root / "camera.json"
            camera.write_text(json.dumps(hands), encoding="utf-8")
            world = root / "world.json"
            slam_code = slam_frames_main([
                "--episode", str(camera),
                "--session", str(session),
                "--tum", str(est_path),
                "--backend", "external_tum",
                "--out", str(world),
            ])
            self.assertEqual(slam_code, 0, world.read_text(encoding="utf-8") if world.is_file() else "no world")
            world_episode = json.loads(world.read_text(encoding="utf-8"))
            world_episode["hand_pose"]["slam"]["accuracy"] = report["accuracy"]
            packaged = root / "pack"
            packaged.mkdir()
            (packaged / "world.json").write_text(json.dumps(world_episode), encoding="utf-8")
            pack_out = root / "pack-out"
            pack_code = export_main([
                "--package-annotated",
                "--episodes", str(packaged),
                "--out", str(pack_out),
            ])
            pack_report = json.loads((pack_out / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(pack_code, 0, pack_report)
            self.assertEqual(pack_report["pose_source"], "orbslam3_stereo_euroc")
            self.assertLess(pack_report["accuracy"]["ate"]["se3"]["rmse_m"], 1e-8)
            pack_note = json.loads((pack_out / "lerobot" / "meta" / "egodata_export.json").read_text(encoding="utf-8"))
            self.assertEqual(pack_note["pose_source"], "orbslam3_stereo_euroc")
            self.assertLess(pack_note["accuracy"]["ate"]["se3"]["rmse_m"], 1e-8)

            relabeled = json.loads(world.read_text(encoding="utf-8"))
            relabeled["hand_pose"]["slam"]["pose_source"] = "hot3d_gt_mocap"
            relabeled["hand_pose"]["slam"]["accuracy"] = report["accuracy"]
            relabel_dir = root / "relabeled"
            relabel_dir.mkdir()
            (relabel_dir / "world.json").write_text(json.dumps(relabeled), encoding="utf-8")
            relabel_out = root / "relabel-out"
            relabel_code = export_main([
                "--package-annotated",
                "--episodes", str(relabel_dir),
                "--out", str(relabel_out),
            ])
            relabel_report = json.loads((relabel_out / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(relabel_code, 0, relabel_report)
            self.assertEqual(relabel_report["pose_source"], "hot3d_gt_mocap")
            self.assertIsNone(relabel_report["accuracy"])

            claimed = json.loads(world.read_text(encoding="utf-8"))
            claimed["hand_pose"]["slam"]["accuracy"] = 0.02
            claimed_dir = root / "claimed"
            claimed_dir.mkdir()
            (claimed_dir / "world.json").write_text(json.dumps(claimed), encoding="utf-8")
            claimed_out = root / "claimed-out"
            claimed_code = export_main([
                "--package-annotated",
                "--episodes", str(claimed_dir),
                "--out", str(claimed_out),
            ])
            claimed_report = json.loads((claimed_out / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(claimed_code, 0, claimed_report)
            self.assertIsNone(claimed_report["accuracy"])


class SlamBatchSummaryTest(unittest.TestCase):
    """多段 ORB-SLAM3 结果的表。失败段 accuracy 是 null，仍要计数。不读 HOT3D 图像。"""

    def _square(self):
        return [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        ]

    def _poses(self, points):
        poses = []
        for point in points:
            pose = np.eye(4)
            pose[:3, 3] = point
            poses.append(pose)
        return poses

    def _measured(self, root):
        from headcam.slam_batch import score_sequence

        points = self._square()
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        shift = np.array([0.2, -0.4, 0.1])
        estimate_points = [rotation.T @ (point - shift) for point in points]
        gt_path = root / "gt.txt"
        est_path = root / "CameraTrajectory.txt"
        gt_path.write_text(
            "# pose_source=hot3d_gt_mocap\n"
            + "".join(_tum_line(stamp, pose) + "\n" for stamp, pose in zip(
                ["0.0", "1.0", "2.0", "3.0"], self._poses(points),
            )),
            encoding="utf-8",
        )
        _write_pose_tum(
            est_path,
            ["0", "1000000000", "2000000000", "3000000000"],
            self._poses(estimate_points),
        )
        log = "\n".join([
            "Creation of new map with id: 0",
            "New Map created with 20 points",
            "*Loop detected",
            "Fail to track local map!",
            "There are 1 maps in the atlas",
            "  Map 0 has 12 KFs",
        ])
        return score_sequence(
            name="P0003_b153ade2",
            sequence_id="P0003_b153ade2",
            participant_id="P0003",
            clip_ids=[480, 491],
            input_frames=6,
            trajectory_path=est_path,
            gt_path=gt_path,
            log_text=log,
            estimate_scale=1e-9,
            gt_scale=1.0,
        )

    def test_failed_sequence_is_counted_with_null_accuracy(self):
        import tempfile

        from headcam.slam_batch import render_markdown, score_sequence, summarize_sequences

        with tempfile.TemporaryDirectory() as tmp:
            measured = self._measured(Path(tmp))
        failed = score_sequence(
            name="P0099_missing",
            sequence_id="P0099_missing",
            participant_id="P0099",
            clip_ids=[1, 2],
            input_frames=300,
            failure="train_quest3/clip-000001.tar 404",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt_path = root / "gt.txt"
            est_path = root / "short.txt"
            pose = np.eye(4)
            gt_path.write_text(
                "# pose_source=hot3d_gt_mocap\n"
                + _tum_line("0.0", pose) + "\n"
                + _tum_line("1.0", pose) + "\n",
                encoding="utf-8",
            )
            _write_pose_tum(est_path, ["0", "1000000000"], [pose, pose])
            short = score_sequence(
                name="P0002_short",
                sequence_id="P0002_short",
                participant_id="P0002",
                clip_ids=[6],
                input_frames=2,
                trajectory_path=est_path,
                gt_path=gt_path,
            )
            summary = summarize_sequences([measured, failed, short])
            text = render_markdown(summary)
        self.assertEqual(measured["status"], "ok")
        self.assertEqual(measured["tracked_frames"], 4)
        self.assertEqual(measured["lost_frames"], 2)
        self.assertEqual(measured["reinits"], 0)
        self.assertEqual(measured["maps"], 1)
        self.assertLess(measured["accuracy"]["ate"]["se3"]["rmse_m"], 1e-8)
        self.assertIsNone(failed["accuracy"])
        self.assertEqual(failed["status"], "failed")
        self.assertIsNone(short["accuracy"])
        self.assertEqual(summary["n_sequences"], 3)
        self.assertEqual(summary["n_failed"], 1)
        self.assertEqual(summary["n_unmeasured"], 1)
        self.assertEqual(summary["n_measured"], 1)
        self.assertEqual(len(summary["sequences"]), 3)
        self.assertIsNone(summary["sequences"][1]["accuracy"])
        self.assertEqual(summary["sequences"][1]["name"], "P0099_missing")
        self.assertEqual(summary["ate_se3_rmse_m"]["n"], 1)
        self.assertLess(summary["ate_se3_rmse_m"]["median"], 1e-8)
        self.assertEqual(summary["rpe_5s"]["n"], 0)
        self.assertNotIn("median", summary["rpe_5s"])
        self.assertIn("不进 RMSE 汇总", summary["aggregate_note"])
        self.assertIn("P0099_missing", text)
        self.assertIn("404", text)
        self.assertIn("不进 RMSE 汇总", text)
        self.assertIn("left out of the RMSE aggregate", text)

    def test_failed_status_drops_a_stuffed_accuracy_from_the_aggregate(self):
        import tempfile

        from headcam.slam_batch import summarize_sequences

        with tempfile.TemporaryDirectory() as tmp:
            measured = self._measured(Path(tmp))
        stuffed = dict(measured)
        stuffed["name"] = "stuffed"
        stuffed["status"] = "failed"
        stuffed["failure"] = "stereo_euroc 退出码 139"
        summary = summarize_sequences([stuffed])
        self.assertEqual(summary["n_sequences"], 1)
        self.assertEqual(summary["n_failed"], 1)
        self.assertEqual(summary["n_measured"], 0)
        self.assertIsNone(summary["sequences"][0]["accuracy"])
        self.assertIsNone(summary["ate_se3_rmse_m"])

    def test_headset_header_on_the_estimate_is_not_a_slam_result(self):
        import tempfile

        from headcam.slam_batch import score_sequence

        pose = np.eye(4)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt_path = root / "gt.txt"
            est_path = root / "est.txt"
            body = "".join(_tum_line(stamp, pose) + "\n" for stamp in ("0.0", "1.0", "2.0", "3.0"))
            gt_path.write_text("# pose_source=hot3d_gt_mocap\n" + body, encoding="utf-8")
            est_path.write_text("# pose_source=hot3d_gt_mocap\n" + body, encoding="utf-8")
            labeled = score_sequence(
                name="copied-gt",
                sequence_id="copied-gt",
                participant_id="P0001",
                clip_ids=[0],
                input_frames=4,
                trajectory_path=est_path,
                gt_path=gt_path,
                estimate_scale=1.0,
            )
            bare = root / "bare.txt"
            bare.write_text(body, encoding="utf-8")
            (root / "no_header_gt.txt").write_text(body, encoding="utf-8")
            missing_header = score_sequence(
                name="no-header",
                sequence_id="no-header",
                participant_id="P0001",
                clip_ids=[0],
                input_frames=4,
                trajectory_path=bare,
                gt_path=root / "no_header_gt.txt",
                estimate_scale=1.0,
            )
        self.assertEqual(labeled["status"], "failed")
        self.assertIsNone(labeled["accuracy"])
        self.assertIn("头戴真值", labeled["failure"])
        self.assertIsNone(missing_header["accuracy"])
        self.assertIn("pose_source", missing_header["failure"])

    def test_summarize_cli_writes_json_and_markdown(self):
        import tempfile

        from headcam.slam_batch import main, score_sequence

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            measured = self._measured(root)
            failed = score_sequence(
                name="P0099_missing",
                sequence_id="P0099_missing",
                participant_id="P0099",
                clip_ids=[3],
                input_frames=10,
                failure="下载失败",
            )
            (root / "ok.json").write_text(json.dumps(measured), encoding="utf-8")
            (root / "bad.json").write_text(json.dumps(failed), encoding="utf-8")
            code = main([
                "summarize",
                "--records", str(root / "bad.json"), str(root / "ok.json"),
                "--json", str(root / "batch.json"),
                "--md", str(root / "batch.md"),
            ])
            report = json.loads((root / "batch.json").read_text(encoding="utf-8"))
            text = (root / "batch.md").read_text(encoding="utf-8")
        self.assertEqual(code, 0)
        self.assertEqual(report["n_sequences"], 2)
        self.assertEqual(report["sequences"][0]["name"], "P0099_missing")
        self.assertIsNone(report["sequences"][0]["accuracy"])
        self.assertEqual(report["n_measured"], 1)
        self.assertIn("P0003_b153ade2", text)
        self.assertIn("下载失败", text)

    def test_train_index_uses_contiguous_runs(self):
        from headcam.slam_batch import quest3_split_runs

        step = 33333333
        definitions = {
            "0": {
                "sequence_id": "P0002_a",
                "device": "Quest3",
                "per_frame_timestamps_ns": [{"1201-1": i * step, "1201-2": i * step} for i in range(4)],
            },
            "1": {
                "sequence_id": "P0002_a",
                "device": "Quest3",
                "per_frame_timestamps_ns": [{"1201-1": (4 + i) * step, "1201-2": (4 + i) * step} for i in range(4)],
            },
            "2": {
                "sequence_id": "P0002_a",
                "device": "Quest3",
                "per_frame_timestamps_ns": [{"1201-1": (4 + i) * step + 2_000_000_000, "1201-2": 0} for i in range(4)],
            },
            "9": {
                "sequence_id": "Aria_x",
                "device": "Aria",
                "per_frame_timestamps_ns": [{"1201-1": i * step, "1201-2": i * step} for i in range(4)],
            },
        }
        splits = {"train": {"Quest3": [2, 0, 1], "Aria": [9]}}
        runs = quest3_split_runs(definitions, splits)
        by_ids = {tuple(run["clip_ids"]): run for run in runs}
        self.assertEqual(by_ids[(0, 1)]["n_frames"], 8)
        self.assertIn((2,), by_ids)
        self.assertNotIn((9,), by_ids)

    def test_next_unused_run_is_the_second_longest_per_participant(self):
        from headcam.slam_batch import next_unused_runs, run_span

        def run(sequence_id, clips, frames):
            return {
                "sequence_id": sequence_id,
                "clip_ids": clips,
                "n_frames": frames,
            }

        runs = [
            run("P0002_long", [6, 7], 1050),
            run("P0002_next", [68, 69], 900),
            run("P0002_short", [13], 150),
            run("P0018_same", [1125, 1126], 750),
            run("P0018_same", [1121, 1122], 600),
            run("P0003_only", [1], 300),
        ]
        used = [run_span(runs[0]), run_span(runs[3])]
        chosen = next_unused_runs(runs, used, per_participant=1)
        spans = [run_span(item) for item in chosen]
        self.assertEqual(spans, [
            ("P0002_next", 68, 69),
            ("P0003_only", 1, 1),
            ("P0018_same", 1121, 1122),
        ])


class TrajectoryTimelineTest(unittest.TestCase):
    """逐帧误差和接缝、速度的对照。合成轨迹，不读 HOT3D，也不把真值叫做 SLAM。"""

    def _line_poses(self, steps):
        poses = []
        x = 0.0
        for step in steps:
            x += step
            pose = np.eye(4)
            pose[0, 3] = x
            poses.append(pose)
        return poses

    def test_constant_speed_has_no_pearson_and_a_seam_spike_stays_at_the_seam(self):
        from headcam.trajectory_qc import (
            motion_series,
            near_seam_mask,
            pearson,
            sse_share_worst_fraction,
            summarize_timeline,
        )

        times = [float(i) for i in range(8)]
        # 1 米的步长在二进制里是精确的。用 0.1 累加会留下一点点起伏，Pearson 就不该是 null。
        poses = self._line_poses([0.0] + [1.0] * 7)
        linear, angular = motion_series(times, poses)
        self.assertTrue(np.isnan(linear[0]))
        self.assertTrue(np.allclose(linear[1:], 1.0))
        self.assertTrue(np.allclose(angular[1:], 0.0))
        self.assertIsNone(pearson(linear, linear))
        errors = np.zeros(8)
        errors[3] = 0.5
        mask = near_seam_mask(8, [4, 4], 0)
        self.assertEqual(mask.tolist(), [False, False, False, True, True, False, False, False])
        report = summarize_timeline(
            times, errors, poses, clip_lengths=[4, 4], clip_ids=[6, 7], seam_radius=0,
            keyframe_times=[0.0, 3.0],
        )
        self.assertEqual(report["max_error"]["index"], 3)
        self.assertEqual(report["max_error"]["clip_id"], 6)
        self.assertEqual(report["max_error"]["offset_in_clip"], 3)
        self.assertGreater(report["seams"]["near"]["rmse_m"], report["seams"]["away"]["rmse_m"])
        self.assertIsNone(report["headset_motion"]["error_vs_linear"]["pearson"])
        self.assertIsNone(report["tracked_features"])
        self.assertEqual(report["tail_runs"][0]["n"], 1)
        self.assertEqual(report["tail_runs"][0]["t0_s"], 3.0)
        share = sse_share_worst_fraction(errors, 0.1)
        self.assertEqual(share["n_worst"], 1)
        self.assertAlmostEqual(share["share"], 1.0)
        self.assertIn("不是原因", report["note"])
        self.assertIn("不是 SLAM", report["note"])

    def test_error_tracks_speed_when_the_spike_is_not_a_seam(self):
        from headcam.trajectory_qc import motion_series, summarize_timeline

        times = [float(i) for i in range(9)]
        steps = [0.0, 0.01, 0.02, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24]
        poses = self._line_poses(steps)
        linear, _angular = motion_series(times, poses)
        report = summarize_timeline(
            times, linear, poses, clip_lengths=[9], seam_radius=2, bin_s=3.0,
        )
        self.assertGreater(report["headset_motion"]["error_vs_linear"]["pearson"], 0.99)
        high = report["headset_motion"]["error_vs_linear"]["tercile"]["high"]["rmse_m"]
        low = report["headset_motion"]["error_vs_linear"]["tercile"]["low"]["rmse_m"]
        self.assertGreater(high, low)
        self.assertIsNone(report["seams"]["near"])
        self.assertEqual(len(report["bins_s"]), 3)

    def test_blur_and_keyframe_gap_are_measured_on_synthetic_inputs(self):
        from headcam.trajectory_qc import brightness_and_blur, seconds_since_keyframe, summarize_timeline

        flat = np.full((8, 8), 40, dtype=np.uint8)
        sharp = np.zeros((8, 8), dtype=np.uint8)
        sharp[::2, ::2] = 255
        flat_stats = brightness_and_blur(flat)
        sharp_stats = brightness_and_blur(sharp)
        self.assertAlmostEqual(flat_stats["brightness"], 40.0)
        self.assertAlmostEqual(flat_stats["laplacian_var"], 0.0)
        self.assertGreater(sharp_stats["laplacian_var"], flat_stats["laplacian_var"])
        gaps = seconds_since_keyframe([0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 3.0])
        self.assertTrue(np.allclose(gaps, [0.0, 1.0, 2.0, 0.0, 1.0]))
        poses = self._line_poses([0.0, 0.1, 0.1, 0.1, 0.1])
        report = summarize_timeline(
            [0.0, 1.0, 2.0, 3.0, 4.0],
            gaps,
            poses,
            clip_lengths=[5],
            keyframe_times=[0.0, 3.0],
            brightness=np.array([10.0, 10.0, 80.0, 10.0, 10.0]),
            blur=np.array([5.0, 4.0, 1.0, 4.0, 5.0]),
        )
        self.assertGreater(report["keyframes"]["error_vs_seconds_since_keyframe"]["pearson"], 0.99)
        self.assertLess(report["images"]["error_vs_laplacian_var"]["pearson"], -0.5)

    def test_headset_header_on_the_estimate_is_not_a_timeline(self):
        import tempfile

        from headcam.trajectory_qc import diagnose_files, main

        pose = np.eye(4)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            body = "".join(_tum_line(stamp, pose) + "\n" for stamp in ("0.0", "1.0", "2.0", "3.0"))
            gt = root / "gt.txt"
            est = root / "est.txt"
            gt.write_text("# pose_source=hot3d_gt_mocap\n" + body, encoding="utf-8")
            est.write_text("# pose_source=hot3d_gt_mocap\n" + body, encoding="utf-8")
            refused = diagnose_files(est, gt, estimate_scale=1.0, gt_scale=1.0)
            shifted = []
            for index, stamp in enumerate(("0.0", "1.0", "2.0", "3.0", "4.0")):
                item = np.eye(4)
                item[0, 3] = 0.2 * index
                shifted.append(item)
            spike = [item.copy() for item in shifted]
            spike[3] = spike[3].copy()
            spike[3][2, 3] = 0.3
            gt_path = root / "line_gt.txt"
            est_path = root / "line_est.txt"
            gt_path.write_text(
                "# pose_source=hot3d_gt_mocap\n"
                + "".join(_tum_line(stamp, item) + "\n" for stamp, item in zip(
                    ("0.0", "1.0", "2.0", "3.0", "4.0"), shifted,
                )),
                encoding="utf-8",
            )
            est_path.write_text(
                "# pose_source=orbslam3_stereo_euroc\n"
                + "".join(_tum_line("%d" % int(float(stamp) * 1e9), item) + "\n" for stamp, item in zip(
                    ("0.0", "1.0", "2.0", "3.0", "4.0"), spike,
                )),
                encoding="utf-8",
            )
            out = root / "timeline.json"
            code = main([
                "--estimate", str(est_path),
                "--gt", str(gt_path),
                "--estimate-scale", "1e-9",
                "--clip-lengths", "5",
                "--seam-radius", "0",
                "--out", str(out),
            ])
            report = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(refused["status"], "failed")
        self.assertIsNone(refused["rmse_m"])
        self.assertIn("头戴真值", refused["failure"])
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["max_error"]["index"], 3)
        self.assertGreater(report["rmse_m"], 0.0)
        self.assertLess(report["median_m"], report["max_m"])

    def test_aligned_rmse_matches_trajectory_ate(self):
        import tempfile

        from headcam.hot3d_check import trajectory_ate
        from headcam.trajectory_qc import diagnose_files

        points = [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        ]
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        shift = np.array([0.2, -0.4, 0.1])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt_poses = []
            est_poses = []
            for point in points:
                gt_pose = np.eye(4)
                gt_pose[:3, 3] = point
                est_pose = np.eye(4)
                est_pose[:3, 3] = rotation.T @ (point - shift)
                gt_poses.append(gt_pose)
                est_poses.append(est_pose)
            gt_path = root / "gt.txt"
            est_path = root / "est.txt"
            gt_path.write_text(
                "# pose_source=hot3d_gt_mocap\n"
                + "".join(_tum_line(stamp, pose) + "\n" for stamp, pose in zip(
                    ("0.0", "1.0", "2.0", "3.0"), gt_poses,
                )),
                encoding="utf-8",
            )
            _write_pose_tum(
                est_path,
                ["0", "1000000000", "2000000000", "3000000000"],
                est_poses,
            )
            ate = trajectory_ate(est_path, gt_path, estimate_scale=1e-9, gt_scale=1.0, alignment="se3")
            report = diagnose_files(est_path, gt_path, estimate_scale=1e-9, gt_scale=1.0, clip_lengths=[4])
        self.assertAlmostEqual(report["rmse_m"], ate["rmse_m"])
        self.assertLess(report["rmse_m"], 1e-8)

    def test_a_constant_shift_keeps_stepwise_motion_and_a_tight_offset(self):
        import tempfile

        from headcam.trajectory_qc import diagnose_files, error_vector_offset, stepwise_relative_error

        tight = error_vector_offset(np.tile(np.array([0.06, 0.0, 0.0]), (6, 1)))
        self.assertAlmostEqual(tight["mean_norm_m"], 0.06)
        self.assertLess(tight["rms_component_std_m"], 1e-12)
        poses = self._line_poses([0.0] + [1.0] * 7)
        same = stepwise_relative_error(poses, poses)
        self.assertLess(same["trans_rmse_m"], 1e-12)
        self.assertLess(same["rot_rmse_deg"], 1e-8)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt_poses = self._line_poses([0.0] + [1.0] * 29)
            est_poses = [pose.copy() for pose in gt_poses]
            for pose in est_poses[:4]:
                pose[2, 3] += 0.5
            stamps = ["%.1f" % float(i) for i in range(30)]
            ns = ["%d" % int(i * 1e9) for i in range(30)]
            gt_path = root / "gt.txt"
            est_path = root / "est.txt"
            gt_path.write_text(
                "# pose_source=hot3d_gt_mocap\n"
                + "".join(_tum_line(stamp, pose) + "\n" for stamp, pose in zip(stamps, gt_poses)),
                encoding="utf-8",
            )
            est_path.write_text(
                "# pose_source=orbslam3_stereo_euroc\n"
                + "".join(_tum_line(stamp, pose) + "\n" for stamp, pose in zip(ns, est_poses)),
                encoding="utf-8",
            )
            report = diagnose_files(
                est_path, gt_path, estimate_scale=1e-9, gt_scale=1.0, clip_lengths=[30],
            )
        self.assertEqual(report["status"], "ok")
        self.assertGreater(report["rmse_m"], 0.01)
        self.assertTrue(report["tail_runs"])
        self.assertLess(report["tail_runs"][0]["t0_s"], 1e-9)
        opening = report["tail_geometry"][0]
        self.assertGreater(opening["offset"]["mean_norm_m"], 0.2)
        self.assertLess(opening["offset"]["rms_component_std_m"], 0.02)
        self.assertLess(opening["step"]["trans_rmse_m"], 1e-8)
        self.assertLess(report["outside_tail_step"]["trans_rmse_m"], 1e-8)
        self.assertEqual(
            report["outside_tail"]["n"] + sum(run["n"] for run in report["tail_runs"]),
            report["n"],
        )
        self.assertLess(report["outside_tail"]["rmse_m"], report["tail_runs"][0]["rmse_m"])
        self.assertAlmostEqual(report["motion_in_tail"]["linear_speed_m_s"]["median"], 1.0, places=6)
        self.assertAlmostEqual(report["motion_outside_tail"]["linear_speed_m_s"]["median"], 1.0, places=6)
        self.assertGreater(
            report["motion_outside_tail"]["linear_speed_m_s"]["max"],
            report["motion_in_tail"]["linear_speed_m_s"]["max"] - 1e-9,
        )


class BatchAccuracyExportTest(unittest.TestCase):
    """batch 行里的实测 ATE 可以进包。单位阵、真值冒充、手写数字、对不上的 clip 都不进。"""

    BATCH = ROOT / "docs" / "headcam" / "orbslam3_quest3_batch.json"

    def _square(self):
        return [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        ]

    def _pair(self, root, points=None):
        points = self._square() if points is None else points
        stamps = [index / 30.0 for index in range(len(points))]
        shift = np.array([0.4, -0.2, 0.1])
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        gt_poses = []
        est_poses = []
        for point in points:
            gt = np.eye(4)
            gt[:3, 3] = point
            est = np.eye(4)
            est[:3, :3] = rotation
            est[:3, 3] = rotation @ point + shift
            gt_poses.append(gt)
            est_poses.append(est)
        gt_path = root / "gt.txt"
        est_path = root / "slam.txt"
        gt_path.write_text(
            "# pose_source=hot3d_gt_mocap\n"
            + "".join(_tum_line(stamp, pose) + "\n" for stamp, pose in zip(stamps, gt_poses)),
            encoding="utf-8",
        )
        est_path.write_text(
            "# pose_source=orbslam3_stereo_euroc\n"
            + "".join(_tum_line(stamp, pose) + "\n" for stamp, pose in zip(stamps, est_poses)),
            encoding="utf-8",
        )
        return stamps, est_path, gt_path, est_poses

    def _record(self, accuracy, frames, name="synth_seq", status="ok", failure=None):
        return {
            "name": name,
            "sequence_id": name,
            "pose_source": "orbslam3_stereo_euroc",
            "reference_pose_source": "hot3d_gt_mocap",
            "clip_ids": [1],
            "input_frames": frames,
            "tracked_frames": frames,
            "trajectory_rows": frames,
            "estimate_time_scale": 1.0,
            "gt_time_scale": 1.0,
            "status": status,
            "failure": failure,
            "accuracy": accuracy,
        }

    def _write_batch(self, path, record, median=123.0):
        path.write_text(json.dumps({
            "ate_se3_rmse_m": {"n": 1, "min": 0.0, "median": median, "max": median},
            "sequences": [record],
        }), encoding="utf-8")

    def test_real_batch_clip_span_must_match_before_accuracy_is_used(self):
        from headcam.batch_export import load_batch, match_sequence, measured_accuracy_from_record

        batch = load_batch(self.BATCH)
        packaged = [
            ("P0002_210fc0da", [0]),
            ("P0003_5e1a34f8", [300]),
            ("P0010_3224a967", [700]),
            ("P0017_d2495f49", [1100]),
        ]
        for sequence_id, clips in packaged:
            self.assertIsNone(match_sequence(batch, sequence_id, clips))
        row = match_sequence(batch, "P0010_3224a967", list(range(706, 718)))
        accuracy, note = measured_accuracy_from_record(row)
        self.assertEqual(accuracy["ate"]["se3"]["rmse_m"], row["accuracy"]["ate"]["se3"]["rmse_m"])
        self.assertEqual(accuracy["ate"]["se3"]["rmse_m"], 0.008205109394453782)
        self.assertNotEqual(accuracy["ate"]["se3"]["rmse_m"], batch["ate_se3_rmse_m"]["median"])
        self.assertNotEqual(accuracy["ate"]["se3"]["rmse_m"], batch["ate_se3_rmse_m"]["min"])
        self.assertIn("P0010_3224a967", note)
        short = match_sequence(batch, "P0017_1a07e9e7", [1036, 1037, 1038, 1039])
        short_accuracy, _short_note = measured_accuracy_from_record(short)
        self.assertEqual(short_accuracy["ate"]["se3"]["rmse_m"], 0.0008185619223686899)

    def test_identity_and_ground_truth_copies_are_refused(self):
        import tempfile

        from headcam.batch_export import export_from_batch, poses_all_identity
        from headcam.export_mvp import main as export_main
        from headcam.hot3d_check import trajectory_accuracy

        eye = np.eye(4)
        moved = np.eye(4)
        moved[0, 3] = 1.0
        self.assertFalse(poses_all_identity([eye, moved]))
        self.assertTrue(poses_all_identity([eye, eye]))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stamps, est_path, gt_path, _est_poses = self._pair(root)
            measured = trajectory_accuracy(
                est_path, gt_path,
                estimate_pose_source="orbslam3_stereo_euroc",
                reference_pose_source="hot3d_gt_mocap",
            )
            batch_path = root / "batch.json"
            self._write_batch(batch_path, self._record(measured, len(stamps)))
            session = _write_session(root / "sess", stamps)
            identity = root / "identity.txt"
            identity.write_text(
                "# pose_source=orbslam3_stereo_euroc\n"
                + "".join(_tum_line(stamp, np.eye(4)) + "\n" for stamp in stamps),
                encoding="utf-8",
            )
            out = root / "out-identity"
            with self.assertRaises(ValueError) as raised:
                export_from_batch(batch_path, "synth_seq", session, identity, out)
            self.assertIn("单位阵", str(raised.exception))
            self.assertFalse(out.exists())
            code = export_main([
                "--batch", str(batch_path),
                "--sequence", "synth_seq",
                "--session", str(session),
                "--tum", str(identity),
                "--out", str(out),
                "--skip-hands",
            ])
            self.assertEqual(code, 1)
            self.assertFalse((out / "lerobot").exists())
            self.assertFalse((out / "report.json").exists())

            copied = root / "copied.txt"
            body = "\n".join(
                line for line in gt_path.read_text(encoding="utf-8").splitlines() if not line.startswith("#")
            )
            copied.write_text("# pose_source=orbslam3_stereo_euroc\n" + body + "\n", encoding="utf-8")
            with self.assertRaises(ValueError) as copied_raised:
                export_from_batch(
                    batch_path, "synth_seq", session, copied, root / "out-copied", gt_tum_path=gt_path,
                )
            self.assertIn("头戴真值", str(copied_raised.exception))
            self.assertFalse((root / "out-copied").exists())

            labeled = root / "labeled.txt"
            labeled.write_text(
                "# pose_source=hot3d_gt_mocap\n"
                + "".join(_tum_line(stamp, pose) + "\n" for stamp, pose in zip(stamps, [moved, moved, moved, moved])),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as labeled_raised:
                export_from_batch(batch_path, "synth_seq", session, labeled, root / "out-labeled")
            self.assertIn("pose_source", str(labeled_raised.exception))

            hands = export_main([
                "--batch", str(batch_path),
                "--sequence", "synth_seq",
                "--session", str(session),
                "--tum", str(est_path),
                "--out", str(root / "out-hands"),
            ])
            self.assertEqual(hands, 1)
            self.assertFalse((root / "out-hands").exists())

    def test_package_takes_batch_measurement_not_the_aggregate_or_a_float(self):
        import shutil
        import tempfile

        try:
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("未安装 pyarrow，跳过 batch accuracy 的 LeRobot 包")
        if shutil.which("ffmpeg") is None:
            self.skipTest("没有 ffmpeg，跳过 batch accuracy 的 LeRobot 包")

        from headcam.batch_export import export_from_batch
        from headcam.export_mvp import main as export_main
        from headcam.hot3d_check import trajectory_accuracy

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stamps, est_path, gt_path, est_poses = self._pair(root)
            measured = trajectory_accuracy(
                est_path, gt_path,
                estimate_pose_source="orbslam3_stereo_euroc",
                reference_pose_source="hot3d_gt_mocap",
            )
            self.assertLess(measured["ate"]["se3"]["rmse_m"], 1e-8)
            batch_path = root / "batch.json"
            self._write_batch(batch_path, self._record(measured, len(stamps)), median=123.0)
            session = _write_session(root / "sess", stamps)
            out = root / "out"
            code = export_main([
                "--batch", str(batch_path),
                "--sequence", "synth_seq",
                "--session", str(session),
                "--tum", str(est_path),
                "--gt-tum", str(gt_path),
                "--out", str(out),
                "--skip-hands",
            ])
            report = json.loads((out / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(code, 0, report)
            self.assertEqual(report["pose_source"], "orbslam3_stereo_euroc")
            self.assertEqual(report["accuracy_source"], "batch")
            self.assertEqual(report["accuracy"]["ate"]["se3"]["rmse_m"], measured["ate"]["se3"]["rmse_m"])
            self.assertNotEqual(report["accuracy"]["ate"]["se3"]["rmse_m"], 123.0)
            self.assertEqual(
                report["accuracy"]["ate"]["se3"]["rotation_row_major"],
                measured["ate"]["se3"]["rotation_row_major"],
            )
            note = json.loads((out / "lerobot" / "meta" / "egodata_export.json").read_text(encoding="utf-8"))
            self.assertEqual(note["pose_source"], "orbslam3_stereo_euroc")
            self.assertEqual(note["hands"], "absent")
            self.assertEqual(note["accuracy"]["ate"]["se3"]["rmse_m"], measured["ate"]["se3"]["rmse_m"])
            info = json.loads((out / "lerobot" / "meta" / "info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["pose_source"], "orbslam3_stereo_euroc")
            self.assertNotIn("accuracy", info)
            table = pq.read_table(out / "lerobot" / "data" / "chunk-000" / "file-000.parquet")
            pose = table.column("observation.pose")[0].as_py()
            self.assertAlmostEqual(pose[0], float(est_poses[0][0, 3]), places=6)
            self.assertAlmostEqual(pose[1], float(est_poses[0][1, 3]), places=6)
            self.assertAlmostEqual(pose[2], float(est_poses[0][2, 3]), places=6)
            self.assertFalse(np.allclose(pose[:3], 0.0))

            stuffed = json.loads(json.dumps(measured))
            stuffed["ate"]["se3"]["rmse_m"] = 0.5
            disagree_path = root / "disagree.json"
            self._write_batch(disagree_path, self._record(stuffed, len(stamps)))
            disagree_out = root / "disagree"
            disagree = export_from_batch(
                disagree_path, "synth_seq", session, est_path, disagree_out, gt_tum_path=gt_path,
            )
            self.assertEqual(disagree["pose_source"], "orbslam3_stereo_euroc")
            self.assertIsNone(disagree["accuracy"])
            self.assertEqual(disagree["accuracy_source"], "unmeasured")
            self.assertIn("不一致", disagree["accuracy_note"])
            saved = json.loads((disagree_out / "report.json").read_text(encoding="utf-8"))
            self.assertIsNone(saved["accuracy"])
            self.assertNotIn("0.5", json.dumps(saved["accuracy"]))

            bare = json.loads(json.dumps(self._record(0.02, len(stamps))))
            bare_path = root / "bare.json"
            self._write_batch(bare_path, bare)
            bare_out = root / "bare"
            bare_report = export_from_batch(bare_path, "synth_seq", session, est_path, bare_out)
            self.assertEqual(bare_report["pose_source"], "orbslam3_stereo_euroc")
            self.assertIsNone(bare_report["accuracy"])
            self.assertIn("手写数字", bare_report["accuracy_note"])

            missing = self._record(None, len(stamps), status="unmeasured", failure="对上的不同时间戳少于 3，不算 ATE。")
            missing_path = root / "missing.json"
            self._write_batch(missing_path, missing)
            missing_out = root / "missing"
            missing_report = export_from_batch(
                missing_path, "synth_seq", session, est_path, missing_out, gt_tum_path=gt_path,
            )
            self.assertEqual(missing_report["accuracy_source"], "measured_this_run")
            self.assertLess(missing_report["accuracy"]["ate"]["se3"]["rmse_m"], 1e-8)
            self.assertIn("这次用真值测到了", missing_report["accuracy_note"])

            null_only = export_from_batch(missing_path, "synth_seq", session, est_path, root / "null-only")
            self.assertIsNone(null_only["accuracy"])
            self.assertEqual(null_only["accuracy_source"], "unmeasured")
            self.assertIn("没有测到 ATE", null_only["accuracy_note"])

            wrong_frames = self._record(measured, 4)
            wrong_frames["input_frames"] = 1800
            wrong_path = root / "wrong.json"
            self._write_batch(wrong_path, wrong_frames)
            with self.assertRaises(ValueError) as wrong:
                export_from_batch(wrong_path, "synth_seq", session, est_path, root / "wrong")
            self.assertIn("input_frames", str(wrong.exception))
            self.assertFalse((root / "wrong").exists())

    def test_origin_first_frame_is_kept_and_a_lost_frame_stays_null(self):
        import shutil
        import tempfile

        try:
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("未安装 pyarrow，跳过丢失帧的 LeRobot 包")
        if shutil.which("ffmpeg") is None:
            self.skipTest("没有 ffmpeg，跳过丢失帧的 LeRobot 包")

        from headcam.batch_export import export_from_batch
        from headcam.hot3d_check import trajectory_accuracy

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            points = self._square()
            stamps = [index / 30.0 for index in range(4)]
            shift = np.array([0.2, 0.0, 0.0])
            est_poses = []
            gt_poses = []
            for point in points:
                est = np.eye(4)
                est[:3, 3] = point
                gt = np.eye(4)
                gt[:3, 3] = point + shift
                est_poses.append(est)
                gt_poses.append(gt)
            self.assertTrue(np.allclose(est_poses[0], np.eye(4)))
            est_path = root / "slam.txt"
            gt_path = root / "gt.txt"
            est_path.write_text(
                "# pose_source=orbslam3_stereo_euroc\n"
                + "".join(_tum_line(stamp, pose) + "\n" for stamp, pose in zip(stamps, est_poses)),
                encoding="utf-8",
            )
            gt_path.write_text(
                "# pose_source=hot3d_gt_mocap\n"
                + "".join(_tum_line(stamp, pose) + "\n" for stamp, pose in zip(stamps, gt_poses)),
                encoding="utf-8",
            )
            measured = trajectory_accuracy(
                est_path, gt_path,
                estimate_pose_source="orbslam3_stereo_euroc",
                reference_pose_source="hot3d_gt_mocap",
            )
            batch_path = root / "batch.json"
            self._write_batch(batch_path, self._record(measured, 4))
            session = _write_session(root / "sess", stamps)
            report = export_from_batch(
                batch_path, "synth_seq", session, est_path, root / "origin", gt_tum_path=gt_path,
            )
            self.assertEqual(report["pose_source"], "orbslam3_stereo_euroc")
            self.assertLess(report["accuracy"]["ate"]["se3"]["rmse_m"], 1e-8)

            short_stamps = stamps[:3]
            short_path = root / "short.txt"
            short_path.write_text(
                "# pose_source=orbslam3_stereo_euroc\n"
                + "".join(_tum_line(stamp, pose) + "\n" for stamp, pose in zip(short_stamps, est_poses[:3])),
                encoding="utf-8",
            )
            short_gt = root / "short_gt.txt"
            short_gt.write_text(
                "# pose_source=hot3d_gt_mocap\n"
                + "".join(_tum_line(stamp, pose) + "\n" for stamp, pose in zip(short_stamps, gt_poses[:3])),
                encoding="utf-8",
            )
            short_measured = trajectory_accuracy(
                short_path, short_gt,
                estimate_pose_source="orbslam3_stereo_euroc",
                reference_pose_source="hot3d_gt_mocap",
            )
            session_stamps = short_stamps + [short_stamps[-1] + 10.0]
            record = self._record(short_measured, 3)
            record["input_frames"] = 4
            record["tracked_frames"] = 3
            record["trajectory_rows"] = 3
            lost_batch = root / "lost.json"
            self._write_batch(lost_batch, record)
            lost_session = _write_session(root / "lost-sess", session_stamps)
            lost = export_from_batch(lost_batch, "synth_seq", lost_session, short_path, root / "lost")
            self.assertEqual(lost["pose_source"], "orbslam3_stereo_euroc")
            self.assertEqual(lost["accuracy"]["ate"]["se3"]["rmse_m"], short_measured["ate"]["se3"]["rmse_m"])
            self.assertIn("丢失帧不补单位阵", lost["accuracy_note"])
            episode = json.loads((root / "lost" / "episodes" / "session.json").read_text(encoding="utf-8"))
            slam = episode["hand_pose"]["slam"]
            self.assertEqual(slam["pose_status"][-1], "tracking_lost")
            self.assertIsNone(slam["T_world_cam"][-1])
            self.assertIsNotNone(slam["T_world_cam"][0])
            table = pq.read_table(root / "lost" / "lerobot" / "data" / "chunk-000" / "file-000.parquet")
            self.assertIsNone(table.column("observation.pose")[-1].as_py())


if __name__ == "__main__":
    unittest.main()
