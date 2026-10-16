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


class PrepareSessionGuardTest(unittest.TestCase):
    def test_refuses_to_nest_work_copy_inside_source(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_session(Path(tmp) / "sess", [0.0])
            slam = run_slam(root, backend="none")
            with self.assertRaises(ValueError):
                prepare_hand_session(root, root / "nested", slam)


if __name__ == "__main__":
    unittest.main()
