# -*- coding: utf-8 -*-
"""合成双目：已知三维点投影到左右相机，检查三角化、尺度和世界系。"""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import convert_headcam  # noqa: E402
from egodata.qc import qc_episode  # noqa: E402
from egodata.schema import validate_episode  # noqa: E402
from headcam.hand_pose import (  # noqa: E402
    HaMeRBackend,
    MediaPipeHandsBackend,
    WiLoRBackend,
    apply_scale,
    associate_camera_poses,
    correct_monocular_with_stereo,
    estimate_scale,
    get_backend,
    hamer_available,
    load_calibration,
    mean_wrist_error_m,
    mediapipe_available,
    project_pinhole,
    transform_points,
    world_to_camera,
    triangulate_pair,
    wilor_available,
    wrist_rotation,
    write_calibration_yaml,
)


def _K(fx=200.0, cx=160.0, cy=120.0):
    return np.array([[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _calib(rotation=None, translation=None):
    return {
        "image_width": 320,
        "image_height": 240,
        "K_left": _K(),
        "K_right": _K(),
        "dist_left": np.zeros(5),
        "dist_right": np.zeros(5),
        "R": np.eye(3) if rotation is None else np.asarray(rotation, dtype=np.float64),
        "T": np.array([-0.08, 0.0, 0.0]) if translation is None else np.asarray(translation, dtype=np.float64),
    }


def _project_independent(points, K, rotation=None, translation=None):
    """测试自己的投影，不调用被测的 project_pinhole。"""
    rotation = np.eye(3) if rotation is None else np.asarray(rotation, dtype=np.float64)
    translation = np.zeros(3) if translation is None else np.asarray(translation, dtype=np.float64)
    pixels = []
    for point in np.asarray(points, dtype=np.float64):
        camera = rotation @ point + translation
        pixels.append([
            float(K[0, 0] * camera[0] / camera[2] + K[0, 2]),
            float(K[1, 1] * camera[1] / camera[2] + K[1, 2]),
        ])
    return np.asarray(pixels, dtype=np.float64)


def _cv_triangulate(uv_left, uv_right, calib):
    import cv2
    K_left = calib["K_left"]
    K_right = calib["K_right"]
    rotation = calib["R"]
    translation = np.asarray(calib["T"], dtype=np.float64).reshape(3, 1)
    P_left = K_left @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P_right = K_right @ np.hstack([rotation, translation])
    homogeneous = cv2.triangulatePoints(
        P_left, P_right, np.asarray(uv_left, dtype=np.float64).T, np.asarray(uv_right, dtype=np.float64).T,
    )
    return (homogeneous[:3] / homogeneous[3]).T


def _hand(wrist):
    wrist = np.asarray(wrist, dtype=np.float64)
    joints = np.zeros((21, 3), dtype=np.float64)
    for index in range(21):
        joints[index] = wrist + np.array([
            0.01 * ((index % 5) - 2),
            0.004 * (index % 3),
            0.008 * (index // 7),
        ])
    joints[0] = wrist
    joints[5] = wrist + np.array([0.04, 0.0, 0.01])
    joints[17] = wrist + np.array([-0.03, 0.012, 0.01])
    return joints


class StereoTriangulationTest(unittest.TestCase):
    def test_rectified_pair_matches_opencv_and_truth(self):
        calib = _calib()
        truth = np.array([
            [0.02, -0.01, 0.45],
            [-0.03, 0.02, 0.70],
            [0.00, 0.00, 1.10],
            [0.05, 0.04, 0.55],
        ])
        left = _project_independent(truth, calib["K_left"])
        right = _project_independent(truth, calib["K_right"], calib["R"], calib["T"])
        recovered, confidence = triangulate_pair(left, right, calib)
        oracle = _cv_triangulate(left, right, calib)
        self.assertLess(np.max(np.abs(recovered - truth)), 1e-4)
        self.assertLess(np.max(np.abs(recovered - oracle)), 1e-6)
        self.assertTrue(np.all(confidence > 0.99))

    def test_rotated_right_camera(self):
        angle = math.radians(8.0)
        rotation = np.array([
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ])
        calib = _calib(rotation=rotation, translation=np.array([-0.08, 0.001, 0.002]))
        truth = np.array([
            [0.01, 0.02, 0.50],
            [-0.02, -0.01, 0.80],
            [0.03, 0.00, 0.62],
        ])
        left = _project_independent(truth, calib["K_left"])
        right = _project_independent(truth, calib["K_right"], calib["R"], calib["T"])
        recovered, confidence = triangulate_pair(left, right, calib)
        self.assertLess(np.max(np.abs(recovered - truth)), 1e-4)
        self.assertTrue(np.all(confidence > 0.99))

    def test_points_behind_the_camera_have_zero_confidence(self):
        calib = _calib()
        truth = np.array([[0.0, 0.0, -0.4]])
        left = _project_independent(truth, calib["K_left"])
        right = _project_independent(truth, calib["K_right"], calib["R"], calib["T"])
        recovered, confidence = triangulate_pair(left, right, calib)
        self.assertEqual(float(confidence[0]), 0.0)
        self.assertFalse(np.isfinite(recovered[0]).all())

    def test_scale_about_the_wrist_undoes_a_factor_of_two(self):
        calib = _calib()
        truth = _hand([0.02, -0.01, 0.6])
        mono = truth * 2.0
        left = _project_independent(truth, calib["K_left"])
        right = _project_independent(truth, calib["K_right"], calib["R"], calib["T"])
        triangulated, confidence = triangulate_pair(left, right, calib)
        scale = estimate_scale(mono, triangulated, confidence)
        self.assertAlmostEqual(scale, 0.5, places=4)
        aligned = apply_scale(mono, triangulated, scale)
        self.assertLess(np.max(np.abs(aligned - truth)), 1e-3)
        fused, fused_confidence, fused_scale = correct_monocular_with_stereo(mono, left, right, calib)
        self.assertAlmostEqual(fused_scale, 0.5, places=4)
        self.assertLess(np.max(np.abs(fused - truth)), 1e-3)
        self.assertTrue(np.all(fused_confidence > 0.99))


class WorldPoseTest(unittest.TestCase):
    def test_ry90_maps_optical_axis_point(self):
        pose = np.eye(4)
        pose[:3, :3] = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        pose[:3, 3] = [1.0, 2.0, 3.0]
        world = transform_points([0.0, 0.0, 1.0], pose)
        self.assertTrue(np.allclose(world, [2.0, 2.0, 3.0]))
        self.assertTrue(np.allclose(world_to_camera(world, pose), [0.0, 0.0, 1.0]))

    def test_tum_exact_timestamp_and_gap(self):
        pose = np.eye(4)
        pose[:3, :3] = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        pose[:3, 3] = [1.0, 2.0, 3.0]
        half = math.sqrt(0.5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "slam.tum"
            path.write_text(
                "1.5 1 2 3 0 %.17g 0 %.17g\n" % (half, half),
                encoding="utf-8",
            )
            poses, gaps = associate_camera_poses([1.5], path)
            self.assertAlmostEqual(gaps[0], 0.0)
            self.assertTrue(np.allclose(poses[0], pose, atol=1e-8))
            world = transform_points([0.0, 0.0, 1.0], poses[0])
            self.assertTrue(np.allclose(world, [2.0, 2.0, 3.0], atol=1e-6))
            _, later = associate_camera_poses([1.55], path)
            self.assertAlmostEqual(later[0], 0.05)

    def test_missing_slam_is_identity(self):
        poses, gaps = associate_camera_poses([0.0, 0.1], None)
        self.assertEqual(gaps, [0.0, 0.0])
        self.assertTrue(np.allclose(poses[0], np.eye(4)))
        self.assertTrue(np.allclose(transform_points([0.2, 0.0, 0.5], poses[1]), [0.2, 0.0, 0.5]))

    def test_wrist_axes_are_right_handed(self):
        joints = _hand([0.0, 0.0, 0.6])
        rotation = wrist_rotation(joints)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=6)
        toward_index = joints[5] - joints[0]
        toward_index = toward_index / np.linalg.norm(toward_index)
        self.assertTrue(np.allclose(rotation[:, 0], toward_index))

    def test_one_centimeter_wrist_shift(self):
        truth = np.stack([_hand([0.0, 0.0, 0.5]), _hand([0.01, 0.0, 0.5])])
        shifted = truth.copy()
        shifted[:, 0, 0] += 0.01
        self.assertAlmostEqual(mean_wrist_error_m(shifted, truth) * 100.0, 1.0, places=6)


class CalibrationLoadTest(unittest.TestCase):
    def test_simple_yaml_roundtrip(self):
        angle = math.radians(5.0)
        rotation = np.array([
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ])
        original = _calib(rotation=rotation, translation=np.array([-0.08, 0.002, -0.001]))
        with tempfile.TemporaryDirectory() as tmp:
            path = write_calibration_yaml(Path(tmp) / "calib.yaml", original)
            loaded = load_calibration(path)
        self.assertTrue(np.allclose(loaded["K_left"], original["K_left"]))
        self.assertTrue(np.allclose(loaded["R"], original["R"]))
        self.assertTrue(np.allclose(loaded["T"], original["T"]))

    def test_kalibr_camchain(self):
        text = """
cam0:
  camera_model: pinhole
  distortion_coeffs: [0.0, 0.0, 0.0, 0.0]
  intrinsics: [200.0, 200.0, 160.0, 120.0]
  resolution: [320, 240]
cam1:
  camera_model: pinhole
  distortion_coeffs: [0.0, 0.0, 0.0, 0.0]
  intrinsics: [200.0, 200.0, 160.0, 120.0]
  resolution: [320, 240]
  T_cn_cnm1:
    - [1.0, 0.0, 0.0, -0.08]
    - [0.0, 1.0, 0.0, 0.0]
    - [0.0, 0.0, 1.0, 0.0]
    - [0.0, 0.0, 0.0, 1.0]
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "camchain.yaml"
            path.write_text(text, encoding="utf-8")
            loaded = load_calibration(path)
        self.assertEqual(loaded["image_width"], 320)
        self.assertTrue(np.allclose(loaded["T"], [-0.08, 0.0, 0.0]))
        self.assertTrue(np.allclose(loaded["K_left"][0, 0], 200.0))

    def test_opencv_filestorage(self):
        import cv2
        calib = _calib()
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "stereo.yaml")
            storage = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
            storage.write("image_width", 320)
            storage.write("image_height", 240)
            storage.write("cameraMatrix1", calib["K_left"])
            storage.write("cameraMatrix2", calib["K_right"])
            storage.write("distCoeffs1", calib["dist_left"])
            storage.write("distCoeffs2", calib["dist_right"])
            storage.write("R", calib["R"])
            storage.write("T", calib["T"].reshape(3, 1))
            storage.release()
            loaded = load_calibration(path)
        self.assertEqual((loaded["image_width"], loaded["image_height"]), (320, 240))
        self.assertTrue(np.allclose(loaded["K_right"], calib["K_right"]))
        self.assertTrue(np.allclose(loaded["T"], calib["T"]))


class ConvertSessionTest(unittest.TestCase):
    def _write_session(self, root, with_stereo, with_slam):
        calib = _calib()
        write_calibration_yaml(root / "calib.yaml", calib)
        stamps = [index / 30.0 for index in range(4)]
        (root / "timestamps.csv").write_text(
            "frame_index,timestamp_s\n" + "\n".join("%d,%.8f" % (index, stamp) for index, stamp in enumerate(stamps)),
            encoding="utf-8",
        )
        (root / "imu.csv").write_text(
            "timestamp_s,gx,gy,gz,ax,ay,az\n0,0,0,0,0,0,9.8\n",
            encoding="utf-8",
        )
        (root / "metadata.json").write_text(json.dumps({
            "episode_id": "headcam/synth",
            "fps": 30,
            "task": "pick",
            "instruction": "拿起",
            "environment": "tabletop",
            "image_width": 320,
            "image_height": 240,
            "objects": ["cup"],
            "verbs": ["pick"],
        }), encoding="utf-8")
        if with_slam:
            lines = ["%.8f 0 0 0 0 0 0 1" % stamp for stamp in stamps]
            (root / "slam.tum").write_text("\n".join(lines) + "\n", encoding="utf-8")
        frames = []
        expected = {"left": [], "right": []}
        for index, stamp in enumerate(stamps):
            del stamp
            left = _hand([0.01 * index, 0.02, 0.60])
            right = _hand([-0.02 + 0.01 * index, -0.03, 0.70])
            expected["left"].append(left)
            expected["right"].append(right)
            frame = {
                "left": {
                    "joints_cam": (left * 2.0).tolist(),
                    "keypoints_2d": _project_independent(left, calib["K_left"]).tolist(),
                    "confidence": [0.2] * 21,
                },
                "right": {
                    "joints_cam": (right * 2.0).tolist(),
                    "keypoints_2d": _project_independent(right, calib["K_left"]).tolist(),
                    "confidence": [0.2] * 21,
                },
            }
            if with_stereo:
                frame["stereo_keypoints_right_view"] = {
                    "left": _project_independent(left, calib["K_right"], calib["R"], calib["T"]).tolist(),
                    "right": _project_independent(right, calib["K_right"], calib["R"], calib["T"]).tolist(),
                }
            frames.append(frame)
        (root / "hands.json").write_text(json.dumps({"frames": frames}), encoding="utf-8")
        return expected

    def test_stereo_session_becomes_metric_world_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = self._write_session(root, with_stereo=True, with_slam=True)
            out = root / "episode.json"
            convert_headcam.main(["--session", str(root), "--out", str(out)])
            episode = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(validate_episode(episode), [])
        self.assertEqual(episode["source"], "headcam")
        self.assertEqual(episode["coordinate_frame"], "slam_world")
        self.assertEqual(episode["annotation"]["subtasks"], [])
        self.assertEqual(episode["annotation"]["instructions"], [])
        self.assertEqual(episode["hand_pose"]["imu_samples"], 1)
        self.assertTrue(episode["hand_pose"]["stereo"])
        self.assertAlmostEqual(episode["hand_pose"]["scale"]["left"][0], 0.5, places=3)
        self.assertEqual(episode["hand_pose"]["slam_time_gap_s"], [0.0, 0.0, 0.0, 0.0])
        for side in ("left", "right"):
            got = np.asarray(episode["hands"][side]["joints"], dtype=float)
            self.assertLess(np.max(np.abs(got - np.stack(expected[side]))), 1e-3)
            wrist = episode["hands"][side]["wrist_pose"][0]
            self.assertEqual(len(wrist), 7)
            self.assertTrue(all(value is not None for value in wrist))
        report = qc_episode(episode)
        self.assertTrue(report["accepted"], report)

    def test_without_stereo_or_slam_keeps_monocular_scale_in_camera(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = self._write_session(root, with_stereo=False, with_slam=False)
            episode = convert_headcam.build_episode(root)
        self.assertEqual(episode["coordinate_frame"], "camera")
        self.assertFalse(episode["hand_pose"]["stereo"])
        self.assertIsNone(episode["hand_pose"]["scale"]["left"][0])
        got = np.asarray(episode["hands"]["left"]["joints"][0], dtype=float)
        self.assertLess(np.max(np.abs(got - expected["left"][0] * 2.0)), 1e-8)
        poses, _gaps = associate_camera_poses(episode["timestamps"], None)
        self.assertTrue(np.allclose(poses[0], np.eye(4)))


class BackendAvailabilityTest(unittest.TestCase):
    def test_unknown_backend_is_rejected(self):
        with self.assertRaises(ValueError):
            get_backend("nope")

    def test_hamer_names_mano_license_when_weights_are_absent(self):
        if hamer_available():
            self.skipTest("本机已装 HaMeR")
        with self.assertRaises(RuntimeError) as caught:
            HaMeRBackend().predict(np.zeros((48, 64, 3), dtype=np.uint8))
        message = str(caught.exception)
        self.assertIn("MANO", message)
        self.assertIn("mano.is.tue.mpg.de", message)

    def test_wilor_names_mano_license_when_weights_are_absent(self):
        if wilor_available():
            self.skipTest("本机已装 WiLoR")
        with self.assertRaises(RuntimeError) as caught:
            WiLoRBackend().predict(np.zeros((48, 64, 3), dtype=np.uint8))
        self.assertIn("MANO", str(caught.exception))

    @unittest.skipUnless(mediapipe_available(), "mediapipe 未安装")
    def test_mediapipe_blank_frame(self):
        prediction = MediaPipeHandsBackend().predict(np.zeros((64, 64, 3), dtype=np.uint8))
        self.assertIn("left", prediction)
        self.assertIn("right", prediction)

    @unittest.skipUnless(hamer_available(), "hamer 未安装或缺少 MANO / 权重")
    def test_hamer_blank_frame(self):
        prediction = HaMeRBackend().predict(np.zeros((64, 64, 3), dtype=np.uint8))
        self.assertIn("left", prediction)

    @unittest.skipUnless(wilor_available(), "wilor 未安装或缺少 MANO / 权重")
    def test_wilor_blank_frame(self):
        prediction = WiLoRBackend().predict(np.zeros((64, 64, 3), dtype=np.uint8))
        self.assertIn("left", prediction)


class LibraryProjectorAgreesTest(unittest.TestCase):
    def test_project_pinhole_matches_the_independent_formula(self):
        calib = _calib()
        truth = np.array([[0.02, -0.01, 0.5], [0.0, 0.03, 0.8]])
        own = _project_independent(truth, calib["K_left"])
        library = project_pinhole(truth, calib["K_left"])
        self.assertTrue(np.allclose(own, library))


if __name__ == "__main__":
    unittest.main()
