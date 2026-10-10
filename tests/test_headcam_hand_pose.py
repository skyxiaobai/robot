# -*- coding: utf-8 -*-
"""合成双目：已知三维点投影到左右相机，检查三角化、尺度和世界系。"""
import inspect
import json
import math
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import convert_headcam  # noqa: E402
import egodata.egodex as egodex  # noqa: E402
from egodata.egodex import EGODEX_NONCORRESPONDING_JOINTS  # noqa: E402
from egodata.qc import qc_episode  # noqa: E402
from egodata.schema import validate_episode  # noqa: E402
from headcam import hand_pose  # noqa: E402
from headcam.hand_pose import (  # noqa: E402
    DEFAULT_WRIST_DEPTH_M,
    HAND_LANDMARKER_URL,
    HaMeRBackend,
    MediaPipeHandsBackend,
    WiLoRBackend,
    apply_scale,
    associate_camera_poses,
    choose_mediapipe_api,
    correct_monocular_with_stereo,
    ensure_hand_landmarker_model,
    estimate_scale,
    evaluate_hand_frames,
    format_hand_eval_report,
    get_backend,
    hamer_available,
    load_calibration,
    mean_wrist_error_m,
    mediapipe_available,
    mediapipe_tasks_failure_message,
    open_mediapipe_detector,
    prediction_from_mediapipe_hands,
    prediction_from_solutions_result,
    prediction_from_tasks_result,
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


def _mark(x, y, z=0.0, visibility=None):
    point = type("Lm", (), {})()
    point.x = float(x)
    point.y = float(y)
    point.z = float(z)
    if visibility is not None:
        point.visibility = float(visibility)
    return point


def _grid(x, y, z=0.0, visibility=None):
    return [_mark(x, y, z, visibility) for _ in range(21)]


def _blank_prediction_sides(prediction):
    return prediction["left"]["joints_cam"] is None and prediction["right"]["joints_cam"] is None


class _RecordingLandmarker(object):
    options = None

    @staticmethod
    def create_from_options(options):
        _RecordingLandmarker.options = options
        return "tasks-detector"


class _BoomLandmarker(object):
    @staticmethod
    def create_from_options(options):
        raise OSError("libGLESv2.so.2: cannot open shared object file")


class _Delegate(object):
    CPU = "CPU"
    GPU = "GPU"


class _BaseOptions(object):
    Delegate = _Delegate

    def __init__(self, model_asset_path=None, delegate=None):
        self.model_asset_path = model_asset_path
        self.delegate = delegate


class _RunningMode(object):
    IMAGE = "IMAGE"


class _HandOptions(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _HandsApi(object):
    calls = []

    def Hands(self, **kwargs):
        _HandsApi.calls.append(kwargs)
        return "solutions-detector"


class _Solutions(object):
    hands = _HandsApi()


def _tasks_module(landmarker):
    vision = type("Vision", (), {})()
    vision.HandLandmarker = landmarker
    vision.HandLandmarkerOptions = _HandOptions
    vision.RunningMode = _RunningMode
    tasks = type("Tasks", (), {})()
    tasks.BaseOptions = _BaseOptions
    tasks.vision = vision
    return tasks


def _fake_mp(landmarker=None, with_solutions=True):
    module = type("Mp", (), {})()
    if landmarker is not None:
        module.tasks = _tasks_module(landmarker)
    if with_solutions:
        module.solutions = _Solutions()
    return module


class MediaPipeApiTest(unittest.TestCase):
    def test_choose_prefers_tasks_and_falls_back_to_solutions(self):
        self.assertEqual(choose_mediapipe_api(True, True), "tasks")
        self.assertEqual(choose_mediapipe_api(True, False), "tasks")
        self.assertEqual(choose_mediapipe_api(False, True), "solutions")
        with self.assertRaises(RuntimeError):
            choose_mediapipe_api(False, False)

    def test_tasks_detector_uses_cpu_delegate_by_default(self):
        _RecordingLandmarker.options = None
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "hand_landmarker.task"
            model.write_bytes(b"model")
            kind, detector = open_mediapipe_detector(_fake_mp(_RecordingLandmarker), model_path=model)
        self.assertEqual((kind, detector), ("tasks", "tasks-detector"))
        self.assertEqual(_RecordingLandmarker.options.base_options.delegate, "CPU")
        self.assertEqual(_RecordingLandmarker.options.num_hands, 2)
        self.assertEqual(_RecordingLandmarker.options.running_mode, "IMAGE")

    def test_gpu_delegate_is_selectable(self):
        _RecordingLandmarker.options = None
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "hand_landmarker.task"
            model.write_bytes(b"model")
            open_mediapipe_detector(
                _fake_mp(_RecordingLandmarker, with_solutions=False),
                model_path=model,
                delegate="gpu",
            )
        self.assertEqual(_RecordingLandmarker.options.base_options.delegate, "GPU")

    def test_tasks_init_failure_falls_back_to_solutions(self):
        _HandsApi.calls = []
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "hand_landmarker.task"
            model.write_bytes(b"model")
            kind, detector = open_mediapipe_detector(_fake_mp(_BoomLandmarker), model_path=model)
        self.assertEqual((kind, detector), ("solutions", "solutions-detector"))
        self.assertTrue(_HandsApi.calls)
        self.assertTrue(_HandsApi.calls[-1]["static_image_mode"])
        self.assertEqual(_HandsApi.calls[-1]["max_num_hands"], 2)

    def test_tasks_init_failure_without_legacy_explains_headless_setup(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "hand_landmarker.task"
            model.write_bytes(b"model")
            with self.assertRaises(RuntimeError) as caught:
                open_mediapipe_detector(
                    _fake_mp(_BoomLandmarker, with_solutions=False),
                    model_path=model,
                )
        message = str(caught.exception)
        self.assertIn("libGLESv2", message)
        self.assertIn("CPU", message)
        self.assertIn("libGLESv2", mediapipe_tasks_failure_message(OSError("libGLESv2.so.2")))

    def test_solutions_used_when_tasks_is_missing(self):
        _HandsApi.calls = []
        kind, detector = open_mediapipe_detector(_fake_mp(landmarker=None), model_path=None)
        self.assertEqual((kind, detector), ("solutions", "solutions-detector"))

    def test_cached_task_model_is_not_redownloaded(self):
        self.assertIn("hand_landmarker.task", HAND_LANDMARKER_URL)
        self.assertTrue(HAND_LANDMARKER_URL.startswith("https://storage.googleapis.com/"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hand_landmarker.task"
            path.write_bytes(b"abc")
            self.assertEqual(ensure_hand_landmarker_model(path), path)
            previous = os.environ.get("MEDIAPIPE_HAND_LANDMARKER")
            os.environ["MEDIAPIPE_HAND_LANDMARKER"] = str(path)
            try:
                self.assertEqual(ensure_hand_landmarker_model(), path)
            finally:
                if previous is None:
                    os.environ.pop("MEDIAPIPE_HAND_LANDMARKER", None)
                else:
                    os.environ["MEDIAPIPE_HAND_LANDMARKER"] = previous

    def test_backend_depth_prior_is_configurable(self):
        previous = os.environ.pop("MEDIAPIPE_HAND_DELEGATE", None)
        try:
            self.assertAlmostEqual(DEFAULT_WRIST_DEPTH_M, 0.55)
            backend = MediaPipeHandsBackend(wrist_depth_m=0.29)
            self.assertAlmostEqual(backend.wrist_depth_m, 0.29)
            self.assertEqual(backend.delegate, "cpu")
            with self.assertRaises(ValueError):
                MediaPipeHandsBackend(wrist_depth_m=0.0)
            with self.assertRaises(ValueError):
                MediaPipeHandsBackend(delegate="tpu")
            os.environ["MEDIAPIPE_HAND_DELEGATE"] = "gpu"
            self.assertEqual(MediaPipeHandsBackend().delegate, "gpu")
            routed = get_backend("mediapipe", wrist_depth_m=0.4)
            self.assertAlmostEqual(routed.wrist_depth_m, 0.4)
        finally:
            if previous is None:
                os.environ.pop("MEDIAPIPE_HAND_DELEGATE", None)
            else:
                os.environ["MEDIAPIPE_HAND_DELEGATE"] = previous

    def test_docs_cover_gles_cpu_delegate_and_stereo_metric_depth(self):
        text = (hand_pose.__doc__ or "") + "\n" + (MediaPipeHandsBackend.__doc__ or "")
        self.assertIn("libGLESv2", text)
        self.assertIn("CPU", text)
        self.assertIn("双目", text)
        self.assertIn("0.55", text)

    def test_duplicate_handedness_keeps_both_hands_by_image_x(self):
        # 两只手都被标成 Left。自拍翻转后都会进 right，旧逻辑只留高分的那只。
        prediction = prediction_from_mediapipe_hands(
            [
                ("Left", 0.95, _grid(0.2, 0.5), None),
                ("Left", 0.55, _grid(0.8, 0.5), None),
            ],
            width=100,
            height=80,
            calib=None,
            wrist_depth_m=0.55,
        )
        self.assertIsNotNone(prediction["left"]["joints_cam"])
        self.assertIsNotNone(prediction["right"]["joints_cam"])
        self.assertLess(
            prediction["left"]["keypoints_2d"][0, 0],
            prediction["right"]["keypoints_2d"][0, 0],
        )
        self.assertAlmostEqual(prediction["left"]["keypoints_2d"][0, 0], 20.0)
        self.assertAlmostEqual(prediction["right"]["keypoints_2d"][0, 0], 80.0)

    def test_tied_image_x_gives_the_labeled_side_to_the_higher_score(self):
        prediction = prediction_from_mediapipe_hands(
            [
                ("Left", 0.9, _grid(0.5, 0.25), None),
                ("Left", 0.4, _grid(0.5, 0.75), None),
            ],
            width=100,
            height=80,
            calib=None,
            wrist_depth_m=0.55,
        )
        # Left 翻转后的标签是 right。x 相同，高分留在 right，低分改到另一侧。
        self.assertAlmostEqual(prediction["right"]["keypoints_2d"][0, 1], 20.0)
        self.assertAlmostEqual(prediction["left"]["keypoints_2d"][0, 1], 60.0)

    def test_distinct_labels_stay_on_their_flipped_sides(self):
        prediction = prediction_from_mediapipe_hands(
            [
                ("Left", 0.9, _grid(0.2, 0.5), None),
                ("Right", 0.9, _grid(0.8, 0.5), None),
            ],
            width=100,
            height=80,
            calib=None,
            wrist_depth_m=0.4,
        )
        self.assertAlmostEqual(prediction["right"]["keypoints_2d"][0, 0], 20.0)
        self.assertAlmostEqual(prediction["left"]["keypoints_2d"][0, 0], 80.0)

    def test_depth_prior_and_intrinsics_set_the_monocular_wrist(self):
        prediction = prediction_from_mediapipe_hands(
            [("Right", 0.8, _grid(0.75, 0.5), None)],
            width=100,
            height=80,
            calib={"K_left": _K(fx=50.0, cx=50.0, cy=40.0)},
            wrist_depth_m=0.29,
        )
        wrist = prediction["left"]["joints_cam"][0]
        self.assertAlmostEqual(wrist[2], 0.29, places=6)
        self.assertAlmostEqual(wrist[0], (75.0 - 50.0) * 0.29 / 50.0, places=6)
        self.assertIsNone(prediction["right"]["joints_cam"])

    def test_world_landmarks_are_added_relative_to_the_wrist(self):
        world = _grid(0.0, 0.0, 0.0)
        world[8] = _mark(0.02, -0.01, 0.03)
        prediction = prediction_from_mediapipe_hands(
            [("Right", 0.8, _grid(0.5, 0.5), world)],
            width=64,
            height=64,
            calib=None,
            wrist_depth_m=0.4,
        )
        joints = prediction["left"]["joints_cam"]
        self.assertAlmostEqual(joints[0, 2], 0.4, places=6)
        self.assertAlmostEqual(joints[8, 0] - joints[0, 0], 0.02, places=6)
        self.assertAlmostEqual(joints[8, 1] - joints[0, 1], -0.01, places=6)

    def test_tasks_result_ignores_zero_visibility(self):
        landmarks = _grid(0.5, 0.5, visibility=0.0)
        result = type("Result", (), {})()
        result.hand_landmarks = [landmarks]
        result.hand_world_landmarks = [None]
        category = type("Cat", (), {})()
        category.category_name = "Right"
        category.score = 0.8
        result.handedness = [[category]]
        prediction = prediction_from_tasks_result(result, 32, 32, None, 0.55)
        self.assertIsNotNone(prediction["left"]["joints_cam"])
        self.assertAlmostEqual(float(np.mean(prediction["left"]["confidence"])), 0.8, places=6)

    def test_solutions_result_scales_confidence_by_visibility(self):
        landmarks = type("Container", (), {})()
        landmarks.landmark = _grid(0.5, 0.5, visibility=0.5)
        handed = type("Handed", (), {})()
        handed.classification = [type("Cls", (), {"label": "Left", "score": 0.8})()]
        results = type("Results", (), {})()
        results.multi_hand_landmarks = [landmarks]
        results.multi_hand_world_landmarks = [None]
        results.multi_handedness = [handed]
        prediction = prediction_from_solutions_result(results, 32, 32, None, 0.55)
        self.assertIsNotNone(prediction["right"]["joints_cam"])
        self.assertAlmostEqual(float(np.mean(prediction["right"]["confidence"])), 0.4, places=6)

    def test_empty_detector_results_have_no_hands(self):
        solutions = type("Results", (), {})()
        solutions.multi_hand_landmarks = None
        solutions.multi_hand_world_landmarks = None
        solutions.multi_handedness = None
        self.assertTrue(_blank_prediction_sides(prediction_from_solutions_result(solutions, 8, 8, None, 0.55)))
        tasks = type("Result", (), {})()
        tasks.hand_landmarks = []
        tasks.hand_world_landmarks = []
        tasks.handedness = []
        self.assertTrue(_blank_prediction_sides(prediction_from_tasks_result(tasks, 8, 8, None, 0.55)))


def _spread_hand(wrist):
    wrist = np.asarray(wrist, dtype=np.float64)
    joints = np.zeros((21, 3), dtype=np.float64)
    for index in range(21):
        joints[index] = wrist + np.array([0.004 * (index % 5 - 2), 0.003 * (index // 5), 0.0])
    joints[0] = wrist
    return joints


def _pred(joints, keypoints):
    return {"joints_cam": np.asarray(joints, dtype=np.float64), "keypoints_2d": np.asarray(keypoints, dtype=np.float64)}


def _eval_sample(gt, pred, K, width, height, confidence=None, episode_id="ep"):
    if confidence is None:
        confidence = {"left": 0.99, "right": 0.99}
    return {
        "episode_id": episode_id,
        "width": width,
        "height": height,
        "K": K,
        "gt": gt,
        "gt_confidence": confidence,
        "pred": pred,
    }


class EgoDexHandEvalTest(unittest.TestCase):
    def test_hand_and_thumb_knuckle_are_not_mediapipe_joints(self):
        self.assertEqual(EGODEX_NONCORRESPONDING_JOINTS, (0, 1))
        source = inspect.getsource(egodex)
        self.assertIn("前臂", source)
        self.assertIn("ThumbKnuckle", source)

    def test_close_wrist_is_detected_and_same_side_is_not_a_swap(self):
        intrinsic = _K(fx=50.0, cx=40.0, cy=30.0)
        gt = _spread_hand([0.0, 0.0, 1.0])
        keypoints = _project_independent(gt, intrinsic)
        report = evaluate_hand_frames([
            _eval_sample(
                {"left": gt, "right": None},
                {"left": _pred(gt, keypoints), "right": None},
                intrinsic, 80, 60,
            ),
        ])
        self.assertEqual(report["gt_visible"], 1)
        self.assertEqual(report["matched"], 1)
        self.assertAlmostEqual(report["detection_rate"], 1.0)
        self.assertAlmostEqual(report["swap_rate"], 0.0)
        self.assertAlmostEqual(report["px_error_median"], 0.0, places=4)
        self.assertAlmostEqual(report["root_relative_m_median"], 0.0, places=6)

    def test_pixel_error_follows_episode_intrinsics(self):
        intrinsic = _K(fx=80.0, cx=40.0, cy=30.0)
        gt = _spread_hand([0.02, 0.0, 1.0])
        keypoints = _project_independent(gt, intrinsic)
        keypoints = keypoints + np.array([3.0, -4.0])
        report = evaluate_hand_frames([
            _eval_sample(
                {"left": gt, "right": None},
                {"left": _pred(gt, keypoints), "right": None},
                intrinsic, 200, 120,
            ),
        ])
        self.assertAlmostEqual(report["px_error_median"], 5.0, places=4)
        wider = _K(fx=1920.0, cx=40.0, cy=30.0)
        mismatched = evaluate_hand_frames([
            _eval_sample(
                {"left": gt, "right": None},
                {"left": _pred(gt, keypoints), "right": None},
                wider, 200, 120,
            ),
        ])
        self.assertGreater(mismatched["px_error_median"], 20.0)

    def test_wrist_beyond_match_distance_is_a_miss(self):
        intrinsic = _K(fx=100.0, cx=50.0, cy=50.0)
        gt = _spread_hand([0.0, 0.0, 1.0])
        keypoints = _project_independent(gt, intrinsic)
        keypoints = keypoints + np.array([300.0, 0.0])
        report = evaluate_hand_frames([
            _eval_sample(
                {"left": gt, "right": None},
                {"left": _pred(gt, keypoints), "right": None},
                intrinsic, 500, 200,
            ),
        ], match_px=250.0)
        self.assertEqual(report["gt_visible"], 1)
        self.assertEqual(report["matched"], 0)
        self.assertAlmostEqual(report["detection_rate"], 0.0)
        self.assertTrue(math.isnan(report["swap_rate"]))

    def test_opposite_side_match_counts_as_a_swap(self):
        intrinsic = _K(fx=100.0, cx=80.0, cy=40.0)
        left = _spread_hand([-0.1, 0.0, 1.0])
        right = _spread_hand([0.15, 0.0, 1.0])
        crossed = _eval_sample(
            {"left": left, "right": right},
            {
                "left": _pred(left, _project_independent(right, intrinsic)),
                "right": _pred(right, _project_independent(left, intrinsic)),
            },
            intrinsic, 200, 100,
        )
        report = evaluate_hand_frames([crossed])
        self.assertEqual(report["matched"], 2)
        self.assertAlmostEqual(report["swap_rate"], 1.0)
        correct = _eval_sample(
            {"left": left, "right": None},
            {"left": _pred(left, _project_independent(left, intrinsic)), "right": None},
            intrinsic, 200, 100,
        )
        swapped = _eval_sample(
            {"left": left, "right": None},
            {"left": None, "right": _pred(left, _project_independent(left, intrinsic))},
            intrinsic, 200, 100,
        )
        mixed = evaluate_hand_frames([correct, swapped])
        self.assertEqual(mixed["matched"], 2)
        self.assertAlmostEqual(mixed["swap_rate"], 0.5)

    def test_low_confidence_and_partial_visibility_are_not_gt_visible(self):
        intrinsic = _K(fx=50.0, cx=50.0, cy=40.0)
        visible = _spread_hand([0.0, 0.0, 1.0])
        mostly_out = visible.copy()
        mostly_out[5:] = [2.0, 0.0, 1.0]
        wrist_out = visible.copy()
        wrist_out[0] = [2.0, 0.0, 1.0]
        keypoints = np.zeros((21, 2))
        low = evaluate_hand_frames([
            _eval_sample(
                {"left": visible, "right": None},
                {"left": _pred(visible, keypoints), "right": None},
                intrinsic, 100, 80,
                confidence={"left": 0.4, "right": 0.99},
            ),
        ])
        self.assertEqual(low["gt_visible"], 0)
        unknown = evaluate_hand_frames([
            _eval_sample(
                {"left": visible, "right": None},
                {"left": _pred(visible, _project_independent(visible, intrinsic)), "right": None},
                intrinsic, 100, 80,
                confidence={"left": None, "right": None},
            ),
        ])
        self.assertEqual(unknown["gt_visible"], 1)
        self.assertAlmostEqual(unknown["detection_rate"], 1.0)
        hidden = evaluate_hand_frames([
            _eval_sample({"left": mostly_out, "right": wrist_out}, {"left": None, "right": None}, intrinsic, 100, 80),
        ])
        self.assertEqual(hidden["gt_visible"], 0)

    def test_exclude_hand_and_thumb_knuckle_from_joint_errors(self):
        intrinsic = _K(fx=100.0, cx=50.0, cy=40.0)
        gt = _spread_hand([0.0, 0.0, 1.0])
        pred_joints = gt.copy()
        pred_joints[1, 0] += 0.05
        keypoints = _project_independent(gt, intrinsic)
        keypoints[0] = [50.0, 80.0]
        keypoints[1] = [90.0, 40.0]
        sample = _eval_sample(
            {"left": gt, "right": None},
            {"left": _pred(pred_joints, keypoints), "right": None},
            intrinsic, 100, 80,
        )
        full = evaluate_hand_frames([sample])
        dropped = evaluate_hand_frames([sample], exclude_joints=EGODEX_NONCORRESPONDING_JOINTS)
        self.assertAlmostEqual(full["root_relative_m_median"], 0.05 / 20.0, places=6)
        self.assertAlmostEqual(dropped["root_relative_m_median"], 0.0, places=6)
        self.assertGreater(full["px_error_median"], 1.0)
        self.assertAlmostEqual(dropped["px_error_median"], 0.0, places=4)
        self.assertEqual(dropped["exclude_joints"], (0, 1))

    def test_one_scale_per_episode_hand_aligns_the_wrist(self):
        intrinsic = _K(fx=100.0, cx=50.0, cy=40.0)
        samples = []
        for pred_z, gt_z in ((0.40, 0.20), (0.80, 0.20)):
            gt = _spread_hand([0.0, 0.0, gt_z])
            pred_joints = _spread_hand([0.0, 0.0, pred_z])
            samples.append(_eval_sample(
                {"left": gt, "right": None},
                {"left": _pred(pred_joints, _project_independent(gt, intrinsic)), "right": None},
                intrinsic, 100, 80,
                episode_id="clip",
            ))
        report = evaluate_hand_frames(samples)
        self.assertAlmostEqual(report["scale_median"], 0.3, places=5)
        self.assertAlmostEqual(report["wrist_error_m_median"], 0.4, places=5)
        self.assertAlmostEqual(report["wrist_error_scaled_m_median"], 0.06, places=5)
        pure = _spread_hand([0.02, -0.01, 0.29])
        mono = pure * (0.55 / 0.29)
        aligned = evaluate_hand_frames([
            _eval_sample(
                {"left": pure, "right": None},
                {"left": _pred(mono, _project_independent(pure, intrinsic)), "right": None},
                intrinsic, 200, 120,
            ),
        ])
        self.assertAlmostEqual(aligned["scale_median"], 0.29 / 0.55, places=4)
        self.assertAlmostEqual(aligned["wrist_error_scaled_m_median"], 0.0, places=4)
        self.assertGreater(aligned["wrist_error_m_median"], 0.2)
        shifted = pure.copy()
        shifted[:, 2] += 0.2
        relative = evaluate_hand_frames([
            _eval_sample(
                {"left": pure, "right": None},
                {"left": _pred(shifted, _project_independent(pure, intrinsic)), "right": None},
                intrinsic, 200, 120,
            ),
        ])
        self.assertAlmostEqual(relative["root_relative_m_median"], 0.0, places=5)
        self.assertAlmostEqual(relative["wrist_error_m_median"], 0.2, places=5)

    def test_format_names_detection_swap_pixels_and_scale(self):
        text = format_hand_eval_report({
            "gt_visible": 4,
            "matched": 2,
            "detection_rate": 0.5,
            "swap_rate": 0.066,
            "px_error_median": 40.0,
            "px_error_mean": 47.1,
            "root_relative_m_median": 0.0844,
            "wrist_error_m_median": 0.281,
            "wrist_error_scaled_m_median": 0.0396,
            "scale_median": 0.53,
            "exclude_joints": (),
        })
        self.assertIn("检出率", text)
        self.assertIn("50.0%", text)
        self.assertIn("左右交换", text)
        self.assertIn("6.6%", text)
        self.assertIn("40.0", text)
        self.assertIn("8.44", text)
        self.assertIn("28.10", text)
        self.assertIn("3.96", text)
        self.assertIn("0.53", text)
        excluded = format_hand_eval_report({
            "gt_visible": 0,
            "matched": 0,
            "detection_rate": float("nan"),
            "swap_rate": float("nan"),
            "px_error_median": float("nan"),
            "px_error_mean": float("nan"),
            "root_relative_m_median": float("nan"),
            "wrist_error_m_median": float("nan"),
            "wrist_error_scaled_m_median": float("nan"),
            "scale_median": float("nan"),
            "exclude_joints": (0, 1),
        })
        self.assertIn("0", excluded)
        self.assertIn("1", excluded)
        self.assertIn("n/a", excluded)


class ColabEgoDexEvalNotebookTest(unittest.TestCase):
    def test_last_cell_uses_intrinsics_all_frames_and_joint_metrics(self):
        notebook = json.loads((ROOT / "notebooks" / "headcam_hand_pose_colab.ipynb").read_text(encoding="utf-8"))
        code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        source = "".join(code[-1]["source"])
        markdown = [cell for cell in notebook["cells"] if cell["cell_type"] == "markdown"]
        prose = "".join(markdown[-1]["source"])
        self.assertIn("camera_intrinsic", source)
        self.assertIn("K_left", source)
        self.assertIn("evaluate_hand_frames", source)
        self.assertIn("format_hand_eval_report", source)
        self.assertIn("EGODEX_NONCORRESPONDING_JOINTS", source)
        self.assertIn("exclude_joints", source)
        self.assertIn("num_frames", source)
        self.assertRegex(source, r"predict\([\s\S]*calib\s*=")
        self.assertIsNone(re.search(r"MAX_FRAMES\s*=\s*20", source))
        self.assertNotIn("range(20", source)
        self.assertGreaterEqual(source.count("evaluate_hand_frames"), 2)
        self.assertIn("736", prose)
        self.assertIn("双目", prose)
        self.assertIn("Hand", prose)
        self.assertIn("libGLESv2", prose)


if __name__ == "__main__":
    unittest.main()
