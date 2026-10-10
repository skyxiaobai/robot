# -*- coding: utf-8 -*-
"""时序精修：平滑、补洞、固定骨长，以及 QC 能拒绝被补上的帧。"""
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import convert_headcam  # noqa: E402
import eval_hand_refine  # noqa: E402
from egodata.qc import frame_qc_flags  # noqa: E402
from egodata.schema import validate_episode  # noqa: E402
from headcam.hand_pose import (  # noqa: E402
    evaluate_hand_frames,
    transform_points,
    world_to_camera,
    write_calibration_yaml,
)
from headcam.hand_track_refine import (  # noqa: E402
    MEDIAPIPE_PARENTS,
    RefineParams,
    ablation_presets,
    accel_samples,
    apply_fixed_bone_lengths,
    bone_lengths,
    fill_gaps,
    fit_shape_coefficients,
    format_ablation_table,
    mean_accel,
    metrics_row,
    refine_hands,
    robust_median_bone_lengths,
    smooth_one_euro,
)


def _hand(bone=0.03, wrist=(0.0, 0.0, 0.5)):
    joints = np.zeros((21, 3), dtype=np.float64)
    joints[0] = np.asarray(wrist, dtype=np.float64)
    for child in range(1, 21):
        parent = int(MEDIAPIPE_PARENTS[child])
        direction = np.array([
            0.02 + 0.001 * child,
            0.01 * ((child % 5) - 2),
            0.004 * ((child % 3) - 1),
        ], dtype=np.float64)
        direction = direction / np.linalg.norm(direction)
        joints[child] = joints[parent] + direction * float(bone)
    return joints


def _sequence(count, wrist_fn, bone=0.03):
    frames = [_hand(bone=bone, wrist=wrist_fn(index)) for index in range(count)]
    return np.stack(frames, axis=0)


def _times(count, fps=30.0):
    return np.arange(count, dtype=np.float64) / float(fps)


def _conf(count, value=0.95):
    return np.full((count, 21), float(value), dtype=np.float64)


class FilterTest(unittest.TestCase):
    def test_one_euro_leaves_a_constant_signal_unchanged(self):
        values = np.full((30, 1), 0.4, dtype=np.float64)
        valid = np.ones((30, 1), dtype=bool)
        out = smooth_one_euro(values, valid, _times(30), RefineParams(smooth="one_euro"))
        self.assertTrue(np.allclose(out, 0.4))

    def test_one_euro_rejects_a_single_spike(self):
        count = 40
        values = np.full((count, 1), 0.5, dtype=np.float64)
        values[20, 0] = 1.5
        params = RefineParams(smooth="one_euro", min_cutoff=0.3, beta=0.0, d_cutoff=1.0)
        out = smooth_one_euro(values, np.ones((count, 1), dtype=bool), _times(count), params)
        self.assertLess(abs(float(out[20, 0]) - 0.5), 0.25)
        self.assertGreater(abs(float(out[20, 0]) - 1.5), 0.5)

    def test_lower_cutoff_smooths_noise_more(self):
        rng = np.random.default_rng(0)
        values = rng.normal(0.0, 0.01, size=(120, 1))
        valid = np.ones_like(values, dtype=bool)
        times = _times(120)
        tight = smooth_one_euro(values, valid, times, RefineParams(min_cutoff=0.2, beta=0.0, d_cutoff=1.0))
        loose = smooth_one_euro(values, valid, times, RefineParams(min_cutoff=8.0, beta=0.0, d_cutoff=1.0))
        self.assertLess(float(np.std(tight[10:])), float(np.std(loose[10:])))

    def test_kalman_reduces_noise_on_a_still_hand(self):
        rng = np.random.default_rng(1)
        count = 80
        clean = _sequence(count, lambda index: (0.0, 0.0, 0.5))
        noisy = clean + rng.normal(0.0, 0.01, size=clean.shape)
        empty = np.full_like(noisy, np.nan)
        out = refine_hands(
            noisy, empty, _conf(count), np.zeros((count, 21)), _times(count),
            RefineParams(smooth="kalman", kalman_accel_std=0.5, kalman_meas_std=0.01),
        )
        raw = float(np.std(noisy[15:, 0, 2]))
        filtered = float(np.std(out["left"]["joints"][15:, 0, 2]))
        self.assertLess(filtered, raw * 0.7)

    def test_unknown_smooth_is_rejected(self):
        with self.assertRaises(ValueError):
            RefineParams(smooth="spline")


class GapAndIdentityTest(unittest.TestCase):
    def test_short_gap_is_the_midpoint_and_flagged(self):
        count = 5
        joints = _sequence(count, lambda index: (0.01 * index, 0.0, 0.5))
        joints[1:4] = np.nan
        observed = np.array([True, False, False, False, True])
        filled_joints, filled = fill_gaps(joints, observed, max_gap=5)
        self.assertTrue(np.allclose(filled_joints[2, 0], 0.5 * (joints[0, 0] + joints[4, 0])))
        self.assertTrue(np.array_equal(filled, np.array([False, True, True, True, False])))
        self.assertTrue(np.allclose(filled_joints[0], joints[0]))
        self.assertTrue(np.allclose(filled_joints[4], joints[4]))

    def test_gap_fill_duration_does_not_depend_on_fps(self):
        # 0.12 s 的洞：30 fps 大约 3 帧，60 fps 大约 7 帧。按 5/30 秒两种都补；按 5 帧则 60 fps 补不上。
        for fps in (30.0, 60.0):
            count = int(fps)
            times = np.arange(count, dtype=np.float64) / fps
            hole = (times > 0.40) & (times < 0.52)
            joints = _sequence(count, lambda index: (0.01 * float(times[index]), 0.0, 0.5))
            joints[hole] = np.nan
            _, filled = fill_gaps(joints, ~hole, max_gap=5, timestamps=times, max_gap_s=5.0 / 30.0)
            self.assertTrue(hole.any())
            self.assertTrue(filled[hole].all(), fps)
            long_hole = (times > 0.40) & (times < 0.70)
            longer = joints.copy()
            longer[long_hole] = np.nan
            _, not_filled = fill_gaps(longer, ~long_hole, max_gap=5, timestamps=times, max_gap_s=5.0 / 30.0)
            self.assertFalse(not_filled.any(), fps)
        count = 60
        times = np.arange(count, dtype=np.float64) / 60.0
        hole = (times > 0.40) & (times < 0.52)
        joints = _sequence(count, lambda index: (0.0, 0.0, 0.5))
        joints[hole] = np.nan
        _, legacy = fill_gaps(joints, ~hole, max_gap=5)
        self.assertGreater(int(hole.sum()), 5)
        self.assertFalse(legacy.any())

    def test_gap_longer_than_n_stays_empty(self):
        joints = _sequence(6, lambda index: (0.0, 0.0, 0.5))
        joints[1:5] = np.nan
        observed = np.array([True, False, False, False, False, True])
        filled_joints, filled = fill_gaps(joints, observed, max_gap=3)
        self.assertFalse(filled.any())
        self.assertTrue(np.isnan(filled_joints[2, 0, 0]))

    def test_low_confidence_frame_is_replaced_and_marked_for_qc(self):
        count = 5
        joints = _sequence(count, lambda index: (0.02 * index, 0.0, 0.6))
        joints[2] = joints[2] + np.array([0.4, 0.0, 0.0])
        confidence = _conf(count)
        confidence[2] = 0.1
        out = refine_hands(
            joints, np.full_like(joints, np.nan), confidence, np.zeros((count, 21)),
            _times(count), RefineParams(gap_fill=True, max_gap=5),
        )
        self.assertTrue(out["left"]["filled"][2])
        self.assertFalse(out["left"]["filled"][0])
        self.assertAlmostEqual(float(out["left"]["confidence"][2, 0]), 0.0)
        self.assertLess(abs(float(out["left"]["joints"][2, 0, 0]) - 0.04), 1e-8)
        self.assertGreater(float(out["left"]["confidence"][0, 0]), 0.9)

    def test_leading_gap_is_not_extrapolated(self):
        joints = _sequence(4, lambda index: (0.0, 0.0, 0.5))
        joints[0] = np.nan
        confidence = _conf(4)
        confidence[0] = 0.0
        out = refine_hands(
            joints, np.full_like(joints, np.nan), confidence, np.zeros((4, 21)),
            _times(4), RefineParams(gap_fill=True, max_gap=5),
        )
        self.assertFalse(out["left"]["filled"][0])
        self.assertTrue(np.isnan(out["left"]["joints"][0, 0, 0]))

    def test_swapped_labels_follow_the_wrist_track(self):
        count = 12
        left = _sequence(count, lambda index: (-0.15, 0.0, 0.50 + 0.001 * index))
        right = _sequence(count, lambda index: (0.18, 0.02, 0.55 + 0.001 * index))
        left[6], right[6] = right[6].copy(), left[6].copy()
        out = refine_hands(
            left, right, _conf(count), _conf(count), _times(count),
            RefineParams(lr_consistency=True, lr_margin_m=0.02),
        )
        self.assertLess(float(out["left"]["joints"][6, 0, 0]), 0.0)
        self.assertGreater(float(out["right"]["joints"][6, 0, 0]), 0.0)
        self.assertTrue(out["left"]["swapped"][6])
        self.assertFalse(out["left"]["swapped"][5])

    def test_hands_closer_than_the_margin_keep_the_detector_label(self):
        count = 8
        left = _sequence(count, lambda index: (0.0, 0.0, 0.5))
        right = _sequence(count, lambda index: (0.01, 0.0, 0.5))
        left[4], right[4] = right[4].copy(), left[4].copy()
        out = refine_hands(
            left, right, _conf(count), _conf(count), _times(count),
            RefineParams(lr_consistency=True, lr_margin_m=0.05),
        )
        self.assertFalse(out["left"]["swapped"][4])
        self.assertGreater(float(out["left"]["joints"][4, 0, 0]), 0.005)


class ShapeTest(unittest.TestCase):
    def test_median_bone_length_rescales_a_stretched_frame(self):
        count = 11
        joints = _sequence(count, lambda index: (0.0, 0.0, 0.5), bone=0.03)
        stretched = joints[-1].copy()
        parent = int(MEDIAPIPE_PARENTS[8])
        direction = stretched[8] - stretched[parent]
        stretched[8] = stretched[parent] + direction / np.linalg.norm(direction) * 0.09
        joints[-1] = stretched
        target = robust_median_bone_lengths(joints, np.ones(count, dtype=bool))
        self.assertAlmostEqual(float(target[8]), 0.03, places=6)
        fixed = apply_fixed_bone_lengths(joints[-1], target)
        self.assertAlmostEqual(float(bone_lengths(fixed)[8]), 0.03, places=6)
        self.assertTrue(np.allclose(fixed[0], joints[-1, 0]))
        out = refine_hands(
            joints, np.full_like(joints, np.nan), _conf(count), np.zeros((count, 21)),
            _times(count), RefineParams(fixed_shape=True),
        )
        self.assertAlmostEqual(float(bone_lengths(out["left"]["joints"][-1])[8]), 0.03, places=6)
        self.assertTrue(np.allclose(out["left"]["joints"][0, 0], joints[0, 0]))

    def test_linear_shape_coefficients_match_a_fake_mano(self):
        base = np.linspace(0.02, 0.05, 21)
        base[0] = np.nan
        matrix = np.zeros((21, 2), dtype=np.float64)
        matrix[1:, 0] = 0.001
        matrix[1:, 1] = np.linspace(-0.0005, 0.0005, 20)
        truth = np.array([0.4, -0.2])

        def length_fn(betas):
            return base + matrix @ np.asarray(betas, dtype=np.float64)

        target = length_fn(truth)
        fitted = fit_shape_coefficients(target, length_fn, n_coeff=2, ridge=0.0)
        self.assertTrue(np.allclose(fitted, truth, atol=1e-6))


class WorldFrameTest(unittest.TestCase):
    def test_smoothing_follows_a_moving_camera_when_the_hand_is_still(self):
        count = 25
        hand = _hand(wrist=(0.0, 0.0, 0.6))
        world = np.repeat(hand.reshape(1, 21, 3), count, axis=0)
        poses = np.repeat(np.eye(4).reshape(1, 4, 4), count, axis=0)
        for index in range(count):
            poses[index, 0, 3] = 0.01 * index
        camera = np.stack([world_to_camera(world[index], poses[index]) for index in range(count)])
        empty = np.full_like(camera, np.nan)
        out = refine_hands(
            camera, empty, _conf(count), np.zeros((count, 21)), _times(count),
            RefineParams(smooth="one_euro", min_cutoff=0.5, beta=0.0),
            camera_poses=poses,
        )
        back = np.stack([
            transform_points(out["left"]["joints"][index], poses[index]) for index in range(count)
        ])
        self.assertTrue(np.allclose(back[:, 0], hand[0], atol=1e-6))
        self.assertEqual(out["coordinate"], "world_then_camera")


class QcAndConvertTest(unittest.TestCase):
    def _episode(self, confidence, filled):
        point = [0.0, 0.0, 1.0]
        joints = [[point] + [[0.02, 0.0, 1.0]] * 20]
        return {
            "num_frames": 1,
            "fps": 30.0,
            "image_width": 200,
            "image_height": 200,
            "camera_intrinsic": [[100.0, 0.0, 100.0], [0.0, 100.0, 100.0], [0.0, 0.0, 1.0]],
            "camera_poses": [np.eye(4).tolist()],
            "hands": {
                "left": {"joints": joints, "confidence": confidence, "filled": filled},
                "right": {"joints": joints, "confidence": confidence, "filled": filled},
            },
        }

    def test_filled_flag_is_out_of_frame_even_if_confidence_is_high(self):
        flags = frame_qc_flags(self._episode([0.99], [True]))
        self.assertTrue(bool(flags["hands_out_of_frame"][0]))
        kept = frame_qc_flags(self._episode([0.99], [False]))
        self.assertFalse(bool(kept["hands_out_of_frame"][0]))

    def test_convert_gap_fill_is_optional_and_qc_can_reject_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calib = {
                "image_width": 320,
                "image_height": 240,
                "K_left": np.array([[200.0, 0.0, 160.0], [0.0, 200.0, 120.0], [0.0, 0.0, 1.0]]),
                "K_right": np.array([[200.0, 0.0, 160.0], [0.0, 200.0, 120.0], [0.0, 0.0, 1.0]]),
                "dist_left": np.zeros(5),
                "dist_right": np.zeros(5),
                "R": np.eye(3),
                "T": np.array([-0.06, 0.0, 0.0]),
            }
            write_calibration_yaml(root / "calib.yaml", calib)
            (root / "metadata.json").write_text(json.dumps({
                "episode_id": "headcam/refine",
                "fps": 30,
                "task": "pick",
                "instruction": "拿起",
                "environment": "tabletop",
                "image_width": 320,
                "image_height": 240,
            }), encoding="utf-8")
            frames = []
            for index in range(5):
                if index == 2:
                    hand = None
                else:
                    wrist = [0.01 * index, 0.0, 0.6]
                    hand = {
                        "joints_cam": _hand(wrist=wrist).tolist(),
                        "keypoints_2d": None,
                        "confidence": [0.95] * 21,
                    }
                frames.append({"left": hand, "right": hand, "stereo_keypoints_right_view": None})
            (root / "hands.json").write_text(json.dumps({"frames": frames}), encoding="utf-8")
            plain = convert_headcam.build_episode(root)
            self.assertIsNone(plain["hand_pose"]["refine"])
            self.assertTrue(all(point[0] is None for point in plain["hands"]["left"]["joints"][2]))
            self.assertNotIn("filled", plain["hands"]["left"])
            filled = convert_headcam.build_episode(
                root, refine=RefineParams(gap_fill=True, max_gap=5),
            )
        self.assertEqual(validate_episode(filled), [])
        self.assertTrue(filled["hands"]["left"]["filled"][2])
        self.assertFalse(filled["hands"]["left"]["filled"][0])
        self.assertEqual(filled["hands"]["left"]["confidence"][2], 0.0)
        wrist = filled["hands"]["left"]["joints"][2][0]
        self.assertAlmostEqual(wrist[0], 0.02, places=5)
        flags = frame_qc_flags(filled)
        self.assertTrue(bool(flags["hands_out_of_frame"][2]))
        self.assertEqual(filled["hand_pose"]["refine"]["filled_frames"]["left"], 1)

    def test_cli_preset_turns_the_four_switches_on(self):
        parser_args = type("Args", (), {})()
        parser_args.refine = True
        parser_args.smooth = None
        parser_args.gap_fill = None
        parser_args.fixed_shape = False
        parser_args.no_lr_consistency = False
        parser_args.min_cutoff = None
        parser_args.beta = None
        parser_args.d_cutoff = None
        parser_args.kalman_accel_std = None
        parser_args.kalman_meas_std = None
        params = convert_headcam.refine_params_from_args(parser_args)
        self.assertEqual(params.smooth, "one_euro")
        self.assertTrue(params.gap_fill)
        self.assertTrue(params.fixed_shape)
        self.assertTrue(params.lr_consistency)
        parser_args.refine = False
        parser_args.smooth = "kalman"
        params = convert_headcam.refine_params_from_args(parser_args)
        self.assertEqual(params.smooth, "kalman")
        self.assertFalse(params.gap_fill)
        self.assertIsNone(convert_headcam.refine_params_from_args(type("Off", (), {
            "refine": False, "smooth": None, "gap_fill": None, "fixed_shape": False,
            "no_lr_consistency": False, "min_cutoff": None, "beta": None, "d_cutoff": None,
            "kalman_accel_std": None, "kalman_meas_std": None,
        })()))


class MetricTableTest(unittest.TestCase):
    def test_table_uses_root_relative_mean_and_does_not_invent_nan(self):
        intrinsic = np.array([[80.0, 0.0, 40.0], [0.0, 80.0, 30.0], [0.0, 0.0, 1.0]])
        gt = _hand(wrist=(0.0, 0.0, 1.0))
        pred = gt.copy()
        pred[:, 2] += 0.02
        sample = {
            "episode_id": "ep",
            "width": 80,
            "height": 60,
            "K": intrinsic,
            "gt": {"left": gt, "right": None},
            "gt_confidence": {"left": 0.99, "right": None},
            "pred": {"left": {"joints_cam": pred, "keypoints_2d": None}, "right": None},
        }
        report = evaluate_hand_frames([sample])
        self.assertAlmostEqual(report["root_relative_m_mean"], 0.0, places=6)
        self.assertAlmostEqual(report["wrist_error_m_mean"], 0.02, places=6)
        row = metrics_row("baseline", report, jitter=1.5, confident_report=report)
        text = format_ablation_table([row])
        self.assertIn("baseline", text)
        self.assertIn("0.00", text)
        self.assertIn("2.00", text)
        self.assertIn("1.500", text)
        empty = metrics_row("empty", {
            "detection_rate": float("nan"),
            "root_relative_m_mean": float("nan"),
            "root_relative_m_median": float("nan"),
            "wrist_error_m_median": float("nan"),
            "wrist_error_m_mean": float("nan"),
            "wrist_error_scaled_m_median": float("nan"),
            "wrist_error_scaled_m_mean": float("nan"),
            "swap_rate": float("nan"),
            "gt_visible": 0,
            "matched": 0,
            "exclude_joints": (),
        }, jitter=float("nan"), confident_report={"detection_rate": float("nan")})
        self.assertIn("n/a", format_ablation_table([empty]))
        self.assertEqual([name for name, _params in ablation_presets()], [
            "baseline", "smoothing", "gap_fill", "fixed_shape", "all",
        ])

    def test_held_out_list_keeps_the_previously_inspected_clip(self):
        self.assertIn(
            "test/open_close_insert_remove_case/8.hdf5",
            eval_hand_refine.EVAL_EPISODES,
        )
        self.assertEqual(len(eval_hand_refine.EVAL_EPISODES), 4)
        notebook = json.loads((ROOT / "notebooks" / "headcam_hand_pose_colab.ipynb").read_text(encoding="utf-8"))
        last = "".join(notebook["cells"][-1]["source"])
        self.assertIn("hand_track_refine", last)
        self.assertIn("HAND_REFINE", last)
        self.assertIn("evaluate_hand_frames", last)

    def test_fmt_delta_puts_percent_outside_the_format(self):
        self.assertEqual(eval_hand_refine._fmt_delta(37.2, 47.4, 1, "%"), "37.2 → 47.4%")
        self.assertEqual(eval_hand_refine._fmt_delta(float("nan"), 1.0, 2), "n/a")


class ZipRangeDownloadTest(unittest.TestCase):
    def _source(self, force_zip64):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            for name, payload in (
                ("test/add_remove_lid/8.hdf5", b"hdf5-bytes"),
                ("test/add_remove_lid/8.mp4", b"mp4-bytes-longer"),
            ):
                with archive.open(name, "w", force_zip64=force_zip64) as member:
                    member.write(payload)
        return eval_hand_refine.BytesRangeSource(buffer.getvalue())

    def test_range_source_extracts_deflated_and_zip64_members(self):
        for force_zip64 in (False, True):
            source = self._source(force_zip64)
            index = eval_hand_refine.read_zip_index(source)
            self.assertEqual(
                eval_hand_refine.extract_member(source, index["test/add_remove_lid/8.hdf5"]),
                b"hdf5-bytes",
            )
            self.assertEqual(
                eval_hand_refine.extract_member(source, index["test/add_remove_lid/8.mp4"]),
                b"mp4-bytes-longer",
            )

    def test_ensure_episodes_skips_files_already_present(self):
        source = self._source(True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episodes = ("test/add_remove_lid/8.hdf5",)
            written = eval_hand_refine.ensure_episodes(root, episodes, source=source)
            self.assertEqual(written, [
                "test/add_remove_lid/8.hdf5",
                "test/add_remove_lid/8.mp4",
            ])
            self.assertEqual((root / "test/add_remove_lid/8.hdf5").read_bytes(), b"hdf5-bytes")
            again = eval_hand_refine.ensure_episodes(
                root,
                episodes,
                source=eval_hand_refine.BytesRangeSource(b"not-a-zip"),
            )
            self.assertEqual(again, [])

    def test_missing_member_names_the_path(self):
        source = self._source(False)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                eval_hand_refine.ensure_episodes(
                    directory,
                    ("test/open_close_insert_remove_case/8.hdf5",),
                    source=source,
                )


class JitterSanityTest(unittest.TestCase):
    def test_constant_velocity_has_no_acceleration(self):
        count = 10
        points = np.stack([np.array([0.01 * index, 0.0, 0.0]) for index in range(count)])
        self.assertAlmostEqual(mean_accel(points, _times(count), np.ones(count, dtype=bool)), 0.0, places=6)
        noisy = points.copy()
        noisy[4, 0] += 0.05
        self.assertGreater(len(accel_samples(noisy, _times(count), np.ones(count, dtype=bool))), 0)
        self.assertGreater(mean_accel(noisy, _times(count), np.ones(count, dtype=bool)), 1.0)


if __name__ == "__main__":
    unittest.main()
