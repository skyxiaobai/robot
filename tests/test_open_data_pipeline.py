# -*- coding: utf-8 -*-
"""开放数据集流水线：统一 episode、EgoDex 适配、QC/产出率、覆盖度、四级标注。"""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import h5py  # noqa: E402

from egodata.coverage import (  # noqa: E402
    accumulate_coverage,
    coarse_object_class,
    coverage_report,
    finalize_coverage,
    normalize_action,
    normalize_environment,
    normalize_object_name,
    write_coverage_reports,
)
from egodata.egodex import (  # noqa: E402
    active_language,
    convert_tree,
    load_episode_hdf5,
)
from egodata.labels import validate_annotation  # noqa: E402
from egodata.qc import qc_episode, write_yield_reports, yield_report  # noqa: E402
from egodata.schema import (  # noqa: E402
    MEDIAPIPE_21,
    load_episode,
    make_quaternions_continuous,
    rotmat_to_quat_xyzw,
    save_episode,
    validate_episode,
)


def _se3(translation, rotation=None):
    pose = np.eye(4, dtype=np.float32)
    if rotation is not None:
        pose[:3, :3] = np.asarray(rotation, dtype=np.float32)
    pose[:3, 3] = np.asarray(translation, dtype=np.float32)
    return pose


def _write_egodex_hdf5(path, n, wrist_world, camera_poses, confidence=0.99,
                       which="1", extra_joints=True, with_confidence=True,
                       wrist_rotations=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    intrinsic = np.array(
        [[736.6339, 0, 960], [0, 736.6339, 540], [0, 0, 1]], dtype=np.float32
    )
    with h5py.File(path, "w") as handle:
        handle.create_dataset("camera/intrinsic", data=intrinsic)
        cam = np.stack(camera_poses).astype(np.float32)
        handle.create_dataset("transforms/camera", data=cam)
        for side, prefix in (("left", "left"), ("right", "right")):
            names = [
                f"{prefix}Hand",
                f"{prefix}ThumbKnuckle",
                f"{prefix}ThumbIntermediateBase",
                f"{prefix}ThumbIntermediateTip",
                f"{prefix}ThumbTip",
                f"{prefix}IndexFingerKnuckle",
                f"{prefix}IndexFingerIntermediateBase",
                f"{prefix}IndexFingerIntermediateTip",
                f"{prefix}IndexFingerTip",
                f"{prefix}MiddleFingerKnuckle",
                f"{prefix}MiddleFingerIntermediateBase",
                f"{prefix}MiddleFingerIntermediateTip",
                f"{prefix}MiddleFingerTip",
                f"{prefix}RingFingerKnuckle",
                f"{prefix}RingFingerIntermediateBase",
                f"{prefix}RingFingerIntermediateTip",
                f"{prefix}RingFingerTip",
                f"{prefix}LittleFingerKnuckle",
                f"{prefix}LittleFingerIntermediateBase",
                f"{prefix}LittleFingerIntermediateTip",
                f"{prefix}LittleFingerTip",
            ]
            if not extra_joints:
                names = names[:1]
            for index, name in enumerate(names):
                # 只沿 z 错开，食指尖与手腕共享 xy，便于核对 MediaPipe 序号。
                offset = np.array([0.0, 0.0, 0.02 * index], dtype=np.float32)
                series = []
                for frame_index in range(n):
                    rotation = None if wrist_rotations is None else wrist_rotations[frame_index]
                    series.append(_se3(np.asarray(wrist_world[frame_index], dtype=np.float32) + offset, rotation))
                handle.create_dataset(f"transforms/{name}", data=np.stack(series).astype(np.float32))
                if with_confidence:
                    conf = np.full((n,), confidence, dtype=np.float32)
                    handle.create_dataset(f"confidences/{name}", data=conf)
        handle.attrs["llm_type"] = "reversible"
        handle.attrs["which_llm_description"] = which
        handle.attrs["llm_description"] = "Open the case, insert the pad."
        handle.attrs["llm_description2"] = "Open the case, remove the pad."
        handle.attrs["llm_objects"] = np.array(["case", "pad"], dtype=object)
        handle.attrs["llm_verbs"] = np.array(["open", "remove"], dtype=object)
        handle.attrs["environment"] = "table:wood, position:sitting, background:brown"
        handle.attrs["task"] = path.parent.name


def _moving_episode(episode_id, n=30, speed=0.002):
    """相机静止、手腕沿 x 缓慢移动，投影落在画面中央。"""
    poses = [_se3((0, 0, 0)) for _ in range(n)]
    wrists = [(speed * i, 0.0, 1.0) for i in range(n)]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "basic_pick_place" / "0.hdf5"
        _write_egodex_hdf5(path, n, wrists, poses)
        episode = load_episode_hdf5(path)
    episode["episode_id"] = episode_id
    return episode


def _static_episode(episode_id, n=60):
    poses = [_se3((0, 0, 0)) for _ in range(n)]
    wrists = [(0.0, 0.0, 1.0) for _ in range(n)]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "basic_pick_place" / "1.hdf5"
        _write_egodex_hdf5(path, n, wrists, poses)
        episode = load_episode_hdf5(path)
    episode["episode_id"] = episode_id
    return episode


class SchemaTest(unittest.TestCase):
    def test_identity_and_z_rotation_quaternion(self):
        identity = rotmat_to_quat_xyzw(np.eye(3))
        self.assertTrue(np.allclose(identity, [0, 0, 0, 1], atol=1e-6))
        quarter = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        quat = rotmat_to_quat_xyzw(quarter)
        self.assertTrue(np.allclose(quat, [0, 0, math.sqrt(0.5), math.sqrt(0.5)], atol=1e-6))

    def test_quaternion_sequence_stays_in_one_hemisphere(self):
        quats = []
        for deg in (20, 40):
            angle = math.radians(deg)
            cosine, sine = math.cos(angle), math.sin(angle)
            rotation = np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]], dtype=float)
            quats.append(rotmat_to_quat_xyzw(rotation))
        flipped = np.stack([quats[0], -np.asarray(quats[1])])
        self.assertLess(float(np.dot(flipped[0], flipped[1])), 0.0)
        aligned = make_quaternions_continuous(flipped)
        self.assertGreater(float(np.dot(aligned[0], aligned[1])), 0.0)

    def test_mediapipe_order_has_wrist_then_index_tip(self):
        self.assertEqual(len(MEDIAPIPE_21), 21)
        self.assertEqual(MEDIAPIPE_21[0], "wrist")
        self.assertEqual(MEDIAPIPE_21[8], "index_tip")

    def test_roundtrip_and_reject_short_wrist(self):
        episode = _moving_episode("roundtrip")
        errors = validate_episode(episode)
        self.assertEqual(errors, [], errors)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ep.json"
            save_episode(episode, path)
            loaded = load_episode(path)
        self.assertEqual(loaded["episode_id"], "roundtrip")
        self.assertEqual(len(loaded["hands"]["left"]["joints"][0]), 21)
        broken = json.loads(json.dumps(episode))
        broken["hands"]["left"]["wrist_pose"][0] = [0, 0, 0]
        self.assertTrue(validate_episode(broken))


class EgoDexAdapterTest(unittest.TestCase):
    def test_world_wrist_language_and_projection_center(self):
        n = 4
        wrists = [(0.0, 0.0, 1.0) for _ in range(n)]
        poses = [_se3((0, 0, 0)) for _ in range(n)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "open_close_insert_remove_case" / "8.hdf5"
            _write_egodex_hdf5(path, n, wrists, poses, which="2")
            episode = load_episode_hdf5(path)
        self.assertEqual(episode["coordinate_frame"], "arkit_world")
        self.assertEqual(episode["fps"], 30.0)
        self.assertEqual(episode["image_width"], 1920)
        self.assertEqual(episode["image_height"], 1080)
        wrist = episode["hands"]["left"]["joints"][0][0]
        self.assertTrue(np.allclose(wrist, [0, 0, 1], atol=1e-5))
        index_tip = episode["hands"]["left"]["joints"][0][8]
        self.assertTrue(np.allclose(index_tip[:2], [0.0, 0.0], atol=1e-5))
        self.assertGreater(index_tip[2], 1.0)
        pose = episode["hands"]["right"]["wrist_pose"][0]
        self.assertEqual(len(pose), 7)
        self.assertTrue(np.allclose(pose[:3], [0, 0, 1], atol=1e-5))
        self.assertIn("remove the pad", episode["annotation"]["task"]["instruction"])
        self.assertEqual(episode["annotation"]["task"]["name"], "open_close_insert_remove_case")
        self.assertEqual(
            episode["coverage"]["environment"],
            "tabletop|table=wood|position=sitting|background=brown",
        )
        self.assertIn("case", episode["coverage"]["objects"])
        self.assertIn("open", episode["coverage"]["action_types"])
        self.assertEqual(episode["annotation"]["subtasks"], [])
        self.assertEqual(episode["annotation"]["instructions"], [])

    def test_active_language_direction(self):
        self.assertEqual(
            active_language({
                "llm_type": "reversible",
                "which_llm_description": "2",
                "llm_description": "insert",
                "llm_description2": "remove",
            }),
            "remove",
        )
        self.assertEqual(
            active_language({"llm_type": "directional", "llm_description": "pour water"}),
            "pour water",
        )

    def test_convert_tree_writes_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "test"
            _write_egodex_hdf5(
                src / "pour" / "0.hdf5",
                2,
                [(0, 0, 1), (0.01, 0, 1)],
                [_se3((0, 0, 0)), _se3((0, 0, 0))],
            )
            out = root / "unified"
            written = convert_tree(src, out, limit=1)
            self.assertEqual(len(written), 1)
            loaded = load_episode(written[0])
            self.assertEqual(loaded["source"], "egodex")
            self.assertTrue(loaded["episode_id"].endswith("pour/0"))

    def test_convert_tree_follows_symlinked_task_folders(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real = root / "store" / "pour"
            _write_egodex_hdf5(
                real / "0.hdf5",
                2,
                [(0, 0, 1), (0.01, 0, 1)],
                [_se3((0, 0, 0)), _se3((0, 0, 0))],
            )
            src = root / "input"
            src.mkdir()
            (src / "linked_pour").symlink_to(real, target_is_directory=True)
            (src / "self").symlink_to(src)
            written = convert_tree(src, root / "unified")
            self.assertEqual(len(written), 1)
            loaded = load_episode(written[0])
            self.assertIn("linked_pour/0", loaded["episode_id"])
            self.assertTrue(list(src.rglob("*.hdf5")) == [])

    def test_missing_confidence_is_unknown_not_zero(self):
        n = 8
        wrists = [(0.002 * i, 0.0, 1.0) for i in range(n)]
        poses = [_se3((0, 0, 0)) for _ in range(n)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pour" / "0.hdf5"
            _write_egodex_hdf5(path, n, wrists, poses, with_confidence=False)
            episode = load_episode_hdf5(path)
        self.assertIsNone(episode["hands"]["left"]["confidence"][0])
        self.assertTrue(episode["hands"]["left"]["valid"][0])
        result = qc_episode(episode)
        self.assertEqual(result["flags"]["hands_out_of_frame"], 0)
        self.assertNotIn("hands_out_of_frame", result["reasons"])

    def test_loaded_wrist_quaternions_are_continuous(self):
        n = 4
        rotations = []
        for deg in (150, 170, 190, 210):
            angle = math.radians(deg)
            cosine, sine = math.cos(angle), math.sin(angle)
            rotations.append(np.array(
                [[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]], dtype=float,
            ))
        wrists = [(0.0, 0.0, 1.0) for _ in range(n)]
        poses = [_se3((0, 0, 0)) for _ in range(n)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pour" / "0.hdf5"
            _write_egodex_hdf5(path, n, wrists, poses, wrist_rotations=rotations)
            episode = load_episode_hdf5(path)
        quats = np.asarray([pose[3:] for pose in episode["hands"]["left"]["wrist_pose"]], dtype=float)
        dots = [float(np.dot(quats[index], quats[index + 1])) for index in range(n - 1)]
        self.assertTrue(all(dot > 0.0 for dot in dots), dots)

    def test_parallel_convert_matches_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "test"
            for name, task in (("pour", "pour"), ("stack", "stack")):
                _write_egodex_hdf5(
                    src / task / "0.hdf5",
                    2,
                    [(0, 0, 1), (0.01, 0, 1)],
                    [_se3((0, 0, 0)), _se3((0, 0, 0))],
                )
            single = convert_tree(src, root / "one", workers=1)
            multi = convert_tree(src, root / "many", workers=2)
            self.assertEqual(
                sorted(path.relative_to(root / "one").as_posix() for path in single),
                sorted(path.relative_to(root / "many").as_posix() for path in multi),
            )
            self.assertEqual(load_episode(single[0])["episode_id"], load_episode(
                root / "many" / single[0].relative_to(root / "one")
            )["episode_id"])


class QcYieldTest(unittest.TestCase):
    def test_clean_episode_is_accepted(self):
        episode = _moving_episode("clean")
        result = qc_episode(episode)
        self.assertTrue(result["accepted"], result)
        self.assertEqual(result["flags"]["hands_out_of_frame"], 0)
        self.assertEqual(result["flags"]["view_drift"], 0)
        self.assertEqual(result["flags"]["blur"], 0)
        self.assertEqual(result["flags"]["staged_static"], 0)

    def test_wrist_outside_image_rejects_clip(self):
        n = 10
        # 相机在原点朝 +Z，手腕在 x=10m，投影远在画面外。
        wrists = [(10.0, 0.0, 1.0) for _ in range(n)]
        poses = [_se3((0, 0, 0)) for _ in range(n)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pour" / "0.hdf5"
            _write_egodex_hdf5(path, n, wrists, poses)
            episode = load_episode_hdf5(path)
        result = qc_episode(episode)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["flags"]["hands_out_of_frame"], n)
        self.assertIn("hands_out_of_frame", result["reasons"])

    def test_low_confidence_counts_as_hand_out_of_frame(self):
        n = 8
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pour" / "0.hdf5"
            _write_egodex_hdf5(
                path, n, [(0, 0, 1)] * n, [_se3((0, 0, 0))] * n, confidence=0.1,
            )
            episode = load_episode_hdf5(path)
        result = qc_episode(episode)
        self.assertEqual(result["flags"]["hands_out_of_frame"], n)
        self.assertFalse(result["accepted"])

    def test_fast_camera_spin_is_blur(self):
        n = 6
        wrists = []
        poses = []
        for i in range(n):
            angle = i * 1.2  # 每帧约 69 度，30fps 下角速度远超模糊阈值
            c, s = math.cos(angle), math.sin(angle)
            rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)
            poses.append(_se3((0, 0, 0), rotation))
            # 手腕始终在相机前方 1m，避免被判出画。
            wrists.append(rotation @ np.array([0.0, 0.0, 1.0]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pour" / "0.hdf5"
            _write_egodex_hdf5(path, n, wrists, poses)
            episode = load_episode_hdf5(path)
        result = qc_episode(episode)
        self.assertGreater(result["flags"]["blur"], 0)
        self.assertFalse(result["accepted"])
        self.assertIn("blur", result["reasons"])

    def test_looking_away_is_view_drift(self):
        episode = _moving_episode("gaze", n=10, speed=0.0)
        # 手腕在光轴附近但角度超过调用方给定的窄阈值，仍在画面内。
        for frame in episode["hands"]["left"]["joints"]:
            frame[0] = [0.3, 0.0, 1.0]
        for frame in episode["hands"]["right"]["joints"]:
            frame[0] = [0.3, 0.0, 1.0]
        for index, pose in enumerate(episode["hands"]["left"]["wrist_pose"]):
            episode["hands"]["left"]["wrist_pose"][index] = [0.3, 0.0, 1.0, 0, 0, 0, 1]
            episode["hands"]["right"]["wrist_pose"][index] = [0.3, 0.0, 1.0, 0, 0, 0, 1]
        result = qc_episode(episode, view_angle_deg=10)
        self.assertEqual(result["flags"]["hands_out_of_frame"], 0)
        self.assertEqual(result["flags"]["view_drift"], 10)
        self.assertIn("view_drift", result["reasons"])

    def test_held_still_segment_is_staged_static(self):
        episode = _static_episode("still")
        result = qc_episode(episode)
        self.assertEqual(result["flags"]["staged_static"], episode["num_frames"])
        self.assertFalse(result["accepted"])
        self.assertIn("staged_static", result["reasons"])

    def test_yield_is_accepted_frames_over_raw_frames(self):
        clean = _moving_episode("clean-yield", n=30)
        still = _static_episode("still-yield", n=30)
        report = yield_report([qc_episode(clean), qc_episode(still)])
        self.assertAlmostEqual(report["yield"], 0.5)
        self.assertEqual(report["raw_frames"], 60)
        self.assertEqual(report["usable_frames"], 30)
        self.assertEqual(report["rejected_episodes"], 1)

    def test_html_and_csv(self):
        report = yield_report([qc_episode(_moving_episode("ok-html"))])
        with tempfile.TemporaryDirectory() as tmp:
            html_path = Path(tmp) / "yield.html"
            csv_path = Path(tmp) / "yield.csv"
            write_yield_reports(report, html_path, csv_path)
            html = html_path.read_text(encoding="utf-8")
            csv_text = csv_path.read_text(encoding="utf-8")
        self.assertIn("产出率", html)
        self.assertIn("ok-html", csv_text)
        self.assertIn("accepted", csv_text)


class CoverageTest(unittest.TestCase):
    def test_counts_and_taxonomy_gap(self):
        first = _moving_episode("a")
        second = _moving_episode("b")
        second["coverage"] = {
            "environment": "kitchen",
            "objects": ["cup"],
            "object_classes": ["tableware"],
            "task": "pour_water",
            "action_types": ["pour"],
        }
        second["annotation"]["task"]["name"] = "pour_water"
        report = coverage_report([first, second])
        env = {row["value"]: row["episodes"] for row in report["counts"] if row["axis"] == "environment"}
        detail = first["coverage"]["environment"]
        self.assertTrue(detail.startswith("tabletop|"))
        self.assertEqual(env[detail], 1)
        self.assertEqual(env["kitchen"], 1)
        gaps = {(row["axis"], row["value"]) for row in report["gaps"]}
        self.assertIn(("environment", "outdoor"), gaps)
        self.assertNotIn(("environment", "kitchen"), gaps)
        self.assertNotIn(("environment", "tabletop"), gaps)
        self.assertEqual(normalize_object_name("plates"), "plate")
        self.assertEqual(normalize_object_name("plushie"), "plush")
        self.assertEqual(normalize_object_name("square table"), normalize_object_name("squaretable"))
        self.assertEqual(normalize_action("disassemble"), "assemble")
        self.assertEqual(normalize_action("uncharge"), "charge")
        self.assertEqual(normalize_action("unzip"), "zip")
        self.assertEqual(normalize_action("scoop"), "scoop")
        self.assertEqual(normalize_action("gather"), "pick")
        self.assertEqual(normalize_action("take"), "pick")
        self.assertEqual(normalize_action("unstock"), "remove")
        self.assertEqual(normalize_action("stock"), "place")
        self.assertEqual(normalize_action("add"), "place")
        self.assertEqual(normalize_action("push"), "push")
        self.assertEqual(normalize_action("roll"), "roll")
        self.assertEqual(normalize_action("color"), "color")
        self.assertEqual(normalize_action("use"), "other")
        self.assertEqual(coarse_object_class("plush"), "toy")
        self.assertEqual(coarse_object_class("shoelace"), "cloth")
        self.assertEqual(coarse_object_class("battery"), "electronics")
        self.assertEqual(coarse_object_class("ice"), "food")
        self.assertEqual(coarse_object_class("device"), "electronics")
        self.assertEqual(
            normalize_environment("table:wood, position:sitting, background:brown"),
            "tabletop|table=wood|position=sitting|background=brown",
        )
        self.assertEqual(
            normalize_environment("tablecloth:blue, position:sitting, background:pink"),
            "tabletop|tablecloth=blue|position=sitting|background=pink",
        )
        self.assertEqual(
            normalize_environment("tablecloth:lavendar, background:lavender"),
            "tabletop|tablecloth=lavender|background=lavender",
        )
        self.assertEqual(normalize_environment("kitchen"), "kitchen")
        streamed = {}
        accumulate_coverage(streamed, first)
        accumulate_coverage(streamed, second)
        streamed_report = finalize_coverage(streamed, 2)
        self.assertEqual(streamed_report["gaps"], report["gaps"])
        self.assertEqual(streamed_report["episodes"], report["episodes"])
        with tempfile.TemporaryDirectory() as tmp:
            html_path = Path(tmp) / "coverage.html"
            csv_path = Path(tmp) / "coverage.csv"
            write_coverage_reports(report, html_path, csv_path)
            html = html_path.read_text(encoding="utf-8")
            csv_text = csv_path.read_text(encoding="utf-8")
        self.assertIn("覆盖", html)
        self.assertIn("outdoor", csv_text)
        self.assertIn("pour_water", csv_text)


class DemoPipelineTest(unittest.TestCase):
    def test_synthetic_demo_prints_yield(self):
        import demo_open_dataset_pipeline as demo

        with tempfile.TemporaryDirectory() as tmp:
            code = demo.main(["--out", str(Path(tmp) / "demo")])
            html = (Path(tmp) / "demo" / "yield_report.html").read_text(encoding="utf-8")
            coverage = (Path(tmp) / "demo" / "coverage_counts.csv").read_text(encoding="utf-8")
        self.assertEqual(code, 0)
        self.assertIn("50.0%", html)
        self.assertIn("basic_pick_place", coverage)


class HierarchyValidatorTest(unittest.TestCase):
    def _full(self):
        return {
            "environment": {"name": "tabletop", "detail": "table:wood", "source": "egodex_attr"},
            "task": {"name": "open_case", "instruction": "打开盒子"},
            "subtasks": [
                {"t_start": 0.0, "t_end": 1.0, "text": "打开盖子"},
                {"t_start": 1.0, "t_end": 2.0, "text": "取出物品"},
            ],
            "instructions": [
                {"t_start": 0.0, "t_end": 1.0, "hand": "right", "text": "右手扳开卡扣"},
                {"t_start": 1.0, "t_end": 2.0, "hand": "left", "text": "左手取出物品"},
            ],
        }

    def test_strict_full_annotation_passes(self):
        result = validate_annotation(self._full(), duration_s=2.0, strict=True)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["missing_levels"], [])

    def test_missing_subtask_and_instruction_is_incomplete(self):
        ann = self._full()
        ann["subtasks"] = []
        ann["instructions"] = []
        loose = validate_annotation(ann, duration_s=2.0, strict=False)
        self.assertTrue(loose["ok"], loose)
        self.assertEqual(loose["missing_levels"], ["subtask", "instruction"])
        strict = validate_annotation(ann, duration_s=2.0, strict=True)
        self.assertFalse(strict["ok"])
        self.assertIn("subtask", strict["missing_levels"])

    def test_gap_between_subtasks_is_invalid(self):
        ann = self._full()
        ann["subtasks"][1]["t_start"] = 1.4
        result = validate_annotation(ann, duration_s=2.0, strict=True)
        self.assertFalse(result["ok"])
        self.assertTrue(any("SUBTASK" in item for item in result["errors"]))

    def test_unknown_hand_is_invalid(self):
        ann = self._full()
        ann["instructions"][0]["hand"] = "middle"
        result = validate_annotation(ann, duration_s=2.0, strict=False)
        self.assertFalse(result["ok"])
        self.assertTrue(any("hand" in item for item in result["errors"]))

    def test_example_file_validates(self):
        example = json.loads((ROOT / "examples" / "hierarchy_annotation.json").read_text(encoding="utf-8"))
        result = validate_annotation(example, duration_s=example["duration_s"], strict=True)
        self.assertTrue(result["ok"], result)


if __name__ == "__main__":
    unittest.main()
