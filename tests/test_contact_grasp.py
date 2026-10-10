# -*- coding: utf-8 -*-
"""物体 6DoF、接触、抓取：schema、HOT3D 真值、启发式、评测和 LeRobot 掩码。"""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from egodata.interaction import (  # noqa: E402
    FINGERTIPS,
    GT_CONTACT_M,
    HEURISTIC_DEFAULTS,
    derive_gt_interaction,
    estimate_interaction,
    evaluate_interaction,
    events_from_labels,
    min_surface_distance,
    synthetic_disagreement,
)
from egodata.lerobot_export import (  # noqa: E402
    CONTACT_DIM,
    GRASP_DIM,
    OBJECT_POSE_DIM,
    OBJECT_SLOTS,
    export_lerobot,
    pack_contact,
    pack_grasp,
    pack_object_pose,
)
from egodata.qc import qc_episode  # noqa: E402
from egodata.schema import (  # noqa: E402
    EVENT_TYPES,
    GRASP_STATES,
    empty_interaction,
    validate_episode,
)
from headcam.hot3d_adapter import (  # noqa: E402
    export_hot3d_objects,
    parse_objects_json,
    se3_from_hot3d,
)


def _wrist(index):
    return [0.01 * index, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


def _joints(fingertip=None):
    joints = [[0.2, 0.0, 0.2] for _ in range(21)]
    joints[0] = [0.0, 0.0, 0.2]
    if fingertip is not None:
        for index in FINGERTIPS:
            joints[index] = list(fingertip)
        joints[4] = [fingertip[0] - 0.02, fingertip[1], fingertip[2]]
        joints[8] = [fingertip[0] + 0.02, fingertip[1], fingertip[2]]
    return joints


def _episode(n=4, objects=None, contact=None, grasp=None, events=None):
    hands = {"left": {"joints": [], "wrist_pose": [], "confidence": []},
             "right": {"joints": [], "wrist_pose": [], "confidence": []}}
    for index in range(n):
        for side in ("left", "right"):
            hands[side]["joints"].append(_joints([0.0, 0.0, 0.2]))
            hands[side]["wrist_pose"].append(_wrist(index))
            hands[side]["confidence"].append(0.9)
    episode = {
        "schema_version": "1.0",
        "episode_id": "demo",
        "source": "test",
        "fps": 10.0,
        "coordinate_frame": "world",
        "image_width": 16,
        "image_height": 16,
        "num_frames": n,
        "timestamps": [index / 10.0 for index in range(n)],
        "camera_intrinsic": [[1, 0, 8], [0, 1, 8], [0, 0, 1]],
        "camera_poses": [np.eye(4).tolist() for _ in range(n)],
        "hands": hands,
        "annotation": {
            "environment": {"name": "tabletop", "detail": "", "source": "test"},
            "task": {"name": "pick", "instruction": "拿起杯子"},
            "subtasks": [],
            "instructions": [],
        },
        "coverage": {"environment": "tabletop", "objects": ["cup"], "task": "pick", "action_types": ["pick"]},
    }
    blank = empty_interaction(n)
    episode["objects"] = objects if objects is not None else blank["objects"]
    episode["contact"] = contact if contact is not None else blank["contact"]
    episode["grasp"] = grasp if grasp is not None else blank["grasp"]
    episode["events"] = events if events is not None else blank["events"]
    return episode


def _yield_csv(path, episode_id):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode_id", "num_frames", "accepted", "reasons"])
        writer.writeheader()
        writer.writerow({"episode_id": episode_id, "num_frames": 4, "accepted": "yes", "reasons": ""})


class SchemaTest(unittest.TestCase):
    def test_empty_interaction_validates(self):
        episode = _episode()
        self.assertEqual(validate_episode(episode), [])
        self.assertEqual(GRASP_STATES, ("open", "pre_grasp", "grasp", "release"))
        self.assertEqual(EVENT_TYPES, ("contact_start", "contact_end", "grasp", "release"))

    def test_missing_objects_is_an_error(self):
        episode = _episode()
        del episode["objects"]
        self.assertTrue(any("objects" in item for item in validate_episode(episode)))

    def test_bad_grasp_state_and_short_pose_fail(self):
        episode = _episode(n=2, objects=[{
            "id": "cup",
            "category": "tableware",
            "source": "hot3d",
            "pose": [[0, 0, 0, 0, 0, 0, 1], [0, 0, 0]],
            "confidence": [1.0, 1.0],
            "valid": [True, True],
        }])
        episode["grasp"]["right"]["state"] = ["holding", "open"]
        episode["grasp"]["right"]["valid"] = [True, True]
        errors = validate_episode(episode)
        self.assertTrue(any("grasp" in item for item in errors))
        self.assertTrue(any("pose" in item for item in errors))

    def test_event_timestamp_must_fall_inside_the_episode(self):
        episode = _episode(events=[{
            "type": "contact_start",
            "hand": "right",
            "object_id": "cup",
            "timestamp": 99.0,
        }])
        self.assertTrue(any("timestamp" in item for item in validate_episode(episode)))


class Hot3dPoseTest(unittest.TestCase):
    def test_wxyz_translation_becomes_xyzw_pose(self):
        matrix = se3_from_hot3d({
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "translation_xyz": [0.1, 0.2, 0.3],
        })
        self.assertTrue(np.allclose(matrix[:3, 3], [0.1, 0.2, 0.3]))
        self.assertTrue(np.allclose(matrix[:3, :3], np.eye(3)))
        half = float(np.sqrt(0.5))
        spun = se3_from_hot3d({
            "quaternion_wxyz": [half, 0.0, 0.0, half],
            "translation_xyz": [0.0, 0.0, 0.0],
        })
        self.assertTrue(np.allclose(spun[:3, :3], [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-6))

    def test_parse_instance_list_and_skip_missing_frame(self):
        payload = {
            "12": [{
                "object_bop_id": 12,
                "object_name": "mug",
                "T_world_from_object": {
                    "quaternion_wxyz": [1, 0, 0, 0],
                    "translation_xyz": [0.1, 0.2, 0.3],
                },
            }]
        }
        found = parse_objects_json(payload)
        self.assertEqual(found[0]["id"], "12")
        self.assertEqual(found[0]["category"], "mug")
        self.assertTrue(np.allclose(found[0]["pose7"][:3], [0.1, 0.2, 0.3]))
        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp)
            (clip / "000000.objects.json").write_text(json.dumps(payload), encoding="utf-8")
            (clip / "000001.objects.json").write_text("{}", encoding="utf-8")
            vertices = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.0, 0.02, 0.0]], dtype=float)
            faces = np.array([[0, 1, 2]], dtype=int)
            hand = np.zeros((21, 3))
            hand[:] = [0.1, 0.2, 0.3]
            hand[8] = [0.11, 0.21, 0.301]
            summary = export_hot3d_objects(
                clip,
                clip / "out",
                keys=["000000", "000001"],
                timestamps=[0.0, 0.1],
                hand_points={"left": [None, None], "right": [hand, None]},
                surfaces={"12": {"vertices": vertices, "faces": faces}},
            )
            self.assertEqual(summary["objects"], 1)
            tracks = json.loads((clip / "out" / "gt" / "objects_gt.json").read_text(encoding="utf-8"))
            interaction = json.loads((clip / "out" / "gt" / "interaction_gt.json").read_text(encoding="utf-8"))
        self.assertEqual(tracks["objects"][0]["source"], "hot3d")
        self.assertEqual(tracks["objects"][0]["valid"], [True, False])
        self.assertEqual(interaction["contact"]["source"], "hot3d_mesh")
        self.assertEqual(interaction["contact"]["right"]["object_id"][0], "12")
        self.assertTrue(interaction["contact"]["right"]["valid"][0])
        self.assertFalse(interaction["contact"]["right"]["valid"][1])
        self.assertIn(interaction["grasp"]["right"]["state"][0], GRASP_STATES)


class DistanceAndHeuristicTest(unittest.TestCase):
    def test_point_above_triangle_is_the_height(self):
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        faces = np.array([[0, 1, 2]])
        distance = min_surface_distance(
            np.array([[0.2, 0.2, 0.004]]),
            vertices=vertices,
            faces=faces,
        )
        self.assertAlmostEqual(float(distance[0]), 0.004, places=4)

    def test_sphere_contact_grasp_and_release(self):
        # 球心在原点，半径 2 cm。指尖放在半径 2.4 cm 上，离表面 4 mm，小于 1 cm。
        timestamps = [0.0, 0.1, 0.2]
        tip = [0.024, 0.0, 0.0]
        far = [0.20, 0.0, 0.0]
        joints = [
            _joints(far),
            _joints(tip),
            _joints([0.08, 0.0, 0.0]),
        ]
        # 第二帧拇指和食指相距 4 cm，五指都在表面上。
        objects = [{
            "id": "ball",
            "category": "toy",
            "source": "test",
            "pose": [[0, 0, 0, 0, 0, 0, 1]] * 3,
            "confidence": [1, 1, 1],
            "valid": [True, True, True],
        }]
        surfaces = {"ball": {"radius_m": 0.02, "center_local": [0, 0, 0]}}
        hands = {
            "left": {"joints": [None, None, None], "confidence": [None, None, None]},
            "right": {"joints": joints, "confidence": [0.9, 0.9, 0.9]},
        }
        # 第一帧在远处。第二帧需要有前一帧才能算靠近速度；这里第二帧已经贴在表面上，应判抓取。
        # 第三帧离开，前一帧是抓取，应判放开。
        # 这三帧测的是当帧阈值，不经过默认的时间滤波。
        result = estimate_interaction(hands, objects, timestamps, surfaces, temporal_filter=False)
        right = result["grasp"]["right"]["state"]
        contact = result["contact"]["right"]["object_id"]
        self.assertEqual(right[0], "open")
        self.assertIsNone(contact[0])
        self.assertEqual(right[1], "grasp")
        self.assertEqual(contact[1], "ball")
        self.assertEqual(right[2], "release")
        self.assertEqual(result["contact"]["source"], "heuristic")
        types = [item["type"] for item in result["events"] if item["hand"] == "right"]
        self.assertEqual(types, ["contact_start", "grasp", "contact_end", "release"])

    def test_depth_and_external_hook(self):
        timestamps = [0.0, 0.1]
        joints = [_joints([0.0, 0.0, 1.0]), _joints([0.0, 0.0, 1.0])]
        objects = [{
            "id": "cup",
            "category": "tableware",
            "source": "foundationpose",
            "pose": [[0, 0, 1.0, 0, 0, 0, 1], [0, 0, 1.0, 0, 0, 0, 1]],
            "confidence": [0.8, 0.8],
            "valid": [True, True],
        }]
        surfaces = {"cup": {"depth_m": [1.0, 1.0]}}
        hands = {
            "left": {"joints": [None, None], "confidence": [None, None]},
            "right": {"joints": joints, "confidence": [0.9, 0.9]},
        }
        camera = [np.eye(4).tolist(), np.eye(4).tolist()]

        def pose_hook(_hands, _timestamps):
            return objects

        def contact_hook(side, frame_index, _joints, _objects):
            if side == "right" and frame_index == 0:
                return {"object_id": "cup", "confidence": 0.7}
            return None

        hooked = estimate_interaction(
            hands, [], timestamps, surfaces,
            camera_poses=camera,
            pose_hook=pose_hook,
            contact_hook=contact_hook,
            temporal_filter=False,
        )
        self.assertEqual(hooked["objects"][0]["source"], "foundationpose")
        self.assertEqual(hooked["contact"]["right"]["object_id"][0], "cup")
        self.assertEqual(hooked["contact"]["source"], "contacthands")
        # 第二帧钩子没说话，退回深度启发式：指尖深度 1 m，物体深度 1 m，算接触。
        self.assertEqual(hooked["contact"]["right"]["object_id"][1], "cup")


class EvaluateTest(unittest.TestCase):
    def test_synthetic_disagreement_has_exact_counts(self):
        gt, pred, timestamps = synthetic_disagreement()
        metrics = evaluate_interaction(gt, pred, timestamps)
        contact = metrics["contact"]["both"]
        grasp = metrics["grasp"]["both"]
        # 3 帧里右手有效。第 0 帧都没接触；第 1 帧只有真值接触；第 2 帧两边都接触且都在抓。
        self.assertEqual(contact["tp"], 1)
        self.assertEqual(contact["fn"], 1)
        self.assertEqual(contact["fp"], 0)
        self.assertAlmostEqual(contact["precision"], 1.0)
        self.assertAlmostEqual(contact["recall"], 0.5)
        self.assertEqual(grasp["tp"], 1)
        self.assertEqual(grasp["fp"], 0)
        self.assertEqual(grasp["fn"], 0)
        self.assertAlmostEqual(grasp["precision"], 1.0)
        self.assertAlmostEqual(grasp["recall"], 1.0)
        # 接触开始差 1 帧（0.1 秒）；抓取事件两边同一帧，时间差 0。中位数是 0.05 秒。
        self.assertEqual(metrics["events"]["matched"], 2)
        self.assertIn(0.1, [round(item, 5) for item in metrics["events"]["timing_error_s"]])
        self.assertAlmostEqual(metrics["events"]["timing_error_median_s"], 0.05)
        self.assertIsNone(metrics["contact"]["left"]["precision"])

    def test_events_use_timestamps(self):
        events = events_from_labels(
            "right",
            [None, "cup", "cup", None],
            [None, "pre_grasp", "grasp", "release"],
            [0.0, 0.5, 1.0, 1.5],
            [False, True, True, True],
        )
        self.assertEqual(events[0]["type"], "contact_start")
        self.assertEqual(events[0]["timestamp"], 0.5)
        self.assertEqual([item["type"] for item in events], ["contact_start", "grasp", "contact_end", "release"])


class GtMeshTest(unittest.TestCase):
    def test_skin_inside_threshold_is_contact_and_two_tips_grasp(self):
        vertices = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])
        faces = np.array([[0, 1, 2]])
        points = np.zeros((21, 3))
        points[:] = [0.0, 0.0, 0.2]
        points[4] = [0.02, 0.02, GT_CONTACT_M * 0.5]
        points[8] = [0.03, 0.02, GT_CONTACT_M * 0.5]
        objects = [{
            "id": "box",
            "category": "box",
            "source": "hot3d",
            "pose": [[0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1]],
            "confidence": [1.0, 1.0],
            "valid": [True, False],
        }]
        hands = {
            "left": [None, None],
            "right": [{"points": points, "tips": points[list(FINGERTIPS)]}, None],
        }
        gt = derive_gt_interaction(
            objects, hands, [0.0, 0.1], {"box": {"vertices": vertices, "faces": faces}},
        )
        self.assertEqual(gt["contact"]["right"]["object_id"][0], "box")
        self.assertEqual(gt["grasp"]["right"]["state"][0], "grasp")
        self.assertFalse(gt["contact"]["right"]["valid"][1])
        self.assertEqual(gt["contact"]["source"], "hot3d_mesh")
        self.assertLessEqual(HEURISTIC_DEFAULTS["contact_m"], 0.02)


class ExportAndQcTest(unittest.TestCase):
    def test_masks_zero_invalid_slots_and_keep_codes(self):
        objects = [{
            "id": "cup",
            "category": "tableware",
            "source": "hot3d",
            "pose": [[0.1, 0.2, 0.3, 0, 0, 0, 1], None, [0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1]],
            "confidence": [0.9, None, 0.9, 0.9],
            "valid": [True, False, True, True],
        }]
        contact = empty_interaction(4)["contact"]
        contact["right"]["object_id"] = ["cup", None, None, "cup"]
        contact["right"]["confidence"] = [0.8, None, 0.2, 0.4]
        contact["right"]["valid"] = [True, False, True, True]
        grasp = empty_interaction(4)["grasp"]
        grasp["right"]["state"] = ["grasp", None, "open", "release"]
        grasp["right"]["confidence"] = [0.9, None, 0.5, 0.5]
        grasp["right"]["valid"] = [True, False, True, True]
        episode = _episode(objects=objects, contact=contact, grasp=grasp)
        pose, pose_valid = pack_object_pose(episode, 0)
        self.assertEqual(pose.shape, (OBJECT_POSE_DIM,))
        self.assertEqual(OBJECT_SLOTS, 4)
        self.assertAlmostEqual(float(pose[0]), 0.1)
        self.assertEqual(pose_valid.tolist(), [1, 0, 0, 0])
        missing, missing_valid = pack_object_pose(episode, 1)
        self.assertTrue(np.allclose(missing, 0))
        self.assertEqual(missing_valid.tolist(), [0, 0, 0, 0])
        contact_vec, contact_valid = pack_contact(episode, 0)
        self.assertEqual(contact_vec.shape, (CONTACT_DIM,))
        self.assertEqual(contact_valid.tolist(), [0, 1])
        self.assertAlmostEqual(float(contact_vec[2]), 0.0)
        self.assertAlmostEqual(float(contact_vec[3]), 0.8)
        none_vec, none_valid = pack_contact(episode, 2)
        self.assertEqual(none_valid.tolist(), [0, 1])
        self.assertAlmostEqual(float(none_vec[2]), -1.0)
        grasp_vec, grasp_valid = pack_grasp(episode, 0)
        self.assertEqual(grasp_vec.shape, (GRASP_DIM,))
        self.assertEqual(grasp_valid.tolist(), [0, 1])
        self.assertAlmostEqual(float(grasp_vec[1]), 2.0)
        released, released_valid = pack_grasp(episode, 3)
        self.assertAlmostEqual(float(released[1]), 3.0)
        self.assertEqual(released_valid.tolist(), [0, 1])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episodes = root / "episodes"
            episodes.mkdir()
            episode["episode_id"] = "good"
            (episodes / "good.json").write_text(json.dumps(episode), encoding="utf-8")
            csv_path = root / "yield.csv"
            _yield_csv(csv_path, "good")
            out = root / "lerobot"
            export_lerobot(episodes, csv_path, out, repo_id="local/contact")
            info = json.loads((out / "meta" / "info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["features"]["observation.object_pose"]["shape"], [OBJECT_POSE_DIM])
            self.assertEqual(info["features"]["observation.object_pose_valid"]["shape"], [OBJECT_SLOTS])
            self.assertEqual(info["features"]["observation.contact"]["shape"], [CONTACT_DIM])
            self.assertEqual(info["features"]["observation.contact_valid"]["shape"], [2])
            self.assertEqual(info["features"]["action.grasp"]["shape"], [GRASP_DIM])
            self.assertEqual(info["features"]["action.grasp_valid"]["shape"], [2])
            table = pq.read_table(out / "data" / "chunk-000" / "file-000.parquet")
            data = table.to_pydict()
            self.assertEqual(len(data["action.grasp"]), 3)
            self.assertEqual(list(data["action.grasp_valid"][1]), [0.0, 0.0])
            self.assertEqual(list(data["observation.object_pose_valid"][1]), [0.0, 0.0, 0.0, 0.0])
            note = json.loads((out / "meta" / "egodata_export.json").read_text(encoding="utf-8"))
            self.assertIn("object_pose", note)
            self.assertTrue((out / "meta" / "interaction_events.jsonl").is_file())

    def test_qc_reports_grasp_without_contact_and_still_accepts_motion(self):
        import h5py
        from egodata.egodex import load_episode_hdf5
        n = 20
        wrists = [(0.002 * index, 0.0, 1.0) for index in range(n)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "basic_pick_place" / "0.hdf5"
            path.parent.mkdir(parents=True, exist_ok=True)
            intrinsic = np.array([[736.6339, 0, 960], [0, 736.6339, 540], [0, 0, 1]], dtype=np.float32)
            with h5py.File(path, "w") as handle:
                handle.create_dataset("camera/intrinsic", data=intrinsic)
                camera = np.stack([np.eye(4, dtype=np.float32) for _ in range(n)])
                handle.create_dataset("transforms/camera", data=camera)
                for prefix in ("left", "right"):
                    names = [
                        "%sHand" % prefix, "%sThumbKnuckle" % prefix, "%sThumbIntermediateBase" % prefix,
                        "%sThumbIntermediateTip" % prefix, "%sThumbTip" % prefix,
                        "%sIndexFingerKnuckle" % prefix, "%sIndexFingerIntermediateBase" % prefix,
                        "%sIndexFingerIntermediateTip" % prefix, "%sIndexFingerTip" % prefix,
                        "%sMiddleFingerKnuckle" % prefix, "%sMiddleFingerIntermediateBase" % prefix,
                        "%sMiddleFingerIntermediateTip" % prefix, "%sMiddleFingerTip" % prefix,
                        "%sRingFingerKnuckle" % prefix, "%sRingFingerIntermediateBase" % prefix,
                        "%sRingFingerIntermediateTip" % prefix, "%sRingFingerTip" % prefix,
                        "%sLittleFingerKnuckle" % prefix, "%sLittleFingerIntermediateBase" % prefix,
                        "%sLittleFingerIntermediateTip" % prefix, "%sLittleFingerTip" % prefix,
                    ]
                    for joint_index, name in enumerate(names):
                        offset = np.array([0.0, 0.0, 0.02 * joint_index], dtype=np.float32)
                        series = []
                        for frame_index in range(n):
                            pose = np.eye(4, dtype=np.float32)
                            pose[:3, 3] = np.asarray(wrists[frame_index], dtype=np.float32) + offset
                            series.append(pose)
                        handle.create_dataset("transforms/%s" % name, data=np.stack(series))
                        handle.create_dataset("confidences/%s" % name, data=np.full((n,), 0.99, dtype=np.float32))
                handle.attrs["llm_description"] = "pick up the cup"
                handle.attrs["llm_objects"] = np.array(["cup"], dtype=object)
                handle.attrs["llm_verbs"] = np.array(["pick"], dtype=object)
                handle.attrs["environment"] = "table:wood"
                handle.attrs["task"] = "basic_pick_place"
            episode = load_episode_hdf5(path)
        self.assertEqual(validate_episode(episode), [])
        self.assertEqual(episode["objects"], [])
        episode["grasp"]["right"]["state"] = ["grasp"] * n
        episode["grasp"]["right"]["valid"] = [True] * n
        episode["grasp"]["right"]["confidence"] = [1.0] * n
        episode["contact"]["right"]["valid"] = [True] * n
        episode["contact"]["right"]["object_id"] = [None] * n
        result = qc_episode(episode)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["flags"]["grasp_without_contact"], n)
        self.assertNotIn("grasp_without_contact", result["reasons"])
        self.assertEqual(result["interaction"]["object_pose_valid_fraction"], None)
        self.assertEqual(result["interaction"]["grasp_valid_fraction"], 0.5)


def _hand_at_distance(distance, n_near=5, radius=0.02):
    """指尖放在球面外 distance 米。n_near=1 时只有拇指贴着，张合大于 8 cm，不算抓住。"""
    joints = [[0.3, 0.2, 0.2] for _ in range(21)]
    near = [radius + float(distance), 0.0, 0.0]
    far = [0.30, 0.0, 0.0]
    if n_near <= 1:
        joints[4] = list(near)
        for index in FINGERTIPS:
            if index != 4:
                joints[index] = list(far)
        return joints
    for index in list(FINGERTIPS)[:n_near]:
        joints[index] = list(near)
    return joints


def _sphere_inputs(distances, timestamps, n_near=5, radius=0.02):
    n = len(timestamps)
    joints = [None if item is None else _hand_at_distance(item, n_near=n_near, radius=radius) for item in distances]
    objects = [{
        "id": "ball",
        "category": "toy",
        "source": "test",
        "pose": [[0, 0, 0, 0, 0, 0, 1]] * n,
        "confidence": [1] * n,
        "valid": [True] * n,
    }]
    surfaces = {"ball": {"radius_m": radius, "center_local": [0, 0, 0]}}
    hands = {
        "left": {"joints": [None] * n, "confidence": [None] * n},
        "right": {"joints": joints, "confidence": [0.9] * n},
    }
    return hands, objects, surfaces


def _run_sphere(distances, timestamps, n_near=5, temporal_filter="default", params=None):
    hands, objects, surfaces = _sphere_inputs(distances, timestamps, n_near=n_near)
    kwargs = {}
    if temporal_filter != "default":
        kwargs["temporal_filter"] = temporal_filter
    return estimate_interaction(hands, objects, timestamps, surfaces, params=params, **kwargs)


class TemporalFilterTest(unittest.TestCase):
    def test_defaults_are_on_with_hysteresis_band_and_dwell(self):
        self.assertGreater(HEURISTIC_DEFAULTS["dwell_s"], 0.0)
        self.assertGreater(HEURISTIC_DEFAULTS["contact_off_margin_m"], 0.0)
        self.assertAlmostEqual(HEURISTIC_DEFAULTS["contact_m"], 0.01)

    def test_default_drops_a_single_frame_blip(self):
        # 30 fps 量级的一帧贴上又离开，短于默认停留时间，不能记成接触。
        timestamps = [0.0, 1.0 / 30.0, 2.0 / 30.0]
        blip = _run_sphere([0.05, 0.005, 0.05], timestamps)
        self.assertEqual(blip["contact"]["right"]["object_id"], [None, None, None])
        self.assertNotIn("grasp", blip["grasp"]["right"]["state"])

    def test_flag_off_keeps_the_same_blip(self):
        timestamps = [0.0, 1.0 / 30.0, 2.0 / 30.0]
        raw = _run_sphere([0.05, 0.005, 0.05], timestamps, temporal_filter=False)
        self.assertEqual(raw["contact"]["right"]["object_id"], [None, "ball", None])
        self.assertEqual(raw["grasp"]["right"]["state"][1], "grasp")

    def test_hysteresis_uses_separate_on_and_off_distances(self):
        params = {"contact_on_m": 0.01, "contact_off_m": 0.02, "dwell_s": 0.0, "grasp_speed_m_s": 10.0}
        # 0.015 落在两档之间：没进去过就不算接触；进去之后离开到 off 之外才松开。
        result = _run_sphere([0.05, 0.015, 0.008, 0.015, 0.03], [0.0, 0.1, 0.2, 0.3, 0.4], params=params)
        self.assertEqual(result["contact"]["right"]["object_id"], [None, None, "ball", "ball", None])
        self.assertEqual(result["grasp"]["right"]["state"][2], "grasp")
        self.assertEqual(result["grasp"]["right"]["state"][3], "grasp")
        self.assertEqual(result["grasp"]["right"]["state"][4], "release")

    def test_contact_m_sets_the_on_threshold_when_on_is_omitted(self):
        params = {"contact_m": 0.004, "dwell_s": 0.0, "grasp_speed_m_s": 10.0}
        held = _run_sphere([0.003, 0.008], [0.0, 0.1], params=params)
        self.assertEqual(held["contact"]["right"]["object_id"], ["ball", "ball"])
        raw = _run_sphere([0.003, 0.008], [0.0, 0.1], temporal_filter=False, params={"contact_m": 0.004})
        self.assertEqual(raw["contact"]["right"]["object_id"], ["ball", None])

    def test_dwell_is_seconds_and_matches_across_frame_rates(self):
        dwell = 0.2

        def first_contact(step):
            count = int(round(1.0 / step)) + 1
            timestamps = [index * step for index in range(count)]
            distances = [0.005] * count
            result = _run_sphere(distances, timestamps, params={"dwell_s": dwell, "contact_on_m": 0.01, "contact_off_m": 0.02})
            for stamp, object_id in zip(timestamps, result["contact"]["right"]["object_id"]):
                if object_id is not None:
                    return stamp
            return None

        slow = first_contact(0.1)
        fast = first_contact(0.02)
        self.assertAlmostEqual(slow, 0.2)
        self.assertAlmostEqual(fast, 0.2)

    def test_spike_shorter_than_dwell_is_ignored_at_both_rates(self):
        def any_contact(step):
            count = int(round(0.5 / step)) + 1
            timestamps = [index * step for index in range(count)]
            distances = [0.005 if 0.20 <= stamp < 0.24 else 0.05 for stamp in timestamps]
            result = _run_sphere(distances, timestamps, params={"dwell_s": 0.1})
            return any(item is not None for item in result["contact"]["right"]["object_id"])

        self.assertFalse(any_contact(0.01))
        self.assertFalse(any_contact(0.04))

    def test_dropout_shorter_than_dwell_does_not_release(self):
        params = {"dwell_s": 0.1, "contact_on_m": 0.01, "contact_off_m": 0.02, "grasp_speed_m_s": 10.0}
        timestamps = [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]
        distances = [0.005, 0.005, 0.005, 0.05, 0.005, 0.005, 0.05, 0.05, 0.05]
        result = _run_sphere(distances, timestamps, params=params)
        ids = result["contact"]["right"]["object_id"]
        states = result["grasp"]["right"]["state"]
        self.assertEqual(ids[:6], [None, "ball", "ball", "ball", "ball", "ball"])
        self.assertEqual(states[1], "grasp")
        self.assertEqual(states[3], "grasp")
        self.assertEqual(ids[6], "ball")
        self.assertIsNone(ids[7])
        self.assertEqual(states[7], "release")
        self.assertEqual(states[8], "open")
        starts = [item["timestamp"] for item in result["events"] if item["type"] == "contact_start"]
        self.assertEqual(starts, [0.1])

    def test_contact_without_grasp_also_waits_for_dwell(self):
        params = {"dwell_s": 0.1, "contact_on_m": 0.01, "contact_off_m": 0.02}
        result = _run_sphere([0.005, 0.005, 0.005], [0.0, 0.05, 0.1], n_near=1, params=params)
        self.assertEqual(result["contact"]["right"]["object_id"], [None, None, "ball"])
        self.assertEqual(result["grasp"]["right"]["state"], ["open", "open", "pre_grasp"])

    def test_irregular_timestamps_use_elapsed_seconds(self):
        params = {"dwell_s": 0.1, "contact_on_m": 0.01, "contact_off_m": 0.02}
        result = _run_sphere([0.005, 0.005, 0.005], [0.0, 0.05, 0.5], params=params)
        self.assertEqual(result["contact"]["right"]["object_id"], [None, None, "ball"])

    def test_missing_joint_resets_the_latch(self):
        params = {"dwell_s": 0.1, "contact_on_m": 0.01, "contact_off_m": 0.02}
        result = _run_sphere([0.005, 0.005, None, 0.005], [0.0, 0.1, 0.2, 0.3], params=params)
        self.assertEqual(result["contact"]["right"]["object_id"][1], "ball")
        self.assertFalse(result["contact"]["right"]["valid"][2])
        self.assertIsNone(result["contact"]["right"]["object_id"][3])

    def test_eval_script_filter_defaults_on(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "eval_hot3d_contact", ROOT / "scripts" / "eval_hot3d_contact.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        base = ["--clips-dir", "c", "--models", "m", "--mano", "mano", "--tune", "clip", "--out", "o"]
        self.assertFalse(module.build_parser().parse_args(base).no_temporal_filter)
        flagged = module.build_parser().parse_args(base + ["--no-temporal-filter"])
        self.assertTrue(flagged.no_temporal_filter)


if __name__ == "__main__":
    unittest.main()
