# -*- coding: utf-8 -*-
"""统一 episode → LeRobot v3.0，以及只吃 QC 通过片段的线性 BC。"""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from egodata.lerobot_export import (  # noqa: E402
    ACTION_DIM,
    CODEBASE_VERSION,
    STATE_DIM,
    accepted_episode_ids,
    export_lerobot,
    frame_task,
    hand_valid_flags,
    pack_action,
    pack_state,
)
from ego_pretrain_bc import train_bc  # noqa: E402
import scaling_law  # noqa: E402


def _wrist(index, side):
    shift = 0.0 if side == "left" else 0.2
    return [0.01 * index + shift, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


def _hand(index, side, blank_joint=None):
    joints = []
    for joint_index in range(21):
        if blank_joint == joint_index:
            joints.append(None)
        else:
            joints.append([0.01 * joint_index, 0.0, 1.0 + 0.001 * index])
    return {
        "joints": joints,
        "wrist_pose": _wrist(index, side),
        "confidence": 0.9,
        "valid": blank_joint is None,
    }


def _episode(episode_id, n=6, instruction="拿起杯子", subtasks=None, blank=None):
    hands = {"left": {"joints": [], "wrist_pose": [], "confidence": [], "valid": []},
             "right": {"joints": [], "wrist_pose": [], "confidence": [], "valid": []}}
    for index in range(n):
        for side in ("left", "right"):
            hand = _hand(index, side, blank_joint=blank if side == "left" and index == 0 else None)
            hands[side]["joints"].append(hand["joints"])
            hands[side]["wrist_pose"].append(hand["wrist_pose"])
            hands[side]["confidence"].append(hand["confidence"])
            hands[side]["valid"].append(hand["valid"])
    return {
        "schema_version": "1.0",
        "episode_id": episode_id,
        "source": "egodex",
        "fps": 10.0,
        "coordinate_frame": "arkit_world",
        "image_width": 16,
        "image_height": 16,
        "num_frames": n,
        "timestamps": [index / 10.0 for index in range(n)],
        "camera_intrinsic": [[1, 0, 8], [0, 1, 8], [0, 0, 1]],
        "camera_poses": [np.eye(4).tolist() for _ in range(n)],
        "hands": hands,
        "annotation": {
            "environment": {"name": "tabletop", "detail": "", "source": "test"},
            "task": {"name": "pick_cup", "instruction": instruction},
            "subtasks": subtasks or [],
            "instructions": [],
        },
        "coverage": {"environment": "tabletop", "objects": ["cup"], "task": "pick_cup", "action_types": ["pick"]},
    }


def _write_episode(directory, episode):
    path = Path(directory) / ("%s.json" % episode["episode_id"].replace("/", "_"))
    path.write_text(json.dumps(episode), encoding="utf-8")
    return path


def _yield_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode_id", "num_frames", "accepted", "reasons"])
        writer.writeheader()
        writer.writerows(rows)


class PackAndLanguageTest(unittest.TestCase):
    def test_state_layout_and_next_wrist_action(self):
        episode = _episode("good", n=4)
        state = pack_state(episode, 1)
        self.assertEqual(state.shape, (STATE_DIM,))
        self.assertEqual(STATE_DIM, 140)
        self.assertEqual(ACTION_DIM, 14)
        left_wrist = state[126:133]
        self.assertTrue(np.allclose(left_wrist, _wrist(1, "left")))
        action = pack_action(episode, 1)
        self.assertTrue(np.allclose(action[:7], _wrist(2, "left")))
        self.assertTrue(np.allclose(action[7:], _wrist(2, "right")))
        self.assertIsNone(pack_action(episode, 3))

    def test_missing_joint_zeros_that_hand(self):
        episode = _episode("gap", n=2, blank=8)
        state = pack_state(episode, 0)
        valid = hand_valid_flags(episode, 0)
        self.assertEqual(valid.tolist(), [0.0, 1.0])
        # 左手 21 点全空（整手无效时不把缺测写成关节坐标）。
        self.assertTrue(np.allclose(state[:63], 0.0))
        self.assertFalse(np.allclose(state[63:126], 0.0))

    def test_subtask_else_task_instruction(self):
        episode = _episode(
            "lang",
            n=4,
            instruction="整段：拿起杯子",
            subtasks=[
                {"t_start": 0.0, "t_end": 0.2, "text": "伸手"},
                {"t_start": 0.2, "t_end": 0.4, "text": "握住"},
            ],
        )
        self.assertEqual(frame_task(episode, 0.0), "伸手")
        self.assertEqual(frame_task(episode, 0.1), "伸手")
        self.assertEqual(frame_task(episode, 0.2), "握住")
        bare = _episode("bare", n=2, instruction="整段：拿起杯子")
        self.assertEqual(frame_task(bare, 0.0), "整段：拿起杯子")


class ExportTest(unittest.TestCase):
    def test_qc_csv_keeps_only_accepted_and_writes_v3(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episodes = root / "episodes"
            episodes.mkdir()
            good = _episode(
                "good",
                n=6,
                subtasks=[
                    {"t_start": 0.0, "t_end": 0.3, "text": "伸手"},
                    {"t_start": 0.3, "t_end": 0.6, "text": "握住"},
                ],
            )
            bad = _episode("bad", n=6, instruction="不要这条")
            _write_episode(episodes, good)
            _write_episode(episodes, bad)
            csv_path = root / "yield.csv"
            _yield_csv(csv_path, [
                {"episode_id": "bad", "num_frames": 6, "accepted": "no", "reasons": "blur"},
                {"episode_id": "good", "num_frames": 6, "accepted": "yes", "reasons": ""},
            ])
            self.assertEqual(accepted_episode_ids(csv_path), {"good"})
            out = root / "lerobot"
            summary = export_lerobot(episodes, csv_path, out, repo_id="local/egodex_smoke")
            self.assertEqual(summary["episodes"], 1)
            # 6 帧丢掉最后一帧。
            self.assertEqual(summary["frames"], 5)
            info = json.loads((out / "meta" / "info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["codebase_version"], CODEBASE_VERSION)
            self.assertEqual(info["codebase_version"], "v3.0")
            self.assertEqual(info["fps"], 10)
            self.assertEqual(info["total_episodes"], 1)
            self.assertEqual(info["total_frames"], 5)
            self.assertEqual(info["features"]["observation.state"]["shape"], [STATE_DIM])
            self.assertEqual(info["features"]["action"]["shape"], [ACTION_DIM])
            self.assertEqual(info["features"]["observation.image"]["dtype"], "video")
            table = pq.read_table(out / "data" / "chunk-000" / "file-000.parquet")
            data = table.to_pydict()
            self.assertEqual(len(data["action"]), 5)
            self.assertNotIn("bad", json.dumps(data["task_index"]))
            action0 = np.asarray(data["action"][0], dtype=float)
            self.assertTrue(np.allclose(action0[:7], _wrist(1, "left")))
            tasks = pd.read_parquet(out / "meta" / "tasks.parquet")
            # 与 lerobot 0.6.1 load_tasks 一样：行号即 task_index，索引名是句子。
            texts = [tasks.iloc[int(task_index)].name for task_index in data["task_index"]]
            self.assertEqual(texts, ["伸手", "伸手", "伸手", "握住", "握住"])
            video = out / "videos" / "observation.image" / "chunk-000" / "file-000.mp4"
            self.assertTrue(video.is_file())
            self.assertGreater(video.stat().st_size, 100)


class TrainSmokeTest(unittest.TestCase):
    def test_cpu_bc_logs_val_loss_for_scaling_law(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episodes = root / "episodes"
            episodes.mkdir()
            _write_episode(episodes, _episode("keep", n=8, instruction="移动手腕"))
            _write_episode(episodes, _episode("drop", n=8, instruction="丢掉"))
            csv_path = root / "yield.csv"
            _yield_csv(csv_path, [
                {"episode_id": "keep", "num_frames": 8, "accepted": "yes", "reasons": ""},
                {"episode_id": "drop", "num_frames": 8, "accepted": "no", "reasons": "staged_static"},
            ])
            dataset = root / "lerobot"
            export_lerobot(episodes, csv_path, dataset)
            long_log = root / "long.log"
            short_log = root / "short.log"
            long_result = train_bc(dataset, steps=4, log_path=long_log, seed=0)
            short_result = train_bc(dataset, steps=4, log_path=short_log, max_frames=4, seed=0)
            self.assertGreater(long_result["frames"], short_result["frames"])
            long_loss = scaling_law.extract_best_val_loss(str(long_log))
            short_loss = scaling_law.extract_best_val_loss(str(short_log))
            self.assertIsNotNone(long_loss)
            self.assertIsNotNone(short_loss)
            runs = root / "runs.yaml"
            runs.write_text(
                "runs:\n"
                "  - name: ego_short\n    episodes: 4\n    log: short.log\n"
                "  - name: ego_long\n    episodes: %d\n    log: long.log\n" % long_result["frames"],
                encoding="utf-8",
            )
            report = root / "scaling.html"
            code = scaling_law.main(["--runs", str(runs), "--out", str(report)])
            self.assertEqual(code, 0)
            self.assertIn("ego_long", report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
