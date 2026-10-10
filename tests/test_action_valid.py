# -*- coding: utf-8 -*-
"""缺测手腕不能变成「保持不动」的监督。"""
import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from egodata.action_valid import (  # noqa: E402
    DEFAULT_CHUNK_LENGTHS,
    action_dim_mask,
    action_is_pad_from_valid,
    action_valid_flags,
    element_mask,
    episode_action_valid,
    masked_l1,
    masked_mse,
    validity_summary,
    wrist_measured,
)
from egodata.lerobot_export import export_lerobot  # noqa: E402
from egodata.qc import qc_episode, write_yield_reports, yield_report  # noqa: E402
from ego_act_scaling import copy_baseline_l1  # noqa: E402
from ego_pretrain_bc import _select_indices, load_action_valid, train_bc  # noqa: E402
from test_lerobot_export import _episode, _write_episode, _yield_csv  # noqa: E402


def _blank_wrist(episode, side, frame_index):
    episode["hands"][side]["wrist_pose"][frame_index] = [None] * 7


class MeasuredWristTest(unittest.TestCase):
    def test_missing_wrist_marks_that_hand_invalid_and_keeps_identity(self):
        from egodata.lerobot_export import pack_action

        episode = _episode("gap", n=4)
        _blank_wrist(episode, "left", 1)
        self.assertFalse(wrist_measured(episode, "left", 1))
        self.assertTrue(wrist_measured(episode, "right", 1))
        self.assertEqual(action_valid_flags(episode, 0).tolist(), [0.0, 1.0])
        self.assertEqual(action_valid_flags(episode, 1).tolist(), [0.0, 1.0])
        self.assertEqual(action_valid_flags(episode, 2).tolist(), [1.0, 1.0])
        action = pack_action(episode, 0)
        self.assertTrue(np.allclose(action[:7], [0, 0, 0, 0, 0, 0, 1]))
        self.assertTrue(np.allclose(action[7:10], [0.01, 0, 0], atol=1e-6))

    def test_filled_low_confidence_status_and_mask_are_not_measured(self):
        episode = _episode("marks", n=4)
        episode["hands"]["left"]["filled"] = [False, True, False, False]
        episode["hands"]["right"]["confidence"][2] = 0.2
        episode["stereo"] = {
            "per_frame": {
                "left": ["ok", "ok", "ok", "ok"],
                "right": ["ok", "one_view", "ok", "ok"],
            }
        }
        # 左手第 1 帧是补出来的。右手第 1 帧被立体门限丢掉，第 2 帧置信度低于 0.5。
        self.assertEqual(action_valid_flags(episode, 0).tolist(), [0.0, 0.0])
        self.assertEqual(action_valid_flags(episode, 1).tolist(), [0.0, 0.0])
        self.assertEqual(action_valid_flags(episode, 2).tolist(), [1.0, 0.0])
        # 补出来的手腕如果仍是有限数，增量照原样留下，只是标成无效。
        from egodata.lerobot_export import pack_action

        kept = pack_action(episode, 0)
        self.assertTrue(np.allclose(kept[:3], [0.01, 0, 0], atol=1e-6))

        unknown = _episode("unknown-conf", n=2)
        unknown["hands"]["left"]["confidence"] = [None, None]
        self.assertEqual(action_valid_flags(unknown, 0).tolist(), [1.0, 1.0])

        labeled = _episode("iphone", n=3)
        labeled["hands"]["left"]["label_status"] = ["ok", "dropped", "ok"]
        self.assertEqual(action_valid_flags(labeled, 0).tolist(), [0.0, 1.0])
        self.assertEqual(action_valid_flags(labeled, 1).tolist(), [0.0, 1.0])

        masked = _episode("mask", n=4)
        masked["good_frame_mask"] = [True, False, True, True]
        self.assertEqual(action_valid_flags(masked, 0).tolist(), [0.0, 0.0])
        self.assertEqual(action_valid_flags(masked, 1).tolist(), [0.0, 0.0])
        self.assertEqual(action_valid_flags(masked, 2).tolist(), [1.0, 1.0])
        # stereo_qc 不把 status=none 当成丢掉的标注。手腕仍有限时这一帧算实测。
        none_status = _episode("none-status", n=2)
        none_status["stereo"] = {"per_frame": {"left": ["none", "ok"], "right": ["ok", "ok"]}}
        self.assertEqual(action_valid_flags(none_status, 0).tolist(), [1.0, 1.0])

    def test_invalid_dims_add_nothing_to_mse_or_l1(self):
        prediction = np.zeros((1, 14))
        target = np.ones((1, 14))
        only_right = action_dim_mask(np.array([[0.0, 1.0]]), 14)
        self.assertAlmostEqual(masked_mse(prediction, target, only_right), 1.0)
        self.assertAlmostEqual(masked_l1(prediction, target, only_right), 1.0)
        none_valid = action_dim_mask(np.zeros((1, 2)), 14)
        self.assertEqual(masked_mse(prediction, target, none_valid), 0.0)
        self.assertEqual(masked_l1(prediction, target, none_valid), 0.0)
        # 一步里只有左手无效：不把整步标成 pad，改用按维掩码。
        pad = action_is_pad_from_valid(np.array([[0.0, 1.0], [0.0, 0.0]]))
        self.assertEqual(pad.tolist(), [False, True])
        mixed = element_mask(np.array([False, False]), np.array([[0.0, 1.0], [1.0, 1.0]]), 14)
        self.assertTrue(np.all(mixed[0, :7] == 0.0))
        self.assertTrue(np.all(mixed[0, 7:14] == 1.0))
        episode_pad = element_mask(np.array([True]), np.ones((1, 2)), 14)
        self.assertEqual(float(episode_pad.sum()), 0.0)

    def test_chunk_ratios_skip_lengths_the_episode_cannot_fill(self):
        valid = episode_action_valid(_episode("short", n=6))
        summary = validity_summary(valid, DEFAULT_CHUNK_LENGTHS)
        self.assertEqual(summary["action_count"], 10)
        self.assertAlmostEqual(summary["valid_action_ratio"], 1.0)
        self.assertIsNone(summary["valid_full_chunk_ratio_16"])
        self.assertIsNone(summary["valid_full_chunk_ratio_50"])
        self.assertIsNone(summary["valid_full_chunk_ratio_100"])
        self.assertEqual(summary["chunk_count_16"], 0)

    def test_qc_and_export_report_ratios_for_a_gap(self):
        episode = _episode("gap-report", n=8)
        _blank_wrist(episode, "left", 7)
        _blank_wrist(episode, "right", 7)
        qc = qc_episode(episode)
        self.assertAlmostEqual(qc["valid_action_ratio"], 12 / 14)
        self.assertIsNone(qc["valid_full_chunk_ratio_50"])
        report = yield_report([qc])
        self.assertAlmostEqual(report["valid_action_ratio"], 12 / 14)
        self.assertIsNone(report["valid_full_chunk_ratio"][50])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "yield.csv"
            write_yield_reports(report, root / "yield.html", csv_path)
            text = csv_path.read_text(encoding="utf-8")
            html = (root / "yield.html").read_text(encoding="utf-8")
            self.assertIn("valid_action_ratio", text)
            self.assertIn("valid_full_chunk_ratio_16", text)
            self.assertIn("valid_full_chunk_ratio_100", text)
            self.assertIn("有效动作", html)
            episodes = root / "episodes"
            episodes.mkdir()
            _write_episode(episodes, episode)
            _yield_csv(csv_path, [
                {"episode_id": "gap-report", "num_frames": 8, "accepted": "yes", "reasons": ""},
            ])
            summary = export_lerobot(episodes, csv_path, root / "lerobot", horizon=1)
            self.assertAlmostEqual(summary["valid_action_ratio"], 12 / 14)
            self.assertIsNone(summary["valid_full_chunk_ratio"][16])
            table = pq.read_table(root / "lerobot" / "data" / "chunk-000" / "file-000.parquet")
            valid = np.asarray(table.column("action_valid").combine_chunks().flatten().to_numpy()).reshape(-1, 2)
            self.assertEqual(valid.shape, (7, 2))
            self.assertEqual(valid[:-1].tolist(), [[1.0, 1.0]] * 6)
            self.assertEqual(valid[-1].tolist(), [0.0, 0.0])
            note = json.loads((root / "lerobot" / "meta" / "egodata_export.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(note["valid_action_ratio"], 12 / 14)
            # 最后一步是占位的「不动」，不能把左手平移的均值拉向 0。
            stats = json.loads((root / "lerobot" / "meta" / "stats.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(stats["action"]["mean"][0], 0.01, places=5)
            loss = masked_l1(
                np.zeros((1, 14)),
                np.asarray(table.column("action").combine_chunks().flatten().to_numpy()[-14:]).reshape(1, 14),
                action_dim_mask(valid[-1:], 14),
            )
            self.assertEqual(loss, 0.0)
            result = train_bc(root / "lerobot", horizon=1, seed=0, log_path=root / "bc.log")
            self.assertEqual(result["val_loss"], 0.0)
            baseline = copy_baseline_l1(root / "lerobot", result["val_episode_ids"], horizon=1)
            self.assertAlmostEqual(baseline, 0.02 / 14, places=5)

    def test_missing_column_is_all_valid_with_a_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episodes = root / "episodes"
            episodes.mkdir()
            _write_episode(episodes, _episode("legacy", n=4))
            csv_path = root / "yield.csv"
            _yield_csv(csv_path, [
                {"episode_id": "legacy", "num_frames": 4, "accepted": "yes", "reasons": ""},
            ])
            dataset = root / "lerobot"
            export_lerobot(episodes, csv_path, dataset, horizon=1)
            path = dataset / "data" / "chunk-000" / "file-000.parquet"
            table = pq.read_table(path).drop(["action_valid"])
            pq.write_table(table, path)
            info_path = dataset / "meta" / "info.json"
            info = json.loads(info_path.read_text(encoding="utf-8"))
            del info["features"]["action_valid"]
            info_path.write_text(json.dumps(info), encoding="utf-8")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                valid = load_action_valid(dataset, num_rows=3)
            self.assertEqual(valid.shape, (3, 2))
            self.assertTrue(np.all(valid == 1.0))
            self.assertTrue(any("action_valid" in str(item.message) for item in caught))
            trained = train_bc(dataset, horizon=1, seed=0)
            self.assertGreater(trained["frames"], 0)

    def test_ridge_ignores_an_invalid_training_target(self):
        from ego_pretrain_bc import _ridge

        rng = np.random.default_rng(0)
        features = rng.normal(size=(6, 3))
        target = np.zeros((6, 2))
        target[:, 0] = features[:, 0]
        target[:, 1] = features[:, 1]
        target[0, 1] = 1000.0
        mask = np.ones_like(target)
        mask[0, 1] = 0.0
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        _, masked = _ridge(features, target, features[:1], 1e-6, mean, std, mask)
        _, dropped = _ridge(features[1:], target[1:], features[:1], 1e-6, mean, std)
        self.assertAlmostEqual(float(masked[0, 1]), float(dropped[0, 1]), places=5)

    def test_min_valid_fraction_drops_incomplete_chunks(self):
        episodes = np.zeros(8, dtype=np.int64)
        action_valid = np.ones((8, 2))
        action_valid[:4] = 0.0
        train, val, _, _, pool = _select_indices(
            episodes, horizon=2, max_frames=None, seed=0, val_fraction=0.1,
            action_valid=action_valid, min_valid_fraction=1.0,
        )
        self.assertTrue(set(np.concatenate([train, val, pool]).tolist()) <= {4, 5, 6})
        self.assertNotIn(0, set(pool.tolist()))
        with self.assertRaises(ValueError):
            _select_indices(
                episodes, horizon=2, max_frames=None, seed=0, val_fraction=0.1,
                action_valid=np.zeros((8, 2)), min_valid_fraction=0.5,
            )


if __name__ == "__main__":
    unittest.main()
