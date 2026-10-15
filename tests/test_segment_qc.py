# -*- coding: utf-8 -*-
"""分段质检：按 3 s 窗口打分、剪掉坏的那几截；训练只取保留区间里的整块。"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import qc_scorecard as Q  # noqa: E402
from test_pipeline_qc import make_ep  # noqa: E402


class SegmentQCTest(unittest.TestCase):
    def test_clean_clip_kept_whole(self):
        sg = Q.segment_episode(make_ep(180), "c")
        self.assertEqual(sg["window_frames"], 45)
        self.assertEqual(sg["spans"], [[0, 180]])
        self.assertAlmostEqual(sg["kept_frac"], 1.0)

    def test_jittery_window_trimmed_rest_kept(self):
        n = 180; t = np.arange(n) / 15.0
        left = np.stack([-0.15 + 0.10 * np.sin(t), 0.15 + 0 * t, 0.45 + 0 * t], 1)
        rng = np.random.default_rng(0)
        left[45:90] += rng.normal(0, 0.05, (45, 3))  # 第 2 个窗口剧烈抖动
        right = left.copy(); right[:, 0] += 0.3
        sg = Q.segment_episode(make_ep(n, left=left, right=right), "c")
        self.assertEqual(sg["spans"], [[0, 45], [90, 180]])
        self.assertIn("抖动大", sg["windows"][1]["fails"])

    def test_still_window_not_rejected_unless_clip_still(self):
        n = 180; t = np.arange(n) / 15.0
        left = np.stack([-0.15 + 0.10 * np.sin(t), 0.15 + 0 * t, 0.45 + 0 * t], 1)
        left[90:135] = left[90]  # 中间静止 3 s
        sg = Q.segment_episode(make_ep(n, left=left, right=left + [0.3, 0, 0]), "c")
        self.assertEqual(sg["spans"], [[0, 180]])
        still = np.tile([-0.15, 0.15, 0.45], (n, 1))
        sg2 = Q.segment_episode(make_ep(n, left=still, right=still + [0.3, 0, 0]), "c")
        self.assertEqual(sg2["spans"], [])
        self.assertIn("手几乎不动", sg2["windows"][0]["fails"])

    def test_frame_mask_and_subsample(self):
        e = {"fps": 30.0, "frames": 10, "spans": [[2, 6]]}
        self.assertEqual(Q.segment_frame_mask(e).tolist(), [0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
        self.assertEqual(Q.segment_frame_mask(e, 5, 2).tolist(), [0, 1, 1, 0, 0])

    def test_load_segments(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "segments.json"
            p.write_text(json.dumps({"clips": [{"clip": "a", "fps": 15, "frames": 90, "spans": [[0, 45]]}]}))
            self.assertEqual(Q.load_segments([p])["a"]["spans"], [[0, 45]])


class TrainSegmentFilterTest(unittest.TestCase):
    def test_chunks_only_inside_spans(self):
        try:
            import train_xhey as T
        except Exception as e:  # noqa: BLE001
            self.skipTest("train_xhey import: %s" % e)
        H = T.H
        T.SEG_KEEP.clear(); T.SEG_KEEP["a"] = {"fps": 15.0, "frames": 100, "spans": [[0, 40], [60, 100]]}
        try:
            ok = T.seg_row_ok([{"num_frames": 100, "fps": 15.0}, {"num_frames": 50, "fps": 15.0}], ["a", "b"], "t")
        finally:
            T.SEG_KEEP.clear()
        self.assertEqual(len(ok), 99 + 49)
        self.assertTrue(ok[:39].all()); self.assertFalse(ok[39:60].any()); self.assertTrue(ok[99:].all())
        starts = np.arange(0, 99 - H + 1)
        kept = T.seg_chunk_filter({"seg_ok": ok}, starts)
        self.assertTrue(all(s + H <= 39 or s >= 60 for s in kept))
        self.assertIn(0, kept); self.assertIn(60, kept); self.assertNotIn(30, kept)
        self.assertEqual(len(T.seg_chunk_filter({}, starts)), len(starts))


if __name__ == "__main__":
    unittest.main()
