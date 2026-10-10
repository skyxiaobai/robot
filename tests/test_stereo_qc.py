# -*- coding: utf-8 -*-
"""双目 QC：立体门限丢掉的标注不计入片段坏帧。"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from egodata.qc import write_yield_reports, yield_report  # noqa: E402
from egodata import stereo_qc as stereo_qc_mod  # noqa: E402
from egodata.stereo_qc import qc_stereo_episode  # noqa: E402


def _episode(episode_id, n, moving):
    """相机在原点朝 +Z。moving 时手腕沿 x 慢慢走，否则一直停着。"""
    pose = np.eye(4).tolist()
    hands = {}
    for side in ("left", "right"):
        joints, confidence, wrist_pose = [], [], []
        for index in range(n):
            x = 0.002 * index if moving else 0.0
            point = [x, 0.0, 1.0]
            joints.append([point] + [[x, 0.02, 1.0]] * 20)
            confidence.append(0.99)
            wrist_pose.append(point + [0.0, 0.0, 0.0, 1.0])
        hands[side] = {
            "joints": joints,
            "confidence": confidence,
            "wrist_pose": wrist_pose,
            "filled": [False] * n,
        }
    return {
        "episode_id": episode_id,
        "num_frames": n,
        "fps": 30.0,
        "image_width": 640,
        "image_height": 480,
        "camera_intrinsic": [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
        "camera_poses": [pose for _ in range(n)],
        "hands": hands,
    }


def _attach_stereo(episode, left_status, right_status=None, filled_frames=()):
    """写入逐手状态。filled_frames 里的帧两只手都标成补出来的，并拿掉手腕。"""
    n = episode["num_frames"]
    right_status = list(right_status) if right_status is not None else ["ok"] * n
    episode["stereo"] = {"per_frame": {"left": list(left_status), "right": list(right_status)}}
    for index in filled_frames:
        for side in ("left", "right"):
            episode["hands"][side]["filled"][index] = True
            episode["hands"][side]["joints"][index][0] = [None, None, None]
            episode["hands"][side]["confidence"][index] = None
    return episode


class StereoLabelCoverageTest(unittest.TestCase):
    def test_default_coverage_threshold_is_provisional_and_does_not_reject(self):
        self.assertEqual(getattr(stereo_qc_mod, "DEFAULT_MIN_LABEL_COVERAGE", None), 0.0)

    def test_jump_disagreement_and_gap_fill_are_dropped_labels(self):
        """超过 EgoDex 20% 的跳点、左右不一致和补帧，不再把整段判失败。"""
        n = 20
        left = ["ok"] * n
        # 8/20 = 40%，旧逻辑会因坏帧比例拒绝整段。
        for index, status in enumerate(
            ["jump", "jump", "reproj", "depth", "palm", "few_joints", "one_view", "strict"]
        ):
            left[index] = status
        episode = _attach_stereo(_episode("drops", n, moving=True), left, filled_frames=(8, 9))
        # 补帧在管线里会把两只手腕拿掉，基础 QC 会当成手出画。这两帧也必须排除。
        qc = qc_stereo_episode(episode)
        self.assertEqual(qc["dropped_label_frames"], 10)
        self.assertEqual(qc["bad_frames"], 0)
        self.assertAlmostEqual(qc["label_coverage"], 0.5)
        self.assertTrue(qc["accepted"], qc)
        self.assertEqual(qc["reasons"], [])
        self.assertNotIn(False, [qc["good_frame_mask"][i] for i in range(10, n)])
        self.assertTrue(all(not qc["good_frame_mask"][i] for i in range(10)))
        self.assertIn("valid_action_ratio", qc)
        self.assertLess(qc["valid_action_ratio"], 1.0)
        self.assertIn("valid_full_chunk_ratio_16", qc)

    def test_long_gate_reject_is_not_staged_static(self):
        """两只手都被门限丢掉超过 1 秒时，缺测不是摆拍，不能据此拒绝片段。"""
        n = 60
        left = ["jump"] * 40 + ["ok"] * 20
        right = ["jump"] * 40 + ["ok"] * 20
        episode = _attach_stereo(_episode("long-jump", n, moving=True), left, right)
        for index in range(40):
            for side in ("left", "right"):
                episode["hands"][side]["joints"][index][0] = [None, None, None]
                episode["hands"][side]["confidence"][index] = None
        qc = qc_stereo_episode(episode)
        self.assertEqual(qc["flags"]["hands_out_of_frame"], 0)
        # 丢掉的 40 帧本身不是静止。紧接着的那一帧速度因前一帧没手腕而记成 0，最多剩 1 帧。
        self.assertLessEqual(qc["flags"]["staged_static"], 1)
        self.assertLessEqual(qc["bad_frames"], 1)
        self.assertLess(qc["bad_fraction"], 0.2)
        self.assertTrue(qc["accepted"], qc)
        self.assertAlmostEqual(qc["label_coverage"], 20 / 60)

    def test_low_coverage_uses_its_own_threshold(self):
        n = 20
        left = ["jump" if i < 8 else "ok" for i in range(n)]
        episode = _attach_stereo(_episode("coverage", n, moving=True), left)
        qc = qc_stereo_episode(episode, min_label_coverage=0.9)
        self.assertFalse(qc["accepted"])
        self.assertIn("low_label_coverage", qc["reasons"])
        self.assertNotIn("stereo_inconsistent", qc["reasons"])
        self.assertLessEqual(qc["bad_fraction"], 0.2)
        self.assertEqual(qc["min_label_coverage"], 0.9)

    def test_genuine_clip_problem_still_rejects(self):
        episode = _attach_stereo(_episode("still", 60, moving=False), ["ok"] * 60)
        qc = qc_stereo_episode(episode)
        self.assertFalse(qc["accepted"])
        self.assertIn("staged_static", qc["reasons"])
        self.assertAlmostEqual(qc["label_coverage"], 1.0)
        self.assertGreater(qc["bad_fraction"], 0.2)

    def test_yield_csv_reports_label_coverage(self):
        episode = _attach_stereo(_episode("csv", 10, moving=True), ["jump"] + ["ok"] * 9)
        report = yield_report([qc_stereo_episode(episode)])
        with __import__("tempfile").TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "yield.csv"
            write_yield_reports(report, Path(tmp) / "yield.html", csv_path)
            text = csv_path.read_text(encoding="utf-8")
        self.assertIn("label_coverage", text)
        self.assertIn("0.9000", text)


if __name__ == "__main__":
    unittest.main()
