# -*- coding: utf-8 -*-
"""左右手/遮挡修正、统一坐标约定、QC 成绩单 + --qc-filter。"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from common import frames as F  # noqa: E402
from egodata.action_valid import action_valid_flags  # noqa: E402
from egodata.hand_fix import fix_episode_hands  # noqa: E402
import qc_scorecard as Q  # noqa: E402


def make_ep(n=60, fps=15.0, frame="camera", left=None, right=None, conf=None):
    t = np.arange(n) / fps
    if left is None:
        left = np.stack([-0.15 + 0.10 * np.sin(t), 0.15 + 0 * t, 0.45 + 0 * t], 1)
    if right is None:
        right = np.stack([0.15 + 0.10 * np.sin(t), 0.15 + 0 * t, 0.45 + 0 * t], 1)

    def hand(W):
        ok = np.isfinite(W).all(1)
        return {"joints": [[list(w)] * 21 if o else [[None] * 3] * 21 for w, o in zip(W.tolist(), ok)],
                "wrist_pose": [list(w) + [0.0, 0.0, 0.0, 1.0] if o else [None] * 7 for w, o in zip(W.tolist(), ok)],
                "confidence": [1.0 if o else None for o in ok], "valid": ok.tolist(), "filled": [False] * n,
                "label_status": ["ok" if o else "missing" for o in ok]}
    ep = {"num_frames": n, "fps": fps, "coordinate_frame": frame, "source": "test", "episode_id": "t",
          "camera_poses": [np.eye(4).tolist()] * n, "hands": {"left": hand(left), "right": hand(right)}}
    if conf is not None:
        ep["hands"]["left"]["confidence"] = list(conf)
    return ep


class HandFixTest(unittest.TestCase):
    def test_swap_fixed_by_wrist_continuity(self):
        ep = make_ep()
        L, R = ep["hands"]["left"], ep["hands"]["right"]
        for i in (20, 21, 22):  # WiLoR 把这三帧左右认反了
            for k in ("wrist_pose", "joints"):
                L[k][i], R[k][i] = R[k][i], L[k][i]
        s = fix_episode_hands(ep)
        self.assertEqual(s["swaps"], 3)
        self.assertLess(ep["hands"]["left"]["wrist_pose"][21][0], 0)  # 左手回到 x<0
        self.assertGreater(ep["hands"]["right"]["wrist_pose"][21][0], 0)

    def test_no_false_swap_on_clean_clip(self):
        self.assertEqual(fix_episode_hands(make_ep())["swaps"], 0)

    def test_short_gap_interpolated_long_gap_invalid(self):
        n = 90; W = make_ep(n)["hands"]["left"]
        left = np.array([w[:3] for w in W["wrist_pose"]], float)
        left[30:34] = np.nan        # 4 帧 ≈ 0.27 s → 补
        left[50:70] = np.nan        # 20 帧 ≈ 1.3 s → 标 gap
        ep = make_ep(n, left=left)
        s = fix_episode_hands(ep, max_gap_s=0.3)
        h = ep["hands"]["left"]
        self.assertEqual(s["left"]["interp_frames"], 4)
        self.assertEqual(s["left"]["gap_frames"], 20)
        self.assertTrue(all(h["filled"][30:34]) and h["label_status"][31] == "interp")
        mid = np.array(h["wrist_pose"][31][:3]); ends = (left[29] + left[34]) / 2
        self.assertLess(np.linalg.norm(mid - ends), 0.02)
        self.assertFalse(any(h["valid"][50:70]))
        # 补出来的帧和长缺口都不当监督
        self.assertEqual(action_valid_flags(ep, 31)[0], 0.0)
        self.assertEqual(action_valid_flags(ep, 60)[0], 0.0)
        self.assertEqual(action_valid_flags(ep, 10)[0], 1.0)

    def test_low_confidence_treated_as_occluded(self):
        conf = [1.0] * 60; conf[10] = 0.1
        ep = make_ep(conf=conf)
        s = fix_episode_hands(ep)
        self.assertEqual(s["left"]["occluded_or_lowconf"], 1)
        self.assertEqual(ep["hands"]["left"]["label_status"][10], "interp")


class FramesTest(unittest.TestCase):
    def test_cv_yup_roundtrip_and_units(self):
        p = np.array([[0.1, 0.2, 0.5]])
        np.testing.assert_allclose(F.cv_to_yup(p), [[0.1, -0.2, -0.5]])
        np.testing.assert_allclose(F.yup_to_cv(F.cv_to_yup(p)), p)
        np.testing.assert_allclose(F.cm_to_m([50.0]), [0.5])

    def test_rotation_up_to_y(self):
        for up in ([0, 0, 1], [1, 0, 0], [0, -1, 0], [0.3, 0.2, 0.9]):
            R = F.rotation_up_to_y(up)
            np.testing.assert_allclose(R @ (np.array(up) / np.linalg.norm(up)), [0, 1, 0], atol=1e-9)
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)

    def test_world_up_alignment_negative_z(self):
        R = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)
        A, label = F.world_up_alignment(np.stack([R] * 5))
        self.assertEqual(label, "z")
        np.testing.assert_allclose(A.T @ A, np.eye(3), atol=1e-8)
        self.assertAlmostEqual(float(np.linalg.det(A)), 1.0)
        np.testing.assert_allclose(A @ [0, 0, -1], [0, 1, 0], atol=1e-8)

    def test_canonicalization_rejects_missing_world_poses(self):
        ep = make_ep(frame="slam_world")
        ep["camera_poses"] = []
        with self.assertRaisesRegex(ValueError, "camera pose"):
            F.canonicalize_for_xhey(ep)

    def test_canonicalization_camera_yup(self):
        # make_ep 的默认数字是 OpenCV（y 下、z 前）。camera_yup 是 x 右、y 上、z 后，
        # 手在相机前方必须先翻到 z<0；把 OpenCV 数字直接标成 camera_yup 会在正确翻转后落到相机背后。
        src = make_ep(frame="camera")
        left = F.cv_to_yup([p[:3] for p in src["hands"]["left"]["wrist_pose"]])
        right = F.cv_to_yup([p[:3] for p in src["hands"]["right"]["wrist_pose"]])
        ep = make_ep(frame="camera_yup", left=left, right=right)
        normalized = F.canonicalize_for_xhey(ep)
        self.assertEqual(F.convention_of(normalized), F.CAMERA)
        self.assertEqual(F.convention_of(ep), F.CAMERA_YUP)
        np.testing.assert_allclose(normalized["hands"]["left"]["wrist_pose"][0][:3],
                                   F.cv_to_yup(ep["hands"]["left"]["wrist_pose"][0][:3]))
        np.testing.assert_allclose(normalized["hands"]["left"]["wrist_pose"][0][:3],
                                   src["hands"]["left"]["wrist_pose"][0][:3])
        np.testing.assert_allclose(normalized["hands"]["left"]["joints"][0][0],
                                   F.cv_to_yup(ep["hands"]["left"]["joints"][0][0]))
        self.assertGreater(normalized["hands"]["left"]["wrist_pose"][0][2], 0)
        self.assertTrue(F.sanity_check(normalized)["ok"], F.sanity_check(normalized)["flags"])

    def test_opencv_numbers_are_not_a_camera_yup_pose(self):
        ep = make_ep(frame="camera_yup")
        with self.assertRaisesRegex(ValueError, "sanity"):
            F.canonicalize_for_xhey(ep)

    def test_world_up_alignment_ros_zup(self):
        # 光学系 → ROS z 朝上世界：相机朝 +x 看，「上」(-y 光学) = +z
        R = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
        A, lab = F.world_up_alignment(np.stack([R] * 5))
        self.assertEqual(lab, "z")
        np.testing.assert_allclose(A @ R @ [0, -1, 0], [0, 1, 0], atol=1e-9)

    def test_sanity_ok_and_flags(self):
        ep = make_ep(); F.tag(ep, F.CAMERA)
        self.assertTrue(F.sanity_check(ep)["ok"])
        cm = make_ep(left=np.tile([10.0, 15.0, 45.0], (60, 1)), right=np.tile([-10.0, 15.0, 45.0], (60, 1)))
        r = F.sanity_check(cm)
        self.assertFalse(r["ok"]); self.assertTrue(any(f.startswith("units") for f in r["flags"]))
        flipped = make_ep(left=np.tile([0.1, 0.15, -0.45], (60, 1)), right=np.tile([-0.1, 0.15, -0.45], (60, 1)))
        self.assertTrue(any(f.startswith("axes:") for f in F.sanity_check(flipped)["flags"]))

    def test_variants_consistent(self):
        ep = make_ep(); F.tag(ep, F.CAMERA)
        yup = F.wrist_positions(ep, F.CAMERA_YUP)
        self.assertTrue((yup[:, :, 2] < 0).all())  # y 朝上相机系里手在 -z
        # 世界系：相机放在 (1,2,3)，世界 = 相机系平移
        w = make_ep(frame="world_yup"); P = np.eye(4); P[:3, 3] = [1, 2, 3]
        w["camera_poses"] = [P.tolist()] * 60
        for s in ("left", "right"):
            for p in w["hands"][s]["wrist_pose"]:
                p[0] += 1; p[1] += 2; p[2] += 3
        np.testing.assert_allclose(F.wrist_positions(w, F.CAMERA), F.wrist_positions(ep, F.CAMERA), atol=1e-9)

    def test_arkit_world_is_yup_world(self):
        ep = make_ep(frame="arkit_world")
        self.assertEqual(F.convention_of(ep), F.WORLD_YUP)

    def test_align_world_to_yup(self):
        ep = make_ep(frame="slam_world")
        self.assertEqual(F.convention_of(ep), F.WORLD)
        F.align_world_to_yup(ep, F.ROS_ZUP_TO_YUP, "z")
        self.assertEqual(F.convention_of(ep), F.WORLD_YUP)
        q = ep["hands"]["left"]["wrist_pose"][0][3:]
        self.assertAlmostEqual(float(np.linalg.norm(q)), 1.0, places=6)


class ScorecardTest(unittest.TestCase):
    def test_constant_velocity_not_jitter(self):
        for fps in (15.0, 30.0, 60.0):
            n = int(fps * 5)
            t = np.arange(n) / fps
            left = np.column_stack((0.4 * t - 0.2, np.full(n, 0.15), np.full(n, 0.5)))
            ep = make_ep(n=n, fps=fps, left=left, right=left + [0.2, 0, 0])
            card = Q.score_episode(ep, "constant_velocity")
            self.assertLess(card["jitter_cm"], 1e-5)

    def test_missing_depth_is_unknown(self):
        for source in ("stera", "dynamic_intelligence", "zeno_lidar", "iphone_lidar"):
            ep = make_ep()
            ep["source"] = source
            ep["hand_pose"] = {"backend": source + "_lidar"}
            c = Q.score_episode(ep, "no_depth_evidence")
            line = Q.one_line(c)
            self.assertIsNone(c["depth_valid"])
            self.assertNotIn("深度状态未知", c["fails"])
            self.assertNotIn("深度有效率低", c["fails"])
            self.assertIn("深度状态未知", c["warnings"])
            self.assertIn("深度 unknown", line)
            self.assertNotIn("深度 100%", line)
            self.assertTrue(c["pass"], c["fails"])

    def test_explicit_unknown_depth_status_is_warning(self):
        ep = make_ep()
        n = ep["num_frames"]
        ep["source"] = "zeno_lidar"
        ep["hand_pose"] = {"backend": "wilor_lidar"}
        ep["iphone"] = {"per_frame": {"left": ["unknown"] * n, "right": ["unknown"] * n}}
        c = Q.score_episode(ep, "unknown_status")
        self.assertIsNone(c["depth_valid"])
        self.assertNotIn("深度有效率低", c["fails"])
        self.assertIn("深度状态未知", c["warnings"])
        self.assertTrue(c["pass"], c["fails"])

    def test_bad_depth_evidence_fails(self):
        ep = make_ep()
        n = ep["num_frames"]
        ep["source"] = "iphone_lidar"
        ep["hand_pose"] = {"backend": "wilor_lidar"}
        ep["iphone"] = {"per_frame": {"left": ["depth"] * n, "right": ["lowconf"] * n}}
        c = Q.score_episode(ep, "bad_depth")
        self.assertIsNotNone(c["depth_valid"])
        self.assertLess(c["depth_valid"], 0.5)
        self.assertIn("深度有效率低", c["fails"])
        self.assertNotIn("深度状态未知", c.get("warnings") or [])
        self.assertFalse(c["pass"])

    def test_card_and_filter(self):
        good = make_ep(150)
        still = make_ep(150, left=np.tile([-0.1, 0.15, 0.45], (150, 1)), right=np.tile([0.1, 0.15, 0.45], (150, 1)))
        cg, cs = Q.score_episode(good, "good"), Q.score_episode(still, "still")
        self.assertTrue(cg["pass"], cg["fails"])
        self.assertFalse(cs["pass"]); self.assertIn("手几乎不动", cs["fails"])
        for k in ("move_frac", "depth_valid", "detect_rate", "jitter_cm", "swaps", "gap_ratio"):
            self.assertIn(k, cg)
        self.assertIn("good", Q.one_line(cg))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "scorecards.json"; p.write_text(json.dumps({"cards": [cg, cs]}))
            self.assertEqual(Q.load_filter(p), {"still"})
            self.assertEqual(Q.load_scored(p), {"good", "still"})

    def test_train_xhey_qc_keep(self):
        import train_xhey as T
        T.QC_EXCLUDE.clear(); T.QC_EXCLUDE.update({"b"})
        T.QC_SCORED.clear(); T.QC_SCORED.update({"a", "b"})
        T.QC_STRICT = False
        try:
            self.assertEqual([p.stem for p in T._qc_keep([Path("a.json"), Path("b.json")], "x")], ["a"])
            kept = [p.stem for p in T._qc_keep([Path("a.json"), Path("new.json")], "x")]
            self.assertEqual(kept, ["a", "new"])
            T.QC_STRICT = True
            with self.assertRaisesRegex(ValueError, "Unreviewed"):
                T._qc_keep([Path("a.json"), Path("new.json")], "x")
        finally:
            T.QC_EXCLUDE.clear()
            T.QC_SCORED.clear()
            T.QC_STRICT = False

    def test_qc_filter_optional(self):
        import io
        import train_xhey as T
        old_argv, old_err = sys.argv, sys.stderr
        with tempfile.TemporaryDirectory() as d:
            sys.argv = ["train_xhey.py", "--run", "missing-run-dir", "--out", d, "--strict-qc", "--no-final"]
            sys.stderr = io.StringIO()
            try:
                with self.assertRaises(BaseException) as ctx:
                    T.main()
                err = sys.stderr.getvalue()
                if isinstance(ctx.exception, SystemExit) and ctx.exception.code not in (0, None):
                    self.assertNotIn("qc-filter", err)
                    self.assertNotIn("unrecognized arguments", err)
                else:
                    self.assertIsInstance(ctx.exception, ValueError)
            finally:
                sys.argv, sys.stderr = old_argv, old_err
                T.QC_EXCLUDE.clear()
                T.QC_SCORED.clear()
                T.QC_STRICT = False


if __name__ == "__main__":
    unittest.main()
