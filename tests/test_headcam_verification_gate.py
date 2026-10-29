# -*- coding: utf-8 -*-
"""Headcam MVP 验收门：对不上的 accuracy 保持 null，缺向量不编方向，RMSE 对不上就拒绝。"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from headcam.verification_gate import (  # noqa: E402
    REASON_CODES,
    RMSE_IDENTITY_ABS_M,
    RMSE_PUBLISH_ATOL_M,
    check_committed_diagnosis,
    check_repo,
    evaluate_claim,
    main,
)


def _measured(rmse, n=10):
    return {
        "kind": "trajectory_error_v1",
        "method": "umeyama",
        "estimate_pose_source": "orbslam3_stereo_euroc",
        "reference_pose_source": "hot3d_gt_mocap",
        "max_dt_s": 0.05,
        "ate": {
            "se3": {
                "alignment": "umeyama_se3",
                "scale": "fixed_1",
                "n": n,
                "rmse_m": rmse,
                "median_m": rmse,
                "max_m": rmse,
                "rotation_row_major": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                "translation": [0, 0, 0],
            },
            "sim3": {
                "alignment": "umeyama_sim3",
                "scale": "free",
                "n": n,
                "rmse_m": rmse,
                "median_m": rmse,
                "max_m": rmse,
                "similarity_scale": 1.0,
            },
        },
        "rpe": [{"alignment": "none_relative", "delta_s": 1.0, "trans_rmse_m": 0.001}],
    }


def _claim(**overrides):
    claim = {
        "kind": "headcam_mvp_increment_v1",
        "package_accuracy": None,
        "episodes": [],
        "timelines": [],
        "mean_subtract": None,
        "mano_fetch": False,
        "hands_in_slam_world": False,
        "auto_merge": False,
        "new_umeyama": False,
        "orbslam3_binary_present": False,
        "reference_data_present": False,
        "invented_xyz": False,
        "claims_classification": False,
    }
    claim.update(overrides)
    return claim


def _codes(reasons):
    return [item.split(":", 1)[0] for item in reasons]


class VerificationGateClaimTest(unittest.TestCase):
    def test_epsilons_are_the_existing_constants(self):
        self.assertEqual(RMSE_IDENTITY_ABS_M, 1e-12)
        self.assertEqual(RMSE_PUBLISH_ATOL_M, 1e-6)

    def test_handwritten_package_accuracy_is_rejected(self):
        reasons = evaluate_claim(_claim(package_accuracy=0.02))
        self.assertIn("accuracy_not_measured", _codes(reasons))

    def test_disagreeing_measurements_keep_package_accuracy_null(self):
        episodes = [
            {
                "name": "a",
                "published_accuracy": _measured(0.01),
                "rerun_se3_rmse_m": 0.01,
                "batch_se3_rmse_m": 0.01,
            },
            {
                "name": "b",
                "published_accuracy": _measured(0.02),
                "rerun_se3_rmse_m": 0.02,
                "batch_se3_rmse_m": 0.02,
            },
        ]
        invented = evaluate_claim(_claim(package_accuracy=_measured(0.01), episodes=episodes))
        self.assertIn("package_accuracy_invented", _codes(invented))
        self.assertTrue(any("测量不相同" in item for item in invented))

        kept = evaluate_claim(_claim(package_accuracy=None, episodes=episodes))
        self.assertEqual(kept, [])

    def test_remount_mismatch_cannot_publish_accuracy(self):
        episode = {
            "name": "rerun",
            "published_accuracy": _measured(0.001),
            "rerun_se3_rmse_m": 0.002,
            "batch_se3_rmse_m": 0.001,
        }
        invented = evaluate_claim(_claim(
            package_accuracy=_measured(0.001),
            episodes=[episode],
        ))
        codes = _codes(invented)
        self.assertIn("episode_accuracy_invented", codes)
        self.assertIn("package_accuracy_invented", codes)
        self.assertTrue(any("remount" in item for item in invented))

        kept = evaluate_claim(_claim(
            package_accuracy=None,
            episodes=[{
                "name": "rerun",
                "published_accuracy": None,
                "rerun_se3_rmse_m": 0.002,
                "batch_se3_rmse_m": 0.001,
            }],
        ))
        self.assertEqual(kept, [])

    def test_agreeing_measurement_can_publish_the_same_structure(self):
        measured = _measured(0.01)
        reasons = evaluate_claim(_claim(
            package_accuracy=json.loads(json.dumps(measured)),
            episodes=[{
                "name": "same",
                "published_accuracy": measured,
                "rerun_se3_rmse_m": 0.01,
                "batch_se3_rmse_m": 0.01,
            }],
        ))
        self.assertEqual(reasons, [])

    def test_timeline_batch_mismatch_must_be_refused_and_unused(self):
        used = evaluate_claim(_claim(timelines=[{
            "name": "drift",
            "timeline_se3_rmse_m": 0.0,
            "batch_se3_rmse_m": 2e-12,
            "pattern": "unclassified",
            "used_as_this_measurement": True,
        }]))
        self.assertIn("rmse_mismatch_not_refused", _codes(used))

        classified = evaluate_claim(_claim(timelines=[{
            "name": "drift",
            "timeline_se3_rmse_m": 0.0,
            "batch_se3_rmse_m": 2e-12,
            "pattern": "opening_fixed_translation",
            "used_as_this_measurement": False,
        }]))
        self.assertIn("rmse_mismatch_not_refused", _codes(classified))

        refused = evaluate_claim(_claim(timelines=[{
            "name": "drift",
            "timeline_se3_rmse_m": 0.0,
            "batch_se3_rmse_m": 2e-12,
            "pattern": "refused",
            "used_as_this_measurement": False,
        }]))
        self.assertEqual(refused, [])

        inside = evaluate_claim(_claim(timelines=[{
            "name": "tight",
            "timeline_se3_rmse_m": 0.0,
            "batch_se3_rmse_m": 1e-12,
            "pattern": "unclassified",
            "used_as_this_measurement": True,
        }]))
        self.assertEqual(inside, [])

    def test_absent_vectors_keep_mean_subtract_unknown(self):
        unknown = evaluate_claim(_claim(mean_subtract={
            "status": "absent_from_json",
            "rmse_after_m": None,
            "rmse_m": 0.02,
        }))
        self.assertEqual(unknown, [])

        invented = evaluate_claim(_claim(mean_subtract={
            "status": "applied",
            "rmse_after_m": 0.001,
            "rmse_m": 0.02,
        }))
        self.assertIn("mean_subtract_unknown", _codes(invented))

        scalar = evaluate_claim(_claim(mean_subtract={
            "status": "applied",
            "rmse_after_m": 0.001,
            "rmse_m": 0.02,
            "n": 4,
            "error_vectors_m": [0.06, 0.06, 0.06, 0.06],
        }))
        self.assertIn("mean_subtract_unknown", _codes(scalar))
        self.assertTrue(any("direction_unknown" in item for item in scalar))

    def test_vector_rmse_mismatch_does_not_publish_adjusted_rmse(self):
        reasons = evaluate_claim(_claim(mean_subtract={
            "status": "applied",
            "rmse_after_m": 0.0,
            "rmse_m": 0.02,
            "n": 4,
            "error_vectors_m": [[0.5, 0.0, 0.0]] * 4,
        }))
        self.assertIn("mean_subtract_unknown", _codes(reasons))
        self.assertTrue(any("vector_rmse_mismatch" in item for item in reasons))

    def test_stop_markers_fail_closed(self):
        mano = evaluate_claim(_claim(mano_fetch=True))
        self.assertIn("stop_mano", _codes(mano))
        hands = evaluate_claim(_claim(hands_in_slam_world=True))
        self.assertIn("stop_hands_in_slam_world", _codes(hands))
        merge = evaluate_claim(_claim(auto_merge=True))
        self.assertIn("stop_auto_merge", _codes(merge))
        umeyama = evaluate_claim(_claim(new_umeyama=True))
        self.assertIn("stop_umeyama_without_inputs", _codes(umeyama))
        vectors = evaluate_claim(_claim(invented_xyz=True))
        self.assertIn("stop_invented_xyz", _codes(vectors))
        blank = evaluate_claim({
            "kind": "headcam_mvp_increment_v1",
            "package_accuracy": None,
            "episodes": [],
            "timelines": [],
            "mean_subtract": None,
        })
        self.assertIn("stop_unanswered", _codes(blank))

    def test_wrong_kind_fails_closed(self):
        self.assertIn("claim_kind", _codes(evaluate_claim({"kind": "other"})))
        self.assertEqual(evaluate_claim(["nope"]), ["claim_not_object"])


class VerificationGateRepoTest(unittest.TestCase):
    def test_committed_diagnosis_matches_timelines(self):
        self.assertEqual(check_committed_diagnosis(), [])
        self.assertEqual(check_repo(), [])

    def test_missing_report_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            reasons = check_committed_diagnosis(tmp)
        self.assertIn("residual_report_missing", _codes(reasons))

    def test_tampered_accuracy_and_invented_mean_subtract_fail(self):
        src = ROOT / "docs" / "headcam" / "orbslam3_quest3"
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / "orbslam3_quest3"
            shutil.copytree(src, dst)
            report_path = dst / "residual_shape_diagnosis.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["accuracy"] = 0.02
            report["cases"][0]["mean_subtract"]["rmse_after_m"] = 0.001
            report["cases"][0]["mean_subtract"]["status"] = "applied"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            reasons = check_committed_diagnosis(dst)
        codes = _codes(reasons)
        self.assertIn("diagnosis_accuracy_not_null", codes)
        self.assertIn("mean_subtract_unknown", codes)
        self.assertIn("residual_report_inconsistent", codes)

    def test_checklist_lists_every_reason_code(self):
        text = (ROOT / "docs" / "headcam" / "verification_gate.md").read_text(encoding="utf-8")
        missing = [code for code in REASON_CODES if code not in text]
        self.assertEqual(missing, [])

    def test_cli_passes_repo_check_without_tests(self):
        from io import StringIO
        from contextlib import redirect_stdout

        buf = StringIO()
        with redirect_stdout(buf):
            code = main(["--skip-tests"])
        self.assertEqual(code, 0, buf.getvalue())
        self.assertTrue(buf.getvalue().startswith("result: PASS\n"))
        self.assertIn("stop_mano:", buf.getvalue())

    def test_cli_fails_closed_on_a_bad_claim(self):
        from io import StringIO
        from contextlib import redirect_stdout

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "claim.json"
            path.write_text(json.dumps(_claim(package_accuracy=0.02)), encoding="utf-8")
            buf = StringIO()
            with redirect_stdout(buf):
                code = main(["--skip-tests", "--no-repo", "--claim", str(path)])
        self.assertEqual(code, 1)
        self.assertIn("result: FAIL", buf.getvalue())
        self.assertIn("accuracy_not_measured", buf.getvalue())

    def test_claim_fixture_cannot_publish_mean_subtract_without_vectors(self):
        from io import StringIO
        from contextlib import redirect_stdout

        path = (
            ROOT / "tests" / "fixtures" / "headcam"
            / "synthetic_error_vectors" / "claim_rmse_without_vectors.json"
        )
        claim = json.loads(path.read_text(encoding="utf-8"))
        self.assertIsNone(claim["mean_subtract"].get("error_vectors_m"))
        reasons = evaluate_claim(claim)
        self.assertIn("mean_subtract_unknown", _codes(reasons))
        buf = StringIO()
        with redirect_stdout(buf):
            code = main(["--skip-tests", "--no-repo", "--claim", str(path)])
        self.assertEqual(code, 1)
        text = buf.getvalue()
        self.assertTrue(text.startswith("result: FAIL\n"))
        self.assertIn("mean_subtract_unknown", text)
        self.assertIn("absent_from_json", text)


if __name__ == "__main__":
    unittest.main()
