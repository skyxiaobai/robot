# -*- coding: utf-8 -*-
"""一轮 Headcam MVP 增量的验收门。

只做核对：已提交的残差报告是否和 `diagnose_repo` 一致，以及一份增量声明
有没有写出不该写出的 accuracy。不跑 ORB-SLAM3，不新算 Umeyama，不下 MANO，不合并。

合格命令（仓库根目录）::

    PYTHONPATH=scripts python3 -m headcam.verification_gate

退出码 0 且第一行是 ``result: PASS`` 才算这轮合格。``--skip-tests`` 和
``--no-repo`` 只用于调试，不能代替上面这条。
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from headcam.batch_export import RMSE_MATCH_ATOL_M
from headcam.hot3d_check import is_measured_accuracy, shared_measured_accuracy
from headcam.residual_diagnosis import (
    ESTIMATE_POSE_SOURCE,
    PATTERN_KEYS,
    REPORT_KIND,
    RMSE_IDENTITY_ABS_M,
    _finite_rmse,
    _mean_subtract_block,
    diagnose_repo,
)

CLAIM_KIND = "headcam_mvp_increment_v1"
# 发表 accuracy 时，remount（这次重跑）和 batch 行算同一份测量的门限。
# 与 batch_export 写包用的是同一个数。
RMSE_PUBLISH_ATOL_M = RMSE_MATCH_ATOL_M
PACKAGE_ACCURACY_RULE = "null_when_episode_measurements_differ"
_UNKNOWN_VECTOR = ("absent_from_json", "direction_unknown", "vector_rmse_mismatch")
_BOOL_KEYS = (
    "mano_fetch",
    "hands_in_slam_world",
    "auto_merge",
    "new_umeyama",
    "orbslam3_binary_present",
    "reference_data_present",
    "invented_xyz",
    "claims_classification",
)
_REQUIRED_TEST_MODULES = (
    "tests.test_headcam_mvp",
    "tests.test_stereo_qc",
    "tests.test_headcam_verification_gate",
)
REASON_CODES = (
    "claim_not_object",
    "claim_kind",
    "claim_unreadable",
    "package_accuracy_missing",
    "episode_accuracy_missing",
    "episodes_missing",
    "timelines_missing",
    "episode_not_object",
    "timeline_not_object",
    "mean_subtract_unanswered",
    "mean_subtract_not_object",
    "stop_unanswered",
    "stop_mano",
    "stop_hands_in_slam_world",
    "stop_auto_merge",
    "stop_umeyama_without_inputs",
    "stop_invented_xyz",
    "accuracy_not_measured",
    "rmse_not_number",
    "episode_accuracy_invented",
    "accuracy_not_this_measurement",
    "package_accuracy_invented",
    "timeline_used_unanswered",
    "timeline_pattern",
    "rmse_missing_used",
    "rmse_missing_not_refused",
    "rmse_mismatch_not_refused",
    "mean_subtract_unknown",
    "mean_subtract_status",
    "mean_subtract_rmse_invented",
    "residual_report_missing",
    "residual_report_unreadable",
    "residual_report_inconsistent",
    "diagnosis_accuracy_not_null",
    "diagnosis_new_ate",
    "diagnosis_package_rule",
    "diagnosis_case_accuracy_not_null",
    "residual_cases_not_committed_timelines",
    "mean_subtract_inconsistent",
    "checklist_missing",
    "tests_failed",
)
HELD_LINES = (
    "stop_mano: 手进 SLAM 世界系仍要 Justin 的 MANO 许可。这道门不下、不用 MANO。",
    "stop_hands_in_slam_world: 人手放进 SLAM 世界系要停下来问 Justin。",
    "stop_auto_merge: 验收通过也不合并，不打开 auto-merge。",
    "stop_umeyama_without_inputs: 没有 ORB-SLAM3 二进制或参考数据时，不能新算 Umeyama。",
    "stop_invented_xyz: 时间线没有 error_vectors_m 时，不编逐帧 XYZ，减均值之后的 RMSE 保持 unknown。",
)


def repo_root():
    return Path(__file__).resolve().parents[2]


def default_docs():
    return repo_root() / "docs" / "headcam" / "orbslam3_quest3"


def _read_rmse(mapping, key):
    """返回 ``(meters or None, bad)``。缺键和 null 都是没给出，不是错误。"""
    if key not in mapping or mapping[key] is None:
        return None, False
    if _finite_rmse(mapping[key]):
        return float(mapping[key]), False
    return None, True


def _published_rmse(value):
    if not isinstance(value, dict):
        return None
    ate = value.get("ate")
    if not isinstance(ate, dict):
        return None
    se3 = ate.get("se3")
    if not isinstance(se3, dict) or not _finite_rmse(se3.get("rmse_m")):
        return None
    return float(se3["rmse_m"])


def _shell(accuracy):
    return {
        "hand_pose": {
            "slam": {
                "pose_source": ESTIMATE_POSE_SOURCE,
                "accuracy": accuracy,
            }
        }
    }


def _check_stops(claim):
    reasons = []
    for key in _BOOL_KEYS:
        if not isinstance(claim.get(key), bool):
            reasons.append("stop_unanswered:%s" % key)
    if claim.get("mano_fetch") is True:
        reasons.append("stop_mano: 要先问 Justin。这道门不下、不用 MANO")
    if claim.get("hands_in_slam_world") is True:
        reasons.append("stop_hands_in_slam_world: 人手进 SLAM 世界系要先问 Justin")
    if claim.get("auto_merge") is True:
        reasons.append("stop_auto_merge: 不合并")
    binaries = claim.get("orbslam3_binary_present") is True
    data = claim.get("reference_data_present") is True
    if claim.get("new_umeyama") is True and not (binaries and data):
        reasons.append("stop_umeyama_without_inputs: 没有二进制或参考数据，不能新算 Umeyama")
    if claim.get("invented_xyz") is True:
        reasons.append("stop_invented_xyz: 没有 error_vectors_m 时不编逐帧 XYZ")
    return reasons


def _check_episodes(episodes):
    reasons = []
    blocked = False
    shells = []
    for index, episode in enumerate(episodes):
        label = str(index)
        if not isinstance(episode, dict):
            reasons.append("episode_not_object:%s" % label)
            blocked = True
            continue
        if episode.get("name"):
            label = str(episode["name"])
        if "published_accuracy" not in episode:
            reasons.append("episode_accuracy_missing:%s" % label)
            blocked = True
            continue
        published = episode["published_accuracy"]
        if published is not None and not is_measured_accuracy(published, ESTIMATE_POSE_SOURCE):
            reasons.append("accuracy_not_measured:%s" % label)
            blocked = True
            continue
        rerun, rerun_bad = _read_rmse(episode, "rerun_se3_rmse_m")
        batch, batch_bad = _read_rmse(episode, "batch_se3_rmse_m")
        if rerun_bad or batch_bad:
            reasons.append("rmse_not_number:%s" % label)
            blocked = True
            continue
        if rerun is not None and batch is not None and abs(rerun - batch) > RMSE_PUBLISH_ATOL_M:
            blocked = True
            if published is not None:
                reasons.append(
                    "episode_accuracy_invented:%s: remount≠batch（差 %.3e m，门限 %.0e m）"
                    % (label, abs(rerun - batch), RMSE_PUBLISH_ATOL_M)
                )
            continue
        anchor = batch if batch is not None else rerun
        if published is not None and anchor is None:
            reasons.append("episode_accuracy_invented:%s: 没有 remount 或 batch 的 SE3 RMSE" % label)
            blocked = True
            continue
        if published is not None:
            got = _published_rmse(published)
            if got is None or abs(got - anchor) > RMSE_PUBLISH_ATOL_M:
                reasons.append("accuracy_not_this_measurement:%s" % label)
                blocked = True
                continue
        shells.append(_shell(published))
    return reasons, blocked, shells


def _check_package(package, shells, blocked, have_package):
    if not have_package:
        return []
    if package is not None and not is_measured_accuracy(package, ESTIMATE_POSE_SOURCE):
        return ["accuracy_not_measured:package: 包级 accuracy 要是 trajectory_error_v1 或 null"]
    if blocked or not shells:
        if package is not None:
            return [
                "package_accuracy_invented: 各段测量不相同、remount≠batch，或缺测时，包级 accuracy 保持 null"
            ]
        return []
    shared = shared_measured_accuracy(shells)
    if shared is None:
        if package is not None:
            return ["package_accuracy_invented: 各段测量不相同或缺测，包级 accuracy 保持 null"]
        return []
    if package is None:
        return []
    if json.dumps(package, sort_keys=True) != json.dumps(shared, sort_keys=True):
        return ["package_accuracy_invented: 包级 accuracy 不是各段那同一份测量"]
    return []


def _check_timelines(rows):
    reasons = []
    allowed = set(PATTERN_KEYS)
    for index, row in enumerate(rows):
        label = str(index)
        if not isinstance(row, dict):
            reasons.append("timeline_not_object:%s" % label)
            continue
        if row.get("name"):
            label = str(row["name"])
        if not isinstance(row.get("used_as_this_measurement"), bool):
            reasons.append("timeline_used_unanswered:%s" % label)
            continue
        pattern = row.get("pattern")
        if pattern not in allowed:
            reasons.append("timeline_pattern:%s" % label)
            continue
        used = row["used_as_this_measurement"]
        timeline_rmse, timeline_bad = _read_rmse(row, "timeline_se3_rmse_m")
        batch_rmse, batch_bad = _read_rmse(row, "batch_se3_rmse_m")
        if timeline_bad or batch_bad:
            reasons.append("rmse_not_number:%s" % label)
            continue
        if timeline_rmse is None or batch_rmse is None:
            if used:
                reasons.append("rmse_missing_used:%s" % label)
            elif pattern != "refused":
                reasons.append("rmse_missing_not_refused:%s" % label)
            continue
        delta = abs(timeline_rmse - batch_rmse)
        if delta > RMSE_IDENTITY_ABS_M and (pattern != "refused" or used):
            reasons.append(
                "rmse_mismatch_not_refused:%s: |timeline-batch|=%.3e m，门限 %.0e m"
                % (label, delta, RMSE_IDENTITY_ABS_M)
            )
    return reasons


def _timeline_from_block(block):
    timeline = {}
    for key in ("rmse_m", "n", "tail_runs"):
        if key in block:
            timeline[key] = block[key]
    if "error_vectors_m" in block and block["error_vectors_m"] is not None:
        timeline["error_vectors_m"] = block["error_vectors_m"]
    return timeline


def _check_mean_subtract(claim):
    if "mean_subtract" not in claim:
        return ["mean_subtract_unanswered"]
    block = claim["mean_subtract"]
    if block is None:
        return []
    if not isinstance(block, dict):
        return ["mean_subtract_not_object"]
    actual = _mean_subtract_block(_timeline_from_block(block))
    if actual["status"] in _UNKNOWN_VECTOR:
        lied = (
            block.get("status") != actual["status"]
            or block.get("rmse_after_m") is not None
            or block.get("sequence_mean_subtracted_rmse_m") is not None
        )
        if lied:
            return [
                "mean_subtract_unknown: 实际 status=%s，rmse_after_m 必须是 null" % actual["status"]
            ]
        return []
    reasons = []
    if block.get("status") != actual["status"]:
        reasons.append(
            "mean_subtract_status: 声明 %s，按向量算出 %s" % (block.get("status"), actual["status"])
        )
    if block.get("rmse_after_m") != actual["rmse_after_m"]:
        reasons.append("mean_subtract_rmse_invented: 减完的 RMSE 必须来自已有向量")
    return reasons


def evaluate_claim(claim, diagnosis_reasons=None, docs_dir=None):
    """检查一份增量声明。失败原因是字符串列表，空列表表示通过。

    ``claims_classification`` 为真时，还会核对已提交残差报告。
    ``diagnosis_reasons`` 已由调用方算过时直接接上，避免同一轮算两遍。
    """
    if not isinstance(claim, dict):
        return ["claim_not_object"]
    if claim.get("kind") != CLAIM_KIND:
        return ["claim_kind: 需要 %s" % CLAIM_KIND]
    reasons = _check_stops(claim)
    have_package = "package_accuracy" in claim
    if not have_package:
        reasons.append("package_accuracy_missing")
    episodes = claim.get("episodes")
    if not isinstance(episodes, list):
        reasons.append("episodes_missing")
        episodes = []
    timelines = claim.get("timelines")
    if not isinstance(timelines, list):
        reasons.append("timelines_missing")
        timelines = []
    episode_reasons, blocked, shells = _check_episodes(episodes)
    reasons.extend(episode_reasons)
    reasons.extend(_check_package(
        claim.get("package_accuracy"), shells, blocked, have_package,
    ))
    reasons.extend(_check_timelines(timelines))
    reasons.extend(_check_mean_subtract(claim))
    if claim.get("claims_classification") is True:
        if diagnosis_reasons is None:
            diagnosis_reasons = check_committed_diagnosis(docs_dir or default_docs())
        reasons.extend(diagnosis_reasons)
    return reasons


def _case_artifact_reasons(case, timeline):
    reasons = []
    name = case.get("name")
    if case.get("accuracy") is not None:
        reasons.append("diagnosis_case_accuracy_not_null:%s" % name)
    measurement = case.get("measurement") if isinstance(case.get("measurement"), dict) else {}
    timeline_rmse = measurement.get("timeline_rmse_m")
    record_rmse = measurement.get("record_rmse_m")
    pattern = case.get("pattern")
    same = measurement.get("same_measurement")
    numbers_ok = _finite_rmse(timeline_rmse) and _finite_rmse(record_rmse)
    delta_bad = (
        numbers_ok
        and abs(float(timeline_rmse) - float(record_rmse)) > RMSE_IDENTITY_ABS_M
    )
    # 差超过 1e-12 m，或缺 RMSE 时，只能是 refused，并且不能当成同一次测量。
    # 已经对上的测量也不会再标成 refused。
    if pattern == "refused" and same:
        reasons.append("rmse_mismatch_not_refused:%s" % name)
    elif delta_bad or not numbers_ok:
        if pattern != "refused" or same:
            reasons.append("rmse_mismatch_not_refused:%s" % name)
    elif pattern != "refused" and not same:
        reasons.append("rmse_mismatch_not_refused:%s" % name)
    rerun = case.get("export_rerun") if isinstance(case.get("export_rerun"), dict) else {}
    diff = rerun.get("abs_diff_m")
    if _finite_rmse(diff) and float(diff) > RMSE_IDENTITY_ABS_M and rerun.get("used_as_this_measurement"):
        reasons.append("rmse_mismatch_not_refused:%s:export_rerun" % name)
    if isinstance(timeline, dict):
        actual = _mean_subtract_block(timeline)
        if case.get("mean_subtract") != actual:
            reasons.append("mean_subtract_inconsistent:%s" % name)
        if "error_vectors_m" not in timeline:
            claimed = case.get("mean_subtract") if isinstance(case.get("mean_subtract"), dict) else {}
            if (
                actual["status"] != "absent_from_json"
                or actual["rmse_after_m"] is not None
                or claimed.get("status") != "absent_from_json"
                or claimed.get("rmse_after_m") is not None
            ):
                reasons.append("mean_subtract_unknown:%s" % name)
    short = case.get("short_clip_package") if isinstance(case.get("short_clip_package"), dict) else {}
    if short.get("copied_rmse_m") is not None:
        reasons.append("package_accuracy_invented:%s:short_clip" % name)
    return reasons


def check_committed_diagnosis(docs_dir=None):
    """已提交的残差报告必须和当前时间线重算的结果一致。缺文件即失败。"""
    docs_dir = Path(docs_dir or default_docs())
    report_path = docs_dir / "residual_shape_diagnosis.json"
    if not docs_dir.is_dir() or not report_path.is_file():
        return ["residual_report_missing: %s" % report_path]
    try:
        committed = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return ["residual_report_unreadable: %s" % exc]
    if not isinstance(committed, dict):
        return ["residual_report_unreadable: 报告不是对象"]
    try:
        fresh = diagnose_repo(docs_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return ["residual_report_unreadable: %s" % exc]
    reasons = []
    if committed != fresh:
        reasons.append("residual_report_inconsistent: 已提交报告和 diagnose_repo 不是同一份")
    if committed.get("kind") != REPORT_KIND or fresh.get("kind") != REPORT_KIND:
        reasons.append("residual_report_inconsistent: kind")
    if committed.get("accuracy") is not None or fresh.get("accuracy") is not None:
        reasons.append("diagnosis_accuracy_not_null: 残差报告的 accuracy 必须是 null")
    if committed.get("new_ate") is not False or fresh.get("new_ate") is not False:
        reasons.append("diagnosis_new_ate: 这轮没有新的 Umeyama")
    if committed.get("package_accuracy_rule") != PACKAGE_ACCURACY_RULE:
        reasons.append("diagnosis_package_rule: 包级规则是 %s" % PACKAGE_ACCURACY_RULE)
    timeline_paths = sorted(docs_dir.glob("*_timeline.json"))
    stems = [path.name[: -len("_timeline.json")] for path in timeline_paths]
    cases = committed.get("cases")
    names = [
        case.get("name") if isinstance(case, dict) else None
        for case in cases
    ] if isinstance(cases, list) else None
    if names != stems:
        reasons.append("residual_cases_not_committed_timelines")
        return reasons
    by_name = {case["name"]: case for case in cases}
    for path, stem in zip(timeline_paths, stems):
        try:
            timeline = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            reasons.append("residual_report_unreadable: %s" % exc)
            continue
        reasons.extend(_case_artifact_reasons(by_name[stem], timeline))
    return reasons


def tracked_mano_weights(root=None):
    """仓库里跟踪了 MANO 权重就失败。文档里提到 MANO 不算。"""
    root = Path(root or repo_root())
    try:
        out = subprocess.check_output(
            ["git", "ls-files", "-z"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return ["stop_mano: 不能列出仓库文件（%s）" % exc]
    reasons = []
    for raw in out.split(b"\0"):
        if not raw:
            continue
        name = raw.decode("utf-8", "replace")
        base = name.rsplit("/", 1)[-1].lower()
        if "mano" in base and base.endswith((".pkl", ".npz", ".zip")):
            reasons.append("stop_mano: 仓库跟踪了 %s" % name)
    return reasons


def check_repo(root=None, docs_dir=None):
    root = Path(root or repo_root())
    reasons = []
    checklist = root / "docs" / "headcam" / "verification_gate.md"
    if not checklist.is_file():
        reasons.append("checklist_missing: docs/headcam/verification_gate.md")
    reasons.extend(tracked_mano_weights(root))
    reasons.extend(check_committed_diagnosis(docs_dir))
    return reasons


def run_required_tests():
    """跑头戴 MVP、双目 QC 和这道门自己的单测。skip 只来自缺可选依赖时算通过。"""
    import io
    import unittest

    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromNames(list(_REQUIRED_TEST_MODULES))
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=1, buffer=True).run(suite)
    failed = []
    for test, _trace in list(result.failures) + list(result.errors):
        failed.append("tests_failed: %s" % test.id())
    if result.wasSuccessful():
        line = "pass skipped=%d" % len(result.skipped)
    else:
        line = "fail skipped=%d failed=%d" % (len(result.skipped), len(failed))
    return failed, line, stream.getvalue()


def format_result(ok, fail, tests_line=None):
    lines = ["result: %s" % ("PASS" if ok else "FAIL")]
    if fail:
        lines.append("fail:")
        for item in fail:
            lines.append("- %s" % item)
    lines.append("held:")
    lines.extend("- %s" % item for item in HELD_LINES)
    if tests_line:
        lines.append("tests: %s" % tests_line)
    return "\n".join(lines) + "\n"


def _load_claim(path):
    path = Path(path)
    if not path.is_file():
        return None, ["claim_unreadable: %s" % path]
    try:
        claim = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, ["claim_unreadable: %s" % exc]
    return claim, []


def main(argv=None):
    parser = argparse.ArgumentParser(description="Headcam MVP 一轮增量的验收门。不新算 ATE，不合并。")
    parser.add_argument("--claim", default=None, help="headcam_mvp_increment_v1 JSON。不给则只核对已提交报告。")
    parser.add_argument("--docs", default=None, help="残差时间线目录，默认 docs/headcam/orbslam3_quest3")
    parser.add_argument("--skip-tests", action="store_true", help="不跑单测。这不算合格命令。")
    parser.add_argument("--no-repo", action="store_true", help="不核对已提交报告。这不算合格命令。")
    args = parser.parse_args(argv)
    docs_dir = Path(args.docs) if args.docs else default_docs()
    fail = []
    repo_checked = not args.no_repo
    if repo_checked:
        fail.extend(check_repo(docs_dir=docs_dir))
    if args.claim:
        claim, claim_fail = _load_claim(args.claim)
        fail.extend(claim_fail)
        if claim is not None:
            already = [] if repo_checked else None
            fail.extend(evaluate_claim(claim, diagnosis_reasons=already, docs_dir=docs_dir))
    tests_line = None
    if not args.skip_tests:
        test_fail, tests_line, test_log = run_required_tests()
        fail.extend(test_fail)
        if test_fail:
            print(test_log, file=sys.stderr, end="")
    ok = not fail
    sys.stdout.write(format_result(ok, fail, tests_line))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
