# -*- coding: utf-8 -*-
"""一条已经接好的 Quest3 录制：去畸变，跑 stereo_euroc，写出 slam_batch 的一份记录。

二进制、词汇和图像不进仓库。失败时记录里的 accuracy 是 null，并写下退出码。
不把头戴真值写进 slam 轨迹。
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from headcam.hot3d_session import write_quest3_euroc  # noqa: E402
from headcam.slam_batch import score_sequence  # noqa: E402


def _library_path(orbslam_root):
    root = Path(orbslam_root)
    parts = [
        root / "lib",
        root / "Thirdparty" / "DBoW2" / "lib",
        root / "Thirdparty" / "g2o" / "lib",
        Path("/usr/local/lib"),
    ]
    current = os.environ.get("LD_LIBRARY_PATH", "")
    return ":".join([str(path) for path in parts if path.is_dir()] + ([current] if current else []))


def run_sequence(
    clip_dirs, work, orbslam_bin, vocab, orbslam_root,
    name=None, participant_id=None, session_out=None, episode_id=None,
):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    euroc = work / "euroc"
    gt_tum = work / "gt_headset_tum.txt"
    yaml_path = work / "cam.yaml"
    log_path = work / "stereo_euroc.log"
    record_path = work / "sequence.json"
    prepared = write_quest3_euroc(
        clip_dirs, euroc, gt_tum, yaml_path,
        session_out=session_out, episode_id=episode_id,
    )
    sequence_id = prepared["sequence_id"]
    participant = participant_id or prepared.get("participant_id") or str(sequence_id).split("_")[0]
    label = name or sequence_id
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = _library_path(orbslam_root)
    command = [str(orbslam_bin), str(vocab), str(yaml_path), str(euroc), str(euroc / "times.txt")]
    (work / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    with log_path.open("w", encoding="utf-8", errors="replace") as handle:
        completed = subprocess.run(command, cwd=str(work), env=env, stdout=handle, stderr=subprocess.STDOUT)
    trajectory = work / "CameraTrajectory.txt"
    failure = None
    if completed.returncode != 0:
        failure = "stereo_euroc 退出码 %s" % completed.returncode
    elif not trajectory.is_file():
        failure = "stereo_euroc 退出码 0，但没有 CameraTrajectory.txt"
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    record = score_sequence(
        name=label,
        sequence_id=sequence_id,
        participant_id=participant,
        clip_ids=prepared["clip_ids"],
        input_frames=prepared["frames"],
        trajectory_path=trajectory if trajectory.is_file() else None,
        gt_path=gt_tum,
        log_text=log_text,
        failure=failure,
        baseline_m=prepared["baseline_m"],
        extrinsic_drift_m=prepared["extrinsic_drift_m"],
    )
    record["fps"] = prepared["fps"]
    record["sync_error_ms_max"] = prepared["sync_error_ms_max"]
    record["image_width"] = prepared["image_width"]
    record["image_height"] = prepared["image_height"]
    record["intrinsic_maps_rebuilt"] = prepared["intrinsic_maps_rebuilt"]
    record["orbslam3_exit_code"] = int(completed.returncode)
    import json
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description="Quest3 连续 clip → ORB-SLAM3 stereo_euroc → 一条误差记录")
    parser.add_argument("--clips", nargs="+", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--orbslam-bin", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--orbslam-root", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--participant", default=None)
    parser.add_argument("--session", default=None, help="另写缺测手导出用的会话。不写 slam.tum")
    parser.add_argument("--episode-id", default=None)
    args = parser.parse_args(argv)
    try:
        record = run_sequence(
            args.clips, args.work, args.orbslam_bin, args.vocab, args.orbslam_root,
            name=args.name, participant_id=args.participant,
            session_out=args.session, episode_id=args.episode_id,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(record["status"], record["name"], "frames", record["input_frames"], "tracked", record["tracked_frames"])
    return 0 if record["status"] != "failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
