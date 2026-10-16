# -*- coding: utf-8 -*-
"""打印一节会话的 ok / tracking_lost / pose_missing 帧数。

不跑手部，不写精度。每一帧都是 ok 时 ``world_hands=allowed``，否则
``world_hands=refused`` 并且默认退出码 1。``--allow-partial`` 只把退出码改成 0，
不把残缺轨迹说成世界系双手。

    python scripts/headcam/summarize_slam.py --session /data/sess01 --tum CameraTrajectory.txt
"""
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from headcam.slam_vio import format_slam_counts, run_slam, world_hands_gate  # noqa: E402


def summarize_session(session, tum_path=None, backend="auto", max_gap_s=None, allow_partial=False, out_dir=None):
    """返回 ``(text, exit_code, result)``。``result`` 是 ``run_slam`` 的字典。"""
    result = run_slam(
        session, backend=backend, tum_path=tum_path, max_gap_s=max_gap_s, out_dir=out_dir,
    )
    gate = world_hands_gate(result, allow_partial=allow_partial)
    return format_slam_counts(result, gate), gate["exit_code"], result


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="打印 SLAM 逐帧状态计数。不声称轨迹误差。")
    parser.add_argument("--session", required=True)
    parser.add_argument("--tum", default=None, help="外部 TUM，例如 ORB-SLAM3 的 CameraTrajectory.txt")
    parser.add_argument("--backend", default="auto", choices=("auto", "none", "external_tum", "orbslam3"))
    parser.add_argument("--max-gap-s", type=float, default=None)
    parser.add_argument("--out", default=None, help="可选。写出 slam_status.json；完整轨迹才写 slam.tum")
    parser.add_argument("--allow-partial", action="store_true",
                        help="覆盖不全时退出码改为 0。world_hands 仍是 refused")
    args = parser.parse_args(argv)
    backend = args.backend
    if args.tum and backend == "auto":
        backend = "external_tum"
    try:
        text, code, _result = summarize_session(
            args.session, tum_path=args.tum, backend=backend, max_gap_s=args.max_gap_s,
            allow_partial=args.allow_partial, out_dir=args.out,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    sys.stdout.write(text)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
