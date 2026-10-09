"""Resolve this experiment repo's root for the analysis scripts.

Priority: ``--root``, then ``$ROBOT_ROOT``, then the directory that contains ``scripts/``.
"""
import argparse
import os


def default_root():
    env = os.environ.get("ROBOT_ROOT")
    if env:
        return env
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def add_root_argument(parser):
    parser.add_argument(
        "--root",
        default=default_root(),
        help="工程根目录（默认环境变量 ROBOT_ROOT，否则为仓库根目录）",
    )
    return parser


def resolve_root(argv=None):
    parser = argparse.ArgumentParser()
    add_root_argument(parser)
    args = parser.parse_args(argv)
    return os.path.abspath(args.root)


def dataset_fps(info):
    """Read fps from a LeRobot ``info.json``.

    This dataset stores it at the top level (``fps``). Fall back to the legacy
    ``video_info['video.fps']`` key, then to the per-feature video metadata.
    """
    if not isinstance(info, dict):
        return None
    if info.get("fps") is not None:
        return info.get("fps")
    video_info = info.get("video_info")
    if isinstance(video_info, dict) and video_info.get("video.fps") is not None:
        return video_info.get("video.fps")
    features = info.get("features") or {}
    if not isinstance(features, dict):
        return None
    for feat in features.values():
        if not isinstance(feat, dict):
            continue
        feat_info = feat.get("info")
        if isinstance(feat_info, dict) and feat_info.get("video.fps") is not None:
            return feat_info.get("video.fps")
    return None
