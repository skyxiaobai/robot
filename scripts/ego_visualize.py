#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把统一 episode 画出来：手骨架叠加、质检对照、世界系轨迹。

投影用 ``egodata.qc._project``（OpenCV：相机 +Z 向前、+Y 向下），和出画判断同一套。
没有源 mp4 时画在灰底上，方便合成样本；有视频时按 ``image_width`` / ``image_height``
把关节对齐到画面上。不要把 EgoDex 的帧或 mp4 提交进仓库。

示例:
    python scripts/ego_visualize.py overlay \\
        --episode outputs/open_data_demo/unified/basic_pick_place/0.json \\
        --out outputs/viz/overlay.png --frame 5 --width 640
    python scripts/ego_visualize.py qc_compare \\
        --episodes outputs/open_data_demo/unified --out outputs/viz/qc.png
    python scripts/ego_visualize.py traj3d \\
        --episode outputs/open_data_demo/unified/basic_pick_place/0.json \\
        --out outputs/viz/traj.png
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))

from egodata.qc import DEFAULTS, _FLAG_LABELS, _project, frame_qc_flags, qc_episode  # noqa: E402
from egodata.schema import iter_episode_paths, load_episode  # noqa: E402

# MediaPipe 21 点的连线。0 是手腕。
HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)
HAND_COLOR = {"left": (255, 140, 0), "right": (0, 200, 255)}
_STRIP_COLOR = {
    "hands_out_of_frame": "#ef4444",
    "view_drift": "#a855f7",
    "blur": "#64748b",
    "staged_static": "#f59e0b",
}
_REASON_NOTE = {
    "accepted": "坏帧比例不超过 20%，整段留下",
    "hands_out_of_frame": "两只手腕都不在画面内，或置信度低于 0.5",
    "view_drift": "可见手腕偏离相机朝向超过 50°",
    "blur": "相机角速度超过 1.5 弧度/秒",
    "staged_static": "两只手腕都慢于 1.5 厘米/秒，且连续至少 1 秒",
}
_RESAMPLE = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
)


def task_text(episode):
    task = (episode.get("annotation") or {}).get("task") or {}
    instruction = str(task.get("instruction") or "").strip()
    if instruction:
        return instruction
    name = str(task.get("name") or "").strip()
    if name:
        return name
    return str(episode.get("episode_id") or "")


def resolve_video(episode, explicit=None, video_root=None):
    """显式路径、与 source_path 同名的 mp4，或 ``--video-root`` 下按 episode_id 找。"""
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError("找不到视频 %s" % path)
        return path
    source = episode.get("source_path")
    if source:
        candidate = Path(source).with_suffix(".mp4")
        if candidate.is_file():
            return candidate
    named = episode.get("video_path")
    if named and Path(named).is_file():
        return Path(named)
    if video_root and episode.get("episode_id"):
        relative = str(episode["episode_id"])
        if relative.startswith("egodex/"):
            relative = relative[len("egodex/"):]
        candidate = Path(video_root) / (relative + ".mp4")
        if candidate.is_file():
            return candidate
    return None


def _font(size):
    for path in _FONT_CANDIDATES:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _finite_point(point):
    if point is None:
        return False
    return all(coord is not None for coord in point)


def _confidence_ok(episode, side, index):
    value = episode["hands"][side]["confidence"][index]
    if value is None:
        return True
    return float(value) >= DEFAULTS["confidence_min"]


def project_joint(episode, camera_index, point):
    """用该帧相机把世界点投到原始图像像素。相机后方返回 None。"""
    if not _finite_point(point):
        return None
    return _project(episode["camera_intrinsic"], episode["camera_poses"][camera_index], point)


def _display_size(episode, width):
    image_w = int(episode["image_width"])
    image_h = int(episode["image_height"])
    if width is None or int(width) <= 0 or image_w <= int(width):
        return image_w, image_h
    scale = float(width) / float(image_w)
    return max(2, int(round(image_w * scale))), max(2, int(round(image_h * scale)))


def _even(value):
    value = max(2, int(value))
    if value % 2:
        value -= 1
    return value


def read_video_frame(path, index):
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise RuntimeError("读不到 %s 的第 %d 帧" % (path, index))
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def iter_video_frames(path, limit=None):
    import cv2

    capture = cv2.VideoCapture(str(path))
    count = 0
    try:
        while limit is None or count < int(limit):
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            count += 1
    finally:
        capture.release()


def _blank(width, height):
    return np.full((int(height), int(width), 3), 40, dtype=np.uint8)


def draw_overlay(episode, frame_index, rgb, out_size, trail=30, line_width=3):
    """把双手骨架和手腕轨迹画到一帧上。``out_size`` 是输出宽高。"""
    image = Image.fromarray(np.asarray(rgb)).convert("RGBA")
    if image.size != tuple(out_size):
        image = image.resize(tuple(out_size), _RESAMPLE)
    scale_x = float(out_size[0]) / float(episode["image_width"])
    scale_y = float(out_size[1]) / float(episode["image_height"])
    draw = ImageDraw.Draw(image, "RGBA")
    trail = int(trail)
    for side in ("left", "right"):
        color = HAND_COLOR[side]
        if trail > 0:
            trail_points = []
            start = max(0, int(frame_index) - trail)
            for past in range(start, int(frame_index) + 1):
                joints = episode["hands"][side]["joints"][past]
                wrist = joints[0] if joints else None
                # 过去的手腕位置，用当前相机来看，才是画面上的轨迹。
                uv = project_joint(episode, frame_index, wrist)
                if uv:
                    trail_points.append((uv[0] * scale_x, uv[1] * scale_y))
            if len(trail_points) > 1:
                draw.line(trail_points, fill=color + (180,), width=max(2, line_width - 1))
        if not _confidence_ok(episode, side, frame_index):
            continue
        projected = [
            project_joint(episode, frame_index, point)
            for point in episode["hands"][side]["joints"][frame_index]
        ]
        for start, end in HAND_EDGES:
            if start >= len(projected) or end >= len(projected):
                continue
            if projected[start] and projected[end]:
                draw.line(
                    [
                        (projected[start][0] * scale_x, projected[start][1] * scale_y),
                        (projected[end][0] * scale_x, projected[end][1] * scale_y),
                    ],
                    fill=color,
                    width=line_width,
                )
        for joint_index, uv in enumerate(projected):
            if not uv:
                continue
            radius = (line_width + 3) if joint_index == 0 else line_width
            x, y = uv[0] * scale_x, uv[1] * scale_y
            fill = color if joint_index == 0 else (255, 255, 255)
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=fill, outline=color)
    return image


def _banner(image, lines, size, top, bg=(0, 0, 0, 170)):
    draw = ImageDraw.Draw(image, "RGBA")
    font = _font(size)
    line_h = int(size * 1.45)
    bar = line_h * len(lines) + 16
    y0 = 0 if top else image.height - bar
    draw.rectangle([0, y0, image.width, y0 + bar], fill=bg)
    for index, text in enumerate(lines):
        draw.text((10, y0 + 8 + index * line_h), text, font=font, fill=(255, 255, 255))
    return image


def compose_frame(episode, frame_index, rgb, out_size, trail, line_width):
    image = draw_overlay(episode, frame_index, rgb, out_size, trail=trail, line_width=line_width)
    if out_size[1] < 80:
        return image.convert("RGB")
    text_size = 16 if out_size[0] < 400 else 22
    image = _banner(image, [task_text(episode)], text_size, top=True)
    caption = "橙=左手 21 关节  蓝=右手 21 关节  细线=当前相机下的手腕轨迹  第 %d 帧" % frame_index
    image = _banner(image, [caption], max(12, text_size - 4), top=False)
    return image.convert("RGB")


def _source_frame(episode, frame_index, video):
    if video is None:
        return _blank(episode["image_width"], episode["image_height"])
    return read_video_frame(video, frame_index)


def write_overlay(episode, out_path, video=None, frame=None, width=960, trail=30, limit_frames=None):
    """``.png`` 写一帧，``.mp4`` 写整段（可用 ``limit_frames`` 截短）。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()
    out_w, out_h = _display_size(episode, width)
    if suffix == ".mp4":
        out_w, out_h = _even(out_w), _even(out_h)
    out_size = (out_w, out_h)
    line_width = 2 if out_w < 200 else 4
    if suffix == ".png":
        index = int(frame) if frame is not None else _busiest_frame(episode)
        rgb = _source_frame(episode, index, video)
        compose_frame(episode, index, rgb, out_size, trail, line_width).save(out_path)
        return out_path
    if suffix != ".mp4":
        raise ValueError("overlay 只写 .png 或 .mp4，实际是 %s" % suffix)
    count = int(episode["num_frames"])
    if limit_frames is not None:
        count = min(count, int(limit_frames))
    fps = float(episode["fps"])
    if video is None:
        frames = [
            np.asarray(compose_frame(episode, index, _blank(episode["image_width"], episode["image_height"]), out_size, trail, line_width))
            for index in range(count)
        ]
        _write_mp4(out_path, frames, fps)
        return out_path
    frames = []
    for index, rgb in enumerate(iter_video_frames(video, count)):
        frames.append(np.asarray(compose_frame(episode, index, rgb, out_size, trail, line_width)))
    if not frames:
        raise RuntimeError("视频里没有帧：%s" % video)
    _write_mp4(out_path, frames, fps)
    return out_path


def _busiest_frame(episode):
    best_index, best_score = 0, -1
    for index in range(int(episode["num_frames"])):
        score = 0
        for side in ("left", "right"):
            if not _confidence_ok(episode, side, index):
                continue
            for point in episode["hands"][side]["joints"][index]:
                if project_joint(episode, index, point):
                    score += 1
        if score > best_score:
            best_index, best_score = index, score
    return best_index


def _write_mp4(path, frames, fps):
    first = np.asarray(frames[0])
    height, width = first.shape[:2]
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", "%dx%d" % (width, height),
        "-r", "%g" % float(fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for frame in frames:
            process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
    finally:
        process.stdin.close()
        code = process.wait()
    if code != 0:
        raise RuntimeError("ffmpeg 写 %s 失败，退出码 %s" % (path, code))


def select_qc_panels(episodes):
    """挑一条通过的，以及每种拒绝原因各一条。返回 (kind, episode, result, flags)。"""
    rows = []
    for episode in episodes:
        result = qc_episode(episode)
        flags = frame_qc_flags(episode)
        rows.append((episode, result, flags))
    panels = []
    accepted = [row for row in rows if row[1]["accepted"]]
    if accepted:
        chosen = min(accepted, key=lambda row: (row[1]["bad_fraction"], row[1]["episode_id"]))
        panels.append(("accepted", chosen[0], chosen[1], chosen[2]))
    for reason in _FLAG_LABELS:
        matched = [row for row in rows if reason in row[1]["reasons"]]
        if matched:
            panels.append((reason, matched[0][0], matched[0][1], matched[0][2]))
    return panels


def _shown_frame(kind, flags):
    length = int(len(next(iter(flags.values()))))
    if kind == "accepted":
        bad = np.zeros(length, dtype=bool)
        for values in flags.values():
            bad |= values
        good = np.flatnonzero(~bad)
        if len(good):
            return int(good[len(good) // 2])
        return 0
    hits = np.flatnonzero(flags[kind])
    if len(hits):
        return int(hits[len(hits) // 2])
    return 0


def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for path in _FONT_CANDIDATES:
        if Path(path).is_file():
            font_manager.fontManager.addfont(path)
            name = font_manager.FontProperties(fname=path).get_name()
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def write_qc_compare(episodes, out_path, video_root=None, width=480):
    """通过与各拒绝原因并排。每格下面是该片段的逐帧坏帧条。"""
    panels = select_qc_panels(episodes)
    if not panels:
        raise ValueError("没有可画的片段")
    plt = _setup_matplotlib()
    columns = 1 if len(panels) == 1 else 2
    rows = (len(panels) + columns - 1) // columns
    fig = plt.figure(figsize=(7.4 * columns, 4.8 * rows), dpi=110)
    outer = fig.add_gridspec(rows, columns, hspace=0.45, wspace=0.12)
    for index, (kind, episode, result, flags) in enumerate(panels):
        inner = outer[index // columns, index % columns].subgridspec(
            2, 1, height_ratios=[8, 1.3], hspace=0.12,
        )
        ax = fig.add_subplot(inner[0])
        strip = fig.add_subplot(inner[1])
        frame_index = _shown_frame(kind, flags)
        video = resolve_video(episode, video_root=video_root)
        rgb = _source_frame(episode, frame_index, video)
        out_w, out_h = _display_size(episode, width)
        image = draw_overlay(episode, frame_index, rgb, (out_w, out_h), trail=15, line_width=2)
        ax.imshow(np.asarray(image.convert("RGB")))
        ax.axis("off")
        if kind == "accepted":
            title = "通过（保留）"
            color = "#16a34a"
        else:
            title = "拒绝（丢弃）：%s" % _FLAG_LABELS[kind]
            color = "#dc2626"
        ax.set_title(
            "%s\n%s\n%s  坏帧 %.0f%%" % (
                title, _REASON_NOTE[kind], episode.get("episode_id", ""), 100 * result["bad_fraction"],
            ),
            color=color, fontsize=11, loc="left",
        )
        length = int(episode["num_frames"])
        for reason, values in flags.items():
            strip.fill_between(
                np.arange(length), 0, np.asarray(values, dtype=float),
                step="mid", color=_STRIP_COLOR[reason], alpha=0.85, label=_FLAG_LABELS[reason],
            )
        strip.axvline(frame_index, color="black", lw=1.2)
        strip.set_xlim(0, max(length - 1, 1))
        strip.set_yticks([])
        strip.set_xlabel("帧    黑线 = 上图这一帧", fontsize=9)
        if index == 0:
            strip.legend(loc="upper right", fontsize=8, ncol=4, frameon=False)
    fig.suptitle("质检：同一套规则下，留下的片段和被丢掉的片段", fontsize=14)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _wrist_track(episode, side):
    series = []
    for joints in episode["hands"][side]["joints"]:
        point = joints[0] if joints else None
        if _finite_point(point):
            series.append([float(coord) for coord in point])
        else:
            series.append([np.nan, np.nan, np.nan])
    return np.asarray(series, dtype=float)


def write_traj3d(episode, out_path, frame=None):
    """头（相机原点）和双手腕。图里 y 朝上，对应 ARKit 世界系的 y。"""
    plt = _setup_matplotlib()
    poses = np.asarray(episode["camera_poses"], dtype=float)
    camera = poses[:, :3, 3]
    wrists = {side: _wrist_track(episode, side) for side in ("left", "right")}
    fig = plt.figure(figsize=(8.2, 6.4), dpi=110)
    ax = fig.add_subplot(111, projection="3d")

    def axes_of(points):
        return points[:, 0], -points[:, 2], points[:, 1]

    ax.plot(*axes_of(camera), color="#334155", lw=2.4, label="头（相机）")
    ax.scatter(*axes_of(camera[:1]), color="#334155", s=40)
    span_points = [camera]
    for side, color, label in (
        ("left", "#f97316", "左手腕"),
        ("right", "#0ea5e9", "右手腕"),
    ):
        track = wrists[side]
        good = np.isfinite(track).all(axis=1)
        if good.any():
            ax.plot(*axes_of(track), color=color, lw=2.0, label=label)
            span_points.append(track[good])
    stacked = np.vstack(span_points)
    center = np.nanmean(stacked, axis=0)
    radius = float(np.nanmax(stacked.max(axis=0) - stacked.min(axis=0))) / 2.0
    if radius < 1e-3:
        radius = 0.2
    step = max(1, len(poses) // 8)
    for index in range(0, len(poses), step):
        forward = poses[index, :3, 2]
        norm = np.linalg.norm(forward)
        if norm < 1e-8:
            continue
        forward = forward / norm * (0.35 * radius)
        start = camera[index]
        end = start + forward
        ax.plot(
            [start[0], end[0]], [-start[2], -end[2]], [start[1], end[1]],
            color="#475569", lw=1.6,
        )
    skeleton_index = int(frame) if frame is not None else len(poses) // 2
    skeleton_index = min(max(skeleton_index, 0), len(poses) - 1)
    for side, color in (("left", "#f97316"), ("right", "#0ea5e9")):
        joints = episode["hands"][side]["joints"][skeleton_index]
        points = []
        for point in joints:
            if _finite_point(point):
                points.append([float(coord) for coord in point])
            else:
                points.append([np.nan, np.nan, np.nan])
        cloud = np.asarray(points, dtype=float)
        for start, end in HAND_EDGES:
            if start >= len(cloud) or end >= len(cloud):
                continue
            segment = cloud[[start, end]]
            if not np.isfinite(segment).all():
                continue
            ax.plot(*axes_of(segment), color=color, lw=1.5)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(-center[2] - radius, -center[2] + radius)
    ax.set_zlim(center[1] - radius, center[1] + radius)
    ax.set_xlabel("x（米）")
    ax.set_ylabel("-z（米）")
    ax.set_zlabel("高度 y（米）")
    ax.view_init(elev=22, azim=-60)
    ax.legend(loc="upper left", fontsize=9)
    duration = float(episode["num_frames"]) / float(episode["fps"])
    ax.set_title(
        "世界系轨迹：%s（%.1f 秒）\n灰线 = 头朝向（OpenCV +Z）  骨架 = 第 %d 帧"
        % (episode.get("episode_id") or "", duration, skeleton_index),
        fontsize=11,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _load_one(path):
    return load_episode(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="统一 episode 的骨架、质检对照和三维轨迹")
    sub = parser.add_subparsers(dest="command", required=True)

    overlay = sub.add_parser("overlay", help="手骨架、手腕轨迹和任务文字，输出 PNG 或 MP4")
    overlay.add_argument("--episode", required=True)
    overlay.add_argument("--video", default=None, help="源 mp4。省略时用与 HDF5 同名的 mp4，再没有就画灰底")
    overlay.add_argument("--out", required=True)
    overlay.add_argument("--frame", type=int, default=None, help="PNG 用这一帧。省略时选关节可见最多的一帧")
    overlay.add_argument("--width", type=int, default=960, help="输出宽度。不大于原图宽度时保持原尺寸")
    overlay.add_argument("--trail", type=int, default=30, help="手腕轨迹回溯的帧数")
    overlay.add_argument("--limit-frames", type=int, default=None, help="MP4 最多写这么多帧")

    compare = sub.add_parser("qc_compare", help="通过与各拒绝原因并排，下面是逐帧坏帧条")
    compare.add_argument("--episodes", required=True, help="一条 JSON，或统一 episode 目录")
    compare.add_argument("--video-root", default=None, help="按 episode_id 找 mp4 的目录")
    compare.add_argument("--out", required=True)
    compare.add_argument("--width", type=int, default=480)

    traj = sub.add_parser("traj3d", help="头和双手腕在世界系里的轨迹")
    traj.add_argument("--episode", required=True)
    traj.add_argument("--out", required=True)
    traj.add_argument("--frame", type=int, default=None, help="在这一帧上再画 21 点骨架")

    args = parser.parse_args(argv)
    if args.command == "overlay":
        episode = _load_one(args.episode)
        video = resolve_video(episode, args.video)
        if video is None:
            print("没有源视频，骨架画在灰底上")
        path = write_overlay(
            episode, args.out, video=video, frame=args.frame,
            width=args.width, trail=args.trail, limit_frames=args.limit_frames,
        )
    elif args.command == "qc_compare":
        paths = iter_episode_paths(args.episodes)
        if not paths:
            raise SystemExit("没有找到 episode JSON：%s" % args.episodes)
        episodes = [load_episode(path) for path in paths]
        path = write_qc_compare(episodes, args.out, video_root=args.video_root, width=args.width)
    else:
        path = write_traj3d(_load_one(args.episode), args.out, frame=args.frame)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
