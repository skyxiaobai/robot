# -*- coding: utf-8 -*-
"""骨架叠加、质检对照和三维轨迹。用临时小视频，不读 EgoDex。"""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import ego_visualize as viz  # noqa: E402
from egodata.qc import _project, frame_qc_flags, qc_episode  # noqa: E402


def _episode(episode_id, xs, fx=16.0, size=32, confidence=0.99, right_offset=0.0):
    n = len(xs)
    intrinsic = [[fx, 0.0, size / 2.0], [0.0, fx, size / 2.0], [0.0, 0.0, 1.0]]
    poses = [np.eye(4).tolist() for _ in range(n)]

    def hand(offset):
        joints = []
        for x in xs:
            frame = [[None, None, None] for _ in range(21)]
            frame[0] = [float(x) + offset, 0.0, 1.0]
            frame[1] = [float(x) + offset + 0.55, 0.0, 1.0]
            joints.append(frame)
        return {
            "joints": joints,
            "wrist_pose": [[float(x) + offset, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] for x in xs],
            "confidence": [confidence] * n,
            "valid": [True] * n,
        }

    return {
        "episode_id": episode_id,
        "fps": 30.0,
        "num_frames": n,
        "image_width": size,
        "image_height": size,
        "camera_intrinsic": intrinsic,
        "camera_poses": poses,
        "hands": {"left": hand(0.0), "right": hand(right_offset)},
        "annotation": {"task": {"name": episode_id, "instruction": "拿起杯子"}},
        "source_path": "",
    }


def _video(path, n, size, color):
    raw = bytes(color) * (size * size)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", "%dx%d" % (size, size),
            "-r", "30", "-i", "pipe:0",
            "-frames:v", str(n),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(path),
        ],
        input=raw * n,
        check=True,
    )


class OverlayTest(unittest.TestCase):
    def test_skeleton_uses_qc_projection_on_a_generated_video(self):
        xs = [0.01 * i for i in range(8)]
        episode = _episode("keep", xs, right_offset=0.9)
        wrist = episode["hands"]["left"]["joints"][3][0]
        uv = _project(episode["camera_intrinsic"], episode["camera_poses"][3], wrist)
        drawn = viz.project_joint(episode, 3, wrist)
        self.assertEqual(drawn, uv)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "keep.mp4"
            _video(video, 8, 32, (0, 40, 0))
            png = root / "overlay.png"
            mp4 = root / "overlay.mp4"
            viz.write_overlay(episode, png, video=video, frame=3, width=32, trail=4)
            viz.write_overlay(episode, mp4, video=video, width=32, trail=2, limit_frames=3)
            with Image.open(png) as opened:
                image = np.asarray(opened.convert("RGB"))
            u, v = int(round(uv[0])), int(round(uv[1]))
            pixel = image[v, u]
            self.assertGreater(int(pixel[0]), 180)
            self.assertLess(int(pixel[1]), 180)
            self.assertTrue(mp4.is_file())
            self.assertGreater(mp4.stat().st_size, 200)
            from egodata.schema import save_episode
            saved = root / "keep.json"
            save_episode(episode, saved)
            cli_png = root / "cli.png"
            self.assertEqual(viz.main([
                "overlay", "--episode", str(saved), "--video", str(video),
                "--out", str(cli_png), "--frame", "3", "--width", "32",
            ]), 0)
            self.assertTrue(cli_png.is_file())


class QcCompareTest(unittest.TestCase):
    def test_panels_follow_qc_reasons_and_write_png(self):
        moving = [0.02 * i for i in range(8)]
        keep = _episode("keep", moving, right_offset=0.15)
        outside = _episode("outside", [8.0] * 8)
        drift = _episode("drift", [1.5 + 0.01 * i for i in range(8)], fx=8.0)
        self.assertTrue(qc_episode(keep)["accepted"])
        self.assertIn("hands_out_of_frame", qc_episode(outside)["reasons"])
        self.assertIn("view_drift", qc_episode(drift)["reasons"])
        for episode in (keep, outside, drift):
            result = qc_episode(episode)
            flags = frame_qc_flags(episode)
            for name, values in flags.items():
                self.assertEqual(int(values.sum()), result["flags"][name])
        panels = viz.select_qc_panels([keep, outside, drift])
        kinds = [item[0] for item in panels]
        self.assertEqual(kinds[0], "accepted")
        self.assertIn("hands_out_of_frame", kinds)
        self.assertIn("view_drift", kinds)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "qc.png"
            viz.write_qc_compare([keep, outside, drift], out, width=64)
            with Image.open(out) as image:
                self.assertEqual(image.format, "PNG")
                self.assertGreater(image.size[0], 100)


class Traj3dTest(unittest.TestCase):
    def test_world_trajectories_write_png(self):
        episode = _episode("keep", [0.02 * i for i in range(8)], right_offset=0.2)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "traj.png"
            viz.write_traj3d(episode, out, frame=2)
            with Image.open(out) as image:
                self.assertEqual(image.format, "PNG")
                self.assertGreater(image.size[0], 100)
            from egodata.schema import save_episode
            saved = Path(tmp) / "keep.json"
            save_episode(episode, saved)
            cli = Path(tmp) / "cli_traj.png"
            self.assertEqual(viz.main([
                "traj3d", "--episode", str(saved), "--out", str(cli), "--frame", "2",
            ]), 0)
            self.assertTrue(cli.is_file())
