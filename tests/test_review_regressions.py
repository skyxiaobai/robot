"""Review 的七项回归：真实 HEVC 解码、小样本导出和训练／验证边界。"""
import json
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "tests")]
from test_lerobot_export import _episode, _write_episode, _yield_csv
from egodata.lerobot_export import export_lerobot
from egodata.qc import qc_episode
from egodata.schema import iter_episode_paths, validate_episode
from headcam.rgbd_pipeline import load_rgbd_session
from iphone import mcap_adapter as mcap
from validate_session import validate_iphone
import ego_act_scaling as act
import hot3d_train_compare as ht
import train_xhey as xhey
import wilor_video_to_episodes as wv

try:
    import torch
except ImportError:
    torch = None


class _Message:
    def ParseFromString(self, data):
        self.__dict__.update(data)


def _capture_messages(packets, pose_indices):
    """仅替代 MCAP/protobuf 容器；压缩图像仍是实际 HEVC 数据。"""
    messages = []
    def add(topic, data):
        messages.append((None, NS(topic=topic), NS(data=data)))
    def header(t):
        return NS(timestamp=NS(sec=int(t), nanosec=int(round((t - int(t)) * 1e9))))
    add(mcap.TOPICS["color_info"], {"width": 64, "height": 48, "k": [100, 0, 32, 0, 100, 24, 0, 0, 1]})
    add(mcap.TOPICS["depth_info"], {"width": 4, "height": 4})
    for i, blob in enumerate(packets):
        t = 100.0 + i / 30.0
        add(mcap.TOPICS["color"], {"header": header(t), "data": blob, "format": "h265"})
        add(mcap.TOPICS["depth"], {"header": header(t), "data": np.full((4, 4), 10000).astype(">u2").tobytes(),
                                 "depth_scale": 10000, "format": "16UC1"})
        if i in pose_indices:
            transform = NS(translation=NS(x=i * 0.001, y=0, z=0), rotation=NS(x=0, y=0, z=0, w=1))
            add(mcap.TOPICS["tf"], {"transforms": [NS(header=header(t), child_frame_id="color_optical_frame",
                                                     transform=transform)]})
    return messages


def _convert_mock_capture(path, out, messages, **kwargs):
    reader = types.ModuleType("mcap.reader")
    reader.make_reader = lambda fh: NS(iter_messages=lambda **kw: iter(messages))
    with patch.dict(sys.modules, {"mcap": types.ModuleType("mcap"), "mcap.reader": reader}), \
            patch.object(mcap, "_pool_from_mcap", return_value=None), \
            patch.object(mcap, "_cls", return_value=_Message):
        return mcap.convert_mcap(path, out, **kwargs)


def _read_video(path):
    cap = cv2.VideoCapture(str(path)); frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.asarray(frames), fps


class McapRegressionTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ffmpeg"), "需要 ffmpeg")
    def test_stride_preserves_hevc_references_pixels_and_fps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / "source.h265"
            command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=30",
                       "-frames:v", "12", "-c:v", "libx265", "-x265-params", "keyint=30:bframes=0:aud=1:log-level=error",
                       "-f", "hevc", str(source)]
            result = subprocess.run(command, capture_output=True)
            if result.returncode and b"Unknown encoder" in result.stderr:
                self.skipTest("ffmpeg 未包含 libx265 测试编码器")
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            blob = source.read_bytes()
            starts = [match.start() for match in re.finditer(b"\x00\x00\x00\x01", blob)
                      if (blob[match.end()] >> 1) & 63 == 35]
            packets = [blob[start:starts[i + 1] if i + 1 < len(starts) else len(blob)]
                       for i, start in enumerate(starts)]
            self.assertEqual(len(packets), 12)
            capture = root / "capture.mcap"; capture.touch()
            out = root / "session"
            metadata = _convert_mock_capture(capture, out, _capture_messages(packets, range(12)), stride=2)
            reference = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(source), "-f", "rawvideo",
                                        "-pix_fmt", "rgb24", "pipe:1"], capture_output=True, check=True).stdout
            reference = np.frombuffer(reference, np.uint8).reshape(12, 48, 64, 3)[::2]
            # 参考序列用同一编码参数封装，比较逐像素结果，避免把有损编码误差当成抽帧错误。
            reference_path = root / "expected.mp4"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", "64x48", "-r", str(metadata["fps"]), "-i", "pipe:0", "-an", "-c:v", "libx264",
                            "-crf", "18", "-pix_fmt", "yuv420p", str(reference_path)], input=reference.tobytes(), check=True)
            expected, _ = _read_video(reference_path)
            frames, video_fps = _read_video(out / "rgb.mp4")
            self.assertEqual(len(frames), 6)
            self.assertAlmostEqual(video_fps, 15, places=3)
            self.assertAlmostEqual(metadata["fps"], 15, places=3)
            np.testing.assert_array_equal(frames, expected)
            self.assertEqual(len(np.load(out / "depth.npz")["depth"]), 6)
            self.assertTrue(validate_iphone(out)["ok"])

    def test_missing_or_partial_pose_fails_before_video_output(self):
        for poses in ([], [0]):
            with self.subTest(poses=poses), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); capture = root / "capture.mcap"; capture.touch()
                with self.assertRaisesRegex(ValueError, "相机位姿"):
                    _convert_mock_capture(capture, root / "out", _capture_messages([b"unused"] * 12, poses), stride=2)
                self.assertFalse((root / "out" / "rgb.mp4").exists())
        self.assertIsNone(mcap._nearest_pose([(0, 0, 0, 0, 0, 0, 0, 0)], 0))
        self.assertIsNone(mcap._nearest_pose([(0, float("nan"), 0, 0, 0, 0, 0, 1)], 0))

    def test_old_identity_pose_and_wrong_video_fps_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer = cv2.VideoWriter(str(root / "rgb.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
            for _ in range(6):
                writer.write(np.zeros((48, 64, 3), np.uint8))
            writer.release()
            np.savez(root / "depth.npz", depth=np.ones((6, 4, 4)), conf=np.full((6, 4, 4), 2))
            (root / "calib.yaml").write_text(json.dumps({"K_rgb": [[100, 0, 32], [0, 100, 24], [0, 0, 1]],
                                                        "image_width": 64, "image_height": 48}))
            (root / "metadata.json").write_text(json.dumps({"fps": 15, "capture_app": "xhey_mcap", "n_tf_poses": 0}))
            (root / "timestamps.csv").write_text("frame,timestamp_s\n" + "".join("%d,%.8f\n" % (i, i / 15) for i in range(6)))
            (root / "slam.tum").write_text("".join("%.8f 0 0 0 0 0 0 1\n" % (i / 15) for i in range(6)))
            report = validate_iphone(root)
            self.assertFalse(report["ok"])
            self.assertTrue(any("真实相机位姿" in e for e in report["errors"]))
            self.assertTrue(any("视频帧率" in e for e in report["errors"]))
            with self.assertRaisesRegex(ValueError, "未通过校验"):
                load_rgbd_session(root)


class ExportAndCalibrationTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ffmpeg"), "需要 ffmpeg")
    def test_epoch_and_irregular_source_times_align_with_cfr_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); episodes = root / "episodes"; episodes.mkdir()
            epoch = 1700000000.0
            ep = _episode("offset", n=6, subtasks=[{"t_start": epoch + .04, "t_end": epoch + .20, "text": "源时钟指令"}])
            ep["timestamps"] = [epoch + t for t in (0, .05, .25, .31, .52, .62)]
            _write_episode(episodes, ep)
            for name, data in (("manifest.json", [{"id": "offset"}]), ("index.json", {}), ("convert_summary.json", {})):
                (episodes / name).write_text(json.dumps(data))
            self.assertEqual(len(iter_episode_paths(episodes)), 1)
            self.assertEqual([qc_episode(json.loads(p.read_text()))["episode_id"]
                              for p in iter_episode_paths(episodes)], ["offset"])
            with self.assertRaisesRegex(ValueError, "转换元数据"):
                iter_episode_paths(episodes / "manifest.json")
            _yield_csv(root / "yield.csv", [{"episode_id": "offset", "num_frames": 6, "accepted": "yes", "reasons": ""}])
            export_lerobot(episodes, root / "yield.csv", root / "out")
            rows = pq.read_table(root / "out/data/chunk-000/file-000.parquet").to_pydict()
            np.testing.assert_allclose(rows["timestamp"], np.arange(5) / 10, atol=1e-7)
            metadata = pq.read_table(root / "out/meta/episodes/chunk-000/file-000.parquet").to_pylist()[0]
            self.assertTrue(all(t < metadata["videos/observation.image/to_timestamp"] for t in rows["timestamp"]))
            tasks = pq.read_table(root / "out/meta/tasks.parquet").to_pandas()
            self.assertEqual(tasks.iloc[rows["task_index"][1]].name, "源时钟指令")

    def test_video_converter_requires_and_passes_scaled_real_intrinsics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); video = root / "RGB.mp4"
            with self.assertRaisesRegex(ValueError, "--calib"):
                wv.process_video(video, root / "out", None, "hub_rgb", 1, 0, .5, "")
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (32, 24))
            for _ in range(20):
                writer.write(np.zeros((24, 32, 3), np.uint8))
            writer.release()
            seen = []
            class Backend:
                def predict(self, image, calib):
                    seen.append(calib["K_left"])
                    points = np.zeros((21, 3)); points[5] = [.08, 0, 0]; points[17] = [.02, .07, 0]
                    points += [len(seen) * .001, 0, .5]
                    return {"left": {"joints_cam": points, "confidence": [1] * 21}}
            calibration = {"K_rgb": [[100, 0, 32], [0, 100, 24], [0, 0, 1]], "image_width": 64, "image_height": 48}
            path = root / "calib.yaml"; path.write_text(json.dumps(calibration))
            result = wv.process_video(video, root / "out", Backend(), "hub_rgb", 1, 0, .5, "", wv.load_video_calibration(path))
            ep = json.loads(Path(result["path"]).read_text())
            self.assertEqual(validate_episode(ep), [])
            self.assertTrue(ep["hand_pose"]["metric_translation"])
            np.testing.assert_allclose(seen[0], [[50, 0, 16], [0, 50, 12], [0, 0, 1]])


class TrainingBoundaryTest(unittest.TestCase):
    def tearDown(self):
        ht.FEAT = "abs"

    def test_uncalibrated_exports_and_stale_cache_cannot_enter_metric_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / "source"; source.mkdir(); cache = root / "cache"; cache.mkdir()
            ep = _episode("video", n=40); ep["hand_pose"] = {"backend": "wilor"}; ep["camera_intrinsic"] = None
            path = _write_episode(source, ep)
            np.savez(cache / "hub_rgb_arrays.npz", state=np.zeros((1, 140)))
            with self.assertRaisesRegex(ValueError, "公制手腕标签"):
                xhey.build_external("hub_rgb", source, cache)
            ep["camera_intrinsic"] = [[1, 0, 8], [0, 1, 8], [0, 0, 1]]
            path.write_text(json.dumps(ep))
            data = xhey.build_external("hub_rgb", source, cache)
            self.assertEqual(len(data["state"]), 39)
            ep["camera_intrinsic"] = None; path.write_text(json.dumps(ep))
            with self.assertRaisesRegex(ValueError, "公制手腕标签"):
                xhey.build_external("hub_rgb", source, cache)

    @unittest.skipIf(torch is None, "需要 torch")
    def test_image_cache_rebuilt_when_source_video_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); video = root / "rgb.mp4"
            def write_video(value):
                writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (32, 24))
                for _ in range(2):
                    writer.write(np.full((24, 32, 3), value, np.uint8))
                writer.release()
            class Model(torch.nn.Module):
                def forward(self, x):
                    return x.mean(dim=(1, 2, 3))[:, None].repeat(1, 512)
            fake_vision = types.ModuleType("torchvision")
            fake_vision.models = NS(resnet18=lambda **kw: Model())
            ep = {"video_path": str(video), "num_frames": 2}
            write_video(0)
            with patch.dict(sys.modules, {"torchvision": fake_vision}):
                before = xhey.image_features(ep, "clip", root / "cache")
            with patch.dict(sys.modules, {"torchvision": None}):
                np.testing.assert_allclose(xhey.image_features(ep, "clip", root / "cache"), before, atol=.001)
            write_video(255)
            with patch.dict(sys.modules, {"torchvision": fake_vision}):
                after = xhey.image_features(ep, "clip", root / "cache")
            self.assertGreater(float(np.mean(after - before)), 1)

    def test_mirror_and_fast_augmentation_respect_original_inner_training_intervals(self):
        with tempfile.TemporaryDirectory() as tmp:
            eps = [_episode("a", n=101), _episode("b", n=101)]
            iph = xhey.finish(ht.to_arrays(eps), np.zeros((200, 512)))
            with patch.object(xhey, "image_features", return_value=np.zeros((101, 512))):
                extra = xhey.prepare_external(["iphone+aug"], iph, eps, ["a", "b"], NS(out=tmp))
            tr, va = xhey.split_inner(iph, ht.chunks(iph, [0]))
            allowed = np.zeros(200, bool)
            for offset in range(xhey.H):
                allowed[tr + offset] = True
            for name in ("mirror", "fast"):
                data = extra[name]
                selected = xhey.augmentation_train_starts(data, tr, 200)
                self.assertTrue(len(selected))
                for start in selected:
                    begin = data["src_start"][start]; end = data["src_end"][start + xhey.H - 1]
                    self.assertTrue(allowed[begin:end].all())
                    self.assertLess(end, int(va[0]))
            np.testing.assert_array_equal(xhey.augmentation_train_starts(extra["mirror"], tr, 200), tr)
            self.assertEqual(extra["fast"]["src_start"][50], 100)

    def test_eval_sampling_covers_each_episode_with_complete_windows(self):
        episodes = np.repeat(np.arange(5), 40)
        indices = act.evaluation_indices(episodes, 16, 2)
        np.testing.assert_array_equal(episodes[indices], np.arange(5))
        self.assertTrue(all(np.unique(episodes[i:i + 16]).size == 1 for i in indices))
        indices = act.evaluation_indices(np.repeat([0, 1], 40), 16, 8)
        self.assertEqual(indices.tolist(), [0, 8, 16, 24, 40, 48, 56, 64])

    @unittest.skipIf(torch is None, "需要 torch")
    def test_act_and_still_baseline_share_targets_samples_and_hand_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "meta").mkdir(); (root / "data/chunk-000").mkdir(parents=True)
            (root / "meta/info.json").write_text('{"fps": 10}')
            (root / "meta/egodata_export.json").write_text('{"horizon": 2}')
            policy_dir = root / "policy"; policy_dir.mkdir(); (policy_dir / "config.json").write_text('{}')
            valid = np.tile([1, 0], (12, 1)).astype(np.float32)
            column = pa.FixedSizeListArray.from_arrays(pa.array(valid.reshape(-1)), 2)
            pq.write_table(pa.table({"action_valid": column}), root / "data/chunk-000/file-000.parquet")
            actions = np.tile(ht.IDENT, (12, 1)).astype(np.float32)
            actions[:6, 0] = 1; actions[6:, 0] = 3; actions[:, 7] = 1000
            calls = []
            class Dataset:
                hf_dataset = {"episode_index": np.repeat([0, 1], 6)}
                def __init__(self, *args, **kwargs): pass
                def __getitem__(self, i):
                    calls.append(i)
                    return {"index": torch.tensor(i), "action": torch.tensor(actions[i:i + 2]),
                            "observation.state": torch.zeros(3), "action_is_pad": torch.zeros(2, dtype=torch.bool)}
            class Policy:
                config = NS()
                @classmethod
                def from_pretrained(cls, path): return cls()
                def eval(self): return self
                def to(self, device): return self
                def predict_action_chunk(self, batch): return torch.tensor(ht.IDENT, dtype=torch.float32).repeat(1, 2, 1)
            modules = {}
            for name in ("lerobot", "lerobot.datasets", "lerobot.policies", "lerobot.policies.act"):
                modules[name] = types.ModuleType(name)
            modules["lerobot.datasets.lerobot_dataset"] = NS(LeRobotDataset=Dataset)
            modules["lerobot.policies.act.modeling_act"] = NS(ACTPolicy=Policy)
            modules["lerobot.policies.factory"] = NS(make_pre_post_processors=lambda **kw: (lambda x: x, lambda x: x))
            with patch.dict(sys.modules, modules):
                loss, baseline = act.evaluate_act_l1(root, policy_dir, [0, 1], "cpu", 2, max_batches=1)
                self.assertEqual(calls, [0, 6])
                self.assertAlmostEqual(loss, 2 / 7)
                self.assertAlmostEqual(loss, baseline)
                with self.assertRaisesRegex(ValueError, "掩码滤空"):
                    act.evaluate_act_l1(root, policy_dir, [0, 1], "cpu", 2, max_batches=1, min_valid_fraction=.75)


if __name__ == "__main__":
    unittest.main()
