"""Phase-0 offline regressions. Synthetic episodes/video only; no downloads/launches."""
import ast
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "tests")]
import train_xhey as tx
import xhey_auto_iter as ai
from xhey_ledger import decision_status, reviewed_rows
from xhey_public import constant_velocity, read_split
from xhey_stats import cluster_summary
from test_lerobot_export import _episode
from test_open_data_pipeline import _write_egodex_hdf5
from egodata.egodex import load_episode_hdf5


class TinyEncoder(torch.nn.Module):
    """Only replace pretrained weights, not video decoding or the trainable MLP."""
    def forward(self, x):
        return x.mean((1, 2, 3))[:, None].repeat(1, 512)


def fixture(root, n=48):
    source = root / "episodes"; source.mkdir()
    for i in range(6):
        ep = _episode("egodex/clip%d" % i, n=n)
        ep["fps"] = 30
        ep["timestamps"] = (np.arange(n) / 30).tolist()
        ep["video_path"] = "clip%d.avi" % i
        path = source / ep["video_path"]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (32, 24))
        assert writer.isOpened()
        for f in range(n):
            writer.write(np.full((24, 32, 3), (f * 4 + i) % 255, np.uint8))
        writer.release()
        (source / ("clip%d.json" % i)).write_text(json.dumps(ep))
    ids = ["egodex/clip%d" % i for i in range(6)]
    split = {"unit": "session", "train": ids[:2], "val": ids[2:3], "test": ids[3:],
             "groups": {eid: "session%d" % (3 if i == 4 else i) for i, eid in enumerate(ids)}}
    path = root / "split.json"; path.write_text(json.dumps(split))
    return source, path


@pytest.fixture(autouse=True)
def clean_globals():
    tx.QC_SCORED.clear(); tx.QC_EXCLUDE.clear(); tx.SEG_KEEP.clear()
    old_feat = tx.ht.FEAT
    yield
    tx.QC_SCORED.clear(); tx.QC_EXCLUDE.clear(); tx.SEG_KEEP.clear()
    tx.ht.FEAT = old_feat


def test_missing_empty_zero_sources_never_silently_skip(tmp_path):
    with pytest.raises(ValueError, match="Missing requested source xphi"):
        tx.try_external("xphi", tmp_path / "absent", tmp_path / "cache")
    source = tmp_path / "empty"; source.mkdir()
    with pytest.raises(ValueError, match="No usable episodes.*xphi"):
        tx.try_external("xphi", source, tmp_path / "cache")
    ep = _episode("bad", n=48)
    ep["good_frame_mask"] = [False] * 48
    for hand in ep["hands"].values():
        hand["valid"] = [False] * 48
    (source / "bad.json").write_text(json.dumps(ep))
    with pytest.raises(ValueError, match="zero valid training windows"):
        tx.build_external("egodex", source, tmp_path / "cache", use_img=False)
    tx.QC_SCORED.add("bad"); tx.QC_EXCLUDE.add("bad")
    with pytest.raises(ValueError, match="missing/removed by QC"):
        tx.build_external("egodex", source, tmp_path / "cache", use_img=False, episode_ids=["bad"])


def test_split_rejects_session_and_episode_leakage(tmp_path):
    _, path = fixture(tmp_path)
    s = read_split(path)
    s["groups"][s["val"][0]] = s["groups"][s["train"][0]]
    path.write_text(json.dumps(s))
    with pytest.raises(ValueError, match="group leakage"):
        read_split(path)
    s["val"] = s["train"][:1]; path.write_text(json.dumps(s))
    with pytest.raises(ValueError, match="overlapping episode"):
        read_split(path)


def test_egodex_converter_carries_sibling_video(tmp_path):
    p = tmp_path / "task" / "0.hdf5"
    n = 40
    cameras = np.repeat(np.eye(4)[None], n, axis=0)
    wrists = np.tile([.1, 0., -1.], (n, 1))
    _write_egodex_hdf5(p, n, wrists, cameras)
    ep = load_episode_hdf5(p)
    assert ep["video_path"] == str(p.with_suffix(".mp4"))
    assert ep["video_fps"] == 30


def test_real_rgb_decoding_alignment_cache_and_missing_video(tmp_path):
    source, _ = fixture(tmp_path)
    cache = tmp_path / "cache"
    path = source / "clip0.json"
    ep = json.loads(path.read_text()); ep["video_path"] = str(source / ep["video_path"])
    with patch.object(tx, "image_encoder", return_value=TinyEncoder()):
        full = tx.image_features(ep, "full", cache)
        data = tx.build_external("egodex", source, cache, episode_ids=[ep["episode_id"]])
        np.testing.assert_allclose(data["f_img"], full[::2][:-1], atol=.001)
        assert np.all(data["f_has"] == 1)
        assert np.ptp(data["f_img"][:, 0]) > .5
        # The index mapping is part of the cache key, not only num_frames.
        small = tx.subsample_episode(ep, 2)
        a = tx.image_features(small, "alignment", cache)
        small["video_frame_indices"] = list(range(1, 48, 2))
        b = tx.image_features(small, "alignment", cache)
        assert not np.allclose(a, b)
        short = dict(ep, num_frames=49, timestamps=(np.arange(49) / 30).tolist())
        with pytest.raises(ValueError, match="Short/undecodable"):
            tx.image_features(short, "short", cache)
        with pytest.raises(ValueError, match="FPS misaligned"):
            tx.image_features(dict(ep, fps=15, timestamps=(np.arange(48) / 15).tolist()), "wrong_fps", cache)
        with pytest.raises(ValueError, match="timestamps are not aligned"):
            tx.image_features(dict(ep, timestamps=(np.arange(48) / 25).tolist()), "bad_times", cache)
    with patch.object(tx, "image_features", side_effect=AssertionError("state-only must not read images")):
        data = tx.build_external("egodex", source, cache, use_img=False)
        assert np.all(data["f_has"] == 0)
    (source / "clip0.avi").unlink()
    with pytest.raises(ValueError, match="Missing RGB.*clip0"):
        tx.build_external("egodex", source, cache, episode_ids=[ep["episode_id"]])


def test_image_encoder_missing_weights_cannot_download(tmp_path):
    import torchvision
    with patch.object(torch.hub, "get_dir", return_value=str(tmp_path)), \
            patch.object(torchvision.models, "resnet18", side_effect=AssertionError("must fail before model/download")):
        with pytest.raises(FileNotFoundError, match="Local ResNet18 weights missing"):
            tx.image_encoder()


def test_public_prepare_only_never_trains(tmp_path):
    source, split = fixture(tmp_path)
    out = tmp_path / "prepare"
    argv = ["train_xhey", "--egodex", str(source), "--split", str(split), "--out", str(out),
            "--prepare-only", "--state-only"]
    with patch.object(sys, "argv", argv), patch.object(tx, "XHEY", side_effect=AssertionError("no training")), \
            patch.object(tx, "build_iphone", side_effect=AssertionError("no probe")):
        assert tx.main() == 0
    data = json.loads((out / "actual_data.json").read_text())
    assert data["status"] == "prepared" and data["sampling"] == []
    assert not (out / "metrics.json").exists()
    assert {ep["source"] for ep in data["episodes"]} == {"egodex"}


def test_public_only_rgb_smoke_and_actual_sampling(tmp_path):
    source, split = fixture(tmp_path)
    out = tmp_path / "run"
    argv = ["train_xhey", "--egodex", str(source), "--split", str(split), "--out", str(out),
            "--steps", "2", "--seeds", "2", "--hidden", "8", "--dropout", "0", "--no-final"]
    changed = []
    original = tx.XHEY.fit
    def audited_fit(self, X, Y, V, Xv, Yv, Vv, **kwargs):
        assert np.all(X[:, -1] == 1) and np.all(Xv[:, -1] == 1)
        before = self.net[0].weight.detach().clone()
        result = original(self, X, Y, V, Xv, Yv, Vv, wd=0, **kwargs)
        changed.append(not torch.equal(before[:, -513:-1], self.net[0].weight[:, -513:-1]))
        return result
    with patch.object(sys, "argv", argv), patch.object(tx, "image_encoder", return_value=TinyEncoder()), \
            patch.object(tx, "build_iphone", side_effect=AssertionError("must never load iPhone")), \
            patch.object(tx.XHEY, "fit", audited_fit):
        assert tx.main() == 0
    assert all(changed)
    manifest = json.loads((out / "actual_data.json").read_text())
    assert manifest["mode"] == "public_only" and manifest["status"] == "complete"
    assert manifest["partitions"]["test"]["units"] == 2  # 3 clips, two sessions, not 6 seed*clips
    for run in manifest["sampling"]:
        eg = run["sources"]["egodex"]
        assert eg["sampled_draws"] == 2 * 256 == eg["sampled_image_draws"]
        assert eg["effective_weight"] == 1
        assert 0 < eg["unique_sampled_windows"] <= eg["available_windows"]
    results = json.loads((out / "metrics.json").read_text())["results"]
    assert set(results) == {"zero", "constant_velocity", "egodex"}
    assert results["egodex"]["n_eval_units"] == 2
    assert results["egodex"]["decision_eligible"] is False
    assert results["egodex"]["ci_method"] == "episode_session_cluster_bootstrap"
    assert (out / "window_errors.npz").is_file()
    assert not (out / "xhey.pt").exists()


def test_cli_state_only_without_iphone_or_weights_and_frozen_rejection(tmp_path):
    source, split = fixture(tmp_path)
    cmd = [sys.executable, str(ROOT / "scripts/train_xhey.py"), "--egodex", str(source), "--split", str(split),
           "--out", str(tmp_path / "state"), "--state-only", "--steps", "1", "--hidden", "8", "--seeds", "1", "--no-final"]
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    r = subprocess.run(cmd, text=True, capture_output=True, env=env)
    assert r.returncode == 0, r.stderr
    assert '"image_windows": 0' in r.stdout
    assert "actual_sampling" in r.stdout
    r = subprocess.run(cmd + ["--mix", "iphone+egodex"], text=True, capture_output=True, env=env)
    assert r.returncode != 0 and "probe/eval-only" in r.stderr
    r = subprocess.run(cmd, text=True, capture_output=True, env=env)
    assert r.returncode != 0 and "fresh --out" in r.stderr
    with pytest.raises(ValueError, match="probe/eval-only"):
        tx.train_mix("iphone", None, [], {}, 0, 1, 0)


def test_cluster_bootstrap_does_not_count_overlapping_windows_hands_or_seeds():
    errors = np.array([[-2., -2.], [1., 1.], [3., 3.]])
    groups = np.array(["s0", "s1", "s2"])
    a = cluster_summary(errors, groups)
    b = cluster_summary(np.repeat(errors, 100, axis=0), np.repeat(groups, 100))
    assert a["ci95"] == b["ci95"] and a["n_units"] == b["n_units"] == 3
    assert a["ci95"][0] < 0 < a["ci95"][1]
    assert cluster_summary(errors, ["same"] * 3)["ci95"] is None
    assert cluster_summary(np.full((3, 2), np.nan), groups)["n_units"] == 0
    # Paired identical predictions have exactly zero difference under every draw.
    assert cluster_summary(errors - errors, groups)["ci95"] == [0., 0.]
    row = {"status": "ok", "delta_moving_ci95": [-.02, -.01], "delta_moving_cm": -.015}
    assert decision_status(row) == "evidence_insufficient"
    row.update(decision_eligible=True, ci_method="episode_session_cluster_bootstrap",
               n_moving_eval_units=3, min_eval_units=10, min_useful_improvement_cm=.1)
    assert decision_status(row) == "evidence_insufficient"


def test_constant_velocity_uses_only_observed_actions():
    d = {"episode": np.array([0, 0, 0, 1]), "action": np.tile(tx.IDENT, (4, 1)), "valid": np.ones((4, 2))}
    d["action"][:, 0] = [1, 2, 999, 888]
    pred = constant_velocity(tx, d, np.array([0, 2, 3]))
    np.testing.assert_array_equal(pred[:, :, 0], np.repeat([[0], [2], [0]], tx.H, axis=1))


def test_pause_and_unavailable_best_row_cannot_reintroduce_xphi():
    assert ai.load_queue()["launch_paused"]
    assert ai.propose() == []
    with patch.object(ai, "kaggle", side_effect=AssertionError("no remote calls")):
        with pytest.raises(ValueError, match="launch_paused"):
            ai.launch([dict(ai.DEFAULT)], 999)
        with pytest.raises(ValueError, match="launch_paused"):
            ai.submit_queue_item({"id": "C14"})
    bad = {"status": "ok", "delta_moving_cm": -100, "proposer_eligible": True,
           "config": dict(ai.DEFAULT, mix="iphone+xphi+xhey_rgb+aug")}
    with patch.object(ai, "load_queue", return_value={}):
        assert all(ai.available_config(c) and "xphi" not in c["mix"] for c in ai.propose(50, ledger=[bad]))
        with pytest.raises(ValueError, match="Unavailable"):
            ai.launch([bad["config"]], 999)
    # Run only the generated preflight, not extraction, downloads or Kaggle jobs.
    template = ai.KERNEL_TEMPLATE % {"cands": json.dumps([bad["config"]]), "tgz": ""}
    preflight = template.split("W = '/kaggle/working'")[0]
    with pytest.raises(RuntimeError, match="Unavailable sources"):
        exec(preflight, {})
    good_template = ai.KERNEL_TEMPLATE % {"cands": json.dumps([ai.DEFAULT]), "tgz": ""}
    namespace = {}; exec(good_template.split("W = '/kaggle/working'")[0], namespace)
    with patch("glob.glob", return_value=[]), pytest.raises(RuntimeError, match="Missing requested Kaggle source"):
        namespace["required"]("xphi")
    with pytest.raises(subprocess.CalledProcessError):
        namespace["sh"]("false; echo MUST_NOT_MASK_FAILURE")


def kaggle_training_drivers():
    paths = list((ROOT / "kaggle").rglob("*.py")) + list((ROOT / "scripts").glob("kaggle*.py"))
    paths += list((ROOT / "scripts/kaggle").glob("*.py"))
    return [p for p in paths if "scripts/train_xhey.py" in p.read_text()]


@pytest.mark.parametrize("path", kaggle_training_drivers(), ids=lambda p: str(p.relative_to(ROOT)))
def test_kaggle_training_failure_cannot_be_hidden_by_tail(path):
    tree = ast.parse(path.read_text())
    sh = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "sh")
    # Only define the helper. Never execute Kaggle top-level setup or jobs.
    namespace = {"subprocess": subprocess}
    exec(compile(ast.Module(body=[sh], type_ignores=[]), str(path), "exec"), namespace)
    with pytest.raises(subprocess.CalledProcessError):
        namespace["sh"]("false | tail -1 # scripts/train_xhey.py\necho MUST_NOT_MASK_FAILURE")


def test_append_only_review_overlay_handles_future_rows(tmp_path):
    path = tmp_path / "ledger.jsonl"
    note = {"id": "audit", "record_type": "review_note", "target_ids": ["future"],
            "annotations": {"evidence_status": "evidence_insufficient", "proposer_eligible": False}}
    metric = {"id": "future", "status": "ok", "overall_ade_cm": 9.75}
    path.write_text(json.dumps(note) + "\n" + json.dumps(metric) + "\n")
    before = path.read_bytes()
    rows = reviewed_rows(path)
    assert len(rows) == 1 and rows[0]["overall_ade_cm"] == 9.75
    assert not rows[0]["proposer_eligible"] and rows[0]["review_notes"] == ["audit"]
    assert path.read_bytes() == before
    committed = {r["id"]: r for r in ai.load_ledger()}
    for rid in ("C10", "C11", "C12", "C13", "C14", "C16", "iter003_c2", "iter004_c0", "iter004_c2"):
        assert committed[rid]["evidence_status"] == "evidence_insufficient"
        assert committed[rid]["proposer_eligible"] is False
