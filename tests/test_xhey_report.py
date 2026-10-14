"""xhey_report：合成 metrics（含曲线/每步/轨迹）能出全套中文图。"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import xhey_report as rep  # noqa: E402


def _res(a):
    return {"mean": {"cum_pos_cm@1": 1.0, "cum_pos_cm@16": 10.0}, "n_seeds": 3, "ade_cm_pooled": a, "ade_ci95": [a - .1, a + .1],
            "ade_minus_zero_cm": a - 11, "ade_minus_zero_ci95": [a - 11.1, a - 10.9], "per_clip_ade_cm": {"a": a, "b": a + 1},
            "moving_ade_cm": a + 3, "moving_minus_zero_cm": a - 11, "moving_minus_zero_ci95": [a - 11.1, a - 10.9],
            "horizon_ade_cm": list(np.linspace(1, a, 16)), "horizon_ade_cm_seed_std": [0.1] * 16}


class Report(unittest.TestCase):
    def test_full_run_and_summary(self):
        traj = [{"hand": 1, "chunk": 3, "gt_cm": np.cumsum(np.ones((16, 3)), 0).tolist(), "pred_cm": np.zeros((16, 3)).tolist()}]
        m = {"clips": ["a", "b"], "results": {"zero": _res(11.0), "linear": _res(12.0), "iphone+aug": _res(10.9)},
             "curves": [{"mix": "iphone+aug", "seed": 0, "fold": "a", "finetune": [{"step": 100, "loss": 1.0, "val_ade_cm": 9.0},
                                                                            {"step": 200, "loss": .5, "val_ade_cm": 8.0}, {"shrink": 1.0}]}],
             "examples": {"iphone+aug": {"clip": "a", "chunks": traj}, "zero": {"clip": "a", "chunks": traj}}}
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); (d / "metrics.json").write_text(json.dumps(m))
            made = rep.run_report(d / "metrics.json", "t", out_root=d / "rep")
            self.assertEqual({p.name for p in made}, {"ade.png", "per_clip.png", "horizon.png", "loss_curves.png", "trajectory.png"})
            (d / "l.jsonl").write_text(json.dumps({"id": "x", "status": "ok", "overall_ade_cm": 11, "delta_overall_cm": .1,
                                                   "moving_ade_cm": 14, "delta_moving_cm": -.03, "delta_moving_ci95": [-.04, -.02]}) + "\n")
            out = rep.summary(d / "l.jsonl", d / "rep")
            self.assertTrue(all(p.exists() for p in out))


class TrainHelpers(unittest.TestCase):
    def test_horizon_and_examples(self):
        try:
            import train_xhey as tx
        except Exception as e:  # noqa: BLE001
            self.skipTest("train_xhey 依赖缺失：%s" % e)
        rng = np.random.default_rng(0)
        Y = rng.normal(0, .01, (5, 16, 14)); V = np.ones((5, 16, 2)); P = np.zeros_like(Y)
        num, den = tx.horizon_parts(P, Y, V)
        self.assertEqual(num.shape, (16,)); self.assertTrue((den == 10).all())
        ex = tx.trajectory_examples(P, Y, V, n=2)
        self.assertEqual(len(ex), 2); self.assertEqual(len(ex[0]["gt_cm"]), 16)
