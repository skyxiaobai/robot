import sys, unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


class WristNoise(unittest.TestCase):
    def test_static_series_has_small_true_disp_but_measurable_drift(self):
        from wrist_noise import series_metrics
        rng = np.random.default_rng(0)
        W = np.tile([0.1, 0.2, 0.5], (300, 1)) + rng.normal(0, 0.006, (300, 3))  # ~0.6 cm/axis noise
        m = series_metrics(W, np.ones(300, bool), 15.0)
        self.assertLess(m["disp_1s_cm"]["median"], 2.0)
        self.assertGreater(m["windows"]["static_drift_rms_cm_median"], 0.5)
        self.assertLess(m["windows"]["static_drift_rms_cm_median"], 2.0)

    def test_depth_correction_replaces_z_on_ray(self):
        from wrist_noise import correct_wrist_depth
        W = np.array([[0.1, 0.1, 1.0]] * 5, float)
        depth = np.full((5, 20, 20), 0.8, np.float32); conf = np.full((5, 20, 20), 2, np.uint8)
        K = np.array([[100., 0, 10], [0, 100., 10], [0, 0, 1.]])
        out, znew, used = correct_wrist_depth(W, np.ones(5, bool), K, depth, conf, offset_m=0.0)
        self.assertTrue(used.all()); self.assertTrue(np.allclose(znew, 0.8))
        self.assertTrue(np.allclose(out[:, 2], 0.8)); self.assertTrue(np.allclose(out[:, 0], 0.08))

    def test_cum_weight_flag_default_preserves_old_loss(self):
        src = (ROOT / "scripts" / "train_xhey.py").read_text(encoding="utf-8")
        self.assertIn('"--cum-weight", type=float, default=0.1', src)
        self.assertIn("self.cum_w = CUM_W", src)


if __name__ == "__main__":
    unittest.main()
