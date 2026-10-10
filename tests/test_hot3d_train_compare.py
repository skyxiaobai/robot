import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import hot3d_train_compare as h  # noqa: E402


def _hand(t):
    j = np.zeros((21, 3)); j[5] = [0.08, 0, 0]; j[17] = [0.02, 0.07, 0]
    return (j + [0.01 * t, 0, 0.5]).tolist()


def test_plain_episode_and_linear_recovers_constant_motion():
    n = 40
    ep = {"num_frames": n, "camera_poses": [np.eye(4).tolist()] * n}
    plain = h._plain_episode(ep, {"left": [_hand(t) for t in range(n)], "right": [None] * n})
    d = h.to_arrays([plain])
    assert d["state"].shape == (n - 1, 140)
    assert d["valid"][:, 0].all() and not d["valid"][:, 1].any()
    assert np.allclose(d["action"][:, 0], 0.01, atol=1e-6)
    starts = h.chunks(d, [0]); X, Y, V = h.xy(d, starts)
    pred = h.fit_linear(X, Y, V)(X)
    assert np.allclose(pred[:, :, 0], 0.01, atol=1e-3)
    ca = h.chunk_ade(d, starts, pred)
    assert np.nanmax(ca[:, 0]) < 0.2 and np.isnan(ca[:, 1]).all()


def test_bootstrap_ci_brackets_mean():
    x = np.arange(100, dtype=float)
    lo, hi = h.bootstrap_ci(x)
    assert lo < x.mean() < hi
