"""Cluster bootstrap for paired XHEY errors; windows/hands/seeds are not units."""
import numpy as np


def cluster_summary(values, groups, n=2000, seed=0):
    """Resample complete episodes/sessions, keeping all windows and both hands.

    The estimand remains the pooled finite window-hand mean (ratio of cluster
    sums/counts), for compatibility with ADE point estimates. Group sizes may
    differ. Seeds must be averaged on the same windows *before* calling this.
    For comparisons pass paired differences with a shared finite/moving mask.
    """
    x = np.asarray(values, float)
    g = np.asarray(groups)
    if g.ndim != 1 or len(g) != len(x):
        raise ValueError("groups must identify every window")
    g = np.broadcast_to(g.reshape((-1,) + (1,) * (x.ndim - 1)), x.shape)
    valid = np.isfinite(x)
    ids = np.unique(g[valid])
    sums = np.array([x[valid & (g == unit)].sum() for unit in ids])
    counts = np.array([np.count_nonzero(valid & (g == unit)) for unit in ids])
    result = {"mean": float(sums.sum() / counts.sum()) if len(ids) else None,
              "n_units": len(ids), "n_window_hands": int(counts.sum()), "ci95": None}
    if len(ids) >= 2:
        rng = np.random.default_rng(seed)
        # Bound temporary memory for large public splits.
        draws = []
        for start in range(0, n, 128):
            ix = rng.integers(0, len(ids), (min(128, n - start), len(ids)))
            draws.extend(sums[ix].sum(1) / counts[ix].sum(1))
        result["ci95"] = np.percentile(draws, [2.5, 97.5]).tolist()
    return result
