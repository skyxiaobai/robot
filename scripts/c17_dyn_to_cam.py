"""Dyn-HaMR joints_world.npz (joints (B,T,21,3) world, cam_R (T,3,3)|(B,T,3,3), cam_t, is_right (B,)) -> joints_cam.npz {left_wrist,right_wrist} (T,3) OpenCV camera, metres, NaN where missing.
usage: c17_dyn_to_cam.py KOUT  (processes KOUT/dynhamr/*/joints_world.npz)"""
import sys, glob, os, numpy as np
def convert(p):
    d = np.load(p, allow_pickle=True); J = np.asarray(d['joints'], float)
    if J.ndim == 3: J = J[None]
    B, T = J.shape[:2]; R = np.asarray(d['cam_R'], float); t = np.asarray(d['cam_t'], float)
    if R.ndim == 3: R = np.broadcast_to(R, (B,) + R.shape)
    if t.ndim == 2: t = np.broadcast_to(t, (B,) + t.shape)
    isr = np.asarray(d['is_right']).reshape(-1) if d['is_right'].size else np.ones(B)
    out = {'left_wrist': np.full((T, 3), np.nan), 'right_wrist': np.full((T, 3), np.nan)}
    for b in range(B):
        w = np.einsum('tij,tj->ti', R[b, :T], J[b, :, 0]) + t[b, :T]
        key = 'right_wrist' if isr[b] > 0.5 else 'left_wrist'
        m = np.isfinite(w).all(1) & ~np.isfinite(out[key]).all(1); out[key][m] = w[m]
    np.savez(os.path.join(os.path.dirname(p), 'joints_cam.npz'), **out)
    return {k: int(np.isfinite(v[:, 0]).sum()) for k, v in out.items()}, np.nanmedian(np.concatenate(list(out.values())), 0)
if __name__ == '__main__':
    for p in sorted(glob.glob(f'{sys.argv[1]}/dynhamr/*/joints_world.npz')): print(p, *convert(p))
