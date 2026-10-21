"""C17 metrics (1) wrist jitter/static drift (scripts/wrist_noise.py, C15 definitions) and
(2) absolute wrist depth error vs iPhone LiDAR, per label method on the same 20 s windows."""
import json, glob, os, sys, argparse
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--kout', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--repo', default='/workspace/c17/wt'); a = ap.parse_args()
sys.path.insert(0, a.repo + '/scripts')
import wrist_noise as WN
from egodata.hand_fix import SIDES
os.makedirs(a.out, exist_ok=True)
wins = json.load(open(f'{a.kout}/windows.json')); wins = wins['windows'] if isinstance(wins, dict) else wins
SES = '/workspace/xhey_kaggle/ds'
LIDAR_EP = {os.path.basename(p)[:-5]: p for p in glob.glob('/workspace/c15/data/wilor_labels*/run_s2/episodes/*.json') if os.path.getsize(p) > 0}
LIDAR_EP.update({os.path.basename(p)[:-5]: p for p in glob.glob(f'{a.kout}/lidar_eps/*.json')})
XR = '/workspace/c15/data/wilor_labels/xhey_rgb_unified'
_cache = {}
KMONO = {}
def full_ep(path):
    if path not in _cache:
        _cache.clear(); e = json.load(open(path)); _cache[path] = WN.load_wrist(e)
    return _cache[path]
def methods_for(w):
    """-> {method: {side: (ok, W_cam)}} for this window, camera (OpenCV) frame, metres"""
    out = {}; s, e = w['start'], w['end']
    if w['kind'] == 'depth':
        if w['clip'] in LIDAR_EP:
            ep, hw = full_ep(LIDAR_EP[w['clip']]); out['wilor_lidar (current iPhone)'] = {k: (ok[s:e], W[s:e]) for k, (ok, W) in hw.items()}
        js = glob.glob(f"{a.kout}/wilor_mono/{w['name']}/*.json")
        js = [j for j in js if os.path.basename(j) not in ('index.json', 'manifest.json', 'convert_summary.json')]
        if js:
            ej = json.load(open(js[0])); KMONO[w['clip']] = np.asarray(ej['camera_intrinsic'], float)
            _, hw = WN.load_wrist(ej); out['wilor_mono'] = hw
    else:
        f = glob.glob(f"{XR}/{w['clip']}_tmp__*.json")
        if f:
            ep, hw = full_ep(f[0]); out['wilor_mono'] = {k: (ok[s:e], W[s:e]) for k, (ok, W) in hw.items()}
    dj = f"{a.kout}/dynhamr/{w['name']}/joints_cam.npz"
    if os.path.exists(dj):
        d = np.load(dj); hw = {}
        for side in SIDES:
            W = d[side + '_wrist']; ok = np.isfinite(W[:, 0]); hw[side] = (ok, W)
        out['dyn_hamr'] = hw
    return out
def lidar_z(w, W, ok, K):
    """median LiDAR depth (m) in a 5x5 patch at each projected wrist; nan where unavailable"""
    d = np.load(f"{SES}/depth/{w['clip']}/depth.npz", mmap_mode='r'); depth = d['depth'][w['start']:w['end']]; conf = d['conf'][w['start']:w['end']]
    sx = depth.shape[2] / 1920.0; fx, fy, cx, cy = K[0, 0] * sx, K[1, 1] * sx, K[0, 2] * sx, K[1, 2] * sx
    zl = np.full(len(W), np.nan)
    for i in range(len(W)):
        if not ok[i] or not np.isfinite(W[i]).all() or W[i, 2] <= 0.05: continue
        u = int(round(fx * W[i, 0] / W[i, 2] + cx)); v = int(round(fy * W[i, 1] / W[i, 2] + cy))
        if not (0 <= u < depth.shape[2] and 0 <= v < depth.shape[1]): continue
        p = np.asarray(depth[i, max(0, v - 2):v + 3, max(0, u - 2):u + 3], float); cf = conf[i, max(0, v - 2):v + 3, max(0, u - 2):u + 3]
        vals = p[(p > 0.2) & (p < 2.5) & (cf >= 1)]
        if len(vals) >= 3: zl[i] = float(np.median(vals))
    return zl
SCALES = []
res = {}; depth_rows = {}
for w in wins:
    M = methods_for(w)
    if w['kind'] == 'depth' and 'wilor_mono' in M and w['clip'] in KMONO:  # one scalar per window+hand: median(LiDAR z / mono z)
        hs = {}
        for side, (ok, W) in M['wilor_mono'].items():
            zl = lidar_z(w, W, ok, KMONO[w['clip']]); r = zl / W[:, 2]; r = r[np.isfinite(r)]
            if len(r) >= 30:
                sc = float(np.median(r)); hs[side] = (ok, W * sc); SCALES.append(dict(window=w['name'], side=side, scale=sc, n=int(len(r))))
        if hs: M['wilor_mono x LiDAR scale'] = hs
    dep = None
    if w['kind'] == 'depth':
        dep = True
    for m, hw in M.items():
        for side, (ok, W) in hw.items():
            if ok.sum() < 30: continue
            r = WN.series_metrics(W, ok, w['fps']); res.setdefault(m, []).append(dict(r, window=w['name'], side=side, kind=w['kind']))
            if dep is not None:
                depth_rows.setdefault(m, []).append((w, side, ok, W))
# ---- (2) depth error vs LiDAR
dres = {}
Kc = {}
for m, rows in depth_rows.items():
    dz_all = []; e3_all = []; per = []
    for w, side, ok, W in rows:
        if w['clip'] not in Kc:
            Kc[w['clip']] = np.asarray(json.load(open(LIDAR_EP[w['clip']]))['camera_intrinsic'], float) if w['clip'] in LIDAR_EP else KMONO.get(w['clip'])
        K = Kc[w['clip']]
        if K is None: continue
        d = np.load(f"{SES}/depth/{w['clip']}/depth.npz", mmap_mode='r'); depth = d['depth'][w['start']:w['end']]; conf = d['conf'][w['start']:w['end']]
        sx = depth.shape[2] / 1920.0; Kd = K.copy(); Kd[0] *= sx; Kd[1] *= sx
        fx, fy, cx, cy = Kd[0, 0], Kd[1, 1], Kd[0, 2], Kd[1, 2]; dz = []; e3 = []
        for i in range(len(W)):
            if not ok[i] or not np.isfinite(W[i]).all() or W[i, 2] <= 0.05: continue
            u = int(round(fx * W[i, 0] / W[i, 2] + cx)); v = int(round(fy * W[i, 1] / W[i, 2] + cy))
            if not (0 <= u < depth.shape[2] and 0 <= v < depth.shape[1]): continue
            p = np.asarray(depth[i, max(0, v - 2):v + 3, max(0, u - 2):u + 3], float); cf = conf[i, max(0, v - 2):v + 3, max(0, u - 2):u + 3]
            vals = p[(p > 0.2) & (p < 2.5) & (cf >= 1)]
            if len(vals) < 3: continue
            zl = float(np.median(vals)); dz.append(W[i, 2] - zl); e3.append(abs(W[i, 2] - zl) * np.linalg.norm(W[i]) / W[i, 2])
        dz = np.array(dz) * 100; e3 = np.array(e3) * 100
        dz_all.append(dz); e3_all.append(e3); per.append(dict(window=w['name'], side=side, n=len(dz), dz_med_cm=float(np.median(dz)) if len(dz) else None,
                                                             absdz_med_cm=float(np.median(np.abs(dz))) if len(dz) else None))
    dz = np.concatenate(dz_all) if dz_all else np.array([]); e3 = np.concatenate(e3_all) if e3_all else np.array([])
    dres[m] = dict(n=int(len(dz)), signed_dz_med_cm=float(np.median(dz)) if len(dz) else None, abs_dz_med_cm=float(np.median(np.abs(dz))) if len(dz) else None,
                   abs_dz_p90_cm=float(np.percentile(np.abs(dz), 90)) if len(dz) else None, ray3d_err_med_cm=float(np.median(e3)) if len(e3) else None,
                   within_2cm=float((np.abs(dz) <= 2).mean()) if len(dz) else None, per_window=per)
    np.save(f"{a.out}/dz_{m.split()[0]}{'_scaled' if 'scale' in m else ''}.npy", dz)
# ---- (1) summaries
summ = {}
for m, rows in res.items():
    for kind in ('depth', 'nodepth', 'all'):
        R = [dict(r) for r in rows if kind == 'all' or r['kind'] == kind]
        if not R: continue
        meta = [(r['window'], r['side']) for r in R]
        s, jump, disp = WN.summarise(R)
        s.pop('hands'); s['hand_series'] = len(meta); summ[f'{m}|{kind}'] = s
        if kind != 'all': np.save(f"{a.out}/jump_{m.split()[0]}{'_scaled' if 'scale' in m else ''}_{kind}.npy", jump)
json.dump(dict(noise=summ, depth=dres, windows=wins, mono_scales=SCALES), open(f'{a.out}/c17_metrics.json', 'w'), indent=1, default=float)
for k, v in summ.items(): print('NOISE', k, {kk: (vv['median'] if isinstance(vv, dict) and 'median' in vv else vv) for kk, vv in v.items()})
for k, v in dres.items(): print('DEPTH', k, {kk: vv for kk, vv in v.items() if kk != 'per_window'})
sc = np.array([x['scale'] for x in SCALES]); print('MONO_SCALE n', len(sc), 'median', np.median(sc) if len(sc) else None, 'range', (sc.min(), sc.max()) if len(sc) else None)
