"""C18: compare Flora-TempSeg action intervals with the current motion-threshold(+QC) segment selection.
Timeline at 10 Hz per clip, restricted to the span that has hand labels."""
import json, glob, os, sys, argparse
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--flora', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--move-cm', type=float, default=3.0); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
HZ = 10

def wrist_series(ep):
    """per-hand wrist (n,3) metres in camera frame, nan where invalid; plus fps"""
    n = ep['num_frames']; out = []
    for side in ('left', 'right'):
        h = ep['hands'][side]; W = np.full((n, 3), np.nan)
        for i in range(n):
            p = h['wrist_pose'][i]
            if h['valid'][i] and p and p[0] is not None: W[i] = p[:3]
        out.append(W)
    return out, float(ep['fps'])

def frame_metrics(Ws, fps):
    """resampled to 10 Hz: detect, speed (1 s displacement cm, max over hands), jitter (qc-style 2nd diff cm @15fps norm)"""
    n = len(Ws[0]); T = n / fps; m = int(np.floor(T * HZ)); t = np.arange(m) / HZ
    idx = np.minimum((t * fps).round().astype(int), n - 1); lag = int(round(fps))
    det = np.zeros(m, bool); spd = np.full(m, np.nan); jit = np.full(m, np.nan)
    for W in Ws:
        ok = np.isfinite(W[:, 0]); det |= ok[idx]
        j = np.minimum(idx + lag, n - 1); d = np.linalg.norm(W[j] - W[idx], axis=1) * 100
        spd = np.fmax(spd, d)
        i1 = np.clip(idx, 1, n - 2); r = np.linalg.norm(W[i1 + 1] - 2 * W[i1] + W[i1 - 1], axis=1) * (15.0 / fps) ** 2 * 100
        jit = np.fmax(jit, r)
    return dict(det=det, spd=spd, jit=jit, m=m)

def mask_from_intervals(iv, m):
    k = np.zeros(m, bool)
    for s, e in iv:
        k[max(0, int(np.floor(s * HZ))):min(m, int(np.ceil(e * HZ)))] = True
    return k

def runs(k):
    d = np.diff(np.r_[0, k.astype(int), 0]); return int((d == 1).sum())

def flora_intervals(src, key):
    p = f'{a.flora}/{src}__{key}.json'
    if not os.path.exists(p): return None, None
    r = json.load(open(p)); iv = [(s['start_s'], s['end_s']) for c in r['chunks'] for s in c.get('segments', [])]
    errs = sum('error' in c for c in r['chunks'])
    return iv, dict(errs=errs, wall=r.get('wall_s'), dur=r.get('duration_s'), nseg=len(iv),
                    infer_s=sum(c.get('infer_s', 0) for c in r['chunks']), windows=sum(c.get('windows', 0) for c in r['chunks']))

rows = []
# ---- xPhi: current = QC segment spans (3 s windows passing all rules, incl. move_frac); motion-only = windows not failing 手几乎不动
seg = json.load(open('/workspace/c14/qc_xphi/segments.json'))
for c in seg['clips']:
    key = c['clip'].replace('tmp__v__', '').replace('_mp4', '')
    iv, info = flora_intervals('xphi', key)
    if iv is None: continue
    ep = json.load(open(f"/workspace/c14/xphi_cam/{c['clip']}.json")); Ws, fps = wrist_series(ep); fm = frame_metrics(Ws, fps); m = fm['m']
    wf = c['window_frames']; qc = np.zeros(m, bool); mot = np.zeros(m, bool)
    for w in c['windows']:
        s, e = w['start'] / fps, w['end'] / fps
        sl = slice(int(s * HZ), min(m, int(e * HZ)))
        if not w['fails']: qc[sl] = True
        if '手几乎不动' not in w['fails']: mot[sl] = True
    cur = mask_from_intervals([(s / fps, e / fps) for s, e in c['spans']], m)
    rows.append(dict(src='xphi', clip=key, m=m, flora=mask_from_intervals(iv, m), cur=cur, motion=mot, qcpass=qc, info=info, **{k: fm[k] for k in ('det', 'spd', 'jit')}))
# ---- Open-AoE nano: current = 20 s segments with 1 s wrist-displacement median >= 3 cm (convert_openaoe default)
eps = {}
for f in glob.glob('/workspace/c18/aoe_all/*.json'):
    if f.endswith('convert_summary.json'): continue
    b = os.path.basename(f)[:-5]; name, s0 = b.rsplit('_', 1); eps.setdefault(name, []).append((int(s0), f))
for name, lst in sorted(eps.items()):
    iv, info = flora_intervals('aoe', name)
    if iv is None: continue
    lst.sort(); last = lst[-1]; e_last = json.load(open(last[1])); n_tot = last[0] + e_last['num_frames']
    W = [np.full((n_tot, 3), np.nan), np.full((n_tot, 3), np.nan)]; curiv = []
    for s0, f in lst:
        e = json.load(open(f)); Ws, fps = wrist_series(e)
        for k in range(2): W[k][s0:s0 + e['num_frames']] = Ws[k]
        if e['hand_pose']['move_median_cm_per_s'] >= a.move_cm: curiv.append((s0 / 15.0, (s0 + e['num_frames']) / 15.0))
    fm = frame_metrics(W, 15.0); m = fm['m']; cur = mask_from_intervals(curiv, m)
    rows.append(dict(src='aoe', clip=name, m=m, flora=mask_from_intervals(iv, m), cur=cur, motion=cur.copy(), qcpass=np.ones(m, bool), info=info,
                     **{k: fm[k] for k in ('det', 'spd', 'jit')}))

def stats(sel_name, src):
    R = [r for r in rows if r['src'] == src]
    if not R: return None
    def cat(k): return np.concatenate([r[k] for r in R])
    F, C, D, S, J, Q = cat('flora'), cat('cur'), cat('det'), cat('spd'), cat('jit'), cat('qcpass')
    sel = {'flora': F, 'current': C, 'flora_and_qc': F & Q, 'flora_only': F & ~C, 'current_only': C & ~F, 'both': F & C, 'neither': ~F & ~C, 'all': np.ones_like(F)}[sel_name]
    def med(x): x = x[np.isfinite(x)]; return float(np.median(x)) if len(x) else None
    nseg = sum(runs(r['flora'] if sel_name == 'flora' else r['cur'] if sel_name == 'current' else
                    {'flora_and_qc': r['flora'] & r['qcpass'], 'flora_only': r['flora'] & ~r['cur'], 'current_only': r['cur'] & ~r['flora'],
                     'both': r['flora'] & r['cur'], 'neither': ~r['flora'] & ~r['cur'], 'all': np.ones(r['m'], bool)}[sel_name]) for r in R)
    return dict(src=src, sel=sel_name, hours=round(float(sel.sum()) / HZ / 3600, 3), frac=round(float(sel.mean()), 3), runs=nseg,
                detect=round(float(D[sel].mean()), 3) if sel.any() else None,
                speed_med_cm=med(S[sel]), moving_frac=round(float((S[sel] >= a.move_cm).mean()), 3) if sel.any() else None,
                jitter_med_cm=med(J[sel]), qc_pass_frac=round(float(Q[sel].mean()), 3) if sel.any() else None)
summary = {}
for src in ('xphi', 'aoe'):
    R = [r for r in rows if r['src'] == src]
    if not R: continue
    F = np.concatenate([r['flora'] for r in R]); C = np.concatenate([r['cur'] for r in R])
    summary[src] = dict(clips=len(R), labelled_hours=round(sum(r['m'] for r in R) / HZ / 3600, 3),
                        flora_segments=sum(r['info']['nseg'] for r in R), flora_chunk_errors=sum(r['info']['errs'] for r in R),
                        iou=round(float((F & C).sum() / max(1, (F | C).sum())), 3),
                        flora_recall_of_current=round(float((F & C).sum() / max(1, C.sum())), 3),
                        current_recall_of_flora=round(float((F & C).sum() / max(1, F.sum())), 3),
                        sec_per_video_min=round(sum(r['info']['infer_s'] for r in R) / max(1e-9, sum(r['info']['dur'] or 0 for r in R)) * 60, 1),
                        windows=sum(r['info']['windows'] for r in R),
                        table=[stats(s, src) for s in ('all', 'current', 'flora', 'flora_and_qc', 'both', 'flora_only', 'current_only', 'neither')])
json.dump(summary, open(f'{a.out}/c18_summary.json', 'w'), indent=1)
np.savez_compressed(f'{a.out}/c18_masks.npz', **{f"{r['src']}__{r['clip']}__{k}": r[k] for r in rows for k in ('flora', 'cur', 'spd', 'qcpass')})
for src, s in summary.items():
    print(src, {k: v for k, v in s.items() if k != 'table'})
    for t in s['table']: print('  ', t)
