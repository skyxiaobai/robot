"""C17 charts from c17_metrics.json + npy arrays (+ steps.jsonl for runtime). usage: c17_charts.py RESULTS_DIR OUT_DIR KOUT_DIR [KOUT_DIR ...] (one per kernel version, in order)"""
import json, sys, os, glob
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R, O = sys.argv[1:3]; KOS = sys.argv[3:]; os.makedirs(O, exist_ok=True)
M = json.load(open(f'{R}/c17_metrics.json'))
C15 = {'C15 iphone (full clips)': dict(jump=1.38, jitter=1.80, static=1.06), 'C15 xhey_rgb (full clips)': dict(jump=1.18, jitter=1.42, static=0.99)}
def med(v): return v['median'] if isinstance(v, dict) and 'median' in v else v
rows = {}
for k, v in M['noise'].items():
    m, kind = k.split('|')
    if kind == 'all': continue
    rows[f'{m} [{kind}]'] = dict(jump=med(v['jump_cm']), jitter=med(v['jitter_cm_median_of_hands']), static=med(v['static_drift_rms_cm_median_of_hands']), n=v['hand_series'])
for k, v in C15.items(): rows[k] = dict(v, n=None)
# 1 noise bars
fig, ax = plt.subplots(figsize=(12, 5)); names = list(rows); x = np.arange(len(names)); wd = 0.27
for i, (key, lab) in enumerate([('jump', 'frame jump (median)'), ('jitter', 'jitter (median of hands)'), ('static', 'static drift RMS')]):
    b = ax.bar(x + (i - 1) * wd, [rows[n][key] for n in names], wd, label=lab)
    ax.bar_label(b, fmt='%.2f', fontsize=7)
ax.set_xticks(x); ax.set_xticklabels([n + (f"\n{rows[n]['n']} hand series" if rows[n]['n'] else '') for n in names], fontsize=7, rotation=15)
ax.set_ylabel('cm'); ax.set_title('C17 wrist noise per label method (20 s windows; C15 full-clip reference)'); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f'{O}/c17_noise.png', dpi=130); plt.close(fig)
# 2 depth error CDF
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for f, lab in [('dz_wilor_lidar.npy', 'WiLoR+LiDAR (current iPhone; circular)'), ('dz_wilor_mono.npy', 'WiLoR mono (raw)'),
               ('dz_wilor_mono_scaled.npy', 'WiLoR mono x per-window LiDAR scale'), ('dz_dyn_hamr.npy', 'Dyn-HaMR (ARKit cams)')]:
    p = f'{R}/{f}'
    if not os.path.exists(p): continue
    d = np.load(p); a = np.sort(np.clip(np.abs(d), 0.1, None)); ax[0].plot(a, np.arange(1, len(a) + 1) / len(a), label=f'{lab} (n={len(a)}, med {np.median(a):.1f})')
    ax[1].hist(np.clip(d, -100, 30), bins=130, histtype='step', label=lab)
ax[0].set_xscale('log'); ax[0].set_xlim(0.1, 200); ax[0].set_xlabel('|wrist z - LiDAR z| (cm, log)'); ax[0].set_ylabel('CDF'); ax[0].axvline(2, ls=':', c='k'); ax[0].legend(fontsize=7)
ax[1].set_xlabel('signed z error (cm, clipped [-100, 30])'); ax[1].legend(fontsize=7); fig.suptitle('C17 absolute wrist depth vs iPhone LiDAR')
fig.tight_layout(); fig.savefig(f'{O}/c17_depth_error.png', dpi=130); plt.close(fig)
# 3 mono scale
S = M.get('mono_scales', [])
if S:
    fig, ax = plt.subplots(figsize=(9, 3.8)); lab = [f"{s['window'][5:]}\n{s['side']}" for s in S]
    ax.bar(range(len(S)), [s['scale'] for s in S], color=['C0' if s['side'] == 'left' else 'C1' for s in S]); ax.axhline(1, c='k', lw=.8)
    ax.set_xticks(range(len(S))); ax.set_xticklabels(lab, fontsize=6, rotation=60); ax.set_ylabel('median LiDAR z / mono z')
    ax.set_title(f"WiLoR mono metric scale error per window/hand (median {np.median([s['scale'] for s in S]):.2f}x)")
    fig.tight_layout(); fig.savefig(f'{O}/c17_mono_scale.png', dpi=130); plt.close(fig)
# 4 runtime / step table
steps = []
for j, kd in enumerate(KOS):
    p = f'{kd}/steps.jsonl'
    if os.path.exists(p): steps += [dict(json.loads(l), src=os.path.basename(kd.rstrip('/'))) for l in open(p)
                                    if j == 0 or json.loads(l)['step'] not in ('windows', 'cut', 'wilor_mono', 'partA_done')]
if steps:
    fig, ax = plt.subplots(figsize=(11, 0.32 * len(steps) + 1)); ax.axis('off')
    cell = []; prev = 0
    for s in steps:
        t = s.get('t', 0); extra = {k: v for k, v in s.items() if k not in ('step', 'rc', 't', 'src', 'per_window', 'frames')}
        cell.append([s['src'], s['step'], str(s['rc']), f"{t:.0f}", json.dumps(extra)[:70]])
    tb = ax.table(cellText=cell, colLabels=['run', 'step', 'rc', 't (s)', 'detail'], loc='center', cellLoc='left', colWidths=[.1, .17, .1, .07, .56])
    tb.auto_set_font_size(False); tb.set_fontsize(7)
    for (r, c), ce in tb.get_celld().items():
        if r > 0 and c == 2 and cell[r - 1][2] not in ('0',): ce.set_facecolor('#f4c7c3')
    ax.set_title('C17 Kaggle steps (T4); run koutN = kernel version vN; red = non-zero rc (Part A rows shown for first run only)'); fig.tight_layout(); fig.savefig(f'{O}/c17_steps.png', dpi=130); plt.close(fig)
print('charts', sorted(os.listdir(O)))
