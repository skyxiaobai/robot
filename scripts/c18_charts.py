import json, sys, os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D = sys.argv[1]; O = sys.argv[2]; os.makedirs(O, exist_ok=True)
S = json.load(open(f'{D}/c18_summary.json')); M = np.load(f'{D}/c18_masks.npz')
plt.rcParams.update({'font.size': 10})
srcs = [s for s in ('xphi', 'aoe') if s in S]; NAME = {'xphi': 'xPhi-Ego-100 (WiLoR labels)', 'aoe': 'Open-AoE nano (native MANO)'}
T = {s: {r['sel']: r for r in S[s]['table'] if r} for s in srcs}
# 1 hours + segments
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
sels = ['all', 'current', 'flora', 'flora_and_qc']; lab = ['all labelled', 'current (motion+QC)', 'Flora intervals', 'Flora ∩ QC-pass']
x = np.arange(len(sels)); wdt = 0.38
for i, s in enumerate(srcs):
    h = [T[s][k]['hours'] for k in sels]; b = ax[0].bar(x + (i - 0.5) * wdt, h, wdt, label=NAME[s])
    ax[0].bar_label(b, fmt='%.2f', fontsize=8)
    r = [T[s][k]['runs'] for k in sels]; b = ax[1].bar(x + (i - 0.5) * wdt, r, wdt, label=NAME[s]); ax[1].bar_label(b, fontsize=8)
for a_, t in zip(ax, ['retained hours', 'number of contiguous segments']):
    a_.set_xticks(x); a_.set_xticklabels(lab, rotation=12); a_.set_title(t); a_.legend(fontsize=8)
fig.suptitle('C18: Flora-TempSeg action intervals vs current segment selection'); fig.tight_layout(); fig.savefig(f'{O}/c18_hours_segments.png', dpi=130); plt.close(fig)
# 2 overlap
fig, ax = plt.subplots(figsize=(8, 3.2)); cats = ['both', 'flora_only', 'current_only', 'neither']; col = ['#4c9a2a', '#3b75af', '#ef8636', '#bbbbbb']
for i, s in enumerate(srcs):
    left = 0
    for c, cl in zip(cats, col):
        f = T[s][c]['frac']; ax.barh(i, f, left=left, color=cl, label=c if i == 0 else None)
        if f > 0.04: ax.text(left + f / 2, i, f'{f:.0%}', ha='center', va='center', fontsize=9, color='white')
        left += f
ax.set_yticks(range(len(srcs))); ax.set_yticklabels([f"{NAME[s]}\nIoU={S[s]['iou']:.2f}" for s in srcs]); ax.set_xlim(0, 1)
ax.set_xlabel('fraction of labelled time'); ax.legend(ncol=4, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, 1.25)); fig.tight_layout(); fig.savefig(f'{O}/c18_overlap.png', dpi=130); plt.close(fig)
# 3 quality per category
fig, ax = plt.subplots(1, 4, figsize=(15, 3.8)); cats = ['all', 'current', 'flora', 'both', 'flora_only', 'current_only', 'neither']
for j, (k, t) in enumerate([('speed_med_cm', 'median 1 s wrist displacement (cm)'), ('moving_frac', 'fraction moving (>=3 cm/s)'),
                            ('jitter_med_cm', 'median label jitter (cm, 2nd diff)'), ('qc_pass_frac', 'fraction in QC-pass 3 s windows')]):
    for i, s in enumerate(srcs):
        v = [T[s][c][k] if T[s][c][k] is not None else 0 for c in cats]
        ax[j].bar(np.arange(len(cats)) + (i - 0.5) * 0.38, v, 0.38, label=NAME[s])
    ax[j].set_xticks(range(len(cats))); ax[j].set_xticklabels(cats, rotation=40, fontsize=8); ax[j].set_title(t, fontsize=9)
ax[0].legend(fontsize=7); fig.suptitle('Quality of time picked by each selector (label statistics inside the selection)'); fig.tight_layout(); fig.savefig(f'{O}/c18_quality.png', dpi=130); plt.close(fig)
# 4 timelines
keys = sorted({k.rsplit('__', 1)[0] for k in M.files})
ex = [k for k in keys if k.startswith('xphi')][:3] + [k for k in keys if k.startswith('aoe')][:3]
fig, ax = plt.subplots(len(ex), 1, figsize=(13, 1.5 * len(ex) + 0.5), squeeze=False)
for i, k in enumerate(ex):
    a_ = ax[i, 0]; sp = M[k + '__spd']; t = np.arange(len(sp)) / 10
    a_.plot(t, np.nan_to_num(sp), color='k', lw=0.6); top = max(10, np.nanpercentile(sp, 98) if np.isfinite(sp).any() else 10)
    a_.fill_between(t, top * 1.02, top * 1.22, where=M[k + '__flora'], color='#3b75af', step='mid', label='Flora')
    a_.fill_between(t, top * 1.25, top * 1.45, where=M[k + '__cur'], color='#ef8636', step='mid', label='current')
    a_.axhline(3, color='r', lw=0.5, ls=':'); a_.set_ylim(0, top * 1.5); a_.set_ylabel('cm/s'); a_.set_title(k.replace('__', ': '), fontsize=8, loc='left')
ax[0, 0].legend(fontsize=7, loc='upper right'); ax[-1, 0].set_xlabel('s'); fig.tight_layout(); fig.savefig(f'{O}/c18_timelines.png', dpi=120); plt.close(fig)
print('charts ok')
