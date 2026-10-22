"""C18 charts when Flora cannot finish: current selection stats + measured Flora timings/extrapolation.
usage: c18_feasibility.py CURRENT_SUMMARY_DIR OUT_DIR LOAD_S WINDOW_S[,WINDOW_S2...] [LABELS]"""
import json, sys, os, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
S = json.load(open(f'{sys.argv[1]}/c18_summary.json')); O = sys.argv[2]; os.makedirs(O, exist_ok=True)
load_s = float(sys.argv[3]); win = [float(x) for x in sys.argv[4].split(',')]; labs = sys.argv[5].split(',') if len(sys.argv) > 5 else [f'run{i}' for i in range(len(win))]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
srcs = ['xphi', 'aoe']; T = {s: {r['sel']: r for r in S[s]['table']} for s in srcs}
x = np.arange(2); ax[0].bar(x - .2, [T[s]['all']['hours'] for s in srcs], .4, label='labelled'); b = ax[0].bar(x + .2, [T[s]['current']['hours'] for s in srcs], .4, label='current selection')
ax[0].bar_label(b, labels=[f"{T[s]['current']['hours']:.2f} h\n{T[s]['current']['runs']} segs" for s in srcs], fontsize=8)
ax[0].set_xticks(x); ax[0].set_xticklabels(['xPhi-Ego-100 (102)', 'Open-AoE nano (111)']); ax[0].set_ylabel('hours'); ax[0].legend(fontsize=8); ax[0].set_title('current motion+QC selection')
for i, (k, lab) in enumerate([('qc_pass_frac', 'QC pass frac'), ('jitter_med_cm', 'jitter median (cm)')]):
    pass
ax[1].bar(x - .2, [T[s]['all']['jitter_med_cm'] for s in srcs], .4, label='all labelled'); b = ax[1].bar(x + .2, [T[s]['current']['jitter_med_cm'] for s in srcs], .4, label='current selection')
ax[1].bar_label(b, fmt='%.2f', fontsize=8); ax[1].set_xticks(x); ax[1].set_xticklabels(['xPhi', 'Open-AoE']); ax[1].set_ylabel('wrist jitter median (cm)')
ax[1].set_title('quality: QC pass all/current = ' + ', '.join(f"{s} {T[s]['all']['qc_pass_frac']:.0%}/{T[s]['current']['qc_pass_frac']:.0%}" for s in srcs), fontsize=9); ax[1].legend(fontsize=8)
hours = sum(T[s]['all']['hours'] for s in srcs); nwin = hours * 3600 / 12
need = [nwin * w / 3600 for w in win]
b = ax[2].bar(range(len(win)), need, color='C3'); ax[2].bar_label(b, labels=[f'{w:.0f} s/window\n-> {n:.0f} GPU-h' for w, n in zip(win, need)], fontsize=8)
ax[2].axhline(12, c='k', ls='--'); ax[2].text(0, 13, 'Kaggle session limit 12 h', fontsize=8); ax[2].set_yscale('log'); ax[2].set_ylim(1, 3000)
ax[2].set_xticks(range(len(win))); ax[2].set_xticklabels(labs, fontsize=8); ax[2].set_ylabel('GPU-hours for all videos (log)')
ax[2].set_title(f'Flora-9B on 2xT4: load {load_s:.0f} s, ~{nwin:.0f} windows', fontsize=9)
fig.suptitle('C18 Flora vs current selection: Flora could not complete on Kaggle T4'); fig.tight_layout(); fig.savefig(f'{O}/c18_feasibility.png', dpi=130); plt.close(fig)
print('ok', O, round(nwin), [round(n) for n in need])
