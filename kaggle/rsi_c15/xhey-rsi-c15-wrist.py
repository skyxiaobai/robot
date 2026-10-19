import subprocess, os, glob, json, time
def sh(c):
    print('+', c, flush=True); r = subprocess.run(c, shell=True); print('rc', r.returncode, flush=True); return r.returncode
def find(pat):
    g = glob.glob('/kaggle/input/**/' + pat, recursive=True); return g[0] if g else None
T0 = time.time(); W = '/kaggle/working'; R = '/kaggle/tmp/robot'; os.makedirs(R, exist_ok=True)
SCR = find('xhey-c15-wrist-data/**/scripts.tgz') or find('scripts.tgz'); QC = find('qc_xhey_rgb/scorecards.json')
SES = os.path.dirname(find('xhey-iphone-sessions/**/depth') or find('depth/2026-08-24_21-18'))
LAB = os.path.dirname(find('run_s2/episodes')); XR = find('xhey_rgb_unified')
print(SCR, QC, SES, LAB, XR, flush=True)
sh(f'nvidia-smi -L; tar xzf {SCR} -C {R}')
run = '/kaggle/tmp/run_s2'; os.makedirs(run + '/episodes', exist_ok=True)
for p in glob.glob(LAB + '/episodes/*.json'):
    e = json.load(open(p)); n = os.path.basename(p)[:-5]
    e['video_path'] = f'{SES}/depth/{n}/rgb.mp4'; json.dump(e, open(f'{run}/episodes/{n}.json', 'w'))
os.environ['PYTHONPATH'] = f'{R}/scripts'
# C15: WiLoR wrist-noise hypothesis test (UNVERIFIED). Step1: per-frame jump med ~1.2-1.4 cm vs
# 1 s displacement med ~5.5-6.2 cm; static-window drift RMS ~1.0 cm. Step2 wrist-pixel depth
# correction gave no material noise drop (iPhone wrist is already LiDAR-placed), so no corrected arm.
# XHEY already predicts per-step wrist deltas; this arm instead up-weights the cumulative
# relative-to-current-wrist displacement loss (--cum-weight 1.0 vs default 0.1).
# Same protocol as C11-C14: LOO over the 3 depth clips, 3 seeds, pre 6000, ft 3000, move-cm 5,
# --no-final; --qc-filter on xhey_rgb in both arms. Arm A cum-weight 1.0, arm B control 0.1.
common = (f'--run {run} --xhey-rgb {XR} --seeds 3 --pre-steps 6000 --steps 3000 --move-cm 5 --no-final '
          f'--qc-filter {QC}')
for tag, mix, extra in (('c15', 'iphone+xhey_rgb+aug', '--cum-weight 1.0'),
                        ('c15ctl', 'iphone+xhey_rgb+aug', '--cum-weight 0.1')):
    out = f'{W}/xhey_{tag}'
    sh(f'cd {R} && python scripts/train_xhey.py {common} --ablation --mixes {mix} --mix {mix} --out {out} {extra} '
       f'> {W}/log_{tag}.txt 2>&1; grep -v -i warn {W}/log_{tag}.txt | tail -40')
    sh(f'rm -rf {out}/cache')
    print(tag.upper() + '_DONE', round(time.time() - T0), flush=True)
