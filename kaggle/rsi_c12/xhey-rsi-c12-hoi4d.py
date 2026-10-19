import subprocess, os, glob, json, time
def sh(c):
    print('+', c, flush=True); r = subprocess.run(c, shell=True); print('rc', r.returncode, flush=True); return r.returncode
def find(pat):
    g = glob.glob('/kaggle/input/**/' + pat, recursive=True); return g[0] if g else None
T0 = time.time(); W = '/kaggle/working'; R = '/kaggle/tmp/robot'; os.makedirs(R, exist_ok=True)
SCR = find('xhey-hoi4d-pretrain/**/scripts.tgz') or find('scripts.tgz'); QC = find('qc_hoi4d/scorecards.json')
SES = os.path.dirname(find('xhey-iphone-sessions/**/depth') or find('depth/2026-08-24_21-18'))
LAB = os.path.dirname(find('run_s2/episodes')); XR = find('xhey_rgb_unified'); HO = find('hoi4d_unified')
print(SCR, QC, SES, LAB, XR, HO, flush=True)
sh(f'nvidia-smi -L; tar xzf {SCR} -C {R}')
run = '/kaggle/tmp/run_s2'; os.makedirs(run + '/episodes', exist_ok=True)
for p in glob.glob(LAB + '/episodes/*.json'):
    e = json.load(open(p)); n = os.path.basename(p)[:-5]
    e['video_path'] = f'{SES}/depth/{n}/rgb.mp4'; json.dump(e, open(f'{run}/episodes/{n}.json', 'w'))
os.environ['PYTHONPATH'] = f'{R}/scripts'
# C12: HOI4D (head-mounted RGB-D, MANO GT hands, camera optical frame, metres; QC 1483/1487) pretrain.
# Same protocol as C11: LOO over the 3 depth clips, 3 seeds, pre 6000, ft 3000, move-cm 5, --no-final.
# Arm A iphone+hoi4d+xhey_rgb+aug, then same-code control arm B iphone+xhey_rgb+aug.
common = (f'--run {run} --xhey-rgb {XR} --seeds 3 --pre-steps 6000 --steps 3000 --move-cm 5 --no-final')
for tag, mix, extra in (('c12', 'iphone+hoi4d+xhey_rgb+aug', f'--hoi4d {HO} --qc-filter {QC}'),
                        ('c12ctl', 'iphone+xhey_rgb+aug', '')):
    out = f'{W}/xhey_{tag}'
    sh(f'cd {R} && python scripts/train_xhey.py {common} --ablation --mixes {mix} --mix {mix} --out {out} {extra} '
       f'> {W}/log_{tag}.txt 2>&1; grep -v -i warn {W}/log_{tag}.txt | tail -40')
    sh(f'rm -rf {out}/cache')
    print(tag.upper() + '_DONE', round(time.time() - T0), flush=True)
