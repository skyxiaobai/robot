import subprocess, os, glob, json, time
def sh(c):
    print('+', c, flush=True); r = subprocess.run(c, shell=True); print('rc', r.returncode, flush=True); return r.returncode
T0 = time.time(); W = '/kaggle/working'; R = '/kaggle/tmp/robot'; os.makedirs(R, exist_ok=True)
AOE = os.path.dirname(glob.glob('/kaggle/input/**/xhey-openaoe-pretrain/**/scripts.tgz', recursive=True)[0])
SES = os.path.dirname(glob.glob('/kaggle/input/**/xhey-iphone-sessions/**/depth', recursive=True)[0])
LAB = os.path.dirname(glob.glob('/kaggle/input/**/run_s2/episodes', recursive=True)[0])
XR = glob.glob('/kaggle/input/**/xhey_rgb_unified', recursive=True)[0]
print(AOE, SES, LAB, XR, flush=True)
sh(f'nvidia-smi -L; tar xzf {AOE}/scripts.tgz -C {R}')
# episodes with video_path pointing at the read-only session videos
run = '/kaggle/tmp/run_s2'; os.makedirs(run + '/episodes', exist_ok=True)
for p in glob.glob(LAB + '/episodes/*.json'):
    e = json.load(open(p)); n = os.path.basename(p)[:-5]
    e['video_path'] = f'{SES}/depth/{n}/rgb.mp4'; json.dump(e, open(f'{run}/episodes/{n}.json', 'w'))
UN = glob.glob(AOE + '/**/unified', recursive=True)[0]
os.environ['PYTHONPATH'] = f'{R}/scripts'
jobs = [('nano', f'{UN}/nano', 'iphone,iphone+xhey_rgb+aug,iphone+openaoe+aug,iphone+openaoe+xhey_rgb+aug', 6000),
        ('tiny', f'{UN}/tiny', 'iphone+openaoe+aug,iphone+openaoe+xhey_rgb+aug', 20000)]
for tag, od, mixes, ps in jobs:
    out = f'{W}/xhey_{tag}'
    if os.path.exists(out + '/metrics.json'): continue
    sh(f'cd {R} && python scripts/train_xhey.py --run {run} --out {out} --ablation --mixes {mixes} '
       f'--mix iphone+openaoe+xhey_rgb+aug --xhey-rgb {XR} --openaoe {od} --seeds 3 --pre-steps {ps} '
       f'> {W}/log_{tag}.txt 2>&1; grep -v -i warn {W}/log_{tag}.txt | tail -40')
    sh(f'rm -rf {out}/cache')
    print('JOB', tag, round(time.time() - T0), flush=True)
print('ALL_DONE', round(time.time() - T0))
