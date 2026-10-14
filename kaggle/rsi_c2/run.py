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
run = '/kaggle/tmp/run_s2'; os.makedirs(run + '/episodes', exist_ok=True)
for p in glob.glob(LAB + '/episodes/*.json'):
    e = json.load(open(p)); n = os.path.basename(p)[:-5]
    e['video_path'] = f'{SES}/depth/{n}/rgb.mp4'; json.dump(e, open(f'{run}/episodes/{n}.json', 'w'))
os.environ['PYTHONPATH'] = f'{R}/scripts'
out = f'{W}/xhey_c2'
# C2: same best mix, longer finetune
sh(f'cd {R} && python scripts/train_xhey.py --run {run} --out {out} --ablation '
   f'--mixes iphone+xhey_rgb+aug --mix iphone+xhey_rgb+aug --xhey-rgb {XR} '
   f'--seeds 3 --pre-steps 6000 --steps 10000 '
   f'> {W}/log_c2.txt 2>&1; grep -v -i warn {W}/log_c2.txt | tail -50')
sh(f'rm -rf {out}/cache')
print('C2_DONE', round(time.time() - T0), flush=True)
