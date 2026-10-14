import subprocess, os, glob, json, time
def sh(c):
    print('+', c, flush=True); r = subprocess.run(c, shell=True); print('rc', r.returncode, flush=True); return r.returncode
def find(pat):
    g = glob.glob('/kaggle/input/**/' + pat, recursive=True); return g[0] if g else None
T0 = time.time(); W = '/kaggle/working'; R = '/kaggle/tmp/robot'; os.makedirs(R, exist_ok=True)
SCR = find('xhey-di-pretrain/**/scripts.tgz') or find('xhey-stera-pretrain/**/scripts.tgz')
SES = os.path.dirname(find('xhey-iphone-sessions/**/depth'))
LAB = os.path.dirname(find('run_s2/episodes'))
XR = find('xhey_rgb_unified')
def unpack(name):
    d = find(name)
    if d and d.endswith('.tar'):
        sh(f'mkdir -p /kaggle/tmp/{name} && tar xf {d} -C /kaggle/tmp'); return f'/kaggle/tmp/{name}'
    return d
DI = unpack('di_unified'); ST = unpack('stera_unified')
print(SCR, SES, LAB, XR, DI, ST, flush=True)
sh(f'nvidia-smi -L; tar xzf {SCR} -C {R}')
run = '/kaggle/tmp/run_s2'; os.makedirs(run + '/episodes', exist_ok=True)
for p in glob.glob(LAB + '/episodes/*.json'):
    e = json.load(open(p)); n = os.path.basename(p)[:-5]
    e['video_path'] = f'{SES}/depth/{n}/rgb.mp4'; json.dump(e, open(f'{run}/episodes/{n}.json', 'w'))
os.environ['PYTHONPATH'] = f'{R}/scripts'
out = f'{W}/xhey_c8'
# c8: pretrain on public iPhone-LiDAR data, finetune on XHEY; same LOO eval (3 depth clips, seeds=3) + moving subset (--move-cm 5)
sh(f'cd {R} && python scripts/train_xhey.py --run {run} --out {out} --ablation '
   f'--mixes iphone+xhey_rgb+aug,iphone+di+xhey_rgb+aug --mix iphone+di+xhey_rgb+aug --xhey-rgb {XR} --di {DI} --stera {ST} '
   f'--seeds 3 --pre-steps 6000 --steps 3000 --move-cm 5 '
   f'> {W}/log_c8.txt 2>&1; grep -v -i warn {W}/log_c8.txt | tail -60')
sh(f'rm -rf {out}/cache')
print('C8_DONE', round(time.time() - T0), flush=True)
