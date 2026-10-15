import subprocess, os, glob, json, time
def sh(c):
    print('+', c, flush=True); r = subprocess.run(c, shell=True); print('rc', r.returncode, flush=True); return r.returncode
def find(pat):
    g = glob.glob('/kaggle/input/**/' + pat, recursive=True); return g[0] if g else None
def unpack_from(dataset, name):
    d = find(dataset + '/**/' + name) or find(dataset + '/**/' + name + '.tar')
    if not d:
        return None
    if d.endswith('.tar'):
        sh(f'mkdir -p /kaggle/tmp/{name} && tar xf {d} -C /kaggle/tmp'); return f'/kaggle/tmp/{name}'
    return d
T0 = time.time(); W = '/kaggle/working'; R = '/kaggle/tmp/robot'; os.makedirs(R, exist_ok=True)
# C11 bundle: scripts that canonicalize camera_yup -> camera optical, plus qc_di/scorecards.json.
# Do not take scripts.tgz from xhey-di-pretrain (that is the C9 v1 world-frame / camera_yup path).
SCR = find('xhey-c11-di-cam/**/scripts.tgz')
SES = os.path.dirname(find('xhey-iphone-sessions/**/depth') or '')
LAB = os.path.dirname(find('run_s2/episodes') or '')
XR = find('xhey_rgb_unified')
DI = unpack_from('xhey-c11-di-cam', 'di_unified') or unpack_from('xhey-di-pretrain', 'di_unified')
ST = unpack_from('xhey-stera-pretrain', 'stera_unified')
qc = sorted(set(
    glob.glob('/kaggle/input/**/qc/*/scorecards.json', recursive=True)
    + glob.glob('/kaggle/input/**/qc_di/scorecards.json', recursive=True)))
QC = ' '.join(qc)
print('qc scorecards:', QC, flush=True)
print('stera batches:', sorted(os.listdir(ST)) if ST else None, flush=True)
print(SCR, SES, LAB, XR, DI, ST, flush=True)
if not (SCR and SES and LAB and XR and DI and ST and QC):
    raise SystemExit('C9b missing camera-frame scripts, Stera, DI, xhey_rgb, or QC scorecards')
sh(f'nvidia-smi -L; tar xzf {SCR} -C {R}')
frames = open(f'{R}/scripts/common/frames.py').read()
train = open(f'{R}/scripts/train_xhey.py').read()
if 'def canonicalize_for_xhey' not in frames or 'canonicalize_for_xhey' not in train:
    raise SystemExit('C9b scripts do not canonicalize DI into the camera frame')
run = '/kaggle/tmp/run_s2'; os.makedirs(run + '/episodes', exist_ok=True)
for p in glob.glob(LAB + '/episodes/*.json'):
    e = json.load(open(p)); n = os.path.basename(p)[:-5]
    e['video_path'] = f'{SES}/depth/{n}/rgb.mp4'; json.dump(e, open(f'{run}/episodes/{n}.json', 'w'))
os.environ['PYTHONPATH'] = f'{R}/scripts'
out = f'{W}/xhey_c9b'
# c9b: Stera b01-b05 + DI camera frame + iPhone + xhey_rgb + aug.
# QC-filter like C7b (qc/*/scorecards.json) and C11 (qc_di/scorecards.json).
# Held-out eval is still the depth-clip LOO; a depth clip that fails the scorecard drops out, same as C7b.
sh(f'cd {R} && python scripts/train_xhey.py --run {run} --out {out} --ablation '
   f'--mixes iphone+xhey_rgb+aug,iphone+stera+di+xhey_rgb+aug --mix iphone+stera+di+xhey_rgb+aug --xhey-rgb {XR} --di {DI} --stera {ST} '
   f'--qc-filter {QC} --seeds 3 --pre-steps 6000 --steps 3000 --move-cm 5 '
   f'> {W}/log_c9b.txt 2>&1; grep -v -i warn {W}/log_c9b.txt | tail -60')
sh(f'rm -rf {out}/cache')
print('C9B_DONE', round(time.time() - T0), flush=True)
