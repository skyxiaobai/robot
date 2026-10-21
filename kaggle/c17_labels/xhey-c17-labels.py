# C17 hand-label method comparison on Justin's own XHEY clips (bounded smoke; do not force).
# Part A: WiLoR monocular (current xhey_rgb labeler) on 20 s windows of the iPhone LiDAR clips (baseline for depth error).
# Part B: Dyn-HaMR (github.com/ZhengdiYu/Dyn-HaMR, MIT) with ARKit cameras (no SLAM) on the same windows;
#         every install/run step is logged as C17STEP {...}; first hard failure is recorded and the rest skipped.
# EgoXperience: not attempted (code + weights unreleased as of 2026-10-21).
import subprocess, os, glob, json, time, shutil, sys
T0 = time.time(); W = '/kaggle/working'; TMP = '/kaggle/tmp'; os.makedirs(TMP, exist_ok=True)
BUDGET = 11.0 * 3600
def sh(c, t=None):
    print('+', c[:400], flush=True)
    try: r = subprocess.run(['bash', '-o', 'pipefail', '-c', c], timeout=t); rc = r.returncode
    except subprocess.TimeoutExpired: rc = 'timeout'
    print('rc', rc, flush=True); return rc
def step(name, rc, **k):
    k.update(step=name, rc=rc, t=round(time.time() - T0, 1)); print('C17STEP ' + json.dumps(k), flush=True)
    open(W + '/steps.jsonl', 'a').write(json.dumps(k) + '\n'); return rc
def find(pat):
    g = glob.glob('/kaggle/input/**/' + pat, recursive=True); return g[0] if g else None
sh('nvidia-smi -L; nvcc --version | tail -2; python --version; df -h /kaggle/tmp | tail -1')
SES = os.path.dirname(find('depth/2026-08-24_21-18/rgb.mp4').rsplit('/depth/', 1)[0] + '/depth')
SES = find('depth/2026-08-24_21-18/rgb.mp4').rsplit('/depth/', 1)[0]
LAB = os.path.dirname(find('run_s2/episodes')); XR = find('xhey_rgb_unified'); MANO = os.path.dirname(find('mano/MANO_RIGHT.pkl'))
SCR = find('xhey-c15-wrist-data/**/scripts.tgz') or find('scripts.tgz')
print(SES, LAB, XR, MANO, SCR, flush=True)
R = TMP + '/robot'; os.makedirs(R, exist_ok=True); sh(f'tar xzf {SCR} -C {R}')
# ---------- windows (same rule as box make_windows.py; 20 s = 300 frames @15 fps)
import numpy as np
L = 300; wins = []
os.makedirs(W + '/cams', exist_ok=True)
for p in sorted(glob.glob(LAB + '/episodes/*.json')):
    e = json.load(open(p)); n = e['num_frames']; clip = os.path.basename(p)[:-5]
    lv = np.array(e['hands']['left']['valid'], bool); rv = np.array(e['hands']['right']['valid'], bool)
    v = (lv | rv).astype(float); both = (lv & rv).astype(float)
    c = np.convolve(v + 0.5 * both, np.ones(L) / L, 'valid')
    P = np.asarray(e['camera_poses'], float); K = np.asarray(e['camera_intrinsic'], float)
    for k, q in enumerate((0.2, 0.5, 0.8)):
        a, b = int(q * n - 0.15 * n), int(q * n + 0.15 * n) - L
        s = a + int(np.argmax(c[a:b])); name = f'{clip}_w{k}'
        w2c = np.linalg.inv(P[s:s + L]); sc = 0.5
        d = f'{TMP}/dh/dynhamr/cameras/{name}/shot-0'; os.makedirs(d, exist_ok=True)
        intr = np.tile([K[0, 0] * sc, K[1, 1] * sc, K[0, 2] * sc, K[1, 2] * sc], (L, 1)).astype(np.float32)
        np.savez(d + '/cameras.npz', height=720, width=960, focal=float(K[0, 0] * sc), intrins=intr, w2c=w2c.astype(np.float32))
        shutil.copy(d + '/cameras.npz', f'{W}/cams/{name}.npz')
        wins.append(dict(name=name, clip=clip, kind='depth', start=int(s), end=int(s + L), fps=e['fps']))
    del e
for p in sorted(glob.glob(XR + '/*.json'))[:3]:
    clip = os.path.basename(p).split('_tmp__')[0]; wins.append(dict(name=f'{clip}_w0', clip=clip, kind='nodepth', start=750, end=1050, fps=15.0))
json.dump(wins, open(W + '/windows.json', 'w'), indent=1); step('windows', 0, n=len(wins))
# ---------- cut window videos (full res for WiLoR with real calib; 960x720 jpg frames for Dyn-HaMR)
os.makedirs(TMP + '/wv', exist_ok=True)
for w in wins:
    src = f"{SES}/{'depth' if w['kind'] == 'depth' else 'nodepth'}/{w['clip']}/rgb.mp4"
    w['calib'] = f"{SES}/{'depth' if w['kind'] == 'depth' else 'nodepth'}/{w['clip']}/calib.yaml"
    mp4 = f"{TMP}/wv/{w['name']}.mp4"; img = f"{TMP}/dh/images/{w['name']}"; os.makedirs(img, exist_ok=True)
    sh(f"ffmpeg -v error -y -i {src} -vf \"trim=start_frame={w['start']}:end_frame={w['end']},setpts=PTS-STARTPTS\" -c:v libx264 -crf 16 -an {mp4}")
    sh(f"ffmpeg -v error -y -i {mp4} -vf scale=960:720 -q:v 2 -start_number 0 {img}/%06d.jpg")
    w['n_img'] = len(os.listdir(img))
step('cut', 0, frames=[w['n_img'] for w in wins])
# ---------- Part A: WiLoR monocular (same install as xhey-wilor-labels)
tA = time.time()
sh('git clone -q --depth 1 https://github.com/rolpotamias/WiLoR /kaggle/tmp/WiLoR')
for pk in ['ultralytics==8.1.34', 'pytorch-lightning', 'smplx==0.1.28', 'yacs', 'timm', 'einops', 'pyrootutils', 'hydra-core', 'scikit-image', 'trimesh', 'xtcocotools']:
    if sh(f'pip install -q {pk} 2>&1 | tail -2') and pk.startswith('ultralytics'): sh('pip install -q ultralytics 2>&1 | tail -2')
sh('pip install -q --no-build-isolation git+https://github.com/mattloper/chumpy 2>&1 | tail -2')
ck = TMP + '/wck'; os.makedirs(ck, exist_ok=True)
for f in ['detector.pt', 'wilor_final.ckpt', 'model_config.yaml']:
    sh(f'curl -sL -o {ck}/{f} https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/{f}')
mano = TMP + '/mano'; os.makedirs(mano, exist_ok=True); sh(f'cp {MANO}/*.pkl {mano}/ && cp /kaggle/tmp/WiLoR/mano_data/mano_mean_params.npz {mano}/')
envA = dict(os.environ, WILOR_CHECKPOINT=f'{ck}/wilor_final.ckpt', WILOR_DETECTOR=f'{ck}/detector.pt', WILOR_CONFIG=f'{ck}/model_config.yaml',
            MANO_MODEL_DIR=mano, PYTHONPATH=f'/kaggle/tmp/WiLoR:{R}/scripts')
timesA = {}
SKIP_A = True  # v3: Part A outputs already collected from v2 (kout2/wilor_mono)
for w in [x for x in wins if x['kind'] == 'depth' and not SKIP_A]:
    o = f"{W}/wilor_mono/{w['name']}"; t = time.time()
    r = subprocess.run(f"cd {R} && python scripts/wilor_video_to_episodes.py --videos {TMP}/wv/{w['name']}.mp4 --out {o} --calib {w['calib']} --stride 1 --max-frames 300 --source xhey_rgb --license-note own > {W}/log_wilor_{w['name']}.txt 2>&1", shell=True, env=envA)
    timesA[w['name']] = round(time.time() - t, 1); step('wilor_mono', r.returncode, win=w['name'], s=timesA[w['name']], files=len(glob.glob(o + '/*.json')))
step('partA_done', 0, s=round(time.time() - tA), per_window=timesA)
# ---------- Part B: Dyn-HaMR
DH = TMP + '/Dyn-HaMR'; tB = time.time(); fail = None
def need(name, rc, **k):
    global fail
    step(name, rc, **k)
    if rc not in (0, '0') and fail is None: fail = name
    return fail is None
ok = need('dh_clone', sh(f'git clone -q --recursive https://github.com/ZhengdiYu/Dyn-HaMR.git {DH}', 1800))
ENV = TMP + '/dhenv'; PY = ENV + '/bin/python'
if ok: ok = need('dh_venv_py310', sh(f'pip install -q uv && uv venv -q -p 3.10 {ENV}', 900))
open(TMP + '/cons.txt', 'w').write('numpy<2\n'); UV = f'uv pip install -q --python {PY} -c {TMP}/cons.txt'
if ok: ok = need('dh_pip', sh(f'{UV} pip "setuptools==69.5.1" wheel cython "numpy==1.26.4" 2>&1 | tail -3', 600))
if ok: ok = need('dh_torch', sh(f'{UV} torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -3', 1800))
if ok: ok = need('dh_torch_scatter', sh(f'{UV} torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html 2>&1 | tail -3', 900))
if ok: ok = need('dh_requirements', sh(f'cd {DH} && grep -v "^setuptools" requirements.txt > {TMP}/req.txt && {UV} --no-build-isolation -r {TMP}/req.txt 2>&1 | tail -8 && {UV} "setuptools==69.5.1" 2>&1 | tail -3', 2400))  # v3: repo pins setuptools==59.5.0 (no PEP 660 editable)
if ok: ok = need('dh_pkg', sh(f'cd {DH} && {UV} --no-build-isolation -e . 2>&1 | tail -3', 600))
if ok: ok = need('dh_detectron2', sh(f'cd {DH} && TORCH_CUDA_ARCH_LIST=7.5 MAX_JOBS=4 {UV} --no-build-isolation "git+https://github.com/facebookresearch/detectron2" 2>&1 | tail -5', 2400))
if ok: ok = need('dh_hamer', sh(f'cd {DH}/third-party/hamer && {UV} --no-build-isolation -e ".[all]" 2>&1 | tail -5 && {UV} --no-build-isolation -e third-party/ViTPose 2>&1 | tail -3', 2400))
if ok: ok = need('dh_mano_egl', sh(f'{UV} --no-build-isolation "git+https://github.com/otaheri/MANO" loguru plyfile 2>&1 | tail -3 && {PY} -m pip install -q --no-deps --ignore-requires-python "git+https://github.com/nghorbani/human_body_prior" 2>&1 | tail -2 && (apt-get update -qq && apt-get install -y -qq libegl1 libgl1 libgles2 > /dev/null 2>&1 || true)', 1800))  # v4: v3 failed dh_run: No module named mano (+ pyrender needs EGL); v5: + loguru, plyfile, human_body_prior main (load_model; py>=3.11 pin ignored) -- box replica imports run_opt OK
if ok:
    rc = sh(f'cd {DH} && {PY} -c "import torch, mmcv, smplx, hydra, hamer, detectron2; print(torch.__version__, torch.cuda.is_available(), mmcv.__version__)"', 600)
    ok = need('dh_import_check', rc)
if ok:  # checkpoints: HaMeR demo data (6 GB), WiLoR detector, MANO, BMC (computed from Hand-BMC joints.zip)
    rc = sh(f'cd {DH} && (wget -q https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz || {ENV}/bin/gdown -q 1mv7CUAnm73oKsEEG1xE3xH2C_oqcFSzT) && tar --warning=no-unknown-keyword --exclude=".*" -xf hamer_demo_data.tar.gz && rm -f hamer_demo_data.tar.gz && ls _DATA', 3600)
    ok = need('dh_ckpt_hamer', rc)
if ok:
    sh(f'mkdir -p {DH}/third-party/hamer/pretrained_models {DH}/_DATA/data/mano && cp {ck}/detector.pt {DH}/third-party/hamer/pretrained_models/ && cp {MANO}/MANO_RIGHT.pkl {DH}/_DATA/data/mano/')
    hb = TMP + '/hbmc'
    hull = ("import numpy as np, runpy, sys\n_c = np.cross\n"
            "def cross(a, b, *k, **kw):\n a = np.asarray(a); b = np.asarray(b)\n"
            " return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0] if a.shape[-1] == 2 and b.shape[-1] == 2 else _c(a, b, *k, **kw)\n"
            "np.cross = cross\nsrc = open('calculate_convex_hull.py').read().replace('figManager.window.showMaximized()', 'pass')"
            ".replace('plt.show()', 'plt.close(\"all\")').replace('all_del_hulls = np.array(all_del_hulls)', 'all_del_hulls = np.array(all_del_hulls, dtype=object)')\n"
            "open('cch_patched.py', 'w').write(src); sys.argv = ['x']; runpy.run_path('cch_patched.py', run_name='__main__')\n")
    os.makedirs(hb, exist_ok=True)
    rc = sh(f'git clone -q --depth 1 https://github.com/MengHao666/Hand-BMC-pytorch.git {hb}/r && cd {hb}/r && pip install -q gdown rdp && gdown -q 1_wV8QjmsVCMBEBhm56gFA2XTyU8VEHzk -O joints.zip && unzip -q -o joints.zip && MPLBACKEND=Agg python calculate_bmc.py > /dev/null', 2400)
    open(f'{hb}/r/run_hull.py', 'w').write(hull)
    if rc == 0: rc = sh(f'cd {hb}/r && MPLBACKEND=Agg python run_hull.py > /dev/null && mkdir -p {DH}/_DATA/BMC && cp BMC/*.npy {DH}/_DATA/BMC/ && ls {DH}/_DATA/BMC', 2400)
    ok = need('dh_bmc', rc)
if ok: ok = need('dh_runopt_import', sh(f'cd {DH}/dyn-hamr && {PY} -c "import run_opt; print(\'RUN_OPT_IMPORT_OK\')" 2>&1 | grep -v -i warn | tail -5', 900))  # v6: v5 ran this check before the BMC data existed
timesB = {}
if ok:
    sh(f'ls -R {DH}/_DATA | head -40')
    for w in [x for x in wins if x['kind'] == 'depth']:
        if time.time() - T0 > BUDGET - 1800: step('budget_stop', 0); break
        t = time.time(); log = f"{W}/log_dyn_{w['name']}.txt"
        cmd = (f"cd {DH}/dyn-hamr && {PY} run_opt.py data=video_driod data.root={TMP}/dh data.seq={w['name']} "
               f"data.src_path={TMP}/wv/{w['name']}.mp4 data.start_idx=0 data.end_idx={L} data.frame_opts.fps=15 "
               f"run_opt=True run_vis=False is_static=False > {log} 2>&1")
        rc = sh(cmd, 3 * 3600); timesB[w['name']] = round(time.time() - t, 1)
        sh(f'grep -v -i warn {log} | tail -30')
        res = sorted(glob.glob(f"{DH}/**/{w['name']}*_world_results.npz", recursive=True), key=os.path.getmtime)
        o = f"{W}/dynhamr/{w['name']}"; os.makedirs(o, exist_ok=True)
        for f in res: shutil.copy(f, o)
        for f in glob.glob(f"{DH}/**/{w['name']}*/track_info.json", recursive=True)[:1] + glob.glob(f"{DH}/**/{w['name']}*/cameras.json", recursive=True)[:1]: shutil.copy(f, o)
        trk = f'{TMP}/dh/dynhamr/track_preds/{w["name"]}'
        if os.path.isdir(trk): sh(f'tar czf {o}/track_preds.tgz -C {trk} . 2>/dev/null')
        good = need('dh_run', rc if res else (rc or 'no_results'), win=w['name'], s=timesB[w['name']], results=[os.path.basename(f) for f in res])
        if not good: break
        if res:  # export camera-frame MANO joints with Dyn-HaMR's own hand model
            exp = f'''
import sys, json, numpy as np, torch
sys.path.insert(0, "{DH}/dyn-hamr")
from body_model import MANO
from body_model.utils import run_mano
r = dict(np.load("{res[-1]}", allow_pickle=True)); print({{k: getattr(v, "shape", None) for k, v in r.items()}})
B, T = r["trans"].shape[:2]
hm = MANO(model_path="{DH}/_DATA/data/mano", is_rhand=True, batch_size=B * T, pose2rot=True, use_pca=False, flat_hand_mean=False,
          num_pca_comps=45, mean_params="{DH}/_DATA/data/mano_mean_params.npz")
tt = lambda k: torch.as_tensor(r[k]).float()
isr = torch.as_tensor(r["is_right"]) if "is_right" in r else None
out = run_mano(hm, tt("trans"), tt("root_orient"), tt("pose_body"), isr, betas=tt("betas") if "betas" in r else None)
J = out["joints"].detach().numpy() if isinstance(out, dict) else out[0].detach().numpy()
np.savez("{o}/joints_world.npz", joints=J, cam_R=r["cam_R"], cam_t=r["cam_t"], is_right=r.get("is_right", np.zeros(0)))
print("EXPORT_OK", J.shape)
'''
            open(f'{TMP}/exp.py', 'w').write(exp); step('dh_export', sh(f'{PY} {TMP}/exp.py 2>&1 | tail -15', 900), win=w['name'])
step('partB_done', fail or 0, first_fail=fail, s=round(time.time() - tB), per_window=timesB)
# DROID-SLAM for no-depth (xhey_rgb) clips is not attempted when Part B already failed or ran out of budget.
sh(f'find {W} -name "*.mp4" -delete; du -sh {W}/* | tail -20')
print('ALL_DONE', round(time.time() - T0), flush=True)
