# xPhi-Ego-100 -> WiLoR camera-frame episodes (same setup as xhey-wilor-labels).
# Videos pre-transcoded on box to 960x540 @10 fps. Intrinsics are ASSUMED (no calibration published):
# pinhole, HFOV 90 deg (fx=fy=960 @1920x1080, principal point at centre), distortion ignored -> metric scale uncertain.
import subprocess, os, glob, time, json, shutil
def sh(c):
    print('+', c, flush=True); r = subprocess.run(c, shell=True); print('rc', r.returncode, flush=True); return r.returncode
T0 = time.time(); BUDGET = 10.8 * 3600
D = glob.glob('/kaggle/input/**/xphi_scripts.tgz', recursive=True)[0]; ROOT = os.path.dirname(D)
MANO = os.path.dirname(glob.glob('/kaggle/input/**/mano/MANO_RIGHT.pkl', recursive=True)[0])
CAL = glob.glob('/kaggle/input/**/calib_assumed_hfov90.yaml', recursive=True)[0]
W = '/kaggle/working'; R = '/kaggle/tmp/robot'; os.makedirs(R, exist_ok=True)
sh(f'nvidia-smi -L; tar xzf {D} -C {R}')
sh('git clone -q --depth 1 https://github.com/rolpotamias/WiLoR /kaggle/tmp/WiLoR')
for pk in ['ultralytics==8.1.34', 'pytorch-lightning', 'smplx==0.1.28', 'yacs', 'timm', 'einops', 'pyrootutils', 'hydra-core', 'scikit-image', 'trimesh', 'xtcocotools']:
    if sh(f'pip install -q {pk} 2>&1 | tail -4') and pk.startswith('ultralytics'): sh('pip install -q ultralytics 2>&1 | tail -3')
sh('pip install -q --no-build-isolation git+https://github.com/mattloper/chumpy 2>&1 | tail -2')
ck = '/kaggle/tmp/wck'; os.makedirs(ck, exist_ok=True)
for f in ['detector.pt', 'wilor_final.ckpt', 'model_config.yaml']:
    sh(f'curl -sL -o {ck}/{f} https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/{f}')
mano = '/kaggle/tmp/mano'; os.makedirs(mano, exist_ok=True)
sh(f'cp {MANO}/*.pkl {mano}/ && cp /kaggle/tmp/WiLoR/mano_data/mano_mean_params.npz {mano}/')
os.environ.update(WILOR_CHECKPOINT=f'{ck}/wilor_final.ckpt', WILOR_DETECTOR=f'{ck}/detector.pt',
                  WILOR_CONFIG=f'{ck}/model_config.yaml', MANO_MODEL_DIR=mano,
                  PYTHONPATH=f'/kaggle/tmp/WiLoR:{R}/scripts')
sh(f'cd {R} && python -c "from headcam.hand_pose import get_backend; print(\'BACKEND_OK\', get_backend(\'wilor\'))" 2>&1 | grep -v -i warn | tail -5')
order = json.load(open(glob.glob('/kaggle/input/**/xphi_order.json', recursive=True)[0]))  # round-robin over tasks
TAR = glob.glob('/kaggle/input/**/xphi_vids.tar', recursive=True)[0]; sh(f'tar xf {TAR} -C /kaggle/tmp'); VD = '/kaggle/tmp/xphi_vids'
for name in order:
    if time.time() - T0 > BUDGET: print('BUDGET_STOP', name, flush=True); break
    o = f'{W}/xphi/{name}'; v = f'/kaggle/tmp/v/{name}.mp4'; os.makedirs('/kaggle/tmp/v', exist_ok=True)
    shutil.copy(f'{VD}/{name}/rgb.mp4', v)
    sh(f'cd {R} && python scripts/wilor_video_to_episodes.py --videos {v} --out {o} --calib {CAL} --stride 1 --max-frames 1800 '
       f'--source xphi --license-note "xPhi-Ego-100 Apache-2.0; intrinsics assumed HFOV90" > {W}/log_{name}.txt 2>&1; tail -3 {W}/log_{name}.txt')
    os.remove(v); shutil.rmtree(f'{VD}/{name}', ignore_errors=True); print('CLIP', name, round(time.time() - T0), flush=True)
sh(f'ls {W}/xphi | wc -l; du -sh {W}/xphi')
print('ALL_DONE', round(time.time() - T0))
