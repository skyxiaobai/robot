import subprocess, os, glob, time, json, shutil
def sh(c):
    print('+', c, flush=True); r = subprocess.run(c, shell=True); print('rc', r.returncode, flush=True); return r.returncode
T0 = time.time(); BUDGET = 11.0 * 3600
IN = [p for p in glob.glob('/kaggle/input/*') ]; print(IN)
D = glob.glob('/kaggle/input/**/scripts.tgz', recursive=True)[0]; ROOT = os.path.dirname(D)
W = '/kaggle/working'; R = '/kaggle/tmp/robot'; os.makedirs(R, exist_ok=True)
sh(f'nvidia-smi -L; tar xzf {D} -C {R}')
sh('git clone -q --depth 1 https://github.com/rolpotamias/WiLoR /kaggle/tmp/WiLoR')
for pk in ['ultralytics==8.1.34', 'pytorch-lightning', 'smplx==0.1.28', 'yacs', 'timm', 'einops', 'pyrootutils', 'hydra-core', 'scikit-image', 'trimesh', 'xtcocotools']:
    if sh(f'pip install -q {pk} 2>&1 | tail -4') and pk.startswith('ultralytics'): sh('pip install -q ultralytics 2>&1 | tail -3')
sh('python -c "import ultralytics;print(ultralytics.__version__)"; ls -la /kaggle/tmp/WiLoR /kaggle/tmp/WiLoR/wilor')
sh('pip install -q --no-build-isolation git+https://github.com/mattloper/chumpy 2>&1 | tail -2')
ck = '/kaggle/tmp/wck'; os.makedirs(ck, exist_ok=True)
for f in ['detector.pt', 'wilor_final.ckpt', 'model_config.yaml']:
    if not os.path.exists(f'{ck}/{f}'):
        sh(f'curl -sL -o {ck}/{f} https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/{f}')
mano = '/kaggle/tmp/mano'; os.makedirs(mano, exist_ok=True)
sh(f'cp {ROOT}/mano/*.pkl {mano}/ && cp /kaggle/tmp/WiLoR/mano_data/mano_mean_params.npz {mano}/')
os.environ.update(WILOR_CHECKPOINT=f'{ck}/wilor_final.ckpt', WILOR_DETECTOR=f'{ck}/detector.pt',
                  WILOR_CONFIG=f'{ck}/model_config.yaml', MANO_MODEL_DIR=mano,
                  PYTHONPATH=f'/kaggle/tmp/WiLoR:{R}/scripts')
sh(f'cd {R} && python -c "from headcam.hand_pose import wilor_missing as m; print(\'MISSING\', m())"')
sh(f'cd {R} && python -c "import traceback\ntry:\n from headcam.hand_pose import get_backend; b=get_backend(\'wilor\'); print(\'BACKEND_OK\', b)\nexcept BaseException as e:\n traceback.print_exc(); print(\'BACKEND_FAIL\', repr(e)[:3000])" 2>&1 | grep -v -i warn | tail -40')
# depth clips (iPhone RGB-D pipeline); copy sessions to writable place
ses = sorted(glob.glob(f'{ROOT}/depth/*/'))
loc = []
for s in ses:
    t = '/kaggle/tmp/ses/' + os.path.basename(s.rstrip('/')); shutil.copytree(s, t, dirs_exist_ok=True); loc.append(t)
if loc: sh(f'cd {R} && python scripts/run_stereo_pipeline.py --iphone {" ".join(loc)} --out {W}/run_s2 --backend wilor --no-export > {W}/logs_depth.txt 2>&1; grep -v -i warn {W}/logs_depth.txt | tail -40')
print('DEPTH_DONE_S', round(time.time()-T0), flush=True)
# non-depth clips -> WiLoR camera-frame episodes
for d in sorted(glob.glob(f'{ROOT}/nodepth/*/')):
    name = os.path.basename(d.rstrip('/')); o = f'{W}/xhey_rgb/{name}'
    if time.time() - T0 > BUDGET: print('BUDGET_STOP', name); break
    if glob.glob(o + '/*.json'): continue
    v = f'/kaggle/tmp/v/{name}.mp4'; os.makedirs('/kaggle/tmp/v', exist_ok=True); shutil.copy(d + 'rgb.mp4', v)
    sh(f'cd {R} && python scripts/wilor_video_to_episodes.py --videos {v} --out {o} --calib {d}calib.yaml --stride 1 --max-frames 1800 --source xhey_rgb --license-note own > {W}/logs_{name}.txt 2>&1; grep -v -i warn {W}/logs_{name}.txt | head -30; grep -v -i warn {W}/logs_{name}.txt | tail -15')
    print('CLIP', name, round(time.time()-T0), flush=True)
# gather nodepth episodes into one dir (unique names)
U = f'{W}/xhey_rgb_unified'; os.makedirs(U, exist_ok=True)
for j in glob.glob(f'{W}/xhey_rgb/*/*.json'):
    if os.path.basename(j) in ('index.json','manifest.json','convert_summary.json'): continue
    shutil.copy(j, f'{U}/' + os.path.basename(os.path.dirname(j)) + '_' + os.path.basename(j))
sh(f'ls {W}/run_s2/episodes; ls {U} | wc -l')
for tag, mixes in [('A', 'iphone,iphone+aug,iphone+xhey_rgb,iphone+xhey_rgb+aug')]:
    sh(f'cd {R} && python scripts/train_xhey.py --run {W}/run_s2 --out {W}/xhey --ablation --mixes {mixes} --mix iphone+xhey_rgb+aug --xhey-rgb {U} --seeds 3 2>&1 | grep -v -i warn | tail -60')
sh(f'find {W} -name "*.mp4" -delete; du -sh {W}/*')
print('ALL_DONE', round(time.time()-T0))
