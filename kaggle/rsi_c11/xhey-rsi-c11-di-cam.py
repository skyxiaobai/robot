import subprocess, os, glob, json, time, zipfile, shlex
def sh(c):
    print('+', c, flush=True); r = subprocess.run(c, shell=True); print('rc', r.returncode, flush=True); return r.returncode
def find(pat, root='/kaggle/input'):
    g = sorted(glob.glob(root.rstrip('/') + '/**/' + pat, recursive=True))
    return g[0] if g else None
def real_file(p):
    # v1 did f'--qc-filter {None}' and train_xhey opened the literal path "None".
    if not isinstance(p, str):
        return None
    s = p.strip()
    if not s or s.lower() == 'none' or not os.path.isfile(s):
        return None
    return s
def real_path(p):
    if not isinstance(p, str):
        return None
    s = p.strip()
    if not s or s.lower() == 'none' or not os.path.exists(s):
        return None
    return s
def locate_di_scorecards(root='/kaggle/input', tmp='/kaggle/tmp/qc_di'):
    # Only named DI cards. A bare scorecards.json from another input (iPhone, xhey_rgb)
    # is a different QC file and must not be passed as the DI filter.
    for pat in ('qc_di/scorecards.json', 'di_scorecards.json', 'qc/di/scorecards.json'):
        hit = real_file(find(pat, root))
        if hit:
            return hit
    z = real_file(find('qc_di.zip', root))
    if not z:
        return None
    os.makedirs(tmp, exist_ok=True)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(tmp)
    for pat in ('qc_di/scorecards.json', 'di_scorecards.json', 'qc/di/scorecards.json', 'scorecards.json'):
        for h in sorted(glob.glob(tmp.rstrip('/') + '/**/' + pat, recursive=True)):
            hit = real_file(h)
            if hit:
                return hit
    return None
def di_arm_extra(di, qc):
    di, qc = real_path(di), real_file(qc)
    if not qc:
        raise SystemExit(
            'missing DI scorecards (di_scorecards.json / qc_di/scorecards.json / '
            'qc/di/scorecards.json / qc_di.zip). Refusing to pass None or an empty path to --qc-filter.')
    if not di:
        raise SystemExit('missing di_unified (refusing to pass None or an empty path to --di)')
    return f'--di {shlex.quote(di)} --qc-filter {shlex.quote(qc)}'
def main():
    SCR = find('xhey-c11-di-cam/**/scripts.tgz') or find('scripts.tgz')
    QC = locate_di_scorecards(); DI = find('di_unified')
    print(SCR, QC, DI, flush=True)
    extra_di = di_arm_extra(DI, QC)  # fail before unpacking if the DI cards are absent
    T0 = time.time(); W = '/kaggle/working'; R = '/kaggle/tmp/robot'; os.makedirs(R, exist_ok=True)
    SES = os.path.dirname(find('xhey-iphone-sessions/**/depth') or find('depth/2026-08-24_21-18'))
    LAB = os.path.dirname(find('run_s2/episodes')); XR = find('xhey_rgb_unified')
    print(SES, LAB, XR, flush=True)
    sh(f'nvidia-smi -L; tar xzf {SCR} -C {R}')
    run = '/kaggle/tmp/run_s2'; os.makedirs(run + '/episodes', exist_ok=True)
    for p in glob.glob(LAB + '/episodes/*.json'):
        e = json.load(open(p)); n = os.path.basename(p)[:-5]
        e['video_path'] = f'{SES}/depth/{n}/rgb.mp4'; json.dump(e, open(f'{run}/episodes/{n}.json', 'w'))
    os.environ['PYTHONPATH'] = f'{R}/scripts'
    # C11: DI (iPhone LiDAR, camera_yup -> camera optical via frames.canonicalize_for_xhey; QC 25/25) pretrain, camera frame.
    # Same protocol as C10: LOO over the 3 depth clips, 3 seeds, pre 6000, ft 3000, move-cm 5, --no-final.
    # Arm A iphone+di+xhey_rgb+aug, then same-code control arm B iphone+xhey_rgb+aug.
    common = (f'--run {run} --xhey-rgb {XR} --seeds 3 --pre-steps 6000 --steps 3000 --move-cm 5 --no-final')
    for tag, mix, extra in (('c11', 'iphone+di+xhey_rgb+aug', extra_di),
                            ('c11ctl', 'iphone+xhey_rgb+aug', '')):
        out = f'{W}/xhey_{tag}'
        sh(f'cd {R} && python scripts/train_xhey.py {common} --ablation --mixes {mix} --mix {mix} --out {out} {extra} '
           f'> {W}/log_{tag}.txt 2>&1; grep -v -i warn {W}/log_{tag}.txt | tail -40')
        sh(f'rm -rf {out}/cache')
        print(tag.upper() + '_DONE', round(time.time() - T0), flush=True)
if __name__ == '__main__':
    main()
