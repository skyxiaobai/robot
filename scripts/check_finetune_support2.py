#!/usr/bin/env python3
"""Check LeRobot train CLI args, resume/pretrained support, and robomimic->lerobot conversion tools."""
import os, re, glob

from repo_root import resolve_root

ROOT = resolve_root()

print("=== 1. lerobot/scripts/train.py CLI 参数 ===")
tp = os.path.join(ROOT, "lerobot/lerobot/scripts/train.py")
if os.path.exists(tp):
    txt = open(tp).read()
    print(" file found:", tp)
    # argparse-style: argparse.ArgumentParser / parser.add_argument("--x")
    for m in re.finditer(r'add_argument\(\s*"(--[\w-]+)"', txt):
        print("   arg:", m.group(1))
    # check for 'pretrained_path' references
    for kw in ("pretrained_path", "resume", "--resume", "hydra", "parse_args"):
        n = txt.count(kw)
        print(f"   contains '{kw}': {n}")
else:
    print(" NOT FOUND:", tp)

print()
print("=== 2. ACT policy 是否支持 pretrained_path 配置 ===")
acfg = os.path.join(ROOT, "lerobot/lerobot/policies/act/config.py")
if os.path.exists(acfg):
    txt = open(acfg).read()
    for kw in ("pretrained_path", "resume_training", "freeze", "vision_backbone", "backbone"):
        n = txt.count(kw)
        print(f"   '{kw}': {n}")

print()
print("=== 3. robomimic -> LeRobot 转换工具 ===")
for pat in ("**/convert*.py", "**/*convert*", "**/mimicgen_to_lerobot*", "**/*to_lerobot*"):
    for f in glob.glob(os.path.join(ROOT, "lerobot", pat), recursive=True)[:8]:
        print("  ", os.path.relpath(f, ROOT))
for f in glob.glob(os.path.join(ROOT, "lerobot", "**", "*.py"), recursive=True):
    try:
        txt = open(f, encoding="utf-8", errors="ignore").read()
    except Exception:
        continue
    if "robomimic" in txt and "convert" in txt.lower():
        print("  mentions robomimic+convert:", os.path.relpath(f, ROOT))

print()
print("=== 4. LeRobot 版本 ===")
for f in ("lerobot/pyproject.toml", "lerobot/setup.py"):
    fp = os.path.join(ROOT, f)
    if os.path.exists(fp):
        txt = open(fp).read()
        m = re.search(r"version\s*=\s*[\"']([^\"']+)[\"']", txt)
        if m:
            print("  ", f, "version:", m.group(1))

print()
print("=== 5. 现有 ACT checkpoint 目录（pretrained 候选） ===")
ck = os.path.join(ROOT, "outputs/checkpoints")
if os.path.isdir(ck):
    dirs = sorted(glob.glob(os.path.join(ck, "*")))
    print("  checkpoints:", [os.path.basename(d) for d in dirs][:40])

print()
print("=== 6. MimicGen 生成脚本的相机参数 ===")
ds = os.path.join(ROOT, "mimicgen/mimicgen/scripts/generate_dataset.py")
if os.path.exists(ds):
    txt = open(ds).read()
    for kw in ("--camera", "camera", "eye_in_hand", "agentview", "--interface-type", "single_task"):
        print(f"   '{kw}': {txt.count(kw)}")
