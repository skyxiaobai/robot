#!/usr/bin/env python3
"""Find ACT config.py and check lerobot_train.py CLI for fine-tuning args."""
import os, re, glob

ROOT = "/mnt/sda/app/robot"

print("=== 1. ACT policy config 位置 ===")
for pat in ("lerobot/src/lerobot/policies/act/*.py", "lerobot/lerobot/policies/act/*.py"):
    for f in glob.glob(os.path.join(ROOT, pat)):
        print("  ", os.path.relpath(f, ROOT))
        if f.endswith("config.py"):
            txt = open(f).read()
            for kw in ("pretrained_path", "resume_training", "freeze", "vision_backbone", "use_imagenet_stats", "unfreeze"):
                print(f"     '{kw}': {txt.count(kw)}")

print()
print("=== 2. lerobot_train.py CLI 参数 ===")
tp = os.path.join(ROOT, "lerobot/src/lerobot/scripts/lerobot_train.py")
if os.path.exists(tp):
    txt = open(tp).read()
    for m in re.finditer(r'add_argument\(\s*"(--[\w-]+)"', txt):
        print("   arg:", m.group(1))
    for kw in ("pretrained_path", "resume", "--resume"):
        print(f"   contains '{kw}': {txt.count(kw)}")
else:
    print("   NOT FOUND:", tp)

print()
print("=== 3. ACT policy.py 里 pretrained 加载逻辑 ===")
for f in glob.glob(os.path.join(ROOT, "leroder/src/lerobot/policies/act/policy.py")) + glob.glob(os.path.join(ROOT, "lerobot/src/lerobot/policies/act/policy.py")):
    if os.path.exists(f):
        txt = open(f).read()
        for kw in ("pretrained_path", "load_state_dict", "from_pretrained", "resume"):
            n = txt.count(kw)
            if n:
                print(f"   {os.path.basename(f)} '{kw}': {n}")
        break
