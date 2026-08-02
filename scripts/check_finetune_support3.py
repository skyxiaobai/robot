#!/usr/bin/env python3
"""Check LeRobot src layout: train.py CLI, ACT config, MimicGen cameras."""
import os, re, glob

ROOT = "/mnt/sda/app/robot"

print("=== 1. lerobot/src/lerobot/scripts/train.py ===")
tp = os.path.join(ROOT, "lerobot/src/lerobot/scripts/train.py")
if os.path.exists(tp):
    txt = open(tp).read()
    for m in re.finditer(r'add_argument\(\s*"(--[\w-]+)"', txt):
        print("   arg:", m.group(1))
    for kw in ("pretrained_path", "resume", "--resume"):
        print(f"   contains '{kw}': {txt.count(kw)}")
    # show the training entrypoint main signature hints
else:
    print("   NOT FOUND")
    for f in glob.glob(os.path.join(ROOT, "lerobot/src/lerobot/scripts/*.py")):
        print("   scripts:", os.path.basename(f))

print()
print("=== 2. ACT config: pretrained_path / fine-tune hooks ===")
acfg = os.path.join(ROOT, "lerobot/src/lerobot/policies/act/config.py")
if os.path.exists(acfg):
    txt = open(acfg).read()
    for kw in ("pretrained_path", "resume_training", "freeze", "vision_backbone", "use_imagenet_stats", "unet"):
        print(f"   '{kw}': {txt.count(kw)}")
else:
    print("   NOT FOUND:", acfg)

print()
print("=== 3. MimicGen 相机名（robosuite 标准） ===")
for f in glob.glob(os.path.join(ROOT, "mimicgen/mimicgen/**/*.py"), recursive=True):
    try:
        txt = open(f, encoding="utf-8", errors="ignore").read()
    except Exception:
        continue
    for kw in ("eye_in_hand", "agentview", "sideview", "frontview", "topview"):
        if kw in txt:
            print("   ", os.path.relpath(f, ROOT), "->", kw)

print()
print("=== 4. robosuite env 支持的相机（sensor） ===")
rob = os.path.join(ROOT, "robosuite")
if os.path.isdir(rob):
    hits = set()
    for f in glob.glob(os.path.join(rob, "robosuite/**/*.py"), recursive=True)[:400]:
        try:
            txt = open(f, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        for kw in ("camera_names", "eye_in_hand", "agentview"):
            if kw in txt:
                hits.add(kw)
    print("   robosuite camera kw hits:", sorted(hits))
    # list sensors dir
    sd = os.path.join(rob, "robosuite/sensors")
    if os.path.isdir(sd):
        print("   sensors:", sorted(os.listdir(sd))[:20])

print()
print("=== 5. data/pusht meta info.json 完整 features dtype ===")
info = json_load(os.path.join(ROOT, "data/pusht/meta/info.json"))
if info:
    for k, v in list(info.get("features", {}).items()):
        print("   ", k, "->", v.get("dtype"), v.get("shape"), "| fps:", v.get("fps"))

print()
print("=== 6. ACT 训练脚本实际调用（train.log 所在 scripts） ===")
for f in glob.glob(os.path.join(ROOT, "scripts/*.py")):
    print("   ", os.path.basename(f))
