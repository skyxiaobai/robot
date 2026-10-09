#!/usr/bin/env python3
"""Check LeRobot finetune support + pretrained_path + MimicGen camera options."""
import json, os, re, glob

from repo_root import resolve_root

ROOT = resolve_root()

print("=== 1. 现有 config 的 pretrained_path / resume ===")
for p in sorted(glob.glob(os.path.join(ROOT, "outputs/checkpoints/*/pretrained_model/train_config.json"))):
    with open(p) as f:
        c = json.load(f)
    if c.get("pretrained_path") or c.get("resume"):
        print(" ", os.path.relpath(p, ROOT))
        print("    pretrained_path:", c.get("pretrained_path"))
        print("    resume:", c.get("resume"))
        break

print()
print("=== 2. LeRobot train 脚本参数（resume / pretrained / policy） ===")
for cand in (
    os.path.join(ROOT, "lerobot/lerobot/scripts/train.py"),
    os.path.join(ROOT, "lerobot/examples/7_get_started_with_act.py"),
):
    if os.path.exists(cand):
        txt = open(cand).read()
        print(" file:", os.path.relpath(cand, ROOT))
        for m in re.finditer(r"(--[\w-]+)", txt):
            print("   arg:", m.group(1))
        break

print()
print("=== 3. MimicGen 相机配置支持（能否自定义视角） ===")
mg = os.path.join(ROOT, "mimicgen")
for pat in ("**/utils/*.py", "**/envs/**/*.py", "**/mimicgen/**/*.py"):
    for f in glob.glob(os.path.join(mg, pat), recursive=True)[:0]:
        pass
hits = []
for f in glob.glob(os.path.join(mg, "**/*.py"), recursive=True):
    try:
        txt = open(f, encoding="utf-8", errors="ignore").read()
    except Exception:
        continue
    if "camera_names" in txt or "camera_configs" in txt or "eye_in_hand" in txt:
        hits.append(os.path.relpath(f, ROOT))
print(" files mentioning camera config:", sorted(set(hits))[:12])

print()
print("=== 4. 现有数据/目录（Square、robosuite 环境、core configs） ===")
for d in ("/tmp/core_datasets", "/tmp/core_train_configs"):
    if os.path.isdir(d):
        print(" ", d, "->", sorted(os.listdir(d))[:10])
for f in glob.glob(os.path.join(ROOT, "outputs/**/*.json"), recursive=True)[:5]:
    print("  out json:", os.path.relpath(f, ROOT))

print()
print("=== 5. 论文 2607.24744 关键章节（两段式/合成数据/微调相关） ===")
pdf_txt = "/tmp/paper_2607.txt"
if os.path.exists(pdf_txt):
    txt = open(pdf_txt).read()
    for kw in ("synthetic", "fine-tun", "finetun", "pretrain", "pre-train", "sim-to-real", "egocentric"):
        idxs = [m.start() for m in re.finditer(kw, txt, re.I)][:3]
        print(f"  '{kw}': {len(re.findall(kw, txt, re.I))} hits; samples:", [txt[max(0,i-60):i+90].replace(chr(10),' ')[:150] for i in idxs])
else:
    print("  paper txt not extracted (ok)")
