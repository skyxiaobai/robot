#!/usr/bin/env python3
"""Collect local evidence for the two-stage (synthetic pretrain -> real finetune) evaluation."""
import json, glob, os

from repo_root import dataset_fps, resolve_root

ROOT = resolve_root()
OUT = []
def p(*msg):
    line = " ".join(str(m) for m in msg)
    OUT.append(line)
    print(line)

p("=== 1. headcam_data_spec.md 章节结构 ===")
spec = os.path.join(ROOT, "docs/headcam_data_spec.md")
if os.path.exists(spec):
    with open(spec) as f:
        lines = f.readlines()
    p("lines:", len(lines))
    for i, ln in enumerate(lines, 1):
        if ln.startswith("#") and not ln.startswith("##") or ln.startswith("## "):
            p(f"  {i}: {ln.rstrip()[:90]}")
else:
    p("MISSING")

p()
p("=== 2. 训练/评估/生成脚本 ===")
for d in ("scripts", "examples"):
    dd = os.path.join(ROOT, d)
    if os.path.isdir(dd):
        p(f"{d}/:", sorted(os.listdir(dd)))

p()
p("=== 3. outputs/checkpoints 训练配置 ===")
cfgs = sorted(glob.glob(os.path.join(ROOT, "outputs/checkpoints/*/pretrained_model/train_config.json")))
p("train_config count:", len(cfgs))
if cfgs:
    with open(cfgs[-1]) as f:
        c = json.load(f)
    p("last cfg keys:", sorted(c.keys())[:40])
    p("  steps:", c.get("steps"), "| batch_size:", c.get("batch_size"), "| lr:", c.get("learning_rate") or c.get("optimizer", {}).get("learning_rate"))
    p("  dataset:", json.dumps(c.get("dataset", {}), ensure_ascii=False)[:300])
    p("  policy:", json.dumps(c.get("policy", {}), ensure_ascii=False)[:400])

p()
p("=== 4. data/pusht meta ===")
pi = os.path.join(ROOT, "data/pusht/meta/info.json")
if os.path.exists(pi):
    with open(pi, encoding="utf-8") as f:
        info = json.load(f)
    p("  repo_id:", info.get("repo_id"), "| total_episodes:", info.get("total_episodes"))
    p("  fps:", dataset_fps(info))
    p("  features:", sorted(info.get("features", {}).keys()))
    for k, v in list(info.get("features", {}).items())[:6]:
        p("   ", k, v.get("shape"))
else:
    p("MISSING pusht info")

p()
p("=== 5. Square 数据集（找 demo.hdf5） ===")
cands = [
    "/tmp/core_datasets/square/demo_src_square_task_D1/demo.hdf5",
]
for c in cands:
    if os.path.exists(c):
        try:
            import h5py
            f = h5py.File(c, "r")
            data = f["data"]
            names = list(data.keys())
            p("  found:", c, "| demos:", len(names))
            d0 = data[names[0]]
            p("  first demo:", names[0], "| action shape:", d0["actions"].shape, "| obs keys:", sorted(d0["obs"].keys()) if "obs" in d0 else "n/a")
            p("  attrs:", {k: v for k, v in list(data.attrs.items())[:8]})
        except Exception as e:
            p("  h5py error:", e)
        break
else:
    p("  demo.hdf5 not at expected path; searching shallow...")
    for base in ("/tmp/core_datasets", os.path.join(ROOT, "data")):
        for f in glob.glob(os.path.join(base, "**/demo.hdf5"), recursive=True)[:3]:
            p("   found:", f)

p()
p("=== 6. MimicGen 生成日志 ===")
for lg in ("outputs/mimicgen_gen.log", "outputs/mimicgen_gen_official.log"):
    lp = os.path.join(ROOT, lg)
    if os.path.exists(lp):
        with open(lp) as f:
            txt = f.read()
        p(lg, "len:", len(txt))
        import re
        m = re.search(r"Final Data Generation Stats\s*\n(\{.*?\})", txt, re.S)
        p("  final stats:", m.group(1)[:500] if m else "not found")

p()
p("=== 7. LeRobot ACT 相关脚本/环境 ===")
for f in glob.glob(os.path.join(ROOT, "lerobot/examples/**/*act*"), recursive=True)[:6]:
    p("  lerobot:", os.path.relpath(f, ROOT))
for f in glob.glob(os.path.join(ROOT, "lerobot/lerobot/scripts/*.py")):
    p("  lerobot script:", os.path.basename(f))

p()
p("=== 8. 官方复现相关 /tmp 日志（重启后可能被清） ===")
for lg in ("/tmp/train_official.log", "/tmp/train_official_200k.log"):
    p("  ", lg, os.path.exists(lg))

with open(os.path.join(ROOT, "outputs/twostage_evidence.txt"), "w") as f:
    f.write("\n".join(OUT))
p()
p("saved -> outputs/twostage_evidence.txt")
