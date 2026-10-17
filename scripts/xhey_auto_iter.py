#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""XHEY 自动迭代（唯一循环驱动）：固定候选 → 提议 → Kaggle → 记账 → 镜像 CSV → 画图。

    python scripts/xhey_auto_iter.py step        # 主命令：幂等
    python scripts/xhey_auto_iter.py propose     # 只看下一轮会先交哪些固定候选、再提哪些配置
    python scripts/xhey_auto_iter.py chart       # 只重画进度图
    python scripts/xhey_auto_iter.py mirror      # 把 ledger.jsonl 镜像进 docs/xhey/ledger.csv
    python scripts/xhey_auto_iter.py ingest-c1   # 把已完成的 justinz2010/xhey-rsi-c1-moveloss 写入账本

step 的逻辑：
  1. 收已完成的固定内核（C1，queue.json 里跑完的 slug，以及 auto_state 里
     queue_running——单个对象或列表——和 queue_running_also），写入 ledger.jsonl 并镜像 CSV。
  2. experiments/auto_state.json 里的自动迭代内核还在 RUNNING/QUEUED → 不提交任何新内核。
  3. 上一轮自动迭代跑完（COMPLETE）→ 下载、记账、镜像、重画。跑挂（ERROR）→ 记成 failed。
  4. 有空闲 GPU 时，先提交 docs/xhey/queue.json 里就绪的固定候选（同时最多 1 个）：
     只要还有正在跑的队列内核（含 queue_running 列表和 queue_running_also），就不提交新内核。
     C2、C8；C7/C9 仅在 Stera 数据已存在时。内核脚本没有 --qc-filter 的就绪候选不提交
     （已经在跑的照常入账），队列里还有这种候选时也不交给提议器。队列没清空之前不跑提议器。
  5. 队列空了才提议：岭回归代理 + 探索奖励，去掉已跑过的，挑前 K 个（默认 3），
     推送到 justinz2010/xhey-auto-iter。

需要：kaggle CLI 已登录（KAGGLE_API_TOKEN 或 ~/.kaggle）；私有数据集
justinz2010/xhey-iphone-sessions、justinz2010/xhey-openaoe-pretrain；内核 justinz2010/xhey-wilor-labels 的输出。
"""
import argparse, base64, csv, io, itertools, json, subprocess, sys, tarfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
LEDGER, STATE, CHART = EXP / "ledger.jsonl", EXP / "auto_state.json", EXP / "progress.png"
CSV_PATH = ROOT / "docs" / "xhey" / "ledger.csv"
QUEUE_PATH = ROOT / "docs" / "xhey" / "queue.json"
KERNEL = "justinz2010/xhey-auto-iter"
DATASETS = ["justinz2010/xhey-openaoe-pretrain", "justinz2010/xhey-iphone-sessions"]
KERNEL_SOURCES = ["justinz2010/xhey-wilor-labels"]
STERA_DATASET = "justinz2010/xhey-stera-pretrain"
STERA_LOCAL = ROOT / "data" / "stera" / "unified"
CSV_FIELDS = ["id", "date", "mix", "overall_ade", "moving_ade", "vs_zero", "status", "source"]
INFLIGHT_STATUS = {"running", "running_kaggle", "submitted"}
# queue.json 里的内核各自的训练配比（与 kaggle/rsi_c*/run.py 一致）。C1 单独记在 C1_SPEC。
QUEUE_CONFIG = {
    "C2": {"mix": "iphone+xhey_rgb+aug", "steps": 10000},
    "C7": {"mix": "iphone+stera+xhey_rgb+aug"},
    "C8": {"mix": "iphone+di+xhey_rgb+aug"},
    "C9": {"mix": "iphone+stera+di+xhey_rgb+aug"},
    "C9b": {"mix": "iphone+stera+di+xhey_rgb+aug"},
    "C11": {"mix": "iphone+di+xhey_rgb+aug"},
    "C12": {"mix": "iphone+hoi4d+xhey_rgb+aug"},
    "C13": {"mix": "iphone+egodex+xhey_rgb+aug"},
    "C14": {"mix": "iphone+xphi+xhey_rgb+aug"},
}

DEFAULT = {"mix": "iphone+xhey_rgb+aug", "openaoe": None, "pre_steps": 6000, "steps": 3000,
           "hidden": 512, "dropout": 0.2, "lr_scale": 1.0, "aug_noise": 0.1, "loss_move_cm": 0.0, "seeds": 3}
SPACE = {"mix": ["iphone+xhey_rgb+aug", "iphone+openaoe+xhey_rgb+aug", "iphone+openaoe+aug"],
         "openaoe": ["nano", "nano_cam", "tiny"], "pre_steps": [6000, 20000],
         "hidden": [256, 512, 1024], "dropout": [0.1, 0.2, 0.3], "lr_scale": [0.5, 1.0, 2.0],
         "aug_noise": [0.05, 0.1, 0.2], "loss_move_cm": [0.0, 5.0]}
# 第一轮固定先试：相机系 Open-AoE（避开世界系没有重力方向的问题）
PINNED = [dict(DEFAULT, mix="iphone+openaoe+xhey_rgb+aug", openaoe="nano_cam"),
          dict(DEFAULT, mix="iphone+openaoe+aug", openaoe="nano_cam")]
# 已完成、需要收进账本的固定内核（不在 queue.json 里）
C1_SPEC = {"id": "C1", "slug": "justinz2010/xhey-rsi-c1-moveloss", "mix": "iphone+xhey_rgb+aug",
           "config": dict(DEFAULT, mix="iphone+xhey_rgb+aug", loss_move_cm=5.0)}


def norm_cfg(c):
    c = dict(DEFAULT, **{k: v for k, v in c.items() if k in DEFAULT})
    if "openaoe" not in (c.get("mix") or ""):
        c["openaoe"] = None
    return c


def key(c):
    c = norm_cfg(c)
    return json.dumps({k: c[k] for k in sorted(c) if k != "seeds"}, sort_keys=True)


def load_ledger():
    if not LEDGER.exists():
        return []
    return [json.loads(l) for l in LEDGER.read_text(encoding="utf-8").splitlines() if l.strip()]


def append_ledger(rows):
    EXP.mkdir(exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    mirror_ledger_csv()


def row_from_metrics(m, mix, cfg, run_id, iteration, source):
    r, z = m["results"][mix], m["results"]["zero"]
    return {"id": run_id, "iteration": iteration, "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "source": source, "config": norm_cfg(dict(cfg, mix=mix)), "status": "ok",
            "overall_ade_cm": r["ade_cm_pooled"], "zero_overall_ade_cm": z["ade_cm_pooled"],
            "delta_overall_cm": r["ade_minus_zero_cm"], "delta_overall_ci95": r.get("ade_minus_zero_ci95"),
            "moving_ade_cm": r.get("moving_ade_cm"), "zero_moving_ade_cm": r.get("moving_zero_ade_cm"),
            "delta_moving_cm": r.get("moving_minus_zero_cm"), "delta_moving_ci95": r.get("moving_minus_zero_ci95"),
            "seed_std_cm": r.get("ade_cm_seed_std")}


def _fmt(x):
    if x is None or x == "":
        return ""
    if isinstance(x, float):
        return "%.4f" % x
    return str(x)


def mix_label(cfg):
    cfg = cfg or {}
    mix = cfg.get("mix") or ""
    oa = cfg.get("openaoe")
    if oa and "openaoe" in mix:
        mix = mix.replace("openaoe", "openaoe_%s" % oa, 1)
    bits = [mix]
    if cfg.get("loss_move_cm"):
        bits.append("loss_move_cm=%s" % cfg["loss_move_cm"])
    if cfg.get("steps") not in (None, 3000):
        bits.append("steps=%s" % cfg["steps"])
    if cfg.get("hidden") not in (None, 512):
        bits.append("hidden=%s" % cfg["hidden"])
    if cfg.get("dropout") not in (None, 0.2):
        bits.append("dropout=%s" % cfg["dropout"])
    if cfg.get("lr_scale") not in (None, 1.0):
        bits.append("lr_scale=%s" % cfg["lr_scale"])
    if cfg.get("aug_noise") not in (None, 0.1):
        bits.append("aug_noise=%s" % cfg["aug_noise"])
    if cfg.get("pre_steps") not in (None, 6000):
        bits.append("pre_steps=%s" % cfg["pre_steps"])
    return " ".join(b for b in bits if b)


def csv_status_for(row):
    st = row.get("status") or ""
    if st != "ok":
        return st
    ci = row.get("delta_moving_ci95") or [None, None]
    d = row.get("delta_moving_cm")
    if d is not None and ci[0] is not None and ci[1] is not None and ci[1] < 0:
        return "win_moving"
    if d is not None and d < 0:
        return "better_moving"
    overall = row.get("delta_overall_cm")
    if overall is not None and overall > 0:
        return "done_worse"
    return "ok"


def jsonl_to_csv(row):
    cfg = row.get("config") or {}
    date = (row.get("date") or "").split(" ")[0]
    return {"id": row["id"], "date": date, "mix": mix_label(cfg),
            "overall_ade": _fmt(row.get("overall_ade_cm")), "moving_ade": _fmt(row.get("moving_ade_cm")),
            "vs_zero": _fmt(row.get("delta_overall_cm")), "status": csv_status_for(row),
            "source": row.get("source") or ""}


def mirror_ledger_csv(rows=None):
    """把 ledger.jsonl 的每一行镜像进 docs/xhey/ledger.csv。同 id 更新，没有的追加，手写行保留。"""
    rows = load_ledger() if rows is None else rows
    existing = []
    if CSV_PATH.exists():
        with CSV_PATH.open(newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
    index = {r["id"]: i for i, r in enumerate(existing)}
    for row in rows:
        csv_row = jsonl_to_csv(row)
        if row["id"] in index:
            existing[index[row["id"]]] = csv_row
        else:
            existing.append(csv_row)
            index[row["id"]] = len(existing) - 1
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\r\n")
        w.writeheader()
        for r in existing:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})
    return len(rows)


# ---------------- 队列（固定候选，先于提议器） ----------------
def load_queue():
    if not QUEUE_PATH.exists():
        return {"note": "", "queue": []}
    return json.loads(QUEUE_PATH.read_text(encoding="utf-8"))


def save_queue(data):
    QUEUE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stera_data_exists():
    """Stera 预训练数据是否已经能给 C7/C9 用：本地 unified 有 episode，或 Kaggle 私有数据集在。"""
    if STERA_LOCAL.exists() and any(STERA_LOCAL.rglob("*.json")):
        return True
    try:
        out = kaggle("datasets", "status", STERA_DATASET, check=False)
    except SystemExit:
        return False
    text = out.lower()
    if "not found" in text or "404" in text or "does not exist" in text or "error" in text:
        return False
    return "ready" in text or "stera" in text or bool(out.strip())


def _needs_stera(item):
    return item.get("id") in ("C7", "C9") or "stera" in (item.get("slug") or "")


def _closed(status):
    st = status or ""
    return st in INFLIGHT_STATUS or st.startswith("done") or st.startswith("failed")


def kernel_has_qc_filter(item):
    """内核脚本的代码（去掉行内注释）里要有 --qc-filter，才允许新提交。

    只挡提交。status 已经是 running 的内核走入账，不经过这里，所以 C9 v1 这种
    未质检对照仍会记进账本。
    """
    path = ROOT / (item.get("dir") or "")
    meta = path / "kernel-metadata.json"
    name = "run.py"
    if meta.is_file():
        try:
            name = json.loads(meta.read_text(encoding="utf-8")).get("code_file") or name
        except json.JSONDecodeError:
            return False
    script = path / name
    if not script.is_file():
        return False
    for line in script.read_text(encoding="utf-8").splitlines():
        if "--qc-filter" in line.split("#", 1)[0]:
            return True
    return False


def split_qc_ready(items):
    """把就绪候选分成「脚本里带了 --qc-filter」和「没有，不能提交」。"""
    ok, skipped = [], []
    for item in items:
        (ok if kernel_has_qc_filter(item) else skipped).append(item)
    return ok, skipped


def ready_queue_items(stera_exists=None, queue=None):
    """按 queue.json 顺序返回可以提交的固定候选。C2、C8 为 ready；C7/C9 要等 Stera 数据。"""
    if stera_exists is None:
        stera_exists = stera_data_exists()
    elif not isinstance(stera_exists, bool):
        stera_exists = bool(stera_exists())
    data = queue if queue is not None else load_queue()
    # 不需要 Stera 的就绪项在前（C2、C8），Stera 项在后（C7、C9，且仅当数据已存在或被标成 ready）。
    normal, stera = [], []
    for item in data.get("queue", []):
        st = item.get("status") or ""
        if _closed(st):
            continue
        needs = _needs_stera(item)
        if st == "ready":
            (stera if needs else normal).append(item)
        elif needs and st == "blocked" and stera_exists:
            stera.append(item)
    return normal + stera


def set_queue_status(item_id, status):
    data = load_queue()
    for item in data.get("queue", []):
        if item.get("id") == item_id:
            item["status"] = status
            if status != "blocked":
                item.pop("blocked_by", None)
    save_queue(data)


# ---------------- 提议 ----------------
def features(c):
    c = norm_cfg(c); f = [1.0]
    for k, vals in SPACE.items():
        f += [1.0 if c[k] == v else 0.0 for v in vals]
    return f


def propose(k=3, ledger=None):
    import numpy as np
    ledger = load_ledger() if ledger is None else ledger
    done = {key(r["config"]) for r in ledger if r.get("config")}
    ok = [r for r in ledger if r.get("status") == "ok" and r.get("delta_moving_cm") is not None]
    picks = [c for c in PINNED if key(c) not in done][:k]
    # 候选：从默认出发，一次改 1~2 个维度（局部搜索，便宜）
    best = min(ok, key=lambda r: r["delta_moving_cm"])["config"] if ok else DEFAULT
    cands = {}
    dims = list(SPACE)
    for d1, d2 in itertools.chain(((d, None) for d in dims), itertools.combinations(dims, 2)):
        for v1 in SPACE[d1]:
            for v2 in (SPACE[d2] if d2 else [None]):
                c = dict(norm_cfg(best)); c[d1] = v1
                if d2: c[d2] = v2
                if "openaoe" in c["mix"] and not c.get("openaoe"):
                    c["openaoe"] = "nano_cam"
                c = norm_cfg(c); kk = key(c)
                if kk not in done and kk not in {key(p) for p in picks}:
                    cands[kk] = c
    cands = list(cands.values())
    if not cands:
        return picks
    X = np.array([features(r["config"]) for r in ok]) if ok else np.zeros((0, len(features(DEFAULT))))
    y = np.array([r["delta_moving_cm"] for r in ok])
    Xc = np.array([features(c) for c in cands])
    if len(ok) >= 2:
        lam = 1.0
        w = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
        pred = Xc @ w
        cnt = X[:, 1:].sum(0)  # 每个取值出现次数 → 越少探索奖励越大
        bonus = 0.02 * (Xc[:, 1:] / np.sqrt(1 + cnt)).sum(1)
    else:
        pred, bonus = np.zeros(len(cands)), np.zeros(len(cands))
    score = pred - bonus
    order = np.argsort(score)
    for i in order:
        if len(picks) >= k:
            break
        c = dict(cands[i]); c["_pred_delta_moving_cm"] = round(float(pred[i]), 4)
        picks.append(c)
    return picks


# ---------------- Kaggle ----------------
def kaggle(*args, check=True):
    try:
        r = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    except FileNotFoundError as e:
        raise SystemExit("找不到 kaggle 命令（需要已登录的 Kaggle CLI）") from e
    if check and r.returncode:
        raise SystemExit("kaggle %s 失败：%s" % (" ".join(args), (r.stderr or r.stdout)[-500:]))
    return (r.stdout or "") + (r.stderr or "")


def kernel_status(slug=None):
    out = kaggle("kernels", "status", slug or KERNEL, check=False)
    for s in ("RUNNING", "QUEUED", "COMPLETE", "ERROR", "CANCEL"):
        if s in out.upper():
            return s
    return "UNKNOWN:" + out.strip()[-200:]


def scripts_b64():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        t.add(ROOT / "scripts", arcname="scripts",
              filter=lambda ti: None if "__pycache__" in ti.name else ti)
    return base64.b64encode(buf.getvalue()).decode()


KERNEL_TEMPLATE = r'''
import subprocess, os, glob, json, time, base64, io, tarfile
def sh(c):
    print('+', c, flush=True); r = subprocess.run(c, shell=True); print('rc', r.returncode, flush=True); return r.returncode
CANDS = json.loads(%(cands)r)
W = '/kaggle/working'; R = '/kaggle/tmp/robot'; os.makedirs(R, exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode(%(tgz)r)), mode='r:gz').extractall(R)
SES = os.path.dirname(glob.glob('/kaggle/input/**/xhey-iphone-sessions/**/depth', recursive=True)[0])
LAB = os.path.dirname(glob.glob('/kaggle/input/**/run_s2/episodes', recursive=True)[0])
XR = glob.glob('/kaggle/input/**/xhey_rgb_unified', recursive=True)[0]
UN = glob.glob('/kaggle/input/**/xhey-openaoe-pretrain/**/unified', recursive=True)[0]
run = '/kaggle/tmp/run_s2'; os.makedirs(run + '/episodes', exist_ok=True)
for p in glob.glob(LAB + '/episodes/*.json'):
    e = json.load(open(p)); n = os.path.basename(p)[:-5]
    e['video_path'] = f'{SES}/depth/{n}/rgb.mp4'; json.dump(e, open(f'{run}/episodes/{n}.json', 'w'))
os.environ['PYTHONPATH'] = f'{R}/scripts'; os.makedirs('/kaggle/tmp/cache', exist_ok=True)
QC_FILES = sorted(set(glob.glob('/kaggle/input/**/qc/*/scorecards.json', recursive=True) + glob.glob('/kaggle/input/**/qc_*/scorecards.json', recursive=True)))
QC = ('--qc-filter ' + ' '.join(QC_FILES)) if QC_FILES else ''
print('qc scorecards:', QC_FILES, flush=True)
if not QC_FILES:
    print('QC 警告：这次提议没有找到 scorecards.json，训练不会过滤。固定候选必须在内核脚本里写 --qc-filter。', flush=True)
for i, c in enumerate(CANDS):
    out = f'{W}/c{i}'; os.makedirs(out, exist_ok=True)
    if not os.path.exists(out + '/cache'): os.symlink('/kaggle/tmp/cache', out + '/cache')
    oa = f"--openaoe {UN}/{c['openaoe']}" if c.get('openaoe') else ''
    sh(f"cd {R} && python scripts/train_xhey.py --run {run} --out {out} --ablation --mixes {c['mix']} --mix {c['mix']} "
       f"--xhey-rgb {XR} {oa} {QC} --seeds {c['seeds']} --steps {c['steps']} --pre-steps {c['pre_steps']} "
       f"--hidden {c['hidden']} --dropout {c['dropout']} --lr-scale {c['lr_scale']} --aug-noise {c['aug_noise']} --loss-move-cm {c.get('loss_move_cm', 0.0)} --no-final "
       f"> {W}/log_c{i}.txt 2>&1; tail -15 {W}/log_c{i}.txt")
    os.unlink(out + '/cache')
print('ALL_DONE')
'''


def load_state():
    if not STATE.exists():
        return {"running": None}
    return json.loads(STATE.read_text(encoding="utf-8"))


def save_state(state):
    EXP.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


def launch(cands, iteration):
    kd = EXP / "kernel"; kd.mkdir(parents=True, exist_ok=True)
    clean = [{k: v for k, v in c.items() if not k.startswith("_")} for c in cands]
    (kd / "run.py").write_text(KERNEL_TEMPLATE % {"cands": json.dumps(clean), "tgz": scripts_b64()}, encoding="utf-8")
    (kd / "kernel-metadata.json").write_text(json.dumps({
        "id": KERNEL, "title": KERNEL.split("/")[1], "code_file": "run.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": DATASETS, "kernel_sources": KERNEL_SOURCES}, indent=1))
    print(kaggle("kernels", "push", "-p", str(kd)).strip())
    state = load_state()
    state["running"] = {"iteration": iteration, "kernel": KERNEL, "version": iteration,
                        "candidates": cands, "launched": datetime.now().strftime("%Y-%m-%d %H:%M")}
    save_state(state)


def submit_queue_item(item):
    """推送 queue.json 里的一个固定候选内核。不改 auto_state.running（那是自动迭代轮次）。"""
    path = ROOT / item["dir"]
    if not (path / "kernel-metadata.json").exists():
        raise SystemExit("固定候选 %s 缺少 %s" % (item.get("id"), path))
    if not kernel_has_qc_filter(item):
        raise SystemExit("固定候选 %s 的内核没有 --qc-filter，拒绝提交。已经在跑的结果仍会入账。" % item.get("id"))
    print("提交固定候选 %s → %s（先于提议器）" % (item.get("id"), item.get("slug")))
    print(kaggle("kernels", "push", "-p", str(path)).strip())
    set_queue_status(item["id"], "running")
    spec = _spec_from_queue_item(item)
    state = load_state()
    state["queue_running"] = {"id": item.get("id"), "slug": item.get("slug"),
                              "mix": spec["mix"],
                              "launched": datetime.now().strftime("%Y-%m-%d %H:%M")}
    save_state(state)


def _find_metrics(dest):
    hits = sorted(dest.rglob("metrics.json"))
    return hits[0] if hits else None


def report_charts(metrics_path, name):
    """每收一次结果就出一组详细图表（失败不影响入账）。"""
    try:
        import xhey_report
        xhey_report.run_report(metrics_path, name, out_root=EXP / "report")
        xhey_report.summary(ledger_path=LEDGER, out_root=EXP / "report")
    except Exception as e:  # noqa: BLE001
        print("图表生成失败：", e)


def _record_metrics(spec, metrics, iteration=0):
    row = row_from_metrics(metrics, spec["mix"], spec.get("config") or {}, spec["id"], iteration, spec["slug"])
    append_ledger([row])
    print("ingested", row["id"], row["status"], row.get("delta_moving_cm"))
    return row


def ingest_metrics_file(path, spec, iteration=0):
    """把一份已经下载的 metrics.json 写入账本（id 已存在则跳过）。"""
    if any(r.get("id") == spec["id"] and r.get("status") == "ok" for r in load_ledger()):
        print("已在账本", spec["id"])
        mirror_ledger_csv()
        return None
    metrics = json.loads(Path(path).read_text(encoding="utf-8"))
    row = _record_metrics(spec, metrics, iteration)
    report_charts(path, spec["id"])
    return row


def ingest_kernel_output(spec, iteration=0, status=None):
    """下载一个已结束的 Kaggle 内核并写入账本。还在跑则跳过。"""
    if any(r.get("id") == spec["id"] and str(r.get("status", "")).startswith(("ok", "failed")) for r in load_ledger()):
        return None
    status = status or kernel_status(spec["slug"])
    if status in ("RUNNING", "QUEUED"):
        print(spec["id"], spec["slug"], "还在跑", status)
        return None
    if str(status).startswith("UNKNOWN"):
        print("拿不到", spec["slug"], "的状态：", status)
        return None
    dest = EXP / "runs" / spec["id"]
    dest.mkdir(parents=True, exist_ok=True)
    if status == "COMPLETE":
        kaggle("kernels", "output", spec["slug"], "-p", str(dest), check=False)
    mf = _find_metrics(dest)
    if mf is not None:
        row = _record_metrics(spec, json.loads(mf.read_text(encoding="utf-8")), iteration)
        report_charts(mf, spec["id"])
    else:
        row = {"id": spec["id"], "iteration": iteration, "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
               "source": spec["slug"], "config": norm_cfg(spec.get("config") or {}), "status": "failed:" + str(status)}
        append_ledger([row])
        print("ingested", row["id"], row["status"])
    for big in dest.rglob("*.pt"):
        big.unlink()
    if spec["id"] != "C1":
        kind = "done_worse" if row.get("status") == "ok" and (row.get("delta_overall_cm") or 0) > 0 else (
            "done_win" if row.get("status") == "ok" else "failed")
        if str(row.get("status", "")).startswith("failed"):
            kind = "failed"
        elif row.get("status") == "ok" and (row.get("delta_moving_cm") or 0) < 0:
            kind = "done_better_moving" if (row.get("delta_overall_cm") or 0) >= 0 else "done_win"
        set_queue_status(spec["id"], kind)
    return row


def queue_running_items(state=None):
    """正在跑的固定候选。queue_running 可以是一个对象或列表；queue_running_also 追加其余内核。"""
    if state is None:
        state = load_state()
    items = []

    def add(raw):
        if isinstance(raw, dict):
            if raw.get("slug"):
                items.append(raw)
        elif isinstance(raw, list):
            for one in raw:
                add(one)

    add(state.get("queue_running"))
    add(state.get("queue_running_also"))
    seen, out = set(), []
    for item in items:
        slug = item.get("slug")
        if not slug or slug in seen:
            continue
        seen.add(slug)
        out.append(item)
    return out


def _spec_from_queue_item(item):
    """入账用的规格。mix 以这条正在跑的记录为准，否则用 queue.json / QUEUE_CONFIG。

    C7b 不在 QUEUE_CONFIG 里。丢掉 mix 时会落到默认的 iphone+xhey_rgb+aug，
    把 Stera 臂记成 iphone 对照组。
    """
    cfg = dict(DEFAULT, **QUEUE_CONFIG.get(item.get("id"), {}))
    if not item.get("mix") and not (item.get("config") or {}).get("mix"):
        for queued in load_queue().get("queue", []):
            if queued.get("id") == item.get("id") and queued is not item and queued.get("mix"):
                cfg["mix"] = queued["mix"]
                break
    if item.get("config"):
        cfg.update(item["config"])
    if item.get("mix"):
        cfg["mix"] = item["mix"]
    return {"id": item.get("id"), "slug": item.get("slug"), "mix": cfg["mix"], "config": cfg}


def external_specs():
    """C1 总是尝试收；queue.json 里已提交的，以及 auto_state 里记下的正在跑的内核，都下载。"""
    specs, seen_ids, seen_slugs = [], set(), set()

    def add(spec):
        slug = spec.get("slug")
        if not slug or spec.get("id") in seen_ids or slug in seen_slugs:
            return
        seen_ids.add(spec.get("id"))
        seen_slugs.add(slug)
        specs.append(spec)

    add(C1_SPEC)
    for item in load_queue().get("queue", []):
        if (item.get("status") or "") not in INFLIGHT_STATUS or not item.get("slug"):
            continue
        add(_spec_from_queue_item(item))
    for item in queue_running_items():
        add(_spec_from_queue_item(item))
    return specs


def ingest_finished_external():
    """收已经跑完、且还没进账本的固定内核。不碰正在跑的自动迭代轮次。"""
    got = []
    for spec in external_specs():
        try:
            row = ingest_kernel_output(spec)
        except SystemExit as e:
            print(spec["id"], e)
            continue
        if row:
            got.append(row)
    return got


def ingest(state, status):
    run = state["running"]; it = run["iteration"]
    od = EXP / "runs" / ("iter%03d" % it); od.mkdir(parents=True, exist_ok=True)
    slug = run.get("kernel") or KERNEL
    rows = []
    if status == "COMPLETE":
        kaggle("kernels", "output", slug, "-p", str(od), check=False)
    for i, c in enumerate(run["candidates"]):
        mf = od / ("c%d" % i) / "metrics.json"
        rid = "iter%03d_c%d" % (it, i)
        if mf.exists():
            m = json.loads(mf.read_text(encoding="utf-8"))
            r = row_from_metrics(m, c["mix"], c, rid, it, slug)
            r["pred_delta_moving_cm"] = c.get("_pred_delta_moving_cm")
            rows.append(r)
        else:
            rows.append({"id": rid, "iteration": it, "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                         "source": slug, "config": norm_cfg(c), "status": "failed:" + status})
        for big in (od / ("c%d" % i)).glob("*.pt"):
            big.unlink()
    append_ledger(rows)
    for i in range(len(run["candidates"])):
        mf = od / ("c%d" % i) / "metrics.json"
        if mf.exists():
            report_charts(mf, "iter%03d_c%d" % (it, i))
    state = load_state()
    state["running"] = None
    state["last_iteration"] = it
    state["last_kernel"] = slug
    save_state(state)
    for r in rows:
        print("ingested", r["id"], r["status"], r.get("delta_moving_cm"))


def chart():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Serif CJK SC", "PingFang SC", "DejaVu Sans"]
    L = [r for r in load_ledger() if r.get("status") == "ok" and r.get("delta_moving_cm") is not None]
    if not L:
        return
    its = sorted({r["iteration"] for r in L}); best, cur = [], 1e9
    for it in its:
        cur = min(cur, min(r["delta_moving_cm"] for r in L if r["iteration"] == it)); best.append(cur)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.scatter([r["iteration"] for r in L], [r["delta_moving_cm"] for r in L], c="#bbb", label="每次尝试")
    ax.plot(its, best, "-o", c="#3a3", label="目前最好")
    ax.axhline(0, ls="--", c="#888"); ax.text(its[0], 0.005, "猜手不动", color="#888")
    ax.set_xlabel("迭代轮次（0 = 以前手动跑的）"); ax.set_ylabel("手在动时 ADE − 猜不动（cm，越低越好）")
    ax.set_title("XHEY 自动迭代进度（%s）" % datetime.now().strftime("%Y-%m-%d")); ax.legend()
    fig.tight_layout(); fig.savefig(CHART, dpi=120); print("chart", CHART)


def _remember_queue_running(items):
    state = load_state()
    state.pop("queue_running_also", None)
    if not items:
        state["queue_running"] = None
    elif len(items) == 1:
        state["queue_running"] = items[0]
    else:
        state["queue_running"] = list(items)
    save_state(state)


def _queue_slot_busy(state):
    """任一队列内核还在跑就占住 GPU。跑完的全部入账，剩下的继续记在 queue_running。"""
    items = queue_running_items(state)
    if not items:
        return False
    kept, changed, busy = [], False, False
    for q in items:
        s = kernel_status(q["slug"])
        if s in ("RUNNING", "QUEUED"):
            print("固定候选 %s 还在跑（%s），不提交新内核。" % (q.get("id"), s))
            kept.append(q)
            busy = True
            continue
        if s == "COMPLETE" or s == "ERROR" or str(s).startswith("CANCEL"):
            spec = next((x for x in external_specs() if x.get("slug") == q.get("slug")), None)
            if spec:
                ingest_kernel_output(spec, status=s)
            changed = True
            continue
        kept.append(q)
    if changed:
        _remember_queue_running(kept)
    return busy


def step(a):
    state = load_state()
    try:
        ingest_finished_external()
    except SystemExit as e:
        print(e)
    run = (load_state() or state).get("running")
    if run:
        slug = run.get("kernel") or KERNEL
        s = kernel_status(slug)
        if s in ("RUNNING", "QUEUED"):
            print("内核还在跑（%s %s，iteration %s），这次不提交新内核。" % (slug, s, run.get("iteration")))
            mirror_ledger_csv()
            return 0
        if str(s).startswith("UNKNOWN"):
            print("拿不到内核状态，稍后再试：", s)
            return 1
        ingest(load_state(), s); chart()
    if _queue_slot_busy(load_state()):
        mirror_ledger_csv()
        return 0
    if a.no_launch:
        mirror_ledger_csv()
        return 0
    ready, skipped = split_qc_ready(ready_queue_items())
    if skipped:
        print("跳过没有 --qc-filter 的就绪候选（新实验默认只训练质检通过的片段；已经在跑的内核照常入账）：",
              ", ".join(i.get("id") or "?" for i in skipped))
    if ready:
        print("固定候选优先于提议器：", ", ".join(i["id"] for i in ready))
        submit_queue_item(ready[0])
        return 0
    if skipped:
        print("队列里还有未质检的就绪候选，不交给提议器。")
        mirror_ledger_csv()
        return 0
    it = max([r["iteration"] for r in load_ledger()] + [0]) + 1
    cands = propose(a.k)
    if not cands:
        print("搜索空间都试过了。"); return 0
    print("提议第 %d 轮：" % it); [print("  ", json.dumps(c, ensure_ascii=False)) for c in cands]
    launch(cands, it)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["step", "propose", "chart", "mirror", "ingest-c1"])
    ap.add_argument("-k", type=int, default=3, help="每轮挑几个候选")
    ap.add_argument("--no-launch", action="store_true", help="step 只收结果，不推下一轮")
    ap.add_argument("--metrics", default=None, help="ingest-c1 时用本地 metrics.json，不访问 Kaggle")
    a = ap.parse_args()
    if a.cmd == "propose":
        ready, skipped = split_qc_ready(ready_queue_items())
        if skipped:
            print("没有 --qc-filter，不会提交（已在跑的仍入账）：")
            for item in skipped:
                print(" ", item["id"], item.get("slug"), item.get("status"))
        if ready:
            print("固定候选优先（提议器之前）：")
            for item in ready:
                print(" ", item["id"], item.get("slug"), item.get("status"))
            print("队列清空后才会提议：")
        elif skipped:
            print("队列里还有未质检的就绪候选，提议器先不跑。")
            return 0
        else:
            print("队列里没有可提交的固定候选，下面才是提议器：")
        for c in propose(a.k):
            print(json.dumps(c, ensure_ascii=False))
        return 0
    if a.cmd == "chart":
        chart(); return 0
    if a.cmd == "mirror":
        n = mirror_ledger_csv(); print("mirrored", n, "jsonl rows ->", CSV_PATH); return 0
    if a.cmd == "ingest-c1":
        if a.metrics:
            ingest_metrics_file(a.metrics, C1_SPEC)
        else:
            row = ingest_kernel_output(C1_SPEC)
            if row is None and not any(r.get("id") == "C1" and r.get("status") == "ok" for r in load_ledger()):
                return 1
        chart()
        return 0
    return step(a)


if __name__ == "__main__":
    sys.exit(main())
