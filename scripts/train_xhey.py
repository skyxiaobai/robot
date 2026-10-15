#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""XHEY：用 xhey App 在 iPhone（带激光雷达）上采的第一视角数据训练的手腕动作模型。

一条命令训练 + 留一段评测 + 保存模型：
    python scripts/train_xhey.py

数据配比消融（同批留一段 iPhone 评测）：
    python scripts/train_xhey.py --ablation

XHEY 看三样东西，预测接下来 16 步（15 fps，约 1 秒）双手手腕的位移和旋转：
  1. 画面：每帧 RGB 经 ImageNet 预训练 ResNet18 得到 512 维特征（冻结，缓存到磁盘）
  2. 当前手：21 个关节相对手腕的位置 + 两手腕朝向 + 两手相对位置（137 维）
  3. 最近 8 帧手腕轨迹：相对当前手腕的位置和有无检测（ACT 的 n_obs_steps 思路）
网络：3 层 MLP（512 宽，dropout），预测相对「手不动」的残差，只在有效手上算损失
（action_valid 掩码，等价于 ACT 的 action_is_pad）。评测时再乘一个 shrink 系数，
在内部验证集上选；shrink=0 就退回「手不动」。

--mix 用 + 连接：iphone（必选）/ egodex / hot3d / zeno_lidar（HF iPhone LiDAR，CC BY-NC）/ hub_rgb / hub_imu / zeno_raw（后三者需先 WiLoR）/ aug（镜像换手 + 2 倍速 + 输入噪声）。
评测：留一段交叉验证，指标复用 egodex_act_eval.metrics_for，对比 zero / linear / history_revert。
最终模型存到 <out>/xhey.pt（不进 git）。详见 docs/xhey_model.md。
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import egodex_act_eval as ev  # noqa: E402
import hot3d_train_compare as ht  # noqa: E402
from egodata.schema import iter_episode_paths  # noqa: E402

H, STEP, IDENT = ht.H, ht.STEP, ht.IDENT
K_HIST = 8
QC_EXCLUDE = set()  # scorecard failures
QC_SCORED = set()   # clips named by the supplied scorecards
QC_STRICT = False   # --strict-qc: missing-from-scorecard is an error; default warns and keeps
SEG_KEEP = {}       # --qc-segments: clip -> {"fps","frames","spans"}；训练块必须整块落在保留区间里（测试段不剪）


def seg_row_ok(eps, names, kind):
    """每个动作行（帧 f→f+1）是否落在保留区间里。没有 --qc-segments 时全 True；成绩单没覆盖的段保留并警告。"""
    from qc_scorecard import segment_frame_mask
    out, miss = [], []
    for ep, name in zip(eps, names):
        n = int(ep["num_frames"]); entry = SEG_KEEP.get(name)
        if entry is None:
            if SEG_KEEP:
                miss.append(name)
            out.append(np.ones(n - 1, bool)); continue
        step = max(1, int(round(entry["fps"] / float(ep.get("fps") or entry["fps"]))))
        m = segment_frame_mask(entry, n, step)
        out.append(m[:-1] & m[1:])
    if miss:
        print("QC 分段警告 %s：%d 段不在分段成绩单里，整段保留：%s" % (kind, len(miss), miss[:10]), flush=True)
    return np.concatenate(out) if out else np.zeros(0, bool)


def seg_chunk_filter(d, starts):
    """只留整块（H 个动作行）都在保留区间里的训练块起点。"""
    ok = d.get("seg_ok")
    if ok is None or not len(starts):
        return starts
    bad = np.concatenate(([0], np.cumsum(~ok.astype(bool))))
    return starts[(bad[starts + H] - bad[starts]) == 0]


def _qc_keep(paths, kind):
    paths = list(paths)
    if not QC_SCORED:
        return paths
    unseen = [p for p in paths if Path(p).stem not in QC_SCORED]
    if unseen:
        shown = [str(p) for p in unseen[:10]]
        if QC_STRICT:
            raise ValueError("Unreviewed QC episodes for %s: %s" % (kind, shown))
        print("QC 警告 %s：%d 段不在成绩单里，保留（--strict-qc 才会报错）：%s" % (
            kind, len(unseen), shown), flush=True)
    kept = [p for p in paths if Path(p).stem not in QC_EXCLUDE]
    if len(kept) != len(paths):
        print("QC 排除 %s：%d/%d 段" % (kind, len(paths) - len(kept), len(paths)), flush=True)
    return kept


# ---------------- 数据 ----------------
def load_episodes(run):
    paths = _qc_keep(sorted((Path(run) / "episodes").glob("*.json")), "iphone")
    from common.frames import canonicalize_for_xhey
    eps = [canonicalize_for_xhey(json.loads(p.read_text())) for p in paths]
    return eps, [p.stem for p in paths]


def subsample_episode(ep, k):
    """30 fps → 15 fps：所有按帧的列表每 k 帧取一帧，动作随之变成 k 帧的累计增量。"""
    n = ep["num_frames"]
    def walk(o):
        if isinstance(o, dict):
            return {kk: walk(v) for kk, v in o.items()}
        if isinstance(o, list) and len(o) == n:
            return o[::k]
        return o
    out = walk(ep); out["num_frames"] = len(range(0, n, k)); out["fps"] = (ep.get("fps") or 30) / k
    return out


def image_features(ep, name, cache_dir, batch=256, flip=False):
    """整段视频 → (num_frames, 512) ResNet18 特征，缓存为 .npy。flip=True 时先左右翻转画面（镜像增强）。"""
    f = Path(cache_dir) / (name + (".flip" if flip else "") + ".resnet18.npy")
    video = Path(ep["video_path"])
    stat = video.stat()
    signature = {"video": str(video.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                 "num_frames": ep["num_frames"], "flip": bool(flip)}
    meta_path = f.with_suffix(".meta")
    if f.exists() and meta_path.exists() and json.loads(meta_path.read_text()) == signature:
        return np.load(f).astype(np.float32)
    import cv2, torch, torchvision
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    net.fc = torch.nn.Identity(); net = net.eval().to(dev).half()
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    cap = cv2.VideoCapture(ep["video_path"]); out, buf = [], []
    def flush():
        x = torch.from_numpy(np.stack(buf)).to(dev).permute(0, 3, 1, 2).float() / 255
        x = ((x - mean) / std).half()
        with torch.no_grad():
            out.append(net(x).float().cpu().numpy())
        buf.clear()
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        fr = cv2.resize(fr, (224, 168))[:, :, ::-1]
        buf.append((fr[:, ::-1] if flip else fr).copy())
        if len(buf) == batch:
            flush()
    if buf:
        flush()
    feat = np.concatenate(out)[: ep["num_frames"]]
    if len(feat) < ep["num_frames"]:
        feat = np.vstack([feat, np.zeros((ep["num_frames"] - len(feat), feat.shape[1]), np.float32)])
    f.parent.mkdir(parents=True, exist_ok=True); np.save(f, feat.astype(np.float16))
    meta_path.write_text(json.dumps(signature))
    print("image features", f.name, feat.shape, flush=True)
    return feat.astype(np.float32)


def history(state, episode):
    """最近 K 帧手腕位置（相对当前手腕）+ 是否检测到，每只手。"""
    n = len(state); out = []
    for h, (js, ws) in enumerate(((slice(0, 63), slice(126, 129)), (slice(63, 126), slice(133, 136)))):
        seen = np.abs(state[:, js]).sum(1) > 0
        cur = state[:, ws]
        for k in range(1, K_HIST + 1):
            idx = np.arange(n) - k
            okk = (idx >= 0) & (episode[np.clip(idx, 0, None)] == episode)
            idx = np.clip(idx, 0, None)
            m = okk & seen[idx] & seen
            out.append(np.where(m[:, None], state[idx, ws] - cur, 0)); out.append(m[:, None].astype(float))
    return np.hstack(out)


def finish(d, img):
    """统一特征：局部手状态 137 + 历史 48 + 画面 512 + 有无画面 1。"""
    ht.FEAT = "local"
    d["f_state"] = ht.features(d["state"]); d["f_hist"] = history(d["state"], d["episode"])
    has = img is not None
    d["f_img"] = (img if has else np.zeros((len(d["state"]), 512))).astype(np.float32)
    d["f_has"] = np.full((len(d["state"]), 1), 1.0 if has else 0.0, np.float32)
    return d


def build_iphone(run, cache_dir):
    eps, names = load_episodes(run)
    d = ht.to_arrays(eps)
    d["seg_ok"] = seg_row_ok(eps, names, "iphone")
    img = np.vstack([image_features(e, n, cache_dir)[: e["num_frames"] - 1] for e, n in zip(eps, names)])
    return finish(d, img), eps, names


# ---- 镜像（左右翻转世界 x 轴，并交换左右手）----
def _mirror_pose_block(a):
    """(..., 7) 位置 + xyzw 四元数，关于 x=0 平面反射。"""
    a = a.copy(); a[..., 0] *= -1; a[..., 4] *= -1; a[..., 5] *= -1
    return a


def mirror_arrays(d):
    st = d["state"].copy()
    lj, rj = st[:, 0:63].reshape(-1, 21, 3).copy(), st[:, 63:126].reshape(-1, 21, 3).copy()
    lj[..., 0] *= -1; rj[..., 0] *= -1
    lw, rw = _mirror_pose_block(st[:, 126:133]), _mirror_pose_block(st[:, 133:140])
    nst = np.hstack([rj.reshape(-1, 63), lj.reshape(-1, 63), rw, lw])
    a = d["action"]
    na = np.concatenate([_mirror_pose_block(a[:, 7:14]), _mirror_pose_block(a[:, 0:7])], 1)
    return {"state": nst, "action": na, "valid": d["valid"][:, ::-1].copy(),
            "episode": d["episode"].copy(), "index": d["index"].copy()}


def subset(d, rows, ep_offset=0):
    out = {k: d[k][rows] for k in d}
    out["episode"] = out["episode"] + ep_offset
    return out


def concat(ds):
    off, parts = 0, []
    for d in ds:
        e = d["episode"]; u = np.unique(e); remap = np.searchsorted(u, e) + off
        x = dict(d); x["episode"] = remap; parts.append(x); off += len(u)
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def build_external(kind, root, cache_dir):
    """开源数据 → 和 iPhone 同样 140 维状态 / 14 维动作，30 fps 下采样到 15 fps。没有画面特征（has_image=0）。"""
    paths = _qc_keep(iter_episode_paths(root), kind)
    signature = json.dumps([(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in paths])
    if SEG_KEEP:  # 分段成绩单变了缓存就失效
        signature += json.dumps({p.stem: SEG_KEEP.get(p.stem) for p in paths}, sort_keys=True)
    f = Path(cache_dir) / ("%s_arrays_p0v1.npz" % kind)
    if f.exists():
        with np.load(f) as cached:
            if "metric_checked" in cached and bool(cached["metric_checked"]) and (
                "source_signature" in cached and str(cached["source_signature"]) == signature
            ):
                d = {k: cached[k] for k in ("state", "action", "valid", "episode", "index")}
                d["seg_ok"] = cached["seg_ok"] if "seg_ok" in cached else np.ones(len(d["state"]), bool)
                return finish(d, None)
        print("重建尺度未校验或源文件已变更的缓存", f, flush=True)
    eps, kept_names = [], []
    for p in paths:
        ep = json.loads(p.read_text())
        pose = ep.get("hand_pose") or {}
        if ep.get("usable_for_metric_xhey") is False or pose.get("metric_translation") is False or (
            pose.get("backend") == "wilor" and ep.get("camera_intrinsic") is None
        ):
            raise ValueError("%s 没有公制手腕标签；请用真实 --calib 重新转换，不能混入米制训练" % p)
        if (ep.get("fps") or 30) >= 25:
            ep = subsample_episode(ep, 2)
        if ep["num_frames"] > H + 1:
            from common.frames import canonicalize_for_xhey
            eps.append(canonicalize_for_xhey(ep)); kept_names.append(p.stem)
    if not eps:
        raise SystemExit("找不到 %s 数据：%s（见 docs/xhey_model.md 的准备步骤）" % (kind, root))
    d = ht.to_arrays(eps)
    d["seg_ok"] = seg_row_ok(eps, kept_names, kind)
    f.parent.mkdir(parents=True, exist_ok=True); np.savez(f, **d, metric_checked=True, source_signature=signature)
    print(kind, "episodes", len(eps), "rows", len(d["state"]), "valid", round(float(d["valid"].mean()), 3), flush=True)
    return finish(d, None)


def feats(d, starts, use_img=True, use_hist=True):
    parts = [d["f_state"][starts]]
    if use_hist: parts.append(d["f_hist"][starts])
    parts.append(d["f_img"][starts] if use_img else np.zeros((len(starts), 512), np.float32))
    parts.append(d["f_has"][starts] if use_img else np.zeros((len(starts), 1), np.float32))
    return np.hstack(parts).astype(np.float32)


# ---------------- 模型 ----------------
class XHEY:
    def __init__(self, in_dim, seed=0, hidden=512, dropout=0.2):
        import torch
        self.t = torch; torch.manual_seed(seed); self.seed = seed
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nn = torch.nn
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, H * STEP)).to(self.dev)
        self.norm = None; self.cum_w = 0.1; self.cfg = dict(in_dim=in_dim, hidden=hidden, dropout=dropout)

    def set_norm(self, X, Y):
        sd = X.std(0) + 1e-6; sd[sd < 1e-4] = 1.0
        self.norm = (X.mean(0), sd, (Y - IDENT).reshape(-1, STEP).std(0) + 1e-6)

    def fit(self, X, Y, V, Xv, Yv, Vv, steps=3000, lr=3e-4, wd=1e-2, bs=256, eval_every=100, noise=0.0):
        """noise>0：合成增强——给输入加高斯噪声（标准化后的单位；相当于手腕轨迹抖动 + 画面特征扰动）。"""
        t = self.t; self.shrink = 1.0
        mu, sd, ysd = self.norm
        T = lambda a: t.tensor(a, dtype=t.float32, device=self.dev)
        Xn, Yn = T((X - mu) / sd), T((Y - IDENT) / ysd)
        M = T(np.repeat(V >= 0.5, 7, axis=-1))
        # 累计位移损失（和评测 ADE 一致）：每只手连续有效的前缀上，比较累加后的手腕位置
        C = T(np.cumprod(V >= 0.5, axis=1)); ysd_t = T(ysd); id_t = T(IDENT)
        Ycum = [t.cumsum(Yn[:, :, 7 * h:7 * h + 3] * ysd_t[7 * h:7 * h + 3] + id_t[7 * h:7 * h + 3], 1) for h in range(2)]
        pos_scale = float(np.mean(ysd[[0, 1, 2, 7, 8, 9]]))
        opt = t.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=wd)
        sched = t.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        g = np.random.default_rng(self.seed); best = (np.inf, None, 0); curve = []
        for it in range(steps):
            self.net.train()
            b = t.tensor(g.integers(0, len(Xn), bs), device=self.dev)
            xb = Xn[b]
            if noise > 0:
                xb = xb + noise * t.randn_like(xb)
            out = self.net(xb).view(-1, H, STEP)
            loss = (((out - Yn[b]) ** 2) * M[b]).sum() / M[b].sum().clamp(min=1)
            for h in range(2):
                pc = t.cumsum(out[:, :, 7 * h:7 * h + 3] * ysd_t[7 * h:7 * h + 3] + id_t[7 * h:7 * h + 3], 1)
                cm = C[b][:, :, h:h + 1]
                loss = loss + self.cum_w * (((pc - Ycum[h][b]) / pos_scale) ** 2 * cm).sum() / (3 * cm.sum()).clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if (it + 1) % eval_every == 0:
                v = ade_prefix(self(Xv), Yv, Vv)
                curve.append({"step": it + 1, "loss": float(loss), "val_ade_cm": v})
                if v is not None and v < best[0]:
                    best = (v, {k: x.detach().clone() for k, x in self.net.state_dict().items()}, it + 1)
        if best[1] is not None:
            self.net.load_state_dict(best[1])
        self.best_step = best[2]
        # 收缩系数（在内部验证集上选）：pred = 不动 + s·(网络残差)。s=0 就退回「手不动」
        self.shrink = 1.0; p = self(Xv); bestv = None
        for sh in (0.0, 0.25, 0.5, 0.75, 1.0):
            v = ade_prefix(IDENT + sh * (p - IDENT), Yv, Vv)
            if v is not None and (bestv is None or v < bestv - 1e-6):
                bestv, self.shrink = v, sh
        curve.append({"shrink": self.shrink, "val_ade_prefix_cm": bestv})
        return curve

    def __call__(self, X):
        t = self.t; mu, sd, ysd = self.norm; self.net.eval(); outs = []
        with t.no_grad():
            for i in range(0, len(X), 8192):
                xt = t.tensor((X[i:i + 8192] - mu) / sd, dtype=t.float32, device=self.dev)
                outs.append(self.net(xt).view(-1, H, STEP).cpu().numpy())
        return IDENT + getattr(self, "shrink", 1.0) * (np.concatenate(outs) * ysd)

    def save(self, path, extra):
        self.t.save({"state_dict": self.net.state_dict(), "norm": self.norm, "cfg": self.cfg, "shrink": self.shrink,
                     "H": H, "STEP": STEP, "K_HIST": K_HIST, **extra}, path)


def ade_prefix(pred, Y, V):
    """验证用：每只手在「连续有效前缀」上的累计位移误差（cm）。比只看 16 步全有效的块多很多样本。"""
    num = den = 0.0
    for h in range(2):
        s = slice(7 * h, 7 * h + 3)
        c = np.cumprod(V[:, :, h] >= 0.5, 1)
        e = np.linalg.norm(np.cumsum(pred[:, :, s], 1) - np.cumsum(Y[:, :, s], 1), axis=-1)
        num += float((e * c).sum()); den += float(c.sum())
    return num / den * 100 if den else None


def prefix_parts(pred, Y, V):
    num = den = 0.0
    for h in range(2):
        s = slice(7 * h, 7 * h + 3)
        c = np.cumprod(V[:, :, h] >= 0.5, 1)
        e = np.linalg.norm(np.cumsum(pred[:, :, s], 1) - np.cumsum(Y[:, :, s], 1), axis=-1)
        num += float((e * c).sum()); den += float(c.sum())
    return num, den


def ade(pred, Y, V):
    e = []
    for h in range(2):
        s = slice(7 * h, 7 * h + 3)
        cont = np.cumprod(V[:, :, h] >= 0.5, 1)[:, -1].astype(bool)
        cp = np.linalg.norm(np.cumsum(pred[:, :, s], 1) - np.cumsum(Y[:, :, s], 1), axis=-1).mean(1)
        e.append(cp[cont])
    e = np.concatenate(e)
    return float(e.mean() * 100) if len(e) else None


def hist_revert(d, starts, alpha):
    """简单基线：下一步把手腕拉回最近 8 帧的平均位置（乘 alpha）。用来检验「去抖动」能解释多少收益。"""
    p = np.tile(IDENT, (len(starts), H, 1)); hist = d["f_hist"][starts]
    for h in range(2):
        hb = hist[:, h * 32:(h + 1) * 32].reshape(-1, K_HIST, 4); m = hb[:, :, 3:4]
        p[:, 0, 7 * h:7 * h + 3] = alpha * (hb[:, :, :3] * m).sum(1) / np.clip(m.sum(1), 1, None)
    return p


def moving_mask_from_Y(Y, thresh_cm):
    """True if either hand moves > thresh_cm over the horizon (GT cumulative xyz, cm)."""
    m = np.zeros(len(Y), dtype=bool)
    for h in range(2):
        m |= np.linalg.norm(Y[:, :, 7 * h:7 * h + 3].sum(1), axis=-1) * 100 > thresh_cm
    return m


def useful(d, starts):
    """只用至少一只手有有效动作的块（全无效的块对掩码损失没贡献）。"""
    _, V = ev.gather(d, starts)
    return starts[(V >= 0.5).any((1, 2))]


def split_inner(d, starts, frac=0.15):
    """每个训练片段最后 15% 作为验证（选最佳步数），与测试片段完全无关。"""
    tr, va = [], []
    for e in np.unique(d["episode"][starts]):
        s = starts[d["episode"][starts] == e]; cut = int(len(s) * (1 - frac))
        tr.append(s[:max(cut - H, 0)]); va.append(s[cut:])
    return np.concatenate(tr), np.concatenate(va)


def xyv(d, starts, use_img=True):
    Y, V = ev.gather(d, starts)
    return feats(d, starts, use_img), Y, V


def augmentation_train_starts(extra, train_starts, source_rows):
    """增强块的全部源动作必须在内部训练区间，镜像与倍速都遵守同一隔离区。"""
    allowed = np.zeros(source_rows, dtype=bool)
    for offset in range(H):
        allowed[train_starts + offset] = True
    starts = useful(extra, ev.chunk_starts(extra["episode"], H, 1))
    begin = extra["src_start"][starts]
    end = extra["src_end"][starts + H - 1]
    rejected = np.concatenate(([0], np.cumsum(~allowed)))
    return starts[(rejected[end] - rejected[begin]) == 0]


MIX_HELP = (
    "iphone 必选；egodex=EgoDex；hot3d=HOT3D；"
    "zeno_lidar=HF zeno iPhone LiDAR 家务（米制手腕，CC BY-NC）；"
    "hub_rgb=Hubdata iPhone RGB 经 WiLoR；"
    "hub_imu=Hubdata iPhone RGB+IMU 经 WiLoR；"
    "zeno_raw=zeno 开放样例视频经 WiLoR；"
    "aug=合成增强（镜像换手 + 2 倍速 + 输入噪声）"
)


# 自动迭代（scripts/xhey_auto_iter.py）可调的超参；默认值 = PR #47
TUNE = {"hidden": 512, "dropout": 0.2, "lr_scale": 1.0, "aug_noise": 0.1}


def train_mix(mix, iph, train_eps, ext, seed, steps, pre_steps, use_img=True, loss_move_cm=0.0):
    """在 iPhone 训练片段（+ 选中的开源/合成数据）上训一个 XHEY。返回模型。"""
    tr_all = seg_chunk_filter(iph, ht.chunks(iph, train_eps))  # 只剪训练块；测试段在 main 里不动
    tr, va = split_inner(iph, tr_all)
    aug_train = tr.copy()
    tr, va = useful(iph, tr), useful(iph, va)
    X, Y, V = xyv(iph, tr, use_img); Xv, Yv, Vv = xyv(iph, va, use_img)
    if loss_move_cm and loss_move_cm > 0:
        keep = moving_mask_from_Y(Y, loss_move_cm)
        if keep.any():
            X, Y, V = X[keep], Y[keep], V[keep]
            print("loss_move_cm", loss_move_cm, "kept", int(keep.sum()), "/", int(len(keep)), flush=True)
    if "aug" in mix:  # 合成数据：镜像（只取训练片段，画面用翻转后的特征）+ 2 倍速
        for extra in (ext["mirror"], ext["fast"]):
            ex_starts = augmentation_train_starts(extra, aug_train, len(iph["state"]))
            if not len(ex_starts):
                continue
            Xe, Ye, Ve = xyv(extra, ex_starts, use_img)
            X, Y, V = np.vstack([X, Xe]), np.concatenate([Y, Ye]), np.concatenate([V, Ve])
    m = XHEY(X.shape[1], seed, hidden=TUNE["hidden"], dropout=TUNE["dropout"]); m.set_norm(X, Y)
    pre_keys = ("egodex", "hot3d", "zeno_lidar", "hub_rgb", "hub_imu", "zeno_raw", "zeno_phone", "xhey_rgb", "openaoe", "stera", "di")
    pre = [ext[k] for k in pre_keys if k in mix and k in ext]
    if pre:
        P = concat(pre); ps = useful(P, seg_chunk_filter(P, ev.chunk_starts(P["episode"], H, 1)))
        Xp, Yp, Vp = xyv(P, ps, use_img)
        m.curves = {"pretrain": m.fit(Xp, Yp, Vp, Xv, Yv, Vv, steps=pre_steps, lr=3e-4 * TUNE["lr_scale"])}
    else:
        m.curves = {}
    m.curves["finetune"] = m.fit(X, Y, V, Xv, Yv, Vv, steps=steps, lr=(1e-4 if pre else 3e-4) * TUNE["lr_scale"],
                                 noise=TUNE["aug_noise"] if "aug" in mix else 0.0)
    return m


def horizon_parts(pred, Y, V):
    """每个预测步 k（1..H）的累计手腕位置误差：返回 (误差和[cm], 计数)，只算连续有效前缀。"""
    num = np.zeros(pred.shape[1]); den = np.zeros(pred.shape[1])
    for h in range(2):
        s = slice(7 * h, 7 * h + 3)
        c = np.cumprod(V[:, :, h] >= 0.5, 1)
        e = np.linalg.norm(np.cumsum(pred[:, :, s], 1) - np.cumsum(Y[:, :, s], 1), axis=-1) * 100
        num += (e * c).sum(0); den += c.sum(0)
    return num, den


def trajectory_examples(pred, Y, V, n=3):
    """挑手腕走得最远的几段（全有效），保存预测/真值的累计轨迹（cm，相对起点）。"""
    out = []
    for h in range(2):
        s = slice(7 * h, 7 * h + 3)
        full = np.cumprod(V[:, :, h] >= 0.5, 1)[:, -1].astype(bool)
        gt = np.cumsum(Y[:, :, s], 1) * 100; pr = np.cumsum(pred[:, :, s], 1) * 100
        dist = np.where(full, np.linalg.norm(gt[:, -1], axis=-1), -1)
        for i in np.argsort(-dist)[:n]:
            if dist[i] > 0:
                out.append({"hand": h, "chunk": int(i), "gt_cm": gt[i].round(3).tolist(), "pred_cm": pr[i].round(3).tolist()})
    out.sort(key=lambda e: -np.linalg.norm(e["gt_cm"][-1]))
    return out[:n]


DEFAULT_ABLATION = [
    "iphone",
    "iphone+aug",
    "iphone+zeno_lidar",
    "iphone+zeno_lidar+aug",
    "iphone+egodex",
    "iphone+egodex+zeno_lidar",
    "iphone+hub_rgb",
    "iphone+hub_imu",
    "iphone+zeno_raw",
    "iphone+zeno_phone",
    "iphone+egodex+hot3d+zeno_lidar+hub_rgb+hub_imu+zeno_raw+zeno_phone+aug",
]



def try_external(kind, root, cache):
    """开源/HF 目录缺失或没有 episode 时跳过该配比分量，不整场退出。"""
    root = Path(root)
    if not root.exists():
        print("skip %s: missing %s" % (kind, root), flush=True)
        return None
    paths = iter_episode_paths(root)
    if not paths:
        print("skip %s: no episode json under %s" % (kind, root), flush=True)
        return None
    try:
        return build_external(kind, root, cache)
    except SystemExit as e:
        print("skip %s: %s" % (kind, e), flush=True)
        return None

def prepare_external(mixes, iph, eps, names, a):
    ext = {}
    need = set("+".join(mixes).split("+"))
    cache = Path(a.out) / "cache"
    if "egodex" in need:
        _d = try_external("egodex", a.egodex, cache)
        if _d is not None:
            ext["egodex"] = _d
    if "hot3d" in need:
        _d = try_external("hot3d", Path(a.hot3d), cache)
        if _d is not None:
            ext["hot3d"] = _d
    if "zeno_lidar" in need:
        _d = try_external("zeno_lidar", Path(a.zeno_lidar), cache)
        if _d is not None:
            ext["zeno_lidar"] = _d
    if "hub_rgb" in need:
        _d = try_external("hub_rgb", Path(a.hub_rgb), cache)
        if _d is not None:
            ext["hub_rgb"] = _d
    if "hub_imu" in need:
        _d = try_external("hub_imu", Path(a.hub_imu), cache)
        if _d is not None:
            ext["hub_imu"] = _d
    if "zeno_raw" in need:
        _d = try_external("zeno_raw", Path(a.zeno_raw), cache)
        if _d is not None:
            ext["zeno_raw"] = _d
    if "zeno_phone" in need:
        _d = try_external("zeno_phone", Path(a.zeno_phone), cache)
        if _d is not None:
            ext["zeno_phone"] = _d
    if "xhey_rgb" in need:  # 你自己另外 17 段没深度的 xhey 录像，WiLoR 认手（相机系）
        _d = try_external("xhey_rgb", Path(a.xhey_rgb), cache)
        if _d is not None:
            ext["xhey_rgb"] = _d
    if "openaoe" in need:  # Open-AoE（inclusionAI/OpenAoE-2000h）MANO 手 + SLAM 世界系，已过滤「手在动」
        _d = try_external("openaoe", Path(a.openaoe), cache)
        if _d is not None:
            ext["openaoe"] = _d
    if "stera" in need:  # Stera-10M（fpvlabs/stera-10m）iPhone Pro LiDAR + ARKit 世界系（y 朝上），已过滤「手在动」
        _d = try_external("stera", Path(a.stera), cache)
        if _d is not None:
            ext["stera"] = _d
    if "di" in need:  # DynamicIntelligence Record3D，MediaPipe+深度 3 点/手，相机系 y 朝上，已过滤「手在动」
        _d = try_external("di", Path(a.di), cache)
        if _d is not None:
            ext["di"] = _d
    if "aug" in need:
        mir = mirror_arrays({k: iph[k] for k in ("state", "action", "valid", "episode", "index")})
        img = np.vstack([image_features(e, n, cache, flip=True)[: e["num_frames"] - 1] for e, n in zip(eps, names)])
        mir = finish(mir, img)
        mir["src_start"] = iph["index"].copy(); mir["src_end"] = mir["src_start"] + 1
        ext["mirror"] = mir
        fast_eps = [subsample_episode(e, 2) for e in eps]
        fd = ht.to_arrays(fast_eps)
        fimg = np.vstack([image_features(e, n, cache)[: e["num_frames"] - 1][::2][: f["num_frames"] - 1]
                          for e, n, f in zip(eps, names, fast_eps)])
        fd = finish(fd, fimg)
        offsets = np.cumsum([0] + [e["num_frames"] - 1 for e in eps[:-1]])
        fd["src_start"] = np.concatenate([off + 2 * np.arange(f["num_frames"] - 1)
                                          for off, f in zip(offsets, fast_eps)])
        fd["src_end"] = fd["src_start"] + 2
        ext["fast"] = fd
    return ext


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="outputs/iphone_mcap/run_s2", help="iPhone 管线输出目录（含 episodes/*.json）")
    ap.add_argument("--out", default="outputs/xhey", help="结果和模型保存目录")
    ap.add_argument("--mix", default="iphone+egodex+aug",
                    help="最终模型用哪些数据，用 + 连接。" + MIX_HELP + "。默认 iphone+egodex+aug")
    ap.add_argument("--ablation", action="store_true",
                    help="评测多种数据配比：" + ", ".join(DEFAULT_ABLATION) + "（加上 --mixes 可自定义）")
    ap.add_argument("--mixes", default=None, help="自定义消融列表，逗号分隔，例如 iphone,iphone+aug")
    ap.add_argument("--egodex", default="data/egodex_sub/unified", help="EgoDex 统一 episode JSON 目录")
    ap.add_argument("--hot3d", default="outputs/hot3d_scale/run/episodes", help="HOT3D episode JSON 目录")
    ap.add_argument("--zeno-lidar", default="data/hf_iphone/unified/zeno_lidar",
                    help="HF zeno iPhone LiDAR 统一 episode（CC BY-NC）")
    ap.add_argument("--hub-rgb", default="data/hf_iphone/unified/hub_rgb",
                    help="Hubdata RGB 经 WiLoR 的 episode 目录")
    ap.add_argument("--hub-imu", default="data/hf_iphone/unified/hub_imu",
                    help="Hubdata RGB+IMU 经 WiLoR 的 episode 目录")
    ap.add_argument("--zeno-raw", default="data/hf_iphone/unified/zeno_raw",
                    help="zeno raw 开放样例经 WiLoR 的 episode 目录")
    ap.add_argument("--zeno-phone", default="data/hf_iphone/unified/zeno_phone_wilor",
                    help="zeno phone sample 经 WiLoR 的 episode 目录")
    ap.add_argument("--xhey-rgb", default="data/xhey_rgb/unified",
                    help="另外 17 段无深度 xhey mcap 经 WiLoR 的 episode 目录（预训练用）")
    ap.add_argument("--openaoe", default="data/openaoe/unified",
                    help="Open-AoE 转换后的 episode 目录（scripts/convert_openaoe.py）")
    ap.add_argument("--stera", default="data/stera/unified",
                    help="Stera-10M 转换后的 episode 目录（scripts/convert_stera.py）")
    ap.add_argument("--di", default="data/di/unified",
                    help="DynamicIntelligence 转换后的 episode 目录（scripts/convert_di.py）")
    ap.add_argument("--move-cm", type=float, default=5.0,
                    help="「手在动」子集：未来 16 步手腕真实位移超过多少 cm")
    ap.add_argument("--no-final", action="store_true", help="只做留一段评测，不再训最终模型（自动迭代省时间）")
    ap.add_argument("--hidden", type=int, default=512, help="MLP 隐层宽度")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr-scale", type=float, default=1.0, help="学习率倍数（预训练和微调一起缩放）")
    ap.add_argument("--aug-noise", type=float, default=0.1, help="合成增强时的输入噪声")
    ap.add_argument("--loss-move-cm", type=float, default=0.0,
                    help=">0 时微调只保留 GT 手腕位移超过该 cm 的训练块（逼模型学「会动」）")
    ap.add_argument("--qc-filter", nargs="*", default=None,
                    help="可选。QC 成绩单 scorecards.json（可给多个）；成绩单里不合格的段不进训练。"
                         "成绩单没覆盖到的段默认警告并保留")
    ap.add_argument("--strict-qc", action="store_true",
                    help="和 --qc-filter 一起用：成绩单里没有的段直接报错。"
                         "默认只警告并保留（Open-AoE、Stera b02–b05 等还没有成绩单）")
    ap.add_argument("--qc-segments", nargs="*", default=None,
                    help="可选。qc_scorecard.py --segments 输出的 segments.json（可多个）：训练块只取整块落在"
                         "保留区间里的；留一段评测的测试段不剪（和旧基线可比）")
    ap.add_argument("--seeds", type=int, default=3, help="每个配比训练几个随机种子")
    ap.add_argument("--steps", type=int, default=3000, help="iPhone（微调）阶段最大步数，按内部验证集挑最佳步")
    ap.add_argument("--pre-steps", type=int, default=6000, help="开源数据预训练步数")
    ap.add_argument("--no-eval", action="store_true", help="跳过留一段评测，只用全部片段训最终模型")
    a = ap.parse_args()
    global QC_STRICT
    QC_EXCLUDE.clear()
    QC_SCORED.clear()
    QC_STRICT = bool(a.strict_qc)
    if a.qc_filter:
        from qc_scorecard import load_filter, load_scored
        for f in a.qc_filter:
            QC_EXCLUDE.update(load_filter(f))
            QC_SCORED.update(load_scored(f))
        print("QC 过滤：共 %d 段不合格，训练时排除" % len(QC_EXCLUDE), flush=True)
    SEG_KEEP.clear()
    if a.qc_segments:
        from qc_scorecard import load_segments
        SEG_KEEP.update(load_segments(a.qc_segments))
        print("QC 分段：%d 段有保留区间，训练块只取区间内" % len(SEG_KEEP), flush=True)
    TUNE.update(hidden=a.hidden, dropout=a.dropout, lr_scale=a.lr_scale, aug_noise=a.aug_noise)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    mixes = (a.mixes.split(",") if a.mixes else DEFAULT_ABLATION) if a.ablation else [a.mix]
    iph, eps, names = build_iphone(a.run, out / "cache")
    if not eps:
        raise ValueError("No approved iPhone episodes remain after QC")
    ext = prepare_external(mixes + [a.mix], iph, eps, names, a)
    print("clips", names, "rows", len(iph["state"]), "valid", round(float(iph["valid"].mean()), 4), flush=True)
    if not a.no_eval:
        per, chunks_ade, prefix, moving = {}, {}, {}, []
        horizon, curves, examples = {}, [], {}
        for test_ep in range(len(names)):
            train_eps = [e for e in range(len(names)) if e != test_ep]
            te, tr = ht.chunks(iph, [test_ep]), ht.chunks(iph, train_eps)
            _t, _ = ev.gather(iph, te)
            moving.append(np.stack([np.linalg.norm(_t[:, :, 7 * h:7 * h + 3].sum(1), axis=-1) * 100 > a.move_cm
                                    for h in range(2)], 1))
            def add(name, pred):
                per.setdefault(name, []).append(ev.metrics_for(iph, te, pred, [1, 8, 16]))
                chunks_ade.setdefault(name, []).append(ht.chunk_ade(iph, te, pred))
                Yte, Vte = ev.gather(iph, te)
                prefix.setdefault(name, []).append(prefix_parts(pred, Yte, Vte))
                horizon.setdefault(name, []).append(horizon_parts(pred, Yte, Vte))
                g0 = name.split("#")[0]
                if test_ep == 0 and g0 not in examples:
                    examples[g0] = {"clip": names[test_ep], "chunks": trajectory_examples(pred, Yte, Vte)}
            add("zero", np.tile(IDENT, (len(te), H, 1)))
            X, (Y, V) = ht.features(iph["state"][tr]), ev.gather(iph, tr)
            add("linear", ht.fit_linear(X, Y, V, ht.tune(iph, train_eps, "linear"))(ht.features(iph["state"][te])))
            Ytr, Vtr = ev.gather(iph, tr)
            al = min((0.0, 0.25, 0.5, 0.75, 1.0), key=lambda x: ade_prefix(hist_revert(iph, tr, x), Ytr, Vtr) or 1e9)
            add("history_revert", hist_revert(iph, te, al))
            for mix in mixes:
                for s in range(a.seeds):
                    m = train_mix(mix, iph, train_eps, ext, s, a.steps, a.pre_steps, loss_move_cm=a.loss_move_cm)
                    add("%s#%d" % (mix, s), m(feats(iph, te)))
                    curves.append({"mix": mix, "seed": s, "fold": names[test_ep], **getattr(m, "curves", {})})
                print("fold", names[test_ep], mix, "done %.0fs" % (time.time() - t0), flush=True)
        results = summarize_results(per, chunks_ade, names, np.concatenate(moving))
        for g in results:
            mem = [n for n in prefix if n.split("#")[0] == g]
            vals = [sum(x[0] for x in prefix[n]) / max(sum(x[1] for x in prefix[n]), 1) * 100 for n in mem]
            results[g]["prefix_ade_cm"] = float(np.mean(vals))
            results[g]["prefix_steps"] = int(sum(x[1] for x in prefix[mem[0]]))
            results[g]["n_full_chunks"] = int(np.isfinite(np.concatenate(chunks_ade[mem[0]])).sum())
            hv = [sum(x[0] for x in horizon[n]) / np.maximum(sum(x[1] for x in horizon[n]), 1) for n in mem]
            results[g]["horizon_ade_cm"] = np.mean(hv, 0).tolist()
            results[g]["horizon_ade_cm_seed_std"] = np.std(hv, 0).tolist()
        report = {"model": "XHEY", "run": a.run, "clips": names, "n_rows": int(len(iph["state"])),
                  "valid_frac": float(iph["valid"].mean()), "fps": 15, "horizon": H, "k_hist": K_HIST,
                  "seeds": a.seeds, "steps": a.steps, "pre_steps": a.pre_steps, "loss_move_cm": a.loss_move_cm, "mixes": mixes,
                  "qc_segments": bool(SEG_KEEP), "seg_kept_rows": {"iphone": int(iph["seg_ok"].sum()),
                  **{k: int(v["seg_ok"].sum()) for k, v in ext.items() if "seg_ok" in v}},
                  "external_rows": {k: int(len(v["state"])) for k, v in ext.items()}, "results": results,
                  "curves": curves, "examples": examples}
        (out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps({g: [round(r["ade_cm_pooled"], 2), round(r["ade_minus_zero_cm"], 2), round(r["prefix_ade_cm"], 2),
                              round(r.get("moving_ade_cm", float("nan")), 2), round(r.get("moving_minus_zero_cm", float("nan")), 2)]
                          for g, r in results.items()}, indent=1), flush=True)
        plot(results, out / "xhey_results.png")
    if a.no_final:
        return 0
    m = train_mix(a.mix, iph, list(range(len(names))), ext, 0, a.steps, a.pre_steps, loss_move_cm=a.loss_move_cm)
    m.save(out / "xhey.pt", {"clips": names, "mix": a.mix, "fps": 15,
                             "features": "state_local137+hist48+resnet18_512+has_image1"})
    print("saved", out / "xhey.pt", "%.0fs" % (time.time() - t0), flush=True)
    return 0


def summarize_results(per, chunks_ade, names, moving=None):
    results = {}
    z = np.concatenate(chunks_ade["zero"])
    for g in dict.fromkeys(n.split("#")[0] for n in per):
        mem = [n for n in per if n.split("#")[0] == g]
        seed_sum = [ht.summarize(per[n]) for n in mem]
        keys = seed_sum[0].keys()
        stack = np.stack([np.concatenate(chunks_ade[n]) for n in mem])
        ca = np.nanmean(stack, 0)
        m_ = np.isfinite(ca) & np.isfinite(z)
        seed_ade = [float(np.nanmean(x)) for x in stack]
        results[g] = {"mean": {k: float(np.mean([s[k] for s in seed_sum])) for k in keys},
                      "n_seeds": len(mem), "ade_cm_pooled": float(np.nanmean(ca)),
                      "ade_cm_seed_std": float(np.std(seed_ade)),
                      "ade_ci95": ht.bootstrap_ci(ca[np.isfinite(ca)]),
                      "ade_minus_zero_cm": float((ca[m_] - z[m_]).mean()),
                      "ade_minus_zero_ci95": ht.bootstrap_ci(ca[m_] - z[m_]),
                      "per_clip_ade_cm": {names[i]: float(np.nanmean(np.stack([chunks_ade[n][i] for n in mem])))
                                          for i in range(len(names))}}
        if moving is not None:  # 只看「手真的在动」的片段：猜不动在这里最吃亏
            mm = m_ & moving
            results[g]["moving_frac"] = float(mm.sum() / max(m_.sum(), 1))
            results[g]["moving_ade_cm"] = float(ca[mm].mean())
            results[g]["moving_zero_ade_cm"] = float(z[mm].mean())
            results[g]["moving_minus_zero_cm"] = float((ca[mm] - z[mm]).mean())
            results[g]["moving_minus_zero_ci95"] = ht.bootstrap_ci(ca[mm] - z[mm])
    return results


def plot(results, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    order = sorted(results, key=lambda g: (g not in ("zero", "linear"), g != "zero", results[g]["ade_cm_pooled"] if g not in ("zero", "linear") else 0))
    v = np.array([results[g]["ade_cm_pooled"] for g in order])
    ci = np.array([results[g]["ade_ci95"] for g in order]); err = [v - ci[:, 0], ci[:, 1] - v]
    best = min((g for g in order if g not in ("zero", "linear")), key=lambda g: results[g]["ade_cm_pooled"], default=None)
    col = ["#999" if g == "zero" else "#bbb" if g == "linear" else "#e4572e" if g == best else "#6a9fd8" for g in order]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar([g.replace("iphone", "iPhone") for g in order], v, yerr=err, capsize=4, color=col)
    ax.axhline(results["zero"]["ade_cm_pooled"], ls="--", c="k", lw=0.8, label="zero (hand still)")
    ax.set_ylabel("ADE over 16 steps (cm, lower is better)")
    ax.set_title("XHEY data-mix ablation, held-out iPhone clips (leave-one-clip-out)")
    ax.legend(); plt.xticks(rotation=20, ha="right"); plt.tight_layout(); fig.savefig(path, dpi=150)


if __name__ == "__main__":
    raise SystemExit(main())
