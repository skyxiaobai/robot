#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""XHEY wrist prediction: Phase-0 public-only training with explicit held-out splits.

    python scripts/train_xhey.py --mix egodex --split split.json --state-only

RGB uses frozen ResNet18 features; the trainable MLP image-input weights update.
iPhone / xhey_rgb are frozen probes, never training or tuning inputs in this CLI.
Historical iPhone loaders/augmentation helpers remain for pipeline regression.
See docs/xhey/PHASE0_ASTRA.md for offline smoke tests and split/RGB contracts.
"""
import argparse, hashlib, json, sys, time
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
CUM_W = 0.1  # --cum-weight: cumulative (relative-to-current-wrist) displacement loss weight
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
    # Indices refer to the ORIGINAL video, not the downsampled labels.
    out["video_frame_indices"] = ep.get("video_frame_indices", list(range(n)))[::k]
    out["video_fps"] = ep.get("video_fps", ep.get("fps") or 30)
    return out


def image_encoder():
    """Use pre-provisioned weights only; feature extraction never downloads."""
    import torch, torchvision
    from urllib.parse import urlparse
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(weights.url).path).name
    if not checkpoint.is_file():
        raise FileNotFoundError("Local ResNet18 weights missing: %s; provision approved weights or use --state-only" % checkpoint)
    net = torchvision.models.resnet18(weights=None)
    net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return net


def image_features(ep, name, cache_dir, batch=256, flip=False):
    """整段视频 → (num_frames, 512) ResNet18 特征，缓存为 .npy。flip=True 时先左右翻转画面（镜像增强）。"""
    f = Path(cache_dir) / (name + (".flip" if flip else "") + ".resnet18.npy")
    video = Path(ep["video_path"])
    stat = video.stat()
    indices = np.asarray(ep.get("video_frame_indices", list(range(ep["num_frames"]))), dtype=int)
    if len(indices) != ep["num_frames"] or not len(indices) or indices[0] < 0 or np.any(np.diff(indices) <= 0):
        raise ValueError("Invalid video_frame_indices: %s" % name)
    timestamps = ep.get("timestamps")
    video_fps = ep.get("video_fps", ep.get("fps"))
    if timestamps is not None and video_fps:
        ts = np.asarray(timestamps, float)
        expected = (indices - indices[0]) / float(video_fps)
        if ts.shape != indices.shape or not np.allclose(ts - ts[0], expected, atol=1e-3, rtol=0):
            raise ValueError("Label timestamps are not aligned to video_frame_indices: %s" % name)
    signature = {"version": 2, "video": str(video.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                 "indices": indices.tolist(), "video_fps": video_fps, "flip": bool(flip)}
    meta_path = f.with_suffix(".meta")
    if f.exists() and meta_path.exists() and json.loads(meta_path.read_text()) == signature:
        feat = np.load(f).astype(np.float32)
        if feat.shape == (len(indices), 512) and np.isfinite(feat).all():
            return feat
    import cv2, torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = image_encoder()
    net.fc = torch.nn.Identity(); net = net.eval().to(dev).half()
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    cap = cv2.VideoCapture(str(video)); out, buf = [], []
    expected_fps = signature["video_fps"]
    if not cap.isOpened() or (expected_fps and not np.isclose(cap.get(cv2.CAP_PROP_FPS), expected_fps, rtol=.01)):
        cap.release()
        raise ValueError("Video unavailable or FPS misaligned: %s" % video)
    def flush():
        x = torch.from_numpy(np.stack(buf)).to(dev).permute(0, 3, 1, 2).float() / 255
        x = ((x - mean) / std).half()
        with torch.no_grad():
            out.append(net(x).float().cpu().numpy())
        buf.clear()
    frame_index, selected = 0, 0
    try:
        while selected < len(indices):
            ok, fr = cap.read()
            if not ok:
                raise ValueError("Short/undecodable RGB video %s: missing frame %d" % (video, indices[selected]))
            if frame_index == indices[selected]:
                fr = cv2.resize(fr, (224, 168))[:, :, ::-1]
                buf.append((fr[:, ::-1] if flip else fr).copy())
                selected += 1
                if len(buf) == batch:
                    flush()
            frame_index += 1
        if buf:
            flush()
    finally:
        cap.release()
    feat = np.concatenate(out)
    if feat.shape != (len(indices), 512) or not np.isfinite(feat).all():
        raise ValueError("Invalid RGB features: %s" % video)
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
    if not eps:
        raise ValueError("No approved iPhone probe episodes: %s" % run)
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


def build_external(kind, root, cache_dir, use_img=None, episode_ids=None):
    """Load requested metric episodes. EgoDex RGB is on unless explicitly state-only.

    No array cache is trusted across QC/split/video changes; the expensive image
    encoder has its own content/alignment-aware cache. Relative video paths are
    resolved against the episode JSON. Labels and video share original indices.
    """
    root = Path(root)
    if not root.exists():
        raise ValueError("Missing requested source %s: %s" % (kind, root))
    paths = _qc_keep(iter_episode_paths(root), kind)
    use_img = kind == "egodex" if use_img is None else use_img
    wanted = None if episode_ids is None else set(episode_ids)
    eps, names, ids, images, loaded_paths = [], [], [], [], []
    from common.frames import canonicalize_for_xhey
    for p in paths:
        ep = json.loads(p.read_text())
        eid = ep.get("episode_id") or (kind + "/" + p.stem)
        if wanted is not None and eid not in wanted:
            continue
        if eid in ids:
            raise ValueError("Duplicate episode_id in %s: %s" % (kind, eid))
        pose = ep.get("hand_pose") or {}
        if ep.get("usable_for_metric_xhey") is False or pose.get("metric_translation") is False or (
            pose.get("backend") == "wilor" and ep.get("camera_intrinsic") is None
        ):
            raise ValueError("%s 没有公制手腕标签；请用真实 --calib 重新转换，不能混入米制训练" % p)
        if use_img:
            video = ep.get("video_path")
            if not video and kind == "egodex" and ep.get("source_path"):
                video = str(Path(ep["source_path"]).with_suffix(".mp4"))
            if not video:
                raise ValueError("Missing RGB video_path for %s:%s; use --state-only for a declared ablation" % (kind, eid))
            video = Path(video)
            if not video.is_absolute():
                video = p.parent / video
            if not video.is_file():
                raise ValueError("Missing RGB for %s:%s: %s" % (kind, eid, video))
            ep["video_path"] = str(video.resolve())
        if (ep.get("fps") or 30) >= 25:
            ep = subsample_episode(ep, 2)
        if ep["num_frames"] <= H + 1:
            if wanted is not None:
                raise ValueError("Requested episode has zero usable windows: %s:%s" % (kind, eid))
            continue
        ep = canonicalize_for_xhey(ep)
        eps.append(ep); names.append(p.stem); ids.append(eid); loaded_paths.append(p)
        if use_img:
            cache_name = kind + "_" + hashlib.sha256(str(p.resolve()).encode()).hexdigest()[:20]
            images.append(image_features(ep, cache_name, cache_dir)[:-1])
    if wanted is not None and wanted - set(ids):
        raise ValueError("Requested episodes missing/removed by QC in %s: %s" % (kind, sorted(wanted - set(ids))))
    if not eps:
        raise ValueError("No usable episodes in requested source %s: %s" % (kind, root))
    d = ht.to_arrays(eps)
    d["seg_ok"] = seg_row_ok(eps, names, kind)
    d["episode_id"] = np.asarray(ids)[d["episode"]]
    d["source"] = np.full(len(d["state"]), kind)
    d["fps"] = np.asarray([e.get("fps") or 30 for e in eps])[d["episode"]]
    d["episode_path"] = np.asarray([str(p.resolve()) for p in loaded_paths])[d["episode"]]
    d["video_path"] = np.asarray([e.get("video_path") or "" for e in eps])[d["episode"]]
    d = finish(d, np.vstack(images) if use_img else None)
    starts = useful(d, seg_chunk_filter(d, ev.chunk_starts(d["episode"], H, 1)))
    if not len(starts):
        raise ValueError("Requested source %s has zero valid training windows after QC: %s" % (kind, root))
    print("actual_source", json.dumps({"source": kind, "root": str(root.resolve()), "episodes": len(eps),
          "rows": len(d["state"]), "valid_windows": len(starts), "image_rows": int(d["f_has"].sum()),
          "image_coverage": float(d["f_has"].mean())}), flush=True)
    return d


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
        self.norm = None; self.cum_w = CUM_W; self.cfg = dict(in_dim=in_dim, hidden=hidden, dropout=dropout, cum_weight=CUM_W)

    def set_norm(self, X, Y):
        sd = X.std(0) + 1e-6; sd[sd < 1e-4] = 1.0
        self.norm = (X.mean(0), sd, (Y - IDENT).reshape(-1, STEP).std(0) + 1e-6)

    def fit(self, X, Y, V, Xv, Yv, Vv, steps=3000, lr=3e-4, wd=1e-2, bs=256, eval_every=100, noise=0.0,
            sample_sources=None):
        """noise>0：合成增强——给输入加高斯噪声（标准化后的单位；相当于手腕轨迹抖动 + 画面特征扰动）。"""
        if not len(X) or not len(Xv) or steps < 1:
            raise ValueError("Training/validation must be nonempty and steps positive")
        sources = np.asarray(sample_sources if sample_sources is not None else ["unspecified"] * len(X))
        if len(sources) != len(X):
            raise ValueError("sample_sources length mismatch")
        drawn = np.zeros(len(X), dtype=np.int64)
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
            indices = g.integers(0, len(Xn), bs)
            np.add.at(drawn, indices, 1)
            b = t.tensor(indices, device=self.dev)
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
            if (it + 1) % eval_every == 0 or it == steps - 1:
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
        self.sample_manifest = {
            str(src): {"available_windows": int(np.sum(sources == src)),
                       "sampled_draws": int(drawn[sources == src].sum()),
                       "unique_sampled_windows": int(np.count_nonzero(drawn[sources == src])),
                       "effective_weight": float(drawn[sources == src].sum() / drawn.sum()),
                       "sampled_image_draws": int(drawn[(sources == src) & (X[:, -1] > .5)].sum())}
            for src in np.unique(sources)}
        print("actual_sampling", json.dumps(self.sample_manifest), flush=True)
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


MIX_HELP = "public-only: egodex / hot3d / hoi4d / egoverse / openaoe; iphone is probe-only."
EXTERNAL_KEYS = ("egodex", "hot3d", "zeno_lidar", "hub_rgb", "hub_imu", "zeno_raw", "zeno_phone", "xhey_rgb", "openaoe", "stera", "di", "hoi4d", "xphi", "egoverse")


# 自动迭代（scripts/xhey_auto_iter.py）可调的超参；默认值 = PR #47
TUNE = {"hidden": 512, "dropout": 0.2, "lr_scale": 1.0, "aug_noise": 0.1}


def train_mix(mix, iph, train_eps, ext, seed, steps, pre_steps, use_img=True, loss_move_cm=0.0):
    """Retired training API; loaders and augmentation remain for probe regressions."""
    raise ValueError("iPhone is frozen probe/eval-only; use the explicit public --split path")


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
    """Compatibility name: requested sources now fail closed, never return None."""
    return build_external(kind, root, cache)

def prepare_external(mixes, iph, eps, names, a):
    ext = {}
    need = set("+".join(mixes).split("+"))
    unknown = need - set(EXTERNAL_KEYS) - {"iphone", "aug"}
    if unknown:
        raise ValueError("Unknown requested sources: %s" % sorted(unknown))
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
    if "hoi4d" in need:  # HOI4D 头戴 RGB-D，MANO 真值手（相机光学系，米），只标交互手，已过滤「手在动」
        _d = try_external("hoi4d", Path(a.hoi4d), cache)
        if _d is not None:
            ext["hoi4d"] = _d
    if "xphi" in need:  # xPhi-Ego-100 头戴单目 RGB，WiLoR 伪标签（假设视场角，尺度不可靠），相机光学系，已过滤「手在动」
        _d = try_external("xphi", Path(a.xphi), cache)
        if _d is not None:
            ext["xphi"] = _d
    if "egoverse" in need:  # EgoVerse/Mecka 头戴双目，MANO 真值关键点+头部位姿+真实内参，相机光学系，原生标签 10 Hz（转换已抽关键帧）
        _d = try_external("egoverse", Path(a.egoverse), cache)
        if _d is not None:
            ext["egoverse"] = _d
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
    missing = need - {"iphone", "aug"} - set(ext)
    if missing:
        raise ValueError("Requested sources not loaded: %s" % sorted(missing))
    return ext


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="outputs/iphone_mcap/run_s2", help="旧 iPhone 探针路径；public-only 训练不读取")
    ap.add_argument("--out", default="outputs/xhey", help="结果和模型保存目录")
    ap.add_argument("--mix", default="egodex",
                    help="公开训练数据源，用 + 连接。" + MIX_HELP)
    ap.add_argument("--split", help="显式 train/val/test episode_id 和独立分组 JSON（必需）")
    ap.add_argument("--state-only", action="store_true", help="声明无图消融；不加载视频/视觉权重")
    ap.add_argument("--prepare-only", action="store_true", help="只校验本地数据并写实际清单，不训练")
    ap.add_argument("--ablation", action="store_true", help="旧接口，public-only 拒绝；请分开运行每个配比")
    ap.add_argument("--mixes", default=None, help="旧接口，public-only 拒绝；使用 --mix")
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
    ap.add_argument("--hoi4d", default="data/hoi4d/unified",
                    help="HOI4D 转换后的 episode 目录（scripts/convert_hoi4d.py）")
    ap.add_argument("--egoverse", default="data/egoverse/unified",
                    help="EgoVerse/Mecka 转换后的 episode 目录（scripts/convert_egoverse.py）")
    ap.add_argument("--xphi", default="data/xphi/unified",
                    help="xPhi-Ego-100 WiLoR 伪标签 episode 目录（相机光学系；内参为假设视场角）")
    ap.add_argument("--move-cm", type=float, default=5.0,
                    help="「手在动」子集：未来 16 步手腕真实位移超过多少 cm")
    ap.add_argument("--no-final", action="store_true", help="不保存 checkpoint；不影响显式 test 评测")
    ap.add_argument("--hidden", type=int, default=512, help="MLP 隐层宽度")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr-scale", type=float, default=1.0, help="学习率倍数（预训练和微调一起缩放）")
    ap.add_argument("--aug-noise", type=float, default=0.1, help="合成增强时的输入噪声")
    ap.add_argument("--cum-weight", type=float, default=0.1,
                    help="累计位移损失权重（相对当前手腕，默认 0.1；调大=少盯逐帧增量、多盯整段相对位移，C15 噪声假设验证）")
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
    ap.add_argument("--steps", type=int, default=3000, help="公开训练最大步数；只在显式 val 上早停/shrink")
    ap.add_argument("--pre-steps", type=int, default=None, help="旧接口；public-only 不支持两阶段训练")
    ap.add_argument("--no-eval", action="store_true", help="跳过 test 评测；仍仅在显式 train 上训练、val 上选择")
    a = ap.parse_args()
    global QC_STRICT, CUM_W
    CUM_W = float(a.cum_weight)
    print("cum_weight", CUM_W, flush=True)
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
    from xhey_public import run_public
    return run_public(a, sys.modules[__name__])


def summarize_results(per, chunks_ade, names, moving=None, cluster_ids=None):
    from xhey_stats import cluster_summary
    results = {}
    z = np.concatenate(chunks_ade["zero"])
    units = names if cluster_ids is None else cluster_ids
    if len(units) != len(names):
        raise ValueError("One cluster id per episode is required")
    groups = np.concatenate([np.repeat(unit, len(part)) for unit, part in zip(units, chunks_ade["zero"])])
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
                      "ci_method": "episode_session_cluster_bootstrap",
                      "ci_estimand": "pooled_window_hand_mean",
                      "n_eval_units": cluster_summary(np.where(m_, ca, np.nan), groups)["n_units"],
                      "decision_eligible": False,
                      "evidence_status": "evidence_insufficient",
                      "ade_ci95": cluster_summary(ca, groups)["ci95"],
                      "ade_minus_zero_cm": float((ca[m_] - z[m_]).mean()),
                      "ade_minus_zero_ci95": cluster_summary(np.where(m_, ca - z, np.nan), groups)["ci95"],
                      "per_clip_ade_cm": {names[i]: float(np.nanmean(np.stack([chunks_ade[n][i] for n in mem])))
                                          for i in range(len(names))}}
        if moving is not None:  # 只看「手真的在动」的片段：猜不动在这里最吃亏
            mm = m_ & moving
            results[g]["moving_frac"] = float(mm.sum() / max(m_.sum(), 1))
            results[g]["moving_ade_cm"] = float(ca[mm].mean()) if mm.any() else None
            results[g]["moving_zero_ade_cm"] = float(z[mm].mean()) if mm.any() else None
            results[g]["moving_minus_zero_cm"] = float((ca[mm] - z[mm]).mean()) if mm.any() else None
            summary = cluster_summary(np.where(mm, ca - z, np.nan), groups)
            results[g]["moving_minus_zero_ci95"] = summary["ci95"]
            results[g]["n_moving_eval_units"] = summary["n_units"]
    return results


def plot(results, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    order = sorted(results, key=lambda g: (g not in ("zero", "linear"), g != "zero", results[g]["ade_cm_pooled"] if g not in ("zero", "linear") else 0))
    v = np.array([results[g]["ade_cm_pooled"] for g in order])
    ci = np.array([results[g]["ade_ci95"] or [v[i], v[i]] for i, g in enumerate(order)])
    err = [np.maximum(v - ci[:, 0], 0), np.maximum(ci[:, 1] - v, 0)]
    best = min((g for g in order if g not in ("zero", "linear")), key=lambda g: results[g]["ade_cm_pooled"], default=None)
    col = ["#999" if g == "zero" else "#bbb" if g == "linear" else "#e4572e" if g == best else "#6a9fd8" for g in order]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar([g.replace("iphone", "iPhone") for g in order], v, yerr=err, capsize=4, color=col)
    ax.axhline(results["zero"]["ade_cm_pooled"], ls="--", c="k", lw=0.8, label="zero (hand still)")
    ax.set_ylabel("ADE over 16 steps (cm, lower is better)")
    ax.set_title("XHEY held-out episodes (cluster bootstrap; no Phase-0 win/loss)")
    ax.legend(); plt.xticks(rotation=20, ha="right"); plt.tight_layout(); fig.savefig(path, dpi=150)


if __name__ == "__main__":
    raise SystemExit(main())
