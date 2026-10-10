#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""HOT3D 双目标签能不能帮训练：同一模型、同一目标、同一评测，只换训练标签。

数据：run_stereo_pipeline.py --hot3d 的输出目录（episodes/*.json、sessions/*/gt/hands_gt.json、
cache/*.wilor.json）。每个 clip 150 帧 → 149 行动作，4 个 clip 共 596 行。

标签来源（同一批帧）：
  stereo  双目流水线导出的手（就是 LeRobot 导出的那份）
  gt      HOT3D 动捕真值 21 点 → 同一套 wrist_poses_from_joints 得到手腕位姿（上限）
  mono    WiLoR 左目单目 3D（joints_cam）用 SLAM 相机位姿转到世界系（无双目三角化）
评测：留一 clip 交叉验证（训练 3 个 clip，测剩下 1 个），目标永远是 **GT 手腕增量**，
掩码用 GT 的 action_valid；指标完全复用 scripts/egodex_act_eval.py 的 metrics_for。
模型：linear（同 egodex_act_eval.py 的岭回归）和 mlp（2 层 256，CPU，多种子）。
EgoDex：在 docs/egodex_act/split.json 的 train 上训练，直接在 HOT3D 上测；
egodex+stereo：EgoDex 预训练的 MLP 再用 HOT3D 双目标签微调。
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from egodata.lerobot_export import pack_state, pack_action  # noqa: E402
from egodata.action_valid import action_valid_flags  # noqa: E402
from headcam.hand_pose import wrist_poses_from_joints  # noqa: E402
import egodex_act_eval as ev  # noqa: E402

H, STEP = ev.H, ev.STEP
IDENT = np.zeros(STEP); IDENT[6] = 1; IDENT[13] = 1


def _plain_episode(ep, joints_by_side):
    out = {k: v for k, v in ep.items() if k not in ("stereo", "hand_pose", "qc", "iphone")}
    n = ep["num_frames"]; out["hands"] = {}
    for side in ("left", "right"):
        js = joints_by_side[side]
        arr = [np.full((21, 3), np.nan) if j is None else np.asarray(j, float) for j in js]
        poses = wrist_poses_from_joints(arr)
        ok = [j is not None and np.isfinite(np.asarray(j, float)).all() for j in js]
        out["hands"][side] = {
            "joints": [np.asarray(j).tolist() if o else [[None] * 3] * 21 for j, o in zip(js, ok)],
            "wrist_pose": poses, "confidence": [1.0 if o else None for o in ok],
            "valid": ok, "filled": [False] * n}
    return out


def gt_episode(ep, session):
    g = json.loads((session / "gt" / "hands_gt.json").read_text())
    return _plain_episode(ep, {s: [f.get(s) for f in g["frames"]] for s in ("left", "right")})


def mono_episode(ep, cache):
    c = json.loads(cache.read_text())["views"]["left"]
    js = {"left": [], "right": []}
    for t, fr in enumerate(c):
        T = np.asarray(ep["camera_poses"][t], float)
        for s in ("left", "right"):
            h = fr.get(s) if fr else None
            if not h or h.get("joints_cam") is None or (h.get("confidence") and np.mean(h["confidence"]) < 0.5):
                js[s].append(None); continue
            p = np.asarray(h["joints_cam"], float)
            js[s].append(p @ T[:3, :3].T + T[:3, 3])
    return _plain_episode(ep, js)


def to_arrays(episodes):
    st, ac, va, eid = [], [], [], []
    for i, ep in enumerate(episodes):
        for f in range(ep["num_frames"] - 1):
            st.append(pack_state(ep, f)); ac.append(pack_action(ep, f)[:STEP])
            va.append(action_valid_flags(ep, f)); eid.append(i)
    return {"state": np.asarray(st, float), "action": np.asarray(ac, float),
            "valid": np.asarray(va, float), "episode": np.asarray(eid), "index": np.arange(len(st))}


def load_hot3d(run):
    run = Path(run); src = {"stereo": [], "gt": [], "mono": []}; names = []
    for p in sorted((run / "episodes").glob("*.json")):
        ep = json.loads(p.read_text()); names.append(p.stem)
        src["stereo"].append(ep)
        src["gt"].append(gt_episode(ep, run / "sessions" / p.stem))
        cache = run / "cache" / (p.stem + ".wilor.json")
        if cache.exists():
            src["mono"].append(mono_episode(ep, cache))
    return {k: to_arrays(v) for k, v in src.items() if len(v) == len(names)}, names


def wrist_err_cm(a, b):
    """两份标签的手腕位置差（cm，两边都有效的手）。"""
    e = []
    for h in range(2):
        s = slice(126 + 7 * h, 129 + 7 * h)
        m = (a["valid"][:, h] > 0.5) & (b["valid"][:, h] > 0.5)
        e.append(np.linalg.norm(a["state"][m, s] - b["state"][m, s], axis=1))
    e = np.concatenate(e) * 100
    return float(np.median(e)), float(np.percentile(e, 90)), int(len(e))


def chunks(d, eps):
    s = ev.chunk_starts(d["episode"], H, 1)
    return s[np.isin(d["episode"][s], eps)]


FEAT = "abs"


def features(state):
    """abs: 原始 140 维（和 EgoDex ACT 实验一样）。
    local: 平移不变——关节减去本手手腕（无效手仍为 0）、两手腕四元数、右腕相对左腕位移，共 137 维。
    绝对世界坐标在不同录制里原点不同，几百帧的小数据上会被模型当成"记住位置"的捷径。"""
    if FEAT == "abs":
        return state
    lw, rw = state[:, 126:129], state[:, 133:136]
    lj = state[:, 0:63].reshape(-1, 21, 3); rj = state[:, 63:126].reshape(-1, 21, 3)
    lv = np.abs(lj).sum((1, 2)) > 0; rv = np.abs(rj).sum((1, 2)) > 0
    lj = np.where(lv[:, None, None], lj - lw[:, None], 0); rj = np.where(rv[:, None, None], rj - rw[:, None], 0)
    both = (np.abs(lw).sum(1) > 0) & (np.abs(rw).sum(1) > 0)
    return np.hstack([lj.reshape(-1, 63), rj.reshape(-1, 63), state[:, 129:133], state[:, 136:140],
                      np.where(both[:, None], rw - lw, 0)])


def xy(d, starts):
    Y, V = ev.gather(d, starts)
    return features(d["state"][starts]), Y, V


L2_GRID = [1e-3, 1e-1, 1.0, 10.0, 100.0, 1000.0]
STEPS_GRID = [100, 300, 1000, 2000]


def _ade_own(pred, Y, V):
    e = []
    for h in range(2):
        s = slice(7 * h, 7 * h + 3)
        cont = np.cumprod(V[:, :, h] >= 0.5, 1)[:, -1].astype(bool)
        e.append(np.linalg.norm(np.cumsum(pred[:, :, s], 1) - np.cumsum(Y[:, :, s], 1), axis=-1).mean(1)[cont])
    e = np.concatenate(e)
    return float(e.mean()) if len(e) else np.inf


def tune(d, train_eps, kind):
    """只用训练 clip、只用该来源自己的标签做内层留一 clip，选正则（linear 的 l2 / mlp 的步数）。不看测试 clip，也不看 GT。"""
    grid = L2_GRID if kind == "linear" else STEPS_GRID
    score = np.zeros(len(grid))
    for hold in train_eps:
        inner = [e for e in train_eps if e != hold]
        X, Y, V = xy(d, chunks(d, inner)); Xv, Yv, Vv = xy(d, chunks(d, [hold]))
        if kind == "linear":
            for i, l2 in enumerate(grid):
                score[i] += _ade_own(fit_linear(X, Y, V, l2)(Xv), Yv, Vv)
        else:
            m = MLP(0); c = m.fit(X, Y, V, steps=max(grid), log_every=10 ** 9, eval_at=grid, Xv=Xv, Yv=Yv, Vv=Vv)
            got = {r["step"] + 1: r["test_ade_cm"] for r in c if r.get("test_ade_cm") is not None}
            for i, st in enumerate(grid):
                score[i] += got.get(st, np.inf)
    return grid[int(np.argmin(score))]


def fit_linear(X, Y, V, l2=1e-3):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    A = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    reg = l2 * np.eye(A.shape[1]); reg[-1, -1] = 0
    W = np.zeros((H, 2, A.shape[1], 7))
    for h in range(H):
        for hand in range(2):
            r = V[:, h, hand] >= 0.5
            if r.sum() < 5:
                W[h, hand, -1] = IDENT[:7]; continue
            Ar = A[r]
            W[h, hand] = np.linalg.solve(Ar.T @ Ar + reg * len(Ar), Ar.T @ Y[r, h, hand * 7:hand * 7 + 7])
    def pred(Xt):
        At = np.hstack([(Xt - mu) / sd, np.ones((len(Xt), 1))])
        return np.concatenate([np.einsum("nd,hdk->nhk", At, W[:, k]) for k in range(2)], -1)
    return pred


class MLP:
    """140 → 256 → 256 → 16×14，带掩码的 MSE（只在有效手上算），预测相对零动作的残差。"""

    def __init__(self, seed, hidden=256, in_dim=None):
        in_dim = in_dim or (140 if FEAT == "abs" else 137)
        import torch
        self.t = torch; torch.manual_seed(seed); self.seed = seed
        self.net = torch.nn.Sequential(torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(),
                                       torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
                                       torch.nn.Linear(hidden, H * STEP))
        self.norm = None

    def fit(self, X, Y, V, steps=2000, lr=1e-3, wd=1e-4, keep_norm=False, log_every=50, Xv=None, Yv=None, Vv=None,
            eval_at=()):
        t = self.t
        if self.norm is None or not keep_norm:
            ys = (Y - IDENT).reshape(-1, STEP)
            self.norm = (X.mean(0), X.std(0) + 1e-6, ys.std(0) + 1e-6)
        mu, sd, ysd = self.norm
        Xn = t.tensor((X - mu) / sd, dtype=t.float32)
        Yn = t.tensor((Y - IDENT) / ysd, dtype=t.float32)
        M = t.tensor(np.repeat(V >= 0.5, 7, axis=-1), dtype=t.float32)
        opt = t.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=wd)
        g = np.random.default_rng(self.seed); curve = []
        for it in range(steps):
            b = g.integers(0, len(Xn), min(128, len(Xn)))
            out = self.net(Xn[b]).view(-1, H, STEP)
            loss = (((out - Yn[b]) ** 2) * M[b]).sum() / M[b].sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()
            if it % log_every == 0 or it == steps - 1 or (it + 1) in eval_at:
                row = {"step": it, "train_loss": float(loss.detach())}
                if Xv is not None:
                    row["test_ade_cm"] = self._ade(Xv, Yv, Vv)
                curve.append(row)
        return curve

    def clone(self):
        c = MLP(self.seed); c.net = copy.deepcopy(self.net); c.norm = self.norm
        return c

    def _ade(self, X, Y, V):
        p = self(X)
        e = []
        for h in range(2):
            s = slice(7 * h, 7 * h + 3)
            cont = np.cumprod(V[:, :, h] >= 0.5, 1)[:, -1].astype(bool)
            cp = np.linalg.norm(np.cumsum(p[:, :, s], 1) - np.cumsum(Y[:, :, s], 1), axis=-1).mean(1)
            e.append(cp[cont])
        e = np.concatenate(e)
        return float(e.mean() * 100) if len(e) else None

    def __call__(self, X):
        t = self.t; mu, sd, ysd = self.norm
        with t.no_grad():
            out = self.net(t.tensor((X - mu) / sd, dtype=t.float32)).view(-1, H, STEP).numpy()
        return out * ysd + IDENT


def summarize(rows):
    keys = [k for k in rows[0] if k not in ("chunks",) and not k.startswith("n_hands") and rows[0][k] is not None]
    out = {}
    for k in keys:
        v = np.asarray([r[k] for r in rows if r.get(k) is not None], float)
        out[k] = float(v.mean())
    return out


def bootstrap_ci(per_chunk_err, n=2000, seed=0):
    g = np.random.default_rng(seed); x = np.asarray(per_chunk_err)
    if len(x) == 0:
        return None
    b = g.choice(x, (n, len(x))).mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def chunk_ade(d, starts, pred):
    tgt, val = ev.gather(d, starts); out = []
    for h in range(2):
        s = slice(7 * h, 7 * h + 3)
        cont = np.cumprod(val[:, :, h] >= 0.5, 1)[:, -1].astype(bool)
        cp = np.linalg.norm(np.cumsum(pred[:, :, s], 1) - np.cumsum(tgt[:, :, s], 1), axis=-1).mean(1) * 100
        out.append(np.where(cont, cp, np.nan))
    return np.stack(out, 1)  # (N, 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="run_stereo_pipeline.py --hot3d 的输出目录")
    p.add_argument("--egodex", default=None, help="EgoDex LeRobot 目录（docs/egodex_act 那份）")
    p.add_argument("--egodex-split", default="docs/egodex_act/split.json")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--out", required=True)
    p.add_argument("--tune", type=int, default=0, help="1=内层留一 clip 选 l2 和 MLP 步数")
    p.add_argument("--features", choices=["abs", "local"], default="abs")
    a = p.parse_args()
    global FEAT
    FEAT = a.features
    report_feat = a.features
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    data, names = load_hot3d(a.run)
    gt = data["gt"]
    report = {"clips": names, "rows": {k: int(len(v["state"])) for k, v in data.items()},
              "label_vs_gt_wrist_cm": {k: wrist_err_cm(v, gt) for k, v in data.items() if k != "gt"},
              "valid_frac": {k: float(v["valid"].mean()) for k, v in data.items()}, "features": report_feat}
    eg = None
    if a.egodex:
        eg = ev.load(a.egodex); sp = json.loads(Path(a.egodex_split).read_text())
        etr = chunks(eg, sp["train"]); EX, EY, EV = xy(eg, etr)
        report["egodex_train_chunks"] = int(len(etr))
        eg_lin = fit_linear(EX, EY, EV)
        eg_mlps = []
        for s in range(a.seeds):
            m = MLP(s); m.fit(EX, EY, EV, steps=a.steps); eg_mlps.append(m)
    per = {}; curves = {}; ade_chunks = {}; chosen = {}
    def add(name, fold, metrics, ca):
        per.setdefault(name, []).append(metrics)
        ade_chunks.setdefault(name, []).append(ca)
    for fold, test_ep in enumerate(range(len(names))):
        train_eps = [e for e in range(len(names)) if e != test_ep]
        te = chunks(gt, [test_ep]); GX, GY, GV = xy(gt, te)
        evals = lambda pred: (ev.metrics_for(gt, te, pred, [1, 8, 16]), chunk_ade(gt, te, pred))
        # 输入状态用评测时真实可得的那份：部署时只有双目，所以测试输入=双目状态
        Xtest = {k: features(data[k]["state"][te]) for k in data}
        zero = np.tile(IDENT, (len(te), H, 1)); add("zero", fold, *evals(zero))
        for src in [k for k in ("stereo", "gt", "mono") if k in data]:
            d = data[src]; tr = chunks(d, train_eps); X, Y, V = xy(d, tr)
            # 测试输入：用同一来源的状态（训练/测试分布一致），目标始终是 GT
            l2 = tune(d, train_eps, "linear") if a.tune else 1e-3
            steps = tune(d, train_eps, "mlp") if a.tune else a.steps
            chosen.setdefault(src, []).append({"fold": fold, "l2": l2, "mlp_steps": steps})
            add("linear_" + src, fold, *evals(fit_linear(X, Y, V, l2)(Xtest[src])))
            for s in range(a.seeds):
                m = MLP(s); c = m.fit(X, Y, V, steps=steps, Xv=Xtest[src], Yv=GY, Vv=GV)
                curves.setdefault("mlp_" + src, []).append({"fold": fold, "seed": s, "curve": c})
                add("mlp_%s#%d" % (src, s), fold, *evals(m(Xtest[src])))
        if eg is not None:
            add("linear_egodex", fold, *evals(eg_lin(Xtest["stereo"])))
            d = data["stereo"]; tr = chunks(d, train_eps); X, Y, V = xy(d, tr)
            for s, m in enumerate(eg_mlps):
                add("mlp_egodex#%d" % s, fold, *evals(m(Xtest["stereo"])))
                ft = m.clone()
                c = ft.fit(X, Y, V, steps=chosen["stereo"][-1]["mlp_steps"] if a.tune else a.steps // 2, lr=3e-4, keep_norm=True, Xv=Xtest["stereo"], Yv=GY, Vv=GV)
                curves.setdefault("mlp_egodex_ft_stereo", []).append({"fold": fold, "seed": s, "curve": c})
                add("mlp_egodex_ft_stereo#%d" % s, fold, *evals(ft(Xtest["stereo"])))
        print("fold", fold, "done", flush=True)
    # 汇总：每个方法 = 折内按 chunk 合并（与 egodex 一样的加权），种子再平均，给种子标准差和 bootstrap CI
    groups = {}
    for name in per:
        groups.setdefault(name.split("#")[0], []).append(name)
    results = {}
    for g, members in groups.items():
        seed_vals = []
        allchunks = []
        for name in members:
            ca = np.concatenate(ade_chunks[name]); allchunks.append(ca)
            seed_vals.append(summarize(per[name]))
        keys = seed_vals[0].keys()
        mean = {k: float(np.mean([s[k] for s in seed_vals])) for k in keys}
        std = {k: float(np.std([s[k] for s in seed_vals])) for k in keys} if len(seed_vals) > 1 else None
        ca = np.nanmean(np.stack(allchunks), 0)  # 种子平均后的每 chunk 每手 ADE
        flat = ca[np.isfinite(ca)]
        results[g] = {"mean": mean, "seed_std": std, "n_seeds": len(members),
                      "ade_cm_pooled": float(flat.mean()), "ade_ci95": bootstrap_ci(flat),
                      "ade_per_fold": [float(np.nanmean(np.nanmean(np.stack([ade_chunks[n][f] for n in members]), 0)))
                                       for f in range(len(names))]}
        np.save(out / ("chunk_ade_%s.npy" % g), ca)
    # 相对零动作的配对差及 CI
    z = np.load(out / "chunk_ade_zero.npy")
    for g in results:
        c = np.load(out / ("chunk_ade_%s.npy" % g)); m = np.isfinite(c) & np.isfinite(z)
        results[g]["ade_minus_zero_cm"] = float((c[m] - z[m]).mean())
        results[g]["ade_minus_zero_ci95"] = bootstrap_ci(c[m] - z[m])
    report["results"] = results; report["chosen_hparams"] = chosen; report["tune"] = bool(a.tune)
    (out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    (out / "curves.json").write_text(json.dumps(curves))
    print(json.dumps({g: [round(r["ade_cm_pooled"], 3), r["ade_ci95"], r["ade_minus_zero_ci95"]] for g, r in results.items()}, indent=0))


if __name__ == "__main__":
    main()
