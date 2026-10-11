#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""路径 A：iPhone RGB-D 标签训小模型，是否优于「手不动」。

数据：run_stereo_pipeline --iphone 的 episodes（默认只拿 QC accepted=yes）。
目标与输入都来自同一套融合标签（无动捕真值）。指标同 egodex_act_eval。
"""
import argparse, csv, json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from egodata.lerobot_export import pack_state, pack_action  # noqa: E402
from egodata.action_valid import action_valid_flags  # noqa: E402
import egodex_act_eval as ev  # noqa: E402
import hot3d_train_compare as ht  # noqa: E402

H, STEP = ev.H, ev.STEP


def load_run(run):
    run = Path(run)
    accepted = None
    csv_path = run / "qc" / "yield.csv"
    if csv_path.exists():
        accepted = set()
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if str(row.get("accepted", "")).lower() in ("yes", "true", "1"):
                    accepted.add(Path(row["episode_id"]).name)
    eps, names = [], []
    for p in sorted((run / "episodes").glob("*.json")):
        if accepted is not None and p.stem not in accepted:
            continue
        eps.append(json.loads(p.read_text()))
        names.append(p.stem)
    return ht.to_arrays(eps), names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--tune", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--features", choices=["abs", "local"], default="abs")
    ap.add_argument("--all-episodes", action="store_true", help="忽略 QC，用全部 episode")
    a = ap.parse_args()
    ht.FEAT = a.features
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.all_episodes:
        eps_paths = sorted((Path(a.run) / "episodes").glob("*.json"))
        eps = [json.loads(p.read_text()) for p in eps_paths]
        names = [p.stem for p in eps_paths]
        data = ht.to_arrays(eps)
    else:
        data, names = load_run(a.run)

    import torch
    print("clips", names, "rows", len(data["state"]), "cuda", torch.cuda.is_available(), flush=True)

    per, curves, ade_chunks, chosen = {}, {}, {}, []
    def add(name, metrics, ca):
        per.setdefault(name, []).append(metrics)
        ade_chunks.setdefault(name, []).append(ca)

    for fold, test_ep in enumerate(range(len(names))):
        train_eps = [e for e in range(len(names)) if e != test_ep]
        if len(train_eps) == 0:
            continue
        te = ht.chunks(data, [test_ep])
        tr = ht.chunks(data, train_eps)
        if len(te) < 5 or len(tr) < 10:
            print("skip fold", fold, "tr", len(tr), "te", len(te), flush=True)
            continue
        Xte, Yte, Vte = ht.xy(data, te)
        evals = lambda pred: (ev.metrics_for(data, te, pred, [1, 8, 16]), ht.chunk_ade(data, te, pred))
        zero = np.tile(ht.IDENT, (len(te), H, 1))
        add("zero", *evals(zero))

        X, Y, V = ht.xy(data, tr)
        l2 = ht.tune(data, train_eps, "linear") if a.tune else 1e-3
        steps = ht.tune(data, train_eps, "mlp") if a.tune else a.steps
        chosen.append({"fold": fold, "l2": l2, "mlp_steps": steps, "test": names[test_ep]})
        add("linear_iphone", *evals(ht.fit_linear(X, Y, V, l2)(Xte)))
        for s in range(a.seeds):
            m = ht.MLP(s)
            c = m.fit(X, Y, V, steps=int(steps), Xv=Xte, Yv=Yte, Vv=Vte)
            curves.setdefault("mlp_iphone", []).append({"fold": fold, "seed": s, "curve": c})
            add("mlp_iphone#%d" % s, *evals(m(Xte)))
        print("fold", fold, names[test_ep], "done", "l2", l2, "steps", steps, flush=True)

    groups = {}
    for name in per:
        groups.setdefault(name.split("#")[0], []).append(name)
    results = {}
    for g, members in groups.items():
        seed_vals, allchunks = [], []
        for name in members:
            ca = np.concatenate(ade_chunks[name])
            allchunks.append(ca)
            seed_vals.append(ht.summarize(per[name]))
        keys = seed_vals[0].keys()
        mean = {k: float(np.mean([s[k] for s in seed_vals])) for k in keys}
        std = {k: float(np.std([s[k] for s in seed_vals])) for k in keys} if len(seed_vals) > 1 else None
        ca = np.nanmean(np.stack(allchunks), 0)
        flat = ca[np.isfinite(ca)]
        results[g] = {"mean": mean, "seed_std": std, "n_seeds": len(members),
                      "ade_cm_pooled": float(flat.mean()) if len(flat) else None,
                      "ade_ci95": ht.bootstrap_ci(flat) if len(flat) else None}
        np.save(out / ("chunk_ade_%s.npy" % g), ca)
    zpath = out / "chunk_ade_zero.npy"
    if zpath.exists():
        z = np.load(zpath)
        for g in results:
            c = np.load(out / ("chunk_ade_%s.npy" % g))
            m = np.isfinite(c) & np.isfinite(z)
            results[g]["ade_minus_zero_cm"] = float((c[m] - z[m]).mean()) if m.any() else None
            results[g]["ade_minus_zero_ci95"] = ht.bootstrap_ci(c[m] - z[m]) if m.any() else None

    report = {"clips": names, "features": a.features, "n_rows": int(len(data["state"])),
              "valid_frac": float(data["valid"].mean()), "chosen_hparams": chosen,
              "results": results, "tune": bool(a.tune)}
    (out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "curves.json").write_text(json.dumps(curves), encoding="utf-8")
    print(json.dumps({g: [None if r["ade_cm_pooled"] is None else round(r["ade_cm_pooled"], 3),
                          r.get("ade_minus_zero_cm"), r["ade_ci95"]]
                      for g, r in results.items()}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
