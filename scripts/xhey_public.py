"""Small explicit-split public-only runner for train_xhey (no launch/download code)."""
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

PUBLIC_SOURCES = {"egodex", "egoverse", "hoi4d", "hot3d", "openaoe"}


def read_split(path):
    if not path:
        raise ValueError("Public training requires --split with explicit train/val/test and groups")
    split = json.loads(Path(path).read_text())
    if split.get("unit") not in ("episode", "session", "subject"):
        raise ValueError("split.unit must be episode, session or subject (use the largest dependent unit)")
    groups = split.get("groups", {})
    seen_ids, seen_groups = set(), set()
    for part in ("train", "val", "test"):
        ids = split.get(part)
        if not isinstance(ids, list) or not ids or not all(isinstance(i, str) and i for i in ids):
            raise ValueError("split.%s must be a nonempty episode_id list" % part)
        if len(set(ids)) != len(ids) or seen_ids.intersection(ids):
            raise ValueError("Duplicate/overlapping episode IDs across splits")
        if any(not isinstance(groups.get(i), str) or not groups[i] for i in ids):
            raise ValueError("Every selected episode needs an explicit independent group")
        units = {groups[i] for i in ids}
        if seen_groups.intersection(units):
            raise ValueError("Session/subject group leakage across train/val/test")
        seen_ids.update(ids); seen_groups.update(units)
    return split


def file_identity(path, sha=False):
    p = Path(path).resolve(); st = p.stat()
    result = {"path": str(p), "bytes": st.st_size, "mtime_ns": st.st_mtime_ns}
    if sha:
        result["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    return result


def constant_velocity(tx, d, starts):
    """Repeat last observed translation; never read target action at `starts`."""
    pred = np.tile(tx.IDENT, (len(starts), tx.H, 1))
    prev = np.maximum(starts - 1, 0)
    same = (starts > 0) & (d["episode"][prev] == d["episode"][starts])
    for hand in range(2):
        valid = same & (d["valid"][prev, hand] >= .5)
        sl = slice(7 * hand, 7 * hand + 3)
        pred[:, :, sl] = np.where(valid[:, None], d["action"][prev, sl], 0)[:, None, :]
    return pred


def run_public(a, tx):
    sources = a.mix.split("+")
    frozen = {"iphone", "xhey_rgb"}.intersection(sources)
    if frozen:
        raise ValueError("iPhone/xhey_rgb are frozen probe/eval-only; no training or tuning: %s" % sorted(frozen))
    if len(sources) != len(set(sources)) or set(sources) - PUBLIC_SOURCES:
        raise ValueError("Unsupported/unavailable public sources: %s; allowed: %s" % (a.mix, sorted(PUBLIC_SOURCES)))
    if a.ablation or a.mixes or a.pre_steps is not None:
        raise ValueError("Public-only is single-stage: run each --mix / --state-only arm separately on the same --split")
    if a.seeds < 1 or a.steps < 1:
        raise ValueError("--seeds and --steps must be positive")
    split = read_split(a.split)
    ids = set(split["train"] + split["val"] + split["test"])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    # Avoid stale metrics/checkpoints being ingested as a successful new run.
    if (out / "metrics.json").exists() or (out / "actual_data.json").exists():
        raise ValueError("Output already has a run manifest/metrics; choose a fresh --out: %s" % out)
    parts = []
    for src in sources:
        # Discover IDs first, without materializing arrays for unselected episodes.
        root = Path(getattr(a, src))
        if not root.exists():
            raise ValueError("Missing requested source %s: %s" % (src, root))
        known = {json.loads(p.read_text()).get("episode_id") for p in tx.iter_episode_paths(root)}
        selected = ids & known
        if not selected:
            raise ValueError("Requested source %s has no selected episodes: %s" % (src, root))
        parts.append(tx.build_external(src, root, out / "cache", use_img=not a.state_only, episode_ids=selected))
    actual_ids = [str(d["episode_id"][np.flatnonzero(d["episode"] == ep)[0]])
                  for d in parts for ep in np.unique(d["episode"])]
    if len(set(actual_ids)) != len(actual_ids) or set(actual_ids) != ids:
        raise ValueError("Split IDs missing or duplicated across sources: %s" % sorted(ids - set(actual_ids)))
    d = tx.concat(parts)
    rates = np.unique(d["fps"])
    if len(rates) != 1:
        raise ValueError("Mixed label FPS changes the prediction horizon; resample explicitly first: %s" % rates)
    starts = tx.ev.chunk_starts(d["episode"], tx.H, 1)
    selections = {}
    for part in ("train", "val", "test"):
        s = starts[np.isin(d["episode_id"][starts], split[part])]
        if part == "train":
            s = tx.seg_chunk_filter(d, s)
        s = tx.useful(d, s)
        if part == "train" and a.loss_move_cm:
            y, _ = tx.ev.gather(d, s)
            s = s[tx.moving_mask_from_Y(y, a.loss_move_cm)]
        if not len(s):
            raise ValueError("Zero effective windows in %s after QC/masks/motion filter" % part)
        if set(d["episode_id"][s]) != set(split[part]):
            raise ValueError("Selected %s episodes have zero effective windows" % part)
        selections[part] = s
    tr, va, te = (selections[p] for p in ("train", "val", "test"))
    if set(d["source"][tr]) != set(sources):
        raise ValueError("Every requested source must contribute effective training samples")
    manifest = {"schema_version": 1, "mode": "public_only", "argv": vars(a), "split": split,
                "split_file": file_identity(a.split, sha=True), "fps": float(rates[0]), "horizon": tx.H,
                "image_mode": "state_only" if a.state_only else "rgb", "episodes": [], "partitions": {},
                "sampling": [], "status": "prepared", "decision_eligible": False,
                "qc_inputs": [file_identity(p, sha=True) for p in (a.qc_filter or []) + (a.qc_segments or [])],
                "code_files": [file_identity(Path(__file__).with_name(name), sha=True)
                               for name in ("train_xhey.py", "xhey_public.py", "xhey_stats.py")],
                "note": "Phase-0 plumbing only; no locked scientific decision policy or license approval implied."}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parents[1])
    manifest["code_revision"] = revision.stdout.strip() if revision.returncode == 0 else None
    for ep in np.unique(d["episode"]):
        rows = np.flatnonzero(d["episode"] == ep); i = rows[0]
        item = {"id": str(d["episode_id"][i]), "source": str(d["source"][i]), "rows": len(rows),
                "image_rows": int(d["f_has"][rows].sum()), "episode": file_identity(d["episode_path"][i], sha=True)}
        item["group"] = split["groups"][item["id"]]
        metadata = json.loads(Path(d["episode_path"][i]).read_text())
        item["recorded_source"] = metadata.get("source")
        item["source_path"] = metadata.get("source_path")
        item["license_note"] = metadata.get("license_note")
        if not a.state_only:
            item["video"] = file_identity(d["video_path"][i])
        manifest["episodes"].append(item)
    for part, s in selections.items():
        manifest["partitions"][part] = {
            "units": len({split["groups"][i] for i in d["episode_id"][s]}),
            "sources": {src: {"windows": int(np.sum(d["source"][s] == src)),
                              "image_windows": int(d["f_has"][s[d["source"][s] == src]].sum()),
                              "eligible_weight": float(np.mean(d["source"][s] == src))}
                        for src in sources if np.any(d["source"][s] == src)}}
    mf = out / "actual_data.json"
    def save_manifest():
        mf.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    save_manifest()
    print("actual_data", json.dumps(manifest["partitions"]), flush=True)
    if a.prepare_only:
        return 0
    X, Y, V = tx.xyv(d, tr, not a.state_only)
    Xv, Yv, Vv = tx.xyv(d, va, not a.state_only)
    per, errors, curves = {}, {}, []
    test_eps = [ep for ep in np.unique(d["episode"]) if np.any(d["episode"][te] == ep)]
    names = [str(d["episode_id"][np.flatnonzero(d["episode"] == ep)[0]]) for ep in test_eps]
    # `te` and all per-episode slices use the same order for every method/seed.
    def add(name, pred):
        per[name], errors[name] = [], []
        for ep in test_eps:
            mask = d["episode"][te] == ep
            per[name].append(tx.ev.metrics_for(d, te[mask], pred[mask], [1, 8, tx.H]))
            ca = tx.ht.chunk_ade(d, te[mask], pred[mask])
            if not np.isfinite(ca).any():
                raise ValueError("Test episode has zero full-horizon valid hand samples: %s" % ep)
            errors[name].append(ca)
    if not a.no_eval:
        add("zero", np.tile(tx.IDENT, (len(te), tx.H, 1)))
        add("constant_velocity", constant_velocity(tx, d, te))
    for seed in range(a.seeds):
        m = tx.XHEY(X.shape[1], seed, hidden=a.hidden, dropout=a.dropout)
        m.set_norm(X, Y)  # Never use validation/test/probe rows for normalization.
        curve = m.fit(X, Y, V, Xv, Yv, Vv, steps=a.steps, lr=3e-4 * a.lr_scale,
                      sample_sources=d["source"][tr])
        curves.append({"mix": a.mix, "seed": seed, "train": curve})
        manifest["sampling"].append({"seed": seed, "sources": m.sample_manifest})
        save_manifest()
        if not a.no_eval:
            add("%s#%d" % (a.mix, seed), m(tx.feats(d, te, not a.state_only)))
        if not a.no_final:
            # Save the selected model; never refit on validation/test or iPhone.
            m.save(out / ("xhey.pt" if seed == 0 else "xhey_seed%d.pt" % seed),
                   {"mix": a.mix, "split_sha256": manifest["split_file"]["sha256"],
                    "fps": float(rates[0]), "state_only": a.state_only})
    manifest["status"] = "complete"
    if not a.no_eval:
        yt, _ = tx.ev.gather(d, te)
        moving = np.stack([np.linalg.norm(yt[:, :, 7*h:7*h+3].sum(1), axis=-1) * 100 > a.move_cm
                           for h in range(2)], 1)
        results = tx.summarize_results(per, errors, names, moving, [split["groups"][i] for i in names])
        report = {"model": "XHEY", "mode": "public_only", "clips": names, "mixes": [a.mix],
                  "fps": float(rates[0]), "horizon": tx.H, "seeds": a.seeds,
                  "results": results, "curves": curves, "actual_data": manifest}
        (out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        np.savez(out / "window_errors.npz", starts=te, episode_id=d["episode_id"][te], moving=moving,
                 groups=np.asarray([split["groups"][i] for i in d["episode_id"][te]]),
                 **{name: np.concatenate(parts) for name, parts in errors.items()})
    save_manifest()
    return 0
