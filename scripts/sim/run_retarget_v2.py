# -*- coding: utf-8 -*-
"""HOT3D 物体位姿重定向，以及同一套改进在 101 条 EgoDex 上的消融。

数字只写这次跑出来的。凑不满 20/40/80 就写实际条数和原因，不补成功率。

    python scripts/retarget/fetch_hot3d.py --dest /tmp/hot3d
    python scripts/sim/run_retarget_v2.py --hot3d /tmp/hot3d --out docs/retarget_hot3d_results.json
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from egodata.egodex import load_episode_hdf5  # noqa: E402
from retarget.arkit import (  # noqa: E402
    episode_center,
    fit_shared_calibration,
    height_gain,
    map_centered,
    permute_arkit_xyz,
)
from retarget.calibrate import AxisCalibration  # noqa: E402
from retarget.events import MIN_ROBOT_SEPARATION_M  # noqa: E402
from retarget.frames import CONTROL_DT, R_DOWN, SUCCESS_XY_M, Z_GRASP  # noqa: E402
from retarget.grasp_timing import (  # noqa: E402
    LEAD_FRAMES,
    approximate_joints,
    coupling_mask,
    estimator_grip_egodex,
    estimator_grip_hot3d,
    grip_from_closed,
    grip_from_distance,
)
from retarget.hot3d_clip import (  # noqa: E402
    bbox_centers,
    find_pick_place,
    iter_clip_dirs,
    load_clip,
    window_slice,
)
from retarget.object_centric import align_grasp_height, object_centric_ee, robot_relative  # noqa: E402
from retarget.paths import path_duration, scripted_segments  # noqa: E402
from retarget.retarget import retarget_episode  # noqa: E402
from sim.arm import PickPlaceEnv  # noqa: E402
from sim.replay_correct import replay_online  # noqa: E402
from sim.rollout import follow_reference, replay_joints, run_policy  # noqa: E402
from sim.run_egodex_benchmark import (  # noqa: E402
    SCALE_POINTS,
    TRAIN_SEEDS,
    _accept_mapped,
    _episode_row,
    _fit_policy,
    _held_out_count,
    _load_prepared,
    _rate_stats,
    _resample,
    _retarget_one,
    _stack_prefix,
    _summary,
)
from sim.scene import geometry_key, plan_geometry, scene_xml  # noqa: E402

ABLATIONS = ("rigid", "timing_gt", "timing_estimator", "object_centric", "online", "full")


def _identity():
    return AxisCalibration(scale=np.ones(3), offset=np.zeros(3))


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _retarget_xyz(env, ee_xyz, times, grip):
    quat = np.zeros((ee_xyz.shape[0], 4), dtype=float)
    quat[:, 3] = 1.0
    wrist = np.concatenate([np.asarray(ee_xyz, dtype=float), quat], axis=1)
    episode = {"timestamps": np.asarray(times, dtype=float), "hands": {"right": {"wrist_pose": wrist}}}
    return retarget_episode(episode, env, _identity(), side="right", grip=grip, target_rotation=R_DOWN)


def _with_grip(result, grip):
    result.grip = np.asarray(grip, dtype=float)
    return result


def _safe_index_row(index, cube, goal, result):
    if isinstance(index, str):
        row = _episode_row(0, cube, goal, result)
        row["index"] = index
        return row
    return _episode_row(index, cube, goal, result)


def _row(name, cube, goal, played, extra=None):
    row = _safe_index_row(name, cube, goal, played)
    final = np.asarray(played["cube"], dtype=float)
    row["cube_xy_moved_m"] = float(np.linalg.norm(final[:2] - np.asarray(cube, dtype=float)[:2]))
    if extra:
        row.update(extra)
    return row


def _summary_moved(rows):
    summary = _summary(rows)
    if rows:
        summary["n_cube_moved_under_1cm"] = int(sum(1 for row in rows if row.get("cube_xy_moved_m", 1) < 0.01))
    return summary


def _load_models(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(key): value for key, value in raw.items()}


def _scan_hot3d(root, models):
    segments = []
    scanned = {"n_clips": 0, "n_objects": 0, "n_segments": 0, "missing_bbox": 0, "clips": []}
    for directory in iter_clip_dirs(root):
        clip = load_clip(directory)
        scanned["n_clips"] += 1
        found = 0
        for obj in clip["objects"].values():
            scanned["n_objects"] += 1
            info = models.get(str(obj["bop_id"]))
            if info is None:
                scanned["missing_bbox"] += 1
                continue
            plan = plan_geometry(info)
            center_local = plan["bbox_center_local"]
            centers = bbox_centers(obj["origin"], obj["quat_wxyz"], center_local)
            picked = find_pick_place(centers, clip["left_xyz"], clip["right_xyz"], fps=clip["fps"])
            if picked is None:
                continue
            lo, hi = window_slice(clip["num_frames"], picked["start"], picked["end"], fps=clip["fps"])
            grasp = int(picked["start"] - lo)
            release = int(picked["end"] - lo)
            side = picked["side"]
            wrist = clip["left_xyz"] if side == "left" else clip["right_xyz"]
            other = clip["right_xyz"] if side == "left" else clip["left_xyz"]
            wrist_q = clip["left_wxyz"] if side == "left" else clip["right_wxyz"]
            other_q = clip["right_wxyz"] if side == "left" else clip["left_wxyz"]
            window_objects = []
            for other_obj in clip["objects"].values():
                window_objects.append({
                    "bop_id": other_obj["bop_id"],
                    "name": other_obj["name"],
                    "origin": other_obj["origin"][lo:hi],
                    "quat_wxyz": other_obj["quat_wxyz"][lo:hi],
                })
            segments.append({
                "clip": clip["clip"],
                "sequence_id": clip["sequence_id"],
                "device": clip["device"],
                "bop_id": str(obj["bop_id"]),
                "name": obj["name"],
                "side": side,
                "wrist": wrist[lo:hi],
                "other": other[lo:hi],
                "wrist_wxyz": wrist_q[lo:hi],
                "other_wxyz": other_q[lo:hi],
                "center": centers[lo:hi],
                "objects": window_objects,
                "grasp": grasp,
                "release": release,
                "times": np.arange(hi - lo, dtype=float) / float(clip["fps"]),
                "plan": plan,
                "hand_distance_m": picked["hand_distance_m"],
                "travel_xz_m": picked["travel_xz_m"],
                "lift_m": picked["lift_m"],
            })
            found += 1
        scanned["clips"].append({"clip": clip["clip"], "sequence_id": clip["sequence_id"], "n_segments": found})
        print("scan", clip["clip"], "segments", found, flush=True)
    scanned["n_segments"] = len(segments)
    return segments, scanned


def _accept_hot3d(segments, models):
    if len(segments) < 2:
        return None, [], {"n_segments": len(segments)}
    centered_rows = []
    grasp_heights = []
    carry_heights = []
    for seg in segments:
        permuted = permute_arkit_xyz(seg["wrist"])
        center = episode_center(permuted)
        seg["origin_center"] = center
        centered = permuted - center
        centered_rows.append(centered)
        grasp_heights.append(float(centered[seg["grasp"], 2]))
        carry_heights.append(float(np.max(centered[seg["grasp"]:seg["release"] + 1, 2])))
    calibration, fit = fit_shared_calibration(np.vstack(centered_rows), grasp_heights, carry_heights)
    accepted = []
    too_close = 0
    non_finite = 0
    for seg in segments:
        if not np.isfinite(seg["wrist"]).all() or not np.isfinite(seg["center"]).all():
            non_finite += 1
            continue
        seg["wrist_robot"] = map_centered(calibration, seg["wrist"], seg["origin_center"])
        seg["obj_robot"] = map_centered(calibration, seg["center"], seg["origin_center"])
        grasp = seg["grasp"]
        release = seg["release"]
        separation = float(np.linalg.norm(seg["obj_robot"][grasp, :2] - seg["obj_robot"][release, :2]))
        seg["separation_m"] = separation
        if separation < MIN_ROBOT_SEPARATION_M:
            too_close += 1
            continue
        relative = robot_relative(seg["wrist"], seg["center"], seg["plan"]["scale"])
        seg["relative"] = relative
        seg["ee_rigid"] = seg["wrist_robot"]
        seg["ee_object"] = object_centric_ee(
            seg["obj_robot"], relative, grasp, release, grasp_z=seg["plan"]["grasp_z"]
        )
        seg["grip_distance"] = grip_from_distance(seg["wrist"], seg["center"])
        closed = coupling_mask(seg["wrist"], seg["center"], seg["other"], fps=30.0)
        seg["grip_gt"] = grip_from_closed(closed, lead_frames=LEAD_FRAMES)
        seg["coupling_fraction"] = float(np.mean(closed))
        timestamps = seg["times"].tolist()
        bbox = {}
        known = []
        for obj in seg["objects"]:
            if obj["bop_id"] == seg["bop_id"]:
                info_plan = seg["plan"]
            elif models.get(str(obj["bop_id"])) is None:
                continue
            else:
                info_plan = plan_geometry(models[str(obj["bop_id"])])
            bbox[obj["bop_id"]] = (info_plan["bbox_min"], info_plan["bbox_size"])
            known.append(obj)
        joints = {
            "left": approximate_joints(
                seg["wrist"] if seg["side"] == "left" else seg["other"],
                seg["wrist_wxyz"] if seg["side"] == "left" else seg["other_wxyz"],
                "left",
            ),
            "right": approximate_joints(
                seg["wrist"] if seg["side"] == "right" else seg["other"],
                seg["wrist_wxyz"] if seg["side"] == "right" else seg["other_wxyz"],
                "right",
            ),
        }
        estimated = estimator_grip_hot3d(joints, known, timestamps, bbox)
        seg["grip_est"] = estimated["grip"][seg["side"]]
        seg["estimator"] = {
            "source": estimated["source"],
            "joint_source": estimated["joint_source"],
            "surface_source": estimated["surface_source"],
            "n_grasp_frames": int(np.sum(np.asarray(estimated["states"][seg["side"]]) == "grasp")),
        }
        seg["cube"] = seg["obj_robot"][grasp, :2].copy()
        seg["goal"] = seg["obj_robot"][release, :2].copy()
        seg["offset_rigid"] = seg["ee_rigid"][grasp, :2] - seg["cube"]
        seg["offset_object"] = seg["ee_object"][grasp, :2] - seg["cube"]
        accepted.append(seg)
    meta = {
        "scale": [float(v) for v in calibration.scale],
        "offset": [float(v) for v in calibration.offset],
        "human_min": [float(v) for v in fit["human_min"]],
        "human_max": [float(v) for v in fit["human_max"]],
        "z_from_grasp_median": bool(fit["z_from_grasp_median"]),
        "n_too_close_after_map": int(too_close),
        "n_non_finite": int(non_finite),
        "min_robot_separation_m": float(MIN_ROBOT_SEPARATION_M),
        "centering": "subtract per-segment median wrist before the shared workspace scale",
        "object_relative_scale": "geometry uniform scale (fit longest side to 8 cm, never enlarge), not the workspace scale",
    }
    return calibration, accepted, meta


def _env_for(plan, cache, directory):
    key = geometry_key(plan)
    if key in cache:
        return cache[key]
    path = directory / ("scene_%d.xml" % len(cache))
    path.write_text(scene_xml(plan), encoding="utf-8")
    env = PickPlaceEnv(path)
    env.rest_z = float(plan["rest_z"])
    env.grasp_z = float(plan["grasp_z"])
    cache[key] = env
    return env


def _play_joint(env, cube, goal, retargeted, grip):
    return replay_joints(env, cube, goal, retargeted.times, retargeted.q, grip)


def _ablate_hot3d(accepted, directory):
    cache = {}
    library = []
    for seg in accepted:
        env = _env_for(seg["plan"], cache, directory)
        rigid = _retarget_xyz(env, seg["ee_rigid"], seg["times"], seg["grip_distance"])
        obj = _retarget_xyz(env, seg["ee_object"], seg["times"], seg["grip_distance"])
        library.append({"seg": seg, "env": env, "rigid": rigid, "object": obj})
        print("ik", seg["clip"], seg["name"], "pinchable", seg["plan"]["pinchable"], flush=True)
    names = {
        "rigid": "手腕工作空间映射 + 距离当开合",
        "timing_gt": "同一条手腕，夹爪按位姿耦合（不是网格接触）",
        "timing_estimator": "同一条手腕，夹爪按启发式；关节是手腕系固定偏移，表面是包围盒角点",
        "object_centric": "相对物体的手腕，偏移乘几何缩放，开环关节回放",
        "online": "手腕轨迹加在线伺服和一次再抓",
        "full": "物体相对轨迹 + 位姿耦合夹爪 + 在线伺服和一次再抓",
    }
    buckets = {name: [] for name in ABLATIONS}
    for entry in library:
        seg = entry["seg"]
        env = entry["env"]
        cube, goal = seg["cube"], seg["goal"]
        label = "%s:%s" % (seg["clip"], seg["name"])
        extra = {
            "label": label,
            "clip": seg["clip"],
            "object": seg["name"],
            "bop_id": seg["bop_id"],
            "pinchable": bool(seg["plan"]["pinchable"]),
            "geometry": seg["plan"]["kind"],
            "geometry_scale": float(seg["plan"]["scale"]),
            "coupling_fraction": float(seg["coupling_fraction"]),
            "estimator_grasp_frames": int(seg["estimator"]["n_grasp_frames"]),
            "offset_rigid_m": float(np.linalg.norm(seg["offset_rigid"])),
            "offset_object_m": float(np.linalg.norm(seg["offset_object"])),
        }
        played = {
            "rigid": _play_joint(env, cube, goal, entry["rigid"], seg["grip_distance"]),
            "timing_gt": _play_joint(env, cube, goal, entry["rigid"], seg["grip_gt"]),
            "timing_estimator": _play_joint(env, cube, goal, entry["rigid"], seg["grip_est"]),
            "object_centric": _play_joint(env, cube, goal, entry["object"], seg["grip_distance"]),
        }
        played["online"] = replay_online(
            env, cube, goal, entry["rigid"].times, entry["rigid"].q, seg["grip_distance"],
            entry["rigid"].ee_target, seg["grasp"], seg["release"], seg["offset_rigid"],
            seg["plan"]["grasp_z"], rest_z=seg["plan"]["rest_z"], record=False,
        )
        played["full"] = replay_online(
            env, cube, goal, entry["object"].times, entry["object"].q, seg["grip_gt"],
            entry["object"].ee_target, seg["grasp"], seg["release"], seg["offset_object"],
            seg["plan"]["grasp_z"], rest_z=seg["plan"]["rest_z"], record=False,
        )
        for name, result in played.items():
            row = _row(len(buckets[name]), cube, goal, result, extra)
            row["regrasped"] = bool(result.get("regrasped", False))
            buckets[name].append(row)
        print(
            "replay", label,
            " ".join("%s=%s" % (name, int(played[name]["success"])) for name in ABLATIONS),
            flush=True,
        )
    report = {"definitions": names, "by_ablation": {}}
    for name in ABLATIONS:
        report["by_ablation"][name] = _summary_moved(buckets[name])
    return library, report


def _closed_loop(library, train_items, eval_items, full_name, seed_count, quick):
    """``train_items`` 是训练池里 full 重放成功的条目。评测场景是留出的，不按成功过滤。"""
    if len(train_items) < 2 or len(eval_items) < 1:
        return {
            "blocker": "训练池成功 %d 条、留出 %d 条，不够做闭环。" % (len(train_items), len(eval_items)),
        }
    human_demos = []
    robot_demos = []
    for item in train_items:
        human_demos.append(item["human_demo"])
        robot = follow_reference(
            item["env"],
            item["cube"],
            item["goal"],
            scripted_segments(item["cube"], item["goal"], arc_m=0.0, z_grasp=item["grasp_z"]),
            record=True,
        )
        robot_demos.append(robot)
        print("robot demo", item["label"], robot["success"], flush=True)
    if any(demo["features"].shape[0] < 2 for demo in human_demos):
        return {"blocker": "有的成功重放没有采到特征，不训练。"}
    step_count = int(round(path_duration(scripted_segments(np.zeros(2), np.array([0.12, 0.0]))) / CONTROL_DT))
    horizon = path_duration(scripted_segments(np.zeros(2), np.array([0.12, 0.0])))
    scales = [int(n) for n in SCALE_POINTS if int(n) <= len(human_demos)]
    if 80 not in scales and len(human_demos) >= 2 and len(human_demos) not in scales:
        scales.append(len(human_demos))
    if not scales:
        return {
            "blocker": "成功重放 %d 条，凑不满 20/40/80，而且少于 2 条，不训练。" % len(human_demos),
            "n_train_replay_success": len(human_demos),
        }
    if quick:
        scales = [min(4, len(human_demos))]
    seeds = list(TRAIN_SEEDS[: int(seed_count)])
    oracle_rows = []
    for item in eval_items:
        oracle = follow_reference(
            item["env"],
            item["cube"],
            item["goal"],
            scripted_segments(item["cube"], item["goal"], arc_m=0.0, z_grasp=item["grasp_z"]),
            record=False,
        )
        oracle_rows.append(_safe_index_row(item["label"], item["cube"], item["goal"], oracle))
        print("oracle", item["label"], oracle["success"], flush=True)
    closed = {}
    for scale in scales:
        h_feat, h_act = _stack_prefix(human_demos, scale, step_count)
        r_feat, r_act = _stack_prefix(robot_demos, scale, step_count)
        half = max(1, scale // 2) if scale >= 2 else 1
        # 条数是 1 时混合没法对半，仍各用这一条，并在结果里写明。
        m_feat = np.vstack([h_feat[: half * step_count], r_feat[: half * step_count]])
        m_act = np.vstack([h_act[: half * step_count], r_act[: half * step_count]])
        per_seed = {"bc_retargeted_human": [], "bc_robot": [], "bc_mixed": []}
        seed_details = []
        for seed in seeds:
            policy_h, stats_h = _fit_policy(h_feat, h_act, seed)
            policy_r, stats_r = _fit_policy(r_feat, r_act, seed + 100)
            policy_m, stats_m = _fit_policy(m_feat, m_act, seed + 200)
            rows = {key: [] for key in per_seed}
            for item in eval_items:
                human = run_policy(item["env"], policy_h, item["cube"], item["goal"], horizon)
                robot = run_policy(item["env"], policy_r, item["cube"], item["goal"], horizon)
                mixed = run_policy(item["env"], policy_m, item["cube"], item["goal"], horizon)
                rows["bc_retargeted_human"].append(_safe_index_row(item["label"], item["cube"], item["goal"], human))
                rows["bc_robot"].append(_safe_index_row(item["label"], item["cube"], item["goal"], robot))
                rows["bc_mixed"].append(_safe_index_row(item["label"], item["cube"], item["goal"], mixed))
            summarized = {key: _summary(value) for key, value in rows.items()}
            for key, summary in summarized.items():
                per_seed[key].append(summary["success_rate"])
            seed_details.append({
                "seed": int(seed),
                "train": {"human": stats_h, "robot": stats_r, "mixed": stats_m},
                "closed_loop": summarized,
            })
            print(
                full_name, "scale", scale, "seed", seed,
                summarized["bc_retargeted_human"]["n_success"],
                summarized["bc_robot"]["n_success"],
                summarized["bc_mixed"]["n_success"],
                "of", len(eval_items),
                flush=True,
            )
        note = None
        if scale < 20:
            note = "成功重放只有 %d 条，没有 20/40/80 这一档。" % len(human_demos)
        closed["n_%d" % scale] = {
            "n_demos_per_source": int(scale),
            "mixed_n": int(min(half, scale) * 2 if scale >= 2 else 2),
            "mixed_rule": "前一半人类加重放同一场景的机器人演示。只有 1 条时两边都用这一条，混合条数是 2，和单一来源不相等。",
            "note": note,
            "bc_retargeted_human": _rate_stats(per_seed["bc_retargeted_human"]),
            "bc_robot": _rate_stats(per_seed["bc_robot"]),
            "bc_mixed": _rate_stats(per_seed["bc_mixed"]),
            "seeds": seed_details,
        }
    return {
        "method": full_name,
        "n_train_replay_success": len(train_items),
        "horizon_s": horizon,
        "resampled_steps_per_demo": step_count,
        "scales": scales,
        "scale_note": "20/40/80 只在成功重放不少于该数时才有。不够就用实际条数。",
        "oracle_scripted": _summary(oracle_rows),
        "closed_loop": closed,
    }


def _record_full(entry, grip, ee_result, offset, grasp_z, rest_z, grasp, release):
    return replay_online(
        entry["env"], entry["cube"], entry["goal"],
        ee_result.times, ee_result.q, grip, ee_result.ee_target,
        grasp, release, offset, grasp_z, rest_z=rest_z, record=True,
    )


def run_hot3d(root, models_path, seed_count, quick):
    models = _load_models(models_path)
    clip_root = Path(root) / "clips"
    if not clip_root.exists():
        clip_root = Path(root)
    segments, scanned = _scan_hot3d(clip_root, models)
    report = {
        "human_source": "HOT3D-Clips train_quest3, json only, from bop-benchmark/hot3d",
        "mano": "not used. No skin-contact ground truth. Coupling is wrist/object co-motion. Estimator joints are a fixed wrist-frame template.",
        "scene": "box or cylinder from models_info bbox, uniform scale so the longest side is at most 8 cm, never enlarged. Jaw axis is a scaled side in 2.6–7.0 cm. Orientation in sim is axis-aligned, success is xy <= 4 cm.",
        "scan": scanned,
        "thresholds": {
            "lift_m": 0.06,
            "place_tol_m": 0.04,
            "travel_xz_m": 0.08,
            "hand_near_m": 0.25,
            "couple_speed_m_s": 0.05,
            "couple_cos": 0.5,
            "lead_frames": LEAD_FRAMES,
            "servo_step_m": 0.015,
            "regrasp_up_m": 0.03,
        },
    }
    if len(segments) < 2:
        report["blocker"] = "扫描到的拿起再放下只有 %d 段，不够标定。" % len(segments)
        return report
    _calibration, accepted, meta = _accept_hot3d(segments, models)
    report["calibration"] = meta
    report["n_accepted"] = len(accepted)
    if len(accepted) < 1:
        report["blocker"] = "映射后起点和终点都近于 6 cm，没有可重放的段。"
        return report
    gains = [height_gain(seg["wrist"], seg["grasp"], seg["release"]) for seg in accepted]
    gains = np.asarray(gains, dtype=float)
    report["wrist_gain"] = {
        "median_gain_xyz_m": [float(v) for v in np.median(gains, axis=0)],
        "fraction_y_gain_over_0_02": float(np.mean(gains[:, 1] > 0.02)),
    }
    accepted = sorted(accepted, key=lambda seg: (seg["clip"], seg["bop_id"], seg["name"]))
    n_eval = _held_out_count(len(accepted), 50, quick)
    eval_keys = {(seg["clip"], seg["bop_id"]) for seg in accepted[len(accepted) - n_eval:]} if n_eval else set()
    with tempfile.TemporaryDirectory(prefix="hot3d_scene_") as tmp:
        directory = Path(tmp)
        library, ablation = _ablate_hot3d(accepted, directory)
        report["replay"] = ablation
        # 训练用 full。留出的段不进训练集。特征在这里补采，避免消融循环里每条都记录。
        train_items = []
        eval_items = []
        full_rows = {row["label"]: row for row in ablation["by_ablation"]["full"]["episodes"]}
        for entry in library:
            seg = entry["seg"]
            label = "%s:%s" % (seg["clip"], seg["name"])
            item = {
                "label": label,
                "env": entry["env"],
                "cube": seg["cube"],
                "goal": seg["goal"],
                "grasp_z": float(seg["plan"]["grasp_z"]),
                "held_out": (seg["clip"], seg["bop_id"]) in eval_keys,
            }
            if item["held_out"]:
                eval_items.append(item)
                continue
            if not full_rows[label]["success"]:
                continue
            demo = _record_full(
                {"env": entry["env"], "cube": seg["cube"], "goal": seg["goal"]},
                seg["grip_gt"], entry["object"], seg["offset_object"],
                seg["plan"]["grasp_z"], seg["plan"]["rest_z"], seg["grasp"], seg["release"],
            )
            print("full record", label, demo["success"], flush=True)
            if not demo["success"] or demo["features"].shape[0] < 2:
                continue
            item["human_demo"] = demo
            train_items.append(item)
        report["split"] = {
            "n_accepted": len(accepted),
            "n_eval": len(eval_items),
            "n_train_pool": len(accepted) - len(eval_items),
            "n_train_full_success": len(train_items),
            "eval": [item["label"] for item in eval_items],
            "rule": "按 clip 名和物体 id 排序，最后 n_eval 条留出。行为克隆只用留出之前、full 重放成功的段。",
        }
        report["bc"] = _closed_loop(library, train_items, eval_items, "hot3d_full", seed_count, quick)
        # 场景文件删掉之前，闭环已经跑完。环境对象仍指向已加载的模型，不依赖文件。
    return report


def _egodex_variants(env, prepared_root, accepted):
    """在同一条 IK 上换夹爪，物体坐标系只改高度。"""
    library = []
    for item in accepted:
        path = Path(prepared_root) / ("%d.hdf5" % int(item["index"]))
        episode = load_episode_hdf5(path)
        hand = episode["hands"][item["side"]]
        joints = np.stack([np.asarray(hand["joints"][int(frame)], dtype=float) for frame in item["frames"]])
        item["joints"] = joints
        retargeted, cube, goal = _retarget_one(env, item)
        pickup = int(item["pickup_index"])
        release = int(item["release_index"])
        closed = np.zeros(item["grip"].shape[0], dtype=bool)
        closed[pickup:release + 1] = True
        grip_interval = grip_from_closed(closed, lead_frames=LEAD_FRAMES)
        estimated = estimator_grip_egodex(joints, item["side"], pickup, item["times"].tolist())
        ee_object = align_grasp_height(retargeted.ee_target, pickup, Z_GRASP)
        object_rt = _retarget_xyz(env, ee_object, retargeted.times, item["grip"])
        library.append({
            "item": item,
            "rigid": retargeted,
            "object": object_rt,
            "cube": cube,
            "goal": goal,
            "grip_interval": grip_interval,
            "grip_est": estimated["grip"],
            "estimator_grasp_frames": int(sum(state == "grasp" for state in estimated["states"])),
            "estimator_source": estimated["source"],
        })
        print("egodex ik", item["index"], flush=True)
    return library


def run_egodex(root, seed_count, quick, limit):
    prepared, load_meta = _load_prepared(root, limit=limit)
    report = {
        "human_source": "EgoDex test/basic_pick_place, same filter as docs/retarget_egodex_results.json",
        "object_pose": "none. Improvements that need a mesh are not applied. Estimator cube is placed at the fingertip centroid at pickup.",
        "load": load_meta,
    }
    if len(prepared) < 2:
        report["blocker"] = "抓放筛选后少于 2 条。"
        return report
    _calibration, accepted, meta = _accept_mapped(prepared)
    report["calibration"] = meta
    accepted = sorted(accepted, key=lambda item: int(item["index"]))
    n_eval = _held_out_count(len(accepted), 50, quick)
    eval_ids = {int(item["index"]) for item in accepted[len(accepted) - n_eval:]} if n_eval else set()
    env = PickPlaceEnv()
    library = _egodex_variants(env, root, accepted)
    definitions = {
        "rigid": "和上一份结果相同的开环：映射后的手腕 + 按开合距离连续开合",
        "timing_gt": "同一条末端。夹爪在握住到松开之间合上（仍由开合滞回定区间，不是物体真值），提前 3 帧",
        "timing_estimator": "同一条末端。夹爪用 PR #27 启发式；物体是指尖中心上的方块，不是数据集位姿",
        "object_centric": "没有物体轨迹。只把抓住帧的高度对齐到夹取高度，开环重放",
        "online": "原手腕轨迹，靠近时伺服到方块（偏移为 0）并最多再抓一次",
        "full": "高度对齐 + 握住区间夹爪 + 在线伺服和一次再抓",
    }
    buckets = {name: [] for name in ABLATIONS}
    for entry in library:
        item = entry["item"]
        cube, goal = entry["cube"], entry["goal"]
        extra = {
            "estimator_grasp_frames": entry["estimator_grasp_frames"],
            "held_out": int(item["index"]) in eval_ids,
        }
        played = {
            "rigid": _play_joint(env, cube, goal, entry["rigid"], item["grip"]),
            "timing_gt": _play_joint(env, cube, goal, entry["rigid"], entry["grip_interval"]),
            "timing_estimator": _play_joint(env, cube, goal, entry["rigid"], entry["grip_est"]),
            "object_centric": _play_joint(env, cube, goal, entry["object"], item["grip"]),
        }
        played["online"] = replay_online(
            env, cube, goal, entry["rigid"].times, entry["rigid"].q, item["grip"],
            entry["rigid"].ee_target, int(item["pickup_index"]), int(item["release_index"]),
            np.zeros(2), Z_GRASP, record=False,
        )
        played["full"] = replay_online(
            env, cube, goal, entry["object"].times, entry["object"].q, entry["grip_interval"],
            entry["object"].ee_target, int(item["pickup_index"]), int(item["release_index"]),
            np.zeros(2), Z_GRASP, record=False,
        )
        for name, result in played.items():
            row = _row(int(item["index"]), cube, goal, result, extra)
            row["regrasped"] = bool(result.get("regrasped", False))
            buckets[name].append(row)
        print(
            "egodex", item["index"],
            " ".join("%s=%s" % (name, int(played[name]["success"])) for name in ABLATIONS),
            flush=True,
        )
    report["replay"] = {"definitions": definitions, "by_ablation": {name: _summary_moved(buckets[name]) for name in ABLATIONS}}
    full_ok = {int(row["index"]): row["success"] for row in buckets["full"]}
    train_items = []
    eval_items = []
    by_entry = {int(entry["item"]["index"]): entry for entry in library}
    for entry in library:
        item = entry["item"]
        packed = {
            "label": int(item["index"]),
            "env": env,
            "cube": entry["cube"],
            "goal": entry["goal"],
            "grasp_z": Z_GRASP,
            "held_out": int(item["index"]) in eval_ids,
        }
        if packed["held_out"]:
            eval_items.append(packed)
            continue
        if not full_ok.get(int(item["index"]), False):
            continue
        demo = _record_full(
            packed, entry["grip_interval"], entry["object"], np.zeros(2), Z_GRASP, None,
            int(item["pickup_index"]), int(item["release_index"]),
        )
        print("egodex full record", item["index"], demo["success"], flush=True)
        if not demo["success"] or demo["features"].shape[0] < 2:
            continue
        packed["human_demo"] = demo
        train_items.append(packed)
    report["split"] = {
        "n_accepted": len(accepted),
        "n_eval": len(eval_items),
        "n_train_pool": len(accepted) - len(eval_items),
        "n_train_full_success": len(train_items),
        "train_indices": [int(entry["item"]["index"]) for entry in library if int(entry["item"]["index"]) not in eval_ids],
        "eval_indices": sorted(eval_ids),
    }
    report["bc"] = _closed_loop(by_entry, train_items, eval_items, "egodex_full", seed_count, quick)
    return report


def run(hot3d, models, egodex, seed_count, quick, egodex_limit):
    started = time.time()
    report = {
        "success_xy_m": SUCCESS_XY_M,
        "success_z_m": [0.20, 0.30],
        "policy": "numpy MLP behavior cloning, state only, not image ACT",
        "robot": "mujoco franka-like 6-DoF + parallel gripper, CPU headless",
        "seeds": list(TRAIN_SEEDS[: int(seed_count)]),
    }
    print("hot3d", flush=True)
    report["hot3d"] = run_hot3d(hot3d, models, seed_count, quick)
    print("egodex", flush=True)
    report["egodex"] = run_egodex(egodex, seed_count, quick, egodex_limit)
    report["elapsed_s"] = time.time() - started
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="HOT3D 与 EgoDex 的重定向消融")
    parser.add_argument("--hot3d", default="/tmp/hot3d")
    parser.add_argument("--models", default=None)
    parser.add_argument("--egodex", default="data/egodex_basic_pick_place")
    parser.add_argument("--out", default="docs/retarget_hot3d_results.json")
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--egodex-limit", type=int, default=None)
    args = parser.parse_args(argv)
    models = args.models or str(Path(args.hot3d) / "models_info.json")
    limit = args.egodex_limit
    seed_count = 1 if args.quick else args.seed_count
    if args.quick and limit is None:
        limit = 40
    report = run(args.hot3d, models, args.egodex, seed_count, args.quick, limit)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out, "elapsed", round(report["elapsed_s"], 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
