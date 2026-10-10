# -*- coding: utf-8 -*-
"""物体接触和抓取。

真值来自手和物体表面的距离（HOT3D 的网格，或调用方给的顶点）。
自有数据没有网格真值时，用三条能从关节和物体位姿算出来的量做启发式：

1. 指尖到物体表面的距离。表面可以是网格、球，或相机深度差。
2. 握姿张合：拇指尖到食指尖的距离。
3. 手相对物体的速度：指尖中心相对物体中心，沿「离开物体」方向的分量。

状态只有四个：``open`` 张开、``pre_grasp`` 预备抓、``grasp`` 抓住、``release`` 放开。
接触默认带时间滤波：距离用分开的进入/离开两档（滞回），状态要连续保持
``dwell_s`` 秒才切换，所以跟帧率无关。``temporal_filter=False`` 关掉，回到单阈值、当帧生效。
可选的 ``pose_hook`` / ``contact_hook`` 用来接现成模型（FoundationPose 一类的物体位姿，
100DOH / ContactHands 一类的接触检测）。钩子不实现、也不下载那些模型。
"""
import numpy as np

from egodata.schema import GRASP_STATES

# MediaPipe 顺序里的指尖：拇指、食指、中指、无名指、小指。
FINGERTIPS = (4, 8, 12, 16, 20)

# 网格真值用的距离。5 mm 是一个还没在 HOT3D 上标定过的工作点。
GT_CONTACT_M = 0.005
GT_MIN_TIPS = 2

# 启发式比真值松：关节在骨头上，不在皮肤上。这些数同样没有用真实录像标定。
# 时间滤波默认开。没另外给 contact_on_m 时，进入阈值就是 contact_m；
# 离开阈值再远 contact_off_margin_m。dwell_s 是秒，不是帧数。
HEURISTIC_DEFAULTS = {
    "contact_m": 0.010,
    "approach_m": 0.050,
    "aperture_grasp_m": 0.080,
    "approach_speed_m_s": 0.02,
    "grasp_speed_m_s": 0.15,
    "min_tips": 2,
    "dwell_s": 0.10,
    "contact_off_margin_m": 0.010,
}


def min_surface_distance(points, vertices=None, faces=None, center=None, radius_m=None):
    """每个查询点到表面的距离（米，不为负）。穿进球体里面算 0。"""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if radius_m is not None:
        origin = np.zeros(3) if center is None else np.asarray(center, dtype=float)
        radial = np.linalg.norm(points - origin, axis=1)
        signed = radial - float(radius_m)
        return np.where(signed < 0.0, 0.0, signed)
    if vertices is None:
        raise ValueError("表面需要半径或顶点")
    vertices = np.asarray(vertices, dtype=float).reshape(-1, 3)
    if faces is not None and len(faces):
        return _distance_to_triangles(points, vertices, np.asarray(faces, dtype=int))
    return _distance_to_vertices(points, vertices)


def _distance_to_vertices(points, vertices):
    out = np.empty(points.shape[0], dtype=float)
    for index, point in enumerate(points):
        out[index] = float(np.linalg.norm(vertices - point, axis=1).min())
    return out


def _point_triangle(point, tri):
    """点到一个三角形的距离。来自 Real-Time Collision Detection 的区域划分。"""
    a, b, c = tri
    ab, ac, ap = b - a, c - a, point - a
    d1, d2 = float(np.dot(ab, ap)), float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return float(np.linalg.norm(ap))
    bp = point - b
    d3, d4 = float(np.dot(ab, bp)), float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return float(np.linalg.norm(bp))
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return float(np.linalg.norm(a + v * ab - point))
    cp = point - c
    d5, d6 = float(np.dot(ab, cp)), float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return float(np.linalg.norm(cp))
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return float(np.linalg.norm(a + w * ac - point))
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return float(np.linalg.norm(b + w * (c - b) - point))
    denom = va + vb + vc
    if denom == 0.0:
        return float(np.linalg.norm(ap))
    v = vb / denom
    w = vc / denom
    return float(np.linalg.norm(a + ab * v + ac * w - point))


def _closest_on_triangles(point, a, b, c):
    """一个点到很多三角形（a/b/c 各 Nx3）的最近距离，向量化版的 Ericson 区域划分。"""
    ab, ac, ap = b - a, c - a, point - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = point - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = point - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = va + vb + vc
        v = np.where(denom != 0.0, vb / denom, 0.0)
        w = np.where(denom != 0.0, vc / denom, 0.0)
        closest = a + ab * v[:, None] + ac * w[:, None]
        # 边 bc
        m = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
        wbc = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        closest = np.where(m[:, None], b + wbc[:, None] * (c - b), closest)
        # 边 ac
        m = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
        wac = d2 / (d2 - d6)
        closest = np.where(m[:, None], a + wac[:, None] * ac, closest)
        # 边 ab
        m = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
        vab = d1 / (d1 - d3)
        closest = np.where(m[:, None], a + vab[:, None] * ab, closest)
    closest = np.where(((d6 >= 0.0) & (d5 <= d6))[:, None], c, closest)
    closest = np.where(((d3 >= 0.0) & (d4 <= d3))[:, None], b, closest)
    closest = np.where(((d1 <= 0.0) & (d2 <= 0.0))[:, None], a, closest)
    return np.linalg.norm(closest - point, axis=1)


def _distance_to_triangles(points, vertices, faces):
    out = np.full(points.shape[0], np.inf, dtype=float)
    triangles = vertices[faces]
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    for point_index, point in enumerate(points):
        out[point_index] = float(np.min(_closest_on_triangles(point, a, b, c)))
    return out


def _pose_matrix(pose7):
    pose = np.asarray(pose7, dtype=float)
    x, y, z, qx, qy, qz, qw = pose
    norm = float(np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw))
    if norm == 0.0:
        qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
    else:
        qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    rotation = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=float)
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = [x, y, z]
    return matrix


def _transform_points(pose7, points):
    matrix = _pose_matrix(pose7)
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    return (matrix[:3, :3] @ points.T).T + matrix[:3, 3]


def _transform_point(pose7, point):
    return _transform_points(pose7, np.asarray(point, dtype=float).reshape(1, 3))[0]


def _as_bool(value):
    return bool(value) if isinstance(value, (bool, np.bool_)) else False


def _joints_array(joints):
    if joints is None or not isinstance(joints, (list, tuple, np.ndarray)):
        return None
    array = np.asarray(joints, dtype=float)
    if array.shape != (21, 3) or not np.isfinite(array).all():
        return None
    return array


def _hand_record(num_frames, field):
    return {
        field: [None] * num_frames,
        "confidence": [None] * num_frames,
        "valid": [False] * num_frames,
    }


def events_from_labels(hand, object_ids, states, timestamps, valid):
    """由逐帧接触物体和抓取状态写出事件。无效帧不闭合、也不新开事件。"""
    events = []
    prev_id = None
    prev_state = None
    seen = False
    for index, (object_id, state, stamp, flag) in enumerate(zip(object_ids, states, timestamps, valid)):
        if not flag:
            continue
        if seen and prev_id != object_id:
            if prev_id is not None:
                events.append(_event("contact_end", hand, prev_id, stamp, index))
            if object_id is not None:
                events.append(_event("contact_start", hand, object_id, stamp, index))
        elif not seen and object_id is not None:
            events.append(_event("contact_start", hand, object_id, stamp, index))
        if state == "grasp" and prev_state != "grasp":
            events.append(_event("grasp", hand, object_id, stamp, index))
        if state == "release" and prev_state != "release":
            events.append(_event("release", hand, object_id if object_id is not None else prev_id, stamp, index))
        prev_id = object_id
        prev_state = state
        seen = True
    return events


def _event(kind, hand, object_id, stamp, frame_index):
    return {
        "type": kind,
        "hand": hand,
        "object_id": object_id,
        "timestamp": float(stamp),
        "frame_index": int(frame_index),
    }


def _surface_query(surface, pose7, query, camera_pose=None, depth_m=None):
    """query 是世界系点。返回每个点到表面的距离；这个表面这一帧不能用时返回 None。"""
    if surface is None:
        surface = {}
    if surface.get("radius_m") is not None and pose7 is not None:
        center = _transform_point(pose7, surface.get("center_local") or [0, 0, 0])
        return min_surface_distance(query, center=center, radius_m=surface["radius_m"]), center
    vertices = surface.get("vertices")
    if vertices is not None and pose7 is not None:
        world = _transform_points(pose7, vertices)
        center = world.mean(axis=0)
        return min_surface_distance(query, vertices=world, faces=surface.get("faces")), center
    if depth_m is not None and camera_pose is not None:
        depths = _camera_depth(query, camera_pose)
        return np.abs(depths - float(depth_m)), None
    return None, None


def _camera_depth(points, camera_pose):
    pose = np.asarray(camera_pose, dtype=float)
    local = (np.asarray(points, dtype=float) - pose[:3, 3]) @ pose[:3, :3]
    return local[:, 2]


def _object_observations(objects, surfaces, frame_index, query, camera_pose=None):
    found = []
    for obj in objects:
        valid = obj.get("valid") or []
        if frame_index >= len(valid) or not _as_bool(valid[frame_index]):
            continue
        pose = obj["pose"][frame_index]
        surface = (surfaces or {}).get(obj["id"]) or {}
        depth_series = surface.get("depth_m")
        depth = None
        if isinstance(depth_series, (list, tuple)) and frame_index < len(depth_series):
            depth = depth_series[frame_index]
        distances, center = _surface_query(surface, pose, query, camera_pose=camera_pose, depth_m=depth)
        if distances is None:
            continue
        found.append({
            "id": obj["id"],
            "distance": float(np.min(distances)),
            "distances": distances,
            "center": center,
        })
    return found


def _nearest_object(objects, surfaces, frame_index, query, camera_pose=None):
    best = None
    for item in _object_observations(objects, surfaces, frame_index, query, camera_pose=camera_pose):
        if best is None or item["distance"] < best["distance"]:
            best = item
    return best


def _temporal_settings(options, enabled):
    """返回 (进入距离, 离开距离, 最短停留秒)。关掉滤波时两档都是 contact_m，停留为 0。"""
    contact_m = float(options["contact_m"])
    if not enabled:
        return contact_m, contact_m, 0.0
    if options.get("contact_on_m") is not None:
        on_m = float(options["contact_on_m"])
    else:
        on_m = contact_m
    if options.get("contact_off_m") is not None:
        off_m = float(options["contact_off_m"])
    else:
        off_m = on_m + float(options["contact_off_margin_m"])
    if off_m < on_m:
        off_m = on_m
    dwell_s = max(0.0, float(options["dwell_s"]))
    return on_m, off_m, dwell_s


def _latch_contact(observations, latched_id, on_m, off_m):
    """滞回：已经贴上的物体要远于 off 才松开；没贴上则要近于 on 才算贴上。"""
    by_id = {item["id"]: item for item in observations}
    if latched_id is not None and latched_id in by_id:
        item = by_id[latched_id]
        n_tips = int(np.sum(np.asarray(item["distances"]) <= off_m))
        if item["distance"] <= off_m and n_tips >= 1:
            return latched_id, item, n_tips
    best = None
    best_n = 0
    for item in observations:
        n_tips = int(np.sum(np.asarray(item["distances"]) <= on_m))
        if n_tips >= 1 and item["distance"] <= on_m and (best is None or item["distance"] < best["distance"]):
            best = item
            best_n = n_tips
    if best is None:
        return None, None, 0
    return best["id"], best, best_n


class _Hold(object):
    """候选值要连续出现 dwell_s 秒才替换当前值。比较的是时间戳，不是帧数。"""

    def __init__(self, dwell_s, initial):
        self.dwell_s = float(dwell_s)
        self.initial = initial
        self.value = initial
        self._pending = None
        self._since = None

    def reset(self):
        self.value = self.initial
        self._pending = None
        self._since = None

    def push(self, value, timestamp):
        if value == self.value:
            self._pending = None
            self._since = None
            return self.value
        stamp = float(timestamp)
        if self._pending != value or self._since is None:
            self._pending = value
            self._since = stamp
        if stamp - float(self._since) >= self.dwell_s - 1e-9:
            self.value = value
            self._pending = None
            self._since = None
        return self.value


def _filtered_labels(observations, nearest, latched_id, id_hold, grasp_hold, prev_state,
                     aperture, speed, hooked, options, on_m, off_m, stamp):
    """滞回得到原始接触/抓取，再用停留时间决定真正写出去的标签。"""
    hook_id = None
    hook_state = None
    if hooked is not None:
        if hooked.get("object_id") is not None:
            hook_id = hooked["object_id"]
        if hooked.get("state") in GRASP_STATES:
            hook_state = hooked["state"]
    if hook_id is not None:
        raw_id = hook_id
        selected = None
        for item in observations:
            if item["id"] == raw_id:
                selected = item
                break
        if selected is None:
            n_tips = 0
        else:
            n_tips = int(np.sum(np.asarray(selected["distances"]) <= off_m))
    else:
        raw_id, selected, n_tips = _latch_contact(observations, latched_id, on_m, off_m)
    latched_id = raw_id
    if hook_state is not None:
        raw_grasp = hook_state == "grasp"
    else:
        raw_grasp = (
            raw_id is not None
            and n_tips >= int(options["min_tips"])
            and aperture is not None
            and aperture <= float(options["aperture_grasp_m"])
            and (speed is None or speed <= float(options["grasp_speed_m_s"]))
        )
    committed_id = id_hold.push(raw_id, stamp)
    committed_grasp = bool(grasp_hold.push(bool(raw_grasp) and raw_id is not None, stamp))
    if committed_id is None:
        committed_grasp = False
        if grasp_hold.value:
            grasp_hold.value = False
            grasp_hold._pending = None
            grasp_hold._since = None
    if committed_id is not None and selected is not None and selected["id"] == committed_id:
        distance = selected["distance"]
    elif nearest is not None:
        distance = nearest["distance"]
    else:
        distance = None
    touching = committed_id is not None
    approaching = (
        not touching
        and distance is not None
        and distance <= float(options["approach_m"])
        and speed is not None
        and speed <= -float(options["approach_speed_m_s"])
    )
    if committed_grasp:
        state = "grasp"
    elif prev_state == "grasp":
        state = "release"
    elif approaching or touching:
        state = "pre_grasp"
    else:
        state = "open"
    return committed_id, state, distance, touching, latched_id


def _tips_of(points):
    points = np.asarray(points, dtype=float)
    if points.shape[0] == 21:
        return points[list(FINGERTIPS)]
    return None


def derive_gt_interaction(objects, hands, timestamps, surfaces, contact_m=GT_CONTACT_M, min_tips=GT_MIN_TIPS):
    """由表面距离写接触和抓取真值。``hands[side]`` 每帧是 None，或带 points / tips 的字典。"""
    num_frames = len(timestamps)
    contact = {"source": "hot3d_mesh", "left": _hand_record(num_frames, "object_id"), "right": _hand_record(num_frames, "object_id")}
    grasp = {"source": "hot3d_mesh", "left": _hand_record(num_frames, "state"), "right": _hand_record(num_frames, "state")}
    contact_flags = {side: [False] * num_frames for side in ("left", "right")}
    grasp_flags = {side: [False] * num_frames for side in ("left", "right")}
    chosen = {side: [None] * num_frames for side in ("left", "right")}
    for side in ("left", "right"):
        series = hands.get(side) or [None] * num_frames
        for index in range(num_frames):
            sample = series[index] if index < len(series) else None
            if sample is None:
                continue
            if isinstance(sample, dict):
                points = sample.get("points")
                tips = sample.get("tips")
            else:
                points = sample
                tips = _tips_of(sample)
            if points is None:
                continue
            points = np.asarray(points, dtype=float).reshape(-1, 3)
            nearest = _nearest_object(objects, surfaces, index, points)
            if nearest is None:
                continue
            tip_distance = None
            if tips is not None:
                tip_nearest = _nearest_object(objects, surfaces, index, np.asarray(tips, dtype=float))
                if tip_nearest is not None and tip_nearest["id"] == nearest["id"]:
                    tip_distance = tip_nearest["distances"]
            touching = nearest["distance"] <= float(contact_m)
            tip_hits = 0
            if tip_distance is not None:
                tip_hits = int(np.sum(tip_distance <= float(contact_m)))
            contact_flags[side][index] = bool(touching)
            grasp_flags[side][index] = bool(touching and tip_hits >= int(min_tips))
            chosen[side][index] = nearest["id"] if touching else None
            contact[side]["valid"][index] = True
            contact[side]["confidence"][index] = 1.0 if touching else 0.0
            grasp[side]["valid"][index] = True
            grasp[side]["confidence"][index] = 1.0
    for side in ("left", "right"):
        states = _gt_states(contact_flags[side], grasp_flags[side])
        for index, state in enumerate(states):
            if not contact[side]["valid"][index]:
                continue
            grasp[side]["state"][index] = state
            contact[side]["object_id"][index] = chosen[side][index]
    events = []
    for side in ("left", "right"):
        events.extend(events_from_labels(
            side,
            contact[side]["object_id"],
            grasp[side]["state"],
            timestamps,
            contact[side]["valid"],
        ))
    return {"contact": contact, "grasp": grasp, "events": events}


def _gt_states(contact_flags, grasp_flags):
    """离线真值可以看下一帧：抓住的前一帧记成预备，抓住的后一帧记成放开。"""
    states = []
    for index, grasped in enumerate(grasp_flags):
        if grasped:
            states.append("grasp")
        elif index > 0 and grasp_flags[index - 1]:
            states.append("release")
        elif contact_flags[index] or (index + 1 < len(grasp_flags) and grasp_flags[index + 1]):
            states.append("pre_grasp")
        else:
            states.append("open")
    return states


def _separation_speed(prev, center, centroid, timestamp):
    if prev is None or center is None or prev["center"] is None or prev["centroid"] is None:
        return None
    dt = float(timestamp) - float(prev["timestamp"])
    if dt <= 1e-8:
        return None
    vel = (centroid - prev["centroid"]) / dt - (center - prev["center"]) / dt
    direction = centroid - center
    norm = float(np.linalg.norm(direction))
    if norm < 1e-8:
        return 0.0
    return float(np.dot(vel, direction / norm))


def estimate_interaction(
    hands,
    objects,
    timestamps,
    surfaces,
    camera_poses=None,
    pose_hook=None,
    contact_hook=None,
    params=None,
    temporal_filter=True,
):
    """用指尖距离、张合和相对速度估计接触和抓取。钩子给了结果就用钩子的。

    ``temporal_filter`` 默认开：接触距离分进入/离开两档，并且新状态要连续保持
    ``dwell_s`` 秒才写出去。关掉则只用 ``contact_m``，当帧就切换。
    """
    options = dict(HEURISTIC_DEFAULTS)
    if params:
        options.update(params)
    on_m, off_m, dwell_s = _temporal_settings(options, temporal_filter)
    if pose_hook is not None:
        hooked = pose_hook(hands, timestamps)
        if hooked is not None:
            objects = hooked
    num_frames = len(timestamps)
    contact_source = "contacthands" if contact_hook is not None else "heuristic"
    contact = {"source": contact_source, "left": _hand_record(num_frames, "object_id"), "right": _hand_record(num_frames, "object_id")}
    grasp = {"source": "heuristic", "left": _hand_record(num_frames, "state"), "right": _hand_record(num_frames, "state")}
    for side in ("left", "right"):
        joints_series = (hands.get(side) or {}).get("joints") or [None] * num_frames
        prev = None
        prev_state = None
        latched_id = None
        id_hold = _Hold(dwell_s, None)
        grasp_hold = _Hold(dwell_s, False)
        for index, stamp in enumerate(timestamps):
            joints = _joints_array(joints_series[index] if index < len(joints_series) else None)
            camera = None
            if camera_poses is not None and index < len(camera_poses):
                camera = camera_poses[index]
            hooked = None
            if contact_hook is not None and joints is not None:
                hooked = contact_hook(side, index, joints, objects)
            if joints is None and hooked is None:
                prev_state = None
                prev = None
                latched_id = None
                id_hold.reset()
                grasp_hold.reset()
                continue
            tips = None if joints is None else joints[list(FINGERTIPS)]
            observations = []
            if tips is not None:
                observations = _object_observations(objects, surfaces, index, tips, camera_pose=camera)
            nearest = None
            for item in observations:
                if nearest is None or item["distance"] < nearest["distance"]:
                    nearest = item
            aperture = None
            if joints is not None:
                aperture = float(np.linalg.norm(joints[4] - joints[8]))
            center = None if nearest is None else nearest["center"]
            centroid = None if tips is None else tips.mean(axis=0)
            speed = _separation_speed(prev, center, centroid, stamp) if centroid is not None else None
            if temporal_filter:
                object_id, state, distance, touching, latched_id = _filtered_labels(
                    observations, nearest, latched_id, id_hold, grasp_hold, prev_state,
                    aperture, speed, hooked, options, on_m, off_m, stamp,
                )
            else:
                n_tips = 0
                distance = None
                object_id = None
                if nearest is not None:
                    distance = nearest["distance"]
                    n_tips = int(np.sum(nearest["distances"] <= options["contact_m"]))
                    if distance <= options["contact_m"] and n_tips >= 1:
                        object_id = nearest["id"]
                if hooked is not None and hooked.get("object_id") is not None:
                    object_id = hooked["object_id"]
                touching = object_id is not None
                stable = (
                    touching
                    and n_tips >= int(options["min_tips"])
                    and aperture is not None
                    and aperture <= options["aperture_grasp_m"]
                    and (speed is None or speed <= options["grasp_speed_m_s"])
                )
                approaching = (
                    not touching
                    and distance is not None
                    and distance <= options["approach_m"]
                    and speed is not None
                    and speed <= -options["approach_speed_m_s"]
                )
                if hooked is not None and hooked.get("state") in GRASP_STATES:
                    state = hooked["state"]
                elif stable:
                    state = "grasp"
                elif prev_state == "grasp":
                    state = "release"
                elif approaching or touching:
                    state = "pre_grasp"
                else:
                    state = "open"
            contact[side]["valid"][index] = True
            contact[side]["object_id"][index] = object_id
            if hooked is not None and hooked.get("confidence") is not None:
                contact[side]["confidence"][index] = float(hooked["confidence"])
            elif distance is None:
                contact[side]["confidence"][index] = 0.0
            else:
                scale = off_m if touching else options["approach_m"]
                contact[side]["confidence"][index] = float(np.clip(1.0 - distance / scale, 0.0, 1.0))
            grasp[side]["valid"][index] = True
            grasp[side]["state"][index] = state
            grasp[side]["confidence"][index] = contact[side]["confidence"][index]
            prev_state = state
            if centroid is not None and center is not None:
                prev = {"centroid": centroid, "center": center, "timestamp": stamp}
            else:
                prev = None
    events = []
    for side in ("left", "right"):
        events.extend(events_from_labels(
            side,
            contact[side]["object_id"],
            grasp[side]["state"],
            timestamps,
            contact[side]["valid"],
        ))
    return {"objects": objects, "contact": contact, "grasp": grasp, "events": events}


def _positive_contact(channel, side):
    ids = channel[side]["object_id"]
    valid = channel[side]["valid"]
    return [bool(flag) and value is not None for value, flag in zip(ids, valid)], [bool(flag) for flag in valid]


def _positive_grasp(channel, side):
    states = channel[side]["state"]
    valid = channel[side]["valid"]
    return [bool(flag) and value == "grasp" for value, flag in zip(states, valid)], [bool(flag) for flag in valid]


def _prf(gt_pos, pred_pos, gt_valid, pred_valid):
    tp = fp = fn = 0
    for g, p, gv, pv in zip(gt_pos, pred_pos, gt_valid, pred_valid):
        if not (gv and pv):
            continue
        if g and p:
            tp += 1
        elif p and not g:
            fp += 1
        elif g and not p:
            fn += 1
    precision = None if tp + fp == 0 else tp / float(tp + fp)
    recall = None if tp + fn == 0 else tp / float(tp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall}


def _pool(left, right):
    return {
        "tp": left["tp"] + right["tp"],
        "fp": left["fp"] + right["fp"],
        "fn": left["fn"] + right["fn"],
    }


def _with_rates(counts):
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    counts["precision"] = None if tp + fp == 0 else tp / float(tp + fp)
    counts["recall"] = None if tp + fn == 0 else tp / float(tp + fn)
    return counts


def evaluate_interaction(gt, pred, timestamps):
    """接触和抓取的精确率 / 召回率，以及同类事件的时间差。无效帧不进分母。"""
    del timestamps
    contact = {}
    grasp = {}
    for side in ("left", "right"):
        gt_c, gt_cv = _positive_contact(gt["contact"], side)
        pred_c, pred_cv = _positive_contact(pred["contact"], side)
        contact[side] = _prf(gt_c, pred_c, gt_cv, pred_cv)
        gt_g, gt_gv = _positive_grasp(gt["grasp"], side)
        pred_g, pred_gv = _positive_grasp(pred["grasp"], side)
        grasp[side] = _prf(gt_g, pred_g, gt_gv, pred_gv)
    contact["both"] = _with_rates(_pool(contact["left"], contact["right"]))
    grasp["both"] = _with_rates(_pool(grasp["left"], grasp["right"]))
    return {
        "contact": contact,
        "grasp": grasp,
        "events": _event_timing(gt.get("events") or [], pred.get("events") or []),
    }


def _event_timing(gt_events, pred_events):
    used = set()
    errors = []
    unmatched_gt = 0
    for event in gt_events:
        best = None
        best_dt = None
        for index, other in enumerate(pred_events):
            if index in used or other.get("type") != event.get("type") or other.get("hand") != event.get("hand"):
                continue
            dt = abs(float(other["timestamp"]) - float(event["timestamp"]))
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best = index
        if best is None:
            unmatched_gt += 1
        else:
            used.add(best)
            errors.append(float(best_dt))
    median = None if not errors else float(np.median(np.asarray(errors, dtype=float)))
    return {
        "matched": len(errors),
        "unmatched_gt": unmatched_gt,
        "unmatched_pred": len(pred_events) - len(used),
        "timing_error_s": errors,
        "timing_error_median_s": median,
    }


def synthetic_disagreement():
    """3 帧小球。皮肤比关节近 6 mm，所以真值比启发式早一帧碰到。

    这是合成夹具，不是 HOT3D 录像。返回 (真值, 启发式, 时间戳)。
    """
    timestamps = [0.0, 0.1, 0.2]
    radius = 0.03
    objects = [{
        "id": "ball",
        "category": "toy",
        "source": "synthetic",
        "pose": [[0, 0, 0, 0, 0, 0, 1]] * 3,
        "confidence": [1.0, 1.0, 1.0],
        "valid": [True, True, True],
    }]
    surfaces = {"ball": {"radius_m": radius, "center_local": [0.0, 0.0, 0.0]}}

    def blank():
        joints = np.full((21, 3), 0.5, dtype=float)
        joints[:, 0] = 0.0
        joints[:, 1] = 0.0
        return joints

    far = blank()
    for index in FINGERTIPS:
        far[index] = [0.20, 0.0, 0.0]
    mid = blank()
    for index in FINGERTIPS:
        mid[index] = [0.20, 0.0, 0.0]
    mid[8] = [0.0405, 0.0, 0.0]
    close = blank()
    close[4] = [0.032, -0.02, 0.0]
    close[8] = [0.032, 0.02, 0.0]
    for index in (12, 16, 20):
        close[index] = [0.032, 0.0, 0.0]
    joints = [far, mid, close]

    def skin(frame):
        points = frame.copy()
        tips = []
        for index in FINGERTIPS:
            point = points[index].copy()
            norm = float(np.linalg.norm(point))
            if norm > 1e-8:
                point = point - 0.006 * point / norm
            points[index] = point
            tips.append(point)
        return {"points": points, "tips": np.stack(tips)}

    hands_est = {
        "left": {"joints": [None, None, None], "confidence": [None, None, None]},
        "right": {"joints": [frame.tolist() for frame in joints], "confidence": [1.0, 1.0, 1.0]},
    }
    hands_gt = {"left": [None, None, None], "right": [skin(frame) for frame in joints]}
    gt = derive_gt_interaction(objects, hands_gt, timestamps, surfaces)
    # 3 帧、间隔 0.1 秒，短于默认停留。文档里的合成数是没开时间滤波的。
    pred = estimate_interaction(hands_est, objects, timestamps, surfaces, temporal_filter=False)
    return gt, pred, timestamps


def annotate_episode(episode, surfaces, **kwargs):
    """把估计结果写回 episode。物体列表以钩子或原来的为准。"""
    result = estimate_interaction(
        episode["hands"],
        episode.get("objects") or [],
        episode["timestamps"],
        surfaces,
        camera_poses=episode.get("camera_poses"),
        **kwargs,
    )
    updated = dict(episode)
    updated["objects"] = result["objects"]
    updated["contact"] = result["contact"]
    updated["grasp"] = result["grasp"]
    updated["events"] = result["events"]
    return updated

