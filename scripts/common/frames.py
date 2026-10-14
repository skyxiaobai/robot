# -*- coding: utf-8 -*-
"""所有数据源共用的坐标约定（单位一律米）。

四种叫法，episode 里写在 ``frame_convention``（旧字段 ``coordinate_frame`` 保留原名，见 ALIASES）：

* ``camera``     相机光学系（OpenCV）：x 右、y 下、z 前。WiLoR / LiDAR 反投影的原生系。
* ``camera_yup`` 相机系翻成 y 朝上：x 右、y 上、z 后（= ARKit 相机系）。不保证重力对齐。
* ``world_yup``  重力对齐世界系：y 朝上（反重力），米。Stera（ARKit 世界）走这里。
* ``world``      公制 SLAM 世界系，但「哪根轴朝上」还没核实（iPhone mcap 的 /embosa_tf 就是这样：
                 实测相机「上」方向并不在 y 上）。用 ``align_world_to_yup`` 指定/估计上轴后变成 world_yup。

转换器只用这里的函数，不再各自写 FLIP 矩阵；``sanity_check`` 检查单位和轴向，
发现「手腕离相机 30 m」（多半是 cm 当成 m）或「手在相机背后」（轴反了）就报出来。
"""
import numpy as np

CAMERA, CAMERA_YUP, WORLD_YUP, WORLD = "camera", "camera_yup", "world_yup", "world"
CONVENTIONS = (CAMERA, CAMERA_YUP, WORLD_YUP, WORLD)
CV_TO_YUP = np.diag([1.0, -1.0, -1.0])  # 光学系 ↔ y 朝上相机系（自逆）
ROS_ZUP_TO_YUP = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], float)  # ROS x前 y左 z上 → x右 y上 z后
# 旧 coordinate_frame 名字 → 统一约定
ALIASES = {"camera": CAMERA, "camera_yup": CAMERA_YUP, "arkit_world_yup": WORLD_YUP, "slam_world": WORLD,
           "world_yup": WORLD_YUP}


def cm_to_m(x):
    return np.asarray(x, dtype=np.float64) / 100.0


def cv_to_yup(points):
    """光学系点 (...,3) → y 朝上相机系；再调一次就是反变换。"""
    return np.asarray(points, dtype=np.float64) @ CV_TO_YUP.T


yup_to_cv = cv_to_yup


def rotation_up_to_y(up):
    """返回旋转 R，使 R @ up 指向 +y（Rodrigues）。"""
    u = np.asarray(up, float); u = u / np.linalg.norm(u)
    y = np.array([0.0, 1.0, 0.0])
    v = np.cross(u, y); s = np.linalg.norm(v); c = float(u @ y)
    if s < 1e-8:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)


def world_up_alignment(R_cv2w):
    """世界系 → 重力 y 朝上：用光学系 -y（相机「上」）在世界里的平均方向判断世界哪根轴朝上。

    R_cv2w: (N,3,3) 光学系 → 世界。返回 (A, 标签)，A 左乘世界坐标得到 y 朝上世界。
    """
    up_world = (np.asarray(R_cv2w) @ np.array([0.0, -1.0, 0.0])).mean(0)
    ax = int(np.argmax(np.abs(up_world))); sgn = np.sign(up_world[ax])
    if ax == 1 and sgn > 0:
        return np.eye(3), "y"
    if ax == 2:
        return ROS_ZUP_TO_YUP * np.array([1, sgn, 1])[:, None], "z"
    u = np.zeros(3); u[ax] = sgn
    R = rotation_up_to_y(u)
    return R, ("y-" if ax == 1 else "axis%d" % ax)


def _rot_to_quat(R):
    """3x3 → xyzw。"""
    R = np.asarray(R, float); t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return np.array([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s])
    i = int(np.argmax(np.diag(R))); j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = np.zeros(4); q[i] = 0.25 * s; q[j] = (R[j, i] + R[i, j]) / s; q[k] = (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return q


def _quat_mul(a, b):
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def align_world_to_yup(episode, A, label=""):
    """把 world 系 episode 左乘旋转 A 变成 world_yup（手腕、关节、相机位姿一起转）。"""
    A = np.asarray(A, float)
    for side in ("left", "right"):
        h = episode["hands"][side]
        for i, p in enumerate(h["wrist_pose"]):
            if p and all(v is not None for v in p[:3]):
                q = list(p[3:7])
                if len(q) == 4 and all(v is not None for v in q):
                    q = _quat_mul(_rot_to_quat(A), np.asarray(q, float)).tolist()
                h["wrist_pose"][i] = list((A @ np.asarray(p[:3], float)).tolist()) + list(q)
        for i, jj in enumerate(h.get("joints") or []):
            arr = np.asarray([[np.nan if v is None else v for v in j] for j in jj], float)
            if np.isfinite(arr).all():
                h["joints"][i] = (arr @ A.T).tolist()
    if episode.get("camera_poses"):
        P = np.asarray(episode["camera_poses"], float)
        P[:, :3, :3] = np.einsum("ij,njk->nik", A, P[:, :3, :3]); P[:, :3, 3] = P[:, :3, 3] @ A.T
        episode["camera_poses"] = P.tolist()
    tag(episode, WORLD_YUP, gravity_aligned=True, source_frame="world(up=%s)" % label)
    return episode


def tag(episode, convention, gravity_aligned=None, source_frame=None, units="m"):
    """在 episode 上写统一坐标约定。"""
    if convention not in CONVENTIONS:
        raise ValueError("未知坐标约定 %s（可选 %s）" % (convention, CONVENTIONS))
    episode["frame_convention"] = {
        "name": convention, "units": units,
        "gravity_aligned": bool(convention == WORLD_YUP if gravity_aligned is None else gravity_aligned),
        "source_frame": source_frame or episode.get("coordinate_frame")}
    return episode


def convention_of(episode):
    fc = episode.get("frame_convention")
    if isinstance(fc, dict) and fc.get("name") in CONVENTIONS:
        return fc["name"]
    return ALIASES.get(str(episode.get("coordinate_frame")), str(episode.get("coordinate_frame")))


def _wrists(episode, side):
    rows = episode["hands"][side]["wrist_pose"]
    out = np.full((len(rows), 3), np.nan)
    for i, p in enumerate(rows):
        if p and len(p) >= 3 and all(v is not None for v in p[:3]):
            out[i] = p[:3]
    return out


def _cam_poses(episode, n):
    cp = episode.get("camera_poses")
    if not cp or len(cp) != n:
        return np.tile(np.eye(4), (n, 1, 1))
    return np.asarray(cp, float)


def wrist_positions(episode, variant=None):
    """(n,2,3) 左右手腕，换到 ``variant``（camera / camera_yup / world_yup；None = 原样）。

    世界系 → 相机系用 camera_poses（相机→世界）；相机系没有重力，要 world_yup 时退回 camera_yup。
    """
    conv = convention_of(episode)
    W = np.stack([_wrists(episode, s) for s in ("left", "right")], 1)
    if variant is None or variant == conv:
        return W
    n = len(W); P = _cam_poses(episode, n)
    if conv in (WORLD_YUP, WORLD):
        if variant == WORLD_YUP and conv == WORLD:
            raise ValueError("world 的上轴未核实；先 align_world_to_yup")
        rel = np.einsum("nji,nkj->nki", P[:, :3, :3], W - P[:, None, :3, 3])  # 世界 → 相机（与 pose 的相机系同向）
        cam_is_cv = episode.get("frame_convention", {}).get("camera_axes", "cv") == "cv"
        cv = rel if cam_is_cv else cv_to_yup(rel)
        return cv if variant == CAMERA else cv_to_yup(cv)
    if conv == CAMERA:
        return cv_to_yup(W) if variant in (CAMERA_YUP, WORLD_YUP) else W
    if conv == CAMERA_YUP:
        return cv_to_yup(W) if variant == CAMERA else W
    raise ValueError("无法转换 %s → %s" % (conv, variant))


def sanity_check(episode, min_dist_m=0.08, max_dist_m=1.5):
    """检查单位和轴向。返回 {"ok", "flags", "median_dist_m", "front_frac", "below_frac"}。

    * 单位：手腕到相机的中位距离应在 8 cm–1.5 m；>3 m 多半是 cm/mm 当成了米，<3 cm 多半是反过来。
    * 轴向：光学系里手应在相机前方（z>0）、偏下（y>0）。前方比例 <50% 说明 z 轴反了。
    * 重力系：标为 world_yup 时，相机「上」方向（光学 -y）在世界里的 y 分量应为正。
    """
    flags = []
    try:
        cam = wrist_positions(episode, CAMERA).reshape(-1, 3)
    except ValueError as e:
        return {"ok": False, "flags": [str(e)], "median_dist_m": None, "front_frac": None, "below_frac": None}
    cam = cam[np.isfinite(cam).all(1)]
    if len(cam) == 0:
        return {"ok": True, "flags": ["no_valid_wrist"], "median_dist_m": None, "front_frac": None, "below_frac": None}
    dist = float(np.median(np.linalg.norm(cam, axis=1)))
    front = float((cam[:, 2] > 0).mean()); below = float((cam[:, 1] > 0).mean())
    if dist > 3.0:
        flags.append("units: 手腕离相机中位 %.1f m，像是 cm/mm 没换成米" % dist)
    elif dist > max_dist_m:
        flags.append("units?: 手腕离相机中位 %.2f m，偏远" % dist)
    if dist < 0.03:
        flags.append("units: 手腕离相机中位 %.3f m，像是把米又除了 100" % dist)
    elif dist < min_dist_m:
        flags.append("units?: 手腕离相机中位 %.3f m，偏近" % dist)
    if front < 0.5:
        flags.append("axes: 只有 %.0f%% 的手腕在相机前方，z 轴可能反了" % (100 * front))
    if below < 0.2:
        flags.append("axes?: 只有 %.0f%% 的手腕在画面下半，y 轴可能反了" % (100 * below))
    if convention_of(episode) in (WORLD_YUP, WORLD):
        P = _cam_poses(episode, len(episode["hands"]["left"]["wrist_pose"]))
        if not np.allclose(P[:, :3, :3], np.eye(3)):
            up = float(np.median((P[:, :3, :3] @ np.array([0.0, -1.0, 0.0]))[:, 1]))
            right = float(np.median(np.abs((P[:, :3, :3] @ np.array([1.0, 0.0, 0.0]))[:, 1])))
            # 低头看桌面时「上」的 y 分量可以很小，但不会是负的；相机左右轴应大致水平（侧倾不大）
            if up < -0.1 or right > 0.7:
                flags.append(("axes" if convention_of(episode) == WORLD_YUP else "axes?") + ": 标为重力 y 朝上，但相机「上」y 分量 %.2f、「右」|y| %.2f，上轴多半不是 y" % (up, right))
    hard = [f for f in flags if not f.split(":")[0].endswith("?")]
    return {"ok": not hard, "flags": flags, "median_dist_m": round(dist, 3), "front_frac": round(front, 3),
            "below_frac": round(below, 3)}
