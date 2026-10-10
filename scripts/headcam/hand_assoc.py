# -*- coding: utf-8 -*-
"""双目手部关联：时序跟踪 + 翻转 TTA 投票 + 双目几何，把每帧的 WiLoR 多候选选成“一只左手 + 一只右手”。

输入是 ``headcam.wilor_candidates`` 的逐帧候选（每个手框带原图 / 翻转图的左右投票，以及按左手、右手
两种假设各跑一次 WiLoR 的结果）。输出与 ``stereo_pipeline.run_backend`` 的 ``views`` 同格式，
后面的三角化、一致性检查、速度门限、RTS 平滑、QC 一个字都不用改。

每帧枚举假设：每只手 = (左目候选, 右目候选) 的立体对 / 只有一目 / 没有；同一目的同一个框不能同时当左手和右手
（去掉重复标注）。代价：

* 立体对：中位重投影误差（对极约束的代理，左右框认错就对不上）、WiLoR 单目 21 点刚体对齐到三角化点的残差
  （拿右手模型套左手，形状对不上）、左右投票的惩罚；
* 一目 / 没有：固定代价；
* 帧间：同一只手世界系手腕位移（Viterbi，整段离线求最优路径）。

``cross_view_requests`` 列出要在另一目重新裁剪的位置：某只手只在一目找到、或者立体对重投影太大时，
把这一目的 WiLoR 公制 21 点（有前后帧立体结果时用其插值）投到另一目，裁框重跑 WiLoR。
阈值和权重见 ``AssocParams``；在 HOT3D 上只用一段调、其余段测（docs/hot3d_stereo/hand_assoc.md）。
"""
import numpy as np

from headcam.hand_pose import JOINTS, project_pinhole, transform_points, triangulate_pair

SIDES = ("left", "right")


class AssocParams(object):
    def __init__(self, sigma_reproj_px=4.0, max_reproj_px=25.0, w_fit=0.3, min_det_score=0.3, sigma_fit_m=0.01, w_hand=1.0,
                 cost_one_view=4.0, cost_none=5.0, w_motion=1.0, sigma_motion_m=0.03, motion_cap=6.0, motion_gap=1.0,
                 w_mono=0.0, sigma_mono_m=0.05, dup_px=40.0,
                 use_flip_votes=True, both_hypotheses=True, temporal=True, top_k=6,
                 depth_range_m=(0.10, 1.20), recrop_reproj_px=8.0, recrop_pad=0.25, recrop_window=5):
        self.__dict__.update({k: v for k, v in locals().items() if k != "self"})

    def to_dict(self):
        return dict(self.__dict__)


def _hand_prob(cand, side, use_flip):
    votes = [cand.get("orig")]
    if use_flip:
        votes.append(cand.get("flip"))
    votes = [v for v in votes if v is not None]
    if not votes:
        return 0.5
    p_right = float(np.mean([v[0] for v in votes]))
    return p_right if side == "right" else 1.0 - p_right


def _score(cand):
    vals = [v[1] for v in (cand.get("orig"), cand.get("flip")) if v is not None]
    return float(max(vals)) if vals else 0.3


def _yolo_side(cand):
    v = cand.get("orig") or cand.get("flip")
    if v is None:
        return None
    return "right" if v[0] >= 0.5 else "left"


def _fit_residual(mono, tri, err):
    from headcam.stereo_pipeline import rigid_fit_hand
    w = np.exp(-0.5 * (np.nan_to_num(err, nan=99.0) / 5.0) ** 2)
    _, res = rigid_fit_hand(mono, tri, w)
    return res


def _usable(cands, params):
    """候选过滤：不用翻转 TTA 时只留原图检测、只用 YOLO 标签的那个假设。"""
    out = []
    for i, c in enumerate(cands):
        if not params.use_flip_votes and c.get("orig") is None and not c.get("recrop"):
            continue
        sides = SIDES if (params.both_hypotheses or c.get("recrop")) else (_yolo_side(c),)
        for s in sides:
            if s is not None and c.get(s) is not None:
                out.append((i, s))
    return out


def frame_hypotheses(cl, cr, calib, params):
    """返回 {side: [(cost, li, ri, info)]}，li/ri 为 None 表示那一目没有。"""
    from headcam.stereo_pipeline import reprojection_errors
    ul, ur = _usable(cl, params), _usable(cr, params)
    hyps = {}
    for s in SIDES:
        L = [i for i, ss in ul if ss == s]
        R = [j for j, ss in ur if ss == s]
        lst = [(params.cost_none, None, None, {})]
        for i in L:
            lst.append((params.cost_one_view + params.w_hand * (1 - _hand_prob(cl[i], s, params.use_flip_votes)),
                        i, None, {"uv": {"left": cl[i][s]["keypoints_2d"][0]}}))
        for j in R:
            lst.append((params.cost_one_view + params.w_hand * (1 - _hand_prob(cr[j], s, params.use_flip_votes)),
                        None, j, {"uv": {"right": cr[j][s]["keypoints_2d"][0]}}))
        stereo = []
        for i in L:
            for j in R:
                uvl = np.asarray(cl[i][s]["keypoints_2d"], float)
                uvr = np.asarray(cr[j][s]["keypoints_2d"], float)
                tri, _ = triangulate_pair(uvl, uvr, calib)
                err = reprojection_errors(tri, uvl, uvr, calib)
                fin = np.isfinite(err)
                if fin.sum() < 6 or not np.isfinite(tri[0]).all():
                    continue
                if not (params.depth_range_m[0] <= tri[0, 2] <= params.depth_range_m[1]):
                    continue
                e = float(np.median(err[fin]))
                if e > params.max_reproj_px:
                    continue
                res = _fit_residual(np.asarray(cl[i][s]["joints_cam"], float), tri, err)
                cost = e / params.sigma_reproj_px
                cost += params.w_fit * (min(res, 0.05) / params.sigma_fit_m if res is not None else 5.0)
                cost += params.w_hand * ((1 - _hand_prob(cl[i], s, params.use_flip_votes)) +
                                         (1 - _hand_prob(cr[j], s, params.use_flip_votes)))
                if params.w_mono > 0:
                    # 单目 WiLoR 公制手腕（左目相机系）和三角化手腕应大致一致；左右框配错时深度差很大
                    mono_w = np.asarray(cl[i][s]["joints_cam"], float)[0]
                    cost += params.w_mono * min(np.linalg.norm(mono_w - tri[0]) / params.sigma_mono_m, 4.0)
                stereo.append((cost, i, j, {"wrist_cam": tri[0], "reproj": e,
                                               "uv": {"left": uvl[0].tolist(), "right": uvr[0].tolist()}}))
        stereo.sort(key=lambda h: h[0])
        hyps[s] = lst + stereo[:params.top_k]
    return hyps


def _duplicate(hl, hr, dup_px):
    """左右手在同一目里手腕挨得太近：同一只手被检成两个框（或两种假设），不能同时当左手和右手。"""
    if not dup_px:
        return False
    ul, ur = hl[3].get("uv", {}), hr[3].get("uv", {})
    for view in SIDES:
        if view in ul and view in ur:
            if np.linalg.norm(np.asarray(ul[view], float) - np.asarray(ur[view], float)) < dup_px:
                return True
    return False


def _joint_states(hyps, dup_px=0.0):
    states = []
    for hl in hyps["left"]:
        for hr in hyps["right"]:
            if hl[1] is not None and hl[1] == hr[1]:
                continue
            if hl[2] is not None and hl[2] == hr[2]:
                continue
            if _duplicate(hl, hr, dup_px):
                continue
            states.append((hl[0] + hr[0], hl, hr))
    return states


def associate(cands, calib, poses, params=None):
    """cands: {"left": [帧候选], "right": [...]}。返回 (views, chosen)。"""
    params = params or AssocParams()
    n = min(len(cands["left"]), len(cands["right"]))
    all_states = []
    for t in range(n):
        st = _joint_states(frame_hypotheses(cands["left"][t], cands["right"][t], calib, params), params.dup_px)
        world = []
        for c, hl, hr in st:
            w = {}
            for s, h in (("left", hl), ("right", hr)):
                if "wrist_cam" in h[3]:
                    w[s] = transform_points(h[3]["wrist_cam"][None], poses[t])[0]
            world.append(w)
        all_states.append((st, world))
    # Viterbi
    if params.temporal:
        prev_cost = np.array([s[0] for s in all_states[0][0]])
        back = []
        for t in range(1, n):
            st, world = all_states[t]
            pst, pworld = all_states[t - 1]
            unary = np.array([s[0] for s in st])
            pair = np.zeros((len(pst), len(st)))
            for a, wa in enumerate(pworld):
                for b, wb in enumerate(world):
                    c = 0.0
                    for s in SIDES:
                        if s in wa and s in wb:
                            c += min(np.linalg.norm(wa[s] - wb[s]) / params.sigma_motion_m, params.motion_cap)
                        else:
                            # 有一帧不是立体对：不知道位移，记一个固定代价，免得“不配对”反而在时序上更便宜
                            c += params.motion_gap
                    pair[a, b] = params.w_motion * c
            tot = prev_cost[:, None] + pair
            back.append(np.argmin(tot, axis=0))
            prev_cost = tot.min(axis=0) + unary
        path = [int(np.argmin(prev_cost))]
        for bp in reversed(back):
            path.append(int(bp[path[-1]]))
        path = path[::-1]
    else:
        path = [int(np.argmin([s[0] for s in all_states[t][0]])) for t in range(n)]
    views = {"left": [], "right": []}
    chosen = []
    for t in range(n):
        _, hl, hr = all_states[t][0][path[t]]
        fr = {"left": {}, "right": {}}
        info = {}
        for s, h in (("left", hl), ("right", hr)):
            info[s] = {"li": h[1], "ri": h[2], "reproj": h[3].get("reproj")}
            for view, idx in (("left", h[1]), ("right", h[2])):
                c = None if idx is None else cands[view][t][idx]
                fr[view][s] = None if c is None else {
                    "keypoints_2d": c[s]["keypoints_2d"], "joints_cam": c[s]["joints_cam"],
                    "confidence": [_score(c)] * JOINTS}
        for view in SIDES:
            views[view].append(fr[view])
        chosen.append(info)
    return views, chosen


def _box_from_uv(uv, w, h, pad):
    ok = np.isfinite(uv).all(1)
    if ok.sum() < 10:
        return None
    x0, y0 = uv[ok].min(0)
    x1, y1 = uv[ok].max(0)
    if not (0 <= uv[0, 0] < w and 0 <= uv[0, 1] < h):
        return None
    pw, ph = (x1 - x0) * pad, (y1 - y0) * pad
    box = [max(0.0, x0 - pw), max(0.0, y0 - ph), min(w - 1.0, x1 + pw), min(h - 1.0, y1 + ph)]
    if box[2] - box[0] < 16 or box[3] - box[1] < 16:
        return None
    return box


def cross_view_requests(cands, chosen, calib, poses, params=None):
    """返回 [(t, 目标目, side, box)]：把有手那一目的 3D（WiLoR 公制 21 点，按前后帧立体手腕平移校正）投到另一目。"""
    params = params or AssocParams()
    R = np.asarray(calib["R"], float)
    T = np.asarray(calib["T"], float).reshape(3)
    w, h = int(calib["image_width"]), int(calib["image_height"])
    n = len(chosen)
    reqs = []
    for t in range(n):
        for s in SIDES:
            c = chosen[t][s]
            li, ri = c["li"], c["ri"]
            todo = []
            if li is not None and ri is None:
                todo.append(("right", "left", li))
            elif ri is not None and li is None:
                todo.append(("left", "right", ri))
            elif li is not None and c["reproj"] is not None and c["reproj"] > params.recrop_reproj_px:
                todo += [("right", "left", li), ("left", "right", ri)]
            for target, src, idx in todo:
                mono = np.asarray(cands[src][t][idx][s]["joints_cam"], float)
                if src == "right":   # 右相机系 → 左相机系
                    mono = (mono - T) @ R
                # 用最近的立体手腕（同一侧、±window 帧）校正单目深度
                best = None
                for dt in range(1, params.recrop_window + 1):
                    for tt in (t - dt, t + dt):
                        if 0 <= tt < n and best is None:
                            cc = chosen[tt][s]
                            if cc["li"] is not None and cc["ri"] is not None and cc.get("wrist_cam") is not None:
                                best = tt
                if best is not None:
                    wc = transform_points(np.asarray(chosen[best][s]["wrist_world"])[None],
                                          np.linalg.inv(np.asarray(poses[t], float)))[0]
                    # 只用距离校正：沿视线把单目手腕缩放到立体手腕的深度
                    if mono[0, 2] > 1e-3:
                        mono = mono * (wc[2] / mono[0, 2])
                if target == "right":
                    uv = project_pinhole(mono, calib["K_right"], R, T)
                else:
                    uv = project_pinhole(mono, calib["K_left"])
                box = _box_from_uv(uv, w, h, params.recrop_pad)
                if box is not None:
                    reqs.append((t, target, s, box))
    return reqs


def annotate_chosen_wrists(chosen, cands, calib, poses):
    """给立体对补上左相机系 / 世界系手腕（供 cross_view_requests 插值用）。原地修改。"""
    for t, info in enumerate(chosen):
        for s in SIDES:
            c = info[s]
            if c["li"] is None or c["ri"] is None:
                continue
            uvl = np.asarray(cands["left"][t][c["li"]][s]["keypoints_2d"], float)
            uvr = np.asarray(cands["right"][t][c["ri"]][s]["keypoints_2d"], float)
            tri, _ = triangulate_pair(uvl[:1], uvr[:1], calib)
            if np.isfinite(tri[0]).all():
                c["wrist_cam"] = tri[0]
                c["wrist_world"] = transform_points(tri[:1], poses[t])[0]
