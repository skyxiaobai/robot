# -*- coding: utf-8 -*-
"""手部关联（headcam/hand_assoc.py）的合成数据测试：不需要 WiLoR / torch。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from headcam import hand_assoc as HA  # noqa: E402
from headcam.hand_pose import project_pinhole  # noqa: E402
from headcam.wilor_candidates import merge_boxes  # noqa: E402

K = np.array([[500.0, 0, 512], [0, 500.0, 640], [0, 0, 1]])
CALIB = {"K_left": K, "K_right": K, "R": np.eye(3), "T": np.array([-0.064, 0, 0]),
         "image_width": 1024, "image_height": 1280, "dist_left": None, "dist_right": None}
POSE = np.eye(4)


def _hand(center):
    rng = np.random.RandomState(0)
    return np.asarray(center) + rng.uniform(-0.04, 0.04, size=(21, 3))


def _cand(joints, view, vote_right, both=True):
    uv = project_pinhole(joints, K) if view == "left" else project_pinhole(joints, K, CALIB["R"], CALIB["T"])
    h = {"keypoints_2d": uv.tolist(), "joints_cam": joints.tolist()}
    c = {"box": [float(uv[:, 0].min()), float(uv[:, 1].min()), float(uv[:, 0].max()), float(uv[:, 1].max())],
         "orig": [float(vote_right), 0.8], "flip": [float(vote_right), 0.8]}
    if both:
        c["left"] = h
        c["right"] = h
    else:
        c["right" if vote_right else "left"] = h
    return c


LEFT = _hand([-0.15, 0.05, 0.40])
RIGHT = _hand([0.15, 0.05, 0.40])


def _frames(n, swap_at=None):
    cands = {"left": [], "right": []}
    for t in range(n):
        for view in ("left", "right"):
            vl, vr = 0, 1
            if swap_at is not None and t == swap_at and view == "right":
                vl, vr = 1, 0          # 右目这一帧 YOLO 把左右标反
            cands[view].append([_cand(LEFT, view, vl), _cand(RIGHT, view, vr)])
    return cands


def test_picks_one_left_one_right_stereo():
    views, chosen = HA.associate(_frames(5), CALIB, [POSE] * 5, HA.AssocParams())
    for t in range(5):
        assert chosen[t]["left"]["li"] is not None and chosen[t]["left"]["ri"] is not None
        assert chosen[t]["left"]["li"] != chosen[t]["right"]["li"]
        uv = np.asarray(views["left"][t]["left"]["keypoints_2d"])
        assert uv[0, 0] < np.asarray(views["left"][t]["right"]["keypoints_2d"])[0, 0]


def test_handedness_swap_in_one_view_is_fixed():
    _, chosen = HA.associate(_frames(7, swap_at=3), CALIB, [POSE] * 7, HA.AssocParams())
    # 立体几何 + 时序把右目第 3 帧标反的框重新配回去：左手仍配左手
    assert chosen[3]["left"]["ri"] == 0 and chosen[3]["right"]["ri"] == 1


def test_duplicate_box_cannot_be_both_hands():
    cands = {"left": [], "right": []}
    for view in ("left", "right"):
        a = _cand(LEFT, view, 0)
        b = _cand(LEFT + 0.002, view, 1)    # 同一只手被检成两个框
        cands[view].append([a, b])
    _, chosen = HA.associate(cands, CALIB, [POSE], HA.AssocParams(dup_px=40.0))
    used = [s for s in ("left", "right") if chosen[0][s]["li"] is not None]
    assert len(used) == 1
    _, chosen = HA.associate(cands, CALIB, [POSE], HA.AssocParams(dup_px=0.0))
    assert all(chosen[0][s]["li"] is not None for s in ("left", "right"))


def test_cross_view_request_projects_into_other_view():
    cands = {"left": [[_cand(LEFT, "left", 0)]], "right": [[]]}
    _, chosen = HA.associate(cands, CALIB, [POSE], HA.AssocParams())
    assert chosen[0]["left"]["li"] == 0 and chosen[0]["left"]["ri"] is None
    reqs = HA.cross_view_requests(cands, chosen, CALIB, [POSE], HA.AssocParams())
    assert len(reqs) == 1 and reqs[0][1] == "right" and reqs[0][2] == "left"
    uv = project_pinhole(LEFT, K, CALIB["R"], CALIB["T"])
    box = reqs[0][3]
    assert box[0] <= uv[:, 0].min() and box[2] >= uv[:, 0].max()


def test_merge_boxes_flip_votes():
    out = merge_boxes([([0, 0, 10, 10], 1.0, 0.6)], [([1, 0, 11, 10], 1.0, 0.5), ([50, 50, 60, 60], 0.0, 0.4)])
    assert len(out) == 2
    assert out[0]["flip"] == [1.0, 0.5] and out[1]["orig"] is None
