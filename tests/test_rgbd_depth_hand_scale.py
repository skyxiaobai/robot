# -*- coding: utf-8 -*-
"""RGB-D：LiDAR 手腕深度 + WiLoR 手型（不按错误单目深度缩放）。"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from headcam.rgbd_pipeline import depth_hand, RGBDParams  # noqa: E402
from headcam.stereo_pipeline import MIDDLE_MCP  # noqa: E402


def _fake_hand(z_mono=0.4, palm=0.095, uv0=(960.0, 720.0), span_px=150.0):
    """单目手在 z_mono，手掌 palm 米；2D 手腕在 uv0。"""
    mono = np.zeros((21, 3), dtype=float)
    mono[:, 2] = z_mono
    mono[MIDDLE_MCP] = [palm, 0.0, z_mono]
    mono[0] = [0.0, 0.0, z_mono]
    uv = np.tile([uv0[0], uv0[1]], (21, 1)).astype(float)
    uv[MIDDLE_MCP, 0] = uv0[0] + span_px
    return {
        "keypoints_2d": uv.tolist(),
        "joints_cam": mono.tolist(),
        "confidence": [1.0] * 21,
    }


def test_lidar_wrist_keeps_mano_palm():
    """单目 z=0.4、LiDAR 近层=1.0 时：手腕用 LiDAR，手掌保持 ~9.5 cm。"""
    h, w = 1440, 1920
    dh, dw = 192, 256
    depth = np.full((dh, dw), 1.0, dtype=np.float32)
    conf = np.full((dh, dw), 2, dtype=np.uint8)
    K = np.array([[1455.0, 0.0, 960.0], [0.0, 1455.0, 720.0], [0.0, 0.0, 1.0]])
    hand = _fake_hand(z_mono=0.4, palm=0.095)
    p = RGBDParams()
    out = depth_hand(hand, depth, conf, (w, h), K, p)
    assert out["status"] == "ok", out
    assert abs(out["wrist_depth_m"] - 1.01) < 0.05, out["wrist_depth_m"]  # near + joint_offset
    assert 0.08 < out["palm_m"] < 0.12, out["palm_m"]
    assert out["mono_scale"] is not None and out["mono_scale"] > 2.0


def test_no_hand_is_none():
    h, w = 100, 100
    depth = np.ones((50, 50), np.float32)
    conf = np.full((50, 50), 2, np.uint8)
    K = np.eye(3)
    out = depth_hand(None, depth, conf, (w, h), K, RGBDParams())
    assert out["status"] == "none"
