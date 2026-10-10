# -*- coding: utf-8 -*-
"""iPhone Record3D 适配 + 激光雷达深度手部求解（合成数据，不需要 WiLoR / MANO）。"""
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

liblzfse = pytest.importorskip("liblzfse")
cv2 = pytest.importorskip("cv2")

from headcam.rgbd_pipeline import RGBDParams, depth_hand, make_session_builder, pose_jumps  # noqa: E402
from iphone import record3d_adapter as RA  # noqa: E402

W, H, DW, DH = 192 * 2, 256 * 2, 192, 256
K = np.array([[300.0, 0, W / 2.0], [0, 300.0, H / 2.0], [0, 0, 1]])


def hand_shape():
    """21 点的简易手（米，手腕在原点，手指沿 -y）。"""
    pts = [[0, 0, 0]]
    for f, x in enumerate([-0.035, -0.015, 0.0, 0.015, 0.03]):
        for k in range(1, 5):
            pts.append([x * (1 + 0.1 * k), -0.03 - 0.022 * k, 0.004 * k])
    return np.asarray(pts, dtype=float)


def project(p):
    return np.stack([K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2], K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]], 1)


def render_depth(joints, background=1.2, offset=0.01):
    depth = np.full((DH, DW), background, np.float32)
    uv = project(joints) * [DW / float(W), DH / float(H)]
    hull = cv2.convexHull(np.round(uv).astype(np.int32))
    mask = np.zeros((DH, DW), np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8)).astype(bool)
    depth[mask] = float(np.median(joints[:, 2])) - offset
    for j in range(len(joints)):
        u, v = np.round(uv[j]).astype(int)
        depth[max(0, v - 1):v + 2, max(0, u - 1):u + 2] = joints[j, 2] - offset
    return depth


def gt_hand(t):
    return hand_shape() + np.array([0.02 * np.sin(t), 0.05, 0.45 + 0.02 * t])


def wilor_like(gt):
    """WiLoR 单目：2D 对，三维深度错（尺度/平移偏离），模拟单目误差。"""
    mono = (gt - gt[0]) * 1.1 + gt[0] * np.array([1, 1, 1.4])
    return {"keypoints_2d": project(gt).tolist(), "joints_cam": mono.tolist(), "confidence": [0.9] * 21}


def test_depth_hand_recovers_wrist_despite_wrong_mono_depth():
    gt = gt_hand(0.0)
    depth = render_depth(gt)
    conf = np.full(depth.shape, 2, np.uint8)
    res = depth_hand(wilor_like(gt), depth, conf, (W, H), K, RGBDParams())
    assert res["status"] == "ok", res["status"]
    assert np.linalg.norm(res["joints_cam"][0] - gt[0]) < 0.01
    mono_err = np.linalg.norm(np.asarray(wilor_like(gt)["joints_cam"])[0] - gt[0])
    assert mono_err > 0.1


def test_depth_hand_low_confidence_and_occluder():
    gt = gt_hand(0.0)
    depth = render_depth(gt)
    res = depth_hand(wilor_like(gt), depth, np.zeros(depth.shape, np.uint8), (W, H), K, RGBDParams())
    assert res["status"] == "lowconf"
    # 背景物体边缘：手区域一半像素是远处背景（深度 1.2 m），近层分位数仍然取到手
    depth2 = depth.copy()
    depth2[:, : int(np.median(project(gt)[:, 0]) * DW / W)] = np.where(
        depth2[:, : int(np.median(project(gt)[:, 0]) * DW / W)] < 1.0, depth2[:, : int(np.median(project(gt)[:, 0]) * DW / W)], 1.2)
    res = depth_hand(wilor_like(gt), depth2, np.full(depth.shape, 2, np.uint8), (W, H), K, RGBDParams())
    assert res["status"] == "ok"
    assert np.linalg.norm(res["joints_cam"][0] - gt[0]) < 0.015


def test_pose_convention_and_jumps():
    t = RA.arkit_pose_to_cv([0, 0, 0, 1, 1, 2, 3])
    assert np.allclose(t[:3, :3], np.diag([1, -1, -1]))
    assert np.allclose(t[:3, 3], [1, 2, 3])
    q = RA.matrix_to_quat_xyzw(RA.quat_xyzw_to_matrix([0.1, 0.2, 0.3, 0.9]))
    assert np.allclose(np.abs(q), np.abs(np.array([0.1, 0.2, 0.3, 0.9]) / np.linalg.norm([0.1, 0.2, 0.3, 0.9])))
    poses = [np.eye(4), np.eye(4), np.eye(4)]
    poses[2] = poses[2].copy()
    poses[2][0, 3] = 0.5
    assert list(pose_jumps(poses, 0.1, 20)) == [False, False, True]
    assert np.allclose(RA.intrinsics_from_metadata({"K": [300, 0, 0, 0, 301, 0, 96, 128, 1]}),
                       [[300, 0, 96], [0, 301, 128], [0, 0, 1]])


def make_r3d(path, n=12, float16=False):
    """合成 Record3D .r3d：ARKit 位姿 = 单位阵（OpenGL 约定），RGB 灰图，深度含一只手。"""
    meta = {"K": K.T.reshape(-1).tolist(), "w": W, "h": H, "dw": DW, "dh": DH, "fps": 30, "cameraType": 1,
            "poses": [[0, 0, 0, 1, 0.001 * i, 0, 0] for i in range(n)],
            "frameTimestamps": [100.0 + i / 30.0 for i in range(n)]}
    with zipfile.ZipFile(str(path), "w") as z:
        z.writestr("metadata", json.dumps(meta))
        for i in range(n):
            ok, jpg = cv2.imencode(".jpg", np.full((H, W, 3), 128, np.uint8))
            z.writestr("rgbd/%d.jpg" % i, jpg.tobytes())
            d = render_depth(gt_hand(i / 30.0)).astype(np.float16 if float16 else np.float32)
            z.writestr("rgbd/%d.depth" % i, liblzfse.compress(d.tobytes()))
            z.writestr("rgbd/%d.conf" % i, liblzfse.compress(np.full((DH, DW), 2, np.uint8).tobytes()))
    return meta


@pytest.mark.parametrize("float16", [False, True])
def test_convert_r3d_and_validate(tmp_path, float16):
    from validate_session import validate
    make_r3d(tmp_path / "cap.r3d", float16=float16)
    info = RA.convert(tmp_path / "cap.r3d", tmp_path / "sess", log=lambda *_: None)
    assert info["frames"] == 12 and info["depth"] == [DW, DH] and info["poses"] == 12
    sess = tmp_path / "sess"
    assert RA.is_iphone_session(sess)
    rep = validate(sess)
    assert rep["ok"], rep["errors"]
    assert rep["kind"] == "iphone" and abs(rep["facts"]["measured_fps"] - 30) < 0.5
    calib = json.loads((sess / "calib.yaml").read_text())
    assert np.allclose(calib["K_rgb"], K)
    # 位姿：OpenGL → OpenCV
    row = (sess / "slam.tum").read_text().splitlines()[2].split()
    assert abs(float(row[0]) - 1 / 30.0) < 1e-4 and abs(float(row[1]) - 0.001) < 1e-6


def test_validate_flags_bad_iphone_session(tmp_path):
    from validate_session import validate
    make_r3d(tmp_path / "cap.r3d")
    RA.convert(tmp_path / "cap.r3d", tmp_path / "sess", log=lambda *_: None)
    (tmp_path / "sess" / "slam.tum").unlink()
    data = np.load(tmp_path / "sess" / "depth.npz")
    np.savez_compressed(tmp_path / "sess" / "depth.npz", depth=np.zeros_like(data["depth"]), conf=data["conf"])
    rep = validate(tmp_path / "sess")
    assert not rep["ok"]
    assert any("slam.tum" in e for e in rep["errors"]) and any("有效深度" in e for e in rep["errors"])


class _FakeBackend(object):
    def __init__(self):
        self.i = 0

    def predict(self, image, calib=None):
        gt = gt_hand(self.i / 30.0)
        self.i += 1
        return {"right": wilor_like(gt), "left": None}


def test_end_to_end_pipeline_with_fake_backend(tmp_path):
    from headcam.stereo_pipeline import StereoParams, run_pipeline
    make_r3d(tmp_path / "cap.r3d", n=20)
    RA.convert(tmp_path / "cap.r3d", tmp_path / "sess", log=lambda *_: None)
    summary = run_pipeline([str(tmp_path / "sess")], tmp_path / "out", StereoParams(), backend_name="fake",
                           export=False, backend=_FakeBackend(), log=lambda *_: None,
                           session_builder=make_session_builder())
    ep = json.loads((tmp_path / "out" / "episodes" / "sess.json").read_text())
    assert ep["source"] == "iphone_lidar" and ep["coordinate_frame"] == "slam_world"
    assert ep["stereo"]["per_frame"]["right"].count("ok") >= 18
    # 世界系：ARKit 单位旋转（OpenGL）→ OpenCV 相机 y、z 取反；平移 0.001*i
    w = np.asarray(ep["hands"]["right"]["joints"][5][0], dtype=float)
    g = gt_hand(5 / 30.0)[0] * [1, -1, -1] + [0.005, 0, 0]
    assert np.linalg.norm(w - g) < 0.015
    assert summary["accepted_episodes"] == 1
    assert ep["iphone"]["per_frame"]["right"] == ep["stereo"]["per_frame"]["right"]
    assert ep["hands"]["right"]["label_status"] == ep["stereo"]["per_frame"]["right"]


def test_export_action_valid_matches_measured_wrists(tmp_path):
    """丢掉的激光雷达帧在 LeRobot 里 action_valid 为 0，占位增量仍在。规则与双目导出相同。"""
    import pyarrow.parquet as pq

    from egodata.action_valid import episode_action_valid, wrist_measured
    from egodata.lerobot_export import pack_action
    from headcam.stereo_pipeline import StereoParams, run_pipeline

    make_r3d(tmp_path / "cap.r3d", n=12)
    RA.convert(tmp_path / "cap.r3d", tmp_path / "sess", log=lambda *_: None)
    data = np.load(tmp_path / "sess" / "depth.npz")
    conf = data["conf"].copy()
    conf[3] = 0
    np.savez_compressed(tmp_path / "sess" / "depth.npz", depth=data["depth"], conf=conf)
    summary = run_pipeline(
        [str(tmp_path / "sess")], tmp_path / "out", StereoParams(), backend_name="fake",
        export=True, backend=_FakeBackend(), log=lambda *_: None,
        session_builder=make_session_builder(),
    )
    ep = json.loads((tmp_path / "out" / "episodes" / "sess.json").read_text())
    assert ep["iphone"]["per_frame"] == ep["stereo"]["per_frame"]
    assert ep["hands"]["right"]["label_status"] == ep["iphone"]["per_frame"]["right"]
    assert ep["iphone"]["per_frame"]["right"][3] == "lowconf"
    expected = episode_action_valid(ep)
    assert expected.shape == (11, 2)
    assert np.all(expected[:, 0] == 0.0)
    assert expected[2, 1] == 0.0 and expected[3, 1] == 0.0
    assert expected[0, 1] == 1.0 and expected[5, 1] == 1.0
    # 手腕数字补全之后，仍因 lowconf 状态判成没测到。清掉三处状态才算实测。
    planted = json.loads(json.dumps(ep))
    planted["hands"]["right"]["wrist_pose"][3] = [0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0]
    planted["hands"]["right"]["confidence"][3] = 0.9
    planted["hands"]["right"]["filled"][3] = False
    assert wrist_measured(planted, "right", 3) is False
    for parent in (planted["iphone"]["per_frame"], planted["stereo"]["per_frame"]):
        parent["right"][3] = "ok"
    planted["hands"]["right"]["label_status"][3] = "ok"
    assert wrist_measured(planted, "right", 3) is True
    # 短缺口会被补上：补出来的增量留在 action 里，action_valid 仍是 0。
    # 从未认到的左手没有可补的端点，增量是「保持不动」。
    assert ep["hands"]["right"]["filled"][3] is True
    action = pack_action(ep, 3)
    assert np.allclose(action[:7], [0, 0, 0, 0, 0, 0, 1])
    assert np.isfinite(action[7:14]).all()
    table = pq.read_table(tmp_path / "out" / "lerobot" / "data" / "chunk-000" / "file-000.parquet")
    exported = np.asarray(table.column("action_valid").combine_chunks().flatten().to_numpy(), dtype=np.float32).reshape(-1, 2)
    exported_action = np.asarray(table.column("action").combine_chunks().flatten().to_numpy(), dtype=np.float32).reshape(-1, 14)
    assert np.array_equal(exported, expected)
    assert np.allclose(exported_action[3], action)
    assert exported[3, 1] == 0.0
    info = json.loads((tmp_path / "out" / "lerobot" / "meta" / "info.json").read_text())
    assert info["features"]["action_valid"]["shape"] == [2]
    note = json.loads((tmp_path / "out" / "lerobot" / "meta" / "egodata_export.json").read_text())
    assert note["valid_action_ratio"] == float(np.mean(expected >= 0.5))
    assert summary["lerobot"]["result"]["valid_action_ratio"] == note["valid_action_ratio"]
    assert "valid_action_ratio" in summary["sessions"][0]["qc"]
