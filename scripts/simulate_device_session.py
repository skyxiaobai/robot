#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""假录制器：用 HOT3D 的真实双目画面，按设备会话格式（§7.7）写出会话目录。

目的是在设备到货前，把“录制落盘格式 → 校验 → 管线”整条链路先跑通。设备上的录制程序
只要写出一样的目录，后面一行都不用改。

    python scripts/simulate_device_session.py --hot3d data/train_quest3/clip-000000 --out sim/sess01
    python scripts/simulate_device_session.py ... --synth-imu            # 由真值位姿合成 200 Hz IMU（不是真 IMU）
    python scripts/simulate_device_session.py ... --fault desync_5ms     # 故意注入故障，检查 validate_session 能抓到

故障：drop_frame（删一行时间戳）、desync_5ms（右目晚 5 ms）、baseline_mm（标定平移按毫米写）、
no_calib（删掉 calib.yaml）。
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

FAULTS = ("drop_frame", "desync_5ms", "baseline_mm", "no_calib")


def synth_imu(session, rate_hz=200.0):
    """由 slam.tum 合成 IMU：角速度 = 相邻位姿相对旋转 / dt（相机系），比力 = R^T (a - g)。
    只用来测试格式和对齐，不代表真实 IMU 噪声。"""
    from headcam.hand_pose import load_tum_trajectory
    t, poses = load_tum_trajectory(Path(session) / "slam.tum")
    P = np.stack(poses)
    ti = np.minimum(np.arange(t[0], t[-1] + 1.0 / rate_hz, 1.0 / rate_hz), t[-1])
    pos = np.stack([np.interp(ti, t, P[:, k, 3]) for k in range(3)], 1)
    vel = np.gradient(pos, ti, axis=0)
    acc = np.gradient(vel, ti, axis=0)
    g = np.array([0.0, -9.81, 0.0])  # HOT3D 世界系 y 朝上
    rows = ["timestamp_s,gx,gy,gz,ax,ay,az  # synthesized_from_poses"]
    idx = np.clip(np.searchsorted(t, ti) - 1, 0, len(t) - 2)
    for k, ts in enumerate(ti):
        i = idx[k]
        R0, R1 = P[i, :3, :3], P[i + 1, :3, :3]
        dR = R0.T @ R1
        ang = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
        axis = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
        n = np.linalg.norm(axis)
        w = axis / n * ang / (t[i + 1] - t[i]) if n > 1e-12 else np.zeros(3)
        f = R0.T @ (acc[k] - g)
        rows.append("%.6f,%s" % (ts, ",".join("%.6f" % v for v in list(w) + list(f))))
    (Path(session) / "imu.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return len(ti)


def inject_fault(session, fault):
    s = Path(session)
    if fault == "drop_frame":
        lines = (s / "timestamps.csv").read_text(encoding="utf-8").splitlines()
        del lines[len(lines) // 2]
        (s / "timestamps.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    elif fault == "desync_5ms":
        p = s / "stereo" / "timestamps_lr.csv"
        lines = p.read_text(encoding="utf-8").splitlines()
        out = [lines[0]]
        for line in lines[1:]:
            i, a, b = line.split(",")
            out.append("%s,%s,%.9f" % (i, a, float(b) + 0.005))
        p.write_text("\n".join(out) + "\n", encoding="utf-8")
    elif fault == "baseline_mm":
        p = s / "calib.yaml"
        lines = p.read_text(encoding="utf-8").splitlines()
        k = lines.index("T_right_left:")
        for r in range(1, 4):
            vals = lines[k + r].strip()[3:-1].split(", ")
            vals[3] = "%.10g" % (float(vals[3]) * 1000)
            lines[k + r] = "  - [%s]" % ", ".join(vals)
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    elif fault == "no_calib":
        (s / "calib.yaml").unlink()
    else:
        raise ValueError("未知故障 %s" % fault)


def main(argv=None):
    ap = argparse.ArgumentParser(description="用 HOT3D 生成设备格式的会话目录")
    ap.add_argument("--hot3d", required=True, help="HOT3D-Clips 的 clip 目录")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--with-gt", action="store_true", help="同时写 gt/hands_gt.json（需要 MANO_MODEL_DIR）")
    ap.add_argument("--synth-imu", action="store_true")
    ap.add_argument("--fault", choices=FAULTS, default=None)
    a = ap.parse_args(argv)
    from headcam.hot3d_adapter import convert_clip
    info = convert_clip(a.hot3d, a.out, mano_dir=os.environ.get("MANO_MODEL_DIR") if a.with_gt else None,
                        max_frames=a.max_frames)
    if a.synth_imu:
        info["imu_samples"] = synth_imu(a.out)
    if a.fault:
        inject_fault(a.out, a.fault)
        info["fault"] = a.fault
    print(info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
