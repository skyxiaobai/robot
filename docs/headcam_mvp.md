# 头戴 MVP：短录像到 QC 过滤的 LeRobot 包

English: this is the front half of the headcam pipeline. A session directory can record sync error, carry per-frame camera pose when a trajectory exists, and otherwise say `pose_missing` or `tracking_lost`. It does not invent a world pose, and it does not report a SLAM accuracy number. Hands, QC, and LeRobot export stay in the existing stereo modules.

中文：把一节短录像收成「同步有没有测到、世界位姿有没有测到、然后走现有 QC 和 LeRobot」。没有 SLAM 结果时明确写 `pose_missing`，不把单位阵写进 `slam.tum` 冒充世界系。

## 这条链路做什么

```text
record_session  →  §7.7 会话目录
sync_calib      →  calib.json（含 sync_error_ms，没测到就是 null）
slam_vio        →  slam_status.json；只有每一帧都 ok 才写 slam.tum
export_mvp      →  现有双目手部管线 → egodata QC → LeRobot
```

后半段没有重写。手部仍是 `scripts/headcam/stereo_pipeline.py`，QC 仍是 `scripts/egodata/qc.py` 和 `scripts/egodata/stereo_qc.py`，导出仍是 `scripts/ego_to_lerobot.py`。C 系列实验脚本不动。

## 两周验收

下面是这版 MVP 算做成的条件。不是精度指标，也不是采集小时数。

- [ ] `record_session init` 得到 §7.7 目录：有 metadata、IMU 表头、`stereo/`。没有视频、没有 `timestamps.csv`、没有 `slam.tum`。
- [ ] `record_session import` 能把已有的左右视频、时间戳、IMU、标定放进这个目录。没给时间戳文件时，不按 fps 编出一列时间。
- [ ] `sync_calib` 写出的 `calib.json` 能用 `load_calibration` 的同一套外参读回来（`p_right = R @ p_left + T`）。`sync_error_ms.left_right_max` 与 `validate_session` 的左右时间差同一算法。没有 `stereo/timestamps_lr.csv` 时该字段是 null，不是 0。
- [ ] 没有 `slam.tum`、也没执行 ORB-SLAM3 时，`slam_status.json` 里每一帧都是 `pose_missing`，`T_world_cam` 为 null，输出目录里没有 `slam.tum`。
- [ ] 轨迹盖不住的帧是 `tracking_lost`，不拿相邻位姿填上。这种残缺轨迹不会交给手部管线。
- [ ] 每一帧都有位姿时，`coordinate_frame` 为 `slam_world`，`slam.tum` 只含这些测量。报告里的 `accuracy` 仍是 null。
- [ ] `export_mvp --skip-hands` 能在没有 WiLoR 的机器上跑完同步和位姿状态。加上已有 episode 时，QC 走 `qc_stereo_episode`，通过的片段才进入现有 LeRobot 导出。
- [ ] 10 分钟量级的会话只是同一套按帧文件，不另写一条短片段专用路径。本仓库还没有拿 10 分钟实拍测过耗时。

还不算做成：板上同时录双目和 IMU、Kalibr 求解、ORB-SLAM3 的轨迹误差、自采画面上的认手效果。这些仍见 `docs/PIPELINE_ARCHITECTURE.md` 第 4 节和 `docs/hardware_day1_checklist.md`。

## 命令

```bash
# 空目录。不写视频，不写位姿。
python scripts/headcam/record_session.py init --out /data/sess01 --fps 30 --episode-id headcam/sess01

# 把已经录好的文件放进目录。没有 --timestamps 就不会编造时间。
python scripts/headcam/record_session.py import \
  --out /data/sess01 \
  --left left.mp4 --right right.mp4 \
  --timestamps timestamps.csv --timestamps-lr timestamps_lr.csv \
  --imu imu.csv --calib calib.yaml

# 默认只写 ffmpeg/v4l2 计划，不打开相机。两个进程不能代替硬件同步。
python scripts/headcam/record_session.py capture --out /data/sess01

# 同步误差 + 相机标定 JSON。IMU 时间偏移不在这里估计。
python scripts/headcam/sync_calib.py --session /data/sess01 --out /data/sess01/calib.json

# 位姿。没有轨迹时退出码仍是 0；--strict 才在 world_pose=false 时返回 1。
python scripts/headcam/slam_vio.py --session /data/sess01 --out outputs/sess01_slam

# 有自己跑出来的 TUM 再读入。这不会启动 ORB-SLAM3，也不测量误差。
python scripts/headcam/slam_vio.py --session /data/sess01 --backend external_tum \
  --tum CameraTrajectory.txt --out outputs/sess01_slam

# 串起来。没有手部权重时先看同步和位姿。
python scripts/headcam/export_mvp.py --session /data/sess01 --out outputs/sess01_mvp --skip-hands

# 手部权重齐了再跑现有双目管线、QC、LeRobot。
python scripts/headcam/export_mvp.py --session /data/sess01 --out outputs/sess01_mvp

# 手部之前已经导出过统一 episode：只补位姿状态、QC 和 LeRobot。
python scripts/headcam/export_mvp.py --session /data/sess01 --episodes outputs/episodes --out outputs/sess01_mvp
```

`capture` 的默认就是不打开设备。要真的录，加 `--run`，并且本机要有 ffmpeg 和两个视频节点。失败时不把缺的时间戳填上。

会话里建议有的文件（规格 §7.7）：

```text
session/
  metadata.json
  timestamps.csv                 frame_index,timestamp_s
  imu.csv                        timestamp_s,gx,gy,gz,ax,ay,az
  calib.yaml                     现有 load_calibration 能读的 YAML
  calib.json                     同一套外参 + sync_error_ms
  stereo/left.mp4
  stereo/right.mp4
  stereo/timestamps_lr.csv       frame_index,left_s,right_s
  stereo/exposure.csv            frame_index,left_exposure_s,right_exposure_s
  slam.tum                       只有完整世界轨迹才作为手部管线的输入
  slam_status.json               逐帧 ok / pose_missing / tracking_lost
```

`imu.csv` 只是原始样本。本模块不对它积分。

## 位姿状态

| 状态 | 含义 | `T_world_cam` |
|---|---|---|
| `ok` | 这一帧在半帧门限内有轨迹点。门限与 `validate_session` 相同：相邻帧间隔中位数的一半 | 4×4，来自那一行 TUM |
| `tracking_lost` | 有轨迹文件，但这一帧对不上 | null |
| `pose_missing` | 没有轨迹，或 ORB-SLAM3 没跑成 | null |

`world_pose=true` 只在每一帧都是 `ok` 时成立，`coordinate_frame` 写 `slam_world`。否则是 `camera`。

现有手部代码在没有 `slam.tum` 时仍会把 episode 里的 `camera_poses` 填成单位阵，并把 `coordinate_frame` 写成 `camera`。这表示坐标还在相机系，不是「相机停在世界原点」。`export_mvp` 会在 episode 的 `hand_pose.slam.camera_pose_meaning` 里写 `identity_placeholder_camera_frame`。不要把这种单位阵再存成 `slam.tum`，否则旧的读取逻辑会把它标成 `slam_world`。

覆盖不全时，ok 帧的矩阵留在 `slam_status.json`。不交给 `stereo_pipeline`，因为那边用最近邻，会把 `tracking_lost` 配到别的帧上。

文件里本来就是单位阵的一行（相机就在原点）会保留，状态是 `ok`。那是测量，不是补出来的。

## 标定和同步

`calib.json` 的 schema 是 `headcam_calib_v1`。

- 左右目外参：`T_right_left`，点从左相机变到右相机，与 `scripts/headcam/hand_pose.py` 里的 Kalibr / OpenCV / 简单 YAML 一致。
- `scripts/retarget/calibrate.py` 的 `AxisCalibration` 是人手到机械臂的轴缩放。把它当成相机标定会直接报错。
- `sync_error_ms.left_right_max` / `left_right_median` 来自 `stereo/timestamps_lr.csv`，单位毫秒。
- `sync_error_ms.imu_cam` 只在 metadata 或命令行已经给出 `imu_cam_time_offset_s`（秒）时填写，取绝对值再换成毫秒。这里不估计时间偏移。

## ORB-SLAM3 / 开源 VIO 怎么接

仓库不编译、不内置 ORB-SLAM3。推荐顺序：

1. 按官方说明安装 ORB-SLAM3，准备 `ORBvoc.txt` 和双目惯性 yaml。
2. 把 `stereo/left.mp4`、`stereo/right.mp4` 按 `timestamps.csv` 抽成 EuRoC 的 `mav0/cam0/data` 与 `mav0/cam1/data`。抽帧不在本模块里。
3. 把 `imu.csv`（`timestamp_s,gx,gy,gz,ax,ay,az`）改成 EuRoC `imu0/data.csv` 的纳秒时间戳。
4. 运行 `Examples/Stereo-Inertial/stereo_inertial_euroc`，留下 `CameraTrajectory.txt`。关键帧轨迹盖不住的帧会变成 `tracking_lost`。
5. `python scripts/headcam/slam_vio.py --session SESSION --backend external_tum --tum CameraTrajectory.txt --out OUT`
6. 再跑 `export_mvp`。只有每一帧都对得上，手部才会乘这组 `T_world_cam`。

库存的 `stereo_inertial_euroc` 吃的是 EuRoC 目录，不是 §7.7 会话。环境变量设齐之后可以试着执行，但这不是默认路径：

```bash
export ORB_SLAM3_BIN=/path/to/stereo_inertial_euroc
export ORB_SLAM3_VOCAB=/path/to/ORBvoc.txt
export ORB_SLAM3_SETTINGS=/path/to/EuRoC.yaml
export ORB_SLAM3_RUN=1
python scripts/headcam/slam_vio.py --session /data/sess01 --backend orbslam3 --run-orbslam3 --out outputs/slam
```

进程失败、超时，或没有留下 `CameraTrajectory.txt` / `slam.tum` / `KeyFrameTrajectory.txt`，结果都是 `pose_missing`。非零退出码时不读取半成品轨迹。

OpenVINS、Basalt 只要最后给出同一份 TUM（`timestamp tx ty tz qx qy qz qw`，左目相机在世界系），就走 `--backend external_tum`。不需要把求解器放进仓库。

到货当天若要看闭环，仍按 `docs/hardware_day1_checklist.md` 自己记数。那个清单里的厘米级要求不是本仓库已经测到的结果。

## 已知限制

- 没有 SLAM 轨迹时就是 `pose_missing`。episode 可以仍在相机系里被 QC 和导出，那不是世界系标签。
- `accuracy` 永远是 null。不要把 HOT3D 手腕误差或别人的 SLAM 论文数字写到这次运行上。
- 残缺轨迹不会逐帧拼进现有 episode。下一步才是：只对 `ok` 帧做 `p_world = T_world_cam · p_cam`，丢失帧的手腕保持缺测，而不是最近邻。
- `record_session capture` 不是硬件同步采集。IMU 驱动、曝光寄存器、左右时间戳都还要在板上接。
- 手部仍要 WiLoR / HaMeR / MediaPipe 那套环境和 `scripts/run_stereo_pipeline.py` 的参数。`--skip-hands` 停在位姿和同步。
- LeRobot 导出需要 pyarrow。当前解释器没有时，报告里给出 `ego_to_lerobot.py` 的命令，不写空数据集。
- 本模块没有在 10 分钟实拍上测过运行时间或轨迹误差。

## 下一步

1. 在一台装好 ORB-SLAM3 的机器上，用一节实拍（或 EuRoC 样例）写出 `CameraTrajectory.txt`。
2. 用 `--backend external_tum --tum` 读入，看 `slam_status.json` 里 `ok` / `tracking_lost` 的帧数。先不要填精度。
3. 确认每一帧都 `ok` 之后，再跑不带 `--skip-hands` 的 `export_mvp`，让现有管线把关节变到 `slam_world`。
4. 若只有一部分帧跟踪成功，再改手部合成：按 `pose_status` 逐帧使用 `T_world_cam`，丢失帧不要最近邻。
