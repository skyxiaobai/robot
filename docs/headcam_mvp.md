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
- [ ] `export_mvp --skip-hands` 能在没有 WiLoR 的机器上跑完同步和位姿状态。有双目标定、并且帧间隔能落成整数 fps 时，会写一条手部未测量的 episode，QC 走 `qc_stereo_episode`，通过才进入现有 LeRobot 导出。手部列不写，也不填 0。
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

# 先看帧数。不是每一帧都 ok 时退出码 1，world_hands=refused。
python scripts/headcam/summarize_slam.py --session /data/sess01 --tum CameraTrajectory.txt

# 不跑手部。有标定且 QC 通过时，仍写出 LeRobot 包：位姿和 slam_status 在，手部列不在。
python scripts/headcam/export_mvp.py --session /data/sess01 --out outputs/sess01_mvp \
  --slam-backend external_tum --tum CameraTrajectory.txt --skip-hands

# 只有 summarize 打印 world_hands=allowed 之后，才跑手部、QC、LeRobot。
python scripts/headcam/export_mvp.py --session /data/sess01 --out outputs/sess01_mvp \
  --slam-backend external_tum --tum CameraTrajectory.txt

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

`world_pose=true` 只在每一帧都是 `ok` 时成立。这时 `coordinate_frame` 仍写旧名 `slam_world`，`common.frames` 把它收成 `frame_convention.name = world`（公制 SLAM，上轴未核实）。不是 `world_yup`。覆盖不全时 episode 保持 `camera`。

现有手部代码在没有 `slam.tum` 时仍会把 episode 里的 `camera_poses` 填成单位阵，并把 `coordinate_frame` 写成 `camera`。这表示坐标还在相机系，不是「相机停在世界原点」。`export_mvp` 会在 episode 的 `hand_pose.slam.camera_pose_meaning` 里写 `identity_placeholder_camera_frame`。不要把这种单位阵再存成 `slam.tum`，否则旧的读取逻辑会把它标成 `slam_world`。

覆盖不全时，ok 帧的矩阵留在 `slam_status.json`，不写进 `slam.tum`，也不交给 `stereo_pipeline`（那边会最近邻）。若调用方已经有一份相机系 episode，`export_mvp` 只对 `status=ok` 的帧算 `p_world = T_world_cam @ p_cam`，结果放在 `hand_pose.slam.world_hands`。`tracking_lost` / `pose_missing` 是 null。手腕主数组不动，所以 `frames.sanity_check` 和 `qc_scorecard` 仍按相机系看这段。没有逐帧激光深度时不写 `iphone.per_frame`，成绩单里的 `depth_valid` 保持 null。`stereo.per_frame` 有的话抄到 `track_status`。

每一帧都 ok 时，同一公式写进手腕和 `camera_poses`，并标 `world`。已经是世界系的 episode 不会再乘一次。`camera_yup` 不按这条 OpenCV 公式变换。

文件里本来就是单位阵的一行（相机就在原点）会保留，状态是 `ok`。那是测量，不是补出来的。

## 检查一份已有的相机系 episode

WiLoR（或别的手部程序）写出的统一 episode 若还在 OpenCV 相机系，可以不重跑手部，直接乘已经算好的位姿并做 `frames.sanity_check`。这条命令不启动 WiLoR，也不写精度；打印里的 `accuracy=unmeasured` 是固定的，不会采用 `slam_status.json` 里带来的数字。

```bash
# 真实文件以后换成自己的路径。episode 的手腕必须还在相机系。
python scripts/headcam/slam_frames.py \
  --episode /data/wilor_clip.json \
  --session /data/sess01 \
  --tum CameraTrajectory.txt \
  --out outputs/wilor_clip_checked.json

# 已经有 slam_status.json 时不必再读会话。
python scripts/headcam/slam_frames.py \
  --episode tests/fixtures/headcam/wilor_partial/episode.json \
  --slam-status tests/fixtures/headcam/wilor_partial/slam_status.json \
  --out /tmp/wilor_partial_checked.json
```

`sanity_ok=false` 时退出码是 1。覆盖不全时 `world_points_scope=ok_frames_only`：世界点只在 `ok` 帧，`tracking_lost` / `pose_missing` 是 null，手腕主数组仍是相机系。把 `--out` 的文件再送进同一条命令，不会把世界点乘第二次。

帧数相同还不够。episode 和 slam 帧两边都有时间戳时，同一序号必须落在半帧门限内。一边是秒、一边是纳秒，或时间对不上，就 `timestamp_alignment=refused`，退出码 1，不乘 `T_world_cam`，也不自动换算。缺一边的时间戳时仍按帧数对齐，打印里写 `timestamp_alignment=frame_count`。

`tests/fixtures/headcam/wilor_partial/` 是手写的 4 帧相机系手腕，状态是 ok / tracking_lost / tracking_lost / ok。它不是 WiLoR 输出，也不是测过的轨迹。

## 标定和同步

`calib.json` 的 schema 是 `headcam_calib_v1`。

- 左右目外参：`T_right_left`，点从左相机变到右相机，与 `scripts/headcam/hand_pose.py` 里的 Kalibr / OpenCV / 简单 YAML 一致。
- `scripts/retarget/calibrate.py` 的 `AxisCalibration` 是人手到机械臂的轴缩放。把它当成相机标定会直接报错。
- `sync_error_ms.left_right_max` / `left_right_median` 来自 `stereo/timestamps_lr.csv`，单位毫秒。两列都是秒时，算法与 `validate_session` 相同（绝对值乘 1000）。两列都是纳秒时改为除以 1e6，并在 `time_unit` 里记下 `nanoseconds`。两列一个是秒、一个是纳秒，或有一边无法判断时，这两个字段是 null，不写一个巨大的毫秒数。
- `sync_error_ms.imu_cam` 只在 metadata 或命令行已经给出 `imu_cam_time_offset_s`（秒）时填写，取绝对值再换成毫秒。这里不估计时间偏移。
- 会话自己的 `timestamps.csv` 如果步长像纳秒，只在 SLAM 状态里警告。不把会话时间戳除以 1e9，因为下游按秒解释速度。

## ORB-SLAM3 / 开源 VIO 怎么接

仓库不编译、不内置 ORB-SLAM3。推荐顺序：

1. 按官方说明安装 ORB-SLAM3，准备 `ORBvoc.txt` 和双目惯性 yaml。
2. 把 `stereo/left.mp4`、`stereo/right.mp4` 按 `timestamps.csv` 抽成 EuRoC 的 `mav0/cam0/data` 与 `mav0/cam1/data`。抽帧不在本模块里。
3. 把 `imu.csv`（`timestamp_s,gx,gy,gz,ax,ay,az`）改成 EuRoC `imu0/data.csv` 的纳秒时间戳。
4. 运行 `Examples/Stereo/stereo_euroc` 或 `Examples/Stereo-Inertial/stereo_inertial_euroc`，留下 `CameraTrajectory.txt`。`SaveTrajectoryEuRoC` 的时间是纳秒，四元数是 `qx qy qz qw`。关键帧轨迹盖不住的帧会变成 `tracking_lost`。
5. `python scripts/headcam/slam_vio.py --session SESSION --backend external_tum --tum CameraTrajectory.txt --tum-time-scale 1e-9 --out OUT`
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

到货当天的复制命令在下一节。`docs/hardware_day1_checklist.md` 里的厘米级闭环是采集清单上的目标，不是本仓库已经测到的数。

## Day 1 on a real machine

在一台已经装好求解器的机器上，用一节约 10 分钟的录像走完「外部轨迹 → 帧数 → 再决定要不要手部」。10 分钟只是同一套按帧文件，本仓库没有拿实拍测过耗时，也没有轨迹误差数字。

假设：

- ORB-SLAM3、OpenVINS 或 Basalt 按它自己的 README 编译过。本仓库不下载词表、不编译这些程序。
- `ORBvoc.txt` 由你自己放在求解器目录里。
- 这一节 Python 能 `import numpy`。手部模型这次先不要跑。
- 会话已经是 `record_session` 的目录。时间戳单位是秒。

```text
/data/sess01/
  metadata.json
  timestamps.csv                 frame_index,timestamp_s
  imu.csv                        timestamp_s,gx,gy,gz,ax,ay,az
  calib.yaml
  stereo/left.mp4
  stereo/right.mp4
  stereo/timestamps_lr.csv       frame_index,left_s,right_s
  stereo/exposure.csv            有就放，没有也可以先看位姿
```

`imu.csv` 的轴是否等于求解器要的机体轴，取决于你的标定。这里不改轴，也不对 IMU 积分。

帧数怎么读（`summarize_slam.py` 打出来的四行计数）：

| 计数 | 含义 | 能不能跑世界系手部 |
|---|---|---|
| `ok` 等于 `frames`，`tracking_lost=0`，`pose_missing=0` | 每一帧都在半帧门限内有轨迹点 | `world_hands=allowed`。可以去掉 `--skip-hands` |
| `tracking_lost>0` | 文件里有轨迹，但这几帧最近的点超过半帧 | `world_hands=refused`。不要把最近邻填进去 |
| `pose_missing>0` 且 `ok=0` | 没有可用轨迹（没给文件、文件是空的，或求解器没写出轨迹） | `world_hands=refused` |
| `time_unit_mismatch=true` | 用于关联的轨迹时间和会话帧时间一个是秒、一个是纳秒 | `world_hands=refused`。不自动换算，跨单位的最近邻也不写成位姿。确认后加 `--tum-time-scale 1e-9`，不要放宽门限 |

`accuracy=unmeasured` 一直印着。不要把它换成别的数字。

```bash
# 0. 还没有轨迹时，计数应是 pose_missing，退出码 1。
python scripts/headcam/summarize_slam.py --session /data/sess01

# 1. ORB-SLAM3。下面是上游 stereo_inertial_euroc 的常见参数形状。
#    先看你的二进制不带参数时打印的 Usage，再替换路径。
#    本仓库执行过的是 stereo_euroc（无 IMU），见文末「离线 EuRoC」。
#    stereo_inertial_euroc 还没在这里跑过。
#    会话不是 EuRoC：mav0/cam0/data、mav0/cam1/data 里的 png 要用时间戳命名，
#    times 文件一行一个时间。抽帧请按 ORB-SLAM3 的 EuRoC 说明做，不要用本仓库现编。
export ORB_SLAM3_ROOT=$HOME/ORB_SLAM3
"$ORB_SLAM3_ROOT/Examples/Stereo-Inertial/stereo_inertial_euroc" \
  "$ORB_SLAM3_ROOT/Vocabulary/ORBvoc.txt" \
  "$ORB_SLAM3_ROOT/Examples/Stereo-Inertial/EuRoC.yaml" \
  /data/euroc_sess01 \
  /data/euroc_sess01/times.txt
# 成功时当前目录出现 CameraTrajectory.txt。
# UZ-SLAMLab stereo_euroc 的 SaveTrajectoryEuRoC 把第一列写成纳秒
# （内部秒 × 1e9，fixed 小数 6 位），四元数是 qx qy qz qw，不是 EuRoC 的 qw qx qy qz。
# 不要把这份文件当成秒。summarize 会印 tum_time_unit=nanoseconds 且
# time_unit_mismatch=true。不要改 max_gap_s。确认后在第 2 步加上 --tum-time-scale 1e-9。

# 2. 只读这文件。不启动求解器，不写精度。
python scripts/headcam/summarize_slam.py \
  --session /data/sess01 \
  --tum ./CameraTrajectory.txt
# 纳秒轨迹（Basalt 常见；个别导出会到 1e18）在确认之后才加缩放：
#   --tum-time-scale 1e-9

# 3. 不跑手。有标定且 QC 通过时会写 LeRobot 包，手部列不在里面。
#    第 2 步如果加了 --tum-time-scale，这里和下面的手部命令也要带上同一个数。
python scripts/headcam/export_mvp.py \
  --session /data/sess01 \
  --out outputs/sess01_mvp \
  --slam-backend external_tum \
  --tum ./CameraTrajectory.txt \
  --skip-hands

# 4. 只有上一步 summarize 的退出码是 0 且 world_hands=allowed，才跑手部。
python scripts/headcam/export_mvp.py \
  --session /data/sess01 \
  --out outputs/sess01_mvp \
  --slam-backend external_tum \
  --tum ./CameraTrajectory.txt
```

退出码不是 0 时停在第 3 步。`--allow-partial` 只让 `summarize_slam.py` 的退出码变成 0，打印里仍然是 `world_hands=refused`。把它加到 `export_mvp` 上，手部会在相机系里跑，并在 stderr 警告；残缺帧不会乘上世界位姿。

OpenVINS、Basalt 也停在第 2 步之前，只要你先得到 8 列 TUM（`timestamp tx ty tz qx qy qz qw`，秒，左目在世界系）：

```bash
# OpenVINS：上游 serial 例子的形状。估计日志的列经常是时间、四元数、位置，还带速度。
# 那个文件不是 TUM。抽成上面的 8 列再交给 --tum。确认命令以你本地的 --help 为准。
./run_serial_msckf config/euroc_mav/estimator_config.yaml /data/euroc_sess01

# Basalt：上游 vio 的形状。trajectory 的时间常常是纳秒。
# 不要把原文件直接交给 --tum。确认后用 --tum-time-scale 1e-9，或先写成秒。
basalt_vio \
  --dataset-path /data/euroc_sess01 \
  --cam-calib /data/euroc_calib.json \
  --dataset-type euroc \
  --config-path /data/euroc_config.json \
  --show-gui 0

python scripts/headcam/summarize_slam.py --session /data/sess01 --tum trajectory_seconds.txt
python scripts/headcam/export_mvp.py --session /data/sess01 --out outputs/sess01_mvp \
  --slam-backend external_tum --tum trajectory_seconds.txt --skip-hands
```

## 已知限制

- 没有 SLAM 轨迹时就是 `pose_missing`。episode 可以仍在相机系里被 QC 和导出，那不是世界系标签。
- `accuracy` 永远是 null。不要把 HOT3D 手腕误差或别人的 SLAM 论文数字写到这次运行上。
- 残缺轨迹的世界点只出现在 `hand_pose.slam.world_hands`。丢失帧是 null，不用最近邻，也不把整段标成 `world` 或 `world_yup`。
- `record_session capture` 不是硬件同步采集。IMU 驱动、曝光寄存器、左右时间戳都还要在板上接。
- 手部仍要 WiLoR / HaMeR / MediaPipe 那套环境和 `scripts/run_stereo_pipeline.py` 的参数。轨迹不完整时，不带 `--allow-partial` 的 `export_mvp` 拒绝跑手部。`--skip-hands` 不跑手部；有标定且 QC 通过时仍可导出，手部列不写。
- LeRobot 导出需要 pyarrow 和 pandas。当前解释器没有时，报告里给出 `ego_to_lerobot.py` 的命令，不写空数据集。仓库测试用的系统 numpy 是 1.26.4，配 pyarrow 18.1.0。官方 `lerobot==0.6.1` 要 numpy 2 和 torch，装在单独的 venv 里，不改系统 numpy。
- 有手部测量的导出仍把无效手的关节块写成 0，并用 `observation.hand_valid` 标成无效。没测量过的片段不走这条，避免把 0 当成手腕在原点。
- 本模块没有在 10 分钟实拍上测过运行时间或轨迹误差。
- 纳秒轨迹不会自动除以 1e9。`tests/fixtures/headcam/offline_skip_hands/` 用合成的秒轨迹和纳秒轨迹检查这件事，不代表实机精度。

## 离线 EuRoC：真实 ORB-SLAM3 轨迹进了 external_tum

这是无人机公开数据上的管线检查，不是头戴精度。

2026-10-16 在这台 Linux VM 上编过上游 UZ-SLAMLab ORB-SLAM3 `4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4`（Pangolin v0.6、apt 的 OpenCV 4.6 和 Eigen）。`Examples/Stereo/stereo_euroc` 能链上。查看器在那份本地源码里关掉了，改动不在本仓库。ETH 的官方压缩包超时，序列来自 Hugging Face `pepijn223/euroc-mirror` 的 `V1_01_easy.zip`。只跑了 `EuRoC_TimeStamps/V101.txt` 的前 400 帧（约 19.95 秒），不是整段飞行。

`CameraTrajectory.txt` 有 400 行，没有表头。第一列是纳秒，带 `.000000`。四元数是 `qx qy qz qw`。`KeyFrameTrajectory.txt` 来自 `SaveKeyFrameTrajectoryEuRoC`，同样是纳秒，这一次只有 11 个关键帧。会话时间戳是这些图像纳秒除以 1e9，写成秒。

`summarize_slam.py --backend external_tum`：

| 命令 | 计数 |
|---|---|
| 不传 `--tum-time-scale` | `frames=400 ok=0 tracking_lost=400 pose_missing=0 time_unit_mismatch=true world_hands=refused`，退出码 1 |
| `--tum-time-scale 1e-9` | `frames=400 ok=400 tracking_lost=0 pose_missing=0 world_pose=true world_hands=allowed accuracy=unmeasured`，退出码 0 |

最大关联间隙是 2.4e-7 秒，来自把 1e18 的纳秒放进 float64 再乘 1e-9。半帧门限大约 0.025 秒，对得上，没有改门限。当时的 `export_mvp --skip-hands` 退出码 0，`hands=skipped`，`accuracy` 仍是 null，还没有写 LeRobot 包。解析器接受了表头注释、`.000000` 和 xyzw。这一次 400 帧都在轨迹里，没有丢帧可标成 `tracking_lost`。包是下一节用其中 40 行写的。

`tests/fixtures/headcam/euroc_orbslam3_v101/CameraTrajectory_excerpt.txt` 是其中第 361–400 行，文件头写了来源。回归测试用这 40 行，不提交数据集、词表或二进制。

同一段前缀对 EuRoC 地面真值做了一次 SE3 对齐（左相机位置：机体位姿乘 `mav0/cam0/sensor.yaml` 的 `T_BS`，Umeyama，尺度固定）。地面真值大约晚 1 秒才开始，对上 379/400 帧。ATE RMSE 0.026 m，中位数 0.022 m，最大 0.051 m。这是 EuRoC 无人机上 ORB-SLAM3 双目管线的检查，不是头戴设备的精度，也没有写进 `accuracy`。

`slam_frames.py` 现在会拒绝时钟对不上的序号配对。`tests/fixtures/headcam/wilor_partial/` 仍是 4 帧合成相机系，时间戳是 0 到 0.1 秒。把它改成纳秒、或整段平移 10 秒再对着这份 slam 状态，结果是 `timestamp_alignment=refused`，手腕不被乘。没有改时间戳去凑 EuRoC。

## 离线 EuRoC：skip-hands 写出了 LeRobot 包

仍是无人机公开数据上的管线检查，不是头戴精度。这次没有重编 ORB-SLAM3。轨迹用的是上面那份摘录（第 361–400 行，40 帧）。图像和 `sensor.yaml`、`imu0/data.csv` 从同一份 `V1_01_easy.zip` 按这 40 个纳秒文件名抽出来。

`scripts/headcam/euroc_session.py` 做的换算：

- 图像文件名的纳秒用整数除法写成秒（9 位小数），不先塞进 float64。左右文件名相同，所以 `sync_error_ms.left_right_max` 测得是 0。这是测到的差，不是缺测填 0。
- `T_right_left = inv(T_BS_cam1) @ T_BS_cam0`，`p_B = T_BS @ p_C`。`imu0` 的 `T_BS` 是单位阵，所以 `T_cam_imu = inv(T_BS_cam0)`。
- 畸变是 radial-tangential 的 4 个数，没有把缺的 k3 补成 0。
- IMU 只保留落在这 40 帧时间里的 391 行，列顺序原样抄，不旋转轴。
- 曝光、任务、手部、`imu_cam_time_offset_s` 没有来源，metadata 的 `unknown` 里写着。不写 `stereo/exposure.csv`。

fps 用图像间隔中位数，是 20，和 yaml 里的 `rate_hz` 一致。

```bash
python scripts/headcam/euroc_session.py \
  --euroc /tmp/euroc/mav --tum tests/fixtures/headcam/euroc_orbslam3_v101/CameraTrajectory_excerpt.txt \
  --out /tmp/euroc/sess --episode-id euroc/V1_01_easy_excerpt
python scripts/headcam/export_mvp.py \
  --session /tmp/euroc/sess --out /tmp/euroc/mvp \
  --slam-backend external_tum \
  --tum tests/fixtures/headcam/euroc_orbslam3_v101/CameraTrajectory_excerpt.txt \
  --tum-time-scale 1e-9 --skip-hands
```

结果：退出码 0。`world_pose=true`，`ok=40`，`tracking_lost=0`，`pose_missing=0`，`hands=skipped`，`accuracy=null`。QC：`accepted=yes`，`bad_fraction=0`，`stereo_unmeasured=40`，`label_coverage=0`（手没测，默认覆盖率门限是 0，不因此拒绝），`blur=0`。

QC 的 CSV 里 `valid_action_ratio=0`：没有实测手腕，所以没有可监督的动作。包本身不写这些手部列，导出说明里的 `valid_action_ratio` 是 null。

第一版包把图像秒写进 `timestamp`（float64）。用 pyarrow 18.1.0 读，40 行，位姿对回 `T_world_cam` 的平移差是 0、旋转差约 7e-16。官方加载器不接受这个钟。

隔离环境（不改系统 numpy 1.26.4）：

```bash
python3 -m venv /tmp/lerobot-venv
/tmp/lerobot-venv/bin/pip install 'torch==2.11.0' 'torchvision==0.26.0' 'numpy>=2,<2.3' \
  --index-url https://download.pytorch.org/whl/cpu
printf 'torch==2.11.0+cpu\ntorchvision==0.26.0+cpu\nnumpy==2.2.6\n' > /tmp/lerobot-constraints.txt
/tmp/lerobot-venv/bin/pip install 'lerobot[dataset]==0.6.1' \
  -c /tmp/lerobot-constraints.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

装上的版本：`lerobot 0.6.1`（`CODEBASE_VERSION` 是 `v3.0`），`torch 2.11.0+cpu`（`cuda False`，`torch.version.cuda is None`），`torchvision 0.26.0+cpu`，`numpy 2.2.6`，`pyarrow 26.0.0`，`pandas 2.3.3`，`av 15.1.0`，`torchcodec 0.11.1+cpu`，`datasets 4.8.5`。

`LeRobotDataset(repo_id="local/euroc_excerpt", root=/tmp/euroc/mvp/lerobot, video_backend="pyav")` 能打开元数据，40 帧，fps 20，`codebase_version v3.0`。逐帧 `__getitem__` 在解视频时拒绝：

```text
FrameTimestampError: query timestamps tensor([1.4037e+09]) > tolerance_s=0.0001
loaded timestamps: 0.0000, 0.0500, ... 1.9500
video: .../videos/observation.image/chunk-000/file-000.mp4
backend: pyav
```

加载器把 `timestamp` 加上 episode 的 `from_timestamp`（这里是 0）去对视频时间轴，容差默认 `1e-4` 秒。绝对时间大约 1.4e9 秒，对不上 0 到 1.95 秒的视频。而且 `__getitem__` 把标量浮点收成 float32，这一量级的 40 个图像秒变成同一个 `1403715328.0`。

缺测手的导出现在按 0.6.1 写 `timestamp = frame_index / fps`（float32）。图像时刻改放 `observation.capture_time_ns`（int64 纳秒，由 episode 里的秒 `round(秒 * 1e9)` 得到）。仍不写 `observation.state`、`action`、手腕列，也不把缺测手填 0。`accuracy` 仍是 null。位姿只在 `slam_status=ok` 时有 7 个数，否则是 null。

按上面的命令重新导出后，用默认后端 torchcodec 逐帧读完 40 帧，没有版本或 schema 错误：

- 帧数 40，等于 ok 帧。`slam_status` 全是 `ok`。任务文本是 `unknown`。
- `timestamp` 相对 `frame_index/20` 的最大差 `4.77e-8` 秒，小于加载器容差 `1e-4`。
- `observation.capture_time_ns` 从 `1403715291262142976` 到 `1403715293212143104`，和 `timestamps.csv` 的纳秒一致。
- `observation.pose` 对回 `T_world_cam`：平移最大差 `4.99e-10`，旋转最大差 `6.33e-8`（加载器把列表收成 float32）。
- 解码图像是 `torch.float32`、形状 `(3, 224, 224)`，数值在 `[0, 1]`。40 帧的标准差大约 0.194 到 0.225，相邻帧都不相同。来源是真实 png 缩到 224，不是 16×16 占位。
- `info.json` 的 `codebase_version` 是加载器自己的 `v3.0`。`stats` 含图像、timestamp、capture_time_ns、pose。episodes 长度 40，`from_timestamp=0`，`to_timestamp=2`。tasks 只有 `unknown`。`hands=absent`，`accuracy=null`。

回归：`UnmeasuredLeRobotPackageTest` 仍用摘录前 3 行加一帧对不上的时间。图像是测试里的 8×8 色块，不是 EuRoC 像素。没装 pyarrow 或 ffmpeg 时跳过。`timestamp` 是 `frame_index/20`，`observation.capture_time_ns` 等于原来的纳秒，第 4 行位姿是 null。`OfficialLeRobotLoaderTest` 用官方 `LeRobotDataset` 再读这份小包（含那一帧 `tracking_lost`）。没装 lerobot 或 torch 时跳过，原因写在 `skipTest` 里。系统解释器上 `python3 -m unittest tests.test_headcam_mvp tests.test_stereo_qc` 是 42 项、跳过 1 项、通过。装了 lerobot 的 venv 里这两个测试类通过。

## 下一步

下一次可以做的事：把一份真实 WiLoR 相机系 episode 交给 `slam_frames.py`。时间戳必须和轨迹在同一个钟上，否则会 `timestamp_alignment=refused`。本仓库没有这样的文件。不要把 `sanity_ok` 写成精度，也不要把残缺轨迹标成 `world_yup`。头戴相机、曝光寄存器和 WiLoR 继续缺着。整段 V1_01_easy 和 stereo-inertial 这次没有跑。短记录到 QC 过滤后的 LeRobot 包，官方 0.6.1 加载器已经读过上面这 40 帧。
