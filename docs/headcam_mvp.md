# 头戴 MVP：短录像到 QC 过滤的 LeRobot 包

English: this is the front half of the headcam pipeline. A session directory can record sync error, carry per-frame camera pose when a trajectory exists, and otherwise say `pose_missing` or `tracking_lost`. It does not invent a world pose. `accuracy` is a measured Umeyama ATE/RPE only when a separate reference trajectory is supplied and the estimate is not itself headset ground truth; otherwise it stays null. Hands, QC, and LeRobot export stay in the existing stereo modules.

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
- [ ] 每一帧都有位姿时，`coordinate_frame` 为 `slam_world`，`slam.tum` 只含这些测量。另给一份真值轨迹、并且估计来源不是头戴真值时，`accuracy` 是测到的 Umeyama ATE/RPE。否则是 null。
- [ ] `export_mvp --skip-hands` 能在没有 WiLoR 的机器上跑完同步和位姿状态。有双目标定、并且帧间隔能落成整数 fps 时，会写一条手部未测量的 episode，QC 走 `qc_stereo_episode`，通过才进入现有 LeRobot 导出。手部列不写，也不填 0。
- [ ] 10 分钟量级的会话只是同一套按帧文件，不另写一条短片段专用路径。本仓库还没有拿 10 分钟实拍测过耗时。

还不算做成：板上同时录双目和 IMU、Kalibr 求解、自采画面上的轨迹和认手效果。这些仍见 `docs/PIPELINE_ARCHITECTURE.md` 第 4 节和 `docs/hardware_day1_checklist.md`。离线 HOT3D 上的轨迹误差见文末。

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

没有另给真值轨迹时，打印是 `accuracy=unmeasured`。给了真值、并且估计来源不是头戴真值、对上至少 3 个不同时间戳时，打印 `accuracy=measured ate_se3_rmse_m=...`。手写数字不会被抄进去。`slam.tum` 的文件头仍写 `accuracy=unmeasured`，测到的结构在 `slam_status.json`。

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
- `accuracy` 只接受这次对上的 Umeyama ATE/RPE（`trajectory_error_v1`）。没有真值、来源是头戴真值、对上的不同时间戳少于 3，或只是一个手写数字时，它是 null。不要把 HOT3D 手腕误差或别人的 SLAM 论文数字写到这次运行上。
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

## HOT3D Quest3 clip-000000：带手导出

这一节用的是 Hugging Face `bop-benchmark/hot3d` 的 Quest3 `clip-000000`（150 帧，序列 `P0002_210fc0da`，参与者 `P0002`）。图像、真值和权重都没有提交。`origin/main` 仍是 `b51dcc9` 的祖先，没有要合并的新提交。

### 会话

`scripts/headcam/hot3d_session.py` 按该帧自己的 FISHEYE624 去畸变，Quest3 侧放的图和外参一起转 -90°，写成虚拟针孔。不写 `slam.tum`。头戴定位写到 `gt/headset_tum.txt`，文件头是 `pose_source=hot3d_gt_mocap`。

```bash
python3 scripts/headcam/hot3d_session.py \
  --clip /tmp/hot3d/clip-000000 \
  --out /tmp/hot3d/sess \
  --euroc-out /tmp/hot3d/euroc \
  --orb-yaml /tmp/hot3d/quest3_pinhole.yaml
```

对得上的量：150 帧，整数 30 fps（中位间隔 33333333 ns），去畸变后 1024×1280，基线 `0.06368567581770954` m，第一帧外参相对其余帧的平移最大差 `1.34e-7` m。左右 `image_timestamps_ns` 相同，`sync_error_ms.left_right_max=0`。`calib.yaml` 的 `dist_coeffs` 是 0，因为图已经是针孔，不是把鱼眼系数填 0。原始模型和 15 个投影参数留在 `metadata.hot3d.source_cameras`。未知：曝光、任务、指令、IMU 样本、IMU 时间偏移、accuracy。这个 tar 没有 IMU，没有跑 stereo-inertial。

### ORB-SLAM3

上游 `4452a3c` 的 `stereo_euroc`，查看器关掉（`System::STEREO, false`）。词汇、yaml、`mav0/cam0|cam1/data/{ns}.png` 和 `times.txt`。没有把绝对路径当成输出文件名，轨迹写在工作目录的 `CameraTrajectory.txt`。

```bash
/tmp/src/ORB_SLAM3/Examples/Stereo/stereo_euroc \
  /tmp/src/ORB_SLAM3/Vocabulary/ORBvoc.txt \
  /tmp/hot3d/quest3_pinhole.yaml \
  /tmp/hot3d/euroc \
  /tmp/hot3d/euroc/times.txt
```

地图 0 只有 1 个关键帧。日志里 `Fail to track local map!` 146 次。`CameraTrajectory.txt` 53 行，不同时间戳只有 2 个（`42139400000000` 和 `42139433333333` 纳秒），其余是同一行的重复。`external_tum --tum-time-scale 1e-9`：`frames=150 ok=2 tracking_lost=148 pose_missing=0 world_pose=false`。这不是可用轨迹，没有拿它乘手，也没有把它叫成导出位姿。

`hot3d_check.trajectory_ate` 先丢掉重复时间戳。剩下 2 个不同时刻，少于 3，不算 ATE，报告里没有 RMSE。若不去重，53 行重复会被最近邻配成看起来很小的误差，那个数没有采用。

### 手、同一个钟、带手的包

WiLoR 用官方源码树 `github.com/rolpotamias/WiLoR`（`PYTHONPATH`），不是 `wilor_mini`：后者的包名是 `wilor_mini`，没有 `wilor.models.load_wilor`。权重在仓库外：`WILOR_CHECKPOINT=/tmp/wilor/wilor_final.ckpt`，`WILOR_DETECTOR=/tmp/wilor/detector.pt`，`WILOR_CONFIG=/tmp/wilor/model_config.yaml`，`MANO_MODEL_DIR=/tmp/wilor/mano`（只有 `MANO_RIGHT.pkl`）。推理 venv：`torch 2.11.0+cpu`，`ultralytics 8.1.34`，`pytorch-lightning 2.6.6`。`chumpy` 在 Python 3.12 / numpy 2 上要补 `inspect.getfullargspec` 和 `np.int` 这些别名，改的是 venv 里的副本，没有进仓库。

```bash
PYTHONPATH=/workspace/scripts:/tmp/src/WiLoR \
MANO_MODEL_DIR=/tmp/wilor/mano \
WILOR_CHECKPOINT=/tmp/wilor/wilor_final.ckpt \
WILOR_DETECTOR=/tmp/wilor/detector.pt \
WILOR_CONFIG=/tmp/wilor/model_config.yaml \
/tmp/wilor-venv/bin/python -c "
from headcam.stereo_pipeline import StereoParams, run_pipeline
run_pipeline(['/tmp/hot3d/sess'], '/tmp/hot3d/hands_cam', params=StereoParams(), export=False)
"
```

`StereoParams()` 是现有默认：关联、另一目重裁、RTS。会话里没有 `slam.tum`，所以这段 episode 的 `coordinate_frame=camera`，位姿是单位阵。后端 565.6 秒。逐目状态：左 `ok=144 jump=6`，右 `ok=122 none=14 jump=13 palm=1`。`filled` 左 5、右 17。管线 QC：`accepted`，坏帧 0，`stereo_inconsistent=20`，`stereo_filled=22`，`stereo_one_view=0`，丢掉标注的帧 30，标注覆盖 0.8（120/150）。这些是丢掉的标注，不是从包里删掉的行。

```bash
python3 scripts/headcam/slam_frames.py \
  --episode /tmp/hot3d/check/camera_episode.json \
  --session /tmp/hot3d/sess \
  --tum /tmp/hot3d/sess/gt/headset_tum.txt \
  --backend external_tum \
  --out /tmp/hot3d/world/sess.json
```

`ok=150 tracking_lost=0 pose_missing=0 timestamp_alignment=timestamps world_points_scope=all_frames pose_source=hot3d_gt_mocap accuracy=unmeasured`，退出码 0。时钟对不上的测试仍然拒绝，没有为了配上而改时间戳。

```bash
python3 scripts/headcam/export_mvp.py \
  --session /tmp/hot3d/sess \
  --episodes /tmp/hot3d/world \
  --out /tmp/hot3d/mvp \
  --slam-backend external_tum \
  --tum /tmp/hot3d/sess/gt/headset_tum.txt \
  --pose-source hot3d_gt_mocap
```

退出码 0。`world_pose=true ok=150 hands=external_episodes pose_source=hot3d_gt_mocap accuracy=null`。世界系 episode 再跑 QC 仍然接受（坏帧 0，yield 1）。LeRobot 149 帧，等于 150 减掉没有下一步动作的最后一帧。`observation.state` 形状 `(140,)`，`action` 形状 `(14,)`。`timestamp` 是 `frame_index/30` 的 float32，第 0 帧是 0，第 148 帧是 `4.933333`（148/30）。`observation.capture_time_ns` 第 0 帧是 `42136166666666`，和图像纳秒一致。`observation.pose` 对回 `T_world_cam` 的最大差 `8.49e-9`。`slam_status` 第 0 帧是 `ok`。`meta/info.json` 和 `meta/egodata_export.json` 都写了 `pose_source=hot3d_gt_mocap`，`accuracy` 是 null。官方 `DatasetInfo` 会警告不认识 `info.json` 里的 `pose_source`，文件里的值还在。

第 0 帧左手置信度 0.413，低于 0.5，`observation.hand_valid` 是 `[0, 0]`，关节块按现有约定写成 0，左手手腕 7 个数仍写入。第 10 帧两手都有效，140 维都不是 0。置信度不低于 0.5 的帧：左 142，右 102，两手同时 97。

### 相对 HOT3D 的手腕误差

比的是 `umetrack_pose.T_world_from_wrist.translation_xyz`（世界坐标，米）。WiLoR 空间只发布了 `MANO_RIGHT.pkl`，没有 `MANO_LEFT.pkl`，没有做 MANO 前向运动学，所以这不是以前报告里的 OpenPose 第 0 点，也不能和「4 段 Quest3、中位约 1.22 cm」直接比。

| 量 | n | 中位 | p90 |
| --- | --- | --- | --- |
| 相机系手腕，相对 `inv(T) @ umetrack` | 278 | 2.064 cm | 2.740 cm |
| 世界系手腕，实测（`filled=false`），相对 umetrack | 256 | 2.094 cm | 2.773 cm |
| 世界系手腕，时序补出来的（`filled=true`） | 22 | 1.651 cm | 2.133 cm |

位姿来源是头戴真值，所以世界系和相机系接近。这些数没有写进 `accuracy`。

### 回归

`HandsPresentPackageTest` 用 4 帧合成相机系（不是 HOT3D 图像）走 `slam_frames` 和 `export_mvp --pose-source hot3d_gt_mocap`。文件头和命令行不一致会拒绝，避免把真值改称 `orbslam3_stereo_euroc`。时钟错 10 秒仍是 `timestamp_alignment=refused`。`HandsPresentOfficialLoaderTest` 没装 lerobot 或 torch 就跳过。`DuplicateTimestampAteTest` 确认两行不同时间加重复行不会写出 `rmse_m`。

系统解释器 `python3 -m unittest tests.test_headcam_mvp tests.test_stereo_qc`：46 项，跳过 2 项（两个官方加载器测试），其余通过。系统 numpy 仍是 1.26.4，pyarrow 18.1.0。

隔离加载器：`lerobot 0.6.1`，`torch 2.11.0+cpu`，`numpy 2.2.6`，`datasets 3.6.0`，`av 15.1.0`，`pyarrow 26.0.0`。`av 19` 没有 `av.option`，0.6.1 要 `av>=15,<16`。这次加载器用的是 pyav 回退，没有 torchcodec。

## 四段 Quest3，一个多 episode 包

`main` 已合进这条分支（C16 EgoVerse 入账，合并提交 `5398af9`）。上面 `clip-000000` 那一节是上一次单独跑的记录，数字留在那里。这一节把同一条路径再跑四段，打成一个数据集。四段都是 Hugging Face `bop-benchmark/hot3d` 的 Quest3，各 150 帧、30 fps。图像、真值、权重没有入库。

| clip | 序列 | 参与者 | 物体 | 手腕路程（左 / 右） |
| --- | --- | --- | --- | --- |
| clip-000000 | `P0002_210fc0da` | P0002 | 芥末瓶、沙拉酱瓶、遥控器、烧瓶、木勺、aria_small | 1.765 m / 1.560 m |
| clip-000300 | `P0003_5e1a34f8` | P0003 | 杯子、花瓶、白板笔、板擦、黑色支架、aria_small | 1.136 m / 0.031 m |
| clip-000700 | `P0010_3224a967` | P0010 | 碗、咖啡壶、蔬菜、华夫、芥末瓶、木勺 | 0.560 m / 0.616 m |
| clip-001100 | `P0017_d2495f49` | P0017 | 手机、键盘、鼠标、白板笔、板擦、黑色支架 | 0.609 m / 1.324 m |

四个参与者，四组物体。clip-000300 的右手几乎不动。去畸变后都是 1024×1280，基线 0.06361–0.06369 m，第一帧外参相对其余帧的平移最大差约 1.3e-7 到 1.8e-7 m。不写 `slam.tum`。头戴定位仍是 `gt/headset_tum.txt`，`pose_source=hot3d_gt_mocap`。这台机器没有现成的 ORB-SLAM3，编译它不是顺手的事，所以这四段没有再跑 stereo，也没有 ATE。

每段先在相机系跑 WiLoR（`StereoParams()` 默认关联），再用该段自己的头戴真值做 `slam_frames.py`。四段都是 `ok=150 timestamp_alignment=timestamps world_points_scope=all_frames`。后端秒数：515.2、469.8、564.1、557.4。

导出入口多了 `--package-annotated`。它只读已经写好的 episode，不用某一个会话的轨迹覆盖其他段。`pose_source` 不一致就拒绝，不写出包。QC 没过的片段留在 `rejected_episodes`，不写进包。`accuracy` 仍是 null，episode 里如果写了数字也不抄。

```bash
python3 scripts/headcam/export_mvp.py \
  --package-annotated \
  --episodes /tmp/hot3d/world_episodes \
  --out /tmp/hot3d/mvp \
  --repo-id local/headcam_mvp
```

退出码 0。`episodes=4 accepted=4 rejected=0 pose_source=hot3d_gt_mocap accuracy=null`。

下面的 QC 和手腕是世界系 episode 上的，也就是进包的那一份。相机系管线当时的坏帧更少：那时候 `camera_poses` 还是单位阵，头戴转动不算运动模糊。乘上头戴真值之后，clip-000300 多出 3 帧模糊（坏帧 13→16），clip-001100 多出 10 帧模糊（坏帧 1→11）。门限仍是坏帧 20%，四段都过了。`stereo_filled` 是「这一帧至少有一只手是补出来的」的帧数，不是两只手的个数相加。

| clip | 帧 | QC | 坏帧 | 丢掉的标注 | stereo_filled | 覆盖 | 世界系手腕中位 | p90 | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clip-000000 | 150 | 接受 | 0 | 29 | 19 | 0.807 | 1.986 cm | 2.794 cm | 258 |
| clip-000300 | 150 | 接受 | 16 | 27 | 24 | 0.820 | 2.108 cm | 3.054 cm | 168 |
| clip-000700 | 150 | 接受 | 0 | 45 | 42 | 0.700 | 2.326 cm | 3.641 cm | 250 |
| clip-001100 | 150 | 接受 | 11 | 84 | 37 | 0.440 | 2.285 cm | 6.084 cm | 196 |
| 四段合在一起 | 600 | 4/4 |  |  |  |  | 2.153 cm | 3.557 cm | 872 |

手腕比的仍是 `umetrack_pose.T_world_from_wrist.translation_xyz`。没有 `MANO_LEFT.pkl`，没有做 MANO 前向。这些数没有写进 `accuracy`。

clip-000000 这次和上一节不是同一次进程。上一节世界系实测是中位 2.094 cm、p90 2.773 cm、n=256。这次左目 `ok=143 jump=7`（上次 144/6），右目 `ok=126 none=13 jump=8 one_view=3`。中位差大约 0.1 cm。表里用的是这一次，四段同一个进程。

clip-000300 的右手路程只有 0.031 m。WiLoR 右目 `none=104 ok=45 jump=1`，104 帧没有预测，所以不进 n。测到的右手中位 2.01 cm、p90 2.21 cm（n=45）。左手实测中位 2.20 cm（n=123）。坏帧 16 里 13 帧是手出画、3 帧是头戴转动带来的模糊。段还在包里。

clip-000700 两只手都在动，跳变多（左 jump 16，右 jump 25），丢掉 45 帧标注，覆盖 0.70。实测中位 2.326 cm，比 clip-000000 大约高 0.3 cm。坏帧 0。段还在包里。

clip-001100 覆盖只有 0.440，是四段里差得明显的一段，没有把它从包里拿掉。左目 `ok=82 one_view=17 none=15 jump=26 palm=10`，右目 `ok=116 jump=26 one_view=7`。头戴角速度中位 27 度/秒，p90 80 度/秒，最大 109 度/秒。状态是 `ok` 的帧里，左手中位 2.40 cm、p90 4.91 cm（n=82），右手中位 2.20 cm、p90 6.14 cm（n=114）。尾巴是少数帧：左手第 63、90 帧大约 26 cm，右手第 29 帧 15 cm。补出来的 40 只手中位 6.69 cm、p90 16.1 cm，比实测差一截，因为补的是 jump。中位数还在 2 cm 附近，所以不是整段坐标系反了。

官方 `lerobot==0.6.1` 的 `LeRobotDataset`，默认后端（装了 `torchcodec 0.11.1+cpu`），逐帧读完 596 行：

- `num_episodes=4`，`num_frames=596`，fps 30，`codebase_version v3.0`。每段 149 行，等于 150 减掉没有下一步的最后一帧。
- 行范围 0–149、149–298、298–447、447–596。每段 `from_timestamp=0`，第 0 行 `timestamp=0`，第 148 行 `4.933333`（148/30）。相对 `frame_index/30` 的最大差 `2.23e-7` 秒。
- `observation.capture_time_ns` 和该段图像纳秒一致，596 行没有一帧对错。clip-000000 第 0 帧仍是 `42136166666666`。
- `observation.state` `(140,)`，`action` `(14,)`。每一行的 `action` 等于该段自己的 `pack_action`（只用段内下一帧）。clip-000000 和 clip-000700 的最后一行两只手腕都缺，增量是占位的保持不动；下一段第 0 帧并没有被写进这一行。clip-000300 最后一行也是占位，它和「接到下一段第 0 帧」的差是 1.31。
- `observation.pose` 对回该段自己的 `T_world_cam`：平移最大差 `5.00e-10`，旋转最大差 `7.42e-8`。四段的平移不一样，没有被收成同一条轨迹。
- `meta/info.json` 和 `meta/egodata_export.json` 的 `pose_source` 是 `hot3d_gt_mocap`，`accuracy` 是 null。

加载器版本：`lerobot 0.6.1`，`torch 2.11.0+cpu`（`cuda False`），`torchvision 0.26.0+cpu`，`numpy 2.2.6`，`pyarrow 26.0.0`，`pandas 2.3.3`，`av 15.1.0`，`torchcodec 0.11.1+cpu`，`datasets 4.8.5`。系统 numpy 仍是 1.26.4。

`MultiEpisodePackageTest` 用两段合成世界 episode（平移 1 m 和 3 m，不是 HOT3D 图像）加上一段置信度被写成 0 的片段。包里只有前两段，时间戳各自从 0 开始，动作不跨段，位姿对回各自的 `T_world_cam`。被拒绝的那段写在 `rejected_episodes`。`pose_source` 一个写成 `orbslam3_stereo_euroc` 就退出码 1，并且不写 `lerobot/`。

系统解释器 `python3 -m unittest tests.test_headcam_mvp tests.test_stereo_qc`：48 项，跳过 2 项（两个官方加载器测试），其余通过。

## ORB-SLAM3 立体轨迹，相对头戴定位

上一节写文档时这台机器没有 ORB-SLAM3。这一节在本机编译了上游 `4452a3c`（`4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4`）的 `Examples/Stereo/stereo_euroc`，并在 HOT3D Quest3 上去畸变针孔上跑完。二进制、词汇、Pangolin、图像和轨迹都没有入库。OpenCV 4.6.0，Eigen 3.4.0，Pangolin 0.6。示例里的查看器参数在本机改成了 `false`，仓库里的代码没有这个补丁。

`origin/main` 仍是 `98a0abc`，已经在 `5398af9` 合进这条分支，没有新提交要再合并。

### 上次 2/150 的原因

150 帧、5 秒不够初始化，这个猜测对不上这次的对照。同一批 `clip-000000` 去畸变图，只改 yaml 里的 `Stereo.T_c1_c2`：

| 外参 | 轨迹行数 | 不同时间戳 | 保存的地图 | `Fail to track local map!` | `New Map created with` |
| --- | --- | --- | --- | --- | --- |
| OpenCV 的 `(R, t)` 原样写入 | 53 | 2 | 地图 0 有 1 个关键帧 | 146 | 4 |
| OpenCV 外参的逆 | 150 | 150 | 地图 0 有 17 个关键帧 | 0 | 1（启动时那张图） |

第一行就是上次文档里的失败：53 行、2 个纳秒时间戳、1 个关键帧、146 次局部地图跟踪失败。第二行同一段 150 帧全部留下，没有 `Frames set to lost`。

OpenCV 的约定是 `p_right = R @ p_left + t`，基线沿 +X 时平移的 x 是负的。ORB-SLAM3 的 `Stereo.T_c1_c2` 存的是 cam2 在 cam1 里的位姿，它在 `precomputeRectificationMaps` 里再取逆，然后才做 `stereoRectify`。把 OpenCV 的矩阵直接写进去，等于校正用了反的外参。`write_orbslam3_yaml` 现在写逆矩阵。`clip-000000` 上 OpenCV 平移是 `[-0.06365264, -0.00196321, -0.00059352]`，yaml 里对应的 x 是 `+0.06367663941`。

图只做了 FISHEYE624 去畸变和与外参一起的 -90°，没有预先做行对齐。ORB-SLAM3 自己校正。`Camera.type` 用的是 `PinHole`。去畸变之后、校正之前，网格上的竖向视差：`clip-000000` 中位 7.78 px、p90 19.11 px、最大 27.74 px（n=192）；下面那条 1800 帧中位 2.75 px、p90 6.25 px、最大 10.28 px（n=192）。双目匹配在第 0 层只看大约 2 个像素的行带，所以这一步要靠 SLAM 内部的校正，不能靠把未对齐的针孔当成已经校正。

### 哪些 clip 能接上，IMU

`clip_definitions.json` 里相邻编号经常是同一个 `sequence_id`，中间仍可能空几秒。接上的条件写在 `hot3d_sequence.contiguous_runs`：同一设备、同一 `sequence_id`，下一段第一帧减上一段最后一帧等于这一段的帧间隔（33333333 或 33333334 ns，允许 2 ns 或 2%）。`concat_clip_records` 对不上就抛 `ValueError`，不返回半截。

训练集 Quest3（`clip_splits.json` 的 train，编号 0–1287）里最长的连续段是 1800 帧、59.967 秒、12 个 clip，三条并列：`P0003_b153ade2`（480–491）、`P0010_3224a967`（706–717）、`P0010_5edabbe9`（747–758）。全部 Quest3 里更长的在 test：`P0006_f065ced6` 1757–1777 共 3150 帧，`P0005_88557ee9` 1486–1499 共 2100 帧。这两个编号向 `train_quest3` 要 tar 是 404，没有下。

已经打进多 episode 包的四段是四条序列，不能拼。`clip-000700` 和 706–717 都是 `P0010_3224a967`，但 700 的末帧到 706 的首帧空了 32.167 秒，没有放进这次的 1800 帧。`clip-000000` 的邻居 1–5 是同一条 `P0002_210fc0da`，clip 1 比 clip 0 的末帧晚 17.467 秒。

`clip-000706.tar` 有 1051 个成员：每帧左右图 `image_1201-1.jpg`、`image_1201-2.jpg`，加上 cameras、hands、hand_crops、objects、info 的 json。没有文件名里带 imu、gyro、accel 的成员。没有 IMU 样本，没有跑 stereo-inertial。

### 跑过的两条轨迹

图像是去畸变针孔，1024×1280，30 fps。yaml 的畸变是 0，因为图已经是针孔。真值轨迹是去畸变并转过 -90° 的左目头戴定位，文件头 `pose_source=hot3d_gt_mocap`。SLAM 保存的 `SaveTrajectoryEuRoC` 把第一帧放在原点，时间戳是纳秒。

`P0010_3224a967`，clip 706–717，1800 帧，基线 0.06361126942862305 m。日志：`Fail to track local map!` 0 次，`Frames set to lost` 0 次，`*Loop detected` 1 次，图集 1 张地图、88 个关键帧。轨迹 1800 行、1800 个不同时间戳。头戴定位路程 7.874 m，平移范围 `[0.154, 0.227, 0.561]` m，时间 44432.967–44492.933 秒。

`clip-000000`，`P0002_210fc0da`，150 帧，基线 0.06368567581770954 m。日志：局部地图失败 0 次，丢帧 0 次，回环 0 次，1 张地图、17 个关键帧。150 行、150 个不同时间戳。头戴定位路程 1.101 m，时长 4.967 秒。

### ATE 和 RPE

`hot3d_check.trajectory_ate` / `trajectory_rpe`。估计轨迹的时间乘 `1e-9`。重复时间戳只留第一行；这两次重复行都是 0。配对是最近邻，超过 0.05 秒不配。对上的不同时间戳少于 3 时不写 rmse。

ATE 比的是相机中心。`alignment=se3` 用 Umeyama，尺度固定为 1。`alignment=sim3` 尺度自由，`similarity_scale` 乘在估计轨迹上再贴近头戴定位；小于 1 表示估计的位移比头戴定位大。RPE 不做这次全局对齐。间隔取 1.0 秒和 5.0 秒，配对应的时间差超过 0.05 秒就丢掉。误差是 `inv(gt_i^{-1} gt_j) @ (est_i^{-1} est_j)` 的平移长度和旋转角。

| 序列 | 对齐 | n | RMSE | 中位 | 最大 | 尺度 |
| --- | --- | --- | --- | --- | --- | --- |
| clip-000000 | SE3 | 150 | 0.006408 m | 0.004792 m | 0.025158 m | 固定 1 |
| clip-000000 | Sim3 | 150 | 0.005039 m | 0.002606 m | 0.024180 m | 1.03103 |
| P0010 706–717 | SE3 | 1800 | 0.005394 m | 0.001590 m | 0.047951 m | 固定 1 |
| P0010 706–717 | Sim3 | 1800 | 0.005275 m | 0.001250 m | 0.047862 m | 1.00694 |

| 序列 | RPE 间隔 | 配对数 | 平移 RMSE | 平移中位 | 旋转 RMSE | 旋转中位 |
| --- | --- | --- | --- | --- | --- | --- |
| clip-000000 | 1.0 s | 121 | 0.010552 m | 0.007837 m | 0.427° | 0.374° |
| P0010 706–717 | 1.0 s | 1771 | 0.007297 m | 0.002146 m | 0.542° | 0.340° |
| P0010 706–717 | 5.0 s | 1651 | 0.010076 m | 0.005947 m | 0.848° | 0.765° |

clip-000000 只有大约 5 秒，5 秒 RPE 只配上 1 对，上表没有把它当成一段误差。主数是尺度固定的 SE3。P0010 的 Sim3 尺度 1.00694，和 6.36 cm 的双目基线放在一起，尺度没有明显漂开。这些数没有写进 `slam_status.accuracy`，也没有抄进四段那个 LeRobot 包。

### `external_tum`

1800 行写成秒，文件头 `pose_source=orbslam3_stereo_euroc`。会话时间戳和它对得上。

```text
run_slam(..., backend="external_tum", pose_source="orbslam3_stereo_euroc")
```

`frame_count=1800 ok_frames=1800 tracking_lost_frames=0 pose_missing_frames=0 pose_coverage=complete world_pose=true imu_present=false imu_used_for_pose=false accuracy=null`。同一文件配 `pose_source=hot3d_gt_mocap` 会 `ValueError`，来源不会被改掉。

### 没有做的手腕对比

这台机器没有 `/tmp/wilor`，也没有 `MANO_RIGHT.pkl` 或 `wilor_final.ckpt`。MANO 要 MPI 的许可确认，没有从别处下载权重。所以没有用这条 SLAM 轨迹重导一个带手的 episode，也没有和同一段头戴定位下的 UmeTrack 手腕误差对照。四段那个包仍是 `pose_source=hot3d_gt_mocap`。没有开 Kaggle。

### 回归

`SyntheticTrajectoryErrorTest`：已知刚体上 SE3 的 RMSE 接近 0；两倍相似变换上 Sim3 找回 `similarity_scale≈0.5`，SE3 的 RMSE 大于 0.5。1.0 秒 RPE 在每步 1.1 倍的平移上是 0.1 m、旋转 0；固定的 SE3 偏置不进 RPE。两行轨迹不写 RPE。

`Hot3dSequenceContinuityTest`：帧间隔正好接上的 clip 拼成严格递增的时间戳；差 2 秒或 `sequence_id` 不同就分成两段；对不上的一组交给 `concat_clip_records` 会抛错。日志夹具只数文本里真的出现的失败和关键帧。

`Orbslam3ExtrinsicConventionTest`：OpenCV 平移 x 为 -0.1 时，yaml 的 `T_c1_c2` 平移 x 是 +0.1。有 cv2 时，校正后的竖向误差小于 `1e-6` px，基线约 0.10 m。

系统解释器 `python3 -m unittest tests.test_headcam_mvp tests.test_stereo_qc`：55 项，跳过 2 项（两个官方加载器测试，没装 lerobot 或 torch），其余通过。numpy 1.26.4，pyarrow 18.1.0，pandas 2.2.3。

## P0010 的 ORB-SLAM3 位姿进了 LeRobot 包

上一节结束时，1800 行轨迹走过了 `external_tum`，`accuracy` 仍是 null，四段那个包仍是 `pose_source=hot3d_gt_mocap`。这一节没有重编 ORB-SLAM3，用的是同一份已经写成秒的轨迹和同一份头戴定位。二进制仍不入库。

`export_mvp --skip-hands --slam-backend external_tum --pose-source orbslam3_stereo_euroc --gt-tum gt_headset_tum.txt` 写出一份缺测手的 LeRobot v3.0 包。一条 episode，1800 帧，fps 30，`hands=absent`。`meta/info.json` 和 `meta/egodata_export.json` 的 `pose_source` 都是 `orbslam3_stereo_euroc`。`accuracy` 不进 `info.json`。四段那个包没有改写。

`--gt-tum` 是另一份轨迹，只用来算误差，不会变成输出位姿。估计文件头和 `--pose-source` 不一致仍是 `ValueError`。估计来源以 `hot3d_gt` 开头、真值文件头没有 `pose_source`、两条来源相同、对上的不同时间戳少于 3，或 `accuracy` 只是一个浮点数时，写出的 `accuracy` 是 null。能写进去的结构是 `kind=trajectory_error_v1`、`method=umeyama`，里面有 SE3、Sim3 和 RPE。

配对和上一节相同：相机中心，最近邻，超过 0.05 秒不配，重复时间戳只留第一行。两条轨迹都已经是秒，时间尺度是 1。SE3 尺度固定为 1，`p_ref = R @ p_est + t`。Sim3 尺度自由。RPE 不做这次全局对齐，间隔 1.0 秒和 5.0 秒，时间差超过 0.05 秒丢掉。写进 `slam_status.json`、`report.json` 和 `egodata_export.json` 的是同一组数：

| 对齐 | n | RMSE | 中位 | 最大 | 尺度 |
| --- | --- | --- | --- | --- | --- |
| SE3 | 1800 | 0.005394100893517117 m | 0.001589850253842316 m | 0.047951287650436665 m | 固定 1 |
| Sim3 | 1800 | 0.005274693901563593 m | 0.0012502277971635127 m | 0.04786236045267196 m | 1.0069400671166455 |

| RPE 间隔 | 配对数 | 平移 RMSE | 平移中位 | 旋转 RMSE | 旋转中位 |
| --- | --- | --- | --- | --- | --- |
| 1.0 s | 1771 | 0.007297399548963262 m | 0.002146327107200282 m | 0.5419324552605177° | 0.340242530578304° |
| 5.0 s | 1651 | 0.010076111886557456 m | 0.005947393886367191 m | 0.8477993870097495° | 0.7647483733644982° |

存下来的 SE3 就是算出这个 RMSE 的对齐。`rotation_row_major` 按行展开：

```text
-0.46141285561926304  -0.5303466731495318   0.7112176762064495
-0.02894455831918674  -0.7922304597534583  -0.6095351599231434
 0.8867132508387278   -0.3018332402470758   0.3501946685330411
```

`translation` 是 `[0.15137217017848723, 1.498233537960144, -0.07064499311260304]`。把这个 `R`、`t` 乘回 1800 个 SLAM 相机中心，RMSE 与上表相同，差是 0。

图像是这段去畸变左目 png（1024×1280），按会话时间戳顺序编成 `left.mp4`，导出时缩到 224。`image_source=source_mp4`。不是占位色块。

官方 `lerobot==0.6.1` 的 `LeRobotDataset`（默认 torchcodec）读完 1800 帧：`codebase_version v3.0`，1 段，fps 30，任务 `unknown`。`slam_status` 全是 `ok`。`timestamp` 相对 `frame_index/30` 的差小于 `1e-4` 秒。`observation.capture_time_ns` 从 `44432966666666` 到 `44492933333333`，和轨迹时间一致。`observation.pose` 对回 `slam.tum` 的平移，最大差 `5.00e-10`。图像 `(3, 224, 224)` float32，抽查的帧落在 `[0, 1]`，相邻帧最大差 0.525。`DatasetInfo` 会警告不认识 `info.json` 里的 `pose_source`，文件里的值还在。加载器版本：lerobot 0.6.1，torch 2.11.0+cpu，torchvision 0.26.0+cpu，numpy 2.2.6，pyarrow 26.0.0，pandas 2.3.3，av 15.1.0，torchcodec 0.11.1+cpu，datasets 4.8.5。系统 numpy 仍是 1.26.4。

### 手腕没有测

要的是 WiLoR 在这条 SLAM 世界系里的手腕，再和 UmeTrack 的 `T_world_from_wrist` 比。比之前先用上面这个 SE3 把 SLAM 世界系对齐到头戴定位。这一步没有做。

这台机器上没有 `MANO_RIGHT.pkl`，也没有 `wilor_final.ckpt` 或 `detector.pt`。`MANO_MODEL_DIR` 和 `WILOR_CHECKPOINT` 是空的。MANO 的许可还要 Justin 确认，没有下载。没有用 MediaPipe 代替 WiLoR。clip 706–717 也没有现成的相机系 WiLoR episode。以前测过手腕的是 clip 0、300、700、1100；clip-000700 和 706 之间空了 32.167 秒，不能拿来填这段。所以没有 SLAM 世界系的手腕厘米数。四段那个包的手腕数仍是头戴真值下的数，没有和这次的 ATE 加在一起。

### 回归

`SlamAccuracyPackagingTest`：已知刚体上 SE3 的 RMSE 小于 `1e-8`，存下来的 `R`、`t` 乘回估计中心，最大差小于 `1e-8`。估计来源写成 `hot3d_gt_mocap` 时 `trajectory_accuracy` 返回 None。浮点 `0.01` 不是测量。两行轨迹不写 accuracy。不给 `--gt-tum` 时 `run_slam` 的 accuracy 是 null；给了就写入 SE3。头戴真值文件配 `--gt-tum` 仍是 null。同一份头戴真值文件请求 `pose_source=orbslam3_stereo_euroc` 会 `ValueError`。

缺测手的导出把 `pose_source` 和测到的 accuracy 写进报告和 `egodata_export.json`，不写进 `info.json`。`--package-annotated` 在每一份被接受的 episode 上都有同一份测量时才抄进包；把 `pose_source` 改成 `hot3d_gt_mocap` 但仍带着那份测量，或把 accuracy 写成 `0.02`，包里的 accuracy 是 null。

系统解释器 `python3 -m unittest tests.test_headcam_mvp tests.test_stereo_qc`：58 项，跳过 2 项（两个官方加载器测试，没装 lerobot 或 torch），其余通过。numpy 1.26.4，pyarrow 18.1.0，pandas 2.2.3。没有开 Kaggle。

## 三条别的 Quest3 录制

上一节只有 `P0010_3224a967` 的 1800 帧。这一节先把 `main` 合进来（`7979ee4`，C17 手部打标对比和 C18 Flora），再在另外三条训练集录制上用同一份 yaml 跑 `stereo_euroc`。没有重跑 P0010 706–717，也没有把上次的 0.005394 m 抄进这张表。

选段读的是 `bop-benchmark/hot3d` 的 `clip_definitions.json` 和 `clip_splits.json`，接上与否交给 `hot3d_sequence.contiguous_runs`。训练集 Quest3 是 1288 个 clip、601 条接得上的段。最长的仍是 1800 帧、三条并列：`P0003_b153ade2`（480–491）、上次跑过的 `P0010_3224a967`（706–717）、`P0010_5edabbe9`（747–758）。后一条和上次是同一个参与者，这次没选。另外两位参与者各自最长的是 `P0021_5d8b0988`（1221–1228，1200 帧）和 `P0002_273c2819`（6–12，1050 帧）。`clip-000000` 是另一条 `P0002_210fc0da`，没有拼进来。这 27 个 `train_quest3` tar 的 HEAD 都是 200。test split 没有下。

配置没改：`write_orbslam3_yaml`，`nFeatures` 1500，`Stereo.T_c1_c2` 是 OpenCV 外参的逆。图是 FISHEYE624 去畸变再转 -90° 的 1024×1280 针孔。二进制仍是上游 `4452a3c`。查看器关掉，帧间 sleep 去掉，这两处只在本机 `/tmp` 的源码里。OpenCV 4.6.0，Eigen 3.4.0，Pangolin 0.6。clip tar 里没有 IMU，没有跑 stereo-inertial。三条的左右时间戳差都是 0。内参采样表每路只建了一次。

三条都跟踪完，没有失败段，所以下面的 RMSE 汇总就是这三条，没有人被排除。失败段的规则写在 `slam_batch.summarize_sequences`：accuracy 是 null，仍计入 `n_sequences`，不进 RMSE。这次失败数是 0，对不上 3 个时间戳的也是 0。日志没有打印 `Frames set to lost`，那一项是 null，不是打印出来的 0。丢失帧是输入帧数减去轨迹里的不同时间戳。`reinits` 是 `Creation of new map with id` 的次数减 1。地图数是保存轨迹时打印的 atlas 张数。

头戴路程是 `gt_headset_tum.txt` 里相邻相机中心的长度，不是 SLAM 轨迹的长度。

| 序列 | clips | 帧 | 跟踪 | 丢失 | 重建 | 地图 | 关键帧 | 时长 | 头戴路程 | 回环 | 局部地图失败 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P0003_b153ade2 | 480–491 | 1800 | 1800 | 0 | 0 | 1 | 157 | 59.967 s | 8.179914918155205 m | 0 | 0 |
| P0021_5d8b0988 | 1221–1228 | 1200 | 1200 | 0 | 0 | 1 | 27 | 39.967 s | 4.315110908077026 m | 0 | 0 |
| P0002_273c2819 | 6–12 | 1050 | 1050 | 0 | 0 | 1 | 79 | 34.967 s | 2.546441790835899 m | 0 | 0 |

| 序列 | SE3 n | SE3 RMSE | SE3 中位 | SE3 最大 | Sim3 RMSE | Sim3 中位 | Sim3 最大 | similarity_scale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P0003_b153ade2 | 1800 | 0.0034270419514742423 m | 0.0012934256425185644 m | 0.04010080541218232 m | 0.0033676029832445825 m | 0.0011297669945467608 m | 0.039541091708160725 m | 1.003924141484529 |
| P0021_5d8b0988 | 1200 | 0.0021233505018789477 m | 0.001463507062016174 m | 0.004974785538924677 m | 0.0008713869718825835 m | 0.0006412149411721794 m | 0.0031657664649935493 m | 1.0115904859598073 |
| P0002_273c2819 | 1050 | 0.015275329451131102 m | 0.003917379911546922 m | 0.0695756897769086 m | 0.015258673604096072 m | 0.004146573387133221 m | 0.06962048955448945 m | 1.011581843720372 |

| 序列 | RPE | 配对数 | 平移 RMSE | 平移中位 | 旋转 RMSE | 旋转中位 |
| --- | --- | --- | --- | --- | --- | --- |
| P0003_b153ade2 | 1.0 s | 1771 | 0.005504165103410066 m | 0.0026301115770167526 m | 0.4623691221276891° | 0.23009896028107774° |
| P0003_b153ade2 | 5.0 s | 1651 | 0.006997038746681433 m | 0.004072977498619308 m | 0.6733943538304771° | 0.42340482799542134° |
| P0021_5d8b0988 | 1.0 s | 1171 | 0.003139545469203001 m | 0.001650462001532968 m | 0.23144775863925898° | 0.1337786285957902° |
| P0021_5d8b0988 | 5.0 s | 1051 | 0.005795969612684733 m | 0.00524491179999531 m | 0.3070281141596381° | 0.22468797163662968° |
| P0002_273c2819 | 1.0 s | 1021 | 0.012348502643444868 m | 0.0016897340478720505 m | 0.46246699197838087° | 0.19860975322346444° |
| P0002_273c2819 | 5.0 s | 901 | 0.017221487009842298 m | 0.0019458886704274534 m | 0.6035279452557174° | 0.26296612194949176° |

RMSE 汇总只含这三条，因为失败数是 0。三个数的中位取排序后的中间那个。

| 量 | n | 最小 | 中位 | 最大 |
| --- | --- | --- | --- | --- |
| SE3 RMSE | 3 | 0.0021233505018789477 m | 0.0034270419514742423 m | 0.015275329451131102 m |
| Sim3 RMSE | 3 | 0.0008713869718825835 m | 0.0033676029832445825 m | 0.015258673604096072 m |
| RPE 1 s 平移 RMSE | 3 | 0.003139545469203001 m | 0.005504165103410066 m | 0.012348502643444868 m |
| RPE 1 s 旋转 RMSE | 3 | 0.23144775863925898° | 0.4623691221276891° | 0.46246699197838087° |
| RPE 5 s 平移 RMSE | 3 | 0.005795969612684733 m | 0.006997038746681433 m | 0.017221487009842298 m |
| RPE 5 s 旋转 RMSE | 3 | 0.3070281141596381° | 0.6035279452557174° | 0.6733943538304771° |

P0003 的尺度 1.0039，和上次 P0010 的 1.0069 差不多，SE3 RMSE 是 3.4 mm。P0021 只有 27 个关键帧，路程 4.32 m，SE3 最大只有 5.0 mm；Sim3 尺度 1.0116，RMSE 降到 0.87 mm，尺度比另外两条离 1 更远。P0002 是这三条里差的那条：SE3 RMSE 15.3 mm，最大 69.6 mm，中位仍是 3.9 mm。1 秒 RPE 的平移中位 1.7 mm，RMSE 12.3 mm，是少数窗口把 RMSE 拉高，不是整段都在 1 cm 以上。它的头戴平移包络大约 0.195 × 0.075 × 0.196 m，路程 2.55 m，人在一小块地方里转。段没有丢掉，也没有重建地图。Sim3 尺度 1.0116，RMSE 几乎没降，所以这段的差主要不是一个全局尺度。

基线：P0003 0.06362609893380695 m，P0021 0.06384111990762019 m，P0002 0.0636856735337763 m。外参漂移都在 2.3e-7 m 以下。

同一批数在 `docs/headcam/orbslam3_quest3_batch.json`。三条原始记录在 `docs/headcam/orbslam3_quest3/`。表可以重算：

```text
PYTHONPATH=scripts python3 -m headcam.slam_batch summarize \
  --records docs/headcam/orbslam3_quest3/P0003_b153ade2.json \
            docs/headcam/orbslam3_quest3/P0021_5d8b0988.json \
            docs/headcam/orbslam3_quest3/P0002_273c2819.json \
  --json docs/headcam/orbslam3_quest3_batch.json \
  --md docs/headcam/orbslam3_quest3_batch.md
```

`SlamBatchSummaryTest` 用合成轨迹。失败段 accuracy 是 null，序列数仍算它，RMSE 汇总不含它。估计文件头写成 `hot3d_gt_mocap` 时不算 SLAM。系统解释器 `python3 -m unittest tests.test_headcam_mvp tests.test_stereo_qc`：63 项，跳过 2 项（两个官方加载器测试，没装 lerobot 或 torch），其余通过。没有开 Kaggle。没有下 MANO。

## 还缺什么

四段合成的 LeRobot 包仍是 `pose_source=hot3d_gt_mocap`。P0010 clip 706–717 的缺测手包仍是上一节那一份，这次没有重跑，也没有并进上面的三序列汇总。没有 WiLoR 手，没有和 UmeTrack 比过手腕。MANO 许可还要 Justin 确认，权重没有下。

这三条都跟踪完了，不能当成全部 HOT3D 都这样。没跑的还有 `P0010_5edabbe9`（747–758，1800 帧）、`P0013_0ec32d10`（846–852，1050 帧）、`P0017_44db2c1d`（1061–1066，900 帧）、`P0018_15ad9743`（1125–1129，750 帧）。P0002 的 SE3 RMSE 是 0.015275329451131102 m，最大 0.0695756897769086 m。Quest3 clip 里没有 IMU，没有 stereo-inertial，没有自采头戴。test split 里 3150 帧和 2100 帧的连续段没有下。没有开 Kaggle。

