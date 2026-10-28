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

# 这一轮算不算合格。不合并，不新算 Umeyama，不下 MANO。
PYTHONPATH=scripts python3 -m headcam.verification_gate
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


## 其余 Quest3 训练段，以及 P0002 的误差落在哪

`origin/main` 是 `476b9e7`，已经在这条分支里（合并 `7979ee4`）。这次没有新的 main 要再合。

上一节的表是三条。这一节用同一份 yaml、同一个 `stereo_euroc`，把当时没进表的训练集最长段跑完：`P0013_0ec32d10`（846–852，1050 帧）、`P0017_44db2c1d`（1061–1066，900 帧）、`P0018_15ad9743`（1125–1129，750 帧），以及 P0010 的两条 1800 帧。`P0010_3224a967`（706–717）在这台机器上重跑，没有把上次的 0.005394100893517117 m 抄进表。`P0003_b153ade2` 和 `P0021_5d8b0988` 没有在这台机器上重跑，表里仍是上一节的记录。`P0002_273c2819` 的表行也仍是上一节那一次（SE3 RMSE 0.015275329451131102 m，79 个关键帧）。这台机器另跑了一次 P0002，用来看误差落在哪，没有再占一行，否则同一条录制会算两次。重跑的序列记录是 `docs/headcam/orbslam3_quest3/P0002_273c2819_rerun.json`，时间线是 `P0002_273c2819_timeline.json`。

配置没改。`Stereo.T_c1_c2` 是 OpenCV 外参的逆，`nFeatures` 1500。图是 FISHEYE624 去畸变再转 -90° 的 1024×1280 针孔。二进制是上游 `4452a3c`。查看器关掉、帧间 sleep 去掉，这两处只在本机 `/tmp` 的源码里。Quest3 的 tar 里没有 IMU，没有跑 stereo-inertial。六条都跟踪完，丢失帧 0，重建 0，地图 1。失败数是 0，所以下面的 RMSE 汇总是全部 8 条。失败段的规则没变：accuracy 为 null，仍计入 `n_sequences`，不进 RMSE。

头戴路程是 `gt_headset_tum.txt` 里相邻相机中心的长度，不是 SLAM 轨迹的长度。这台机器上六条的路程在 `docs/headcam/orbslam3_quest3/headset_path.json`。P0002 这次仍是 2.546441790835899 m，和上一节同一条录制。P0003 和 P0021 的路程仍是上一节的数，那两份真值不在这台机器上。

| 序列 | clips | 帧 | 跟踪 | 关键帧 | 局部地图失败 | 回环 | 头戴路程 | SE3 RMSE | SE3 中位 | SE3 最大 | Sim3 RMSE | 尺度 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P0003_b153ade2 | 480–491 | 1800 | 1800 | 157 | 0 | 0 | 8.179914918155205 m | 0.0034270419514742423 m | 0.0012934256425185644 m | 0.04010080541218232 m | 0.0033676029832445825 m | 1.003924141484529 |
| P0021_5d8b0988 | 1221–1228 | 1200 | 1200 | 27 | 0 | 0 | 4.315110908077026 m | 0.0021233505018789477 m | 0.001463507062016174 m | 0.004974785538924677 m | 0.0008713869718825835 m | 1.0115904859598073 |
| P0002_273c2819 | 6–12 | 1050 | 1050 | 79 | 0 | 0 | 2.546441790835899 m | 0.015275329451131102 m | 0.003917379911546922 m | 0.0695756897769086 m | 0.015258673604096072 m | 1.011581843720372 |
| P0010_3224a967 | 706–717 | 1800 | 1800 | 83 | 1 | 0 | 7.873505693348391 m | 0.008205109394453782 m | 0.0020800494546338866 m | 0.08389409291523889 m | 0.008100251379623766 m | 1.0080545963907157 |
| P0010_5edabbe9 | 747–758 | 1800 | 1800 | 88 | 0 | 0 | 8.293934545287039 m | 0.04942053479288229 m | 0.018706983087599 m | 0.15811514024780318 m | 0.0474363595017323 m | 0.9183096427504743 |
| P0013_0ec32d10 | 846–852 | 1050 | 1050 | 57 | 0 | 0 | 3.7130443101606123 m | 0.006196191926232237 m | 0.0047783261252145924 m | 0.013933751896470524 m | 0.0021213276661974896 m | 1.0301082162274657 |
| P0017_44db2c1d | 1061–1066 | 900 | 900 | 41 | 0 | 0 | 4.487354715790248 m | 0.007054496104415027 m | 0.003974965173258681 m | 0.03713993532543655 m | 0.007045156718925644 m | 1.0023946088947122 |
| P0018_15ad9743 | 1125–1129 | 750 | 750 | 107 | 0 | 0 | 2.3366922201385307 m | 0.004247753259978912 m | 0.0020693655177331195 m | 0.017867154336932808 m | 0.0038702017042583432 m | 1.0250505738252662 |

前三行是上一台机器的记录。后五行是这台机器的。P0002 的表行也是上一台机器的。同一份配置两次跑出来的数并不逐位相同：P0010 706–717 上次 SE3 RMSE 0.005394100893517117 m、88 个关键帧、1 次回环、局部地图失败 0 次；这次 0.008205109394453782 m、83 个关键帧、回环 0、局部地图失败 1 次。P0002 上次 0.015275329451131102 m、79 个关键帧；这次重跑 0.014536769894140768 m、80 个关键帧。

RMSE 汇总是 `slam_batch.summarize` 写出来的，8 条都在里面。偶数条时中位是排序后下标 `len//2` 的那个，也就是第 5 个，不是两个中间数的平均。

| 量 | n | 最小 | 中位 | 最大 |
| --- | --- | --- | --- | --- |
| SE3 RMSE | 8 | 0.0021233505018789477 m | 0.007054496104415027 m | 0.04942053479288229 m |
| Sim3 RMSE | 8 | 0.0008713869718825835 m | 0.007045156718925644 m | 0.0474363595017323 m |
| RPE 1 s 平移 RMSE | 8 | 0.003139545469203001 m | 0.010156141119185518 m | 0.021701871293961936 m |
| RPE 1 s 旋转 RMSE | 8 | 0.23144775863925898° | 0.46246699197838087° | 2.18104128038373° |
| RPE 5 s 平移 RMSE | 8 | 0.004999968218054385 m | 0.01216880834929074 m | 0.05195165630965154 m |
| RPE 5 s 旋转 RMSE | 8 | 0.3070281141596381° | 0.6733943538304771° | 2.5702983472974688° |

SE3 中位是 P0017。最大是 `P0010_5edabbe9`。P0013 是这批里 Sim3 真的把 RMSE 降下来的一条：尺度 1.0301082162274657，SE3 0.006196191926232237 m，Sim3 0.0021213276661974896 m。其余几条尺度靠近 1 时 RMSE 几乎不动；`P0010_5edabbe9` 的尺度是 0.9183096427504743，估计位移比头戴真值大，Sim3 RMSE 仍是 0.0474363595017323 m。

表可以重算：

```text
PYTHONPATH=scripts python3 -m headcam.slam_batch summarize \
  --records docs/headcam/orbslam3_quest3/P0003_b153ade2.json \
            docs/headcam/orbslam3_quest3/P0021_5d8b0988.json \
            docs/headcam/orbslam3_quest3/P0002_273c2819.json \
            docs/headcam/orbslam3_quest3/P0010_3224a967.json \
            docs/headcam/orbslam3_quest3/P0010_5edabbe9.json \
            docs/headcam/orbslam3_quest3/P0013_0ec32d10.json \
            docs/headcam/orbslam3_quest3/P0017_44db2c1d.json \
            docs/headcam/orbslam3_quest3/P0018_15ad9743.json \
  --json docs/headcam/orbslam3_quest3_batch.json \
  --md docs/headcam/orbslam3_quest3_batch.md
```

`P0002_273c2819_rerun.json` 不在这组输入里。

### 逐帧误差

`trajectory_qc` 用和 `trajectory_ate` 相同的 SE3 Umeyama、尺度固定为 1，写出每一帧的相机中心距离，再和能从现有数据算出的量放在一起：头戴真值的线速度和角速度、clip 接缝、关键帧间隔、去畸变左目的亮度和四邻域拉普拉斯方差。Pearson 在样本少于 3 或某一列没有起伏时是 null，不写成 0。这些相关不是原因。上游 `stereo_euroc` 在跟踪成功时不打印该帧的 inlier 个数，所以 `tracked_features` 是 null，不是 0。

超过 `max(0.02 m, 3×中位)` 的连续帧记成一段尾巴。这段里误差向量均值的长度是整段平移，分量标准差是它散开了多少。相邻帧的相对运动只在两端都落在这段里时才算，不跨出边界。尾巴外面的位置误差和帧间误差另记。合成轨迹上，接缝处的尖峰留在接缝；速度没有起伏时相关是 null；开头一段固定平移的帧间误差接近 0，整段平移还在。估计文件头若是头戴真值，不算 SLAM。

时间线里的亮度和模糊来自 png 还在时的同一次轨迹。png 删掉之后又跑了一次 `diagnose_files`，补上尾巴的平移和尾巴内外的速度。两份用的是同一份 `CameraTrajectory.txt`。

P0002 重跑的时间线，1050 帧，RMSE 0.014536769894140768 m，中位 0.0040423255435268765 m，最大 0.06606663538241843 m。最大在第 39 帧，t = 1.3000000000029104 s，clip 6，clip 内第 39 帧。最差的 10%（105 帧）占误差平方和的 0.9139904154615902。

只有一段尾巴：t = 0 到 1.7000000000043656 s，52 帧，RMSE 0.061654164438708026 m，最大 0.06606663538241843 m。误差向量均值的长度 0.06062457775745827 m，分量标准差 0.006478080074763555 m。这段内部 51 步的平移 RMSE 0.002916849944043936 m，旋转 RMSE 0.15150243565023788°。尾巴外面 998 帧的位置 RMSE 0.004926238619860379 m，中位 0.0037992139759612216 m，最大 0.016672495626819422 m。外面 997 步的平移 RMSE 0.0008946829663664574 m，旋转 RMSE 0.10195609464164868°。

钉得住的是：这次重跑的 RMSE 集中在开头大约 1.7 秒、clip 6 的 52 帧。对齐之后这 52 帧的误差接近一个 6.06 cm 的固定向量，帧与帧的运动仍贴着头戴真值。1.7 秒之后位置 RMSE 是 4.9 mm。上一节那次的最大是 0.0695756897769086 m、中位 0.003917379911546922 m，形状相近。那次的轨迹没有留在这台机器上，所以不能说上次的最大也在 t = 1.3 s。

接缝半径 2 帧：靠近接缝的 36 帧 RMSE 0.004596351884180073 m，其余 1014 帧 0.014767194393526986 m。最大误差不在接缝上。Sim3 尺度 1.0111402616823868，RMSE 0.014520470758477963 m，和 SE3 几乎一样。日志只有一行 `New Map created with 473 points`。`Fail to track local map!` 0 次，回环 0 次，丢失帧 0。

头戴真值的速度：尾巴里线速度中位 0.08106907524195156 m/s、最大 0.11823643444934767 m/s，角速度中位 27.157818804918172°/s、最大 59.36103521175984°/s。尾巴外线速度中位 0.05523558855871445 m/s、最大 0.28569690017074567 m/s，角速度中位 13.08001922435474°/s、最大 89.36073854298883°/s。全程最快的线速度和角速度都在尾巴外面，而外面的位置 RMSE 是 4.9 mm。误差和线速度的 Pearson 是 -0.0042732365635008845，和角速度是 0.21729428169529802。角速度高的三分之一 RMSE 0.02061055204441826 m，低的三分之一 0.0048591037958722666 m；开头那 52 帧比后面的中位更快，所以这个三分之一和开头叠在一起。

去畸变左目：误差和亮度的 Pearson -0.3413054670531752，暗的三分之一 RMSE 0.024318594902062025 m，亮的三分之一 0.0037304203226282313 m。和拉普拉斯方差的 Pearson -0.32667803649941196。png 已经删掉，没有把开头 1.7 秒拿掉再算一次，所以亮度和模糊没有和开头分开。关键帧 80 个。误差和「距上一关键帧的秒数」的 Pearson 是 -0.2062428236710675，靠近关键帧的帧误差更大，因为开头的关键帧更密。

假设，没有钉死：开头这一段和后面差的是一个整段平移，一次全局 SE3 对不齐，两段各自的运动仍贴着头戴真值。保存下来的是退出之后的位姿。日志里没有失败行，也就没有时间戳可以把某一次重定位对上 t = 1.3 s。

### 另外几条的时间线

`P0010_5edabbe9` 是表里最差的一条，和 P0002 重跑是同一种开头。尾巴 t = 0 到 5.866666666996025 s，177 帧，RMSE 0.1483826203615348 m，最大 0.15811514024780318 m，在 t = 5.666666666998935 s、clip 748、clip 内第 20 帧。误差向量均值的长度 0.14796632492630718 m，分量标准差 0.006412714276026166 m。内部 176 步平移 RMSE 0.003801115231638534 m，旋转 RMSE 0.31066738078634787°。外面 1623 帧位置 RMSE 0.017538277969787148 m，中位 0.018312769746322736 m，最大 0.04188936898933839 m。开头之后的大约 54 秒仍在 1.8 cm 这一档，所以这条录制不只有开头那一段。尾巴里的头戴线速度最大 0.555320686855585 m/s，外面最大 0.6961325536796875 m/s；角速度最大 113.54883660146606°/s 对 159.93871974466782°/s。接缝附近 RMSE 0.04930948987117039 m，远离接缝 0.04942475648740109 m。日志只有 `New Map created with 433 points`。没有 `Relocalized!!`。

`P0010_3224a967` 这次重跑的开头是好的：第一秒 RMSE 0.0047590293371802955 m。两段尾巴在后面。t = 40.83333333399787 s 到 41.5 s，21 帧，RMSE 0.06690665923535118 m，最大 0.08389409291523889 m（第 1231 帧，t = 41.03333333399496 s，clip 714，clip 内第 31 帧）。误差向量均值的长度 0.06511305113924523 m，分量标准差 0.008884286153530414 m，内部平移 RMSE 0.006545046898055068 m。第二段 t = 50.83333333399787 s 到 51.13333333400078 s，10 帧，RMSE 0.03755588929274601 m，均值长度 0.036698439960333284 m。尾巴外 1769 帧 RMSE 0.0027183510591479282 m。接缝附近 0.0018910876604215576 m，远离接缝 0.008351658500222259 m。日志按出现顺序是 `New Map created with 557 points`、一次 `Fail to track local map!`、一次 `Relocalized!!`。这两行没有时间戳。假设：这次重定位和 t = 41 s 的那段平移是同一件事。证据只到「日志里有这两行，时间对不上」。

`P0017_44db2c1d` 的位置尾巴在结尾。t = 29.0 s 到 29.86666666599922 s，27 帧，RMSE 0.029923890175277655 m，最大 0.03713993532543655 m（第 883 帧，clip 1066，clip 内第 133 帧）。均值长度 0.028725053615471167 m，分量标准差 0.004841160260735892 m。内部 26 步平移 RMSE 0.0019381870832655263 m，旋转 RMSE 0.2309233971861965°。外面 871 帧位置 RMSE 0.004759691623143043 m。1 秒 RPE 的旋转 RMSE 是 2.18104128038373°，中位 0.22044795593624356°，p90 0.47864267760142754°，p99 11.990798346732126°，最大 12.474401512021744°。最差的 5%（44 对）占旋转平方和的 0.9856310088054617。旋转最大的几对从 t = 27.93333333299961 s 到 28.66666666600213 s，平移最大 0.07328094290740568 m。这些 1 秒窗口的末端落进 t = 29 s 的位置尾巴，单帧旋转仍是 0.23°。配对和 `trajectory_rpe` 相同，多出来的分布在 `docs/headcam/orbslam3_quest3/P0017_44db2c1d_rpe1s.json`。这段位置尾巴里的头戴线速度中位 0.5475637987657411 m/s、最大 0.9806722779835109 m/s，最大线速度就在这段里。误差和线速度的 Pearson 是 0.5574313691089604。和 P0002 的开头不同，P0002 全程最快的运动在尾巴外面。误差和拉普拉斯方差的 Pearson 是 -0.6127724210338129，没有把 t = 29 s 拿掉再算。接缝附近 RMSE 0.0037513392883389506 m，远离接缝 0.007141198856562141 m。

`P0013_0ec32d10` 没有超过 2 cm 的尾巴。最大 0.013933751896470524 m 在 t = 33.533333334002236 s，clip 852，clip 内第 106 帧。第一秒 RMSE 0.010057037681941924 m。接缝附近 0.004847035873468951 m，远离接缝 0.006238730183498315 m。上面的 Sim3 尺度是这条上能对上的量。假设：一个大约 3% 的全局尺度。帧间平移 RMSE 0.0011723643108388338 m。

`P0018_15ad9743` 也没有 2 cm 的尾巴。最大 0.017867154336932808 m 在 t = 24.033333334002236 s，clip 1129，clip 内第 121 帧。那一秒的 RMSE 0.016202075663577436 m。第一秒 0.002614617514255781 m。接缝附近 0.0020929679223527014 m，远离接缝 0.00430058991803123 m。

这五条时间线上，接缝半径 2 帧的 RMSE 都没有高于远离接缝的那一组。

### 重算一条时间线

```text
PYTHONPATH=scripts python3 -m headcam.trajectory_qc \
  --estimate /tmp/slam/P0002_273c2819/CameraTrajectory.txt \
  --gt /tmp/slam/P0002_273c2819/gt_headset_tum.txt \
  --estimate-scale 1e-9 \
  --clip-lengths 150,150,150,150,150,150,150 \
  --clip-ids 6,7,8,9,10,11,12 \
  --seam-radius 2 --bin-s 1 \
  --keyframes /tmp/slam/P0002_273c2819/KeyFrameTrajectory.txt \
  --out /tmp/slam/records/P0002_273c2819_timeline.json
```

亮度和模糊要在 euroc png 还在时加上 `--image-dir .../euroc/mav0/cam0/data`。轨迹、png 和二进制不入库。

序列本身：

```text
PYTHONPATH=scripts python3 -m headcam.run_quest3_stereo \
  --clips /tmp/hot3d/clips/clip-000006 /tmp/hot3d/clips/clip-000007 \
          /tmp/hot3d/clips/clip-000008 /tmp/hot3d/clips/clip-000009 \
          /tmp/hot3d/clips/clip-000010 /tmp/hot3d/clips/clip-000011 \
          /tmp/hot3d/clips/clip-000012 \
  --work /tmp/slam/P0002_273c2819 \
  --orbslam-bin /tmp/src/ORB_SLAM3/Examples/Stereo/stereo_euroc \
  --vocab /tmp/src/ORB_SLAM3/Vocabulary/ORBvoc.txt \
  --orbslam-root /tmp/src/ORB_SLAM3 \
  --name P0002_273c2819 --participant P0002
```

另外五条是同一条命令，clip 目录换成上面表里的编号。

### 回归

`TrajectoryTimelineTest`：合成轨迹上接缝尖峰留在接缝；速度没有起伏时 Pearson 是 null；逐帧 RMSE 和 `trajectory_ate` 的 SE3 一致；估计文件头写成头戴真值时 status 是 failed、rmse 是 null。开头 4 帧加 0.5 m 的固定平移时，尾巴从 t = 0 开始，帧间平移 RMSE 小于 1e-8，尾巴内外的线速度中位都是 1 m/s，尾巴外的位置 RMSE 小于尾巴里的。

系统解释器 `python3 -m unittest tests.test_headcam_mvp tests.test_stereo_qc`：69 项，跳过 2 项（两个官方加载器测试，没装 lerobot），其余通过。numpy 1.26.4，pyarrow 18.1.0，pandas 2.2.3。没有开 Kaggle。没有下 MANO。没有把人手放进 SLAM 世界系。

### 还缺的

601 条训练集连续段里，跑过的是每个参与者最长的那条，再加上 P0010 的第二条 1800 帧。P0003 和 P0021 没有在这台机器上重跑。test split 的 3150 帧和 2100 帧没有下。Quest3 tar 里没有 IMU，没有 stereo-inertial，没有自采头戴。逐帧特征数日志里没有。P0002 开头那段固定平移的原因没有钉死。亮度和模糊没有把开头拿掉再算。四段那个 LeRobot 包仍是 `pose_source=hot3d_gt_mocap`。没有 WiLoR 权重。

手要进 SLAM 世界系，卡在 MANO 许可。这是 Justin 的决定，权重没有下。这次的轨迹误差没有停在账号或机器上。


## 同一台机器上的 15 条

`origin/main` 仍是 `476b9e7`，已经在这条分支里。这次没有再合并。

P0003_b153ade2（480–491）和 P0021_5d8b0988（1221–1228）在这台机器上按同一份配置重跑了。配置没改：`Stereo.T_c1_c2` 是 OpenCV 外参的逆，`nFeatures` 1500，去畸变再转 -90° 的 1024×1280 针孔，二进制是上游 `4452a3c`。查看器和帧间 sleep 只在本机 `/tmp` 里改过。没有 IMU。

P0002_273c2819 的表行换成这台机器上一次已经测过的重跑（80 个关键帧，SE3 RMSE 0.014536769894140768 m）。上一台机器的三份原件在 `P0002_273c2819_prevhost.json`、`P0003_b153ade2_prevhost.json`、`P0021_5d8b0988_prevhost.json`。上一节里那条 summarize 命令现在读到的是换过的文件；上一台机器的数在这三份 prevhost 里。

新的七条是每位参与者还没进表的最长连续训练段，由 `slam_batch.next_unused_runs` 从 `contiguous_runs` 里挑出来。P0018 的这段和已经入表的 1125–1129 是同一个 `sequence_id`，记录名叫 `P0018_15ad9743_c1121`。合成数据上的选择在 `SlamBatchSummaryTest.test_next_unused_run_is_the_second_longest_per_participant`。

15 条都有测到的 ATE。失败 0，未测 0，RMSE 汇总是这 15 条。`P0002_45904c71` 的轨迹有重复时间戳，`lost_frames` 按「输入帧数减不同时间戳」是 25，accuracy 仍然有值，留在表里。

| 序列 | clips | 帧 | 跟踪 | 关键帧 | 局部地图失败 | SE3 RMSE | SE3 中位 | SE3 最大 | Sim3 RMSE | 尺度 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P0003_b153ade2 | 480–491 | 1800 | 1800 | 180 | 0 | 0.034938410199040267 m | 0.03130053034361041 m | 0.053364086668572314 m | 0.033468149532695377 m | 0.9407163993842738 |
| P0021_5d8b0988 | 1221–1228 | 1200 | 1200 | 25 | 0 | 0.002077283831836545 m | 0.0014290889134091398 m | 0.005070125688359554 m | 0.000898350932502304 m | 1.0112071854116933 |
| P0002_273c2819 | 6–12 | 1050 | 1050 | 80 | 0 | 0.014536769894140768 m | 0.0040423255435268765 m | 0.06606663538241843 m | 0.014520470758477963 m | 1.0111402616823868 |
| P0010_3224a967 | 706–717 | 1800 | 1800 | 83 | 1 | 0.008205109394453782 m | 0.0020800494546338866 m | 0.08389409291523889 m | 0.008100251379623766 m | 1.0080545963907157 |
| P0010_5edabbe9 | 747–758 | 1800 | 1800 | 88 | 0 | 0.04942053479288229 m | 0.018706983087599 m | 0.15811514024780318 m | 0.0474363595017323 m | 0.9183096427504743 |
| P0013_0ec32d10 | 846–852 | 1050 | 1050 | 57 | 0 | 0.006196191926232237 m | 0.0047783261252145924 m | 0.013933751896470524 m | 0.0021213276661974896 m | 1.0301082162274657 |
| P0017_44db2c1d | 1061–1066 | 900 | 900 | 41 | 0 | 0.007054496104415027 m | 0.003974965173258681 m | 0.03713993532543655 m | 0.007045156718925644 m | 1.0023946088947122 |
| P0018_15ad9743 | 1125–1129 | 750 | 750 | 107 | 0 | 0.004247753259978912 m | 0.0020693655177331195 m | 0.017867154336932808 m | 0.0038702017042583432 m | 1.0250505738252662 |
| P0002_45904c71 | 68–73 | 900 | 875 | 91 | 27 | 0.027915580877991106 m | 0.01896957330455942 m | 0.05678426959613544 m | 0.02600061728855521 m | 1.0786898121797996 |
| P0003_46b683a2 | 258–266 | 1350 | 1350 | 53 | 0 | 0.0019564760870109414 m | 0.0013573732166447063 m | 0.006834883975367377 m | 0.001719328492950887 m | 1.006326385870652 |
| P0010_086ff32a | 648–657 | 1500 | 1500 | 69 | 0 | 0.003269682361286761 m | 0.0011249178226932033 m | 0.03534755101592558 m | 0.003020135408092692 m | 1.009058717076233 |
| P0013_a2b36677 | 961–966 | 900 | 900 | 19 | 0 | 0.003780502220618381 m | 0.0023469759601996154 m | 0.021079866273816156 m | 0.0019658686792309718 m | 1.0316941005605043 |
| P0017_1a07e9e7 | 1036–1039 | 600 | 600 | 9 | 0 | 0.0008185619223686899 m | 0.0006448887028987194 m | 0.00405390449561427 m | 0.0008148215552086108 m | 1.0009715900342202 |
| P0018_15ad9743_c1121 | 1121–1124 | 600 | 600 | 106 | 0 | 0.003080392875958846 m | 0.0025172974758072373 m | 0.011144374967580596 m | 0.0024354863459798964 m | 1.0209913406291327 |
| P0021_8f2f3894 | 1255–1261 | 1050 | 1050 | 18 | 0 | 0.004552055280436843 m | 0.003943796765189701 m | 0.012591377779989378 m | 0.0015453583273660441 m | 1.016750941532254 |

`slam_batch.summarize`：SE3 RMSE 最小 0.0008185619223686899 m，中位 0.004552055280436843 m，最大 0.04942053479288229 m。15 条时中位是排序后下标 `len//2` 的那个，也就是第 8 个，是 `P0021_8f2f3894`。Sim3 RMSE 最小 0.0008148215552086108 m，中位 0.003020135408092692 m，最大 0.0474363595017323 m。RPE 1 s 平移 RMSE 中位 0.0046249488252417025 m，最大 0.021701871293961936 m。RPE 5 s 平移 RMSE 中位 0.00810188211907968 m，最大 0.05195165630965154 m。

头戴路程仍是 `gt_headset_tum.txt` 相邻相机中心的长度。P0003 这台机器重算是 8.179914918155205 m，P0021 是 4.315110908077026 m，和上一台机器写下的路程相同。新的七条写在 `headset_path.json`。

### P0003 在这台机器上

上一台机器的 P0003 是 SE3 RMSE 0.0034270419514742423 m、157 个关键帧。这台机器是 0.034938410199040267 m、180 个关键帧，1800/1800，局部地图失败 0，回环 0，地图 1。Sim3 尺度 0.9407163993842738，RMSE 0.033468149532695377 m。1 秒 RPE 平移 RMSE 0.00957224245017761 m（1771 对），中位 0.0026636872268652016 m。

时间线：中位已经是 0.03130053034361041 m，尾巴门限 `max(0.02, 3×中位)` 是 0.09390159103083123 m，没有一帧到这个门限，所以 `tail_runs` 是空的。最差 10%（180 帧）只占误差平方和的 0.19338232065630634。第一秒 RMSE 0.028459201275034866 m。最差的一秒从 t = 34 s 起，RMSE 0.04997181196923633 m。最大 0.053364086668572314 m 在 t = 34.166666666998935 s，clip 486，clip 内第 125 帧。整段 1799 步的平移 RMSE 是 0.001817144902473394 m，旋转 RMSE 0.08433025879716792°。接缝半径 2 帧的 RMSE 0.033607351081445413 m，远离接缝 0.03498807298792872 m。误差和头戴线速度的 Pearson 是 0.19425520345682684。

钉得住的是：这次的几厘米残差铺在整段 60 秒上，帧间运动仍贴着头戴真值，一次带尺度的对齐也没把它收掉。上一台机器那次的轨迹不在这里，两次数不能对到同一帧。假设：两次非确定性的建图落在不同的结果上。原因没有钉死。

P0021 两次接近。这台机器 SE3 RMSE 0.002077283831836545 m、25 个关键帧；上一台机器 0.0021233505018789477 m、27 个关键帧。

### 次长的七条

九条新跑的时间线里，没有一条尾巴从 t = 0 开始。P0002 / `P0010_5edabbe9` 那种开头固定平移，这次没有再出现。

`P0002_45904c71` 是这七条里差的一条，也是唯一没有跟踪满的。`CameraTrajectory.txt` 900 行、875 个不同时间戳、25 行是重复时间戳。重复落在相对起点 16.66666666700621 s（2 行）、16.733333334006602 s（14 行）、17.200000000004366 s（4 行）、17.333333334005147 s（7 行）、17.56666666700039 s（2 行）、17.63333333400078 s（2 行）。头戴真值里有 16 个时间戳在 0.05 秒内找不到估计，都在 t = 16.80000000000291 s 到 17.5 s：11 帧在 16.80000000000291–17.13333333400078 s，1 帧在 17.26666666699748 s，4 帧在 17.400000000001455–17.5 s。日志有 `Fail to track local map!` 27 次，`Relocalized!!` 5 次，`Less than 15 matches` 4 次，都没有时间戳。地图 1，重建 0。SE3 RMSE 0.027915580877991106 m，中位 0.01896957330455942 m，最大 0.05678426959613544 m 在 t = 16.66666666700621 s。最差的一秒从 t = 12 s 起，RMSE 0.05020883235758875 m。第一秒 RMSE 0.0177283027929863 m。门限 0.056908719913678255 m，没有尾巴。874 步平移 RMSE 0.002703870252434238 m。Sim3 尺度 1.0786898121797996，RMSE 0.02600061728855521 m。第一次 `trajectory_qc` 带上 6×150 的 clip 长度，因为 900 对不上 875 个时间戳而退出。现在这份时间线没有传 clip 长度，所以没有接缝对照。假设：日志里的重定位和 t = 16.8–17.5 s 的空档是同一段。证据只到这两件事都在，日志没有时间。

`P0010_086ff32a` 开头是好的，第一秒 RMSE 0.001669897216364096 m。三段短尾巴在 t = 38.700000000004366–38.833333333001065 s（5 帧，均值长度 0.0264695880904903 m，内部平移 RMSE 0.008472528476430858 m）、t = 38.93333333299961–38.966666665997764 s（2 帧）、t = 39.599999999998545 s（1 帧）。最差 10% 占误差平方和的 0.8784529380738049。接缝附近 RMSE 0.003419177324678025 m，远离接缝 0.003263966939392707 m。这是后面不到一秒的鼓包，内部平移误差有 8.5 mm，和 P0002 开头那种 1.7 秒、帧间 2.9 mm 的固定平移不是同一份证据。

`P0021_8f2f3894` 的 Sim3 尺度 1.016750941532254，SE3 RMSE 0.004552055280436843 m，Sim3 RMSE 0.0015453583273660441 m。`P0013_a2b36677` 的尺度 1.0316941005605043，SE3 0.003780502220618381 m，Sim3 0.0019658686792309718 m。这两条上，放开尺度会把 RMSE 降下来。`P0003_46b683a2` 全程跟踪，SE3 RMSE 0.0019564760870109414 m。`P0017_1a07e9e7` 是这 15 条里最小的，0.0008185619223686899 m，9 个关键帧。

### 重算

```text
PYTHONPATH=scripts python3 -m headcam.slam_batch summarize \
  --records docs/headcam/orbslam3_quest3/P0003_b153ade2.json \
            docs/headcam/orbslam3_quest3/P0021_5d8b0988.json \
            docs/headcam/orbslam3_quest3/P0002_273c2819.json \
            docs/headcam/orbslam3_quest3/P0010_3224a967.json \
            docs/headcam/orbslam3_quest3/P0010_5edabbe9.json \
            docs/headcam/orbslam3_quest3/P0013_0ec32d10.json \
            docs/headcam/orbslam3_quest3/P0017_44db2c1d.json \
            docs/headcam/orbslam3_quest3/P0018_15ad9743.json \
            docs/headcam/orbslam3_quest3/P0002_45904c71.json \
            docs/headcam/orbslam3_quest3/P0003_46b683a2.json \
            docs/headcam/orbslam3_quest3/P0010_086ff32a.json \
            docs/headcam/orbslam3_quest3/P0013_a2b36677.json \
            docs/headcam/orbslam3_quest3/P0017_1a07e9e7.json \
            docs/headcam/orbslam3_quest3/P0018_15ad9743_c1121.json \
            docs/headcam/orbslam3_quest3/P0021_8f2f3894.json \
  --json docs/headcam/orbslam3_quest3_batch.json \
  --md docs/headcam/orbslam3_quest3_batch.md
```

一条录制：

```text
PYTHONPATH=scripts python3 -m headcam.run_quest3_stereo \
  --clips /tmp/hot3d/clips/clip-000480 /tmp/hot3d/clips/clip-000481 \
          /tmp/hot3d/clips/clip-000482 /tmp/hot3d/clips/clip-000483 \
          /tmp/hot3d/clips/clip-000484 /tmp/hot3d/clips/clip-000485 \
          /tmp/hot3d/clips/clip-000486 /tmp/hot3d/clips/clip-000487 \
          /tmp/hot3d/clips/clip-000488 /tmp/hot3d/clips/clip-000489 \
          /tmp/hot3d/clips/clip-000490 /tmp/hot3d/clips/clip-000491 \
  --work /tmp/slam/P0003_b153ade2 \
  --orbslam-bin /tmp/src/ORB_SLAM3/Examples/Stereo/stereo_euroc \
  --vocab /tmp/src/ORB_SLAM3/Vocabulary/ORBvoc.txt \
  --orbslam-root /tmp/src/ORB_SLAM3 \
  --name P0003_b153ade2 --participant P0003
```

其余几条是同一条命令，clip 目录换成表里的编号。`P0018_15ad9743_c1121` 的 `--name` 要用这个名字，`--participant P0018`。

`P0002_45904c71` 的时间线：

```text
PYTHONPATH=scripts python3 -m headcam.trajectory_qc \
  --estimate /tmp/slam/P0002_45904c71/CameraTrajectory.txt \
  --gt /tmp/slam/P0002_45904c71/gt_headset_tum.txt \
  --estimate-scale 1e-9 \
  --bin-s 1 \
  --keyframes /tmp/slam/P0002_45904c71/KeyFrameTrajectory.txt \
  --out docs/headcam/orbslam3_quest3/P0002_45904c71_timeline.json
```

系统解释器 `python3 -m unittest tests.test_headcam_mvp tests.test_stereo_qc`：70 项，跳过 2 项，通过。没有 Kaggle，没有 MANO，没有 stereo-inertial，没有把人手放进 SLAM 世界系。

### 还缺的

601 条训练段里，表上是每个参与者最长的一条、P0010 的另一条 1800 帧，再加上这次每位的次长一条，一共 15 条。test split 没下。没有 IMU，没有逐帧特征数。P0003 这次整段几厘米残差的原因没有钉死。P0002 开头那段固定平移的原因也还没有钉死。四段 LeRobot 包仍是 `pose_source=hot3d_gt_mocap`。手要进 SLAM 世界系，卡在 MANO 许可，这是 Justin 的决定。

## 用 batch 里的实测误差导出

`scripts/headcam/batch_export.py`。缺测手的包可以改从外部 TUM 来，`pose_source=orbslam3_stereo_euroc`。`accuracy` 只取 batch 里那一行的 `trajectory_error_v1`。给了真值、并且这次重算的 SE3 RMSE 和那一行一致（差不超过 1e-6 m）时，写进行里的那一份，不另造一个数。行里是 null、这次真值对上了至少 3 个时间戳，就写这次测到的结构，报告里写明 batch 那一行是 null。两份 RMSE 不一致、行里是手写数字、或对不上 3 个时间戳，包里的 `accuracy` 保持 null，报告写原因。汇总的 min / median / max 不进包。

下面这些情况直接拒绝，不写 `out`：估计轨迹每一帧都是单位阵；文件头不是 `orbslam3_stereo_euroc`（头戴真值改个文件头也不行）；估计轨迹和头戴真值逐帧相同；会话帧数、轨迹行数或不同时间戳个数和这一行对不上。第一帧在原点、后面的帧在动，仍然接受。丢失帧的 `observation.pose` 是 null，不补单位阵。不跑手。

```text
python3 scripts/headcam/export_mvp.py \
  --batch docs/headcam/orbslam3_quest3_batch.json \
  --sequence P0010_3224a967 \
  --session /path/to/session \
  --tum /path/to/CameraTrajectory_seconds.txt \
  --gt-tum /path/to/gt_headset_tum.txt \
  --out /path/to/out \
  --skip-hands
```

纳秒轨迹还要 `--tum-time-scale 1e-9`。不传尺度时用这一行的 `estimate_time_scale`。`--sequence` 对不上、或者没加 `--skip-hands`，退出码是 1。

已经打进四段包的 clip 和这 15 行的 `clip_ids` 对不上，所以没有把表上的 ATE 套上去：

| 包里的 clip | 序列 | batch 里同名的 clip | 结果 |
| --- | --- | --- | --- |
| clip-000000 | `P0002_210fc0da` | 无 | 不匹配 |
| clip-000300 | `P0003_5e1a34f8` | 无 | 不匹配 |
| clip-000700 | `P0010_3224a967` | 706–717 | 同名，clip 不是同一段。700 的末帧到 706 空了 32.167 秒 |
| clip-001100 | `P0017_d2495f49` | 无 | 不匹配 |

`P0010_3224a967` 706–717 这一行的 SE3 RMSE 仍是表上的 0.008205109394453782 m。这台机器上没有 `CameraTrajectory.txt`、`gt_headset_tum.txt`，也没有 Quest3 图像。`docs/headcam/orbslam3_quest3/P0010_3224a967_timeline.json` 写明当时的 euroc png 已经删了。没有重下 clip，没有重跑 ORB-SLAM3，没有编一份位姿。15 条都没有在这台机器上重导出。

`BatchAccuracyExportTest` 用合成刚体轨迹：单位阵和逐帧抄来的头戴真值会被拒绝；对得上的 batch 行把同一个 `rmse_m` 和 `rotation_row_major` 写进 `report.json` 和 `egodata_export.json`，不写汇总的 median，也不写进 `info.json`。把 RMSE 改成 0.5 再配上真值，包里的 accuracy 是 null。

系统解释器 `python3 -m unittest tests.test_headcam_mvp tests.test_stereo_qc`：74 项，跳过 3 项（两个官方加载器，没装 lerobot 或 torch；`test_rectified_pair_has_no_vertical_error`，没装 cv2），其余通过。numpy 1.26.4，pyarrow 18.1.0，pandas 2.2.3。没有 Kaggle。没有下 MANO。没有把人手放进 SLAM 世界系。

### 还缺的

四段那个包仍是 `pose_source=hot3d_gt_mocap`，因为没有和它们的 clip 对应的实测轨迹。15 条里有实测 ATE 的，要等外部 TUM 和会话还在时才能按上面的命令导出。P0003 这次整段几厘米残差、P0002 开头那段固定平移，这次没有再查。手进 SLAM 世界系仍等 Justin 的 MANO 许可。

## P0017 的缺测手包

下了 Hugging Face `bop-benchmark/hot3d` 的 `train_quest3` clip-001036 到 clip-001039，每段 150 帧，序列 `P0017_1a07e9e7`。`write_quest3_euroc`：600 帧，30 fps，基线 0.06368154174105198 m，外参漂移 1.1450259460238877e-07 m，左右同步 0 ms，内参表各重建 1 次。和 batch 行的基线一致。图像、tar、去畸变 png 和轨迹没有入库。

ORB-SLAM3 是上游 `4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4` 的 `stereo_euroc`。OpenCV 4.6.0，Eigen 3，Pangolin v0.6。查看器关掉，帧间 sleep 去掉，只在本机 `/tmp`。没有 IMU。日志：局部地图失败 0，回环 0，地图 1，关键帧 9，`New Map created with 639 points`。`CameraTrajectory.txt` 600 行、600 个时间戳、丢失 0。第一帧在原点。

这次相对头戴真值的 SE3 RMSE 是 0.0008689981649285166 m（n=600）。batch 行是 0.0008185619223686899 m。差的绝对值是 5.0436242559826774e-05 m，大于 1e-6 m。没有把 batch 那一行的 RMSE 写进这份新轨迹。

导出：

```text
PYTHONPATH=scripts python3 scripts/headcam/export_mvp.py \
  --batch docs/headcam/orbslam3_quest3_batch.json \
  --sequence P0017_1a07e9e7 \
  --session /tmp/slam/P0017_1a07e9e7/session \
  --tum /tmp/slam/P0017_1a07e9e7/CameraTrajectory_labeled.txt \
  --gt-tum /tmp/slam/P0017_1a07e9e7/gt_headset_tum.txt \
  --out /tmp/slam/P0017_1a07e9e7/export \
  --skip-hands
```

`CameraTrajectory_labeled.txt` 只是在原始轨迹前加了 `# pose_source=orbslam3_stereo_euroc`。位姿没有改。退出码 0。包在 `/tmp/slam/P0017_1a07e9e7/export/lerobot`：1 段，600 帧，`ok=600`，`hands=absent`。`info.json` 和 `egodata_export.json` 的 `pose_source` 是 `orbslam3_stereo_euroc`。`info.json` 没有 accuracy。`report.json`、`slam_status.json`、`egodata_export.json` 的 accuracy 都是 null。`accuracy_source=unmeasured`。报告写的是：batch 的 SE3 RMSE 是 0.0008185619223686899，这次测到 0.0008689981649285166，两份不一致。parquet 里 600 行 `observation.pose` 都不是 null。第 0 帧平移是 0，最后一帧平移是 `[0.01006262, 0.047735944, -0.014272315]`，和 `CameraTrajectory.txt` 最后一行一致。这台机器没有 lerobot venv，没有用官方加载器逐帧读。

数在 `docs/headcam/orbslam3_quest3/P0017_1a07e9e7_export_rerun.json`。batch 那 15 行没有改。

`python3 -m unittest tests.test_headcam_mvp tests.test_stereo_qc`：74 项，跳过 2 项（两个官方加载器，没装 lerobot 或 torch），其余通过。numpy 1.26.4，pyarrow 18.1.0，pandas 2.2.3，cv2 4.6.0。没有 Kaggle。没有下 MANO。没有把人手放进 SLAM 世界系。

### 还缺的

这份包的 accuracy 是 null，因为重跑的 RMSE 和 batch 行对不上 1e-6 m。四段那个包仍是 `pose_source=hot3d_gt_mocap`。P0003 整段残差和 P0002 开头固定平移没有再查。手进 SLAM 世界系仍等 Justin 的 MANO 许可。

## P0018 clips 1121–1124 的缺测手包

下了 Hugging Face `bop-benchmark/hot3d` 的 `train_quest3` clip-001121 到 clip-001124，每段 150 帧。`write_quest3_euroc`：序列 `P0018_15ad9743`，记录名 `P0018_15ad9743_c1121`，600 帧，30 fps，基线 0.0639632910011052 m，外参漂移 2.1169715027180658e-07 m，左右同步 0 ms，内参表各重建 1 次。基线和外参漂移与 batch 里 `P0018_15ad9743_c1121` 那一行相同。这不是同序列已经入表的 1125–1129（750 帧）。图像、tar、去畸变 png 和轨迹没有入库。15 行的表没有改。

ORB-SLAM3 是上游 `4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4` 的 `stereo_euroc`。OpenCV 4.6.0，Pangolin v0.6。查看器关掉，帧间 sleep 去掉，只在本机 `/tmp`。没有 IMU。日志：局部地图失败 0，回环 0，地图 1，关键帧 110，`New Map created with 232 points`。batch 行是 106 个关键帧。`CameraTrajectory.txt` 600 行、600 个时间戳、丢失 0。第一帧在原点。

这次相对头戴真值的 SE3 RMSE 是 0.0027235435865570787 m（n=600，中位 0.002246254571217467 m，最大 0.008812988185761003 m）。Sim3 RMSE 是 0.0020056904218865978 m，尺度 1.0204944836544005。batch 行的 SE3 RMSE 是 0.003080392875958846 m。差的绝对值是 0.00035684928940176754 m，大于 1e-6 m。没有把 batch 那一行的 RMSE 写进这份新轨迹。

导出：

```text
PYTHONPATH=scripts python3 scripts/headcam/export_mvp.py \
  --batch docs/headcam/orbslam3_quest3_batch.json \
  --sequence P0018_15ad9743_c1121 \
  --session /tmp/slam/P0018_15ad9743_c1121/session \
  --tum /tmp/slam/P0018_15ad9743_c1121/CameraTrajectory_labeled.txt \
  --gt-tum /tmp/slam/P0018_15ad9743_c1121/gt_headset_tum.txt \
  --out /tmp/slam/P0018_15ad9743_c1121/export \
  --skip-hands
```

`CameraTrajectory_labeled.txt` 只是在原始轨迹前加了 `# pose_source=orbslam3_stereo_euroc`。位姿没有改。退出码 0。包在 `/tmp/slam/P0018_15ad9743_c1121/export/lerobot`：1 段，600 帧，`ok=600`，`hands=absent`。`info.json` 和 `egodata_export.json` 的 `pose_source` 是 `orbslam3_stereo_euroc`。`info.json` 没有 accuracy。`report.json`、`slam_status.json`、`egodata_export.json` 的 accuracy 都是 null。`accuracy_source=unmeasured`。报告写的是：batch 的 SE3 RMSE 是 0.003080392875958846，这次测到 0.0027235435865570787。parquet 里 600 行 `observation.pose` 都不是 null。第 0 帧平移是 0，最后一帧平移是 `[-0.142737329, -0.111938842, 0.079168409]`，和 `CameraTrajectory.txt` 最后一行一致。这台机器没有 lerobot venv，没有用官方加载器逐帧读。

数在 `docs/headcam/orbslam3_quest3/P0018_15ad9743_c1121_export_rerun.json`。batch 那 15 行没有改。

`python3 -m unittest tests.test_headcam_mvp tests.test_stereo_qc`：74 项，跳过 2 项（两个官方加载器，没装 lerobot 或 torch），其余通过。numpy 1.26.4，pyarrow 18.1.0，pandas 2.1.4，cv2 4.6.0。没有 Kaggle。没有下 MANO。没有把人手放进 SLAM 世界系。

### 还缺的

这份包的 accuracy 是 null，因为重跑的 RMSE 和 batch 行对不上 1e-6 m。四段那个包仍是 `pose_source=hot3d_gt_mocap`。P0003 整段残差和 P0002 开头固定平移没有再查。手进 SLAM 世界系仍等 Justin 的 MANO 许可。

## 四段 ORB-SLAM3 缺测手包

上面 clip-000000、clip-000300、clip-000700、clip-001100 的头戴真值四段包仍是 `pose_source=hot3d_gt_mocap`。那四段的 clip 不在 `docs/headcam/orbslam3_quest3_batch.json` 的 `clip_ids` 里，这次没有把表上的 ATE 套上去，也没有改那 15 行。

这次另下了表里已经有轨迹误差的四段 `train_quest3`，在这台机器上重跑 ORB-SLAM3，再打成一个缺测手的多 episode 包。图像、tar、去畸变 png、轨迹和 LeRobot 目录都在 `/tmp`，没有入库。

| 记录名 | clip | 帧 |
| --- | --- | --- |
| `P0017_1a07e9e7` | 1036–1039 | 600 |
| `P0018_15ad9743_c1121` | 1121–1124 | 600 |
| `P0018_15ad9743` | 1125–1129 | 750 |
| `P0017_44db2c1d` | 1061–1066 | 900 |

`P0018_15ad9743` 和 `P0018_15ad9743_c1121` 的 `sequence_id` 都是 `P0018_15ad9743`。导出按唯一记录名取行：750 帧对 1125–1129，600 帧对 1121–1124。

ORB-SLAM3 是上游 `4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4` 的 `stereo_euroc`。OpenCV 4.6.0，Pangolin v0.6。查看器关掉，帧间 sleep 去掉，只在本机 `/tmp`。二进制不入库。没有 IMU。四段都是 `stereo_euroc` 退出码 0，局部地图失败 0，回环 0，地图 1，丢失 0。`CameraTrajectory.txt` 的行数等于输入帧数。每一段第一帧平移是 0。日志里的新地图点数：`P0017_1a07e9e7` 639，`P0018_15ad9743_c1121` 232，`P0018_15ad9743` 450，`P0017_44db2c1d` 619。

每段都是 30 fps、1024×1280，左右同步 0 ms，左右内参表各重建 1 次。基线相对 batch 行的绝对差：`P0017_1a07e9e7` 1.3877787807814457e-17 m（这次 0.06368154174105199，表 0.06368154174105198），`P0018_15ad9743_c1121` 0（两边都是 0.0639632910011052），`P0018_15ad9743` 2.7755575615628914e-17 m（这次 0.06396328071855298，表 0.063963280718553），`P0017_44db2c1d` 1.3877787807814457e-17 m（这次 0.06368154288115634，表 0.06368154288115635）。

这次相对头戴真值的 SE3 RMSE 和 batch 行都对不上 1e-6 m。包和每一段的 accuracy 都是 null，`accuracy_source=unmeasured`。没有把表上的 RMSE 写进这四份新轨迹。

| 记录名 | 这次 SE3 RMSE (m) | batch SE3 RMSE (m) | 绝对差 (m) | 关键帧（这次 / 表） | 末帧平移 |
| --- | --- | --- | --- | --- | --- |
| `P0017_1a07e9e7` | 0.0008098764209253006 | 0.0008185619223686899 | 8.685501443389228e-06 | 8 / 9 | `[0.009366003, 0.046851031, -0.014788805]` |
| `P0018_15ad9743_c1121` | 0.0032966843202244865 | 0.003080392875958846 | 0.0002162914442656403 | 110 / 106 | `[-0.146049410, -0.110140949, 0.077572860]` |
| `P0018_15ad9743` | 0.004994147404644316 | 0.004247753259978912 | 0.0007463941446654047 | 112 / 107 | `[-0.001062672, 0.143115312, 0.035223052]` |
| `P0017_44db2c1d` | 0.013931679274417118 | 0.007054496104415027 | 0.006877183170002091 | 38 / 41 | `[0.024094917, 0.265315384, 0.278289735]` |

中位、最大和 Sim3：`P0017_1a07e9e7` 中位 0.0006712533178188623、最大 0.003930738608523243、Sim3 RMSE 0.0008061087074877067、尺度 1.0009699252634265。`P0018_15ad9743_c1121` 中位 0.0027770546810274785、最大 0.010167575268578477、Sim3 RMSE 0.0028177055877934977、尺度 1.0190120153981526。`P0018_15ad9743` 中位 0.0023643101810740487、最大 0.021643403644813625、Sim3 RMSE 0.004527301932967464、尺度 1.0303398297209463。`P0017_44db2c1d` 中位 0.012513345813771865、最大 0.038500098126321235、Sim3 RMSE 0.013890129226933728、尺度 1.0071508594326033。n 分别是 600、600、750、900。

单段导出走 `export_from_batch`，轨迹文件头是 `# pose_source=orbslam3_stereo_euroc`，位姿没有改。四段都是 `hands=skipped`，`world_pose=true`，`ok` 等于帧数，`tracking_lost=0`，`pose_missing=0`。`P0017_1a07e9e7` 和 `P0018_15ad9743_c1121` 上面两节的单段重跑是另一次轨迹，RMSE 和末帧平移以这一节和 `quest3_four_episode_orbslam3_export.json` 为准。

四段 episode 再交给 `package_unmeasured_pose_episodes`（命令行是 `--package-pose-only`）。退出码 0。日志：

```text
package=pose_only episodes=4 accepted=4 rejected=0 pose_source=orbslam3_stereo_euroc ok=2850 tracking_lost=0 pose_missing=0 accuracy=null
```

包在 `/tmp/slam/multi_pose/lerobot`，`repo_id=local/headcam_mvp_quest3_orbslam3`。`meta/info.json`：`pose_source=orbslam3_stereo_euroc`，`robot_type=headcam_pose_only`，`total_episodes=4`，`total_frames=2850`，`fps=30`，没有 accuracy。`meta/egodata_export.json`：`hands=absent`，`pose_source=orbslam3_stereo_euroc`，`accuracy=null`，`ok_frames=2850`，`frames=2850`。parquet 2850 行，`slam_status` 全是 `ok`，`observation.pose` 没有 null。平移为 0 的行正好 4 行，各是一段的第 0 帧。按 `episode_index`：0 是 `P0017_1a07e9e7`（600），1 是 `P0017_44db2c1d`（900），2 是 `P0018_15ad9743`（750），3 是 `P0018_15ad9743_c1121`（600）。四段末帧平移和各自的 `CameraTrajectory.txt` 最后一行一致。这台机器没有 lerobot venv，没有用官方加载器逐帧读。

数在 `docs/headcam/orbslam3_quest3/quest3_four_episode_orbslam3_export.json`。

`python3 -m unittest tests.test_headcam_mvp tests.test_stereo_qc`：77 项，跳过 2 项（两个官方加载器，没装 lerobot 或 torch），其余通过。numpy 1.26.4，pyarrow 18.1.0，pandas 2.2.3，cv2 4.6.0。没有 Kaggle。没有下 MANO。没有把人手放进 SLAM 世界系。

### 还缺的

这个四段包的 accuracy 仍是 null，因为四段这次的 SE3 RMSE 和 batch 行的绝对差都大于 1e-6 m。clip-000000 / 000300 / 000700 / 001100 那个头戴真值包没有被这次的轨迹替换。P0003 整段残差和 P0002 开头固定平移没有再查。手进 SLAM 世界系仍等 Justin 的 MANO 许可。

## 四段短 clip 的 ORB-SLAM3 缺测手包

上一节的 2850 帧包留在 `/tmp/slam/multi_pose/lerobot`，这次没有覆盖，也没有改 `docs/headcam/orbslam3_quest3_batch.json`。clip-000000、clip-000300、clip-000700、clip-001100 仍不在那 15 行的 `clip_ids` 里。这一节另下了这四段 Hugging Face `bop-benchmark/hot3d` 的 `train_quest3`，各 150 帧，在这台机器上跑 `stereo_euroc`，再打成另一个缺测手的多 episode 包。图像、tar、去畸变 png、轨迹和 LeRobot 目录都在 `/tmp`，没有入库。

| clip | 序列 | 参与者 | 帧 |
| --- | --- | --- | --- |
| clip-000000 | `P0002_210fc0da` | P0002 | 150 |
| clip-000300 | `P0003_5e1a34f8` | P0003 | 150 |
| clip-000700 | `P0010_3224a967` | P0010 | 150 |
| clip-001100 | `P0017_d2495f49` | P0017 | 150 |

`write_quest3_euroc`：都是 30 fps、1024×1280，左右同步 0 ms，左右内参表各重建 1 次。基线：clip-000000 是 0.06368567581770954 m，clip-000300 是 0.06361127711681205 m，clip-000700 是 0.06361127219638592 m，clip-001100 是 0.06368155119263902 m。外参漂移：1.3407708431633166e-07 m、1.766177832163487e-07 m、1.4517020204469397e-07 m、1.804308712059654e-07 m。没有 IMU。

ORB-SLAM3 是上游 `4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4` 的 `stereo_euroc`。OpenCV 4.6.0，Eigen 3.4.0，Pangolin v0.6（`dd801d244db3a8e27b7fe8020cd751404aa818fd`）。查看器关掉，帧间 sleep 去掉，这两处只在本机 `/tmp` 的 `Examples/Stereo/stereo_euroc.cc`。二进制不入库。本机 gcc 13 默认按 C++17 编 `Thirdparty/g2o` 时，clip-000000 在日志 `New Map created with 551 points` 之后退出码 -6，`double free or corruption (out)`，落在 `g2o::HyperGraph::clear` / `Optimizer::PoseOptimization`，没有写出 `CameraTrajectory.txt`。把同一份 g2o 用 `-std=c++11` 重编，和 ORB-SLAM3 的 `-std=c++11` 对齐之后，四段都是退出码 0。下面的轨迹和 ATE 都来自这次重编。

四段都是局部地图失败 0，回环 0，地图 1，丢失 0，`CameraTrajectory.txt` 150 行、150 个时间戳。第一帧平移是 0。日志里的新地图点数：clip-000000 是 551，clip-000300 是 656，clip-000700 是 543，clip-001100 是 486。关键帧：17、8、10、21。

这次相对头戴真值的 Umeyama 写进了每一段自己的 `report.json` 和 episode。来源是 `pose_source=orbslam3_stereo_euroc`，真值文件头是 `hot3d_gt_mocap`。`accuracy_source=measured_this_run`。文档上一节 clip-000000 的 SE3 RMSE 0.006408 m 留在那一节，没有写入这四份轨迹。

| clip | SE3 RMSE (m) | SE3 中位 (m) | SE3 最大 (m) | Sim3 RMSE (m) | Sim3 尺度 | 末帧平移 |
| --- | --- | --- | --- | --- | --- | --- |
| clip-000000 | 0.0045912349693376705 | 0.004226443939589596 | 0.014209958267527703 | 0.002830270810618332 | 1.0282527167188087 | `[-0.094589248, -0.245565057, 0.168445095]` |
| clip-000300 | 0.0017128617629292439 | 0.0012812311134126073 | 0.003413361805056417 | 0.0005832235613042844 | 1.0140845282400313 | `[-0.191976130, 0.164640561, -0.126090199]` |
| clip-000700 | 0.0007797332618919564 | 0.0005260818839636823 | 0.003662961842560437 | 0.0007601198771456078 | 1.0031196366517134 | `[-0.017725043, 0.030247804, 0.187362045]` |
| clip-001100 | 0.02475762699101022 | 0.023573684693554423 | 0.039992538061840374 | 0.02424834786772622 | 0.9544006686528443 | `[-0.033097558, -0.079486132, -0.122066431]` |

n 都是 150。1 秒 RPE 都配上 121 对：平移 RMSE 分别是 0.0079132006277709 m、0.002753304207956389 m、0.0018040903972870541 m、0.030710074375809204 m；旋转 RMSE 分别是 0.38270844589435427°、0.4704201773495364°、0.29795402912763896°、1.0720475347259262°。5 秒 RPE 每段只配上 1 对，主数仍是尺度固定的 SE3。`summarize_sequences` 对这四段 SE3 RMSE 的汇总是 n=4，min 0.0007797332618919564 m，median 0.0045912349693376705 m（排序后取 `n//2`），max 0.02475762699101022 m。这个汇总没有写进 LeRobot 包。

单段导出走 `export_mvp --skip-hands`，轨迹文件头是 `# pose_source=orbslam3_stereo_euroc`，位姿没有改，`--tum-time-scale 1e-9`，`--gt-tum` 是这次写出的头戴真值。四段都是 `hands=skipped`，`world_pose=true`，`ok=150`，`tracking_lost=0`，`pose_missing=0`。没有走 `--batch`，因为这四段不在 15 行里。

四段 episode 再交给 `--package-pose-only`。退出码 0。日志：

```text
package=pose_only episodes=4 accepted=4 rejected=0 pose_source=orbslam3_stereo_euroc ok=600 tracking_lost=0 pose_missing=0 accuracy=null
```

包级 accuracy 是 null，因为四段测到的 `trajectory_error_v1` 不是同一份，`shared_measured_accuracy` 返回 null。每一段 episode 里仍留着上面这次测到的 Umeyama。包在 `/tmp/slam/short_four/pack/lerobot`，`repo_id=local/headcam_mvp_quest3_short_orbslam3`。`meta/info.json`：`pose_source=orbslam3_stereo_euroc`，`robot_type=headcam_pose_only`，`total_episodes=4`，`total_frames=600`，`fps=30`，没有 accuracy。`meta/egodata_export.json`：`hands=absent`，`pose_source=orbslam3_stereo_euroc`，`accuracy=null`，`ok_frames=600`，`frames=600`。parquet 600 行，`slam_status` 全是 `ok`，`observation.pose` 没有 null。平移为 0 的行正好 4 行，各是一段的第 0 帧。按 `episode_index`：0 是 clip-000000，1 是 clip-000300，2 是 clip-000700，3 是 clip-001100。四段末帧平移和各自的 `CameraTrajectory.txt` 最后一行一致。这台机器没有 lerobot venv，没有用官方加载器逐帧读。

数在 `docs/headcam/orbslam3_quest3/quest3_four_short_clips_orbslam3_export.json`。

`python3 -m unittest tests.test_headcam_mvp tests.test_stereo_qc`：77 项，跳过 2 项（两个官方加载器，没装 lerobot 或 torch），其余通过。numpy 1.26.4，pyarrow 18.1.0，pandas 2.1.4，cv2 4.6.0。没有 Kaggle。没有下 MANO。没有把人手放进 SLAM 世界系。

### 还缺的

这个短 clip 包的包级 accuracy 是 null，因为四段这次的 Umeyama 不是同一份测量。2850 帧那个包没有改。15 行的 batch 没有加这四段。P0003 整段残差和 P0002 开头固定平移没有再查。手进 SLAM 世界系仍等 Justin 的 MANO 许可。

## P0002 开头固定平移、P0003 整段残差

这一节只读已经提交的时间线和 batch 行。这台机器没有 ORB-SLAM3 二进制，也没有 HOT3D 的 tar，没有重跑 `stereo_euroc`，没有新的 Umeyama。`accuracy` 仍是 null。短 clip 包的包级 accuracy 规则没有改：各段测量对象不相同的时候，包级 accuracy 保持 null。

`scripts/headcam/residual_diagnosis.py` 把时间线归成两种形状。门限写在模块里，合成轨迹锁在 `ResidualShapeDiagnosisTest`。时间线、表行、重跑记录的 SE3 RMSE 必须是同一个数（差超过 1e-12 m 就拒绝，不从里面挑一个）。1 秒箱的 `pooled_rmse_m` 是各箱 `n * rmse^2` 的加总，不是重新对齐。

短 clip 包对不上这两条录制，所以没有抄它们的 RMSE：

| 短 clip | 序列 | clip | 这次诊断用的录制 | clip |
| --- | --- | --- | --- | --- |
| clip-000000 | `P0002_210fc0da` | 0 | `P0002_273c2819` | 6–12 |
| clip-000300 | `P0003_5e1a34f8` | 300 | `P0003_b153ade2` | 480–491 |

`P0002_45904c71` 和 `P0003_46b683a2` 是另外的 sequence_id，这份诊断不读它们。

重算：

```text
PYTHONPATH=scripts python3 -m headcam.residual_diagnosis \
  --docs docs/headcam/orbslam3_quest3 \
  --out docs/headcam/orbslam3_quest3/residual_shape_diagnosis.json
```

### P0002_273c2819：开头固定平移

`P0002_273c2819_timeline.json`、`P0002_273c2819.json`、`P0002_273c2819_rerun.json` 的 SE3 RMSE 都是 0.014536769894140768 m。归类是 `opening_fixed_translation`。

抄自时间线的数：一条尾巴，t = 0 到 1.7000000000043656 s，52 帧，RMSE 0.061654164438708026 m，最大 0.06606663538241843 m。最大在第 39 帧，t = 1.3000000000029104 s，clip 6，clip 内第 39 帧。误差向量均值的长度 0.06062457775745827 m，分量标准差 0.006478080074763555 m，比值 9.35841747211978。尾巴内部 51 步的平移 RMSE 0.002916849944043936 m，是尾巴位置 RMSE 的 0.047309860908806095。尾巴外 998 帧的位置 RMSE 0.004926238619860379 m，是尾巴的 0.07990114965806856。

1 秒箱：在尾巴结束前开始的箱共 60 帧，含跨过 1.7 s 的那一箱，平方和占比 0.8941310398242173，pooled RMSE 0.057502605856651304 m。从尾巴结束起的箱共 990 帧，pooled RMSE 0.0048711251976878405 m。完全落在尾巴里的是第一秒，30 帧，RMSE 0.06434934008061646 m。

头戴线速度的最大在尾巴外：尾巴里 0.11823643444934767 m/s，外面 0.28569690017074567 m/s。接缝附近的最大 0.008857777451921423 m，小于全程最大。丢失 0，局部地图失败 0，回环 0，关键帧 80。

上一台机器的 `P0002_273c2819_prevhost.json` 是 SE3 RMSE 0.015275329451131102 m、79 个关键帧。绝对差 0.0007385595569903338 m，大于 1e-6 m。仓库里没有那次的时间线，开头是不是也在 t = 1.3 s，写的是 unknown。0.015275329451131102 m 没有当成这次的 RMSE。

亮度只有整段 Pearson。时间线没有逐帧亮度，开头 1.7 秒没有单独拿掉再算。`tracked_features` 是 null。时间线没有逐帧误差向量，不能把那 6.06 cm 的均值减掉再算一次 RMSE。日志字段里没有带时间戳的重定位。

### P0003_b153ade2：整段残差

`P0003_b153ade2_timeline.json` 和 `P0003_b153ade2.json` 的 SE3 RMSE 都是 0.034938410199040267 m，1800 帧。没有单独的 rerun 文件；当前这行就是这台机器的那一次。归类是 `whole_sequence_residual`。

`tail_runs` 是空的。门限 `max(0.02, 3×中位)` 是 0.09390159103083123 m，中位已经是 0.03130053034361041 m。最差 10%（180 帧）占误差平方和的 0.19338232065630634。1799 步的平移 RMSE 是 0.001817144902473394 m，是位置 RMSE 的 0.05200994813791812。1 秒箱最小 0.02024805329022599 m（全程的 0.5795356221097373），最大 0.04997181196923633 m（全程的 1.4302829374477095）。最大在 t = 34.166666666998935 s，clip 486，clip 内第 125 帧。接缝附近 RMSE 0.033607351081445413 m，远离接缝 0.03498807298792872 m。

表上的 Sim3 RMSE 是 0.033468149532695377 m，尺度 0.9407163993842738，相对 SE3 的比值 0.9579185012148813。放开尺度没有把这段残差收掉。丢失 0，局部地图失败 0，回环 0，重建 0，关键帧 180。

上一台机器的 `P0003_b153ade2_prevhost.json` 是 SE3 RMSE 0.0034270419514742423 m、157 个关键帧。绝对差 0.03151136824756602 m。两份都是丢失 0、局部地图失败 0。仓库里没有上一台机器的时间线，那一次是不是同一种整段残差，写的是 unknown。0.0034270419514742423 m 没有写进这次的时间线。

两次完整跟踪的 SE3 RMSE 差了一个数量级。原因没有钉死。假设仍是两次建图落在不同结果上。证据只到这两份已提交的 RMSE、关键帧数，以及这次时间线的形状。

### 回归

`ResidualShapeDiagnosisTest`：合成的开头固定平移归成 `opening_fixed_translation`；尾巴不从 0 开始、前半段明显更安静、或者 Sim3 把 RMSE 收到 SE3 的 0.8 以下，都不归成这两种。表行 RMSE 和手写的 0.5 对不上，或者文件头是 `hot3d_gt_mocap`，结果是 `refused`，`proven` 是 null。丢失帧留在 `unclassified`，RMSE 仍是时间线里的数。短 clip 的 0.0045912349693376705 和重叠 clip 上的手写 0.123456 都不出现在报告里。已提交的两份时间线锁住上面的数。报告里的 `accuracy` 是 null，没有 `trajectory_error_v1`。

系统解释器 `python3 -m unittest tests.test_headcam_mvp tests.test_stereo_qc`：86 项，跳过 2 项（两个官方加载器，没装 lerobot 或 torch），其余通过。numpy 1.26.4，pyarrow 18.1.0，pandas 2.2.3，cv2 4.6.0。没有 Kaggle。没有下 MANO。没有把人手放进 SLAM 世界系。没有新的 ATE。

### 还缺的

形状归类没有给出原因。上一台机器的逐帧形状不在仓库里。P0002 的亮度没有把开头拿掉。时间线里没有逐帧误差向量，也没有逐帧特征数。Quest3 这几份记录里没有 IMU。四段短 clip 包和 2850 帧那个包的包级 accuracy 仍是 null。手进 SLAM 世界系仍等 Justin 的 MANO 许可。

## 15 条已提交时间线的残差形状

上一节只归了 `P0002_273c2819` 和 `P0003_b153ade2`。这一节用同一个脚本读 `docs/headcam/orbslam3_quest3/` 里全部 15 份 `*_timeline.json`。每份都有同名的 batch 行。时间线和表行的 SE3 RMSE 差是 0，没有超过 1e-12 m，所以没有拒绝。没有新的 Umeyama，没有新的 ATE。报告里的 `accuracy` 仍是 null。各段测量不相同的时候，包级 accuracy 保持 null，这里没有改这条规则，也没有把 15 个 RMSE 收成一个中位数。

短 clip 的 RMSE 没有抄进报告。`P0010_3224a967` 和 clip-000700 的 `sequence_id` 相同，`copied_rmse_m` 仍是 null。`P0017_1a07e9e7_export_rerun.json` 的这次 SE3 RMSE 是 0.0008689981649285166 m，batch 行是 0.0008185619223686899 m；`P0018_15ad9743_c1121_export_rerun.json` 的这次是 0.0027235435865570787 m，batch 行是 0.003080392875958846 m。这两份导出重跑都标了 `used_as_this_measurement=false`，时间线用的仍是和 batch 行相同的那个 RMSE。

`pattern_counts`：`opening_fixed_translation` 2，`whole_sequence_residual` 1，`unclassified` 12，`refused` 0。未归类是现有门限没通过，不是把它们硬写成两种形状之一。

| 时间线 | 归类 | SE3 RMSE (m) | 已测到、因而没归进那两种形状的点 |
| --- | --- | --- | --- |
| `P0002_273c2819` | `opening_fixed_translation` | 0.014536769894140768 | 上一节 |
| `P0010_5edabbe9` | `opening_fixed_translation` | 0.04942053479288229 | 下面这一段 |
| `P0003_b153ade2` | `whole_sequence_residual` | 0.034938410199040267 | 上一节 |
| `P0002_45904c71` | `unclassified` | 0.027915580877991106 | 丢失 25 帧。没有尾巴。1 秒箱相对全程是 0.49333048422588566 到 1.7985952926085749 |
| `P0003_46b683a2` | `unclassified` | 0.0019564760870109414 | 没有尾巴。帧间平移 RMSE 0.0006870864155126594 m，是位置 RMSE 的 0.35118569558515483。1 秒箱相对全程是 0.16292549148619498 到 2.0221723134273555 |
| `P0010_086ff32a` | `unclassified` | 0.003269682361286761 | 三段尾巴都在 t = 38.700000000004366 s 之后，不是从 0 开始的一段。最差 10% 占平方和 0.8784529380738049 |
| `P0010_3224a967` | `unclassified` | 0.008205109394453782 | 尾巴在 t = 40.83333333399787 s（21 帧，RMSE 0.06690665923535118 m）和 t = 50.83333333399787 s（10 帧）。最差 10% 占平方和 0.9408732420825507 |
| `P0013_0ec32d10` | `unclassified` | 0.006196191926232237 | Sim3 RMSE 0.0021213276661974896 m，相对 SE3 的比值 0.3423599028972333，尺度把残差收到 0.8 以下。帧间 / 位置是 0.18920722998839093 |
| `P0013_a2b36677` | `unclassified` | 0.003780502220618381 | 唯一的尾巴是 t = 14.73333333300252 s 的 1 帧。Sim3 相对 SE3 是 0.5200019903465134 |
| `P0017_1a07e9e7` | `unclassified` | 0.0008185619223686899 | 没有尾巴。帧间平移 RMSE 0.0006955262794541821 m，是位置 RMSE 的 0.8496929315274317。1 秒箱相对全程是 0.5896002828475024 到 1.8146678082432335，尺度没有收掉（Sim3 / SE3 = 0.9954305629691942） |
| `P0017_44db2c1d` | `unclassified` | 0.007054496104415027 | 尾巴在 t = 29.0 s（27 帧）和 t = 29.93333333299961 s（2 帧），不是开头一段。1 秒箱相对全程最大 4.12887176930324 |
| `P0018_15ad9743` | `unclassified` | 0.004247753259978912 | 没有尾巴。最差 10% 占平方和 0.7608861505448176。帧间 / 位置是 0.23428472640805612。1 秒箱相对全程最大 3.8142694907043335 |
| `P0018_15ad9743_c1121` | `unclassified` | 0.003080392875958846 | Sim3 RMSE 0.0024354863459798964 m，相对 SE3 是 0.7906414681672035。帧间 / 位置是 0.44542421183130615 |
| `P0021_5d8b0988` | `unclassified` | 0.002077283831836545 | Sim3 RMSE 0.000898350932502304 m，相对 SE3 是 0.43246422021590764。帧间 / 位置是 0.3642359366544009 |
| `P0021_8f2f3894` | `unclassified` | 0.004552055280436843 | Sim3 RMSE 0.0015453583273660441 m，相对 SE3 是 0.3394858436819647。帧间 / 位置是 0.2383542628781757 |

### P0010_5edabbe9：另一条开头固定平移

`P0010_5edabbe9_timeline.json` 和 `P0010_5edabbe9.json` 的 SE3 RMSE 都是 0.04942053479288229 m，1800 帧。归类是 `opening_fixed_translation`。没有单独的 rerun，也没有 prevhost。

一条尾巴，t = 0 到 5.866666666996025 s，177 帧，RMSE 0.1483826203615348 m，最大 0.15811514024780318 m。最大在第 170 帧，t = 5.666666666998935 s，clip 748，clip 内第 20 帧。误差向量均值的长度 0.14796632492630718 m，分量标准差 0.006412714276026166 m，比值 23.073899531042105。尾巴内部 176 步的平移 RMSE 0.003801115231638534 m，是尾巴位置 RMSE 的 0.02561698413451052。尾巴外 1623 帧的位置 RMSE 0.017538277969787148 m，是尾巴的 0.11819630848313009。

1 秒箱：在尾巴结束前开始的箱共 180 帧，含跨过尾巴结束的那一箱，平方和占比 0.8869489098670089，pooled RMSE 0.14718269228856976 m。从尾巴结束起的箱共 1620 帧，pooled RMSE 0.017515541298556904 m。完全落在尾巴里的箱共 150 帧，pooled RMSE 0.14892263641594128 m。

头戴线速度的最大在尾巴外：尾巴里 0.555320686855585 m/s，外面 0.6961325536796875 m/s。表上的 Sim3 RMSE 是 0.0474363595017323 m，尺度 0.9183096427504743。丢失 0，局部地图失败 0，回环 0，重建 0，关键帧 88。

仓库里没有逐帧 XYZ 误差向量。0.14796632492630718 m 的均值因此没有减掉再算一次 RMSE。`mean_subtract.rmse_after_m` 是 null。

### 有向量时可以减开头均值

`diagnose_files` 在对齐之后把 `error_vectors_m` 写进时间线。这台机器没有 ORB-SLAM3 二进制，也没有 HOT3D tar，没有重跑，所以 15 份已提交的时间线仍没有这个字段。`opening_tail_mean_subtract` 只在调用方已经给出 N×3 向量时计算。它从 t = 0 的那一段尾巴减掉这段自己的均值平移，再算全程 RMSE。这不是新的 Umeyama。`accuracy` 仍是 null。

合成夹具，不读已提交文件：开头 10 帧是 `[0.06, 0, 0]`，其余 90 帧是 0。减掉开头均值之后全程 RMSE 小于 1e-12。整段减去自身均值不会塌掉，因为开头只占 10 帧。另一条合成是整段绕圈、每帧长度 0.03 m，时间线归成 `whole_sequence_residual`。没有从 t = 0 开始的单段尾巴，所以 `status` 是 `no_opening_tail`，`rmse_after_m` 是 null。就算只减前 10 帧的均值，全程 RMSE 仍大于减之前的 0.8。整段减去自身均值之后 RMSE 仍大于 0.025 m。标量列表 `[0.06] * 100` 是 `direction_unknown`，不编方向。向量 RMSE 和时间线 `rmse_m` 差超过 1e-12 m 时是 `vector_rmse_mismatch`，不改写时间线上的 RMSE。

15 份已提交时间线的 `mean_subtract.status` 都是 `absent_from_json`。P0002 那个 0.06062457775745827 m 的均值还不能减掉再算。

重算：

```text
PYTHONPATH=scripts python3 -m headcam.residual_diagnosis \
  --docs docs/headcam/orbslam3_quest3 \
  --out docs/headcam/orbslam3_quest3/residual_shape_diagnosis.json
```

### 回归

`ResidualShapeDiagnosisTest` 锁住上面 15 条的归类和 P0010 的数。合成开头固定平移在减掉尾巴均值后塌掉；合成整段残差不塌。标量误差和跟时间线对不上的向量都不产生减完之后的 RMSE。短 clip 的四个 RMSE 不出现在报告里。`accuracy` 是 null。

系统解释器 `python3 -m unittest tests.test_headcam_mvp tests.test_stereo_qc`：89 项，跳过 12 项（这台机器没装 pyarrow、cv2、lerobot、torch；其中两项是官方加载器），其余通过。numpy 1.26.4。没有 Kaggle。没有下 MANO。没有把人手放进 SLAM 世界系。没有新的 ATE。

### 还缺的

形状归类没有给出原因。上一台机器的时间线不在仓库里，P0002、P0003、P0021 的 prevhost 形状仍是 unknown。亮度没有把开头拿掉：`P0002_273c2819`、`P0010_3224a967`、`P0010_5edabbe9`、`P0013_0ec32d10`、`P0017_44db2c1d`、`P0018_15ad9743` 只有整段 Pearson（`aggregate_only`），其余是 `unmeasured`。euroc png 不在这台机器上。15 份已提交时间线没有逐帧 XYZ 向量，减掉均值平移之后的 RMSE 是 unknown，也没有逐帧特征数。这些记录里没有 IMU。四段短 clip 包和 2850 帧那个包的包级 accuracy 仍是 null。手进 SLAM 世界系仍等 Justin 的 MANO 许可。没有下 MANO，没有 Kaggle，没有新的 ATE。

## 一轮增量的验收门

上面每一轮结束时，都用同一句话收束：没有新的 ATE，包级 accuracy 在测量不相同的时候保持 null，没有下 MANO。从这一节起，这句话有一道机器可跑的门。清单在 `docs/headcam/verification_gate.md`。

合格增量是下面这条命令退出码为 0，并且第一行是 `result: PASS`：

```bash
PYTHONPATH=scripts python3 -m headcam.verification_gate
```

它做五件事。包级和 episode 的 `accuracy` 在 remount（这次重跑）与 batch 行相差超过 1e-6 m，或各段测量不是同一份时，必须是 null。时间线与 batch 的 SE3 RMSE 差超过 1e-12 m 时，这条是 `refused`，并且不用作这次测量。声称残差归类时，`residual_shape_diagnosis.json` 必须和 `diagnose_repo` 对已提交时间线的重算一致，报告里的 `accuracy` 仍是 null。没有 `error_vectors_m` 时，减均值之后的 RMSE 保持 unknown，不从标量编 XYZ。`tests.test_headcam_mvp`、`tests.test_stereo_qc`、`tests.test_headcam_verification_gate` 要通过；缺可选依赖的 skip 算通过。

`held:` 里仍然要问 Justin 的是：MANO 许可、人手进 SLAM 世界系、自动合并、没有二进制或参考数据时新算 Umeyama、没有逐帧向量时编 XYZ。这道门不合并，不下载 MANO，不跑 ORB-SLAM3，也不新造 RMSE。PASS 只说明这轮声明和已提交文件没有越过这些线。

这轮若另外写出包级 accuracy 或 remount 相对 batch 的数，用 `--claim` 交一份 `headcam_mvp_increment_v1`。不交声明时，门核对已提交的 15 份时间线和残差报告。`--skip-tests` 和 `--no-repo` 只用于调试，不能代替合格命令。

系统解释器 `python3 -m unittest tests.test_headcam_mvp tests.test_stereo_qc tests.test_headcam_verification_gate`：105 项，跳过 12 项（这台机器没装 pyarrow、cv2、lerobot、torch；其中两项是官方加载器），其余通过。numpy 1.26.4。没有 Kaggle。没有下 MANO。没有新的 ATE。

