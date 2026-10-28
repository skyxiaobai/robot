# Headcam MVP 一轮增量的验收门

这是 headcam 轨道上每一轮自主改动共用的尺子。先跑检查，通过之后才把这轮叫做合格。通过不等于合并，也不产生新的轨迹误差数字。

## 合格命令

在仓库根目录：

```bash
PYTHONPATH=scripts python3 -m headcam.verification_gate
```

stdout 第一行是 `result: PASS`，退出码是 0，这轮才算合格。`fail:` 下面每一行是一条原因，原因码见文末。`held:` 是仍然要停下来问 Justin 的边界，通过时也打印。

`--skip-tests` 不跑单测。`--no-repo` 不核对已提交的残差报告。这两条都只用于调试，不能代替合格命令。

只跑单测、不印这道门的结论时：

```bash
python3 -m unittest tests.test_headcam_mvp tests.test_stereo_qc tests.test_headcam_verification_gate
```

缺 pyarrow、cv2、lerobot、torch 之类可选依赖时，对应测试 skip。skip 算通过。失败和错误算不合格。

这轮如果还要声明包级 accuracy、remount（这次重跑）相对 batch 的 SE3，或减均值的结果，把声明写成 `headcam_mvp_increment_v1`，加上 `--claim 文件`。不交声明时，门核对已提交的残差报告、15 份时间线和单测。

## 合格增量

下面五条都成立，才叫合格增量。

1. 包级和每一段 episode 的 `accuracy` 只接受 null，或一份真实的 `trajectory_error_v1`。remount 的 SE3 RMSE 和 batch 行相差超过 `1e-6` m（`batch_export.RMSE_MATCH_ATOL_M`）时，这段和包级都保持 null。各段测量不是同一份时，包级保持 null。手写数字保持 null。
2. 时间线 SE3 RMSE 和 batch 行相差不超过 `1e-12` m（`residual_diagnosis.RMSE_IDENTITY_ABS_M`），才可以把这条当作这次测量。超过时 `pattern` 是 `refused`，并且 `used_as_this_measurement` 是 false。
3. 声称残差归类时，`docs/headcam/orbslam3_quest3/residual_shape_diagnosis.json` 在，并且和 `diagnose_repo` 对这批已提交时间线的重算结果一致。报告的 `accuracy` 是 null，`new_ate` 是 false。短 clip 的 RMSE 不抄进报告。
4. 时间线没有 `error_vectors_m` 时，`mean_subtract.status` 是 `absent_from_json`，`rmse_after_m` 是 null。不从标量 `error_m` 编出逐帧 XYZ。向量和时间线 RMSE 差超过 `1e-12` m 时，不改写时间线上的 RMSE。
5. 上面的单测通过。skip 只来自本机缺的可选依赖。

门不重算 ATE。PASS 只说明声明和已提交文件没有违反这些规则，不说明新跑出了一条轨迹。

## 停下来问 Justin

这些事项门会失败，并在 `held:` 里保留：

- MANO：没有 Justin 的明确许可，不下、不用 MANO 权重。仓库里跟踪了 MANO 的 pkl、npz、zip 也失败。
- 人手放进 SLAM 世界系：和 MANO 是同一件事。
- 自动合并：这道门不合并，也不打开 auto-merge。
- 新的 Umeyama：ORB-SLAM3 二进制或参考数据不在时，不能新算。
- 没有 `error_vectors_m` 时编造逐帧 XYZ。

硬件、账号、不可逆操作、真正的产品取舍，同样停下来问 Justin。门不代替这四类决定。

## 声明里要写明的字段

`kind` 是 `headcam_mvp_increment_v1`。布尔字段必须出现：`mano_fetch`、`hands_in_slam_world`、`auto_merge`、`new_umeyama`、`orbslam3_binary_present`、`reference_data_present`、`invented_xyz`、`claims_classification`。

还要有 `package_accuracy`（null 或测量结构）、`episodes`、`timelines`、`mean_subtract`（null 表示这轮不声称减过均值）。

每一段 episode 可以带 `published_accuracy`、`rerun_se3_rmse_m`、`batch_se3_rmse_m`。每一条时间线可以带 `timeline_se3_rmse_m`、`batch_se3_rmse_m`、`pattern`、`used_as_this_measurement`。`mean_subtract` 为对象时，`status` 和 `rmse_after_m` 必须和 `opening_tail_mean_subtract` 那条路径算出来的一致。没有向量时 `rmse_after_m` 是 null。

## 原因码

```text
claim_not_object
claim_kind
claim_unreadable
package_accuracy_missing
episode_accuracy_missing
episodes_missing
timelines_missing
episode_not_object
timeline_not_object
mean_subtract_unanswered
mean_subtract_not_object
stop_unanswered
stop_mano
stop_hands_in_slam_world
stop_auto_merge
stop_umeyama_without_inputs
stop_invented_xyz
accuracy_not_measured
rmse_not_number
episode_accuracy_invented
accuracy_not_this_measurement
package_accuracy_invented
timeline_used_unanswered
timeline_pattern
rmse_missing_used
rmse_missing_not_refused
rmse_mismatch_not_refused
mean_subtract_unknown
mean_subtract_status
mean_subtract_rmse_invented
residual_report_missing
residual_report_unreadable
residual_report_inconsistent
diagnosis_accuracy_not_null
diagnosis_new_ate
diagnosis_package_rule
diagnosis_case_accuracy_not_null
residual_cases_not_committed_timelines
mean_subtract_inconsistent
checklist_missing
tests_failed
```
