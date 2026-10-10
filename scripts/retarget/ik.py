# -*- coding: utf-8 -*-
"""阻尼最小二乘逆运动学的一步更新，带关节限位。

雅可比由调用方提供（仿真里用 MuJoCo 的 site 雅可比）。这里不依赖物理引擎，
方便单独测试限位。
"""
import numpy as np


def damped_least_squares(q, err, jacobian, lower, upper, damp=0.08, step=0.6):
    """沿末端误差走一步，并把新关节角夹进 ``[lower, upper]``。

    ``jacobian`` 形状是 (任务维, 关节数)，``err`` 是任务空间误差（希望减小的量）。
    返回 ``(q_next, dq)``。``dq`` 是裁切之前的步长。
    """
    current = np.asarray(q, dtype=float).reshape(-1)
    error = np.asarray(err, dtype=float).reshape(-1)
    matrix = np.asarray(jacobian, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != current.shape[0] or matrix.shape[0] != error.shape[0]:
        raise ValueError("雅可比形状应是 (len(err), len(q))")
    gram = matrix @ matrix.T + float(damp) * np.eye(matrix.shape[0])
    delta = matrix.T @ np.linalg.solve(gram, error)
    nxt = np.clip(current + float(step) * delta, lower, upper)
    return nxt, delta


def orientation_error(current_R, target_R):
    """从当前旋转到目标旋转的旋转向量（世界系）。"""
    relative = np.asarray(target_R, dtype=float) @ np.asarray(current_R, dtype=float).T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-8:
        return np.zeros(3, dtype=float)
    vee = np.array(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ],
        dtype=float,
    )
    return vee * (angle / (2.0 * np.sin(angle)))
