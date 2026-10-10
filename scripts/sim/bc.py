# -*- coding: utf-8 -*-
"""小多层感知机行为克隆。只依赖 numpy，CPU 上几秒能训完。

状态是末端、夹爪、方块和目标（以及相对向量）。动作是下一次控制周期要到达的
末端 xyz 和夹爪开合。这是状态 BC，不是图像 ACT：当前环境没有 GPU，也不装 torch。
"""
from dataclasses import dataclass

import numpy as np

from retarget.frames import GRIP_OPEN


def relative_features(obs):
    """用方块和目标做参照，不把桌子上的绝对坐标交给网络。

    这样换一个起点时，同样的「走到方块上方」仍然是同一类输入。
    """
    ee = np.asarray(obs["ee"], dtype=float)
    cube = np.asarray(obs["cube"], dtype=float)
    goal = np.asarray(obs["goal"], dtype=float)
    return np.concatenate(
        [ee - cube, ee - goal, [float(obs["grip"]), float(ee[2]), float(cube[2])]]
    ).astype(np.float64)


def encode_waypoint(position, grip, cube_xy, goal_xy):
    """把末端目标写成：沿起点→目标的比例 u、侧向偏移 v（米）、高度、夹爪。"""
    cube_xy = np.asarray(cube_xy, dtype=float)
    goal_xy = np.asarray(goal_xy, dtype=float)
    delta = goal_xy - cube_xy
    length = float(np.linalg.norm(delta))
    if length < 1e-8:
        along = np.array([1.0, 0.0])
        lateral = np.array([0.0, 1.0])
        u = 0.0
    else:
        along = delta / length
        lateral = np.array([-along[1], along[0]])
        offset = np.asarray(position[:2], dtype=float) - cube_xy
        u = float(np.dot(offset, along) / length)
    v = float(np.dot(np.asarray(position[:2], dtype=float) - cube_xy, lateral))
    return np.array([u, v, float(position[2]), float(grip)], dtype=float)


def decode_action(action, cube_xy, goal_xy):
    """网络输出还原成基座坐标里的末端目标和夹爪。超出演示范围的数会被夹住。"""
    cube_xy = np.asarray(cube_xy, dtype=float)
    goal_xy = np.asarray(goal_xy, dtype=float)
    delta = goal_xy - cube_xy
    length = float(np.linalg.norm(delta))
    if length < 1e-8:
        lateral = np.array([0.0, 1.0])
    else:
        along = delta / length
        lateral = np.array([-along[1], along[0]])
    u = float(np.clip(action[0], -0.2, 1.3))
    v = float(np.clip(action[1], -0.06, 0.06))
    z = float(np.clip(action[2], 0.23, 0.40))
    grip = float(np.clip(action[3], 0.0, GRIP_OPEN))
    xy = cube_xy + u * delta + v * lateral
    return np.array([xy[0], xy[1], z], dtype=float), grip


def featurize(ee, grip, cube, goal):
    ee = np.asarray(ee, dtype=float)
    cube = np.asarray(cube, dtype=float)
    goal = np.asarray(goal, dtype=float)
    return np.concatenate(
        [ee, [float(grip)], cube, goal, ee - cube, ee - goal]
    ).astype(np.float64)


def featurize_obs(obs):
    return featurize(obs["ee"], obs["grip"], obs["cube"], obs["goal"])


@dataclass
class TrainStats:
    train_mse_pose: float
    train_mse_grip: float
    epochs: int
    n_samples: int


class BCPolicy:
    """两层 ReLU 的回归网络。输入输出都按训练集做标准化。"""

    def __init__(self, hidden=128, seed=0, lr=1e-3, epochs=120, l2=1e-6):
        self.hidden = int(hidden)
        self.seed = int(seed)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.l2 = float(l2)
        self.ready = False

    def fit(self, features, actions):
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(actions, dtype=np.float64)
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError("features 和 actions 必须是同样行数的二维数组")
        self.x_mean = x.mean(axis=0)
        self.x_std = np.maximum(x.std(axis=0), 1e-6)
        self.y_mean = y.mean(axis=0)
        self.y_std = np.maximum(y.std(axis=0), 1e-6)
        xs = (x - self.x_mean) / self.x_std
        ys = (y - self.y_mean) / self.y_std
        rng = np.random.default_rng(self.seed)
        din = xs.shape[1]
        dout = ys.shape[1]
        h = self.hidden
        self.w1 = rng.normal(0.0, np.sqrt(2.0 / din), size=(din, h))
        self.b1 = np.zeros(h)
        self.w2 = rng.normal(0.0, np.sqrt(2.0 / h), size=(h, h))
        self.b2 = np.zeros(h)
        self.w3 = rng.normal(0.0, np.sqrt(2.0 / h), size=(h, dout))
        self.b3 = np.zeros(dout)
        m = {name: np.zeros_like(value) for name, value in self._params()}
        v = {name: np.zeros_like(value) for name, value in self._params()}
        step = 0
        order = np.arange(xs.shape[0])
        batch = min(256, xs.shape[0])
        for _epoch in range(self.epochs):
            rng.shuffle(order)
            for start in range(0, xs.shape[0], batch):
                step += 1
                idx = order[start:start + batch]
                self._adam(xs[idx], ys[idx], m, v, step)
        self.ready = True
        pred = self._predict_raw(x)
        pose = float(np.mean((pred[:, :-1] - y[:, :-1]) ** 2))
        grip = float(np.mean((pred[:, -1] - y[:, -1]) ** 2))
        return TrainStats(pose, grip, self.epochs, int(x.shape[0]))

    def act(self, features):
        """返回训练集原来的单位。不做工作空间裁剪，调用方自己解释动作。"""
        if not self.ready:
            raise RuntimeError("策略还没有训练")
        feat = np.asarray(features, dtype=float).reshape(1, -1)
        return self._predict_raw(feat)[0].copy()

    def _params(self):
        return [("w1", self.w1), ("b1", self.b1), ("w2", self.w2), ("b2", self.b2), ("w3", self.w3), ("b3", self.b3)]

    def _forward(self, xs):
        h1 = np.maximum(xs @ self.w1 + self.b1, 0.0)
        h2 = np.maximum(h1 @ self.w2 + self.b2, 0.0)
        out = h2 @ self.w3 + self.b3
        return h1, h2, out

    def _predict_raw(self, features):
        xs = (np.asarray(features, dtype=np.float64) - self.x_mean) / self.x_std
        _h1, _h2, out = self._forward(xs)
        return out * self.y_std + self.y_mean

    def _adam(self, xs, ys, m, v, step, beta1=0.9, beta2=0.999, eps=1e-8):
        h1, h2, out = self._forward(xs)
        # 标准化空间里的均方误差，加上很小的 L2。
        diff = (out - ys) / xs.shape[0]
        dw3 = h2.T @ diff + self.l2 * self.w3
        db3 = diff.sum(axis=0)
        dh2 = diff @ self.w3.T
        dh2 *= h2 > 0
        dw2 = h1.T @ dh2 + self.l2 * self.w2
        db2 = dh2.sum(axis=0)
        dh1 = dh2 @ self.w2.T
        dh1 *= h1 > 0
        dw1 = xs.T @ dh1 + self.l2 * self.w1
        db1 = dh1.sum(axis=0)
        grads = {"w1": dw1, "b1": db1, "w2": dw2, "b2": db2, "w3": dw3, "b3": db3}
        for name, value in self._params():
            g = grads[name]
            m[name] = beta1 * m[name] + (1 - beta1) * g
            v[name] = beta2 * v[name] + (1 - beta2) * (g * g)
            mhat = m[name] / (1 - beta1 ** step)
            vhat = v[name] / (1 - beta2 ** step)
            value -= self.lr * mhat / (np.sqrt(vhat) + eps)
