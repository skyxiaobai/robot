#!/usr/bin/env python
"""可视化"模型驾驶"PushT 模拟器:加载训练好的 ACT checkpoint,实时推理控制。

用法:
    python launch_simulator_act.py
    python launch_simulator_act.py --checkpoint outputs/checkpoints/010000/pretrained_model
    python launch_simulator_act.py --root /path/to/repo --checkpoint .../last/pretrained_model

窗口操作:q / ESC 退出,r 立即重开一局。
"""
import argparse
import os

import cv2
import gym_pusht  # 注册 gym_pusht/PushT-v0
import gymnasium as gym
import torch

from lerobot.envs.utils import preprocess_observation
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTPolicy
from repo_root import add_root_argument

# 导入即完成 gym 环境注册；保留引用以便静态检查看到这次导入。
_GYM_PUSHT = gym_pusht.__name__


def main():
    parser = argparse.ArgumentParser(description="模型驾驶 PushT 模拟器")
    add_root_argument(parser)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="ACT checkpoint（默认 <root>/outputs/checkpoints/010000/pretrained_model）",
    )
    args = parser.parse_args()
    root = os.path.abspath(args.root)
    checkpoint = args.checkpoint or os.path.join(
        root, "outputs", "checkpoints", "010000", "pretrained_model"
    )

    policy = ACTPolicy.from_pretrained(checkpoint)
    policy.to("cuda")
    policy.eval()
    device = next(policy.parameters()).device
    print(f"loaded ACT policy from {checkpoint} (device={device})")

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=checkpoint
    )

    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        max_episode_steps=300,
    )
    obs, _ = env.reset()

    ep = 1
    while True:
        obs_t = preprocess_observation(obs)
        obs_t = preprocessor(obs_t)
        with torch.inference_mode():
            action = policy.select_action(obs_t)
        action = postprocessor(action)
        action = action.squeeze(0).cpu().numpy()

        obs, reward, terminated, truncated, _ = env.step(action)

        frame = env.render()
        cv2.imshow("PushT Simulator (ACT policy)", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        key = cv2.waitKey(16) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            obs, _ = env.reset()
            policy.reset()
            ep = 1
            continue

        if terminated or truncated:
            print(f"episode {ep} ended (reward={reward:.2f})")
            ep += 1
            obs, _ = env.reset()
            policy.reset()

    env.close()
    cv2.destroyAllWindows()
    print("simulator closed")


if __name__ == "__main__":
    main()
