#!/usr/bin/env python
"""可视化启动 LeRobot 模拟器:PushT 或 ALOHA 插孔。

用法:
    python launch_simulator.py --env pusht
    python launch_simulator.py --env aloha

窗口操作:
    q / ESC  退出
    r        立即重开一局
默认随机动作演示;把 action 换成模型推理结果即可变成"模型驾驶"。
"""
import argparse

import cv2
import gym_aloha  # 导入即注册 gym_aloha/AlohaInsertion-v0
import gym_pusht  # 导入即注册 gym_pusht/PushT-v0
import gymnasium as gym

_ = (gym_aloha, gym_pusht)


def main():
    parser = argparse.ArgumentParser(description="可视化模拟器")
    parser.add_argument("--env", choices=["pusht", "aloha"], default="pusht")
    parser.add_argument("--max-episodes", type=int, default=0, help="0=无限运行")
    args = parser.parse_args()

    if args.env == "pusht":
        env = gym.make(
            "gym_pusht/PushT-v0",
            obs_type="pixels_agent_pos",
            render_mode="rgb_array",
            max_episode_steps=300,
        )
        window_name = "PushT Simulator (random actions)"
    else:
        env = gym.make(
            "gym_aloha/AlohaInsertion-v0",
            obs_type="pixels_agent_pos",
            render_mode="rgb_array",
            max_episode_steps=400,
        )
        window_name = "ALOHA Insertion Simulator (random actions)"

    obs, _ = env.reset()
    print(f"started {args.env}: obs keys={list(obs.keys())}")

    ep = 1
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)

        frame = env.render()
        cv2.imshow(window_name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        key = cv2.waitKey(16) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            obs, _ = env.reset()
            ep = 1
            continue

        if terminated or truncated:
            print(f"episode {ep} ended (reward={reward:.2f})")
            ep += 1
            obs, _ = env.reset()
            if args.max_episodes and ep > args.max_episodes:
                break

    env.close()
    cv2.destroyAllWindows()
    print("simulator closed")


if __name__ == "__main__":
    main()
