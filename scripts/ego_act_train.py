#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""调用 lerobot 的训练入口，并在 ACT 损失里按 action_valid 按手屏蔽。

参数与 ``lerobot-train`` 相同。找不到 lerobot 时改去执行原来的命令，此时
没有按手掩码，只会打印警告。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ego_act_scaling  # noqa: E402


def _lerobot_main():
    try:
        from lerobot.scripts.lerobot_train import main
        return main
    except ImportError:
        pass
    try:
        from lerobot.scripts.train import main
        return main
    except ImportError:
        return None


def main():
    installed = ego_act_scaling.install_act_hand_mask()
    train_main = _lerobot_main()
    if train_main is None:
        command = ego_act_scaling._lerobot_train_command()
        if not installed:
            sys.stderr.write("未安装 lerobot，改为执行 %s\n" % command)
        os_exec = __import__("os").execvp
        os_exec(command, [command, *sys.argv[1:]])
    return train_main()


if __name__ == "__main__":
    raise SystemExit(main())
