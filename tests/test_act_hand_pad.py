# -*- coding: utf-8 -*-
"""lerobot 0.6.1 预处理只保留带 _is_pad 的附加键；按手掩码必须以 action_hand_is_pad 传到 ACT。"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

import ego_act_scaling as eas  # noqa: E402


@unittest.skipIf(torch is None, "需要 torch")
class HandPadKeyTest(unittest.TestCase):
    def test_item_carries_hand_pad_that_survives_is_pad_filter(self):
        table = np.array([[1, 1], [1, 0], [0, 0], [1, 1]], dtype=np.float32)
        eas._cached_action_valid.cache = {"root": table}

        class D:
            root = "root"

        item = {"action": torch.zeros(3, 14), "index": torch.tensor(0), "action_is_pad": torch.zeros(3, dtype=torch.bool)}
        out = eas._attach_action_valid_item(D(), item)
        self.assertIn(eas.HAND_PAD_KEY, out)
        self.assertIn("_is_pad", eas.HAND_PAD_KEY)
        self.assertEqual(out[eas.HAND_PAD_KEY].tolist(), [[False, False], [False, True], [True, True]])
        self.assertEqual(out["action_is_pad"].tolist(), [False, False, True])


if __name__ == "__main__":
    unittest.main()
