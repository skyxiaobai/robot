import sys, types, unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


class EgoverseSourceKey(unittest.TestCase):
    def test_egoverse_is_pretrain_key_and_loaded(self):
        import train_xhey as tx
        src = (ROOT / "scripts" / "train_xhey.py").read_text(encoding="utf-8")
        self.assertIn('"xphi", "egoverse")', src)
        self.assertIn('"--egoverse"', src)
        calls = []
        a = types.SimpleNamespace(out="/tmp/egoverse_test_out", egoverse="/nonexistent/egoverse")
        with mock.patch.object(tx, "try_external", side_effect=lambda k, r, c: calls.append((k, str(r))) or None):
            ext = tx.prepare_external(["iphone+egoverse"], {}, [], [], a)
        self.assertEqual(calls, [("egoverse", "/nonexistent/egoverse")])
        self.assertNotIn("egoverse", ext)

    def test_converter_native_rate_and_head_const_flag(self):
        conv = (ROOT / "scripts" / "convert_egoverse.py").read_text(encoding="utf-8")
        self.assertIn('"egoverse_mecka"', conv)
        self.assertIn("head_const", conv)
        self.assertIn("intrinsics", conv)


if __name__ == "__main__":
    unittest.main()
