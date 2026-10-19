import sys, types, unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


class XphiSourceKey(unittest.TestCase):
    def test_xphi_is_pretrain_key_and_loaded(self):
        import train_xhey as tx
        src = (ROOT / "scripts" / "train_xhey.py").read_text(encoding="utf-8")
        self.assertIn('"hoi4d", "xphi", "egoverse")', src)
        self.assertIn('"--xphi"', src)
        calls = []
        a = types.SimpleNamespace(out="/tmp/xphi_test_out", xphi="/nonexistent/xphi")
        with mock.patch.object(tx, "try_external", side_effect=lambda k, r, c: calls.append((k, str(r))) or None):
            ext = tx.prepare_external(["iphone+xphi"], {}, [], [], a)
        self.assertEqual(calls, [("xphi", "/nonexistent/xphi")])
        self.assertNotIn("xphi", ext)


if __name__ == "__main__":
    unittest.main()
