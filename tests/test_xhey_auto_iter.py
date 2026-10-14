"""xhey_auto_iter：固定队列先于提议器，jsonl 镜像到 csv，C1 指标可入账。"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import xhey_auto_iter as ai  # noqa: E402


def _queue():
    return {"queue": [
        {"id": "C2", "dir": "kaggle/rsi_c2", "slug": "justinz2010/xhey-rsi-c2-longft", "status": "ready"},
        {"id": "C7", "dir": "kaggle/rsi_c7", "slug": "justinz2010/xhey-rsi-c7-stera", "status": "blocked"},
        {"id": "C8", "dir": "kaggle/rsi_c8", "slug": "justinz2010/xhey-rsi-c8-di", "status": "ready"},
        {"id": "C9", "dir": "kaggle/rsi_c9", "slug": "justinz2010/xhey-rsi-c9-stera-di", "status": "blocked"},
    ]}


class QueueOrder(unittest.TestCase):
    def test_ready_before_stera(self):
        ids = [i["id"] for i in ai.ready_queue_items(stera_exists=False, queue=_queue())]
        self.assertEqual(ids, ["C2", "C8"])

    def test_stera_appended_when_data_exists(self):
        ids = [i["id"] for i in ai.ready_queue_items(stera_exists=True, queue=_queue())]
        self.assertEqual(ids, ["C2", "C8", "C7", "C9"])

    def test_inflight_not_resubmitted(self):
        q = _queue()
        q["queue"][0]["status"] = "running"
        ids = [i["id"] for i in ai.ready_queue_items(stera_exists=True, queue=q)]
        self.assertEqual(ids, ["C8", "C7", "C9"])


class MirrorAndIngest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.ledger = root / "ledger.jsonl"
        self.csv = root / "ledger.csv"
        self.state = root / "auto_state.json"
        self.queue = root / "queue.json"
        self.chart = root / "progress.png"
        self.csv.write_text(
            "id,date,mix,overall_ade,moving_ade,vs_zero,status,source\r\n"
            "E0,baseline,zero,11.0255,14.0813,0.0,baseline,always\r\n"
            "C1,2026-10-14,iphone+xhey_rgb+aug +moving_loss,,,,running_kaggle,justinz2010/xhey-rsi-c1-moveloss\r\n",
            encoding="utf-8")
        self.patches = [
            patch.object(ai, "LEDGER", self.ledger),
            patch.object(ai, "CSV_PATH", self.csv),
            patch.object(ai, "STATE", self.state),
            patch.object(ai, "QUEUE_PATH", self.queue),
            patch.object(ai, "CHART", self.chart),
            patch.object(ai, "EXP", root),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_mirror_appends_and_keeps_hand_rows(self):
        row = {"id": "iter001_c0", "iteration": 1, "date": "2026-10-14 13:54",
               "source": "justinz2010/xhey-auto-iter",
               "config": dict(ai.DEFAULT, mix="iphone+openaoe+aug", openaoe="nano_cam"),
               "status": "ok", "overall_ade_cm": 11.07, "moving_ade_cm": 14.05,
               "delta_overall_cm": 0.0444, "delta_moving_cm": -0.03,
               "delta_moving_ci95": [-0.04, -0.02]}
        ai.append_ledger([row])
        text = self.csv.read_text(encoding="utf-8")
        self.assertIn("E0,baseline,zero", text)
        self.assertIn("iter001_c0,2026-10-14,iphone+openaoe_nano_cam+aug,11.0700,14.0500,0.0444,win_moving,justinz2010/xhey-auto-iter", text)
        self.assertTrue(text.startswith("id,date,mix"))

    def test_ingest_c1_updates_csv_row(self):
        metrics = {"results": {
            "zero": {"ade_cm_pooled": 11.0255},
            "iphone+xhey_rgb+aug": {
                "ade_cm_pooled": 11.10, "ade_minus_zero_cm": 0.0745,
                "ade_minus_zero_ci95": [0.06, 0.09],
                "moving_ade_cm": 14.01, "moving_zero_ade_cm": 14.0813,
                "moving_minus_zero_cm": -0.0713, "moving_minus_zero_ci95": [-0.09, -0.05],
                "ade_cm_seed_std": 0.02,
            },
        }}
        path = Path(self.tmp.name) / "metrics.json"
        path.write_text(json.dumps(metrics), encoding="utf-8")
        ai.ingest_metrics_file(path, ai.C1_SPEC)
        rows = {r["id"]: r for r in ai.load_ledger()}
        self.assertEqual(rows["C1"]["status"], "ok")
        self.assertAlmostEqual(rows["C1"]["delta_moving_cm"], -0.0713)
        self.assertEqual(rows["C1"]["config"]["loss_move_cm"], 5.0)
        csv = self.csv.read_text(encoding="utf-8")
        self.assertIn("C1,2026-10-14,iphone+xhey_rgb+aug loss_move_cm=5.0,11.1000,14.0100,0.0745,win_moving,justinz2010/xhey-rsi-c1-moveloss", csv)
        self.assertNotIn("running_kaggle", csv)

    def test_running_iter_does_not_submit_queue(self):
        self.state.write_text(json.dumps({"running": {"iteration": 2, "kernel": ai.KERNEL, "version": 2,
                                                      "candidates": []}}), encoding="utf-8")
        self.queue.write_text(json.dumps(_queue()), encoding="utf-8")
        pushed = []
        with patch.object(ai, "kernel_status", return_value="RUNNING"), \
                patch.object(ai, "submit_queue_item", side_effect=lambda item: pushed.append(item["id"])), \
                patch.object(ai, "launch", side_effect=AssertionError("should not launch")):
            rc = ai.step(type("A", (), {"no_launch": False, "k": 3})())
        self.assertEqual(rc, 0)
        self.assertEqual(pushed, [])
        state = json.loads(self.state.read_text(encoding="utf-8"))
        self.assertEqual(state["running"]["iteration"], 2)
        self.assertEqual(state["running"]["kernel"], "justinz2010/xhey-auto-iter")
        self.assertEqual(state["running"]["version"], 2)

    def test_idle_submits_queue_before_proposer(self):
        self.state.write_text(json.dumps({"running": None}), encoding="utf-8")
        self.queue.write_text(json.dumps(_queue()), encoding="utf-8")
        pushed = []
        with patch.object(ai, "kernel_status", return_value="UNKNOWN:none"), \
                patch.object(ai, "stera_data_exists", return_value=False), \
                patch.object(ai, "submit_queue_item", side_effect=lambda item: pushed.append(item["id"])), \
                patch.object(ai, "launch", side_effect=AssertionError("proposer must wait")):
            rc = ai.step(type("A", (), {"no_launch": False, "k": 3})())
        self.assertEqual(rc, 0)
        self.assertEqual(pushed, ["C2"])


class CommittedState(unittest.TestCase):
    def test_iteration_2_ingested(self):
        state = json.loads((ROOT / "experiments" / "auto_state.json").read_text(encoding="utf-8"))
        self.assertIsNone(state["running"])
        self.assertEqual(state["last_iteration"], 2)
        ids = {json.loads(l)["id"] for l in (ROOT / "experiments" / "ledger.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
        self.assertTrue({"iter002_c0", "iter002_c1", "iter002_c2", "C1", "C8"} <= ids)


def _kernel_metrics(mix, moving=-0.05):
    return {"results": {
        "zero": {"ade_cm_pooled": 11.0255},
        mix: {
            "ade_cm_pooled": 11.0, "ade_minus_zero_cm": -0.0255,
            "ade_minus_zero_ci95": [-0.04, -0.01],
            "moving_ade_cm": 14.03, "moving_zero_ade_cm": 14.0813,
            "moving_minus_zero_cm": moving, "moving_minus_zero_ci95": [-0.08, -0.02],
            "ade_cm_seed_std": 0.01,
        },
    }}


class ManyQueueRunning(unittest.TestCase):
    """C7 与 C8 可以同时在跑：queue_running 为列表，或另有 queue_running_also，都要入账。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.ledger = root / "ledger.jsonl"
        self.csv = root / "ledger.csv"
        self.state = root / "auto_state.json"
        self.queue = root / "queue.json"
        self.chart = root / "progress.png"
        self.csv.write_text("id,date,mix,overall_ade,moving_ade,vs_zero,status,source\r\n", encoding="utf-8")
        q = _queue()
        for item in q["queue"]:
            if item["id"] in ("C7", "C8"):
                item["status"] = "ready"
        self.queue.write_text(json.dumps(q), encoding="utf-8")
        self.patches = [
            patch.object(ai, "LEDGER", self.ledger),
            patch.object(ai, "CSV_PATH", self.csv),
            patch.object(ai, "STATE", self.state),
            patch.object(ai, "QUEUE_PATH", self.queue),
            patch.object(ai, "CHART", self.chart),
            patch.object(ai, "EXP", root),
        ]
        for p in self.patches:
            p.start()
        self.mixes = {
            "justinz2010/xhey-rsi-c7-stera": "iphone+stera+xhey_rgb+aug",
            "justinz2010/xhey-rsi-c8-di": "iphone+di+xhey_rgb+aug",
        }

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def _kaggle(self, *args, check=True):
        if args[:2] == ("kernels", "output") and args[2] in self.mixes:
            slug = args[2]
            dest = Path(args[4])
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "metrics.json").write_text(
                json.dumps(_kernel_metrics(self.mixes[slug])), encoding="utf-8")
        return ""

    def _run(self, state, statuses, no_launch):
        self.state.write_text(json.dumps(state), encoding="utf-8")
        pushed = []

        def status(slug=None):
            return statuses.get(slug, "UNKNOWN:none")

        with patch.object(ai, "kernel_status", side_effect=status), \
                patch.object(ai, "kaggle", side_effect=self._kaggle), \
                patch.object(ai, "chart"), \
                patch.object(ai, "submit_queue_item", side_effect=lambda item: pushed.append(item["id"])), \
                patch.object(ai, "launch", side_effect=AssertionError("proposer must wait")):
            rc = ai.step(type("A", (), {"no_launch": no_launch, "k": 3})())
        return rc, pushed

    def test_queue_running_also_ingests_c7_and_c8(self):
        state = {
            "running": None,
            "queue_running": {
                "id": "C7", "slug": "justinz2010/xhey-rsi-c7-stera", "launched": "2026-10-14 14:40",
            },
            "queue_running_also": [{
                "id": "C8", "slug": "justinz2010/xhey-rsi-c8-di", "launched": "2026-10-14 14:18",
            }],
        }
        done = {slug: "COMPLETE" for slug in self.mixes}
        rc, pushed = self._run(state, done, no_launch=True)
        self.assertEqual(rc, 0)
        self.assertEqual(pushed, [])
        rows = {r["id"]: r for r in ai.load_ledger()}
        self.assertEqual(set(rows), {"C7", "C8"})
        self.assertEqual(rows["C7"]["status"], "ok")
        self.assertEqual(rows["C8"]["status"], "ok")
        self.assertEqual(rows["C7"]["source"], "justinz2010/xhey-rsi-c7-stera")
        self.assertEqual(rows["C8"]["source"], "justinz2010/xhey-rsi-c8-di")
        self.assertAlmostEqual(rows["C7"]["delta_moving_cm"], -0.05)
        self.assertAlmostEqual(rows["C8"]["delta_moving_cm"], -0.05)

    def test_queue_running_list_ingests_both_when_finished(self):
        state = {
            "running": None,
            "queue_running": [
                {"id": "C7", "slug": "justinz2010/xhey-rsi-c7-stera", "launched": "2026-10-14 14:40"},
                {"id": "C8", "slug": "justinz2010/xhey-rsi-c8-di", "launched": "2026-10-14 14:18"},
            ],
        }
        done = {slug: "COMPLETE" for slug in self.mixes}
        rc, pushed = self._run(state, done, no_launch=True)
        self.assertEqual(rc, 0)
        rows = {r["id"]: r for r in ai.load_ledger()}
        self.assertEqual(rows["C7"]["status"], "ok")
        self.assertEqual(rows["C8"]["status"], "ok")
        self.assertEqual(pushed, [])

    def test_second_kernel_holds_slot_after_first_finishes(self):
        state = {
            "running": None,
            "queue_running": {
                "id": "C7", "slug": "justinz2010/xhey-rsi-c7-stera", "launched": "2026-10-14 14:40",
            },
            "queue_running_also": [{
                "id": "C8", "slug": "justinz2010/xhey-rsi-c8-di", "launched": "2026-10-14 14:18",
            }],
        }
        statuses = {
            "justinz2010/xhey-rsi-c7-stera": "COMPLETE",
            "justinz2010/xhey-rsi-c8-di": "RUNNING",
        }
        rc, pushed = self._run(state, statuses, no_launch=False)
        self.assertEqual(rc, 0)
        self.assertEqual(pushed, [])
        rows = {r["id"]: r for r in ai.load_ledger()}
        self.assertEqual(rows["C7"]["status"], "ok")
        self.assertNotIn("C8", rows)
        saved = json.loads(self.state.read_text(encoding="utf-8"))
        still = saved.get("queue_running")
        also = saved.get("queue_running_also") or []
        ids = []
        if isinstance(still, dict) and still.get("id"):
            ids.append(still["id"])
        elif isinstance(still, list):
            ids.extend(item.get("id") for item in still)
        ids.extend(item.get("id") for item in also if isinstance(item, dict))
        self.assertEqual(ids, ["C8"])

    def test_finished_auto_iter_does_not_drop_other_running_kernels(self):
        state = {
            "running": {"iteration": 2, "kernel": ai.KERNEL, "version": 2, "candidates": []},
            "queue_running": {
                "id": "C7", "slug": "justinz2010/xhey-rsi-c7-stera", "launched": "2026-10-14 14:40",
            },
            "queue_running_also": [{
                "id": "C8", "slug": "justinz2010/xhey-rsi-c8-di", "launched": "2026-10-14 14:18",
            }],
        }
        statuses = {
            ai.KERNEL: "COMPLETE",
            "justinz2010/xhey-rsi-c7-stera": "RUNNING",
            "justinz2010/xhey-rsi-c8-di": "RUNNING",
        }
        rc, pushed = self._run(state, statuses, no_launch=False)
        self.assertEqual(rc, 0)
        self.assertEqual(pushed, [])
        saved = json.loads(self.state.read_text(encoding="utf-8"))
        self.assertIsNone(saved.get("running"))
        ids = []
        still = saved.get("queue_running")
        if isinstance(still, dict) and still.get("id"):
            ids.append(still["id"])
        elif isinstance(still, list):
            ids.extend(item.get("id") for item in still)
        ids.extend(item.get("id") for item in (saved.get("queue_running_also") or []) if isinstance(item, dict))
        self.assertEqual(ids, ["C7", "C8"])


class TrainFlags(unittest.TestCase):
    def test_stera_di_and_tune_flags(self):
        src = (ROOT / "scripts" / "train_xhey.py").read_text(encoding="utf-8")
        self.assertIn('"stera", "di"', src)
        self.assertIn("--stera", src)
        self.assertIn("--di", src)
        self.assertIn("--hidden", src)
        self.assertIn("--no-final", src)
        self.assertIn('TUNE = {"hidden": 512', src)
        self.assertNotIn("<<<<<<<", src)


if __name__ == "__main__":
    unittest.main()
