"""xhey_auto_iter：固定队列先于提议器，jsonl 镜像到 csv，C1 指标可入账。"""
import json
import re
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
        # 入账日期来自 row_from_metrics 的 datetime.now()。钉在 CSV 里已有的实验日，
        # 避免断言跟墙上时钟走（这条用例写于 2026-10-14，次日会把日期写成当天）。
        class _Clock(ai.datetime):
            @classmethod
            def now(cls, tz=None):
                return cls(2026, 10, 14, 12, 0, 0)

        with patch.object(ai, "datetime", _Clock):
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
                patch.object(ai, "kernel_has_qc_filter", return_value=True), \
                patch.object(ai, "submit_queue_item", side_effect=lambda item: pushed.append(item["id"])), \
                patch.object(ai, "launch", side_effect=AssertionError("proposer must wait")):
            rc = ai.step(type("A", (), {"no_launch": False, "k": 3})())
        self.assertEqual(rc, 0)
        self.assertEqual(pushed, ["C2"])

    def test_unfiltered_ready_is_not_submitted_and_blocks_proposer(self):
        """C9 v1 没有 --qc-filter。循环不能再交这种候选，也不能因此改跑提议器。"""
        self.state.write_text(json.dumps({"running": None, "queue_running": None}), encoding="utf-8")
        self.queue.write_text(json.dumps({"queue": [{
            "id": "C9", "dir": "kaggle/rsi_c9",
            "slug": "justinz2010/xhey-rsi-c9-stera-di", "status": "ready",
        }]}), encoding="utf-8")
        pushed = []
        with patch.object(ai, "kernel_status", return_value="UNKNOWN:none"), \
                patch.object(ai, "stera_data_exists", return_value=True), \
                patch.object(ai, "submit_queue_item", side_effect=lambda item: pushed.append(item["id"])), \
                patch.object(ai, "launch", side_effect=AssertionError("proposer must wait")):
            rc = ai.step(type("A", (), {"no_launch": False, "k": 3})())
        self.assertEqual(rc, 0)
        self.assertEqual(pushed, [])

    def test_qc_candidate_is_submitted_ahead_of_unfiltered(self):
        self.state.write_text(json.dumps({"running": None, "queue_running": None}), encoding="utf-8")
        self.queue.write_text(json.dumps({"queue": [
            {"id": "C9", "dir": "kaggle/rsi_c9", "slug": "justinz2010/xhey-rsi-c9-stera-di", "status": "ready"},
            {"id": "C9b", "dir": "kaggle/rsi_c9b", "slug": "justinz2010/xhey-rsi-c9b-stera-di-qc", "status": "ready"},
        ]}), encoding="utf-8")
        pushed = []
        with patch.object(ai, "kernel_status", return_value="UNKNOWN:none"), \
                patch.object(ai, "stera_data_exists", return_value=True), \
                patch.object(ai, "submit_queue_item", side_effect=lambda item: pushed.append(item["id"])), \
                patch.object(ai, "launch", side_effect=AssertionError("proposer must wait")):
            rc = ai.step(type("A", (), {"no_launch": False, "k": 3})())
        self.assertEqual(rc, 0)
        self.assertEqual(pushed, ["C9b"])

    def test_running_unfiltered_c9_still_ingests(self):
        mix = "iphone+stera+di+xhey_rgb+aug"
        slug = "justinz2010/xhey-rsi-c9-stera-di"
        self.queue.write_text(json.dumps({"queue": [{
            "id": "C9", "dir": "kaggle/rsi_c9", "slug": slug, "status": "running", "mix": mix,
        }]}), encoding="utf-8")
        self.state.write_text(json.dumps({
            "running": None,
            "queue_running": {"id": "C9", "slug": slug, "mix": mix, "launched": "2026-10-15 21:24"},
        }), encoding="utf-8")

        def kaggle(*args, check=True):
            if args[:2] == ("kernels", "output") and args[2] == slug:
                dest = Path(args[4])
                dest.mkdir(parents=True, exist_ok=True)
                (dest / "metrics.json").write_text(json.dumps(_kernel_metrics(mix)), encoding="utf-8")
            return ""

        def status(slug_arg=None):
            return "COMPLETE" if slug_arg == slug else "UNKNOWN:none"

        with patch.object(ai, "kernel_status", side_effect=status), \
                patch.object(ai, "kaggle", side_effect=kaggle), \
                patch.object(ai, "chart"), \
                patch.object(ai, "report_charts"), \
                patch.object(ai, "submit_queue_item", side_effect=AssertionError("must not resubmit")), \
                patch.object(ai, "launch", side_effect=AssertionError("proposer must wait")):
            rc = ai.step(type("A", (), {"no_launch": True, "k": 3})())
        self.assertEqual(rc, 0)
        rows = {r["id"]: r for r in ai.load_ledger()}
        self.assertEqual(rows["C9"]["status"], "ok")
        self.assertEqual(rows["C9"]["config"]["mix"], mix)
        self.assertEqual(rows["C9"]["source"], slug)


class CommittedState(unittest.TestCase):
    def test_iteration_2_ingested(self):
        state = json.loads((ROOT / "experiments" / "auto_state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["last_iteration"], 2)
        self.assertEqual(state["last_kernel"], "justinz2010/xhey-auto-iter")
        running = state["running"]
        self.assertEqual(running["iteration"], 3)
        self.assertEqual(running["kernel"], "justinz2010/xhey-auto-iter")
        self.assertEqual(running["version"], 3)
        self.assertEqual(running["launched"], "2026-10-18 09:22")
        self.assertEqual(len(running["candidates"]), 3)
        ids = {json.loads(l)["id"] for l in (ROOT / "experiments" / "ledger.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
        self.assertTrue({"iter002_c0", "iter002_c1", "iter002_c2", "C1", "C8"} <= ids)

    def test_c9_done_worse_and_c9b_already_running(self):
        state = json.loads((ROOT / "experiments" / "auto_state.json").read_text(encoding="utf-8"))
        self.assertIsNone(state["queue_running"])
        running = state["running"]
        self.assertEqual(running["iteration"], 3)
        self.assertEqual(running["kernel"], "justinz2010/xhey-auto-iter")
        self.assertEqual(running["version"], 3)
        self.assertEqual(running["launched"], "2026-10-18 09:22")
        self.assertNotIn("queue_running_also", state)
        q = json.loads((ROOT / "docs" / "xhey" / "queue.json").read_text(encoding="utf-8"))
        ids = [i["id"] for i in q["queue"]]
        self.assertEqual(ids[ids.index("C9") + 1], "C9b")
        by_id = {i["id"]: i for i in q["queue"]}
        c9 = by_id["C9"]
        self.assertEqual(c9["status"], "done_worse")
        self.assertEqual(c9["control"], "unfiltered/world-frame")
        self.assertFalse(c9["qc_filter"])
        self.assertEqual(c9["result"]["protocol"], "unfiltered/world-frame 3-clip LOO")
        self.assertEqual(c9["result"]["zero_overall_ade_cm"], 11.026)
        self.assertEqual(c9["result"]["C9ctl_overall_ade_cm"], 11.062)
        self.assertEqual(c9["result"]["C9ctl_delta_overall_cm"], 0.037)
        self.assertEqual(c9["result"]["C9ctl_moving_minus_zero_cm"], -0.021)
        self.assertEqual(c9["result"]["C9_overall_ade_cm"], 11.100)
        self.assertEqual(c9["result"]["C9_delta_overall_cm"], 0.075)
        self.assertEqual(c9["result"]["C9_moving_minus_zero_cm"], -0.005)
        self.assertEqual(by_id["C8"]["status"], "done_better_moving")
        self.assertEqual(by_id["C7"]["status"], "done_worse")
        self.assertEqual(by_id["C7b"]["status"], "done_better_moving")
        self.assertEqual(by_id["C2"]["status"], "done_better_moving")
        self.assertEqual(by_id["C11"]["status"], "done_no_gain_vs_ctl")
        c12 = by_id["C12"]
        self.assertEqual(c12["status"], "done_worse_vs_ctl")
        self.assertEqual(c12["result"]["C12_overall_ade_cm"], 9.845)
        self.assertEqual(c12["result"]["C12_delta_overall_cm"], 0.066)
        self.assertEqual(c12["result"]["C12_moving_ade_cm"], 12.619)
        self.assertEqual(c12["result"]["C12_moving_minus_zero_cm"], -0.064)
        self.assertEqual(c12["result"]["C12_moving_ci95"], [-0.071, -0.056])
        self.assertEqual(c12["result"]["zero_overall_ade_cm"], 9.779)
        self.assertEqual(c12["result"]["C11ctl_moving_minus_zero_cm"], -0.137)
        c9b = by_id["C9b"]
        self.assertEqual(c9b["status"], "done_worse_vs_ctl")
        self.assertEqual(c9b["slug"], "justinz2010/xhey-rsi-c9b-stera-di-qc")
        self.assertEqual(c9b["mix"], "iphone+stera+di+xhey_rgb+aug")
        self.assertEqual(c9b["kaggle"], "https://www.kaggle.com/code/justinz2010/xhey-rsi-c9b-stera-di-qc")
        self.assertEqual(c9b["result"]["protocol"], "QC 2-clip LOO (2026-08-22_18-21, 2026-08-24_21-18)")
        self.assertEqual(c9b["result"]["zero_overall_ade_cm"], 7.227)
        self.assertEqual(c9b["result"]["C9bctl_overall_ade_cm"], 7.221)
        self.assertEqual(c9b["result"]["C9bctl_delta_overall_cm"], -0.006)
        self.assertEqual(c9b["result"]["C9bctl_moving_minus_zero_cm"], -0.068)
        self.assertEqual(c9b["result"]["C9b_overall_ade_cm"], 7.235)
        self.assertEqual(c9b["result"]["C9b_delta_overall_cm"], 0.007)
        self.assertEqual(c9b["result"]["C9b_moving_ade_cm"], 10.383)
        self.assertEqual(c9b["result"]["C9b_moving_minus_zero_cm"], -0.05)
        self.assertEqual(c9b["result"]["C9b_moving_ci95"], [-0.056, -0.044])
        self.assertEqual(c9b["qc"]["stera_pass"]["b01"], "213/757")
        self.assertEqual(c9b["qc"]["stera_kept_segments"], 1297)
        self.assertEqual(c9b["qc"]["di_pass"], "25/25")
        self.assertEqual(c9b["qc"]["iphone_depth_pass"], by_id["C7b"]["qc"]["iphone_depth_pass"])
        self.assertEqual(c9b["qc"]["xhey_rgb_pass"], "14/17")
        self.assertIn("--qc-filter", c9b["protocol"])
        self.assertIn("6000", c9b["protocol"])
        self.assertIn("3000", c9b["protocol"])
        ok, skipped = ai.split_qc_ready(ai.ready_queue_items(stera_exists=True, queue=q))
        self.assertEqual([item["id"] for item in ok], [])
        self.assertEqual(skipped, [])
        c13 = by_id["C13"]
        self.assertEqual(c13["status"], "done_worse_vs_ctl")
        self.assertEqual(c13["mix"], "iphone+egodex+xhey_rgb+aug")
        self.assertEqual(c13["result"]["C13_overall_ade_cm"], 9.803)
        self.assertEqual(c13["result"]["C13_delta_overall_cm"], 0.024)
        self.assertEqual(c13["result"]["C13_moving_ade_cm"], 12.557)
        self.assertEqual(c13["result"]["C13_moving_minus_zero_cm"], -0.126)
        self.assertEqual(c13["result"]["C13_moving_ci95"], [-0.134, -0.117])
        self.assertEqual(c13["result"]["C13ctl_overall_ade_cm"], 9.759)
        self.assertEqual(c13["result"]["C13ctl_moving_minus_zero_cm"], -0.137)
        self.assertEqual(c13["result"]["C13ctl_moving_ci95"], [-0.146, -0.129])
        self.assertEqual(c13["result"]["zero_overall_ade_cm"], 9.779)
        self.assertIn("undefined HO", c13["note_v1"])
        self.assertIn("credentials were missing", c13["note_v1"])
        self.assertIn("2026-10-17 09:20", c13["relaunched_v2"])
        self.assertEqual(c13["ingested"], "2026-10-17 21:27")
        self.assertTrue(ai.kernel_has_qc_filter(c13))
        c13_src = (ROOT / "kaggle" / "rsi_c13" / "xhey-rsi-c13-egodex.py").read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"\bHO\b", c13_src))
        self.assertIn("EG = find", c13_src)
        c14 = by_id["C14"]
        self.assertEqual(c14["status"], "done_win")
        self.assertEqual(c14["slug"], "justinz2010/xhey-rsi-c14-xphi")
        self.assertEqual(c14["launched"], "2026-10-17 21:27")
        self.assertEqual(c14["mix"], "iphone+xphi+xhey_rgb+aug")
        self.assertIn("WiLoR", c14["caveats"])
        self.assertIn("HFOV 90", c14["caveats"])
        self.assertIn("--qc-filter", c14["protocol"])
        self.assertIn("--qc-segments", c14["protocol"])
        self.assertEqual(
            c14["data"],
            "102 clips labelled, 5.03 h, QC whole-clip pass 3/102, segment-level retention ~0.32 h; dataset justinz2010/xhey-xphi-cam is uploaded and ready on Kaggle.",
        )
        self.assertIn("justinz2010/xhey-xphi-cam", c14["datasets"][1])
        self.assertTrue(ai.kernel_has_qc_filter(c14))
        self.assertEqual(ai.QUEUE_CONFIG["C14"]["mix"], "iphone+xphi+xhey_rgb+aug")
        c14_meta = json.loads((ROOT / "kaggle" / "rsi_c14" / "kernel-metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(c14_meta["id"], "justinz2010/xhey-rsi-c14-xphi")
        self.assertIn("justinz2010/xhey-xphi-cam", c14_meta["dataset_sources"])
        wilor_meta = json.loads((ROOT / "kaggle" / "xphi_wilor" / "kernel-metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(wilor_meta["id"], "justinz2010/xhey-xphi-wilor")
        self.assertIn("--qc-filter", ai.KERNEL_TEMPLATE)
        meta = json.loads((ROOT / "kaggle" / "rsi_c9b" / "kernel-metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["id"], "justinz2010/xhey-rsi-c9b-stera-di-qc")
        self.assertIn("justinz2010/xhey-c11-di-cam", meta["dataset_sources"])
        self.assertIn("justinz2010/xhey-stera-pretrain", meta["dataset_sources"])
        self.assertTrue(ai.kernel_has_qc_filter(c9b))
        self.assertFalse(ai.kernel_has_qc_filter(by_id["C9"]))
        rows = {r["id"]: r for r in ai.load_ledger()}
        self.assertEqual(rows["C9"]["status"], "ok")
        self.assertEqual(rows["C9"]["source"], "justinz2010/xhey-rsi-c9-stera-di")
        self.assertEqual(rows["C9"]["config"]["mix"], "iphone+stera+di+xhey_rgb+aug")
        self.assertEqual(rows["C9"]["overall_ade_cm"], 11.100)
        self.assertEqual(rows["C9"]["zero_overall_ade_cm"], 11.026)
        self.assertEqual(rows["C9"]["delta_overall_cm"], 0.075)
        self.assertEqual(rows["C9"]["delta_moving_cm"], -0.005)
        self.assertIsNone(rows["C9"]["moving_ade_cm"])
        self.assertIsNone(rows["C9"]["delta_moving_ci95"])
        ctl = rows["C9ctl"]
        self.assertEqual(ctl["status"], "ok")
        self.assertEqual(ctl["source"], rows["C9"]["source"])
        self.assertEqual(ctl["config"]["mix"], "iphone+xhey_rgb+aug")
        self.assertEqual(ctl["overall_ade_cm"], 11.062)
        self.assertEqual(ctl["zero_overall_ade_cm"], 11.026)
        self.assertEqual(ctl["delta_overall_cm"], 0.037)
        self.assertEqual(ctl["delta_moving_cm"], -0.021)
        self.assertIsNone(ctl["moving_ade_cm"])
        ids = [r["id"] for r in ai.load_ledger()]
        self.assertEqual(ids.count("C9"), 1)
        self.assertEqual(ids.count("C9ctl"), 1)
        self.assertEqual(ids.count("C12"), 1)
        c12row = rows["C12"]
        self.assertEqual(c12row["status"], "ok")
        self.assertEqual(c12row["date"], "2026-10-16 09:23")
        self.assertEqual(c12row["source"], "justinz2010/xhey-rsi-c12-hoi4d")
        self.assertEqual(c12row["config"]["mix"], "iphone+hoi4d+xhey_rgb+aug")
        self.assertEqual(c12row["overall_ade_cm"], 9.84528839947837)
        self.assertEqual(c12row["zero_overall_ade_cm"], 9.77909130440752)
        self.assertEqual(c12row["delta_overall_cm"], 0.06619709507084982)
        self.assertEqual(c12row["moving_ade_cm"], 12.61918517304698)
        self.assertEqual(c12row["delta_moving_cm"], -0.06371208950413923)
        self.assertEqual(c12row["delta_moving_ci95"], [-0.0713584667934282, -0.05602845413106699])
        self.assertEqual(ids.count("C9b"), 1)
        self.assertEqual(ids.count("C9bctl"), 1)
        c9brow = rows["C9b"]
        self.assertEqual(c9brow["status"], "ok")
        self.assertEqual(c9brow["date"], "2026-10-16")
        self.assertEqual(c9brow["source"], "justinz2010/xhey-rsi-c9b-stera-di-qc")
        self.assertEqual(c9brow["config"]["mix"], "iphone+stera+di+xhey_rgb+aug")
        self.assertEqual(c9brow["overall_ade_cm"], 7.23459540460082)
        self.assertEqual(c9brow["zero_overall_ade_cm"], 7.227108386200057)
        self.assertEqual(c9brow["delta_overall_cm"], 0.007487018400761065)
        self.assertEqual(c9brow["moving_ade_cm"], 10.382878606425848)
        self.assertEqual(c9brow["delta_moving_cm"], -0.04965893045995437)
        self.assertEqual(c9brow["delta_moving_ci95"], [-0.05559107366730263, -0.04406222087818459])
        c9bctl = rows["C9bctl"]
        self.assertEqual(c9bctl["status"], "ok")
        self.assertEqual(c9bctl["source"], c9brow["source"])
        self.assertEqual(c9bctl["config"]["mix"], "iphone+xhey_rgb+aug")
        self.assertEqual(c9bctl["overall_ade_cm"], 7.221381990047469)
        self.assertEqual(c9bctl["zero_overall_ade_cm"], 7.227108386200057)
        self.assertEqual(c9bctl["delta_overall_cm"], -0.005726396152587624)
        self.assertEqual(c9bctl["moving_ade_cm"], 10.364363733614738)
        self.assertEqual(c9bctl["delta_moving_cm"], -0.06817380327106637)
        self.assertEqual(ids.count("C13"), 1)
        c13row = rows["C13"]
        self.assertEqual(c13row["status"], "ok")
        self.assertEqual(c13row["date"], "2026-10-17 21:27")
        self.assertEqual(c13row["source"], "justinz2010/xhey-rsi-c13-egodex")
        self.assertEqual(c13row["config"]["mix"], "iphone+egodex+xhey_rgb+aug")
        self.assertEqual(c13row["overall_ade_cm"], 9.8031111699139)
        self.assertEqual(c13row["zero_overall_ade_cm"], 9.77909130440752)
        self.assertEqual(c13row["delta_overall_cm"], 0.02401986550637711)
        self.assertEqual(c13row["moving_ade_cm"], 12.557135490486884)
        self.assertEqual(c13row["delta_moving_cm"], -0.1257617720642352)
        self.assertEqual(c13row["delta_moving_ci95"], [-0.1344363025584355, -0.11697807499012294])
        self.assertEqual(ids.count("C14"), 1)
        c14row = rows["C14"]
        self.assertEqual(c14row["status"], "ok")
        self.assertEqual(c14row["date"], "2026-10-18 09:22")
        self.assertEqual(c14row["source"], "justinz2010/xhey-rsi-c14-xphi")
        self.assertEqual(c14row["config"]["mix"], "iphone+xphi+xhey_rgb+aug")
        self.assertEqual(c14row["overall_ade_cm"], 9.750331477450072)
        self.assertEqual(c14row["zero_overall_ade_cm"], 9.77909130440752)
        self.assertEqual(c14row["delta_overall_cm"], -0.02875982695745208)
        self.assertEqual(c14row["moving_ade_cm"], 12.534017037982933)
        self.assertEqual(c14row["delta_moving_cm"], -0.14888022456818356)
        self.assertEqual(c14row["delta_moving_ci95"], [-0.15879802343330682, -0.13937611602942523])


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


class QueueRunningMix(unittest.TestCase):
    """C7b 不在 QUEUE_CONFIG。正在跑的规格必须带着自己的 mix，不能记成 iphone 对照组。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.ledger = root / "ledger.jsonl"
        self.csv = root / "ledger.csv"
        self.state = root / "auto_state.json"
        self.queue = root / "queue.json"
        self.chart = root / "progress.png"
        self.csv.write_text("id,date,mix,overall_ade,moving_ade,vs_zero,status,source\r\n", encoding="utf-8")
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
        self.stera = "iphone+stera+xhey_rgb+aug"
        self.control = "iphone+xhey_rgb+aug"
        self.slug = "justinz2010/xhey-rsi-c7b-stera-all"

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def _metrics(self):
        def arm(overall, delta, moving, moving_delta):
            return {
                "ade_cm_pooled": overall, "ade_minus_zero_cm": delta,
                "ade_minus_zero_ci95": [delta - 0.004, delta + 0.004],
                "moving_ade_cm": moving, "moving_zero_ade_cm": moving - moving_delta,
                "moving_minus_zero_cm": moving_delta,
                "moving_minus_zero_ci95": [moving_delta - 0.006, moving_delta + 0.006],
                "ade_cm_seed_std": 0.01,
            }
        return {"results": {
            "zero": {"ade_cm_pooled": 7.227108386200057},
            self.control: arm(7.221, -0.006, 10.364, -0.068),
            self.stera: arm(7.232, 0.0045, 10.378, -0.054),
        }}

    def test_queue_running_c7b_records_its_mix_not_iphone_control(self):
        q = _queue()
        q["queue"].append({
            "id": "C7b", "dir": "kaggle/rsi_c7b", "slug": self.slug,
            "status": "running", "mix": self.stera,
        })
        self.queue.write_text(json.dumps(q), encoding="utf-8")
        # 和线上一样：queue_running 当时没写 mix，配比在 queue.json 上。
        self.state.write_text(json.dumps({
            "running": None,
            "queue_running": {"id": "C7b", "slug": self.slug, "launched": "2026-10-15 09:28"},
        }), encoding="utf-8")
        metrics = self._metrics()

        def kaggle(*args, check=True):
            if args[:2] == ("kernels", "output") and args[2] == self.slug:
                dest = Path(args[4])
                dest.mkdir(parents=True, exist_ok=True)
                (dest / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
            return ""

        def status(slug=None):
            return "COMPLETE" if slug == self.slug else "UNKNOWN:none"

        with patch.object(ai, "kernel_status", side_effect=status), \
                patch.object(ai, "kaggle", side_effect=kaggle), \
                patch.object(ai, "chart"), \
                patch.object(ai, "report_charts"), \
                patch.object(ai, "submit_queue_item"), \
                patch.object(ai, "launch", side_effect=AssertionError("proposer must wait")):
            rc = ai.step(type("A", (), {"no_launch": True, "k": 3})())
        self.assertEqual(rc, 0)
        rows = {r["id"]: r for r in ai.load_ledger()}
        self.assertEqual(rows["C7b"]["config"]["mix"], self.stera)
        self.assertAlmostEqual(rows["C7b"]["delta_moving_cm"], -0.054)
        self.assertAlmostEqual(rows["C7b"]["moving_ade_cm"], 10.378)
        self.assertNotAlmostEqual(rows["C7b"]["delta_moving_cm"], -0.068)
        csv = self.csv.read_text(encoding="utf-8")
        self.assertIn("iphone+stera+xhey_rgb+aug,7.2320,10.3780,0.0045,", csv)
        self.assertNotIn("10.3640", csv)

    def test_submit_queue_item_stores_mix_on_queue_running(self):
        item = {"id": "C7b", "dir": "kaggle/rsi_c7b", "slug": self.slug,
                "status": "ready", "mix": self.stera}
        self.queue.write_text(json.dumps({"queue": [item]}), encoding="utf-8")
        self.state.write_text(json.dumps({"running": None}), encoding="utf-8")
        with patch.object(ai, "kaggle", return_value="pushed"):
            ai.submit_queue_item(dict(item))
        saved = json.loads(self.state.read_text(encoding="utf-8"))
        self.assertEqual(saved["queue_running"]["id"], "C7b")
        self.assertEqual(saved["queue_running"]["slug"], self.slug)
        self.assertEqual(saved["queue_running"]["mix"], self.stera)
        queued = json.loads(self.queue.read_text(encoding="utf-8"))["queue"][0]
        self.assertEqual(queued["status"], "running")
        self.assertEqual(queued["mix"], self.stera)


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
