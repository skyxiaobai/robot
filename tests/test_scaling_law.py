# -*- coding: utf-8 -*-
"""缩放律分析：最优验证损失对 log(数据量) 的拟合。"""
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import scaling_law  # noqa: E402


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class ExtractBestValLossTest(unittest.TestCase):
    def test_minimum_val_loss_ignores_lower_train_loss_and_later_worse_val(self):
        text = "\n".join([
            "step=100 train_loss: 0.10 val_loss: 1.50",
            "step=200 train_loss=0.05 validation_loss: 0.90",
            'step=300 {"Loss": 0.01, "val_loss": 1.10}',
            "Validation loss: 1.40",
        ])
        self.assertAlmostEqual(scaling_law.extract_best_val_loss(text), 0.90)

    def test_no_validation_loss_returns_none(self):
        self.assertIsNone(scaling_law.extract_best_val_loss("train_loss: 0.2\nloss:0.074"))

    def test_scientific_notation_keeps_distinct_small_losses(self):
        text = "\n".join([
            "step=1 train_loss: 1.000000e-04 val_loss: 1.088600e-04",
            "trans_mse: 1.000000e-06",
            "copy_current_wrist: 1.147300e-04",
            "step=1 val_loss: 1.085100e-04",
        ])
        self.assertAlmostEqual(scaling_law.extract_best_val_loss(text), 1.0851e-04)
        self.assertNotAlmostEqual(scaling_law.extract_best_val_loss(text), 1.0886e-04)

    def test_component_metrics_follow_the_best_val_loss(self):
        text = "\n".join([
            "copy_current_wrist: 9.000000e-04",
            "copy_trans_mse: 8.000000e-04",
            "step=1 train_loss: 1.000000e-03 val_loss: 2.000000e-04",
            "trans_mse: 9.000000e-05",
            "rot_mse: 8.000000e-05",
            "step=2 train_loss: 1.000000e-04 val_loss: 1.085100e-04",
            "copy_current_wrist: 1.200000e-04",
            "trans_mse: 3.000000e-06",
            "rot_mse: 4.000000e-05",
            "copy_trans_mse: 1.000000e-03",
        ])
        metrics = scaling_law.extract_run_metrics(text)
        self.assertAlmostEqual(metrics["val_loss"], 1.0851e-04)
        self.assertAlmostEqual(metrics["copy_current_wrist"], 1.2e-04)
        self.assertAlmostEqual(metrics["trans_mse"], 3.0e-06)
        self.assertAlmostEqual(metrics["rot_mse"], 4.0e-05)
        self.assertAlmostEqual(metrics["val_baseline_ratio"], 1.0851e-04 / 1.2e-04)
        self.assertNotAlmostEqual(metrics["trans_mse"], 1.0e-03)


class FitLogLinearTest(unittest.TestCase):
    def test_recovers_slope_against_natural_log_of_data_size(self):
        hours = [10.0, 40.0, 160.0, 640.0]
        intercept, slope = 1.5, -0.2
        losses = [intercept + slope * math.log(h) for h in hours]
        fit = scaling_law.fit_log_linear(hours, losses)
        self.assertAlmostEqual(fit["intercept"], intercept, places=9)
        self.assertAlmostEqual(fit["slope"], slope, places=9)
        self.assertGreater(fit["r2"], 0.999)


class LoadRunsTest(unittest.TestCase):
    def test_yaml_hours_and_csv_episodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            yaml_path = root / "runs.yaml"
            _write(yaml_path, """
runs:
  - name: pretrain_10h
    hours: 10
    log: logs/a.log
  - name: pretrain_40h
    hours: 40
    log: logs/b.log
""")
            csv_path = root / "runs.csv"
            _write(csv_path, "name,episodes,log\nrun_a,20,logs/a.log\nrun_b,80,logs/b.log\n")
            yaml_runs = scaling_law.load_runs(str(yaml_path))
            csv_runs = scaling_law.load_runs(str(csv_path))
        self.assertEqual([r["name"] for r in yaml_runs], ["pretrain_10h", "pretrain_40h"])
        self.assertEqual(yaml_runs[0]["size"], 10.0)
        self.assertEqual(yaml_runs[0]["size_kind"], "hours")
        self.assertEqual(csv_runs[1]["size"], 80.0)
        self.assertEqual(csv_runs[1]["size_kind"], "episodes")


class AnalyzeAndReportTest(unittest.TestCase):
    def test_example_config_fits_and_html_contains_plot(self):
        out = ROOT / "outputs" / "scaling_law_report_test.html"
        code = scaling_law.main([
            "--runs", str(ROOT / "examples" / "scaling_law_runs.yaml"),
            "--out", str(out),
        ])
        self.assertEqual(code, 0)
        html = out.read_text(encoding="utf-8")
        self.assertIn("data:image/png;base64,", html)
        self.assertIn("pretrain_8h", html)
        self.assertIn("pretrain_512h", html)
        # 示例日志里更小的 train_loss 和更差的后续 val_loss 不能当成最优点
        self.assertIn("1.480140", html)
        self.assertIn("0.440419", html)
        out.unlink(missing_ok=True)

    def test_baseline_split_and_ratio_fit_appear_in_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sizes = [1000.0, 4000.0, 16000.0]
            intercept, slope = 1.2, -0.02
            for size in sizes:
                ratio = intercept + slope * math.log(size)
                loss = ratio * 1.1
                (root / ("n%d.log" % int(size))).write_text(
                    "\n".join([
                        "copy_current_wrist: %r" % 1.1,
                        "step=1 train_loss: 1.0 val_loss: %r" % loss,
                        "trans_mse: %r" % (loss * 0.4),
                        "rot_mse: %r" % (loss * 0.6),
                    ]) + "\n",
                    encoding="utf-8",
                )
            cfg = root / "runs.yaml"
            lines = ["runs:"]
            for size in sizes:
                lines.append("  - name: n%d" % int(size))
                lines.append("    size: %d" % int(size))
                lines.append("    log: n%d.log" % int(size))
            cfg.write_text("\n".join(lines) + "\n", encoding="utf-8")
            out = root / "report.html"
            code = scaling_law.main(["--runs", str(cfg), "--out", str(out)])
            self.assertEqual(code, 0)
            html = out.read_text(encoding="utf-8")
            self.assertIn("保持不动基线", html)
            self.assertIn("平移", html)
            self.assertIn("旋转", html)
            self.assertIn("相对基线", html)
            result = scaling_law.analyze_config(str(cfg))
            self.assertAlmostEqual(result["ratio_fit"]["intercept"], intercept, places=6)
            self.assertAlmostEqual(result["ratio_fit"]["slope"], slope, places=6)
            self.assertGreater(result["ratio_fit"]["r2"], 0.999)

    def test_section_html_for_training_report(self):
        section = scaling_law.section_html(str(ROOT / "examples" / "scaling_law_runs.yaml"))
        self.assertIn("<h2>", section)
        self.assertIn("data:image/png;base64,", section)

    def test_missing_config_skips(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = str(Path(tmp) / "nope.yaml")
            self.assertEqual(scaling_law.section_html(missing), "")
            code = scaling_law.main(["--runs", missing, "--out", str(Path(tmp) / "r.html")])
            self.assertEqual(code, 0)
            self.assertFalse((Path(tmp) / "r.html").exists())

    def test_no_runs_argument_skips(self):
        self.assertEqual(scaling_law.main([]), 0)

    def test_empty_runs_and_logs_without_val_loss_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            empty = root / "empty.yaml"
            _write(empty, "runs: []\n")
            self.assertEqual(scaling_law.section_html(str(empty)), "")
            log = root / "train_only.log"
            _write(log, "train_loss: 0.2\n")
            cfg = root / "one.yaml"
            _write(cfg, "runs:\n  - name: only\n    hours: 10\n    log: train_only.log\n")
            self.assertEqual(scaling_law.section_html(str(cfg)), "")


if __name__ == "__main__":
    unittest.main()
