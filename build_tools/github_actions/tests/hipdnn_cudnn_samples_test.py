# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# test_hipdnn_cudnn_samples.py lives under github_actions/test_executable_scripts/.
sys.path.insert(
    0,
    os.fspath(Path(__file__).parent.parent / "test_executable_scripts"),
)

import test_hipdnn_cudnn_samples as driver


def _tu(tu, tier, outcome, **extra):
    return {"tu": tu, "tier": tier, "outcome": outcome, **extra}


def _case(case, tier, outcome, verdict):
    return {
        "tu": f"legacy_samples::{case}",
        "case": case,
        "tier": tier,
        "outcome": outcome,
        "verdict": verdict,
    }


class SidecarReportTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.report_dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, relpath, entry):
        path = self.report_dir / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entry), encoding="utf-8")

    def _write_mixed_corpus(self):
        self._write("conv_fprop.json", _tu("conv_fprop", "RUN", "ran-clean"))
        self._write("legacy_samples.json", _tu("legacy_samples", "RUN", "compiled"))
        self._write("legacy/a.json", _case("a", "RUN_CLEAN", "ran-clean", "expected"))
        self._write("legacy/b.json", _case("b", "XFAIL_RUN", "crashed", "expected"))
        self._write("legacy/c.json", _case("c", "VACUOUS", "ran-clean", "expected"))
        self._write("legacy/d.json", _case("d", "RUN_CLEAN", "crashed", "unexpected"))

    def test_reads_case_sidecars_in_subdirectories(self):
        self._write_mixed_corpus()
        entries, anomalies = driver.load_sidecars(self.report_dir)
        self.assertEqual(anomalies, [])
        self.assertEqual(len(entries), 6)
        self.assertIn("legacy_samples::d", {e["tu"] for e in entries})

    def test_flat_report_from_older_harness_is_unchanged(self):
        self._write("conv_fprop.json", _tu("conv_fprop", "RUN", "ran-clean"))
        self._write(
            "matmul.json", _tu("matmul", "XFAIL_COMPILE", "xfail-still-failing")
        )
        entries, anomalies = driver.load_sidecars(self.report_dir)
        self.assertEqual(anomalies, [])
        self.assertEqual(driver.summarize(entries)["compiled"], 1)
        self.assertEqual(driver.summarize_cases(entries)["cases"], 0)
        report = driver.format_report(
            entries,
            anomalies,
            driver.summarize(entries),
            driver.summarize_cases(entries),
        )
        self.assertNotIn("Legacy cases", report)
        self.assertIn("| conv_fprop | RUN | ran-clean | - | - |", report)

    def test_cases_are_counted_apart_from_translation_units(self):
        self._write_mixed_corpus()
        entries, _ = driver.load_sidecars(self.report_dir)
        counts = driver.summarize(entries)
        # Two translation units; the expected crash in case b must not appear as a
        # translation unit that ran.
        self.assertEqual(counts["compiled"], 2)
        self.assertEqual(counts["ran"], 1)
        self.assertEqual(counts["ran-clean"], 1)
        self.assertEqual(
            driver.summarize_cases(entries),
            {"cases": 4, "expected": 3, "unexpected": 1, "vacuous": 1},
        )

    def test_report_shows_case_counts_and_marks_unexpected_verdicts(self):
        self._write_mixed_corpus()
        entries, anomalies = driver.load_sidecars(self.report_dir)
        report = driver.format_report(
            entries,
            anomalies,
            driver.summarize(entries),
            driver.summarize_cases(entries),
        )
        self.assertIn(
            "Legacy cases: cases 4, expected 3, unexpected 1, vacuous 1", report
        )
        self.assertIn(
            "| legacy_samples::d | RUN_CLEAN | crashed | **unexpected** | - |", report
        )
        self.assertIn(
            "| legacy_samples::b | XFAIL_RUN | crashed | expected | - |", report
        )

    def test_duplicate_tu_across_directories_is_an_anomaly(self):
        self._write("a.json", _case("a", "RUN_CLEAN", "ran-clean", "expected"))
        self._write("legacy/a.json", _case("a", "RUN_CLEAN", "ran-clean", "expected"))
        _, anomalies = driver.load_sidecars(self.report_dir)
        self.assertEqual(
            anomalies, ["legacy_samples::a: reported by both a.json and legacy/a.json"]
        )

    def test_unknown_verdict_is_an_anomaly(self):
        self._write("legacy/a.json", _case("a", "RUN_CLEAN", "ran-clean", "maybe"))
        _, anomalies = driver.load_sidecars(self.report_dir)
        self.assertEqual(anomalies, ["legacy_samples::a: unrecognized verdict 'maybe'"])


if __name__ == "__main__":
    unittest.main()
