# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for combined PyTorch JUnit result summaries."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).resolve().parent.parent))

from summarize_test_results import write_combined_summary


class SummarizeTestResultsTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.artifacts = self.root / "artifacts"
        self.summary = self.root / "summary.md"
        self.outputs = self.root / "outputs.txt"

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_report(self, artifact_name: str, testcases: str) -> None:
        report_dir = self.artifacts / artifact_name / "python-pytest" / "module.tests"
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "results.xml").write_text(f"<testsuite>{testcases}</testsuite>")

    def _summarize(self, expected_report_count: int) -> tuple[dict, str, dict]:
        result = write_combined_summary(
            self.artifacts,
            "gfx94X-dcgpu",
            os.fspath(self.summary),
            expected_report_count=expected_report_count,
            github_output=os.fspath(self.outputs),
        )
        output_values = dict(
            line.split("=", 1) for line in self.outputs.read_text().splitlines()
        )
        return result, self.summary.read_text(), output_values

    def test_deduplicates_failed_and_error_copies_of_same_case(self):
        self._write_report(
            "test-reports-default-1-1",
            """
            <testcase classname="pkg.TestCase" name="test_duplicate">
              <failure message="first failure"/>
            </testcase>
            <testcase classname="pkg.TestCase" name="test_duplicate">
              <error message="rerun error"/>
            </testcase>
            """,
        )

        result, summary, outputs = self._summarize(expected_report_count=1)

        self.assertEqual(result["failed_test_count"], 1)
        self.assertIn("### 1 failures across all shards", summary)
        self.assertEqual(outputs["failed_test_count"], "1")

    def test_zero_failures_is_complete_when_report_is_present(self):
        self._write_report(
            "test-reports-default-1-1",
            '<testcase classname="pkg.TestCase" name="test_passes"/>',
        )

        result, summary, outputs = self._summarize(expected_report_count=1)

        self.assertEqual(result["failed_test_count"], 0)
        self.assertTrue(result["results_complete"])
        self.assertIn("### All tests passed", summary)
        self.assertEqual(outputs["results_complete"], "true")

    def test_counts_distinct_failed_and_error_cases(self):
        self._write_report(
            "test-reports-default-1-1",
            """
            <testcase classname="pkg.TestCase" name="test_failure">
              <failure message="assertion failed"/>
            </testcase>
            <testcase classname="pkg.TestCase" name="test_error">
              <error message="setup crashed"/>
            </testcase>
            """,
        )

        result, summary, outputs = self._summarize(expected_report_count=1)

        self.assertEqual(result["failed_test_count"], 2)
        self.assertIn("| FAILED |", summary)
        self.assertIn("| ERROR |", summary)
        self.assertEqual(outputs["observed_report_count"], "1")
        self.assertEqual(outputs["expected_report_count"], "1")

    def test_incomplete_reports_do_not_look_like_a_complete_zero(self):
        self._write_report(
            "test-reports-default-1-2",
            '<testcase classname="pkg.TestCase" name="test_passes"/>',
        )

        result, summary, outputs = self._summarize(expected_report_count=2)

        self.assertFalse(result["results_complete"])
        self.assertIn("1 observed / 2 expected; 1 valid (incomplete)", summary)
        self.assertIn("No failed tests found in available reports", summary)
        self.assertNotIn("All tests passed", summary)
        self.assertEqual(outputs["results_complete"], "false")

    def test_missing_artifact_directory_emits_incomplete_outputs(self):
        result, summary, outputs = self._summarize(expected_report_count=1)

        self.assertEqual(result["observed_report_count"], 0)
        self.assertFalse(result["results_complete"])
        self.assertIn("0 observed / 1 expected; 0 valid (incomplete)", summary)
        self.assertEqual(outputs["failed_test_count"], "0")
        self.assertEqual(outputs["results_complete"], "false")

    def test_empty_artifact_is_incomplete(self):
        (self.artifacts / "test-reports-default-1-1").mkdir(parents=True)

        result, summary, outputs = self._summarize(expected_report_count=1)

        self.assertEqual(result["observed_report_count"], 1)
        self.assertFalse(result["results_complete"])
        self.assertIn("1 observed / 1 expected; 0 valid (incomplete)", summary)
        self.assertEqual(outputs["results_complete"], "false")

    def test_malformed_junit_is_incomplete(self):
        self._write_report("test-reports-default-1-1", "<testcase>")

        result, summary, outputs = self._summarize(expected_report_count=1)

        self.assertEqual(result["observed_report_count"], 1)
        self.assertFalse(result["results_complete"])
        self.assertIn("1 observed / 1 expected; 0 valid (incomplete)", summary)
        self.assertEqual(outputs["results_complete"], "false")

    def test_non_junit_xml_is_incomplete(self):
        report_dir = self.artifacts / "test-reports-default-1-1" / "python-pytest"
        report_dir.mkdir(parents=True)
        (report_dir / "results.xml").write_text("<metadata/>")

        result, summary, outputs = self._summarize(expected_report_count=1)

        self.assertEqual(result["observed_report_count"], 1)
        self.assertFalse(result["results_complete"])
        self.assertIn("1 observed / 1 expected; 0 valid (incomplete)", summary)
        self.assertEqual(outputs["results_complete"], "false")


if __name__ == "__main__":
    unittest.main()
