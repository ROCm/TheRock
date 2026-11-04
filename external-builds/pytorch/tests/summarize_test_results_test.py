# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import summarize_test_results as m

LINUX_XML = """<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="pytest">
    <testcase classname="test_nn.TestNN" name="test_pass" time="0.1"/>
    <testcase classname="test_nn.TestNN" name="test_fail" time="0.1">
      <failure message="AssertionError: mismatch">trace</failure>
    </testcase>
    <testcase classname="test_nn.TestNN" name="test_error" time="0.1">
      <error message="RuntimeError: HIP error">trace</error>
    </testcase>
    <testcase classname="test_nn.TestNN" name="test_skip_a" time="0">
      <skipped message="skipIfRocm: not supported"/>
    </testcase>
    <testcase classname="test_nn.TestNN" name="test_skip_b" time="0">
      <skipped message="skipIfRocm: not supported"/>
    </testcase>
    <testcase classname="test_nn.TestNN" name="test_skip_c" time="0">
      <skipped>requires multiple GPUs</skipped>
    </testcase>
    <testcase classname="test_nn.TestNN" name="test_rerun_pass" time="0.2">
      <rerun message="flaky timeout"/>
    </testcase>
    <testcase classname="test_nn.TestNN" name="test_retry" time="0.1">
      <failure message="first attempt"/>
    </testcase>
  </testsuite>
</testsuites>
"""

# A later attempt of the same test file, as written by a rerun.
LINUX_RETRY_XML = """<?xml version="1.0" encoding="utf-8"?>
<testsuite name="pytest">
  <testcase classname="test_nn.TestNN" name="test_retry" time="0.1"/>
</testsuite>
"""

WINDOWS_XML = """<?xml version="1.0" encoding="utf-8"?>
<testsuite name="pytest">
  <testcase classname="test_torch.TestTorch" name="test_ok" time="0.1"/>
  <testcase classname="test_torch.TestTorch" name="test_ok_2" time="0.1"/>
</testsuite>
"""


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


class StandardSummaryTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        linux = self.root / "test-reports-Linux-gfx94X-dcgpu" / "python-pytest"
        _write(linux / "test_nn" / "test_nn-1.xml", LINUX_XML)
        _write(linux / "test_nn" / "test_nn-2.xml", LINUX_RETRY_XML)
        windows = self.root / "test-reports-Windows-gfx120X-all" / "python-pytest"
        _write(windows / "test_torch" / "test_torch-1.xml", WINDOWS_XML)
        # Full-suite artifacts use a different naming scheme and are ignored.
        _write(self.root / "test-reports-default-1-2" / "x.xml", WINDOWS_XML)
        self.summary = self.root / "summary.md"

    def _summarize(self) -> tuple[int, str]:
        failures = m.write_standard_summary(self.root, str(self.summary))
        return failures, self.summary.read_text()

    def test_counts_merge_reruns(self):
        outcomes = m.collect_outcomes(self.root / "test-reports-Linux-gfx94X-dcgpu")
        counts = m.count_outcomes(outcomes)
        self.assertEqual(
            counts,
            {
                "RUN": 5,
                "PASSED": 3,
                "FAILED": 1,
                "ERROR": 1,
                "SKIPPED": 3,
                "FLAKY": 2,
            },
        )

    def test_summary_table_has_a_row_per_platform_and_a_total(self):
        failures, text = self._summarize()
        self.assertEqual(failures, 2)
        self.assertIn("| Linux | gfx94X-dcgpu | 5 | 3 | 1 | 1 | 3 | 2 |", text)
        self.assertIn("| Windows | gfx120X-all | 2 | 2 | 0 | 0 | 0 | 0 |", text)
        self.assertIn(
            "| **Total** | | **7** | **5** | **1** | **1** | **3** | **2** |", text
        )
        self.assertNotIn("default", text)

    def test_failures_and_skip_reasons_are_listed(self):
        _, text = self._summarize()
        self.assertIn("### 2 failed — Linux | gfx94X-dcgpu", text)
        self.assertIn(
            "| test_nn | TestNN | test_fail | FAILED | AssertionError: mismatch |", text
        )
        self.assertNotIn("test_retry | FAILED", text)
        self.assertIn("3 skipped — Linux | gfx94X-dcgpu (2 reasons)", text)
        self.assertIn("| 2 | skipIfRocm: not supported |", text)
        self.assertIn("| 1 | requires multiple GPUs |", text)

    def test_single_artifact_download_is_found_by_metadata(self):
        # download-artifact extracts a lone match directly into its path.
        with tempfile.TemporaryDirectory() as tmp:
            single = Path(tmp)
            reports = single / "python-pytest" / "test_nn"
            _write(reports / "test_nn-1.xml", LINUX_XML)
            _write(reports / "test_nn-2.xml", LINUX_RETRY_XML)
            _write(
                single / m.REPORT_METADATA,
                '{"os": "Linux", "amdgpu_family": "gfx120X-all"}\n',
            )
            summary = single / "summary.md"
            self.assertEqual(m.write_standard_summary(single, str(summary)), 2)
            text = summary.read_text()
        self.assertIn("| Linux | gfx120X-all | 5 | 3 | 1 | 1 | 3 | 2 |", text)
        self.assertNotIn("**Total**", text)

    def test_metadata_overrides_artifact_directory_names(self):
        _write(
            self.root / "test-reports-Linux-gfx94X-dcgpu" / m.REPORT_METADATA,
            '{"os": "Linux", "amdgpu_family": "gfx942"}\n',
        )
        _, text = self._summarize()
        self.assertIn("| Linux | gfx942 | 5 |", text)
        self.assertNotIn("| Linux | gfx94X-dcgpu |", text)
        self.assertIn("| Windows | gfx120X-all | 2 |", text)

    def test_no_reports_is_explained(self):
        with tempfile.TemporaryDirectory() as empty:
            summary = Path(empty) / "summary.md"
            self.assertEqual(m.write_standard_summary(Path(empty), str(summary)), 0)
            self.assertIn("No PyTorch test reports were uploaded", summary.read_text())

    def _count(self, reports: Path) -> list[str]:
        output = self.root / "github_output"
        output.unlink(missing_ok=True)
        m.write_count_outputs(reports, str(output))
        return output.read_text().splitlines()

    def test_count_outputs_are_written_for_quartz(self):
        reports = self.root / "test-reports-Linux-gfx94X-dcgpu"
        (reports / m.COMPLETION_MARKER).touch()
        self.assertEqual(
            self._count(reports),
            [
                "run_test_count=5",
                "passed_test_count=3",
                "failed_test_count=2",
                "skipped_test_count=3",
                "results_complete=true",
            ],
        )

    def test_counts_without_completion_marker_are_incomplete(self):
        lines = self._count(self.root / "test-reports-Linux-gfx94X-dcgpu")
        self.assertIn("failed_test_count=2", lines)
        self.assertEqual(lines[-1], "results_complete=false")

    def test_unparseable_report_is_incomplete(self):
        reports = self.root / "test-reports-Linux-gfx94X-dcgpu"
        (reports / m.COMPLETION_MARKER).touch()
        _write(reports / "python-pytest" / "test_nn" / "broken.xml", "<testsuite")
        self.assertEqual(self._count(reports)[-1], "results_complete=false")

    def test_no_reports_only_writes_incomplete(self):
        self.assertEqual(self._count(self.root / "missing"), ["results_complete=false"])

    def test_failure_only_parser_is_unchanged(self):
        xml = (
            self.root
            / "test-reports-Linux-gfx94X-dcgpu/python-pytest/test_nn/test_nn-1.xml"
        )
        statuses = [f["status"] for f in m.parse_junit_xml(xml)]
        self.assertEqual(statuses, ["FAILED", "ERROR", "FAILED"])


if __name__ == "__main__":
    unittest.main()
