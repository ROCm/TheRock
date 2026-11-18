# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for run recording in run_pytorch_tests_full.py."""

import json
import os
import platform
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).resolve().parent.parent))

import run_pytorch_tests_full as runner
import summarize_test_results as summarize


class RecordedRunTest(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.temp = Path(self._temp_dir.name)
        self.reports_dir = self.temp / "test-reports"

    def tearDown(self):
        self._temp_dir.cleanup()

    def test_started_run_is_incomplete_until_finished(self):
        output_log = runner.start_recorded_run(self.reports_dir, "core", "gfx94X")
        output_log.write_text("test_nn 1/1 was successful\n")

        self.assertIn(
            "the core test runner did not exit",
            summarize.check_report_set(self.reports_dir),
        )

        runner.finish_recorded_run(self.reports_dir, "core")

        self.assertNotIn(
            "the core test runner did not exit",
            summarize.check_report_set(self.reports_dir),
        )

    def test_start_writes_report_metadata(self):
        runner.start_recorded_run(self.reports_dir, "core", "gfx94X")

        metadata = json.loads(
            (self.reports_dir / summarize.REPORT_METADATA).read_text()
        )
        self.assertEqual(metadata, {"os": platform.system(), "amdgpu_family": "gfx94X"})
        self.assertEqual(
            summarize.find_report_sets(self.reports_dir),
            [(platform.system(), "gfx94X", self.reports_dir)],
        )

    def test_restart_truncates_previous_output(self):
        output_log = runner.start_recorded_run(self.reports_dir, "core", "gfx94X")
        output_log.write_text("stale output\n")

        output_log = runner.start_recorded_run(self.reports_dir, "core", "gfx94X")

        self.assertEqual(output_log.read_text(), "")

    def test_run_command_records_output_and_return_code(self):
        output_log = runner.start_recorded_run(self.reports_dir, "core", "gfx94X")
        script = (
            "import sys; print('test_cuda 1/1 failed!'); "
            "print('Received signal: SIGSEGV', file=sys.stderr); sys.exit(3)"
        )

        return_code = runner.run_command(
            [sys.executable, "-c", script], self.temp, output_log
        )

        self.assertEqual(return_code, 3)
        output = output_log.read_text()
        self.assertIn("test_cuda 1/1 failed!", output)
        self.assertIn("Received signal: SIGSEGV", output)

    def test_run_command_appends_across_invocations(self):
        # The inductor config runs run_test.py twice into one log.
        output_log = runner.start_recorded_run(self.reports_dir, "full", "gfx94X")

        for text in ("first", "second"):
            runner.run_command(
                [sys.executable, "-c", f"print('{text}')"], self.temp, output_log
            )

        self.assertEqual(output_log.read_text().split(), ["first", "second"])


if __name__ == "__main__":
    unittest.main()
