#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Ensure real CSAN report formats cannot silently pass a component job."""

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from check_csan_reports import main, scan_csan_reports


class CheckCsanReportsTest(unittest.TestCase):
    def test_host_gpu_and_disabled_detection_reports_fail_and_remain_available(self):
        reports = (
            "WARNING: ConcurrencySanitizer: data race (pid=123)\n",
            "\x1b[1;31mWARNING: ConcurrencySanitizer: intra-wave data race\x1b[0m\n",
            "WARNING: ConcurrencySanitizer: data race of unknown origin\n",
            "ERROR: ConcurrencySanitizer: cannot initialize HSA offload support\n",
            "WARNING: ConcurrencySanitizer: the runtime is loaded too late to "
            "intercept HSA, GPU races will not be reported. Link the runtime first "
            "or use LD_PRELOAD.\n",
            "SUMMARY: ConcurrencySanitizer: data race example.cpp:12\n",
            "ConcurrencySanitizer: CHECK failed: csan.cpp:12\n",
            "ConcurrencySanitizer:DEADLYSIGNAL\n",
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for index, text in enumerate(reports):
                with self.subTest(report=text):
                    path = root / f"csan.{index}"
                    path.write_text(
                        "initialization information\n" + text, encoding="utf-8"
                    )
            findings, report_count = scan_csan_reports(root)
            self.assertEqual(report_count, len(reports))
            self.assertEqual(len(findings), len(reports))
            self.assertTrue(all(finding.line_number == 2 for finding in findings))
            self.assertTrue(all("\x1b" not in finding.message for finding in findings))
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                self.assertEqual(main(["--reports-dir", str(root)]), 1)
            self.assertIn("csan.0:2:", stderr.getvalue())
            for index, text in enumerate(reports):
                self.assertEqual(
                    (root / f"csan.{index}").read_text(encoding="utf-8"),
                    "initialization information\n" + text,
                )

    def test_benign_initialization_and_other_tools_are_not_findings(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "csan.program.123").write_text(
                "ConcurrencySanitizer initialized\n"
                "CSAN_OPTIONS=halt_on_error=1\n"
                "WARNING: another tool: benign warning\n"
                "SUMMARY: 0 tests failed\n"
                "INFO: ConcurrencySanitizer: no diagnostics\n",
                encoding="utf-8",
            )
            findings, report_count = scan_csan_reports(root)
            self.assertEqual(report_count, 1)
            self.assertEqual(findings, [])
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--reports-dir", str(root)]), 0)

    def test_clean_run_may_produce_no_reports(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(scan_csan_reports(Path(temp)), ([], 0))
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--reports-dir", temp]), 0)

    def test_missing_directory_fails_instead_of_masking_missing_setup(self):
        with tempfile.TemporaryDirectory() as temp:
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                self.assertEqual(main(["--reports-dir", str(Path(temp) / "absent")]), 1)
            self.assertIn("configure_csan_env.py", stderr.getvalue())

    def test_per_process_nested_reports_and_non_utf8_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            nested = root / "nested"
            nested.mkdir()
            (nested / "csan.123").write_bytes(
                b"\xff\nWARNING: ConcurrencySanitizer: data race\n"
            )
            findings, report_count = scan_csan_reports(root)
            self.assertEqual(report_count, 1)
            self.assertEqual(len(findings), 1)
            self.assertEqual(findings[0].line_number, 2)


if __name__ == "__main__":
    unittest.main()
