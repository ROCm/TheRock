#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
from teatime import format_returncode


TEATIME = Path(__file__).parent.parent / "teatime.py"


class TeatimeDiagnosticsTest(unittest.TestCase):
    def _run(self, returncode: int) -> tuple[subprocess.CompletedProcess[str], str]:
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "teatime.log"
            result = subprocess.run(
                [
                    sys.executable,
                    TEATIME,
                    "--log-timestamps",
                    "--diagnostics",
                    "--no-interactive",
                    "--label",
                    "test",
                    log_path,
                    "--",
                    sys.executable,
                    "-c",
                    f"raise SystemExit({returncode})",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            return result, log_path.read_text()

    def test_success_records_diagnostics_without_console_output(self):
        result, log = self._run(0)

        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("TEATIME_PROCESS", log)
        self.assertIn("TEATIME_RESOURCE phase=start", log)
        self.assertIn("TEATIME_RESULT decimal=0 hex=0x00000000", log)
        self.assertRegex(log, r"END\t[^\t]+\t[^\t]+\t0\n$")

    def test_failure_reports_diagnostics_and_summary_to_console(self):
        result, log = self._run(3)

        self.assertEqual(result.returncode, 3)
        self.assertIn("TEATIME_RESULT decimal=3 hex=0x00000003", result.stdout)
        self.assertIn("[test FAILED WITH CODE 3", result.stdout)
        self.assertIn("TEATIME_RESULT decimal=3 hex=0x00000003", log)
        self.assertRegex(log, r"END\t[^\t]+\t[^\t]+\t3\n$")

    def test_windows_status_is_preserved(self):
        self.assertEqual(
            format_returncode(-1073741819),
            "decimal=-1073741819 hex=0xC0000005 status=STATUS_ACCESS_VIOLATION",
        )
        self.assertEqual(
            format_returncode(3221356611),
            "decimal=3221356611 hex=0xC0020043 status=RPC_NT_INTERNAL_ERROR",
        )

    def test_diagnostic_path_records_before_and_after_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            watched_path = Path(temp_dir) / "created.txt"
            log_path = Path(temp_dir) / "teatime.log"
            result = subprocess.run(
                [
                    sys.executable,
                    TEATIME,
                    "--log-timestamps",
                    "--diagnostics",
                    "--diagnostic-path",
                    watched_path,
                    "--no-interactive",
                    log_path,
                    "--",
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(watched_path)!r}).touch()",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            log = log_path.read_text()

        self.assertEqual(result.returncode, 0)
        self.assertRegex(log, r"TEATIME_PATH phase=start .* exists=False")
        self.assertRegex(log, r"TEATIME_PATH phase=end .* exists=True")


if __name__ == "__main__":
    unittest.main()
