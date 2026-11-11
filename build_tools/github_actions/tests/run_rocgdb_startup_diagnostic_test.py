# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import io
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_rocgdb_startup_diagnostic import main


class StartupDiagnosticTest(unittest.TestCase):
    def run_helper(self, exit_code, *, outside=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sdk = root / "sdk"
            tests = sdk / "tests/rocgdb"
            suite = tests / "gdb/testsuite"
            suite.mkdir(parents=True)
            inferior = (root if outside else suite) / "inferior"
            inferior.touch()
            log_line = "Starting program: " + str(inferior) + " \n"
            (tests / "test_rocgdb.py").write_text(
                "from pathlib import Path\n"
                f"Path({str(suite / 'gdb.log')!r}).write_text("
                f"{log_line!r})\n"
                f"raise SystemExit({exit_code})\n"
            )
            with patch("run_rocgdb_startup_diagnostic.capture_startup") as capture:
                with redirect_stdout(io.StringIO()):
                    result = main(
                        ["--artifacts-dir", str(sdk), "--output", str(root / "logs")]
                    )
                return result, capture.call_args_list, inferior

    def test_success_never_launches_an_additional_inferior(self):
        result, calls, _ = self.run_helper(0)
        self.assertEqual(result, 0)
        self.assertEqual(calls, [])

    def test_capture_preserves_original_failure_status(self):
        result, calls, inferior = self.run_helper(17)
        self.assertEqual(result, 17)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].args[1], inferior)

    def test_log_cannot_launch_a_file_outside_the_test_suite(self):
        result, calls, _ = self.run_helper(17, outside=True)
        self.assertEqual(result, 17)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
