# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

WRAPPER = (
    Path(__file__).resolve().parent.parent
    / "test_executable_scripts"
    / "retry_test_command.py"
)


class RetryTestCommandTest(unittest.TestCase):
    def run_command(self, command, *, retries=1, enabled="true"):
        return subprocess.run(
            [
                sys.executable,
                str(WRAPPER),
                "--retry-count",
                str(retries),
                "--retry-delay",
                "0",
                "--command",
                command,
            ],
            env={**os.environ, "RETRY_THIS_STEP": enabled},
            capture_output=True,
            text=True,
        )

    def test_success_runs_once(self):
        result = self.run_command("echo passed")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.count("passed"), 1)

    def test_transient_failure_retries_complete_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = shlex.quote(str(Path(directory) / "first attempt"))
            command = (
                f"TEST_CATEGORY='hw common' bash -c 'test \"$TEST_CATEGORY\" = \"hw common\"' "
                f"&& {{ if [ -e {marker} ]; then echo recovered; "
                f"else touch {marker}; exit 9; fi; }}"
            )
            result = self.run_command(command)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("attempt 2/2", result.stdout)
            self.assertIn("recovered", result.stdout)

    def test_persistent_failure_returns_last_exit_code(self):
        result = self.run_command("exit 9", retries=2)
        self.assertEqual(result.returncode, 9)
        self.assertEqual(result.stdout.count("attempt"), 3)

    def test_retries_can_be_disabled(self):
        for options in ({"retries": 0}, {"enabled": "false"}):
            with self.subTest(options=options):
                result = self.run_command("exit 9", **options)
                self.assertEqual(result.returncode, 9)
                self.assertEqual(result.stdout.count("attempt"), 1)

    def test_pipeline_failure_is_preserved(self):
        result = self.run_command("false | true", retries=0)
        self.assertEqual(result.returncode, 1)

    def test_interruption_is_not_retried(self):
        result = self.run_command("exit 130")
        self.assertEqual(result.returncode, 130)
        self.assertEqual(result.stdout.count("attempt"), 1)

    def test_invalid_policy_rejected_before_launch(self):
        result = self.run_command("echo launched", retries=-1)
        self.assertEqual(result.returncode, 2)
        self.assertNotIn("launched", result.stdout)


if __name__ == "__main__":
    unittest.main()
