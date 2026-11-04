# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import retry_test_command as retry


class RetryTestCommandTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.summary = patch.object(retry, "gha_append_step_summary")
        self.summary.start()

    def tearDown(self) -> None:
        self.summary.stop()
        self.temp_dir.cleanup()

    def execute(self, script: str, *, retries: int = 3, timeout: float = 5):
        return retry.retry_command(
            script,
            max_retries=retries,
            timeout_seconds=timeout,
            retry_delay_seconds=0,
            logs_dir=self.root / "logs",
        )

    def test_success_does_not_retry(self) -> None:
        attempts = self.execute("echo success")
        self.assertEqual([item.exit_code for item in attempts], [0])
        self.assertEqual((self.root / "logs/attempt-1.log").read_text(), "success\n")

    def test_failure_then_success_stops_early(self) -> None:
        with patch.object(
            retry, "run_attempt", side_effect=[(1, False), (0, False)]
        ) as run:
            attempts = self.execute("unused", retries=5)
        self.assertEqual(run.call_count, 2)
        self.assertEqual([item.exit_code for item in attempts], [1, 0])
        saved = json.loads((self.root / "logs/attempts.json").read_text())
        self.assertEqual([item["exit_code"] for item in saved], [1, 0])

    def test_real_command_reuses_environment_after_failure(self) -> None:
        with patch.dict(os.environ, {"RETRY_TEST_MARKER": str(self.root / "marker")}):
            attempts = self.execute(
                'if test -e "$RETRY_TEST_MARKER"; then echo recovered; '
                'else printf failed > "$RETRY_TEST_MARKER"; echo initial-failure; exit 1; fi'
            )
        self.assertEqual([item.exit_code for item in attempts], [1, 0])
        self.assertIn("initial-failure", (self.root / "logs/attempt-1.log").read_text())
        self.assertIn("recovered", (self.root / "logs/attempt-2.log").read_text())

    def test_retry_budget_is_additional_attempts(self) -> None:
        attempts = self.execute("echo failed; exit 7", retries=2)
        self.assertEqual([item.exit_code for item in attempts], [7, 7, 7])

    def test_zero_disables_retries(self) -> None:
        attempts = self.execute("exit 9", retries=0)
        self.assertEqual([item.exit_code for item in attempts], [9])

    def test_shell_chains_and_shard_environment_are_preserved(self) -> None:
        with patch.dict(
            os.environ,
            {"SHARD_INDEX": "2", "TOTAL_SHARDS": "4", "TEST_TYPE": "standard"},
        ):
            attempts = self.execute(
                "VALUE=ready bash -c 'echo \"$VALUE $SHARD_INDEX/$TOTAL_SHARDS $TEST_TYPE\"' "
                '&& echo "$RETRY_THIS_STEP $RETRY_COUNT"'
            )
        self.assertEqual(attempts[0].exit_code, 0)
        self.assertEqual(
            (self.root / "logs/attempt-1.log").read_text(),
            "ready 2/4 standard\nfalse 0\n",
        )

    def test_shell_failure_is_not_hidden_by_later_command(self) -> None:
        attempts = self.execute("false\necho should-not-run", retries=0)
        self.assertNotEqual(attempts[0].exit_code, 0)
        self.assertNotIn(
            "should-not-run", (self.root / "logs/attempt-1.log").read_text()
        )

    def test_attempt_timeout_is_retryable(self) -> None:
        with patch.object(retry, "run_attempt", side_effect=[(124, True), (0, False)]):
            attempts = self.execute("unused")
        self.assertEqual([item.timed_out for item in attempts], [True, False])

    @unittest.skipUnless(os.name == "posix", "POSIX process group test")
    def test_timeout_terminates_descendants(self) -> None:
        marker = self.root / "child-survived"
        child = self.root / "child.py"
        child.write_text(
            "import time\nfrom pathlib import Path\ntime.sleep(0.5)\n"
            f"Path({str(marker)!r}).write_text('survived')\n"
        )
        script = f"{shlex.quote(sys.executable)} {shlex.quote(str(child))} & wait"
        attempts = self.execute(script, retries=0, timeout=0.1)
        self.assertTrue(attempts[0].timed_out)
        subprocess.run(
            [sys.executable, "-c", "import time; time.sleep(0.6)"], check=True
        )
        self.assertFalse(marker.exists())

    def test_cancellation_does_not_retry(self) -> None:
        with patch.object(retry, "run_attempt", side_effect=KeyboardInterrupt) as run:
            with self.assertRaises(KeyboardInterrupt):
                self.execute("unused")
        self.assertEqual(run.call_count, 1)

    def test_invalid_retry_limits_are_rejected(self) -> None:
        for value in ("-1", "1.5", "nan", "inf", "abc"):
            with self.subTest(value=value), self.assertRaises(
                argparse.ArgumentTypeError
            ):
                retry.nonnegative_integer(value)
        self.assertEqual(retry.nonnegative_integer("0"), 0)
        self.assertEqual(retry.nonnegative_integer("3.0"), 3)

    def test_retry_timeout_budget_includes_attempts_and_delays(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "retry_test_command.py",
                "--configure-only",
                "--max-retries",
                "3",
                "--timeout-minutes",
                "10",
            ],
        ), patch.object(retry, "gha_set_output") as output:
            self.assertEqual(retry.main(), 0)
        output.assert_called_once_with({"step_timeout_minutes": "42"})

    def test_retry_timeout_budget_is_capped(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "retry_test_command.py",
                "--configure-only",
                "--max-retries",
                "10",
                "--timeout-minutes",
                "120",
            ],
        ), patch.object(retry, "gha_set_output") as output:
            self.assertEqual(retry.main(), 0)
        output.assert_called_once_with({"step_timeout_minutes": "285"})


if __name__ == "__main__":
    unittest.main()
