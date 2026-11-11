#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise diagnostic preservation with an actual, GPU-free child runner."""

from contextlib import redirect_stdout, redirect_stderr
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_debug_agent_tests import main


class DebugAgentRunnerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.artifacts = self.root / "artifacts"
        self.test_dir = self.artifacts / "tests" / "rocm-debug-agent"
        self.test_dir.mkdir(parents=True)
        self.diagnostics = self.root / "diagnostics"

    def run_fake(self, exit_code):
        (self.test_dir / "run-test.py").write_text(
            "import os, pathlib, sys\n"
            "assert pathlib.Path(sys.argv[1]) == pathlib.Path.cwd()\n"
            "assert os.environ['DEBUG_AGENT_PROBE'] == 'preserved'\n"
            "with open('calls', 'a') as calls: calls.write('called\\n')\n"
            "pathlib.Path('run-test.log').write_text('child memory-violation detail\\n')\n"
            f"sys.exit({exit_code})\n",
            encoding="utf-8",
        )
        output = io.StringIO()
        with patch.dict(
            os.environ, {"DEBUG_AGENT_PROBE": "preserved"}
        ), redirect_stdout(output), patch(
            "run_debug_agent_tests.subprocess.run", wraps=subprocess.run
        ) as run:
            result = main(
                [
                    "--artifacts-dir",
                    str(self.artifacts),
                    "--diagnostics-dir",
                    str(self.diagnostics),
                ]
            )
        child_env = run.call_args.kwargs["env"]
        self.assertEqual(child_env.get("LD_PRELOAD"), os.environ.get("LD_PRELOAD"))
        self.assertTrue(child_env["PATH"].startswith(str(self.artifacts / "bin")))
        self.assertTrue(
            child_env["LD_LIBRARY_PATH"].startswith(str(self.artifacts / "lib"))
        )
        self.assertEqual((self.test_dir / "calls").read_text(), "called\n")
        self.assertEqual(
            (self.diagnostics / "run-test.log").read_text(),
            "child memory-violation detail\n",
        )
        return result, output.getvalue()

    def test_failure_runs_once_preserves_and_echoes_log_and_exit_code(self):
        result, output = self.run_fake(7)
        self.assertEqual(result, 7)
        self.assertIn("child memory-violation detail", output)

    def test_success_preserves_log_without_failure_dump(self):
        result, output = self.run_fake(0)
        self.assertEqual(result, 0)
        self.assertNotIn("child memory-violation detail", output)

    def test_missing_runner_fails_with_actionable_path(self):
        output = io.StringIO()
        with redirect_stderr(output):
            result = main(
                [
                    "--artifacts-dir",
                    str(self.artifacts),
                    "--diagnostics-dir",
                    str(self.diagnostics),
                ]
            )
        self.assertEqual(result, 1)
        self.assertIn(str(self.test_dir / "run-test.py"), output.getvalue())


if __name__ == "__main__":
    unittest.main()
