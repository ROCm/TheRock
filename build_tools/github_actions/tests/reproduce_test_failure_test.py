# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import contextlib
import io
import os
import shlex
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import reproduce_test_failure

CI_ARGV = [
    "reproduce_test_failure.py",
    "--run-id",
    "12345678",
    "--repository",
    "ROCm/TheRock",
    "--amdgpu-family",
    "gfx94X-dcgpu",
    "--amdgpu-targets",
    "gfx942",
    "--test-script",
    "python build_tools/github_actions/test_executable_scripts/test_runner.py",
    "--test-component",
    "rocblas",
    "--shard-index",
    "2",
    "--total-shards",
    "4",
]


def _run_main(argv):
    """Run main() with argv; return (stdout, args passed to run_linux)."""
    captured = {}

    def fake_run_linux(args):
        captured["args"] = args
        return 0

    stdout = io.StringIO()
    with mock.patch.object(sys, "argv", argv), mock.patch.object(
        reproduce_test_failure, "is_windows", False
    ), mock.patch.object(
        reproduce_test_failure, "run_linux", fake_run_linux
    ), contextlib.redirect_stdout(
        stdout
    ):
        reproduce_test_failure.main()
    return stdout.getvalue(), captured.get("args")


def _parse(argv):
    return _run_main(argv)[1]


class ReproductionCommandTest(unittest.TestCase):
    def test_printed_command_round_trips_test_component(self):
        printed, _ = _run_main(CI_ARGV + ["--print-cmd"])
        command = printed.strip().splitlines()[-1].strip()

        self.assertIn('--test-component "rocblas"', command)
        argv = shlex.split(command)[1:]  # drop "python"
        args = _parse(argv)
        self.assertEqual(args.test_component, "rocblas")
        self.assertEqual(args.amdgpu_targets, "gfx942")

    def test_test_component_is_optional(self):
        argv = CI_ARGV[: CI_ARGV.index("--test-component")]
        args = _parse(argv)

        self.assertEqual(args.test_component, "")
        self.assertNotIn(
            "--test-component",
            reproduce_test_failure.build_reproduction_command(args),
        )


class CiTestEnvironmentTest(unittest.TestCase):
    def test_sets_what_the_ci_test_step_sets(self):
        env = dict(reproduce_test_failure.ci_test_environment(_parse(CI_ARGV)))

        self.assertEqual(env["TEST_COMPONENT"], "rocblas")
        self.assertEqual(env["AMDGPU_FAMILIES"], "gfx94X-dcgpu")
        self.assertEqual(env["AMDGPU_TARGETS"], "gfx942")
        self.assertEqual(env["SHARD_INDEX"], "2")
        self.assertEqual(env["TOTAL_SHARDS"], "4")
        self.assertEqual(env["THEROCK_BIN_DIR"], "build/bin")

    def test_omits_unset_optional_values(self):
        argv = (
            CI_ARGV[: CI_ARGV.index("--amdgpu-targets")]
            + CI_ARGV[
                CI_ARGV.index("--test-script") : CI_ARGV.index("--test-component")
            ]
        )
        env = dict(reproduce_test_failure.ci_test_environment(_parse(argv)))

        self.assertNotIn("TEST_COMPONENT", env)
        self.assertNotIn("AMDGPU_TARGETS", env)
        self.assertEqual(env["AMDGPU_FAMILIES"], "gfx94X-dcgpu")

    def test_linux_container_script_exports_test_component(self):
        args = _parse(CI_ARGV)
        with mock.patch.object(
            reproduce_test_failure, "check_docker", return_value=True
        ), mock.patch.object(reproduce_test_failure.subprocess, "run") as run:
            run.return_value.returncode = 0
            reproduce_test_failure.run_linux(args)

        script = run.call_args.args[0][-1]
        self.assertIn('export TEST_COMPONENT="rocblas"', script)
        self.assertIn('export AMDGPU_FAMILIES="gfx94X-dcgpu"', script)
        self.assertIn('export AMDGPU_TARGETS="gfx942"', script)


if __name__ == "__main__":
    unittest.main()
