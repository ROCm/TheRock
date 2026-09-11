# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT_DIR = Path(__file__).parent.parent / "test_executable_scripts"
sys.path.insert(0, os.fspath(SCRIPT_DIR))

import test_rand_host_asan
import test_rocsparse_host_asan
import test_stinkytofu_host_asan


class HostAsanPhase2Test(unittest.TestCase):
    def test_rocrand_exact_allowlist_and_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = Path(tmp)
            for binary, _ in test_rand_host_asan.ROCRAND_TESTS:
                (bin_dir / binary).touch()
            with (
                patch.dict(
                    os.environ,
                    {"THEROCK_BIN_DIR": tmp, "TEST_COMPONENT": "rocrand"},
                    clear=False,
                ),
                patch.object(test_rand_host_asan.subprocess, "run") as run,
            ):
                self.assertEqual(test_rand_host_asan.main(), 0)

        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(len(commands), len(test_rand_host_asan.ROCRAND_TESTS))
        generator_command = next(
            command for command in commands if "test_rocrand_generator_type" in command[0]
        )
        self.assertEqual(
            generator_command[1],
            "--gtest_filter=rocrand_generator_type_tests.rocrand_generator:"
            "rocrand_generator_type_tests.generate_test",
        )

    def test_hiprand_runs_only_linkage(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "test_hiprand_linkage").touch()
            with (
                patch.dict(
                    os.environ,
                    {"THEROCK_BIN_DIR": tmp, "TEST_COMPONENT": "hiprand"},
                    clear=False,
                ),
                patch.object(test_rand_host_asan.subprocess, "run") as run,
            ):
                self.assertEqual(test_rand_host_asan.main(), 0)

        self.assertEqual(Path(run.call_args.args[0][0]).name, "test_hiprand_linkage")

    def test_rocsparse_runs_only_gpu_independent_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "rocsparse-unit-test").touch()
            with (
                patch.dict(os.environ, {"THEROCK_BIN_DIR": tmp}, clear=False),
                patch.object(test_rocsparse_host_asan.subprocess, "run") as run,
            ):
                self.assertEqual(test_rocsparse_host_asan.main(), 0)

        self.assertEqual(Path(run.call_args.args[0][0]).name, "rocsparse-unit-test")

    def test_stinkytofu_runs_native_generator_and_filecheck_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            test_dir = Path(tmp) / "stinkytofu"
            filecheck_dir = test_dir / "filecheck"
            filecheck_dir.mkdir(parents=True)
            for name in (
                "unit_tests",
                "test_gen_instructions",
                "stinkytofu-check",
                "stinkytofu-opt",
            ):
                (test_dir / name).touch()
            (test_dir / "architectures.txt").write_text(
                "Gfx1250\nGfx1250v0\n", encoding="utf-8"
            )
            (filecheck_dir / "one.stir").touch()
            (filecheck_dir / "two.s").touch()

            with (
                patch.dict(os.environ, {"THEROCK_BIN_DIR": tmp}, clear=False),
                patch.object(test_stinkytofu_host_asan.subprocess, "run") as run,
            ):
                self.assertEqual(test_stinkytofu_host_asan.main(), 0)

        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(Path(commands[0][0]).name, "unit_tests")
        self.assertEqual(Path(commands[1][0]).name, "test_gen_instructions")
        self.assertEqual(commands[1][-2:], ["Gfx1250", "Gfx1250v0"])
        self.assertEqual(len(commands), 4)


if __name__ == "__main__":
    unittest.main()
