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

import test_composable_kernel_host_asan
import test_rocprim_host_asan
import test_rocthrust_host_asan


class MathCppHostAsanTest(unittest.TestCase):
    def test_rocprim_runs_exact_host_allowlist(self):
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = Path(tmp)
            for binary in test_rocprim_host_asan.ROCPRIM_HOST_TESTS:
                (bin_dir / binary).touch()
            with (
                patch.dict(os.environ, {"THEROCK_BIN_DIR": tmp}, clear=False),
                patch.object(test_rocprim_host_asan, "require_direct_clang_asan"),
                patch.object(test_rocprim_host_asan.subprocess, "run") as run,
            ):
                self.assertEqual(test_rocprim_host_asan.main(), 0)

        self.assertEqual(
            [Path(call.args[0][0]).name for call in run.call_args_list],
            list(test_rocprim_host_asan.ROCPRIM_HOST_TESTS),
        )

    def test_rocthrust_runs_exact_host_allowlist(self):
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = Path(tmp)
            for binary in test_rocthrust_host_asan.ROCTHRUST_HOST_TESTS:
                (bin_dir / binary).touch()
            with (
                patch.dict(os.environ, {"THEROCK_BIN_DIR": tmp}, clear=False),
                patch.object(test_rocthrust_host_asan, "require_direct_clang_asan"),
                patch.object(test_rocthrust_host_asan.subprocess, "run") as run,
            ):
                self.assertEqual(test_rocthrust_host_asan.main(), 0)

        self.assertEqual(
            [Path(call.args[0][0]).name for call in run.call_args_list],
            list(test_rocthrust_host_asan.ROCTHRUST_HOST_TESTS),
        )

    def test_composable_kernel_runs_exact_host_allowlist(self):
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = Path(tmp)
            for binary in test_composable_kernel_host_asan.COMPOSABLE_KERNEL_HOST_TESTS:
                (bin_dir / binary).touch()
            with (
                patch.object(test_composable_kernel_host_asan, "_require_no_gpu_nodes"),
                patch.object(test_composable_kernel_host_asan, "_run_binary") as run,
            ):
                with patch.dict(os.environ, {"THEROCK_BIN_DIR": tmp}, clear=False):
                    self.assertEqual(test_composable_kernel_host_asan.main(), 0)

        self.assertEqual(
            [call.args[0].name for call in run.call_args_list],
            list(test_composable_kernel_host_asan.COMPOSABLE_KERNEL_HOST_TESTS),
        )

    def test_composable_kernel_inventory_totals_409(self):
        inventories = (
            test_composable_kernel_host_asan.COMPOSABLE_KERNEL_HOST_TESTS.values()
        )
        self.assertEqual(sum(count for count, _ in inventories), 409)

    def test_composable_kernel_inventory_mismatch_fails(self):
        with self.assertRaisesRegex(RuntimeError, "inventory changed"):
            test_composable_kernel_host_asan._validate_inventory(
                Path("unit_sequence"), ["Suite.test"], 1, "0" * 64
            )

    def test_composable_kernel_parser_handles_parameter_comments(self):
        output = """Suite.
  one
Typed/Parameterized.
  two/0  # GetParam() = value
"""
        self.assertEqual(
            test_composable_kernel_host_asan._parse_gtest_names(output),
            ["Suite.one", "Typed/Parameterized.two/0"],
        )

    def test_composable_kernel_uses_shared_instrumentation_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            executable = Path(tmp) / "test"
            executable.touch()
            with patch.object(
                test_composable_kernel_host_asan,
                "require_direct_clang_asan",
            ) as require_direct:
                test_composable_kernel_host_asan._require_direct_asan(executable, {})
        require_direct.assert_called_once_with(executable, {})

    def test_composable_kernel_rejects_exposed_gpu_nodes(self):
        with patch.object(Path, "exists", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "unexpectedly exposes GPU"):
                test_composable_kernel_host_asan._require_no_gpu_nodes()

    def test_composable_kernel_environment_forbids_preload_and_enables_lsan(self):
        with patch.dict(
            os.environ,
            {"LD_PRELOAD": "/tmp/not-allowed.so", "ASAN_OPTIONS": "existing=1"},
            clear=False,
        ):
            env = test_composable_kernel_host_asan._test_environment()
        self.assertNotIn("LD_PRELOAD", env)
        self.assertIn("detect_leaks=1", env["ASAN_OPTIONS"])
        self.assertIn("halt_on_error=1", env["ASAN_OPTIONS"])
        self.assertIn("exitcode=23", env["LSAN_OPTIONS"])


if __name__ == "__main__":
    unittest.main()
