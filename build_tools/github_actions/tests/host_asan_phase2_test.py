# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

SCRIPT_DIR = Path(__file__).parent.parent / "test_executable_scripts"
sys.path.insert(0, os.fspath(SCRIPT_DIR))

import test_composable_kernel_host_asan
import test_rand_host_asan
import test_rocsparse_host_asan
import test_rocprim_host_asan
import test_rocthrust_host_asan
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
                patch.object(test_rand_host_asan, "require_direct_clang_asan"),
                patch.object(test_rand_host_asan.subprocess, "run") as run,
            ):
                self.assertEqual(test_rand_host_asan.main(), 0)

        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(len(commands), len(test_rand_host_asan.ROCRAND_TESTS))
        generator_command = next(
            command
            for command in commands
            if "test_rocrand_generator_type" in command[0]
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
                patch.object(test_rand_host_asan, "require_direct_clang_asan"),
                patch.object(test_rand_host_asan.subprocess, "run") as run,
            ):
                self.assertEqual(test_rand_host_asan.main(), 0)

        self.assertEqual(Path(run.call_args.args[0][0]).name, "test_hiprand_linkage")

    def test_rocsparse_runs_only_gpu_independent_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "rocsparse-unit-test").touch()
            with (
                patch.dict(os.environ, {"THEROCK_BIN_DIR": tmp}, clear=False),
                patch.object(test_rocsparse_host_asan, "require_direct_clang_asan"),
                patch.object(test_rocsparse_host_asan.subprocess, "run") as run,
            ):
                self.assertEqual(test_rocsparse_host_asan.main(), 0)

        self.assertEqual(Path(run.call_args.args[0][0]).name, "rocsparse-unit-test")

    def test_stinkytofu_runs_native_generator_and_filecheck_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = Path(tmp) / "bin"
            test_dir = bin_dir / "stinkytofu"
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
                patch.dict(os.environ, {"THEROCK_BIN_DIR": str(bin_dir)}, clear=False),
                patch.object(test_stinkytofu_host_asan, "require_direct_clang_asan"),
                patch.object(test_stinkytofu_host_asan.subprocess, "run") as run,
            ):
                self.assertEqual(test_stinkytofu_host_asan.main(), 0)

            expected_env = run.call_args_list[0].kwargs["env"]
            self.assertEqual(
                expected_env["LD_LIBRARY_PATH"].split(os.pathsep)[:3],
                [
                    str(Path(tmp) / "lib"),
                    str(Path(tmp) / "lib" / "rocm_sysdeps" / "lib"),
                    str(Path(tmp) / "lib" / "llvm" / "lib"),
                ],
            )

        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(Path(commands[0][0]).name, "unit_tests")
        self.assertEqual(Path(commands[1][0]).name, "test_gen_instructions")
        self.assertEqual(commands[1][-2:], ["Gfx1250", "Gfx1250v0"])
        self.assertEqual(len(commands), 4)

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
                patch.dict(os.environ, {"THEROCK_BIN_DIR": tmp}, clear=False),
                patch.object(test_composable_kernel_host_asan, "_require_no_gpu_nodes"),
                patch.object(test_composable_kernel_host_asan, "_run_binary") as run,
            ):
                self.assertEqual(test_composable_kernel_host_asan.main(), 0)

        self.assertEqual(
            [call.args[0].name for call in run.call_args_list],
            list(test_composable_kernel_host_asan.COMPOSABLE_KERNEL_HOST_TESTS),
        )

    def test_composable_kernel_inventory_totals_409(self):
        inventories = (
            test_composable_kernel_host_asan.COMPOSABLE_KERNEL_HOST_TESTS.values()
        )
        self.assertEqual(
            sum(count for count, _ in inventories),
            409,
        )

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

    def test_composable_kernel_rejects_uninstrumented_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            executable = Path(tmp) / "test"
            executable.touch()
            completed = Mock(stdout="NEEDED libc.so.6", returncode=0)
            with patch.object(
                test_composable_kernel_host_asan.subprocess,
                "run",
                return_value=completed,
            ):
                with self.assertRaisesRegex(RuntimeError, "not directly linked"):
                    test_composable_kernel_host_asan._require_direct_asan(
                        executable, {}
                    )

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
