# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import hashlib
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).parent.parent / "test_executable_scripts"
sys.path.insert(0, os.fspath(SCRIPT_DIR))

import test_profiler_host_tsan


class ProfilerHostTsanTest(unittest.TestCase):
    def test_exact_component_binary_inventories(self):
        self.assertEqual(
            [
                test["inventory_count"]
                for test in test_profiler_host_tsan.COMPONENTS[
                    "rocprofiler-compute"
                ]
            ],
            [79, 59],
        )
        self.assertEqual(
            [
                test["inventory_count"]
                for test in test_profiler_host_tsan.COMPONENTS["rocprofiler-sdk"]
            ],
            [55, 33, 7],
        )
        codeobj = test_profiler_host_tsan.COMPONENTS["rocprofiler-sdk"][1]
        self.assertEqual(codeobj["passed"], 33)
        self.assertEqual(codeobj["skipped"], 0)
        self.assertNotIn("skipped_names", codeobj)
        systems = test_profiler_host_tsan.COMPONENTS["rocprofiler-systems"]
        self.assertEqual(len(systems), 1)
        self.assertEqual(systems[0]["inventory_count"], 1452)
        self.assertEqual(
            systems[0]["inventory_sha256"],
            "eca8383c653a46e316a6452b914f726bcbd67353e1e2181625871e0f7540caa2",
        )
        self.assertEqual(systems[0]["passed"], 1452)
        self.assertEqual(systems[0]["skipped"], 0)

    def test_inventory_digest_mismatch_fails_closed(self):
        expected = {
            "inventory_count": 1,
            "inventory_sha256": "0" * 64,
        }
        with self.assertRaisesRegex(RuntimeError, "inventory changed before execution"):
            test_profiler_host_tsan._validate_inventory(
                Path("test"), ["Suite.case"], expected
            )

    def test_inventory_normalization_is_sorted(self):
        normalized = test_profiler_host_tsan._normalize_names(
            ["Suite.second", "Suite.first"]
        )
        self.assertEqual(normalized, "Suite.first\nSuite.second\n")
        self.assertEqual(
            hashlib.sha256(normalized.encode()).hexdigest(),
            hashlib.sha256(b"Suite.first\nSuite.second\n").hexdigest(),
        )

    def test_result_requires_exact_pass_and_skip_contract(self):
        executable = Path("codeobj-library-tests")
        expected = {
            "passed": 32,
            "skipped": 1,
            "skipped_names": ("codeobj_library.dwarf_matches_llvm_symbolizer",),
        }
        output = """[  PASSED  ] 32 tests.
[  SKIPPED ] 1 test, listed below:
[  SKIPPED ] codeobj_library.dwarf_matches_llvm_symbolizer
"""
        test_profiler_host_tsan._validate_result(executable, output, expected)

        with self.assertRaisesRegex(RuntimeError, "unexpected GoogleTest result"):
            test_profiler_host_tsan._validate_result(
                executable, "[  PASSED  ] 31 tests.\n", expected
            )

    def test_environment_never_preloads_and_sets_sdk_metrics(self):
        with mock.patch.dict(os.environ, {"LD_PRELOAD": "/forbidden.so"}, clear=False):
            env = test_profiler_host_tsan._test_environment(
                Path("/opt/rocm"), "rocprofiler-sdk"
            )
        self.assertNotIn("LD_PRELOAD", env)
        self.assertEqual(env["ROCM_PATH"], os.fspath(Path("/opt/rocm")))
        self.assertEqual(
            env["ROCPROFILER_METRICS_PATH"],
            os.fspath(Path("/opt/rocm/share/rocprofiler-sdk")),
        )

    def test_rocprofiler_systems_runs_in_isolated_temporary_directory(self):
        command = ["/opt/rocm/bin/rocprof-sys-unit-tests"]
        env = {"TSAN_OPTIONS": "halt_on_error=1"}
        with mock.patch.object(
            test_profiler_host_tsan.tempfile, "TemporaryDirectory"
        ) as temporary_directory, mock.patch.object(
            test_profiler_host_tsan.subprocess, "run"
        ) as run:
            temporary_directory.return_value.__enter__.return_value = (
                "/writable/rocprofiler-systems"
            )
            test_profiler_host_tsan._run_test_binary(
                command, env, "rocprofiler-systems"
            )

        temporary_directory.assert_called_once_with(
            prefix="therock-rocprofiler-systems-"
        )
        run.assert_called_once_with(
            command,
            cwd="/writable/rocprofiler-systems",
            capture_output=True,
            text=True,
            env={**env, "PWD": "/writable/rocprofiler-systems"},
        )
        temporary_directory.return_value.__exit__.assert_called_once()

    def test_other_profilers_keep_the_callers_working_directory(self):
        command = ["/opt/rocm/bin/common-tests"]
        env = {"TSAN_OPTIONS": "halt_on_error=1"}
        with mock.patch.object(test_profiler_host_tsan.subprocess, "run") as run:
            test_profiler_host_tsan._run_test_binary(
                command, env, "rocprofiler-sdk"
            )

        run.assert_called_once_with(
            command,
            capture_output=True,
            text=True,
            env=env,
        )


if __name__ == "__main__":
    unittest.main()
