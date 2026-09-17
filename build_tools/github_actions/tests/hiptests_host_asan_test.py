# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

GITHUB_ACTIONS_DIR = Path(__file__).parent.parent
SCRIPT_DIR = GITHUB_ACTIONS_DIR / "test_executable_scripts"
sys.path.insert(0, os.fspath(SCRIPT_DIR))

import test_hiptests_host_asan


def _listing(names: tuple[str, ...]) -> str:
    entries = "".join(f"  {name}\n      [host]\n" for name in names)
    return f"Matching test cases:\n{entries}{len(names)} matching test cases\n"


class HiptestsHostAsanTest(unittest.TestCase):
    def test_positive_inventory_is_exact_and_digest_locked(self):
        self.assertEqual(test_hiptests_host_asan.EXPECTED_TEST_COUNT, 35)
        self.assertEqual(len(test_hiptests_host_asan.EXPECTED_TEST_NAMES), 35)
        self.assertEqual(len(set(test_hiptests_host_asan.EXPECTED_TEST_NAMES)), 35)
        self.assertEqual(
            test_hiptests_host_asan._inventory_sha256(
                test_hiptests_host_asan.EXPECTED_TEST_NAMES
            ),
            test_hiptests_host_asan.EXPECTED_INVENTORY_SHA256,
        )

    def test_parser_and_inventory_accept_exact_listing(self):
        output = _listing(test_hiptests_host_asan.EXPECTED_TEST_NAMES)
        names, count = test_hiptests_host_asan._parse_catch_list(output)
        test_hiptests_host_asan._check_inventory(names, count)

    def test_missing_binary_fails_before_listing(self):
        with self.assertRaisesRegex(RuntimeError, "executable is missing"):
            test_hiptests_host_asan.run(Path("/definitely-missing-rocm-prefix"), {})

    def test_zero_match_fails_before_execution(self):
        listed = Mock(
            stdout="Matching test cases:\n0 matching test cases\n",
            stderr="",
            returncode=0,
        )
        with (
            patch.object(test_hiptests_host_asan, "require_direct_clang_asan"),
            patch.object(
                test_hiptests_host_asan.subprocess, "run", return_value=listed
            ) as run,
            self.assertRaisesRegex(RuntimeError, "inventory changed before execution"),
        ):
            test_hiptests_host_asan.run(Path("/opt/rocm"), {})
        self.assertEqual(run.call_count, 1)

    def test_name_or_count_mismatch_fails_before_execution(self):
        mismatched = list(test_hiptests_host_asan.EXPECTED_TEST_NAMES)
        mismatched[-1] = "unexpected_GPU_case"
        listed = Mock(stdout=_listing(tuple(mismatched)), stderr="", returncode=0)
        with (
            patch.object(test_hiptests_host_asan, "require_direct_clang_asan"),
            patch.object(
                test_hiptests_host_asan.subprocess, "run", return_value=listed
            ) as run,
            self.assertRaisesRegex(RuntimeError, "inventory changed before execution"),
        ):
            test_hiptests_host_asan.run(Path("/opt/rocm"), {})
        self.assertEqual(run.call_count, 1)

    def test_child_failure_is_propagated(self):
        listed = Mock(
            stdout=_listing(test_hiptests_host_asan.EXPECTED_TEST_NAMES),
            stderr="",
            returncode=0,
        )
        failed = Mock(stdout="deterministic failure\n", stderr="", returncode=17)
        with (
            patch.object(test_hiptests_host_asan, "require_direct_clang_asan"),
            patch.object(
                test_hiptests_host_asan.subprocess,
                "run",
                side_effect=[listed, failed],
            ),
            self.assertRaises(subprocess.CalledProcessError) as raised,
        ):
            test_hiptests_host_asan.run(Path("/opt/rocm"), {})
        self.assertEqual(raised.exception.returncode, 17)

    def test_environment_removes_preload_and_separates_trace_mode(self):
        with patch.dict(
            os.environ,
            {
                "LD_PRELOAD": "/tmp/preload-is-not-instrumentation.so",
                "ASAN_OPTIONS": "existing=1",
            },
            clear=True,
        ):
            env = test_hiptests_host_asan._test_environment(Path("/opt/rocm"))
        self.assertNotIn("LD_PRELOAD", env)
        self.assertEqual(
            env["ASAN_OPTIONS"], "existing=1:detect_leaks=1:halt_on_error=1"
        )
        self.assertEqual(env["LSAN_OPTIONS"], "exitcode=23")

        with patch.dict(
            os.environ, {"THEROCK_HOST_ASAN_DEVICE_TRACE": "1"}, clear=True
        ):
            trace_env = test_hiptests_host_asan._test_environment(Path("/opt/rocm"))
        self.assertEqual(
            trace_env["ASAN_OPTIONS"], "detect_leaks=0:halt_on_error=1"
        )

    def test_main_rejects_visible_gpu_nodes(self):
        with (
            patch.dict(os.environ, {"THEROCK_BIN_DIR": "/opt/rocm/bin"}),
            patch.object(
                test_hiptests_host_asan,
                "_require_no_gpu_nodes",
                side_effect=RuntimeError("GPU nodes"),
            ),
        ):
            self.assertEqual(test_hiptests_host_asan.main(), 1)


if __name__ == "__main__":
    unittest.main()
