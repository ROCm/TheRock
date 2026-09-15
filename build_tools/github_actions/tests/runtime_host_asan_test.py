# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

SCRIPT_DIR = Path(__file__).parent.parent / "test_executable_scripts"
sys.path.insert(0, os.fspath(SCRIPT_DIR))

import test_runtime_host_asan


class RuntimeHostAsanTest(unittest.TestCase):
    def test_host_asan_comm_stage_rebuilds_only_rocshmem(self):
        workflow_path = (
            Path(__file__).resolve().parents[3]
            / ".github"
            / "workflows"
            / "multi_arch_build_portable_linux.yml"
        )
        workflow = workflow_path.read_text(encoding="utf-8")
        comm_job = workflow.split("\n  comm-libs:\n", maxsplit=1)[1].split(
            "\n  # ==========================================================================",
            maxsplit=1,
        )[0]

        self.assertIn(
            "rebuild_artifacts: ${{ startsWith(inputs.build_variant_label, "
            "'host-asan') && 'rocshmem' || inputs.rebuild_artifacts }}",
            comm_job,
        )

    def test_inventory_is_explicit(self):
        self.assertEqual(set(test_runtime_host_asan.COMPONENTS), {"rocshmem", "rocrtst"})
        self.assertEqual(
            [entry["count"] for entry in test_runtime_host_asan.COMPONENTS["rocshmem"]],
            [79],
        )
        self.assertEqual(
            [entry["count"] for entry in test_runtime_host_asan.COMPONENTS["rocrtst"]],
            [9, 5, 13],
        )

    def test_parser_handles_typed_suite_comments(self):
        output = """Suite.
  one
Typed/0.  # TypeParam = bool
  two  # GetParam() = value
"""
        self.assertEqual(
            test_runtime_host_asan._parse_gtest_names(output),
            ["Suite.one", "Typed/0.two"],
        )

    def test_environment_removes_preload_and_enables_lsan(self):
        with patch.dict(
            os.environ,
            {"LD_PRELOAD": "/tmp/not-allowed.so", "ASAN_OPTIONS": "existing=1"},
            clear=False,
        ):
            env = test_runtime_host_asan._test_environment()
        self.assertNotIn("LD_PRELOAD", env)
        self.assertIn("detect_leaks=1", env["ASAN_OPTIONS"])
        self.assertIn("halt_on_error=1", env["ASAN_OPTIONS"])
        self.assertIn("exitcode=23", env["LSAN_OPTIONS"])

    def test_trace_mode_disables_only_lsan(self):
        with patch.dict(os.environ, {"THEROCK_HOST_ASAN_DEVICE_TRACE": "1"}, clear=False):
            env = test_runtime_host_asan._test_environment()
        self.assertIn("detect_leaks=0", env["ASAN_OPTIONS"])
        self.assertIn("halt_on_error=1", env["ASAN_OPTIONS"])

    def test_direct_asan_rejects_uninstrumented_binary(self):
        executable = Path("/fake/test")
        with (
            patch.object(Path, "is_file", return_value=True),
            patch.object(
                test_runtime_host_asan.subprocess,
                "run",
                return_value=Mock(stdout="NEEDED libc.so.6", returncode=0),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "not directly linked"):
                test_runtime_host_asan._require_direct_asan(executable, {})

    def test_inventory_drift_fails_before_test_execution(self):
        prefix = Path("/fake/prefix")
        listed = Mock(stdout="WrongSuite.\n  wrong\n", stderr="", returncode=0)
        with (
            patch.object(test_runtime_host_asan, "_require_direct_asan"),
            patch.object(test_runtime_host_asan.subprocess, "run", return_value=listed) as run,
        ):
            with self.assertRaisesRegex(RuntimeError, "inventory changed"):
                test_runtime_host_asan._run_component(prefix, "rocshmem", {})
            self.assertEqual(run.call_count, 1)

    def test_child_failure_is_propagated(self):
        test = test_runtime_host_asan.COMPONENTS["rocshmem"][0]
        names = [f"Suite.case{i}" for i in range(test["count"])]
        prefix = Path("/fake/prefix")
        original_digest = test["sha256"]
        test["sha256"] = test_runtime_host_asan._inventory_digest(names)
        listed = Mock(
            stdout="Suite.\n" + "".join(f"  case{i}\n" for i in range(test["count"])),
            stderr="",
            returncode=0,
        )
        failed = Mock(stdout="", stderr="boom", returncode=17)
        try:
            with (
                patch.object(test_runtime_host_asan, "_require_direct_asan"),
                patch.object(
                    test_runtime_host_asan.subprocess,
                    "run",
                    side_effect=[listed, failed],
                ),
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    test_runtime_host_asan._run_component(prefix, "rocshmem", {})
        finally:
            test["sha256"] = original_digest

    def test_visible_gpu_nodes_fail_closed(self):
        with patch.object(Path, "exists", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "unexpectedly exposes GPU nodes"):
                test_runtime_host_asan._require_no_gpu_nodes()


if __name__ == "__main__":
    unittest.main()
