# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import hashlib
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).parent.parent / "test_executable_scripts"
sys.path.insert(0, os.fspath(SCRIPT_DIR))

import test_rpp_host_tsan
import verify_host_tsan_linkage


class RppHostTsanTest(unittest.TestCase):
    def test_host_tsan_build_excludes_only_device_test_targets(self):
        hook = (
            Path(__file__).resolve().parents[3] / "cv-libs" / "post_hook_rpp.cmake"
        ).read_text(encoding="utf-8")
        self.assertIn('THEROCK_SANITIZER STREQUAL "HOST_TSAN"', hook)
        for target in (
            "Tensor_image_hip",
            "Tensor_misc_hip",
            "Tensor_voxel_hip",
            "Tensor_audio_hip",
        ):
            self.assertIn(target, hook)
        self.assertIn("EXCLUDE_FROM_ALL TRUE", hook)
        self.assertNotIn("Tensor_image_host", hook)
        self.assertNotIn("Tensor_misc_host", hook)

    def test_exact_host_inventory(self):
        self.assertEqual(
            test_rpp_host_tsan.EXPECTED_NAMES,
            (
                "rpp_sanity_test_brightness_host_f32",
                "rpp_qa_tests_tensor_image_host_all",
                "rpp_qa_tests_tensor_misc_host_all",
            ),
        )
        normalized = "".join(
            f"{name}\n" for name in sorted(test_rpp_host_tsan.EXPECTED_NAMES)
        )
        self.assertEqual(
            hashlib.sha256(normalized.encode()).hexdigest(),
            test_rpp_host_tsan.EXPECTED_INVENTORY_SHA256,
        )

    def test_inventory_change_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "inventory changed"):
            test_rpp_host_tsan._validate_inventory(
                ["rpp_sanity_test_brightness_host_f32"]
            )

    def test_global_verifier_checks_installed_library(self):
        self.assertEqual(
            verify_host_tsan_linkage.COMPONENT_INVENTORIES["rpp"],
            {"libraries": ("lib/librpp.so",)},
        )

    def test_environment_sets_required_parallel_runtimes_when_ci_omits_limits(self):
        with mock.patch.object(
            test_rpp_host_tsan,
            "native_host_tsan_environment",
            return_value={"LD_LIBRARY_PATH": "/existing"},
        ):
            env = test_rpp_host_tsan._test_environment(Path("/opt/rocm"))

        self.assertEqual(env["OMP_NUM_THREADS"], "4")
        self.assertEqual(env["OPENBLAS_NUM_THREADS"], "1")
        self.assertEqual(env["ROCM_PATH"], os.fspath(Path("/opt/rocm")))
        self.assertIn("/existing", env["LD_LIBRARY_PATH"].split(os.pathsep))

    def test_environment_replaces_inherited_host_parallel_runtime_limits(self):
        with mock.patch.object(
            test_rpp_host_tsan,
            "native_host_tsan_environment",
            return_value={"OMP_NUM_THREADS": "64", "OPENBLAS_NUM_THREADS": "32"},
        ):
            env = test_rpp_host_tsan._test_environment(Path("/opt/rocm"))

        self.assertEqual(env["OMP_NUM_THREADS"], "4")
        self.assertEqual(env["OPENBLAS_NUM_THREADS"], "1")

    def test_environment_does_not_collapse_to_single_affinity_cpu(self):
        with mock.patch.object(
            test_rpp_host_tsan,
            "native_host_tsan_environment",
            return_value={},
        ), mock.patch.object(
            test_rpp_host_tsan.os,
            "sched_getaffinity",
            return_value={0},
            create=True,
        ) as affinity:
            env = test_rpp_host_tsan._test_environment(Path("/opt/rocm"))

        affinity.assert_not_called()
        self.assertEqual(env["OMP_NUM_THREADS"], "4")
        self.assertEqual(env["OPENBLAS_NUM_THREADS"], "1")

    def test_ctest_execution_disables_aslr(self):
        completed = subprocess.CompletedProcess(
            [], 0, "100% tests passed, 0 tests failed out of 1\n", ""
        )
        with mock.patch.object(
            test_rpp_host_tsan, "_run", return_value=completed
        ) as run:
            test_rpp_host_tsan._run_ctest(
                Path("/tmp/rpp"),
                {},
                test_rpp_host_tsan.BRIGHTNESS_NAME,
                timeout_seconds=120,
                repeat_until_pass=3,
            )

        command = run.call_args.args[0]
        self.assertEqual(
            command[:3], ["setarch", test_rpp_host_tsan.platform.machine(), "-R"]
        )
        self.assertIn("^rpp_sanity_test_brightness_host_f32$", command)
        self.assertEqual(command[command.index("--timeout") + 1], "120")
        self.assertEqual(command[command.index("--repeat") + 1], "until-pass:3")

    def test_comprehensive_ctest_uses_extended_timeout_without_parallelism(self):
        completed = subprocess.CompletedProcess(
            [], 0, "100% tests passed, 0 tests failed out of 1\n", ""
        )
        with mock.patch.object(
            test_rpp_host_tsan, "_run", return_value=completed
        ) as run:
            test_rpp_host_tsan._run_ctest(
                Path("/tmp/rpp"),
                {},
                test_rpp_host_tsan.COMPREHENSIVE_NAMES[0],
                timeout_seconds=1500,
            )

        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--timeout") + 1], "1500")
        self.assertNotIn("--parallel", command)
        self.assertNotIn("--repeat", command)


if __name__ == "__main__":
    unittest.main()
