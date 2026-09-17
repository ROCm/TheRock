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

    def test_environment_bounds_parallel_runtimes_when_ci_omits_limits(self):
        with mock.patch.object(
            test_rpp_host_tsan,
            "native_host_tsan_environment",
            return_value={"LD_LIBRARY_PATH": "/existing"},
        ), mock.patch.object(
            test_rpp_host_tsan.os,
            "sched_getaffinity",
            return_value=set(range(64)),
            create=True,
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
        ), mock.patch.object(
            test_rpp_host_tsan.os,
            "sched_getaffinity",
            return_value=set(range(4)),
            create=True,
        ):
            env = test_rpp_host_tsan._test_environment(Path("/opt/rocm"))

        self.assertEqual(env["OMP_NUM_THREADS"], "4")
        self.assertEqual(env["OPENBLAS_NUM_THREADS"], "1")

    def test_ctest_execution_disables_aslr(self):
        source = Path(test_rpp_host_tsan.__file__).read_text(encoding="utf-8")
        self.assertIn('"setarch",\n                    platform.machine(),\n                    "-R",', source)


if __name__ == "__main__":
    unittest.main()
