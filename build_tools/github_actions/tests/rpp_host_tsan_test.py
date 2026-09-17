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
        self.assertEqual(
            test_rpp_host_tsan.EXECUTED_NAMES,
            (
                "rpp_qa_tests_tensor_image_host_all",
                "rpp_qa_tests_tensor_misc_host_all",
            ),
        )
        self.assertEqual(
            test_rpp_host_tsan.PRESUBMIT_CASES[
                "rpp_qa_tests_tensor_image_host_all"
            ],
            (
                "5",
                "21",
                "40",
                "49",
                "61",
                "65",
                "70",
                "90",
            ),
        )
        self.assertEqual(
            test_rpp_host_tsan.PRESUBMIT_CASES[
                "rpp_qa_tests_tensor_misc_host_all"
            ],
            (
                "0",
                "1",
                "2",
                "3",
                "5",
                "8",
                "11",
            ),
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
        self.assertEqual(env["OMP_WAIT_POLICY"], "ACTIVE")
        self.assertEqual(env["KMP_BLOCKTIME"], "infinite")
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

    def test_brightness_runs_prebuilt_binary_without_build_and_test_wrapper(self):
        tests = [
            {
                "name": test_rpp_host_tsan.EXPECTED_NAMES[0],
                "command": [
                    "cmake",
                    "--build-and-test",
                    "/src",
                    "/build/HOST",
                    "--test-command",
                    "Tensor_image_host",
                    "/src/images",
                    "2",
                    "0",
                ],
            }
        ]
        with mock.patch.object(test_rpp_host_tsan, "_run") as run:
            test_rpp_host_tsan._run_brightness(
                Path("/tmp/rpp-build"), {}, tests
            )

        command = run.call_args.args[0]
        self.assertEqual(
            command[:3], ["setarch", test_rpp_host_tsan.platform.machine(), "-R"]
        )
        self.assertEqual(
            command[3], os.fspath(Path("/tmp/rpp-build/HOST/Tensor_image_host"))
        )
        self.assertNotIn("--build-and-test", command)
        self.assertEqual(command[4:], ["/src/images", "2", "0"])

    def test_brightness_fails_closed_when_registered_payload_changes(self):
        tests = [
            {
                "name": test_rpp_host_tsan.EXPECTED_NAMES[0],
                "command": ["cmake", "--build-and-test", "/src", "/build"],
            }
        ]
        with self.assertRaisesRegex(RuntimeError, "wrapper changed"):
            test_rpp_host_tsan._brightness_payload(tests)

    def test_presubmit_suite_uses_registered_command_and_positive_case_list(self):
        completed = subprocess.CompletedProcess(
            [],
            0,
            "Total test cases including all subvariants REQUESTED = 181\n"
            "Total test cases including all subvariants PASSED = 181\n",
            "",
        )
        tests = [
            {
                "name": test_rpp_host_tsan.EXECUTED_NAMES[0],
                "command": ["python", "/src/HOST/runImageTests.py", "--qa_mode", "1"],
            }
        ]
        with mock.patch.object(
            test_rpp_host_tsan, "_run", return_value=completed
        ) as run, mock.patch.object(
            test_rpp_host_tsan, "require_direct_clang_tsan"
        ) as linkage:
            test_rpp_host_tsan._run_presubmit_suite(
                Path("/tmp/rpp"),
                {},
                tests,
                test_rpp_host_tsan.EXECUTED_NAMES[0],
                timeout_seconds=900,
            )

        command = run.call_args.args[0]
        self.assertEqual(
            command[:3], ["setarch", test_rpp_host_tsan.platform.machine(), "-R"]
        )
        self.assertEqual(
            command[3:8],
            [
                "python",
                "-u",
                "/src/HOST/runImageTests.py",
                "--qa_mode",
                "1",
            ],
        )
        self.assertEqual(
            command[command.index("--case_list") + 1 :],
            list(
                test_rpp_host_tsan.PRESUBMIT_CASES[
                    test_rpp_host_tsan.EXECUTED_NAMES[0]
                ]
            ),
        )
        self.assertEqual(run.call_args.kwargs["timeout_seconds"], 900)
        linkage.assert_called_once_with(
            Path("/tmp/rpp/build/Tensor_image_host"), {}
        )

    def test_misc_presubmit_limits_tensor_ranks(self):
        completed = subprocess.CompletedProcess(
            [],
            0,
            "Total test cases including all subvariants REQUESTED = 90\n"
            "Total test cases including all subvariants PASSED = 90\n",
            "",
        )
        name = test_rpp_host_tsan.EXECUTED_NAMES[1]
        tests = [
            {
                "name": name,
                "command": [
                    "python",
                    "/src/HOST/runMiscTests.py",
                    "--qa_mode",
                    "1",
                ],
            }
        ]
        with mock.patch.object(
            test_rpp_host_tsan, "_run", return_value=completed
        ) as run, mock.patch.object(
            test_rpp_host_tsan, "require_direct_clang_tsan"
        ):
            test_rpp_host_tsan._run_presubmit_suite(
                Path("/tmp/rpp"), {}, tests, name, timeout_seconds=600
            )

        command = run.call_args.args[0]
        rank_index = command.index("--num_dims_list")
        self.assertEqual(
            command[rank_index : rank_index + 3],
            ["--num_dims_list", "2", "4"],
        )
        self.assertLess(rank_index, command.index("--case_list"))
        self.assertEqual(run.call_args.kwargs["timeout_seconds"], 600)

    def test_presubmit_suite_rejects_missing_expected_qa_variants(self):
        completed = subprocess.CompletedProcess(
            [],
            0,
            "Total test cases including all subvariants REQUESTED = 180\n"
            "Total test cases including all subvariants PASSED = 180\n",
            "",
        )
        name = test_rpp_host_tsan.EXECUTED_NAMES[0]
        tests = [
            {
                "name": name,
                "command": ["python", "/src/HOST/runImageTests.py"],
            }
        ]
        with mock.patch.object(test_rpp_host_tsan, "_run", return_value=completed):
            with self.assertRaisesRegex(RuntimeError, "expected 181"):
                test_rpp_host_tsan._run_presubmit_suite(
                    Path("/tmp/rpp"), {}, tests, name, timeout_seconds=900
                )

    def test_presubmit_suite_fails_closed_on_child_process_error(self):
        completed = subprocess.CompletedProcess(
            [],
            0,
            "Returned non-zero exit status : 86 WARNING: ThreadSanitizer\n"
            "100% tests passed, 0 tests failed out of 1\n",
            "",
        )
        with mock.patch.object(
            test_rpp_host_tsan, "_run", return_value=completed
        ):
            with self.assertRaisesRegex(RuntimeError, "child process failed"):
                test_rpp_host_tsan._run_presubmit_suite(
                    Path("/tmp/rpp"),
                    {},
                    [
                        {
                            "name": test_rpp_host_tsan.EXECUTED_NAMES[0],
                            "command": [
                                "python",
                                "/src/HOST/runImageTests.py",
                            ],
                        }
                    ],
                    test_rpp_host_tsan.EXECUTED_NAMES[0],
                    timeout_seconds=1500,
                )


if __name__ == "__main__":
    unittest.main()
