#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT_DIR = Path(__file__).parent.parent / "test_executable_scripts"
sys.path.insert(0, os.fspath(SCRIPT_DIR))

import test_native_host_tsan
import test_rccl_host_tsan
import verify_host_tsan_linkage


def _gtest_listing(names: list[str]) -> str:
    suites: dict[str, list[str]] = {}
    for name in names:
        suite, case = name.rsplit(".", 1)
        suites.setdefault(suite, []).append(case)
    return "".join(
        f"{suite}.\n" + "".join(f"  {case}\n" for case in cases)
        for suite, cases in suites.items()
    )


class RcclHostTsanTest(unittest.TestCase):
    def test_exact_binary_inventory_and_evidence(self):
        self.assertEqual(
            [
                (
                    test["name"],
                    test["count"],
                    test["sha256"],
                    test["executed"],
                )
                for test in test_rccl_host_tsan.GTEST_INVENTORIES
            ],
            [
                (
                    "rccl-UnitTestsMicro",
                    110,
                    "0e6225d55fa67a0fa2a642b181a056e125b929be192ed37be3e08756fedccdaf",
                    110,
                ),
                (
                    "rccl-UnitTestsMicroEnqueue",
                    177,
                    "1ee3e97ee5e0d455ddcf072d094c7f4c633b9cdf2be4f43aadfbb3d7570a9778",
                    177,
                ),
                (
                    "rccl-UnitTestsMicroInit",
                    688,
                    "c08ad78a18c4211548fa8904f6d601d51768a2ab470d144926cd139d0fe7099e",
                    685,
                ),
                (
                    "rccl-UnitTestsMicroInit-faultinj",
                    688,
                    "c08ad78a18c4211548fa8904f6d601d51768a2ab470d144926cd139d0fe7099e",
                    685,
                ),
                (
                    "rccl-UnitTestsMicroInit-uncached",
                    687,
                    "1d4dcc58d78bfa3be9c0bb0345279d88d859e344c8ebf489448943080b671dbb",
                    684,
                ),
            ],
        )
        self.assertEqual(len(test_rccl_host_tsan.NET_TELEMETRY_CASES), 17)
        self.assertEqual(len(test_rccl_host_tsan.DEATH_TEST_EXCLUSIONS), 3)

    def test_init_excludes_exact_death_tests_and_sets_allocator_option(self):
        test = test_rccl_host_tsan.GTEST_INVENTORIES[2]
        selected_names = ["SafeSuite.case"] * test["executed"]
        full_names = [*selected_names, *test_rccl_host_tsan.DEATH_TEST_EXCLUSIONS]
        runs = [
            subprocess.CompletedProcess([], 0, _gtest_listing(full_names), ""),
            subprocess.CompletedProcess([], 0, _gtest_listing(selected_names), ""),
            subprocess.CompletedProcess(
                [], 0, f"[  PASSED  ] {test['executed']} tests.\n", ""
            ),
        ]
        with (
            patch.object(test_rccl_host_tsan, "require_direct_clang_tsan"),
            patch.object(test_rccl_host_tsan, "_require_inventory"),
            patch.object(
                test_rccl_host_tsan.subprocess, "run", side_effect=runs
            ) as run,
        ):
            test_rccl_host_tsan._run_gtest(
                Path("/prefix/bin"), test, {"TSAN_OPTIONS": "halt_on_error=1"}
            )

        execute = run.call_args_list[-1]
        self.assertEqual(
            execute.args[0][1],
            "--gtest_filter=-" + ":".join(test_rccl_host_tsan.DEATH_TEST_EXCLUSIONS),
        )
        self.assertIn(
            "allocator_may_return_null=1", execute.kwargs["env"]["TSAN_OPTIONS"]
        )
        self.assertIn("--gtest_brief=1", execute.args[0])
        self.assertIn("--gtest_color=no", execute.args[0])

    def test_rejects_child_process_pass_summaries(self):
        with self.assertRaisesRegex(RuntimeError, "unexpected RCCL pass summary"):
            test_rccl_host_tsan._require_pass_summary(
                "[  PASSED  ] 1 test.\n[  PASSED  ] 685 tests.\n",
                685,
                "rccl-UnitTestsMicroInit",
            )

    def test_net_telemetry_requires_all_17_ordered_cases(self):
        output = "net_telemetry unit tests\n" + "\n".join(
            test_rccl_host_tsan.NET_TELEMETRY_CASES
        ) + "\nALL PASSED\n"
        with (
            patch.object(test_rccl_host_tsan, "require_direct_clang_tsan"),
            patch.object(
                test_rccl_host_tsan.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0, output, ""),
            ),
        ):
            test_rccl_host_tsan._run_net_telemetry(Path("/prefix/bin"), {})

        with (
            patch.object(test_rccl_host_tsan, "require_direct_clang_tsan"),
            patch.object(
                test_rccl_host_tsan.subprocess,
                "run",
                return_value=subprocess.CompletedProcess(
                    [],
                    0,
                    "\n".join(test_rccl_host_tsan.NET_TELEMETRY_CASES[:-1])
                    + "\nALL PASSED\n",
                    "",
                ),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "inventory changed"):
                test_rccl_host_tsan._run_net_telemetry(Path("/prefix/bin"), {})


class RocshmemHostTsanTest(unittest.TestCase):
    def test_exact_79_test_inventory(self):
        self.assertEqual(
            test_native_host_tsan.COMPONENT_TESTS["rocshmem"],
            (("rocshmem_envvar_test", ()),),
        )
        self.assertEqual(
            test_native_host_tsan.EXACT_GTEST_INVENTORIES[
                ("rocshmem", "rocshmem_envvar_test")
            ],
            (
                79,
                "5f517cf84a42fca8000429fbf5a7826bcb089ae8e5e67f070103428d49f29e4e",
            ),
        )


class RuntimeLinkageInventoryTest(unittest.TestCase):
    def test_verifier_covers_every_admitted_runtime_binary(self):
        self.assertEqual(
            verify_host_tsan_linkage.COMPONENT_INVENTORIES["hip-tests"][
                "executables"
            ],
            (
                "share/hip/catch_tests/VectorTypesTest",
                "share/hip/catch_tests/ErrorHandlingTest",
                "share/hip/catch_tests/ChannelDescriptorTest",
                "share/hip/catch_tests/ComplexTest",
            ),
        )
        self.assertEqual(
            verify_host_tsan_linkage.COMPONENT_INVENTORIES["rocrtst"][
                "executables"
            ],
            ("bin/intercept_queue_logic_test",),
        )
        self.assertEqual(
            verify_host_tsan_linkage.COMPONENT_INVENTORIES["rocshmem"]["executables"],
            ("bin/rocshmem_envvar_test",),
        )
        self.assertEqual(
            verify_host_tsan_linkage.COMPONENT_INVENTORIES["rccl"]["executables"],
            tuple(
                f"bin/{name}"
                for name in (
                    "rccl-UnitTestsMicro",
                    "rccl-UnitTestsMicroEnqueue",
                    "rccl-UnitTestsMicroInit",
                    "rccl-UnitTestsMicroInit-faultinj",
                    "rccl-UnitTestsMicroInit-uncached",
                    "rccl-UnitTestsNetTelemetry",
                )
            ),
        )


if __name__ == "__main__":
    unittest.main()
