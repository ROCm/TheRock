#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import hashlib
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

SCRIPT_DIR = Path(__file__).parent.parent / "test_executable_scripts"
sys.path.insert(0, os.fspath(SCRIPT_DIR))

import host_tsan_instrumentation
import test_ctest_host_tsan
import test_hiptests_host_tsan
import test_native_host_tsan
import test_rocroller_host_tsan


class HostTsanInstrumentationTest(unittest.TestCase):
    def test_environment_forbids_preload_and_sets_deterministic_options(self):
        env = host_tsan_instrumentation.native_host_tsan_environment(
            {
                "LD_PRELOAD": "/tmp/forbidden.so",
                "LD_LIBRARY_PATH": "/existing",
                "TSAN_RUNTIME_PATH": "/toolchain/lib/libclang_rt.tsan.so",
            }
        )
        self.assertNotIn("LD_PRELOAD", env)
        self.assertEqual(
            env["TSAN_OPTIONS"], host_tsan_instrumentation.TSAN_OPTIONS
        )
        self.assertEqual(
            env["LD_LIBRARY_PATH"].split(os.pathsep),
            [os.fspath(Path("/toolchain/lib")), "/existing"],
        )

    def test_environment_preserves_configured_symbolizer(self):
        env = host_tsan_instrumentation.native_host_tsan_environment(
            {
                "TSAN_SYMBOLIZER_PATH": "/rocm/llvm/bin/llvm-symbolizer",
                "TSAN_OPTIONS": "exitcode=1",
            }
        )

        self.assertEqual(
            env["TSAN_OPTIONS"],
            host_tsan_instrumentation.TSAN_OPTIONS
            + ":external_symbolizer_path=/rocm/llvm/bin/llvm-symbolizer",
        )

    def test_rejects_executable_without_direct_tsan_dependency(self):
        executable = Path("/tests/test")
        with (
            patch.object(Path, "is_file", return_value=True),
            patch.object(
                host_tsan_instrumentation.subprocess,
                "run",
                return_value=Mock(stdout="NEEDED libc.so.6"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "not directly linked"):
                host_tsan_instrumentation.require_direct_clang_tsan(executable)

    def test_rejects_exposed_gpu_nodes(self):
        with patch.object(Path, "exists", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "unexpectedly exposes GPU"):
                host_tsan_instrumentation.require_no_gpu_nodes()

    def test_gtest_parser_normalizes_comments_and_rejects_false_success(self):
        output = "TypedSuite/0.  # TypeParam = int\n  case  # GetParam() = 1\n"
        self.assertEqual(
            host_tsan_instrumentation.parse_gtest_listed_tests(output),
            ["TypedSuite/0.case"],
        )
        with self.assertRaisesRegex(RuntimeError, "skipped tests"):
            host_tsan_instrumentation.require_gtest_execution(
                "[  SKIPPED ] Suite.case"
            )
        with self.assertRaisesRegex(RuntimeError, "zero tests"):
            host_tsan_instrumentation.require_gtest_execution(
                "Running 0 tests from 0 test suites."
            )


class NativeHostTsanTest(unittest.TestCase):
    def test_exact_component_inventories_match_phase_evidence(self):
        self.assertEqual(
            test_native_host_tsan.COMPONENT_TESTS["rocrtst"],
            (("intercept_queue_logic_test", ()),),
        )
        self.assertEqual(
            test_native_host_tsan.EXACT_GTEST_INVENTORIES[
                ("rocrtst", "intercept_queue_logic_test")
            ],
            (
                9,
                "427e22cbc625d6c60ae4a6707dbbcbdaf2ad49b54bc788a03e1b52f5b505bf36",
            ),
        )
        self.assertEqual(len(test_native_host_tsan.COMPONENT_TESTS["rocrand"]), 9)
        self.assertEqual(
            test_native_host_tsan.COMPONENT_TESTS["hiprand"],
            (("test_hiprand_linkage", ()),),
        )
        self.assertEqual(
            test_native_host_tsan.COMPONENT_TESTS["rocsparse"],
            (("rocsparse-unit-test", ()),),
        )
        self.assertEqual(len(test_native_host_tsan.COMPONENT_TESTS["rocprim"]), 11)
        self.assertEqual(len(test_native_host_tsan.COMPONENT_TESTS["rocthrust"]), 16)
        self.assertEqual(
            test_native_host_tsan.FILTERED_GTEST_INVENTORIES[
                ("rocrand", "test_rocrand_generator_type")
            ],
            (
                "rocrand_generator_type_tests.generate_test",
                "rocrand_generator_type_tests.rocrand_generator",
            ),
        )

    def test_executes_only_selected_native_binaries(self):
        with (
            patch.dict(
                os.environ,
                {"THEROCK_BIN_DIR": "/prefix/bin", "TEST_COMPONENT": "hiprand"},
                clear=False,
            ),
            patch.object(test_native_host_tsan, "require_no_gpu_nodes"),
            patch.object(test_native_host_tsan, "require_direct_clang_tsan"),
            patch.object(
                test_native_host_tsan.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0, "", ""),
            ) as run,
        ):
            self.assertEqual(test_native_host_tsan.main(), 0)
        self.assertEqual(Path(run.call_args.args[0][0]).name, "test_hiprand_linkage")
        self.assertNotIn("LD_PRELOAD", run.call_args.kwargs["env"])


class HipTestsHostTsanTest(unittest.TestCase):
    def test_commands_disable_aslr_for_tsan_fixed_mappings(self):
        command = test_hiptests_host_tsan._without_aslr("/tests/VectorTypesTest")
        self.assertEqual(command[0], "setarch")
        self.assertEqual(command[2:], ["-R", "/tests/VectorTypesTest"])

    def test_exact_device_free_inventory(self):
        self.assertEqual(
            [
                (name, count, digest)
                for name, _, count, digest in test_hiptests_host_tsan.HIP_HOST_TESTS
            ],
            [
                (
                    "VectorTypesTest",
                    44,
                    "4df54143fe42cba3c7d064daef51f4556bb351066a64f9d1f494fc691302c18b",
                ),
                (
                    "ErrorHandlingTest",
                    8,
                    "36fdea2d27c315d7070de2f86004e8f3485826b9ec547b96b01eeceb64b54d0e",
                ),
                (
                    "ChannelDescriptorTest",
                    56,
                    "5c0db148b5623da771ac3daaafab0db5a3ce937878b6630ee5b5a0977aefa5f5",
                ),
                (
                    "ComplexTest",
                    13,
                    "cf08ffdf01ed75347bf52d87f46968e85303db85b21771802899018487f70762",
                ),
            ],
        )

    def test_count_validation_is_fail_closed(self):
        test_hiptests_host_tsan._require_count(
            "44 matching test cases\n",
            44,
            test_hiptests_host_tsan._LIST_COUNT_RE,
            "discovery",
        )
        test_hiptests_host_tsan._require_count(
            "All tests passed (100 assertions in 44 test cases)\n",
            44,
            test_hiptests_host_tsan._PASS_COUNT_RE,
            "execution",
        )
        with self.assertRaisesRegex(RuntimeError, "inventory changed"):
            test_hiptests_host_tsan._require_count(
                "43 test cases\n",
                44,
                test_hiptests_host_tsan._LIST_COUNT_RE,
                "discovery",
            )
        with self.assertRaisesRegex(RuntimeError, "did not execute exactly"):
            test_hiptests_host_tsan._require_count(
                "All tests passed (100 assertions in 44 test cases); 1 skipped\n",
                44,
                test_hiptests_host_tsan._PASS_COUNT_RE,
                "execution",
            )

    def test_discovery_names_are_digest_frozen(self):
        output = (
            "All available test cases:\n  alpha\n      [tag]\n  beta\n2 test cases\n"
        )
        digest = hashlib.sha256(b"alpha\nbeta\n").hexdigest()
        test_hiptests_host_tsan._require_inventory(output, 2, digest)
        with self.assertRaisesRegex(RuntimeError, "inventory changed"):
            test_hiptests_host_tsan._require_inventory(output, 2, "0" * 64)


class RocrollerHostTsanTest(unittest.TestCase):
    def test_filter_is_positive_and_zero_selection_fails(self):
        self.assertTrue(test_rocroller_host_tsan.HOST_SAFE_FILTERS)
        self.assertTrue(
            all(
                not value.startswith("-")
                for value in test_rocroller_host_tsan.HOST_SAFE_FILTERS
            )
        )
        self.assertEqual(test_rocroller_host_tsan._count_listed_tests(""), 0)
        self.assertEqual(
            test_rocroller_host_tsan._count_listed_tests("Suite.\n  case\n"), 1
        )
        self.assertEqual(
            test_rocroller_host_tsan.RELEASE_MODE_NON_EXECUTING_TESTS,
            ("CommandTest.DuplicateOp",),
        )
        self.assertNotIn(
            "CommandTest.DuplicateOp", test_rocroller_host_tsan.HOST_SAFE_FILTERS
        )

    def test_inventory_is_frozen_to_built_cpu_selection(self):
        selected = "\n".join(
            (
                "ArgumentLoaderTest.",
                "  eagerLoadArguments",
                "  IsContiguousRange",
                "  loadArgExtra",
                "  pickInstructionWidth",
                "  releaseArguments",
                "AssemblerTest.",
                "  BadTarget",
                "  Basic",
                "CommandTest.",
                "  Basic",
                "  BlockScaleInline",
                "  BlockScaleSeparate",
                "  CommandKernelPredicates",
                "  ConvertOp",
                "  FindCommandArguments",
                "  GetRuntimeArguments",
                "  SetCommandArguments",
                "  ToString",
                "  VectorAdd",
                "  XopInputOutputs",
                "ComponentTest.",
                "  Basic",
                "ErrorFixtureDeathTest.",
                "  BreakOnAssertFatal",
                "  BreakOnThrow",
            )
        )
        self.assertEqual(test_rocroller_host_tsan._validate_inventory(selected), 21)
        candidate = selected.replace(
            "  ConvertOp\n", "  ConvertOp\n  DuplicateOp\n"
        )
        test_rocroller_host_tsan._validate_candidate_inventory(candidate)
        with self.assertRaisesRegex(RuntimeError, "inventory changed"):
            test_rocroller_host_tsan._validate_inventory(selected + "\n  NewTest")
        with self.assertRaisesRegex(RuntimeError, "candidate inventory changed"):
            test_rocroller_host_tsan._validate_candidate_inventory(selected)


class CtestHostTsanTest(unittest.TestCase):
    def test_inventory_contracts_are_positive_and_exact(self):
        self.assertEqual(
            test_ctest_host_tsan.COMPONENTS["origami"]["expected_names"],
            ("origami-tests",),
        )
        self.assertEqual(
            test_ctest_host_tsan.COMPONENTS["hipfile"]["inventory_count"], 668
        )

    def test_inventory_mismatch_fails_before_execution(self):
        with self.assertRaisesRegex(RuntimeError, "inventory changed before execution"):
            test_ctest_host_tsan._validate_inventory(
                "origami", test_ctest_host_tsan.COMPONENTS["origami"], []
            )

    def test_run_command_requires_nonempty_selection(self):
        config = test_ctest_host_tsan.COMPONENTS["origami"]
        with patch.object(
            test_ctest_host_tsan.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 0, '{"tests": []}', ""),
        ):
            with self.assertRaisesRegex(RuntimeError, "resolved to zero tests"):
                test_ctest_host_tsan._selected_commands(
                    Path("/tests"), config["args"], {}
                )

    def test_ctest_execution_is_fail_closed(self):
        config = test_ctest_host_tsan.COMPONENTS["origami"]
        command = [
            "ctest",
            "--test-dir",
            "/prefix/bin/origami",
            *config["args"],
            "--output-on-failure",
            "--no-tests=error",
        ]
        self.assertIn("--no-tests=error", command)
        self.assertEqual(command[3:5], ["-R", "^origami-tests$"])

    def test_ctest_execution_rejects_skips_and_inexact_summary(self):
        with self.assertRaisesRegex(RuntimeError, "skipped or did not run"):
            test_ctest_host_tsan._validate_ctest_execution(
                "1/1 Test #1: origami-tests ***Skipped", 1
            )
        with self.assertRaisesRegex(RuntimeError, "exact passing inventory"):
            test_ctest_host_tsan._validate_ctest_execution(
                "100% tests passed, 0 tests failed out of 2", 1
            )
        test_ctest_host_tsan._validate_ctest_execution(
            "100% tests passed, 0 tests failed out of 1", 1
        )

    def test_hipfile_runs_each_frozen_test_in_an_isolated_process(self):
        config = {
            "inventory_count": 2,
            "inventory_sha256": hashlib.sha256(
                b"Suite.alpha\nSuite.beta\n"
            ).hexdigest(),
        }
        listed = subprocess.CompletedProcess(
            [], 0, "Suite.\n  alpha\n  beta\n", ""
        )
        passed = subprocess.CompletedProcess(
            [],
            0,
            "[==========] Running 1 test from 1 test suite.\n"
            "[  PASSED  ] 1 test.\n",
            "",
        )
        with (
            patch.object(test_ctest_host_tsan, "require_direct_clang_tsan"),
            patch.object(
                test_ctest_host_tsan.platform, "machine", return_value="x86_64"
            ),
            patch.object(
                test_ctest_host_tsan.subprocess,
                "run",
                side_effect=[listed, passed, passed],
            ) as run,
        ):
            test_ctest_host_tsan._run_hipfile(
                Path("/prefix/share/hipfile/test"), config, {}
            )

        self.assertEqual(run.call_count, 3)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertTrue(
            all(command[:4] == ["cmake", "-E", "env", "--"] for command in commands)
        )
        self.assertTrue(
            all("setarch" in command and "-R" in command for command in commands)
        )
        self.assertIn("--gtest_filter=Suite.alpha", commands[1])
        self.assertIn("--gtest_filter=Suite.beta", commands[2])

    def test_hipfile_isolated_execution_requires_exactly_one_pass(self):
        with self.assertRaisesRegex(RuntimeError, "did not report one pass"):
            test_ctest_host_tsan._require_one_gtest_pass(
                "[==========] Running 1 test from 1 test suite.\n", "Suite.case"
            )


if __name__ == "__main__":
    unittest.main()
