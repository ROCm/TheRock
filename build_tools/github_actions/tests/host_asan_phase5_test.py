# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import io
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

GITHUB_ACTIONS_DIR = Path(__file__).parent.parent
BUILD_TOOLS_DIR = GITHUB_ACTIONS_DIR.parent
SCRIPT_DIR = GITHUB_ACTIONS_DIR / "test_executable_scripts"
sys.path.insert(0, os.fspath(SCRIPT_DIR))
sys.path.insert(0, os.fspath(GITHUB_ACTIONS_DIR))
sys.path.insert(0, os.fspath(BUILD_TOOLS_DIR))

import fetch_test_configurations
import test_phase5_host_asan
from _therock_utils.build_topology import get_topology


class HostAsanPhase5Test(unittest.TestCase):
    def test_workflow_does_not_blanket_preload_host_asan(self):
        workflow = (
            Path(__file__).parents[3] / ".github" / "workflows" / "test_component.yml"
        ).read_text(encoding="utf-8")
        test_step = workflow.split("- name: Test", 1)[1].split(
            "- name: Print test reproduction", 1
        )[0]
        self.assertNotIn("LD_PRELOAD:", test_step)

    def test_component_inventory_is_explicit(self):
        self.assertEqual(
            set(test_phase5_host_asan.GTEST_COMPONENTS),
            {"rocprofiler-compute", "rocprofiler-sdk"},
        )
        self.assertEqual(set(test_phase5_host_asan.CTEST_COMPONENTS), {"hipfile"})
        self.assertEqual(
            [
                entry["inventory_count"]
                for entry in test_phase5_host_asan.GTEST_COMPONENTS[
                    "rocprofiler-compute"
                ]
            ],
            [79, 59],
        )
        self.assertEqual(
            [
                entry["inventory_count"]
                for entry in test_phase5_host_asan.GTEST_COMPONENTS["rocprofiler-sdk"]
            ],
            [55, 33, 7],
        )
        self.assertEqual(
            test_phase5_host_asan.GTEST_COMPONENTS["rocprofiler-sdk"][1][
                "skipped_names"
            ],
            ("codeobj_library.dwarf_matches_llvm_symbolizer",),
        )
        self.assertEqual(
            test_phase5_host_asan.CTEST_COMPONENTS["hipfile"]["inventory_count"],
            668,
        )

    def test_fetch_selectors_have_enabled_producer_stages(self):
        topology = get_topology()
        producer_stages = topology.get_artifact_to_producer_stages()
        expected = {
            "hipfile": ("hipfile", "storage-libs"),
            "rocprofiler-compute": ("rocprofiler-compute", "compiler-runtime"),
            "rocprofiler-sdk": ("rocprofiler-sdk", "compiler-runtime"),
            "rocrtst": ("rocrtst", "runtime-tests"),
        }

        for component, (artifact, stage) in expected.items():
            fetch_args = shlex.split(
                fetch_test_configurations._host_asan_matrix()[component][
                    "fetch_artifact_args"
                ]
            )
            self.assertIn(f"--{artifact}", fetch_args)
            self.assertIn("--tests", fetch_args)
            self.assertIn(stage, producer_stages[artifact])

    def test_gtest_parser_handles_parameter_comments(self):
        output = """Running main() from gtest_main.cc
Suite.
  one
Typed/Parameterized.
  two/0  # GetParam() = value
"""
        self.assertEqual(
            test_phase5_host_asan._parse_gtest_names(output),
            ["Suite.one", "Typed/Parameterized.two/0"],
        )
        self.assertEqual(
            test_phase5_host_asan._normalize_names(["Suite.two", "Suite.one"]),
            "Suite.one\nSuite.two\n",
        )

    def test_test_environment_forbids_preload_and_enables_lsan(self):
        with patch.dict(
            os.environ,
            {
                "LD_PRELOAD": "/tmp/not-allowed.so",
                "LD_LIBRARY_PATH": "/existing/lib",
                "ASAN_RUNTIME_PATH": "/toolchain/lib/libclang_rt.asan.so",
                "ASAN_OPTIONS": "existing=1",
            },
            clear=False,
        ):
            env = test_phase5_host_asan._test_environment(
                Path("/opt/rocm"), "rocprofiler-sdk"
            )
        self.assertNotIn("LD_PRELOAD", env)
        self.assertIn("detect_leaks=1", env["ASAN_OPTIONS"])
        self.assertIn("halt_on_error=1", env["ASAN_OPTIONS"])
        self.assertIn("exitcode=23", env["LSAN_OPTIONS"])
        self.assertEqual(
            env["LD_LIBRARY_PATH"].split(os.pathsep),
            [
                str(Path("/opt/rocm/lib")),
                str(Path("/opt/rocm/lib/rocm_sysdeps/lib")),
                str(Path("/opt/rocm/lib/llvm/lib")),
                str(Path("/toolchain/lib")),
                "/existing/lib",
            ],
        )
        self.assertEqual(
            env["ROCPROFILER_METRICS_PATH"],
            str(Path("/opt/rocm") / "share" / "rocprofiler-sdk"),
        )

        with patch.dict(
            os.environ, {"THEROCK_HOST_ASAN_DEVICE_TRACE": "1"}, clear=False
        ):
            trace_env = test_phase5_host_asan._test_environment(
                Path("/opt/rocm"), "hipfile"
            )
        self.assertTrue(
            trace_env["ASAN_OPTIONS"].endswith("detect_leaks=0:halt_on_error=1")
        )

    def test_direct_asan_check_rejects_uninstrumented_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            executable = Path(tmp) / "test"
            executable.touch()
            completed = Mock(stdout="NEEDED libc.so.6", returncode=0)
            with patch.object(
                test_phase5_host_asan.subprocess, "run", return_value=completed
            ):
                with self.assertRaisesRegex(RuntimeError, "not directly linked"):
                    test_phase5_host_asan._require_direct_asan(executable, {})

    def test_inventory_digest_mismatch_fails_before_execution(self):
        with self.assertRaisesRegex(RuntimeError, "inventory changed before execution"):
            test_phase5_host_asan._check_inventory(
                names=["Suite.test"],
                raw_output="Suite.test\n",
                expected_count=1,
                expected_sha256="0" * 64,
            )

    def test_main_reports_captured_subprocess_diagnostics(self):
        failure = subprocess.CalledProcessError(
            127,
            ["host-test", "--gtest_list_tests"],
            output="discovery stdout\n",
            stderr="loader diagnostic\n",
        )
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch.dict(
                os.environ,
                {
                    "THEROCK_BIN_DIR": "/opt/rocm/bin",
                    "TEST_COMPONENT": "rocprofiler-sdk",
                },
                clear=False,
            ),
            patch.object(test_phase5_host_asan, "_require_no_gpu_nodes"),
            patch.object(
                test_phase5_host_asan,
                "_run_gtest_component",
                side_effect=failure,
            ),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            self.assertEqual(test_phase5_host_asan.main(), 1)

        self.assertIn("discovery stdout", stdout.getvalue())
        self.assertIn("loader diagnostic", stderr.getvalue())
        self.assertIn("exit status 127", stderr.getvalue())

    def test_skipped_name_parser_ignores_timed_event_and_reads_summary(self):
        output = """[  SKIPPED ] Suite.expected_skip (4 ms)
[  SKIPPED ] 1 test, listed below:
[  SKIPPED ] Suite.expected_skip
"""
        self.assertEqual(
            test_phase5_host_asan._SKIPPED_NAME_RE.findall(output),
            ["Suite.expected_skip"],
        )

    def test_main_routes_only_phase5_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = Path(tmp) / "bin"
            bin_dir.mkdir()
            with (
                patch.dict(
                    os.environ,
                    {"THEROCK_BIN_DIR": str(bin_dir), "TEST_COMPONENT": "hipfile"},
                    clear=False,
                ),
                patch.object(test_phase5_host_asan, "_require_no_gpu_nodes"),
                patch.object(test_phase5_host_asan, "_run_ctest_component") as run,
            ):
                self.assertEqual(test_phase5_host_asan.main(), 0)
                self.assertEqual(run.call_args.args[1], "hipfile")

            with (
                patch.dict(
                    os.environ,
                    {
                        "THEROCK_BIN_DIR": str(bin_dir),
                        "TEST_COMPONENT": "hipdnn-integration-tests",
                    },
                    clear=False,
                ),
                patch.object(test_phase5_host_asan, "_require_no_gpu_nodes"),
            ):
                self.assertEqual(test_phase5_host_asan.main(), 1)


if __name__ == "__main__":
    unittest.main()
