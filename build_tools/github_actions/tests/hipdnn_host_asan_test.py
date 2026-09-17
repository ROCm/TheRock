# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

SCRIPT_DIR = Path(__file__).parent.parent / "test_executable_scripts"
sys.path.insert(0, os.fspath(SCRIPT_DIR))

import test_hipdnn_host_asan


class HipdnnHostAsanTest(unittest.TestCase):
    def test_integration_inventory_totals_exactly_541(self):
        self.assertEqual(
            sum(
                selection["count"]
                for selection in test_hipdnn_host_asan.INTEGRATION_SELECTIONS.values()
            ),
            541,
        )

    def test_each_binary_has_an_independent_positive_filter(self):
        selections = test_hipdnn_host_asan.INTEGRATION_SELECTIONS.values()
        self.assertEqual(len({item["filter"] for item in selections}), 3)
        for item in selections:
            self.assertTrue(item["filter"])
            self.assertNotIn("-*", item["filter"])

    def test_inventory_rejects_zero_selection(self):
        expected = {
            "count": 1,
            "sha256": test_hipdnn_host_asan._inventory_digest(["A.a"]),
        }
        with self.assertRaisesRegex(RuntimeError, "inventory changed"):
            test_hipdnn_host_asan._validate_inventory([], expected)

    def test_inventory_rejects_count_mismatch(self):
        expected = {
            "count": 2,
            "sha256": test_hipdnn_host_asan._inventory_digest(["A.a"]),
        }
        with self.assertRaisesRegex(RuntimeError, "inventory changed"):
            test_hipdnn_host_asan._validate_inventory(["A.a"], expected)

    def test_inventory_rejects_name_drift(self):
        expected = {
            "count": 1,
            "sha256": test_hipdnn_host_asan._inventory_digest(["A.a"]),
        }
        with self.assertRaisesRegex(RuntimeError, "inventory changed"):
            test_hipdnn_host_asan._validate_inventory(["B.b"], expected)

    def test_ctest_category_set_mismatch_fails(self):
        payload = json.dumps(
            {"tests": [{"name": "hipdnn_integration_tests_unit_tests_host-asan_suite"}]}
        )
        completed = Mock(stdout=payload, returncode=0)
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = Path(tmp)
            test_dir = bin_dir / "hipdnn_integration_tests_ctest"
            test_dir.mkdir()
            (test_dir / "CTestTestfile.cmake").touch()
            with patch.object(
                test_hipdnn_host_asan.subprocess,
                "run",
                return_value=completed,
            ):
                with self.assertRaisesRegex(RuntimeError, "categories changed"):
                    test_hipdnn_host_asan._installed_ctest_commands(bin_dir, {})

    def test_golden_command_requires_packaged_data(self):
        expected = test_hipdnn_host_asan.INTEGRATION_SELECTIONS[
            "hipdnn_golden_data_tests_host-asan_suite"
        ]
        test = {
            "command": [
                "../hipdnn_golden_data_tests",
                "--reference",
                "cpu",
                "--gd",
                "../../lib/integration-test-bundles/quick/BatchnormFwdInference",
                f"--gtest_filter={expected['filter']}",
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "golden data is missing"):
                test_hipdnn_host_asan._validate_installed_command(
                    test, expected, prefix
                )

    def test_direct_asan_uses_shared_instrumentation_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "native.so"
            artifact.touch()
            with patch.object(
                test_hipdnn_host_asan,
                "require_direct_clang_asan",
            ) as require_direct:
                test_hipdnn_host_asan._require_direct_asan(artifact, {})
        require_direct.assert_called_once_with(artifact, {})

    def test_missing_native_artifact_fails(self):
        with self.assertRaisesRegex(RuntimeError, "artifact is missing"):
            test_hipdnn_host_asan._read_dynamic(
                Path("definitely-missing-hipdnn-artifact"), {}
            )

    def test_native_failure_status_propagates(self):
        completed = Mock(stdout="", stderr="", returncode=9)
        with (
            patch.object(test_hipdnn_host_asan, "_require_direct_asan"),
            patch.object(
                test_hipdnn_host_asan.subprocess,
                "run",
                return_value=completed,
            ),
        ):
            with self.assertRaisesRegex(
                subprocess.CalledProcessError, "returned non-zero exit status 9"
            ):
                test_hipdnn_host_asan._run_install_engine(
                    Path("/tmp/bin"), Path("/tmp"), {}
                )

    def test_environment_forbids_preload_and_enables_lsan(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(
                os.environ,
                {"LD_PRELOAD": "/tmp/not-allowed.so", "ASAN_OPTIONS": "old=1"},
                clear=False,
            ):
                env = test_hipdnn_host_asan._test_environment(Path(tmp))
        self.assertNotIn("LD_PRELOAD", env)
        self.assertIn("detect_leaks=1", env["ASAN_OPTIONS"])
        self.assertIn("halt_on_error=1", env["ASAN_OPTIONS"])
        self.assertIn("exitcode=23", env["LSAN_OPTIONS"])

    def test_trace_mode_disables_only_lsan(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(
                os.environ, {"THEROCK_HOST_ASAN_DEVICE_TRACE": "1"}, clear=False
            ):
                env = test_hipdnn_host_asan._test_environment(Path(tmp))
        self.assertIn("detect_leaks=0", env["ASAN_OPTIONS"])
        self.assertIn("halt_on_error=1", env["ASAN_OPTIONS"])


if __name__ == "__main__":
    unittest.main()
