# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
# Add tests directory to path for extended_tests imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests"))
import configure_ci


class ConfigureCITest(unittest.TestCase):
    def assert_target_output_is_valid(self, target_output, allow_xfail):
        self.assertTrue(all("test-runs-on" in entry for entry in target_output))
        self.assertTrue(all("family" in entry for entry in target_output))

        if not allow_xfail:
            self.assertFalse(
                any(entry.get("expect_failure") for entry in target_output)
            )

    ###########################################################################
    # Tests for matrix_generator and helper functions

    def test_filter_known_target_names(self):
        requested_target_names = ["gfx110X", "abcdef"]
        # Use all trigger types to get a comprehensive matrix for testing
        test_matrix = configure_ci.get_all_families_for_trigger_types(
            ["presubmit", "postsubmit", "nightly"]
        )
        target_names = configure_ci.filter_known_names(
            requested_target_names, "target", test_matrix
        )
        self.assertIn("gfx110x", target_names)
        self.assertNotIn("abcdef", target_names)

    def test_filter_known_test_names(self):
        requested_test_names = ["hipsparse", "hipdense"]
        test_names = configure_ci.filter_known_names(requested_test_names, "test")
        self.assertIn("hipsparse", test_names)
        self.assertNotIn("hipdense", test_names)

    def test_valid_linux_workflow_dispatch_matrix_generator(self):
        build_families = {"amdgpu_families": "   gfx94X , gfx103X"}
        linux_target_output, linux_test_labels = configure_ci.matrix_generator(
            is_pull_request=False,
            is_workflow_dispatch=True,
            is_push=False,
            is_schedule=False,
            base_args={
                "workflow_dispatch_linux_test_labels": "",
                "workflow_dispatch_windows_test_labels": "",
                "build_variant": "release",
            },
            families=build_families,
            platform="linux",
        )
        self.assertTrue(
            any("gfx94X-dcgpu" == entry["family"] for entry in linux_target_output)
        )
        self.assertTrue(
            any("gfx103X-dgpu" == entry["family"] for entry in linux_target_output)
        )
        self.assertGreaterEqual(len(linux_target_output), 2)
        self.assert_target_output_is_valid(
            target_output=linux_target_output, allow_xfail=True
        )
        self.assertEqual(linux_test_labels, [])

    def test_invalid_linux_workflow_dispatch_matrix_generator(self):
        build_families = {
            "amdgpu_families": "",
        }
        linux_target_output, linux_test_labels = configure_ci.matrix_generator(
            is_pull_request=False,
            is_workflow_dispatch=True,
            is_push=False,
            is_schedule=False,
            base_args={"build_variant": "release"},
            families=build_families,
            platform="linux",
        )
        self.assertEqual(linux_target_output, [])
        self.assertEqual(linux_test_labels, [])

    def test_valid_linux_pull_request_matrix_generator(self):
        base_args = {
            "pr_labels": '{"labels":[{"name":"gfx94X-linux"},{"name":"gfx110X-linux"},{"name":"gfx110X-windows"}]}',
            "build_variant": "release",
        }
        linux_target_output, linux_test_labels = configure_ci.matrix_generator(
            is_pull_request=True,
            is_workflow_dispatch=False,
            is_push=False,
            is_schedule=False,
            base_args=base_args,
            families={},
            platform="linux",
        )
        self.assertTrue(
            any("gfx94X-dcgpu" == entry["family"] for entry in linux_target_output)
        )
        self.assertTrue(
            any("gfx110X-all" == entry["family"] for entry in linux_target_output)
        )
        self.assertGreaterEqual(len(linux_target_output), 2)
        self.assert_target_output_is_valid(
            target_output=linux_target_output, allow_xfail=False
        )
        self.assertEqual(linux_test_labels, [])

    def test_duplicate_windows_pull_request_matrix_generator(self):
        base_args = {
            "pr_labels": '{"labels":[{"name":"gfx94X-linux"},{"name":"gfx110X-linux"},{"name":"gfx110X-windows"},{"name":"gfx110X-windows"}]}',
            "build_variant": "release",
        }
        windows_target_output, windows_test_labels = configure_ci.matrix_generator(
            is_pull_request=True,
            is_workflow_dispatch=False,
            is_push=False,
            is_schedule=False,
            base_args=base_args,
            families={},
            platform="windows",
        )
        self.assertTrue(
            any("gfx110X-all" == entry["family"] for entry in windows_target_output)
        )
        self.assertGreaterEqual(len(windows_target_output), 1)
        self.assert_target_output_is_valid(
            target_output=windows_target_output, allow_xfail=False
        )
        self.assertEqual(windows_test_labels, [])

    def test_invalid_linux_pull_request_matrix_generator(self):
        base_args = {
            "pr_labels": '{"labels":[{"name":"gfx10000X-linux"},{"name":"gfx110000X-windows"}]}',
            "build_variant": "release",
        }
        linux_target_output, windows_test_labels = configure_ci.matrix_generator(
            is_pull_request=True,
            is_workflow_dispatch=False,
            is_push=False,
            is_schedule=False,
            base_args=base_args,
            families={},
            platform="linux",
        )
        self.assertGreaterEqual(len(linux_target_output), 1)
        self.assert_target_output_is_valid(
            target_output=linux_target_output, allow_xfail=True
        )
        self.assertEqual(windows_test_labels, [])

    def test_empty_windows_pull_request_matrix_generator(self):
        base_args = {"pr_labels": "{}", "build_variant": "release"}
        windows_target_output, windows_test_labels = configure_ci.matrix_generator(
            is_pull_request=True,
            is_workflow_dispatch=False,
            is_push=False,
            is_schedule=False,
            base_args=base_args,
            families={},
            platform="windows",
        )
        self.assertGreaterEqual(len(windows_target_output), 1)
        self.assert_target_output_is_valid(
            target_output=windows_target_output, allow_xfail=False
        )
        self.assertEqual(windows_test_labels, [])

    def test_valid_test_label_linux_pull_request_matrix_generator(self):
        base_args = {
            "pr_labels": '{"labels":[{"name":"test:hipblaslt"},{"name":"test:rocblas"}]}',
            "build_variant": "release",
        }
        linux_target_output, linux_test_labels = configure_ci.matrix_generator(
            is_pull_request=True,
            is_workflow_dispatch=False,
            is_push=False,
            is_schedule=False,
            base_args=base_args,
            families={},
            platform="linux",
        )
        self.assertGreaterEqual(len(linux_target_output), 1)
        self.assert_target_output_is_valid(
            target_output=linux_target_output, allow_xfail=False
        )
        self.assertTrue(any("hipblaslt" == entry for entry in linux_test_labels))
        self.assertTrue(any("rocblas" == entry for entry in linux_test_labels))
        self.assertGreaterEqual(len(linux_test_labels), 2)

    def test_invalid_test_label_linux_pull_request_matrix_generator(self):
        base_args = {
            "pr_labels": '{"labels":[{"name":"test:hipchalk"},{"name":"test:rocchalk"}]}',
            "build_variant": "release",
        }
        linux_target_output, linux_test_labels = configure_ci.matrix_generator(
            is_pull_request=True,
            is_workflow_dispatch=False,
            is_push=False,
            is_schedule=False,
            base_args=base_args,
            families={},
            platform="linux",
        )
        self.assertGreaterEqual(len(linux_target_output), 1)
        self.assert_target_output_is_valid(
            target_output=linux_target_output, allow_xfail=False
        )
        self.assertEqual(linux_test_labels, [])

    def test_kernel_test_label_linux_pull_request_matrix_generator(self):
        base_args = {
            "pr_labels": '{"labels":[{"name":"test_runner:oem"}]}',
            "build_variant": "release",
        }
        linux_target_output, linux_test_labels = configure_ci.matrix_generator(
            is_pull_request=True,
            is_workflow_dispatch=False,
            is_push=False,
            is_schedule=False,
            base_args=base_args,
            families={},
            platform="linux",
        )
        self.assertGreaterEqual(len(linux_target_output), 1)
        # check that at least one runner name has "oem" in test runner name if "oem" test runner was requested
        self.assertTrue("oem" in item["test-runs-on"] for item in linux_target_output)
        self.assert_target_output_is_valid(
            target_output=linux_target_output, allow_xfail=False
        )
        self.assertEqual(linux_test_labels, [])

    @patch("subprocess.run")
    def test_filter_tests_from_pull_request(self, mock_run):
        base_args = {
            "pr_labels": '{"labels":[{"name":"test_filter:comprehensive"}]}',
            "build_variant": "release",
            "github_event_name": "pull_request",
            "base_ref": "HEAD^",
        }
        mock_process = MagicMock()
        mock_process.stdout = ".github/workflows/ci.yml\nsrc/some_code.cpp"
        mock_run.return_value = mock_process
        captured_out = io.StringIO()
        captured_err = io.StringIO()
        with redirect_stdout(captured_out), redirect_stderr(captured_err):
            configure_ci.main(base_args, {}, {})
        self.assertIn('"test_type": "comprehensive"', captured_out.getvalue())

    @patch("subprocess.run")
    def test_invalid_filter_tests_from_pull_request(self, mock_run):
        base_args = {
            "pr_labels": '{"labels":[{"name":"test_filter:extended"}]}',
            "build_variant": "release",
            "github_event_name": "pull_request",
            "base_ref": "HEAD^",
        }
        mock_process = MagicMock()
        mock_process.stdout = ".github/workflows/ci.yml\nsrc/some_code.cpp"
        mock_run.return_value = mock_process
        captured_out = io.StringIO()
        captured_err = io.StringIO()
        with redirect_stdout(captured_out), redirect_stderr(captured_err):
            configure_ci.main(base_args, {}, {})
        self.assertIn('"test_type": "quick"', captured_out.getvalue())

    @patch("subprocess.run")
    def test_valid_main_push_ci_run(self, mock_run):
        base_args = {
            "build_variant": "release",
            "github_event_name": "push",
            "base_ref": "HEAD^",
        }
        mock_process = MagicMock()
        mock_process.stdout = ".github/workflows/ci.yml"
        mock_run.return_value = mock_process
        configure_ci.main(base_args, {}, {})

    @patch("subprocess.run")
    def test_valid_schedule_ci_run(self, mock_run):
        base_args = {
            "build_variant": "release",
            "github_event_name": "schedule",
            "base_ref": "HEAD^",
        }
        mock_process = MagicMock()
        mock_process.stdout = ".github/workflows/ci.yml"
        mock_run.return_value = mock_process
        captured_out = io.StringIO()
        captured_err = io.StringIO()
        with redirect_stdout(captured_out), redirect_stderr(captured_err):
            configure_ci.main(base_args, {}, {})
        self.assertIn('"test_type": "comprehensive"', captured_out.getvalue())

    @patch("subprocess.run")
    def test_valid_workflow_dispatch_ci_run(self, mock_run):
        base_args = {
            "build_variant": "release",
            "github_event_name": "workflow_dispatch",
            "base_ref": "HEAD^",
        }
        mock_process = MagicMock()
        mock_process.stdout = ".github/workflows/ci.yml"
        mock_run.return_value = mock_process
        configure_ci.main(
            base_args, {"amdgpu_families": "gfx94X"}, {"amdgpu_families": "gfx110X"}
        )

    def test_skip_ci_label(self):
        base_args = {
            "pr_labels": '{"labels":[{"name":"ci:skip"},{"name":"test:hipblaslt"},{"name":"test:rocblas"},{"name":"gfx94X-linux"},{"name":"gfx110X-linux"},{"name":"gfx110X-windows"},{"name":"test_runner:oem"}]}',
            "build_variant": "release",
        }
        linux_target_output, linux_test_labels = configure_ci.matrix_generator(
            is_pull_request=True,
            is_workflow_dispatch=False,
            is_push=False,
            is_schedule=False,
            base_args=base_args,
            families={},
            platform="linux",
        )
        self.assertEqual(len(linux_target_output), 0)
        self.assert_target_output_is_valid(
            target_output=linux_target_output, allow_xfail=False
        )
        self.assertEqual(linux_test_labels, [])

    def test_main_linux_branch_push_matrix_generator(self):
        base_args = {"branch_name": "main", "build_variant": "release"}
        linux_target_output, linux_test_labels = configure_ci.matrix_generator(
            is_pull_request=False,
            is_workflow_dispatch=False,
            is_push=True,
            is_schedule=False,
            base_args=base_args,
            families={},
            platform="linux",
        )
        self.assertGreaterEqual(len(linux_target_output), 1)
        self.assert_target_output_is_valid(
            target_output=linux_target_output, allow_xfail=True
        )
        self.assertEqual(linux_test_labels, [])

    def test_main_windows_branch_push_matrix_generator(self):
        base_args = {"branch_name": "main", "build_variant": "release"}
        windows_target_output, windows_test_labels = configure_ci.matrix_generator(
            is_pull_request=False,
            is_workflow_dispatch=False,
            is_push=True,
            is_schedule=False,
            base_args=base_args,
            families={},
            platform="windows",
        )
        self.assertGreaterEqual(len(windows_target_output), 1)
        self.assert_target_output_is_valid(
            target_output=windows_target_output, allow_xfail=False
        )
        self.assertEqual(windows_test_labels, [])

    def test_linux_branch_push_matrix_generator(self):
        # Push to non-main branches uses presubmit defaults
        base_args = {"branch_name": "test_branch", "build_variant": "release"}
        linux_target_output, linux_test_labels = configure_ci.matrix_generator(
            is_pull_request=False,
            is_workflow_dispatch=False,
            is_push=True,
            is_schedule=False,
            base_args=base_args,
            families={},
            platform="linux",
        )
        # Should use presubmit defaults
        self.assertGreaterEqual(len(linux_target_output), 1)
        self.assert_target_output_is_valid(
            target_output=linux_target_output, allow_xfail=False
        )

    def test_linux_schedule_matrix_generator(self):
        linux_target_output, linux_test_labels = configure_ci.matrix_generator(
            is_pull_request=False,
            is_workflow_dispatch=False,
            is_push=False,
            is_schedule=True,
            base_args={"build_variant": "release"},
            families={},
            platform="linux",
        )
        self.assertGreaterEqual(len(linux_target_output), 1)
        self.assert_target_output_is_valid(
            target_output=linux_target_output, allow_xfail=True
        )
        self.assertEqual(linux_test_labels, [])

    def test_windows_schedule_matrix_generator(self):
        windows_target_output, windows_test_labels = configure_ci.matrix_generator(
            is_pull_request=False,
            is_workflow_dispatch=False,
            is_push=False,
            is_schedule=True,
            base_args={"build_variant": "release"},
            families={},
            platform="windows",
        )
        self.assertGreaterEqual(len(windows_target_output), 1)
        self.assert_target_output_is_valid(
            target_output=windows_target_output, allow_xfail=True
        )
        self.assertEqual(windows_test_labels, [])

    def test_build_pytorch_disabled_when_expect_failure(self):
        """build_pytorch should be False when expect_failure is True."""
        # Schedule trigger includes all families, some with expect_failure
        linux_target_output, _ = configure_ci.matrix_generator(
            is_pull_request=False,
            is_workflow_dispatch=False,
            is_push=False,
            is_schedule=True,
            base_args={"build_variant": "release"},
            families={},
            platform="linux",
        )
        for entry in linux_target_output:
            if entry.get("expect_failure", False):
                self.assertFalse(
                    entry.get("build_pytorch", False),
                    f"build_pytorch should be False when expect_failure is True "
                    f"for family {entry.get('family')}",
                )

    def test_build_pytorch_disabled_when_expect_pytorch_failure(self):
        """build_pytorch should be False when expect_pytorch_failure is True."""
        # Use schedule trigger on windows to include gfx90x which has
        # expect_pytorch_failure on windows
        windows_target_output, _ = configure_ci.matrix_generator(
            is_pull_request=False,
            is_workflow_dispatch=False,
            is_push=False,
            is_schedule=True,
            base_args={"build_variant": "release"},
            families={},
            platform="windows",
        )
        for entry in windows_target_output:
            if entry.get("expect_pytorch_failure", False):
                self.assertFalse(
                    entry.get("build_pytorch", False),
                    f"build_pytorch should be False when expect_pytorch_failure "
                    f"is True for family {entry.get('family')}",
                )

    def test_build_pytorch_enabled_for_supported_families(self):
        """build_pytorch should be True for families without known failures."""
        # Presubmit families (gfx94x, gfx110x, etc.) should have build_pytorch
        # enabled if they don't have expect_failure or expect_pytorch_failure
        linux_target_output, _ = configure_ci.matrix_generator(
            is_pull_request=True,
            is_workflow_dispatch=False,
            is_push=False,
            is_schedule=False,
            base_args={
                "pr_labels": '{"labels":[]}',
                "build_variant": "release",
            },
            families={},
            platform="linux",
        )
        for entry in linux_target_output:
            if not entry.get("expect_failure", False) and not entry.get(
                "expect_pytorch_failure", False
            ):
                self.assertTrue(
                    entry.get("build_pytorch", False),
                    f"build_pytorch should be True for supported family "
                    f"{entry.get('family')}",
                )

    def test_determine_long_lived_branch(self):
        """Test to correctly determine long-lived branch that expect more testing."""

        # long-lived branches
        for branch in [
            "main",
            "release/therock-7.9",
            "release/therock-",
            "release/therock-100",
        ]:
            self.assertTrue(configure_ci.determine_long_lived_branch(branch))
        # non long-lived branches
        for branch in [
            "users/test",
            "release/therock",
            "main-test",
            "newfeature",
            "release/main",
        ]:
            self.assertFalse(configure_ci.determine_long_lived_branch(branch))

    # TODO(#3433): Remove sandbox logic once ASAN tests are passing and environment is no longer required
    def test_sandbox_test_runner_with_asan(self):
        base_args = {"build_variant": "asan"}
        build_families = {"amdgpu_families": "gfx94X"}
        linux_target_output, linux_test_labels = configure_ci.matrix_generator(
            is_pull_request=True,
            is_workflow_dispatch=False,
            is_push=False,
            is_schedule=False,
            base_args=base_args,
            families=build_families,
            platform="linux",
        )
        entry = linux_target_output[0]
        self.assertEqual(entry["test-runs-on"], "rocm-asan-mi325-sandbox")

    ###########################################################################
    # Tests for dual-label runner selection

    def test_gfx94x_alternate_label_selected_when_random_below_weight(self):
        """When random() < weight, alternate label should be selected."""
        base_args = {"build_variant": "release"}
        build_families = {"amdgpu_families": "gfx94X"}

        # Mock random.random() to return 0.1 (< 0.2 weight)
        with patch("random.random", return_value=0.1):
            linux_target_output, _ = configure_ci.matrix_generator(
                is_pull_request=True,
                is_workflow_dispatch=False,
                is_push=False,
                is_schedule=False,
                base_args=base_args,
                families=build_families,
                platform="linux",
            )

        # Find the gfx94X entry
        gfx94x_entries = [
            e for e in linux_target_output if e["family"] == "gfx94X-dcgpu"
        ]
        self.assertEqual(len(gfx94x_entries), 1, "Expected exactly one gfx94X entry")
        entry = gfx94x_entries[0]
        # Should select the alternate CCS label
        self.assertEqual(entry["test-runs-on"], "linux-gfx942-1gpu-ccs-ossci-rocm")

    def test_gfx94x_primary_label_selected_when_random_above_weight(self):
        """When random() >= weight, primary label should be selected."""
        base_args = {"build_variant": "release"}
        build_families = {"amdgpu_families": "gfx94X"}

        # Mock random.random() to return 0.5 (>= 0.35 weight)
        with patch("random.random", return_value=0.5):
            linux_target_output, _ = configure_ci.matrix_generator(
                is_pull_request=True,
                is_workflow_dispatch=False,
                is_push=False,
                is_schedule=False,
                base_args=base_args,
                families=build_families,
                platform="linux",
            )

        # Find the gfx94X entry
        gfx94x_entries = [
            e for e in linux_target_output if e["family"] == "gfx94X-dcgpu"
        ]
        self.assertEqual(len(gfx94x_entries), 1, "Expected exactly one gfx94X entry")
        entry = gfx94x_entries[0]
        # Should select the primary label
        self.assertEqual(entry["test-runs-on"], "linux-gfx942-1gpu-ossci-rocm")

    def test_families_without_alternate_always_use_primary(self):
        """Families without dual-label config should always use primary label."""
        base_args = {"build_variant": "release"}
        build_families = {"amdgpu_families": "gfx103X"}

        # Run multiple times to ensure consistency (no alternate label exists)
        for _ in range(5):
            linux_target_output, _ = configure_ci.matrix_generator(
                is_pull_request=False,
                is_workflow_dispatch=False,
                is_push=False,
                is_schedule=True,
                base_args=base_args,
                families=build_families,
                platform="linux",
            )

            # Find the gfx103X entry
            gfx103x_entries = [
                e for e in linux_target_output if e["family"] == "gfx103X-dgpu"
            ]
            if gfx103x_entries:
                entry = gfx103x_entries[0]
                # Should always use the same primary label
                self.assertEqual(entry["test-runs-on"], "linux-gfx1030-gpu-rocm")


if __name__ == "__main__":
    unittest.main()
