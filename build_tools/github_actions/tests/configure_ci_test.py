import json
from pathlib import Path
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import configure_ci
from benchmarks.benchmark_test_matrix import benchmark_matrix

therock_test_runner_dict = {
    "gfx110x": {
        "linux": "linux-gfx110X-gpu-rocm-test",
        "windows": "windows-gfx110X-gpu-rocm-test",
    },
}

os.environ["ROCM_THEROCK_TEST_RUNNERS"] = json.dumps(therock_test_runner_dict)


class ConfigureCITest(unittest.TestCase):
    def assert_target_output_is_valid(self, target_output, allow_xfail):
        self.assertTrue(all("test-runs-on" in entry for entry in target_output))
        self.assertTrue(all("family" in entry for entry in target_output))

        if not allow_xfail:
            self.assertFalse(
                any(entry.get("expect_failure") for entry in target_output)
            )

    def assert_multi_arch_output_is_valid(self, target_output, allow_xfail):
        """Validate multi-arch matrix output format."""
        import json

        self.assertTrue(
            all("matrix_per_family_json" in entry for entry in target_output)
        )
        self.assertTrue(all("dist_amdgpu_families" in entry for entry in target_output))
        self.assertTrue(all("build_variant_label" in entry for entry in target_output))
        # Multi-arch output should NOT have 'family' field at top level
        self.assertFalse(any("family" in entry for entry in target_output))

        # Validate structure of matrix_per_family_json
        for entry in target_output:
            family_info_list = json.loads(entry["matrix_per_family_json"])
            self.assertTrue(all("amdgpu_family" in f for f in family_info_list))
            self.assertTrue(all("test-runs-on" in f for f in family_info_list))
            self.assertTrue(
                all("sanity_check_only_for_family" in f for f in family_info_list)
            )

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
        # This supports multi_arch_ci.yml which triggers on multi_arch/** branches
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

    ###########################################################################
    # Tests for multi_arch mode

    def test_multi_arch_linux_workflow_dispatch_matrix_generator(self):
        """Test multi_arch mode groups all families into one entry with test-runs-on."""
        import json

        build_families = {"amdgpu_families": "gfx94X, gfx110X"}
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
            multi_arch=True,
        )
        # Multi-arch should produce one entry per build_variant, not per family
        self.assertEqual(len(linux_target_output), 1)
        self.assert_multi_arch_output_is_valid(
            target_output=linux_target_output, allow_xfail=True
        )

        # Check that both families are in the output with structured format
        entry = linux_target_output[0]
        family_info_list = json.loads(entry["matrix_per_family_json"])
        family_names = [f["amdgpu_family"] for f in family_info_list]
        self.assertIn("gfx94X-dcgpu", family_names)
        self.assertIn("gfx110X-all", family_names)

        # Verify test-runs-on is populated for each family
        for family_info in family_info_list:
            self.assertIn("test-runs-on", family_info)

        # Check dist_amdgpu_families is semicolon-separated
        dist_families = entry["dist_amdgpu_families"].split(";")
        self.assertIn("gfx94X-dcgpu", dist_families)
        self.assertIn("gfx110X-all", dist_families)

        self.assertEqual(linux_test_labels, [])

    def test_multi_arch_single_family_linux_workflow_dispatch(self):
        """Test multi_arch mode with single family produces one entry."""
        import json

        build_families = {"amdgpu_families": "gfx94X"}
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
            multi_arch=True,
        )
        self.assertEqual(len(linux_target_output), 1)
        self.assert_multi_arch_output_is_valid(
            target_output=linux_target_output, allow_xfail=True
        )

        entry = linux_target_output[0]
        family_info_list = json.loads(entry["matrix_per_family_json"])
        self.assertEqual(len(family_info_list), 1)
        self.assertEqual(family_info_list[0]["amdgpu_family"], "gfx94X-dcgpu")

    def test_multi_arch_empty_families_linux_workflow_dispatch(self):
        """Test multi_arch mode with empty families produces empty output."""
        build_families = {"amdgpu_families": ""}
        linux_target_output, linux_test_labels = configure_ci.matrix_generator(
            is_pull_request=False,
            is_workflow_dispatch=True,
            is_push=False,
            is_schedule=False,
            base_args={"build_variant": "release"},
            families=build_families,
            platform="linux",
            multi_arch=True,
        )
        self.assertEqual(linux_target_output, [])
        self.assertEqual(linux_test_labels, [])

    def test_multi_arch_postsubmit_matrix_generator(self):
        """Test multi_arch mode with postsubmit (main branch push)."""
        import json

        base_args = {"branch_name": "main", "build_variant": "release"}
        linux_target_output, linux_test_labels = configure_ci.matrix_generator(
            is_pull_request=False,
            is_workflow_dispatch=False,
            is_push=True,
            is_schedule=False,
            base_args=base_args,
            families={},
            platform="linux",
            multi_arch=True,
        )
        # Should produce one entry with all postsubmit families grouped
        self.assertEqual(len(linux_target_output), 1)
        self.assert_multi_arch_output_is_valid(
            target_output=linux_target_output, allow_xfail=True
        )

        entry = linux_target_output[0]
        family_info_list = json.loads(entry["matrix_per_family_json"])
        # Postsubmit should have multiple families
        self.assertGreaterEqual(len(family_info_list), 1)
        # Each entry should have amdgpu_family and test-runs-on
        for family_info in family_info_list:
            self.assertIn("amdgpu_family", family_info)
            self.assertIn("test-runs-on", family_info)

    def test_multi_arch_sanity_check_field_propagation_logic(self):
        """Unit test: Verify sanity_check_only_for_family field is correctly propagated.

        Uses synthetic data to test the code logic in isolation.
        This test should never need updates unless the code behavior changes.
        """
        # Synthetic minimal test matrix
        # Use naming convention matching real matrix (e.g., gfx94x, gfx110x - no underscores)
        synthetic_matrix = {
            "testfamily1": {
                "linux": {
                    "family": "testfamily1-stable",
                    "test-runs-on": "linux-stable-runner",
                    "build_variants": ["release"],
                    # Field not present - should default to False
                }
            },
            "testfamily2": {
                "linux": {
                    "family": "testfamily2-experimental",
                    "test-runs-on": "linux-experimental-runner",
                    "build_variants": ["release"],
                    "sanity_check_only_for_family": True,
                }
            },
            "testfamily3": {
                "linux": {
                    "family": "testfamily3-explicit-false",
                    "test-runs-on": "linux-another-runner",
                    "build_variants": ["release"],
                    "sanity_check_only_for_family": False,  # Explicit False
                }
            },
        }

        with patch(
            "configure_ci.get_all_families_for_trigger_types",
            return_value=synthetic_matrix,
        ):
            build_families = {
                "amdgpu_families": "testfamily1, testfamily2, testfamily3"
            }
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
                multi_arch=True,
            )

            # Validate multi-arch structure
            self.assertEqual(len(linux_target_output), 1)
            self.assert_multi_arch_output_is_valid(
                target_output=linux_target_output, allow_xfail=True
            )

            # Parse and validate field propagation
            entry = linux_target_output[0]
            family_info_list = json.loads(entry["matrix_per_family_json"])
            self.assertEqual(len(family_info_list), 3)

            family_dict = {f["amdgpu_family"]: f for f in family_info_list}

            # Verify field is correctly propagated with proper defaults
            self.assertIn("testfamily1-stable", family_dict)
            self.assertFalse(
                family_dict["testfamily1-stable"]["sanity_check_only_for_family"],
                "Missing field should default to False",
            )

            self.assertIn("testfamily2-experimental", family_dict)
            self.assertTrue(
                family_dict["testfamily2-experimental"]["sanity_check_only_for_family"],
                "Explicit True should be preserved",
            )

            self.assertIn("testfamily3-explicit-false", family_dict)
            self.assertFalse(
                family_dict["testfamily3-explicit-false"][
                    "sanity_check_only_for_family"
                ],
                "Explicit False should be preserved",
            )

            # Verify all entries have the field (even if False)
            for family_info in family_info_list:
                self.assertIn("sanity_check_only_for_family", family_info)
                self.assertIsInstance(family_info["sanity_check_only_for_family"], bool)

    def test_multi_arch_production_sanity_check_configuration(self):
        """Integration test: Verify production matrix sanity_check configuration.

        This documents our expected production configuration and catches unintentional changes.

        When this test fails:
        1. Check if the architecture matured (expected) → update expected_families
        2. Check if someone accidentally changed the matrix (bug) → revert the change

        Update this test when architectures are promoted/demoted intentionally.
        """
        # Get actual production matrix
        matrix = configure_ci.get_all_families_for_trigger_types(["presubmit"])

        # Document expected production configuration as of 2025-02
        # Update these when architectures mature or new experimental archs are added
        expected_families = {
            # Stable architectures - should NOT have sanity_check flag
            "stable": ["gfx94x"],
            # Experimental architectures - SHOULD have sanity_check flag
            "experimental": ["gfx110x", "gfx1151"],
        }

        # Verify stable architectures
        for family in expected_families["stable"]:
            if family not in matrix:
                self.fail(
                    f"Stable family '{family}' not in presubmit matrix. "
                    f"If removed intentionally, update expected_families in this test."
                )
            linux_info = matrix[family].get("linux", {})
            sanity_check = linux_info.get("sanity_check_only_for_family", False)
            self.assertFalse(
                sanity_check,
                f"Stable family '{family}' should not have sanity_check_only_for_family=True",
            )

        # Verify experimental architectures
        for family in expected_families["experimental"]:
            if family not in matrix:
                # Allow experimental families to be removed without breaking CI
                print(
                    f"WARNING: Experimental family '{family}' not in matrix (may have been promoted/removed)"
                )
                continue
            linux_info = matrix[family].get("linux", {})
            sanity_check = linux_info.get("sanity_check_only_for_family", False)
            self.assertTrue(
                sanity_check,
                f"Experimental family '{family}' should have sanity_check_only_for_family=True. "
                f"If promoted to stable, move to 'stable' list in expected_families.",
            )

        # Now test end-to-end: pick one stable + one experimental and verify propagation
        if not expected_families["stable"] or not expected_families["experimental"]:
            self.skipTest("Need at least one stable and one experimental family")

        stable_family = expected_families["stable"][0]
        experimental_family = expected_families["experimental"][0]

        # Skip if experimental family was removed
        if experimental_family not in matrix:
            self.skipTest(f"Experimental family {experimental_family} not available")

        build_families = {"amdgpu_families": f"{stable_family}, {experimental_family}"}
        linux_target_output, _ = configure_ci.matrix_generator(
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
            multi_arch=True,
        )

        self.assertEqual(len(linux_target_output), 1)
        self.assert_multi_arch_output_is_valid(
            target_output=linux_target_output, allow_xfail=True
        )

        # Verify the production values are correctly propagated
        entry = linux_target_output[0]
        family_info_list = json.loads(entry["matrix_per_family_json"])

        stable_arch_name = matrix[stable_family]["linux"]["family"]
        experimental_arch_name = matrix[experimental_family]["linux"]["family"]

        family_dict = {f["amdgpu_family"]: f for f in family_info_list}

        self.assertIn(stable_arch_name, family_dict)
        self.assertFalse(
            family_dict[stable_arch_name]["sanity_check_only_for_family"],
            f"Stable family {stable_arch_name} should have sanity_check=False",
        )

        self.assertIn(experimental_arch_name, family_dict)
        self.assertTrue(
            family_dict[experimental_arch_name]["sanity_check_only_for_family"],
            f"Experimental family {experimental_arch_name} should have sanity_check=True",
        )

    def test_rocm_org_var_names(self):
        os.environ["LOAD_TEST_RUNNERS_FROM_VAR"] = "false"
        test_matrix = configure_ci.get_all_families_for_trigger_types(["presubmit"])
        self.assertIn("linux-gfx110X-gpu-rocm-test", json.dumps(test_matrix))
        self.assertIn("windows-gfx110X-gpu-rocm-test", json.dumps(test_matrix))


if __name__ == "__main__":
    unittest.main()
