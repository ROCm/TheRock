from pathlib import Path
import os
import sys
import unittest

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import configure_ci


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

        if not allow_xfail:
            self.assertFalse(
                any(entry.get("expect_failure") for entry in target_output)
            )

    ###########################################################################
    # Tests for should_ci_run_given_modified_paths

    def test_run_ci_if_source_file_edited(self):
        paths = ["source_file.h"]
        run_ci = configure_ci.should_ci_run_given_modified_paths(paths)
        self.assertTrue(run_ci)

    def test_dont_run_ci_if_only_markdown_files_edited(self):
        paths = ["README.md", "build_tools/README.md"]
        run_ci = configure_ci.should_ci_run_given_modified_paths(paths)
        self.assertFalse(run_ci)

    def test_dont_run_ci_if_only_external_builds_edited(self):
        paths = ["external-builds/pytorch/CMakeLists.txt"]
        run_ci = configure_ci.should_ci_run_given_modified_paths(paths)
        self.assertFalse(run_ci)

    def test_dont_run_ci_if_only_external_builds_edited(self):
        paths = ["experimental/file.h"]
        run_ci = configure_ci.should_ci_run_given_modified_paths(paths)
        self.assertFalse(run_ci)

    def test_run_ci_if_related_workflow_file_edited(self):
        paths = [".github/workflows/ci.yml"]
        run_ci = configure_ci.should_ci_run_given_modified_paths(paths)
        self.assertTrue(run_ci)

        paths = [".github/workflows/build_portable_linux_artifacts.yml"]
        run_ci = configure_ci.should_ci_run_given_modified_paths(paths)
        self.assertTrue(run_ci)

        paths = [".github/workflows/build_artifact.yml"]
        run_ci = configure_ci.should_ci_run_given_modified_paths(paths)
        self.assertTrue(run_ci)

    def test_dont_run_ci_if_unrelated_workflow_file_edited(self):
        paths = [".github/workflows/pre-commit.yml"]
        run_ci = configure_ci.should_ci_run_given_modified_paths(paths)
        self.assertFalse(run_ci)

        paths = [".github/workflows/test_jax_dockerfile.yml"]
        run_ci = configure_ci.should_ci_run_given_modified_paths(paths)
        self.assertFalse(run_ci)

    def test_run_ci_if_source_file_and_unrelated_workflow_file_edited(self):
        paths = ["source_file.h", ".github/workflows/pre-commit.yml"]
        run_ci = configure_ci.should_ci_run_given_modified_paths(paths)
        self.assertTrue(run_ci)

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


if __name__ == "__main__":
    unittest.main()
