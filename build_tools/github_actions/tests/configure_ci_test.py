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
        target_names = configure_ci.filter_known_target_names(requested_target_names)
        self.assertIn("gfx110x", target_names)
        self.assertNotIn("abcdef", target_names)

    def test_valid_linux_workflow_dispatch_matrix_generator(self):
        build_families = {"amdgpu_families": "   gfx94X , gfx103X"}
        linux_target_output = configure_ci.matrix_generator(
            is_pull_request=False,
            is_workflow_dispatch=True,
            is_push=False,
            is_schedule=False,
            base_args={},
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

    def test_invalid_linux_workflow_dispatch_matrix_generator(self):
        build_families = {
            "amdgpu_families": "",
        }
        linux_target_output = configure_ci.matrix_generator(
            is_pull_request=False,
            is_workflow_dispatch=True,
            is_push=False,
            is_schedule=False,
            base_args={},
            families=build_families,
            platform="linux",
        )
        self.assertEqual(linux_target_output, [])

    def test_valid_linux_pull_request_matrix_generator(self):
        base_args = {
            "pr_labels": '{"labels":[{"name":"gfx94X-linux"},{"name":"gfx110X-linux"},{"name":"gfx110X-windows"}]}'
        }
        linux_target_output = configure_ci.matrix_generator(
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
            any("gfx110X-dgpu" == entry["family"] for entry in linux_target_output)
        )
        self.assertGreaterEqual(len(linux_target_output), 2)
        self.assert_target_output_is_valid(
            target_output=linux_target_output, allow_xfail=False
        )

    def test_duplicate_windows_pull_request_matrix_generator(self):
        base_args = {
            "pr_labels": '{"labels":[{"name":"gfx94X-linux"},{"name":"gfx110X-linux"},{"name":"gfx110X-windows"},{"name":"gfx110X-windows"}]}'
        }
        windows_target_output = configure_ci.matrix_generator(
            is_pull_request=True,
            is_workflow_dispatch=False,
            is_push=False,
            is_schedule=False,
            base_args=base_args,
            families={},
            platform="windows",
        )
        self.assertTrue(
            any("gfx110X-dgpu" == entry["family"] for entry in windows_target_output)
        )
        self.assertGreaterEqual(len(windows_target_output), 1)
        self.assert_target_output_is_valid(
            target_output=windows_target_output, allow_xfail=False
        )

    def test_invalid_linux_pull_request_matrix_generator(self):
        base_args = {
            "pr_labels": '{"labels":[{"name":"gfx10000X-linux"},{"name":"gfx110000X-windows"}]}'
        }
        linux_target_output = configure_ci.matrix_generator(
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

    def test_empty_windows_pull_request_matrix_generator(self):
        base_args = {"pr_labels": "{}"}
        windows_target_output = configure_ci.matrix_generator(
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

    def test_main_linux_branch_push_matrix_generator(self):
        base_args = {"branch_name": "main"}
        linux_target_output = configure_ci.matrix_generator(
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
            target_output=linux_target_output, allow_xfail=False
        )

    def test_main_windows_branch_push_matrix_generator(self):
        base_args = {"branch_name": "main"}
        windows_target_output = configure_ci.matrix_generator(
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

    def test_linux_branch_push_matrix_generator(self):
        base_args = {"branch_name": "test_branch"}
        linux_target_output = configure_ci.matrix_generator(
            is_pull_request=False,
            is_workflow_dispatch=False,
            is_push=True,
            is_schedule=False,
            base_args=base_args,
            families={},
            platform="linux",
        )
        self.assertEqual(len(linux_target_output), 0)

    def test_linux_schedule_matrix_generator(self):
        linux_target_output = configure_ci.matrix_generator(
            is_pull_request=False,
            is_workflow_dispatch=False,
            is_push=False,
            is_schedule=True,
            base_args={},
            families={},
            platform="linux",
        )
        self.assertGreaterEqual(len(linux_target_output), 1)
        self.assert_target_output_is_valid(
            target_output=linux_target_output, allow_xfail=True
        )

    def test_windows_schedule_matrix_generator(self):
        windows_target_output = configure_ci.matrix_generator(
            is_pull_request=False,
            is_workflow_dispatch=False,
            is_push=False,
            is_schedule=True,
            base_args={},
            families={},
            platform="windows",
        )
        self.assertGreaterEqual(len(windows_target_output), 1)
        self.assert_target_output_is_valid(
            target_output=windows_target_output, allow_xfail=True
        )


if __name__ == "__main__":
    unittest.main()
