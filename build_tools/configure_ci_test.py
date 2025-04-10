from unittest import TestCase, main
import os

import configure_ci


class ConfigureCITest(TestCase):
    def test_run_ci_if_source_file_edited(self):
        paths = ["source_file.h"]
        run_ci = configure_ci.should_ci_run_given_modified_paths(paths)
        self.assertTrue(run_ci)

    def test_dont_run_ci_if_only_markdown_files_edited(self):
        paths = ["README.md", "build_tools/README.md"]
        run_ci = configure_ci.should_ci_run_given_modified_paths(paths)
        self.assertFalse(run_ci)

    def test_run_ci_if_related_workflow_file_edited(self):
        paths = [".github/workflows/ci.yml"]
        run_ci = configure_ci.should_ci_run_given_modified_paths(paths)
        self.assertTrue(run_ci)

    def test_dont_run_ci_if_unrelated_workflow_file_edited(self):
        paths = [".github/workflows/publish_pytorch_dev_docker.yml"]
        run_ci = configure_ci.should_ci_run_given_modified_paths(paths)
        self.assertFalse(run_ci)

    def test_run_ci_if_source_file_and_unrelated_workflow_file_edited(self):
        paths = ["source_file.h", ".github/workflows/publish_pytorch_dev_docker.yml"]
        run_ci = configure_ci.should_ci_run_given_modified_paths(paths)
        self.assertTrue(run_ci)

    def test_valid_workflow_dispatch_matrix_generator(self):
        build_families = {
            "input_linux_amdgpu_families": "   gfx94X ,|.\\,  gfx1201X, --   gfx90X",
            "input_windows_amdgpu_families": "gfx94X \\., gfx1201X  gfx90X",
        }
        linux_target_output, windows_target_output = configure_ci.matrix_generator(
            False, True, False, {}, build_families, False
        )
        linux_target_to_compare = [
            {
                "family": "gfx94X-dcgpu",
                "pytorch-target": "gfx942",
                "test-runs-on": "linux-mi300-1gpu-ossci-rocm",
            }
        ]
        self.assertEqual(linux_target_output, linux_target_to_compare)
        self.assertEqual(windows_target_output, [])

    def test_invalid_workflow_dispatch_matrix_generator(self):
        build_families = {
            "input_linux_amdgpu_families": "",
            "input_windows_amdgpu_families": "",
        }
        linux_target_output, windows_target_output = configure_ci.matrix_generator(
            False, True, False, {}, build_families, False
        )
        self.assertEqual(linux_target_output, [])
        self.assertEqual(windows_target_output, [])

    def test_valid_pull_request_matrix_generator(self):
        base_args = {
            "pr_labels": '["gfx94X-linux", "gfx1201X-windows", "gfx94X-windows", "gfx1201-linux"]'
        }
        linux_target_output, windows_target_output = configure_ci.matrix_generator(
            True, False, False, base_args, {}, False
        )
        linux_target_to_compare = [
            {"test-runs-on": "", "family": "gfx110X-dgpu", "pytorch-target": "gfx1100"},
            {
                "test-runs-on": "linux-mi300-1gpu-ossci-rocm",
                "family": "gfx94X-dcgpu",
                "pytorch-target": "gfx942",
            },
        ]
        windows_target_to_compare = [{"test-runs-on": "", "family": "gfx110X-dgpu"}]
        self.assertEqual(linux_target_output, linux_target_to_compare)
        self.assertEqual(windows_target_output, windows_target_to_compare)

    def test_duplicate_pull_request_matrix_generator(self):
        base_args = {
            "pr_labels": '["gfx94X-linux", "gfx94X-linux", "gfx1201X-windows", "gfx94X-windows", "gfx1201-linux"]'
        }
        linux_target_output, windows_target_output = configure_ci.matrix_generator(
            True, False, False, base_args, {}, False
        )
        linux_target_to_compare = [
            {"test-runs-on": "", "family": "gfx110X-dgpu", "pytorch-target": "gfx1100"},
            {
                "test-runs-on": "linux-mi300-1gpu-ossci-rocm",
                "family": "gfx94X-dcgpu",
                "pytorch-target": "gfx942",
            },
        ]
        windows_target_to_compare = [{"test-runs-on": "", "family": "gfx110X-dgpu"}]
        self.assertEqual(linux_target_output, linux_target_to_compare)
        self.assertEqual(windows_target_output, windows_target_to_compare)

    def test_invalid_pull_request_matrix_generator(self):
        base_args = {"pr_labels": '["gfx942X-windows", "gfx1201-linux"]'}
        linux_target_output, windows_target_output = configure_ci.matrix_generator(
            True, False, False, base_args, {}, False
        )
        linux_target_to_compare = [
            {"test-runs-on": "", "family": "gfx110X-dgpu", "pytorch-target": "gfx1100"},
            {
                "test-runs-on": "linux-mi300-1gpu-ossci-rocm",
                "family": "gfx94X-dcgpu",
                "pytorch-target": "gfx942",
            },
        ]
        windows_target_to_compare = [{"test-runs-on": "", "family": "gfx110X-dgpu"}]
        self.assertEqual(linux_target_output, linux_target_to_compare)
        self.assertEqual(windows_target_output, windows_target_to_compare)

    def test_empty_pull_request_matrix_generator(self):
        base_args = {"pr_labels": "[]"}
        linux_target_output, windows_target_output = configure_ci.matrix_generator(
            True, False, False, base_args, {}, False
        )
        linux_target_to_compare = [
            {"test-runs-on": "", "family": "gfx110X-dgpu", "pytorch-target": "gfx1100"},
            {
                "test-runs-on": "linux-mi300-1gpu-ossci-rocm",
                "family": "gfx94X-dcgpu",
                "pytorch-target": "gfx942",
            },
        ]
        windows_target_to_compare = [{"test-runs-on": "", "family": "gfx110X-dgpu"}]
        self.assertEqual(linux_target_output, linux_target_to_compare)
        self.assertEqual(windows_target_output, windows_target_to_compare)

    def test_main_branch_push_matrix_generator(self):
        base_args = {"branch_name": "main"}
        linux_target_output, windows_target_output = configure_ci.matrix_generator(
            False, False, True, base_args, {}, False
        )
        linux_target_to_compare = [
            {"test-runs-on": "linux-mi300-1gpu-ossci-rocm", "family": "gfx94X-dcgpu"},
            {"test-runs-on": "", "family": "gfx110X-dgpu"},
        ]
        windows_target_to_compare = [{"test-runs-on": "", "family": "gfx110X-dgpu"}]
        self.assertEqual(linux_target_output, linux_target_to_compare)
        self.assertEqual(windows_target_output, windows_target_to_compare)

    def test_main_branch_push_matrix_generator(self):
        base_args = {"branch_name": "test_branch"}
        build_families = {
            "input_linux_amdgpu_families": "   gfx94X ,|.\\,  gfx1201X, --   gfx90X",
            "input_windows_amdgpu_families": "gfx94X \\., gfx1201X  gfx90X",
        }
        linux_target_output, windows_target_output = configure_ci.matrix_generator(
            False, False, True, base_args, build_families, False
        )
        self.assertEqual(linux_target_output, [])
        self.assertEqual(windows_target_output, [])


if __name__ == "__main__":
    main()
