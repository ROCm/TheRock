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
        args = {
            "input_linux_amdgpu_families": "   gfx94X ,|.\\,  gfx1201X, --   gfx90X",
            "input_windows_amdgpu_families": "gfx94X \\., gfx1201X  gfx90X",
        }
        linux_target_output, windows_target_output = configure_ci.matrix_generator(
            False, True, False, args
        )
        linux_target_to_compare = [
            {"runs-on": "linux-mi300-1gpu-ossci-rocm", "target": "gfx94X-dcgpu"}
        ]
        self.assertEqual(linux_target_output, linux_target_to_compare)
        self.assertEqual(windows_target_output, [])

    def test_invalid_workflow_dispatch_matrix_generator(self):
        args = {"input_linux_amdgpu_families": "", "input_windows_amdgpu_families": ""}
        linux_target_output, windows_target_output = configure_ci.matrix_generator(
            False, True, False, args
        )
        self.assertEqual(linux_target_output, [])
        self.assertEqual(windows_target_output, [])

    def test_valid_pull_request_matrix_generator(self):
        args = {
            "pr_labels": '["gfx94X-linux", "gfx1201X-windows", "gfx94X-windows", "gfx1201-linux"]'
        }
        linux_target_output, windows_target_output = configure_ci.matrix_generator(
            True, False, False, args
        )
        linux_target_to_compare = [
            {"runs-on": "linux-mi300-1gpu-ossci-rocm", "target": "gfx94X-dcgpu"}
        ]
        self.assertEqual(linux_target_output, linux_target_to_compare)
        self.assertEqual(windows_target_output, [])

    def test_duplicate_pull_request_matrix_generator(self):
        args = {
            "pr_labels": '["gfx94X-linux", "gfx94X-linux", "gfx1201X-windows", "gfx94X-windows", "gfx1201-linux"]'
        }
        linux_target_output, windows_target_output = configure_ci.matrix_generator(
            True, False, False, args
        )
        linux_target_to_compare = [
            {"runs-on": "linux-mi300-1gpu-ossci-rocm", "target": "gfx94X-dcgpu"}
        ]
        self.assertEqual(linux_target_output, linux_target_to_compare)
        self.assertEqual(windows_target_output, [])

    def test_invalid_pull_request_matrix_generator(self):
        args = {"pr_labels": '["gfx942X-windows", "gfx1201-linux"]'}
        linux_target_output, windows_target_output = configure_ci.matrix_generator(
            True, False, False, args
        )
        self.assertEqual(linux_target_output, [])
        self.assertEqual(windows_target_output, [])

    def test_empty_pull_request_matrix_generator(self):
        args = {"pr_labels": "[]"}
        linux_target_output, windows_target_output = configure_ci.matrix_generator(
            True, False, False, args
        )
        self.assertEqual(linux_target_output, [])
        self.assertEqual(windows_target_output, [])

    def test_main_branch_push_matrix_generator(self):
        args = {"branch_name": "main"}
        linux_target_output, windows_target_output = configure_ci.matrix_generator(
            False, False, True, args
        )
        linux_target_to_compare = [
            {"runs-on": "linux-mi300-1gpu-ossci-rocm", "target": "gfx94X-dcgpu"}
        ]
        self.assertEqual(linux_target_output, linux_target_to_compare)
        self.assertEqual(windows_target_output, [])

    def test_main_branch_push_matrix_generator(self):
        args = {"branch_name": "test_branch"}
        linux_target_output, windows_target_output = configure_ci.matrix_generator(
            False, False, True, args
        )
        self.assertEqual(linux_target_output, [])
        self.assertEqual(windows_target_output, [])


if __name__ == "__main__":
    main()
