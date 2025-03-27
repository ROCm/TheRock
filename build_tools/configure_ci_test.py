from unittest import mock, TestCase, main
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

    @mock.patch.dict(
        os.environ,
        {
            "INPUT_LINUX_AMDGPU_FAMILIES": "   gfx94X ,|.\\,  gfx1201X, --   gfx90X",
            "INPUT_WINDOWS_AMDGPU_FAMILIES": "gfx94X \\., gfx1201X  gfx90X",
        },
    )
    def test_valid_workflow_dispatch_matrix_generator(self):
        linux_target_output, windows_target_output = configure_ci.matrix_generator(
            False, True, False
        )
        linux_target_to_compare = [
            {"runs-on": "linux-mi300-1gpu-ossci-rocm", "target": "gfx94X-dcgpu"}
        ]
        self.assertEqual(linux_target_output, linux_target_to_compare)
        self.assertEqual(windows_target_output, [])

    def test_invalid_workflow_dispatch_matrix_generator(self):
        linux_target_output, windows_target_output = configure_ci.matrix_generator(
            False, True, False
        )
        self.assertEqual(linux_target_output, [])
        self.assertEqual(windows_target_output, [])

    @mock.patch.dict(
        os.environ,
        {
            "PR_LABELS": '["gfx94X-linux", "gfx1201X-windows", "gfx94X-windows", "gfx1201-linux"]'
        },
    )
    def test_valid_pull_request_matrix_generator(self):
        linux_target_output, windows_target_output = configure_ci.matrix_generator(
            True, False, False
        )
        linux_target_to_compare = [
            {"runs-on": "linux-mi300-1gpu-ossci-rocm", "target": "gfx94X-dcgpu"}
        ]
        self.assertEqual(linux_target_output, linux_target_to_compare)
        self.assertEqual(windows_target_output, [])

    @mock.patch.dict(os.environ, {"PR_LABELS": '["gfx942X-windows", "gfx1201-linux"]'})
    def test_invalid_pull_request_matrix_generator(self):
        linux_target_output, windows_target_output = configure_ci.matrix_generator(
            True, False, False
        )
        self.assertEqual(linux_target_output, [])
        self.assertEqual(windows_target_output, [])


if __name__ == "__main__":
    main()
