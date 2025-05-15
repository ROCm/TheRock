from unittest import TestCase, main
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
        paths = [".github/workflows/build_package.yml"]
        run_ci = configure_ci.should_ci_run_given_modified_paths(paths)
        self.assertTrue(run_ci)
        paths = [".github/workflows/test_some_subproject.yml"]
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

    def test_valid_linux_workflow_dispatch_matrix_generator(self):
        build_families = {"amdgpu_families": "   gfx94X ,|.\\,  gfx1200"}
        linux_target_output = configure_ci.matrix_generator(
            False, True, False, {}, build_families, "linux"
        )
        target_to_compare = [
            {
                "test-runs-on": "linux-mi300-1gpu-ossci-rocm-test",
                "family": "gfx94X-dcgpu",
                "pytorch-target": "gfx942",
            }
        ]
        self.assertEqual(linux_target_output, target_to_compare)

    def test_invalid_linux_workflow_dispatch_matrix_generator(self):
        build_families = {
            "amdgpu_families": "",
        }
        linux_target_output = configure_ci.matrix_generator(
            False, True, False, {}, build_families, "linux"
        )
        self.assertEqual(linux_target_output, [])

    def test_valid_linux_pull_request_matrix_generator(self):
        base_args = {
            "pr_labels": '{"labels":[{"name":"gfx94X-linux"},{"name":"gfx110X-linux"},{"name":"gfx110X-windows"}]}'
        }
        linux_target_output = configure_ci.matrix_generator(
            True, False, False, base_args, {}, "linux"
        )
        linux_target_output.sort(key=lambda item: item["family"])
        linux_target_to_compare = [
            {"test-runs-on": "", "family": "gfx110X-dgpu", "pytorch-target": "gfx1100"},
            {
                "test-runs-on": "linux-mi300-1gpu-ossci-rocm-test",
                "family": "gfx94X-dcgpu",
                "pytorch-target": "gfx942",
            },
        ]
        linux_target_to_compare.sort(key=lambda item: item["family"])
        self.assertEqual(linux_target_output, linux_target_to_compare)

    def test_duplicate_windows_pull_request_matrix_generator(self):
        base_args = {
            "pr_labels": '{"labels":[{"name":"gfx94X-linux"},{"name":"gfx110X-linux"},{"name":"gfx110X-windows"},{"name":"gfx110X-windows"}]}'
        }
        windows_target_output = configure_ci.matrix_generator(
            True, False, False, base_args, {}, "windows"
        )
        windows_target_to_compare = [{"test-runs-on": "", "family": "gfx110X-dgpu"}]
        self.assertEqual(windows_target_output, windows_target_to_compare)

    def test_invalid_linux_pull_request_matrix_generator(self):
        base_args = {
            "pr_labels": '{"labels":[{"name":"gfx10000X-linux"},{"name":"gfx110000X-windows"}]}'
        }
        linux_target_output = configure_ci.matrix_generator(
            True, False, False, base_args, {}, "linux"
        )
        linux_target_output.sort(key=lambda item: item["family"])
        linux_target_to_compare = [
            {"test-runs-on": "", "family": "gfx110X-dgpu", "pytorch-target": "gfx1100"},
            {
                "test-runs-on": "linux-mi300-1gpu-ossci-rocm-test",
                "family": "gfx94X-dcgpu",
                "pytorch-target": "gfx942",
            },
        ]
        linux_target_to_compare.sort(key=lambda item: item["family"])
        self.assertEqual(linux_target_output, linux_target_to_compare)

    def test_empty_windows_pull_request_matrix_generator(self):
        base_args = {"pr_labels": "{}"}
        windows_target_output = configure_ci.matrix_generator(
            True, False, False, base_args, {}, "windows"
        )
        windows_target_to_compare = [{"test-runs-on": "", "family": "gfx110X-dgpu"}]
        self.assertEqual(windows_target_output, windows_target_to_compare)

    def test_main_linux_branch_push_matrix_generator(self):
        base_args = {"branch_name": "main"}
        linux_target_output = configure_ci.matrix_generator(
            False, False, True, base_args, {}, "linux"
        )
        linux_target_output.sort(key=lambda item: item["family"])
        linux_target_to_compare = [
            {"test-runs-on": "", "family": "gfx90X-dcgpu", "pytorch-target": "gfx90a"},
            {
                "test-runs-on": "linux-rx9070-gpu-rocm",
                "family": "gfx120X-all",
                "pytorch-target": "gfx1201",
            },
            {
                "test-runs-on": "linux-mi300-1gpu-ossci-rocm-test",
                "family": "gfx94X-dcgpu",
                "pytorch-target": "gfx942",
            },
            {"test-runs-on": "", "family": "gfx103X-dgpu", "pytorch-target": "gfx1030"},
            {"test-runs-on": "", "family": "gfx101X-dgpu", "pytorch-target": "gfx1010"},
            {"test-runs-on": "", "family": "gfx110X-dgpu", "pytorch-target": "gfx1100"},
            {"test-runs-on": "", "family": "gfx1151", "pytorch-target": "gfx1151"},
        ]
        linux_target_to_compare.sort(key=lambda item: item["family"])
        self.assertEqual(linux_target_output, linux_target_to_compare)

    def test_main_windows_branch_push_matrix_generator(self):
        base_args = {"branch_name": "main"}
        windows_target_output = configure_ci.matrix_generator(
            False, False, True, base_args, {}, "windows"
        )
        windows_target_output.sort(key=lambda item: item["family"])
        windows_target_to_compare = [{"test-runs-on": "", "family": "gfx110X-dgpu"}]
        windows_target_to_compare.sort(key=lambda item: item["family"])
        self.assertEqual(windows_target_output, windows_target_to_compare)

    def test_linux_branch_push_matrix_generator(self):
        base_args = {"branch_name": "test_branch"}
        build_families = {
            "amdgpu_families": "   gfx94X ,|.\\,  gfx1201X, --   gfx90X",
        }
        linux_target_output = configure_ci.matrix_generator(
            False, False, True, base_args, build_families, "linux"
        )
        self.assertEqual(linux_target_output, [])


if __name__ == "__main__":
    main()
