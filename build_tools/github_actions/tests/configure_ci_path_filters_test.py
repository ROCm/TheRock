from pathlib import Path
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from configure_ci_path_filters import is_ci_run_required


class ConfigureCIPathFiltersTest(unittest.TestCase):
    def test_run_ci_if_source_file_edited(self):
        paths = ["source_file.h"]
        run_ci = is_ci_run_required(paths)
        self.assertTrue(run_ci)

    def test_dont_run_ci_if_only_markdown_files_edited(self):
        paths = ["README.md", "build_tools/README.md"]
        run_ci = is_ci_run_required(paths)
        self.assertFalse(run_ci)

    def test_dont_run_ci_if_only_experimental_files_edited(self):
        paths = ["experimental/file.h"]
        run_ci = is_ci_run_required(paths)
        self.assertFalse(run_ci)

    def test_run_ci_if_related_workflow_file_edited(self):
        paths = [".github/workflows/ci.yml"]
        run_ci = is_ci_run_required(paths)
        self.assertTrue(run_ci)

        paths = [".github/workflows/build_portable_linux_artifacts.yml"]
        run_ci = is_ci_run_required(paths)
        self.assertTrue(run_ci)

        paths = [".github/workflows/build_artifact.yml"]
        run_ci = is_ci_run_required(paths)
        self.assertTrue(run_ci)

    def test_dont_run_ci_if_unrelated_workflow_file_edited(self):
        paths = [".github/workflows/pre-commit.yml"]
        run_ci = is_ci_run_required(paths)
        self.assertFalse(run_ci)

        paths = [".github/workflows/test_jax_dockerfile.yml"]
        run_ci = is_ci_run_required(paths)
        self.assertFalse(run_ci)

    def test_run_ci_if_source_file_and_unrelated_workflow_file_edited(self):
        paths = ["source_file.h", ".github/workflows/pre-commit.yml"]
        run_ci = is_ci_run_required(paths)
        self.assertTrue(run_ci)


if __name__ == "__main__":
    unittest.main()
