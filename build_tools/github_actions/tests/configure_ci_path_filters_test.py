# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from configure_ci_path_filters import is_ci_run_required, _GITHUB_WORKFLOWS_CI_FILENAMES
from workflow_utils import get_transitive_workflow_uses


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

        paths = [".github/workflows/build_native_linux_packages.yml"]
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

    def test_ci_workflow_filenames_cover_all_transitive_uses(self):
        """_GITHUB_WORKFLOWS_CI_FILENAMES must exactly match the set of
        workflows transitively called by ci.yml and multi_arch_ci.yml.

        This is a change-detector test that can be removed if
        _GITHUB_WORKFLOWS_CI_FILENAMES is computed dynamically instead of
        maintained by hand.

        If this test fails, update _GITHUB_WORKFLOWS_CI_FILENAMES in
        configure_ci_path_filters.py to match the actual workflow tree.
        """
        all_used = get_transitive_workflow_uses(["ci.yml", "multi_arch_ci.yml"])
        missing = all_used - _GITHUB_WORKFLOWS_CI_FILENAMES
        stale = _GITHUB_WORKFLOWS_CI_FILENAMES - all_used
        errors = []
        if missing:
            errors.append(
                "Missing (add to _GITHUB_WORKFLOWS_CI_FILENAMES):\n"
                + "\n".join(f"  - {f}" for f in sorted(missing))
            )
        if stale:
            errors.append(
                "Stale (remove from _GITHUB_WORKFLOWS_CI_FILENAMES):\n"
                + "\n".join(f"  - {f}" for f in sorted(stale))
            )
        if errors:
            self.fail("\n".join(errors))


if __name__ == "__main__":
    unittest.main()
