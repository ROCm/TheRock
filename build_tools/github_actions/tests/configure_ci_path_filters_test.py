# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from configure_ci_path_filters import (
    _GITHUB_WORKFLOWS_CI_FILENAMES,
    get_git_commit_hash,
    get_git_modified_paths,
    is_ci_run_required,
)
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

    def test_dont_run_ci_if_only_skipped_files_edited(self):
        paths = ["gitleaks.toml", "build_tools/scan_tools/script.py"]
        run_ci = is_ci_run_required(paths)
        self.assertFalse(run_ci)

    def test_run_ci_if_related_workflow_file_edited(self):
        paths = [".github/workflows/multi_arch_ci.yml"]
        run_ci = is_ci_run_required(paths)
        self.assertTrue(run_ci)

        paths = [".github/workflows/multi_arch_build_portable_linux_artifacts.yml"]
        run_ci = is_ci_run_required(paths)
        self.assertTrue(run_ci)

        paths = [".github/workflows/multi_arch_build_native_linux_packages.yml"]
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

    @patch("configure_ci_path_filters.subprocess.run")
    def test_missing_base_sha_is_fetched_before_diffing(self, mock_run):
        base_sha = "f5c168058a7ceaa0f179cc36784b491a11a3adc7"

        def run_side_effect(args, **kwargs):
            if args == ["git", "cat-file", "-e", f"{base_sha}^{{commit}}"]:
                return subprocess.CompletedProcess(args=args, returncode=1)
            if args == ["git", "diff", "--name-only", base_sha]:
                return subprocess.CompletedProcess(
                    args=args,
                    returncode=0,
                    stdout="compiler/amd-llvm\ncompiler/spirv-llvm-translator\n",
                )
            if args == [
                "git",
                "fetch",
                "--no-tags",
                "--no-recurse-submodules",
                "--depth=1",
                "origin",
                base_sha,
            ]:
                return subprocess.CompletedProcess(
                    args=args,
                    returncode=0,
                    stdout="",
                )
            self.fail(f"Unexpected subprocess.run call: {args!r}")

        mock_run.side_effect = run_side_effect

        self.assertEqual(
            get_git_modified_paths(base_sha),
            ["compiler/amd-llvm", "compiler/spirv-llvm-translator"],
        )

    @patch("configure_ci_path_filters.subprocess.run")
    def test_diff_failure_for_available_base_sha_is_not_treated_as_missing(
        self, mock_run
    ):
        base_sha = "f5c168058a7ceaa0f179cc36784b491a11a3adc7"

        def run_side_effect(args, **kwargs):
            if args == ["git", "cat-file", "-e", f"{base_sha}^{{commit}}"]:
                return subprocess.CompletedProcess(args=args, returncode=0)
            if args == ["git", "diff", "--name-only", base_sha]:
                raise subprocess.CalledProcessError(128, args)
            self.fail(f"Unexpected subprocess.run call: {args!r}")

        mock_run.side_effect = run_side_effect

        with self.assertRaises(subprocess.CalledProcessError):
            get_git_modified_paths(base_sha)

    @patch("configure_ci_path_filters.subprocess.run")
    def test_get_git_commit_hash_resolves_ref(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=["git", "rev-parse", "--verify", "HEAD^{commit}"],
            returncode=0,
            stdout="0123456789abcdef0123456789abcdef01234567\n",
        )

        self.assertEqual(
            get_git_commit_hash("HEAD"),
            "0123456789abcdef0123456789abcdef01234567",
        )
        mock_run.assert_called_once_with(
            ["git", "rev-parse", "--verify", "HEAD^{commit}"],
            stdout=subprocess.PIPE,
            check=True,
            text=True,
            timeout=60,
        )

    def test_ci_workflow_filenames_cover_all_transitive_uses(self):
        """_GITHUB_WORKFLOWS_CI_FILENAMES must exactly match the set of
        workflows transitively called by multi_arch_ci.yml.

        This is a change-detector test that can be removed if
        _GITHUB_WORKFLOWS_CI_FILENAMES is computed dynamically instead of
        maintained by hand.

        If this test fails, update _GITHUB_WORKFLOWS_CI_FILENAMES in
        configure_ci_path_filters.py to match the actual workflow tree.
        """
        all_used = get_transitive_workflow_uses(["multi_arch_ci.yml"])
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
