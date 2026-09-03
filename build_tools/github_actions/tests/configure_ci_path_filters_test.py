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
    is_external_repo_ci_required,
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

    def test_dont_run_ci_for_unit_test_only_changes(self):
        # These directories are exercised separately by unit_tests.yml.
        unit_test_paths = [
            "build_tools/tests/example_test.py",
            "build_tools/github_actions/tests/example_test.py",
            "build_tools/packaging/linux/tests/example_test.py",
            "build_tools/packaging/python/tests/example_test.py",
            "build_tools/third_party/s3_management/tests/example_test.py",
            "build_tools/scan_tools/github_actions/tests/example_test.py",
            "test_tools/tests/example_test.py",
        ]

        for path in unit_test_paths:
            with self.subTest(path=path):
                self.assertFalse(is_ci_run_required([path]))

    def test_dont_run_ci_for_path_filter_only_changes(self):
        paths = ["build_tools/github_actions/configure_ci_path_filters.py"]
        self.assertFalse(is_ci_run_required(paths))

    def test_run_ci_for_tests_exercising_built_packages(self):
        integration_test_paths = [
            # Tests for ROCm subprojects.
            "build_tools/github_actions/test_executable_scripts/test_example.py",
            # Install tests for packages (run on CI after building packages)
            "build_tools/packaging/python/templates/example_test.py",
            "build_tools/packaging/linux/example_test.py",
            # Integration tests for ROCm (run on CI after building artifacts)
            "tests/test_rocm_sanity.py",
        ]

        for path in integration_test_paths:
            with self.subTest(path=path):
                self.assertTrue(is_ci_run_required([path]))

    def test_run_ci_for_source_and_unit_test_changes(self):
        # Exclusions for skipping unit tests do not take priority over
        # inclusions for modifying script files.
        paths = [
            "build_tools/build_tarballs.py",
            "build_tools/tests/build_tarballs_test.py",
        ]
        self.assertTrue(is_ci_run_required(paths))

    @patch("configure_ci_path_filters.subprocess.run")
    def test_missing_base_sha_is_fetched_before_diffing(self, mock_run):
        base_sha = "f5c168058a7ceaa0f179cc36784b491a11a3adc7"
        fetched = False

        def run_side_effect(args, **kwargs):
            nonlocal fetched
            if args[:2] == ["git", "cat-file"]:
                return subprocess.CompletedProcess(args=args, returncode=1)
            if args[:2] == ["git", "diff"]:
                if not fetched:
                    raise subprocess.CalledProcessError(128, args)
                return subprocess.CompletedProcess(
                    args=args,
                    returncode=0,
                    stdout="compiler/amd-llvm\ncompiler/spirv-llvm-translator\n",
                )
            if args[:2] == ["git", "fetch"]:
                fetched = True
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
            if args[:2] == ["git", "cat-file"]:
                return subprocess.CompletedProcess(args=args, returncode=0)
            if args[:2] == ["git", "diff"]:
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

    @patch("configure_ci_path_filters.subprocess.run")
    def test_get_git_commit_hash_fetches_missing_sha_before_resolving(self, mock_run):
        base_sha = "f5c168058a7ceaa0f179cc36784b491a11a3adc7"
        fetched = False

        def run_side_effect(args, **kwargs):
            nonlocal fetched
            if args[:2] == ["git", "cat-file"]:
                return subprocess.CompletedProcess(args=args, returncode=1)
            if args[:2] == ["git", "fetch"]:
                fetched = True
                return subprocess.CompletedProcess(
                    args=args,
                    returncode=0,
                    stdout="",
                )
            if args[:2] == ["git", "rev-parse"]:
                if not fetched:
                    raise subprocess.CalledProcessError(128, args)
                return subprocess.CompletedProcess(
                    args=args,
                    returncode=0,
                    stdout=f"{base_sha}\n",
                )
            self.fail(f"Unexpected subprocess.run call: {args!r}")

        mock_run.side_effect = run_side_effect

        self.assertEqual(get_git_commit_hash(base_sha), base_sha)

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


class ExternalRepoPathFiltersTest(unittest.TestCase):
    """Tests for is_external_repo_ci_required function."""

    def test_none_changed_files_requires_ci(self):
        """None changed_files means unknown changes, run CI to be safe."""
        self.assertTrue(is_external_repo_ci_required(None))

    def test_empty_changed_files_skips_ci(self):
        """Empty changed_files means no changes, skip CI."""
        self.assertFalse(is_external_repo_ci_required([]))

    def test_only_markdown_files_skips_ci(self):
        """Only markdown files changed, skip CI."""
        self.assertFalse(is_external_repo_ci_required(["README.md"]))
        self.assertFalse(
            is_external_repo_ci_required(["README.md", "docs/guide.md", "CHANGELOG.md"])
        )

    def test_only_rst_files_skips_ci(self):
        """Only RST files changed, skip CI."""
        self.assertFalse(is_external_repo_ci_required(["docs/index.rst"]))

    def test_only_docs_directory_skips_ci(self):
        """Only docs directory changes, skip CI."""
        self.assertFalse(
            is_external_repo_ci_required(["docs/guide.md", "docs/api/index.rst"])
        )

    def test_only_gitignore_skips_ci(self):
        """Only .gitignore files changed, skip CI."""
        self.assertFalse(is_external_repo_ci_required([".gitignore"]))
        self.assertFalse(is_external_repo_ci_required(["projects/rocblas/.gitignore"]))

    def test_only_project_docs_skips_ci(self):
        """Only project-specific docs changed, skip CI."""
        self.assertFalse(
            is_external_repo_ci_required(
                ["projects/rocblas/docs/README.md", "shared/utils/docs/guide.rst"]
            )
        )

    def test_only_experimental_skips_ci(self):
        """Only experimental files changed, skip CI."""
        self.assertFalse(is_external_repo_ci_required(["experimental/new_feature.cpp"]))

    def test_only_ai_rules_skips_ci(self):
        """Only AI/editor rules files changed, skip CI."""
        self.assertFalse(
            is_external_repo_ci_required([".clinerules", ".cursorrules", "CLAUDE.mdc"])
        )

    def test_source_file_requires_ci(self):
        """Source file change requires CI."""
        self.assertTrue(is_external_repo_ci_required(["projects/rocblas/src/lib.cpp"]))

    def test_cmake_file_requires_ci(self):
        """CMake file change requires CI."""
        self.assertTrue(is_external_repo_ci_required(["CMakeLists.txt"]))

    def test_python_script_requires_ci(self):
        """Python script change requires CI."""
        self.assertTrue(is_external_repo_ci_required(["scripts/build.py"]))

    def test_mixed_skippable_and_non_skippable_requires_ci(self):
        """Mix of skippable and non-skippable requires CI."""
        self.assertTrue(
            is_external_repo_ci_required(["README.md", "projects/rocblas/src/lib.cpp"])
        )

    def test_github_workflow_requires_ci(self):
        """GitHub workflow file requires CI (not in skippable patterns)."""
        self.assertTrue(is_external_repo_ci_required([".github/workflows/ci.yml"]))

    def test_codeowners_skips_ci(self):
        """CODEOWNERS file is skippable."""
        self.assertFalse(is_external_repo_ci_required(["CODEOWNERS"]))
        self.assertFalse(is_external_repo_ci_required([".github/CODEOWNERS"]))

    def test_license_skips_ci(self):
        """LICENSE file is skippable."""
        self.assertFalse(is_external_repo_ci_required(["LICENSE"]))
        self.assertFalse(is_external_repo_ci_required(["LICENSE.md"]))

    def test_repo_name_included_in_logging(self):
        """Repo name is used in logging output."""
        # This is a smoke test - just verify it doesn't crash
        self.assertTrue(
            is_external_repo_ci_required(["src/lib.cpp"], repo_name="rocm-libraries")
        )
        self.assertFalse(
            is_external_repo_ci_required(["README.md"], repo_name="rocm-systems")
        )


if __name__ == "__main__":
    unittest.main()
