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
    get_ci_workflow_filenames,
    get_jobs_for_changes,
    get_git_commit_hash,
    get_git_modified_paths,
    is_ci_run_required,
)
from workflow_utils import get_transitive_workflow_uses


class TestIsCIRunRequired(unittest.TestCase):
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

    def test_dont_run_ci_if_only_uccl_files_edited(self):
        self.assertFalse(
            is_ci_run_required(["external-builds/uccl/build_prod_wheels.py"])
        )

    def test_dont_run_ci_if_only_skipped_files_edited(self):
        paths = [
            "build_tools/scan_tools/gitleaks.toml",
            "build_tools/scan_tools/bandit.yaml",
            "build_tools/scan_tools/trivy.yaml",
            "build_tools/scan_tools/zizmor.yaml",
        ]
        run_ci = is_ci_run_required(paths)
        self.assertFalse(run_ci)

    def test_dont_run_ci_for_path_filter_only_changes(self):
        paths = ["build_tools/github_actions/configure_ci_path_filters.py"]
        self.assertFalse(is_ci_run_required(paths))

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

    def test_run_ci_if_source_file_and_unrelated_workflow_file_edited(self):
        paths = ["source_file.h", ".github/workflows/pre-commit.yml"]
        run_ci = is_ci_run_required(paths)
        self.assertTrue(run_ci)

    def test_dont_run_ci_for_unit_test_only_changes(self):
        # These directories are exercised separately by unit_tests.yml.
        unit_test_paths = [
            "build_tools/tests/example_test.py",
            "build_tools/github_actions/tests/example_test.py",
            "build_tools/packaging/tests/example_test.py",
            "build_tools/packaging/archives/tests/example_test.py",
            "build_tools/packaging/linux/tests/example_test.py",
            "build_tools/packaging/python/tests/example_test.py",
            "build_tools/packaging/windows/tests/example_test.py",
            "build_tools/third_party/s3_management/tests/example_test.py",
            "external-builds/pytorch/tests/example_test.py",
            "test_tools/tests/example_test.py",
        ]

        for path in unit_test_paths:
            with self.subTest(path=path):
                self.assertFalse(is_ci_run_required([path]))

    def test_run_ci_for_source_and_unit_test_changes(self):
        # Exclusions for skipping unit tests do not take priority over
        # inclusions for modifying script files.
        paths = [
            "build_tools/packaging/archives/build_tarballs.py",
            "build_tools/packaging/archives/tests/build_tarballs_test.py",
        ]
        self.assertTrue(is_ci_run_required(paths))

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

    def test_external_repo_patterns_do_not_apply_therock_workflow_allowlist(self):
        self.assertTrue(
            is_ci_run_required(
                [".github/workflows/pre-commit.yml"],
                skip_patterns=["*.md"],
                repo_name="rocm-libraries",
            )
        )


class TestGetJobsForChanges(unittest.TestCase):
    def test_pytorch_changes_select_pytorch(self):
        paths = [
            "external-builds/pytorch/skip_tests/generic.py",
            ".github/workflows/multi_arch_build_windows_pytorch_wheels_ci.yml",
        ]
        self.assertEqual(get_jobs_for_changes(paths), {"pytorch"})

    def test_jax_changes_select_jax(self):
        paths = [
            "external-builds/jax/install_jax_wheels.py",
            ".github/workflows/test_multi_arch_linux_jax_wheels.yml",
        ]
        self.assertEqual(get_jobs_for_changes(paths), {"jax"})

    def test_pytorch_and_jax_changes_select_both_frameworks(self):
        paths = [
            "external-builds/pytorch/skip_tests/generic.py",
            "external-builds/jax/install_jax_wheels.py",
        ]
        self.assertEqual(get_jobs_for_changes(paths), {"pytorch", "jax"})

    def test_unmapped_changes_affect_rocm_without_losing_other_matches(self):
        paths = [
            "external-builds/pytorch/build_prod_wheels.py",
            "build_tools/github_actions/amdgpu_family_matrix.py",
        ]
        self.assertEqual(get_jobs_for_changes(paths), {"pytorch", "rocm"})

    def test_skippable_changes_do_not_add_affected_jobs(self):
        paths = [
            "external-builds/pytorch/build_prod_wheels.py",
            "docs/development/ci_overview.md",
            "external-builds/uccl/build_prod_wheels.py",
            "external-builds/pytorch/tests/repo_management_test.py",
            ".github/workflows/pre-commit.yml",
        ]
        self.assertEqual(get_jobs_for_changes(paths), {"pytorch"})

    def test_release_only_workflows_do_not_select_frameworks(self):
        self.assertEqual(
            get_jobs_for_changes(
                [".github/workflows/multi_arch_build_windows_pytorch_wheels.yml"]
            ),
            set(),
        )

    def test_no_relevant_changes_affect_no_jobs(self):
        for paths in ([], ["README.md"]):
            with self.subTest(paths=paths):
                self.assertEqual(get_jobs_for_changes(paths), set())

    def test_unavailable_diff_affects_all_job_groups(self):
        self.assertEqual(
            get_jobs_for_changes(None),
            {
                "rocm",
                "rocm_python",
                "rocm_native_linux",
                "rocm_native_windows",
                "pytorch",
                "jax",
            },
        )

    def test_python_packaging_changes_select_rocm_python(self):
        paths = ["build_tools/packaging/python/templates/rocm/setup.py"]
        self.assertEqual(get_jobs_for_changes(paths), {"rocm_python"})

    def test_native_linux_packaging_changes_select_rocm_native_linux(self):
        paths = ["build_tools/packaging/linux/build_package.py"]
        self.assertEqual(get_jobs_for_changes(paths), {"rocm_native_linux"})

    def test_native_windows_packaging_changes_select_rocm_native_windows(self):
        paths = ["build_tools/packaging/windows/generate_msi_wxs.py"]
        self.assertEqual(get_jobs_for_changes(paths), {"rocm_native_windows"})

    def test_packaging_workflow_changes_accumulate_with_framework_changes(self):
        paths = [
            "external-builds/pytorch/build_prod_wheels.py",
            ".github/workflows/build_windows_python_packages.yml",
            ".github/workflows/multi_arch_build_native_linux_packages.yml",
            ".github/workflows/multi_arch_build_native_windows_packages.yml",
        ]
        self.assertEqual(
            get_jobs_for_changes(paths),
            {"pytorch", "rocm_python", "rocm_native_linux"},
        )

    def test_shared_packaging_code_affects_rocm(self):
        self.assertEqual(
            get_jobs_for_changes(["build_tools/packaging/archives/compress.py"]),
            {"rocm"},
        )


class TestGetGitModifiedPaths(unittest.TestCase):
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


class TestGetGitCommitHash(unittest.TestCase):
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


class TestGetCIWorkflowFilenames(unittest.TestCase):
    def test_ci_workflow_filenames_cover_all_transitive_uses(self):
        """CI workflow filenames in skip-ci-config.toml must exactly match
        the set of workflows transitively called by multi_arch_ci.yml.

        This is a change-detector test that can be removed if
        ci_workflows.filenames is computed dynamically instead of
        maintained by hand.

        If this test fails, update ci_workflows.filenames in
        build_tools/github_actions/skip-ci-config.toml to match the actual
        workflow tree.
        """
        ci_workflow_filenames = get_ci_workflow_filenames()
        all_used = get_transitive_workflow_uses(["multi_arch_ci.yml"])
        missing = all_used - ci_workflow_filenames
        stale = ci_workflow_filenames - all_used
        errors = []
        if missing:
            errors.append(
                "Missing (add to ci_workflows.filenames in skip-ci-config.toml):\n"
                + "\n".join(f"  - {f}" for f in sorted(missing))
            )
        if stale:
            errors.append(
                "Stale (remove from ci_workflows.filenames in skip-ci-config.toml):\n"
                + "\n".join(f"  - {f}" for f in sorted(stale))
            )
        if errors:
            self.fail("\n".join(errors))


if __name__ == "__main__":
    unittest.main()
