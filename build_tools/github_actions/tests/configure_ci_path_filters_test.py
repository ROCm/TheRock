# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import configure_ci_path_filters
from configure_ci_path_filters import (
    compute_external_repo_changed_projects,
    get_ci_workflow_filenames,
    get_git_commit_hash,
    get_git_modified_paths,
    is_ci_run_required,
    is_external_ci_only_change,
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
        paths = [
            "build_tools/scan_tools/gitleaks.toml",
            "build_tools/scan_tools/bandit.yaml",
            "build_tools/scan_tools/trivy.yaml",
            "build_tools/scan_tools/zizmor.yaml",
        ]
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
            "build_tools/packaging/archives/build_tarballs.py",
            "build_tools/packaging/archives/tests/build_tarballs_test.py",
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


# A TheRock workflow file inside the calling repository.
_CALLER_WORKFLOW = ".github/workflows/therock-multi-arch-ci-nightly.yml"
# The calling repository's ci-env action, holding its pinned TheRock ref and
# baseline run ID. Edited on the same occasions as the workflows above.
_CI_ENV_ACTION = ".github/actions/ci-env/action.yml"
_PROJECTS_CONFIG = ".github/repos-config.json"
_SKIP_PATTERNS = ["*.md", "docs/*", "*CODEOWNERS"]

_LIBRARY_ENTRIES = [
    ("projects", "rocblas"),
    ("projects", "hipblas"),
    ("shared", "stinkytofu"),
]
_LIBRARY_PROJECTS = ["projects/hipblas", "projects/rocblas", "shared/stinkytofu"]
_SYSTEM_ENTRIES = [("projects", "rccl"), ("projects", "hip")]


class ExternalRepoChangedProjectsTest(unittest.TestCase):
    """compute_external_repo_changed_projects: [] means 'no narrowing'."""

    def compute(
        self,
        changed_files,
        *,
        repo="ROCm/rocm-libraries",
        entries=_LIBRARY_ENTRIES,
        config_exists=True,
        skip_patterns=_SKIP_PATTERNS,
    ):
        with tempfile.TemporaryDirectory() as directory:
            if config_exists:
                config_path = Path(directory) / _PROJECTS_CONFIG
                config_path.parent.mkdir(parents=True, exist_ok=True)
                config_path.write_text(
                    json.dumps(
                        {
                            "repositories": [
                                {
                                    "category": category,
                                    "name": name,
                                    "url": f"ROCm/{name}",
                                    "branch": "develop",
                                }
                                for category, name in entries
                            ]
                        }
                    ),
                    encoding="utf-8",
                )

            with patch.object(
                configure_ci_path_filters, "_EXTERNAL_REPO_CONFIG_DIR", directory
            ):
                return compute_external_repo_changed_projects(
                    changed_files=list(changed_files),
                    projects_config_path=_PROJECTS_CONFIG,
                    repo_full_name=repo,
                    skip_patterns=skip_patterns,
                )

    def test_source_change_selects_only_that_project(self):
        self.assertEqual(
            self.compute(["projects/rocblas/src/rocblas.cpp"]), ["projects/rocblas"]
        )

    def test_nested_project_wins_over_parent(self):
        self.assertEqual(
            self.compute(
                ["projects/hipblaslt/tensilelite/foo.py"],
                entries=[
                    ("projects", "hipblaslt"),
                    ("projects", "hipblaslt/tensilelite"),
                ],
            ),
            ["projects/hipblaslt/tensilelite"],
        )

    def test_non_subtree_directory_is_classified(self):
        self.assertEqual(self.compute(["shared/kpack/kpack.cpp"]), ["shared/kpack"])

    def test_libraries_workflow_selects_its_configured_projects(self):
        """A rocm-libraries workflow edit must not schedule rocm-systems tests."""
        projects = self.compute([_CALLER_WORKFLOW])
        self.assertEqual(projects, _LIBRARY_PROJECTS)
        self.assertNotIn("projects/rccl", projects)

    def test_libraries_workflow_with_known_source_selects_all_libraries(self):
        self.assertEqual(
            self.compute([_CALLER_WORKFLOW, "projects/rocblas/src/rocblas.cpp"]),
            _LIBRARY_PROJECTS,
        )

    def test_systems_workflow_keeps_full_coverage(self):
        self.assertEqual(
            self.compute(
                [_CALLER_WORKFLOW], repo="ROCm/rocm-systems", entries=_SYSTEM_ENTRIES
            ),
            [],
        )

    def test_unknown_repo_workflow_keeps_full_coverage(self):
        self.assertEqual(self.compute([_CALLER_WORKFLOW], repo="ROCm/unknown"), [])

    def test_workflow_with_unclassifiable_path_keeps_full_coverage(self):
        self.assertEqual(self.compute([_CALLER_WORKFLOW, "unexpected/config.py"]), [])

    def test_shared_ci_changes_keep_full_coverage(self):
        for path in (
            ".github/repos-config.json",
            ".github/scripts/get_changed_projects.py",
            "shared/ctest/categories.yaml",
        ):
            with self.subTest(path=path):
                self.assertEqual(self.compute([_CALLER_WORKFLOW, path]), [])

    def test_unclassifiable_path_keeps_full_coverage(self):
        self.assertEqual(
            self.compute(["projects/rocblas/src/rocblas.cpp", "unexpected/config.py"]),
            [],
        )

    def test_missing_projects_config_keeps_full_coverage(self):
        self.assertEqual(
            self.compute(["projects/rocblas/src/rocblas.cpp"], config_exists=False), []
        )

    def test_no_changed_files_keeps_full_coverage(self):
        self.assertEqual(self.compute([]), [])
        self.assertEqual(
            compute_external_repo_changed_projects(
                changed_files=None,
                projects_config_path=_PROJECTS_CONFIG,
                repo_full_name="ROCm/rocm-libraries",
            ),
            [],
        )

    def test_skippable_path_does_not_widen_scope(self):
        """A docs file alongside a project change must not force full coverage."""
        self.assertEqual(
            self.compute(["docs/guide.md", "projects/rocblas/src/rocblas.cpp"]),
            ["projects/rocblas"],
        )

    def test_only_skippable_paths_keep_full_coverage(self):
        self.assertEqual(self.compute(["docs/guide.md", "README.md"]), [])

    def test_libraries_workflow_does_not_select_rccl_downstream(self):
        """End-to-end: the selected projects must not pull in rocm-systems tests."""
        therock_root = Path(configure_ci_path_filters.__file__).resolve().parents[2]
        selector = therock_root / "test_tools/determine_rocm_test_dependencies.py"

        selection = subprocess.run(
            [
                sys.executable,
                str(selector),
                "--changed-projects",
                ",".join(self.compute([_CALLER_WORKFLOW])),
                "--format",
                "list",
            ],
            cwd=therock_root,
            capture_output=True,
            text=True,
            check=True,
        )

        selected_tests = set(selection.stdout.splitlines())
        self.assertIn("rocblas", selected_tests)
        self.assertNotIn("rccl", selected_tests)

    def test_ci_env_action_selects_its_configured_projects(self):
        """ci-env is CI plumbing, not an unclassifiable path (#13645)."""
        projects = self.compute([_CI_ENV_ACTION])
        self.assertEqual(projects, _LIBRARY_PROJECTS)
        self.assertNotIn("projects/rccl", projects)

    def test_ci_env_action_with_workflow_selects_its_configured_projects(self):
        """A TheRock ref bump touches both files and must stay narrowed."""
        self.assertEqual(
            self.compute([_CALLER_WORKFLOW, _CI_ENV_ACTION]), _LIBRARY_PROJECTS
        )


class ExternalCiOnlyChangeTest(unittest.TestCase):
    """is_external_ci_only_change: True only for pure CI-plumbing diffs."""

    def check(self, changed_files, skip_patterns=_SKIP_PATTERNS):
        return is_external_ci_only_change(changed_files, skip_patterns)

    def test_workflow_only_is_ci_only(self):
        self.assertTrue(self.check([_CALLER_WORKFLOW]))

    def test_ci_env_action_only_is_ci_only(self):
        self.assertTrue(self.check([_CI_ENV_ACTION]))

    def test_workflow_plus_ci_env_is_ci_only(self):
        self.assertTrue(self.check([_CALLER_WORKFLOW, _CI_ENV_ACTION]))

    def test_skippable_path_does_not_disqualify(self):
        self.assertTrue(self.check(["docs/guide.md", _CALLER_WORKFLOW]))

    def test_source_change_is_not_ci_only(self):
        self.assertFalse(self.check(["projects/rocblas/src/rocblas.cpp"]))

    def test_source_change_alongside_workflow_is_not_ci_only(self):
        self.assertFalse(
            self.check([_CALLER_WORKFLOW, "projects/rocblas/src/rocblas.cpp"])
        )

    def test_unrelated_workflow_is_not_ci_only(self):
        """Only the therock* workflows are TheRock CI plumbing."""
        self.assertFalse(self.check([".github/workflows/close-stale-prs.yml"]))

    def test_shared_ci_script_is_not_ci_only(self):
        """Selection logic can change coverage everywhere; stay conservative."""
        self.assertFalse(self.check([".github/scripts/get_changed_projects.py"]))

    def test_empty_and_unknown_diffs_are_not_ci_only(self):
        """'Nothing to judge' must fall through to the caller's default."""
        self.assertFalse(self.check([]))
        self.assertFalse(self.check(None))
        self.assertFalse(self.check(["docs/guide.md", "README.md"]))


if __name__ == "__main__":
    unittest.main()
