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

import configure_ci_path_filters as pf
from configure_ci_path_filters import (
    get_ci_workflow_filenames,
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


# ---------------------------------------------------------------------------
# External repo changed-project analysis
# ---------------------------------------------------------------------------

# A TheRock workflow file inside the calling repository.
_CALLER_WORKFLOW = ".github/workflows/therock-multi-arch-ci-nightly.yml"

_LIBRARY_ENTRIES = [
    ("projects", "rocblas"),
    ("projects", "hipblas"),
    ("projects", "hipblaslt"),
    ("projects", "hipblaslt/tensilelite"),
    ("projects", "rpp"),
    ("shared", "stinkytofu"),
]
_LIBRARY_PROJECTS = [
    "projects/hipblas",
    "projects/hipblaslt",
    "projects/hipblaslt/tensilelite",
    "projects/rocblas",
    "projects/rpp",
    "shared/stinkytofu",
]
_SYSTEM_ENTRIES = [("projects", "rccl"), ("projects", "hip")]

# Mirrors skip-ci-base.toml; the real patterns are exercised in
# configure_ci_path_filters_test.py.
_SKIP_PATTERNS = ["*.md", "*CODEOWNERS", "docs/*"]


class TestComputeExternalRepoChangedProjects(unittest.TestCase):
    """Test mapping an external repo's diff onto the projects CI should scope to.

    An empty result means "cannot narrow scope" and makes downstream consumers
    fall back to full coverage.
    """

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
                config_dir = Path(directory) / ".github"
                config_dir.mkdir()
                (config_dir / "repos-config.json").write_text(
                    json.dumps(
                        {
                            "repositories": [
                                {"category": category, "name": name}
                                for category, name in entries
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
            with patch.object(pf, "_EXTERNAL_REPO_CONFIG_DIR", directory):
                return pf.compute_external_repo_changed_projects(
                    changed_files=list(changed_files),
                    projects_config_path=".github/repos-config.json",
                    repo_full_name=repo,
                    skip_patterns=skip_patterns,
                )

    # -- Ordinary source changes --------------------------------------------

    def test_source_change_narrows_to_its_project(self):
        self.assertEqual(
            self.compute(["projects/rocblas/src/rocblas.cpp"]), ["projects/rocblas"]
        )

    def test_nested_subtree_wins_over_parent(self):
        """A tensilelite-only change must not also fire hipblaslt-proper."""
        self.assertEqual(
            self.compute(["projects/hipblaslt/tensilelite/foo.py"]),
            ["projects/hipblaslt/tensilelite"],
        )

    def test_parent_change_does_not_match_nested_subtree(self):
        self.assertEqual(
            self.compute(["projects/hipblaslt/src/foo.cpp"]), ["projects/hipblaslt"]
        )

    def test_skippable_paths_do_not_widen_scope(self):
        self.assertEqual(
            self.compute(["projects/rocblas/src/rocblas.cpp", "README.md"]),
            ["projects/rocblas"],
        )

    def test_non_subtree_ci_relevant_prefix_is_surfaced(self):
        """Monorepo dirs absent from repos-config.json still classify."""
        self.assertEqual(self.compute(["shared/kpack/kpack.cpp"]), ["shared/kpack"])

    def test_no_changed_files_runs_all_tests(self):
        self.assertEqual(self.compute([]), [])

    # -- Conservative guards -------------------------------------------------

    def test_unclassified_path_runs_all_tests(self):
        self.assertEqual(self.compute(["unexpected/config.py"]), [])

    def test_mixed_recognized_and_unclassified_runs_all_tests(self):
        """Narrowing to the recognized subset would drop the unknown change."""
        self.assertEqual(
            self.compute(["projects/rocblas/src/rocblas.cpp", "unexpected/config.py"]),
            [],
        )

    def test_shared_ctest_change_runs_all_tests(self):
        self.assertEqual(self.compute(["shared/ctest/categories.yaml"]), [])

    def test_repos_config_change_runs_all_tests(self):
        self.assertEqual(self.compute([".github/repos-config.json"]), [])

    def test_missing_config_runs_all_tests(self):
        self.assertEqual(
            self.compute(["projects/rocblas/src/a.cpp"], config_exists=False), []
        )

    # -- Workflow test scoping ----------------------------------------------

    def test_libraries_workflow_change_scopes_to_configured_projects(self):
        self.assertEqual(self.compute([_CALLER_WORKFLOW]), _LIBRARY_PROJECTS)

    def test_libraries_workflow_with_known_source_scopes_to_configured_projects(self):
        self.assertEqual(
            self.compute([_CALLER_WORKFLOW, "projects/rocblas/src/rocblas.cpp"]),
            _LIBRARY_PROJECTS,
        )

    def test_libraries_workflow_with_unclassified_path_runs_all_tests(self):
        self.assertEqual(self.compute([_CALLER_WORKFLOW, "unexpected/config.py"]), [])

    def test_libraries_workflow_with_shared_ci_change_runs_all_tests(self):
        self.assertEqual(
            self.compute([_CALLER_WORKFLOW, "shared/ctest/categories.yaml"]), []
        )

    def test_unscoped_repo_workflow_change_runs_all_tests(self):
        """rocm-systems is not in the scope map, so it keeps full coverage."""
        self.assertEqual(
            self.compute(
                [_CALLER_WORKFLOW], repo="ROCm/rocm-systems", entries=_SYSTEM_ENTRIES
            ),
            [],
        )

    def test_unknown_repo_workflow_change_runs_all_tests(self):
        self.assertEqual(self.compute([_CALLER_WORKFLOW], repo="ROCm/unknown"), [])

    def test_scope_map_lookup_is_case_insensitive(self):
        self.assertEqual(
            self.compute([_CALLER_WORKFLOW], repo="rocm/ROCm-Libraries"),
            _LIBRARY_PROJECTS,
        )

    def test_skippable_workflow_path_does_not_widen_scope(self):
        """A repo that skips its own workflows scopes to the real change.

        Pairing the workflow edit with a source change must classify on the
        source change, not expand to every configured project.
        """
        self.assertEqual(
            self.compute(
                [_CALLER_WORKFLOW, "projects/rocblas/src/rocblas.cpp"],
                skip_patterns=_SKIP_PATTERNS + [".github/workflows/*"],
            ),
            ["projects/rocblas"],
        )

    def test_all_skippable_paths_still_run_all_tests(self):
        """An entirely skippable diff must not narrow to an empty-ish scope."""
        self.assertEqual(
            self.compute(
                [_CALLER_WORKFLOW],
                skip_patterns=_SKIP_PATTERNS + [".github/workflows/*"],
            ),
            _LIBRARY_PROJECTS,
        )

    def test_skippable_workflow_with_unclassified_path_runs_all_tests(self):
        """Dropping the workflow must not disarm the unclassified-path guard."""
        self.assertEqual(
            self.compute(
                [_CALLER_WORKFLOW, "unexpected/config.py"],
                skip_patterns=_SKIP_PATTERNS + [".github/workflows/*"],
            ),
            [],
        )

    def test_configured_scope_map_entry_applies(self):
        with patch.dict(
            pf._WORKFLOW_TEST_SCOPE_BY_REPO, {"rocm/example": "configured-projects"}
        ):
            result = self.compute(
                [_CALLER_WORKFLOW],
                repo="ROCm/example",
                entries=[("projects", "example")],
            )
        self.assertEqual(result, ["projects/example"])


class TestNonSubtreePrefixContract(unittest.TestCase):
    """The non-subtree prefix list must stay usable by the test selector."""

    def test_declared_prefixes_are_wellformed(self):
        for prefix in pf._CI_RELEVANT_NON_SUBTREE_PREFIXES:
            with self.subTest(prefix=prefix):
                self.assertEqual(prefix.count("/"), 1, "expect 'category/name'")
                self.assertFalse(prefix.startswith("/"))
                self.assertFalse(prefix.endswith("/"))

    def test_declared_prefixes_resolve_to_tests(self):
        """Surfacing a prefix the test selector cannot map just fails the job.

        These directories are absent from every repos-config.json, so TheRock
        is the only place the mapping exists. Keep the two lists in lock-step.
        """
        therock_root = Path(pf.__file__).resolve().parents[2]
        sys.path.insert(0, os.fspath(therock_root / "test_tools"))
        from determine_rocm_test_dependencies import _EXTERNAL_SUBTREE_ALIASES

        for prefix in pf._CI_RELEVANT_NON_SUBTREE_PREFIXES:
            with self.subTest(prefix=prefix):
                self.assertIn(prefix, _EXTERNAL_SUBTREE_ALIASES)


if __name__ == "__main__":
    unittest.main()
