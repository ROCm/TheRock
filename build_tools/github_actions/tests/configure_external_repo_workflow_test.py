"""Regression tests for external-repository workflow changes."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from build_tools.github_actions import configure_external_repo_ci as ci

WORKFLOW = ".github/workflows/therock-multi-arch-ci-nightly.yml"

LIBRARY_ENTRIES = [
    ("projects", "rocblas"),
    ("projects", "hipblas"),
    ("shared", "stinkytofu"),
]
SYSTEM_ENTRIES = [
    ("projects", "rccl"),
    ("projects", "hip"),
]
LIBRARY_PROJECTS = {
    "projects/rocblas",
    "projects/hipblas",
    "shared/stinkytofu",
}


class ExternalRepoWorkflowTest(unittest.TestCase):
    def configure(
        self,
        modified_paths,
        *,
        repo="ROCm/rocm-libraries",
        entries=LIBRARY_ENTRIES,
        config_exists=True,
    ):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "repos-config.json"

            if config_exists:
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
                ci, "get_modified_paths_api", return_value=set(modified_paths)
            ) as get_paths:
                result = ci.configure(
                    event_name="pull_request",
                    github_repo=repo,
                    base_sha="base",
                    head_sha="head",
                    config_path=str(config_path),
                )

            get_paths.assert_called_once_with(repo, "base", "head")
            return result

    def test_libraries_workflow_only_is_skipped(self):
        """TheRock workflow files are in SKIPPABLE_PATH_PATTERNS, so workflow-only
        changes should skip tests (no build/test impact from workflow file changes)."""
        result = self.configure({WORKFLOW})

        self.assertFalse(result.run_all_tests)
        self.assertTrue(result.skip_tests)
        self.assertEqual(result.changed_projects, "")

    def test_libraries_workflow_with_known_source_selects_only_changed_source(self):
        """When a skippable workflow file AND a source file change together,
        only the source change is detected (workflow file is ignored as skippable)."""
        result = self.configure({WORKFLOW, "projects/rocblas/src/rocblas.cpp"})

        self.assertFalse(result.run_all_tests)
        self.assertFalse(result.skip_tests)
        # Only rocblas should be detected, not all library projects
        self.assertEqual(result.changed_projects, "projects/rocblas")

    def test_systems_workflow_only_is_skipped(self):
        """TheRock workflow files are in SKIPPABLE_PATH_PATTERNS even for rocm-systems."""
        result = self.configure(
            {WORKFLOW},
            repo="ROCm/rocm-systems",
            entries=SYSTEM_ENTRIES,
        )

        self.assertFalse(result.run_all_tests)
        self.assertTrue(result.skip_tests)
        self.assertEqual(result.changed_projects, "")

    def test_full_test_trigger_patterns_preserve_full_test_run(self):
        """Paths in FULL_TEST_TRIGGER_PATTERNS should trigger run_all_tests."""
        for path in (
            ".github/repos-config.json",
            ".github/scripts/get_changed_projects.py",
        ):
            with self.subTest(path=path):
                result = self.configure({WORKFLOW, path})

                self.assertTrue(result.run_all_tests)
                self.assertFalse(result.skip_tests)
                self.assertEqual(result.changed_projects, "")

    def test_ctest_change_returns_shared_ctest(self):
        """shared/ctest is in CI_RELEVANT_NON_SUBTREE_PREFIXES, so it should be
        detected as a changed project, not trigger run_all_tests."""
        result = self.configure({WORKFLOW, "shared/ctest/categories.yaml"})

        self.assertFalse(result.run_all_tests)
        self.assertFalse(result.skip_tests)
        self.assertEqual(result.changed_projects, "shared/ctest")

    def test_workflow_with_unclassified_path_preserves_full_test_run(self):
        result = self.configure({WORKFLOW, "unexpected/config.py"})

        self.assertTrue(result.run_all_tests)
        self.assertEqual(result.changed_projects, "")

    def test_workflow_without_config_is_skipped(self):
        """TheRock workflow files are in SKIPPABLE_PATH_PATTERNS, so skipped regardless of config."""
        result = self.configure({WORKFLOW}, config_exists=False)

        self.assertFalse(result.run_all_tests)
        self.assertTrue(result.skip_tests)
        self.assertEqual(result.changed_projects, "")

    def test_configured_repository_workflow_only_is_skipped(self):
        """TheRock workflow files are in SKIPPABLE_PATH_PATTERNS, so skipped for any repo."""
        with patch.dict(
            ci.WORKFLOW_TEST_SCOPE_BY_REPO,
            {"rocm/example": "configured-projects"},
        ):
            result = self.configure(
                {WORKFLOW},
                repo="ROCm/example",
                entries=[("projects", "example")],
            )

        self.assertFalse(result.run_all_tests)
        self.assertTrue(result.skip_tests)
        self.assertEqual(result.changed_projects, "")

    def test_unknown_repository_workflow_only_is_skipped(self):
        """TheRock workflow files are in SKIPPABLE_PATH_PATTERNS, so skipped for unknown repos."""
        result = self.configure({WORKFLOW}, repo="ROCm/unknown")

        self.assertFalse(result.run_all_tests)
        self.assertTrue(result.skip_tests)
        self.assertEqual(result.changed_projects, "")

    def test_ordinary_source_changes_stay_selective(self):
        result = self.configure({"projects/rocblas/src/rocblas.cpp"})

        self.assertFalse(result.run_all_tests)
        self.assertFalse(result.skip_tests)
        self.assertEqual(result.changed_projects, "projects/rocblas")

    def test_source_change_does_not_select_rccl_downstream(self):
        """Test that a library source change doesn't pull in rccl as a downstream dependency."""
        result = self.configure({"projects/rocblas/src/rocblas.cpp"})
        therock_root = Path(ci.__file__).resolve().parents[2]
        selector = therock_root / "test_tools/determine_rocm_test_dependencies.py"

        selection = subprocess.run(
            [
                sys.executable,
                str(selector),
                "--changed-projects",
                result.changed_projects,
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


if __name__ == "__main__":
    unittest.main()
