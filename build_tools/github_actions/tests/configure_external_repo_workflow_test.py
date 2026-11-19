"""Regression tests for external-repository workflow changes.

configure() returns only the project scope: a comma-separated list of project
paths, or "" when the diff cannot be narrowed safely. Downstream, "" means
every project is tested. Test depth is decided separately, by
configure_multi_arch_ci.py, so nothing here asserts on it.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import configure_external_repo_ci as ci

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


def select_tests(changed_projects):
    """Run TheRock's test selector over a project scope, as CI does.

    Returns the selected test names; {"*"} means every test runs.
    """
    therock_root = Path(ci.__file__).resolve().parents[2]
    selection = subprocess.run(
        [
            sys.executable,
            str(therock_root / "test_tools/determine_rocm_test_dependencies.py"),
            "--changed-projects",
            changed_projects,
            "--format",
            "list",
        ],
        cwd=therock_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return set(selection.stdout.split())


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

    def assertProjectScope(self, result, expected):
        self.assertEqual(set(result.split(",")), set(expected))

    def test_libraries_workflow_selects_library_projects(self):
        result = self.configure({WORKFLOW})

        self.assertProjectScope(result, LIBRARY_PROJECTS)
        self.assertNotIn("projects/rccl", result.split(","))

    def test_libraries_workflow_with_known_source_selects_all_libraries(self):
        result = self.configure({WORKFLOW, "projects/rocblas/src/rocblas.cpp"})

        self.assertProjectScope(result, LIBRARY_PROJECTS)

    def test_changes_that_cannot_be_attributed_are_not_scoped(self):
        """Each case yields "" -- which select_tests() pins to a full run."""
        cases = [
            # (description, modified paths, configure() kwargs)
            (
                "repo with no configured-projects policy",
                {WORKFLOW},
                dict(repo="ROCm/rocm-systems", entries=SYSTEM_ENTRIES),
            ),
            (
                "repo absent from WORKFLOW_TEST_SCOPE_BY_REPO",
                {WORKFLOW},
                dict(repo="ROCm/unknown"),
            ),
            (
                "no repos-config.json, so no prefixes to match",
                {WORKFLOW},
                dict(config_exists=False),
            ),
            (
                "path attributable to no known subtree",
                {WORKFLOW, "unexpected/config.py"},
                {},
            ),
            ("project discovery config", {WORKFLOW, ".github/repos-config.json"}, {}),
            (
                "project discovery script",
                {WORKFLOW, ".github/scripts/get_changed_projects.py"},
                {},
            ),
            ("shared CTest logic", {WORKFLOW, "shared/ctest/categories.yaml"}, {}),
        ]

        for description, paths, kwargs in cases:
            with self.subTest(case=description):
                self.assertEqual(self.configure(paths, **kwargs), "")

    def test_configured_repository_workflow_selects_its_projects(self):
        with patch.dict(
            ci.WORKFLOW_TEST_SCOPE_BY_REPO,
            {"rocm/example": "configured-projects"},
        ):
            result = self.configure(
                {WORKFLOW},
                repo="ROCm/example",
                entries=[("projects", "example")],
            )

        self.assertEqual(result, "projects/example")

    def test_ordinary_source_changes_stay_selective(self):
        result = self.configure({"projects/rocblas/src/rocblas.cpp"})

        self.assertEqual(result, "projects/rocblas")

    def test_empty_scope_selects_every_test(self):
        """Pins the contract the tests above rely on: "" means a full run.

        Without this, asserting configure() == "" would only pin an
        intermediate string and never show that a full test run results.
        """
        self.assertEqual(select_tests(""), {"*"})

    def test_libraries_workflow_does_not_select_rccl_downstream(self):
        selected_tests = select_tests(self.configure({WORKFLOW}))

        self.assertNotEqual(selected_tests, {"*"})
        self.assertIn("rocblas", selected_tests)
        self.assertNotIn("rccl", selected_tests)


if __name__ == "__main__":
    unittest.main()
