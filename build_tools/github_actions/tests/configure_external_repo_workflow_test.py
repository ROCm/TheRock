"""Regression tests for external-repository workflow changes.

determine_changed_projects() returns only the project scope: a list of project
paths, or [] when the diff cannot be narrowed safely. Downstream, [] means every
project is tested. Test depth is decided separately, by
is_external_repo_ci_only_change(), so nothing here asserts on it.
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import configure_ci_path_filters as filters
from configure_ci_path_filters import determine_changed_projects

WORKFLOW = ".github/workflows/therock-multi-arch-ci-nightly.yml"
CI_ENV = ".github/actions/ci-env/action.yml"

LIBRARY_PREFIXES = {
    "projects/rocblas",
    "projects/hipblas",
    "shared/stinkytofu",
}
SYSTEM_PREFIXES = {
    "projects/rccl",
    "projects/hip",
}


def select_tests(changed_projects):
    """Run TheRock's test selector over a project scope, as CI does.

    Returns the selected test names; {"*"} means every test runs.
    """
    therock_root = Path(filters.__file__).resolve().parents[2]
    selection = subprocess.run(
        [
            sys.executable,
            str(therock_root / "test_tools/determine_rocm_test_dependencies.py"),
            "--changed-projects",
            ",".join(changed_projects),
            "--format",
            "list",
        ],
        cwd=therock_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return set(selection.stdout.split())


class ExternalRepoWorkflowScopeTest(unittest.TestCase):
    def scope(self, changed_files, *, prefixes=None):
        return determine_changed_projects(
            changed_files,
            LIBRARY_PREFIXES if prefixes is None else prefixes,
        )

    def test_caller_workflow_selects_the_callers_own_projects(self):
        """A workflow edit cannot reach a project outside its own repo."""
        self.assertEqual(set(self.scope({WORKFLOW})), LIBRARY_PREFIXES)

    def test_ci_env_action_selects_the_callers_own_projects(self):
        self.assertEqual(set(self.scope({CI_ENV})), LIBRARY_PREFIXES)

    def test_each_caller_repo_is_scoped_to_its_own_subtrees(self):
        """Scoping is not special-cased per repository."""
        self.assertEqual(
            set(self.scope({WORKFLOW}, prefixes=SYSTEM_PREFIXES)),
            SYSTEM_PREFIXES,
        )

    def test_workflow_alongside_source_narrows_to_that_source(self):
        """The workflow adds no project the component change does not cover."""
        self.assertEqual(
            self.scope({WORKFLOW, "projects/rocblas/src/rocblas.cpp"}),
            ["projects/rocblas"],
        )

    def test_ordinary_source_changes_stay_selective(self):
        self.assertEqual(
            self.scope({"projects/rocblas/src/rocblas.cpp"}), ["projects/rocblas"]
        )

    def test_nested_subtree_wins_over_its_parent(self):
        prefixes = {"projects/hipblaslt", "projects/hipblaslt/tensilelite"}
        self.assertEqual(
            self.scope({"projects/hipblaslt/tensilelite/foo.py"}, prefixes=prefixes),
            ["projects/hipblaslt/tensilelite"],
        )

    def test_monorepo_native_directory_is_attributable(self):
        """Absent from repos-config.json, but mapped TheRock-side."""
        self.assertEqual(
            self.scope({WORKFLOW, "projects/hipccl/hipccl3/CMakeLists.txt"}),
            ["projects/hipccl"],
        )

    def test_changes_that_cannot_be_attributed_are_not_scoped(self):
        """Each case yields [] -- which select_tests() pins to a full run."""
        cases = [
            ("no diff available", None, {}),
            ("empty diff", set(), {}),
            ("no configured subtrees", {WORKFLOW}, dict(prefixes=set())),
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

        for description, changed_files, kwargs in cases:
            with self.subTest(case=description):
                self.assertEqual(self.scope(changed_files, **kwargs), [])

    def test_test_selection_plumbing_outranks_an_attributable_source_change(self):
        """It rewrites the scope, so the scope it would narrow to is unusable."""
        self.assertEqual(
            self.scope({"shared/ctest/categories.yaml", "projects/rocblas/src/a.cpp"}),
            [],
        )

    def test_docs_only_paths_are_not_a_project_scope(self):
        self.assertEqual(
            determine_changed_projects(
                {"projects/rocblas/docs/index.md"},
                LIBRARY_PREFIXES,
                ["*.md"],
            ),
            [],
        )

    def test_empty_scope_selects_every_test(self):
        """Pins the contract the tests above rely on: [] means a full run.

        Without this, asserting determine_changed_projects() == [] would only
        pin an intermediate value and never show that a full test run results.
        """
        self.assertEqual(select_tests([]), {"*"})

    def test_libraries_workflow_does_not_select_rccl_downstream(self):
        """The regression this feature exists to prevent."""
        selected_tests = select_tests(self.scope({WORKFLOW}))

        self.assertNotEqual(selected_tests, {"*"})
        self.assertIn("rocblas", selected_tests)
        self.assertNotIn("rccl", selected_tests)


if __name__ == "__main__":
    unittest.main()
