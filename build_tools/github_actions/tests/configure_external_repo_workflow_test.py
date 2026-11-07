"""Regression tests for external-repository workflow changes."""

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
COMPONENT_ENTRIES = LIBRARY_ENTRIES + [
    ("projects", "hipblaslt"),
    ("projects/hipblaslt", "tensilelite"),
    ("projects", "hipdnn"),
    ("projects", "rocfft"),
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

    def test_libraries_workflow_selects_library_projects(self):
        result = self.configure({WORKFLOW})

        self.assertFalse(result.run_all_tests)
        self.assertEqual(set(result.changed_projects.split(",")), LIBRARY_PROJECTS)
        self.assertNotIn("projects/rccl", result.changed_projects.split(","))

    def test_libraries_workflow_with_known_source_selects_all_libraries(self):
        result = self.configure({WORKFLOW, "projects/rocblas/src/rocblas.cpp"})

        self.assertFalse(result.run_all_tests)
        self.assertEqual(set(result.changed_projects.split(",")), LIBRARY_PROJECTS)

    def test_systems_workflow_preserves_full_test_run(self):
        result = self.configure(
            {WORKFLOW},
            repo="ROCm/rocm-systems",
            entries=SYSTEM_ENTRIES,
        )

        self.assertTrue(result.run_all_tests)
        self.assertEqual(result.changed_projects, "")

    def test_shared_ci_changes_preserve_full_test_run(self):
        for path in (
            ".github/repos-config.json",
            ".github/scripts/get_changed_projects.py",
            "shared/ctest/categories.yaml",
        ):
            with self.subTest(path=path):
                result = self.configure({WORKFLOW, path})

                self.assertTrue(result.run_all_tests)
                self.assertEqual(result.changed_projects, "")

    def test_workflow_with_unclassified_path_preserves_full_test_run(self):
        result = self.configure({WORKFLOW, "unexpected/config.py"})

        self.assertTrue(result.run_all_tests)
        self.assertEqual(result.changed_projects, "")

    def test_workflow_without_config_preserves_full_test_run(self):
        result = self.configure({WORKFLOW}, config_exists=False)

        self.assertTrue(result.run_all_tests)
        self.assertEqual(result.changed_projects, "")

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

        self.assertFalse(result.run_all_tests)
        self.assertEqual(result.changed_projects, "projects/example")

    def test_unknown_repository_workflow_preserves_full_test_run(self):
        result = self.configure({WORKFLOW}, repo="ROCm/unknown")

        self.assertTrue(result.run_all_tests)
        self.assertEqual(result.changed_projects, "")

    def test_ordinary_source_changes_stay_selective(self):
        result = self.configure({"projects/rocblas/src/rocblas.cpp"})

        self.assertFalse(result.run_all_tests)
        self.assertEqual(result.changed_projects, "projects/rocblas")

    def test_component_ci_files_select_their_owners(self):
        cases = [
            (
                {".github/workflows/hipblaslt-asan-ci.yml"},
                {"projects/hipblaslt"},
            ),
            (
                {
                    "projects/hipblaslt/clients/common/include/hipblaslt_test.hpp",
                    ".github/workflows/hipblaslt-asan-ci.yml",
                },
                {"projects/hipblaslt"},
            ),
            (
                {".github/workflows/therock-rocjitsu-race-check-linux.yml"},
                {"projects/hipblaslt/tensilelite"},
            ),
            (
                {".github/workflows/component-ci-rocisa.yml"},
                {"projects/hipblaslt/tensilelite", "shared/stinkytofu"},
            ),
            (
                {
                    ".github/scripts/run_rocjitsu_hipblaslt_gemm_race_checks.sh",
                    ".github/workflows/component-ci-hipblaslt-rocjitsu-scripts.yml",
                    "test/therock/test_rocfft.py",
                },
                {"projects/hipblaslt/tensilelite", "projects/rocfft"},
            ),
        ]
        for paths, expected in cases:
            with self.subTest(paths=paths):
                result = self.configure(paths, entries=COMPONENT_ENTRIES)
                self.assertFalse(result.run_all_tests)
                self.assertEqual(set(result.changed_projects.split(",")), expected)
                self.assertEqual(result.test_type, "standard")

    def test_component_ownership_preserves_broad_coverage_when_needed(self):
        for path in (
            "CMakeLists.txt",
            ".github/actions/ci-env/action.yml",
            ".github/scripts/component_ci.py",
            "shared/ctest/parse_test_categories.py",
            "test/therock/test_runner.py",
            "unexpected/build.py",
            ".github/workflows/hipblaslt-new-workflow.yml",
            "test/therock/test_new_component.py",
        ):
            with self.subTest(path=path):
                result = self.configure(
                    {".github/workflows/hipblaslt-asan-ci.yml", path},
                    entries=COMPONENT_ENTRIES,
                )
                self.assertTrue(result.run_all_tests)
                self.assertEqual(result.changed_projects, "")
                self.assertEqual(result.test_type, "quick")

    def test_component_ownership_requires_the_repository_and_owner(self):
        for repo, entries in (
            ("ROCm/rocm-systems", COMPONENT_ENTRIES),
            ("ROCm/rocm-libraries", LIBRARY_ENTRIES),
        ):
            with self.subTest(repo=repo, entries=entries):
                result = self.configure(
                    {".github/workflows/hipblaslt-asan-ci.yml"},
                    repo=repo,
                    entries=entries,
                )
                self.assertTrue(result.run_all_tests)

    def test_component_file_with_shared_workflow_selects_all_library_projects(self):
        result = self.configure(
            {WORKFLOW, ".github/workflows/hipblaslt-asan-ci.yml"},
            entries=COMPONENT_ENTRIES,
        )
        self.assertFalse(result.run_all_tests)
        self.assertEqual(
            set(result.changed_projects.split(",")),
            {f"{category}/{name}" for category, name in COMPONENT_ENTRIES},
        )

    def test_shared_workflow_retains_changed_non_subtree_project(self):
        result = self.configure({WORKFLOW, "projects/hipccl/CMakeLists.txt"})
        self.assertFalse(result.run_all_tests)
        self.assertEqual(
            set(result.changed_projects.split(",")),
            LIBRARY_PROJECTS | {"projects/hipccl"},
        )

    def test_component_ci_files_retain_consumers_without_unrelated_tests(self):
        therock_root = Path(ci.__file__).resolve().parents[2]
        selector = therock_root / "test_tools/determine_rocm_test_dependencies.py"
        for path, expected in (
            (
                ".github/workflows/hipblaslt-asan-ci.yml",
                {"hipblaslt", "rocblas", "hipsparselt", "hipblasltprovider"},
            ),
            (
                ".github/workflows/component-ci-tensilelite-coverage.yml",
                {"hipblaslt", "tensilelite", "tensilelite-common", "hipsparselt"},
            ),
            (
                "test/therock/test_hipdnn_cudnn_samples.py",
                {"hipdnn", "hipdnn-cudnn-samples"},
            ),
        ):
            with self.subTest(path=path):
                result = self.configure({path}, entries=COMPONENT_ENTRIES)
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
                selected = set(selection.stdout.splitlines())
                self.assertTrue(expected <= selected)
                self.assertTrue(selected.isdisjoint({"rocfft", "hipfft", "rccl", "*"}))

    def test_libraries_workflow_does_not_select_rccl_downstream(self):
        result = self.configure({WORKFLOW})
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
