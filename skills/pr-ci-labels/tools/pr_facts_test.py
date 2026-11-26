# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for pr_facts.py (the pr-ci-labels skill's one-call fact gatherer).

Run locally:
    python -m pytest skills/pr-ci-labels/tools/pr_facts_test.py
"""

import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import pr_facts as pf  # noqa: E402

SKIP_PATTERNS = ["*.md", "docs/*"]


class TouchedProjectsTest(unittest.TestCase):
    def test_project_roots_map_to_two_segments(self):
        self.assertEqual(
            pf.touched_projects(
                [
                    "projects/hipblaslt/clients/common/include/testing_matmul.hpp",
                    "shared/stinkytofu/src/passes/WaitInsertion.cpp",
                    "dnn-providers/miopen-provider/src/a.cpp",
                ]
            ),
            [
                "dnn-providers/miopen-provider",
                "projects/hipblaslt",
                "shared/stinkytofu",
            ],
        )

    def test_nested_tensilelite_is_its_own_project(self):
        self.assertEqual(
            pf.touched_projects(
                ["projects/hipblaslt/tensilelite/Tensile/KernelWriter.py"]
            ),
            ["projects/hipblaslt/tensilelite"],
        )

    def test_top_level_and_docs_paths_name_no_project(self):
        self.assertEqual(
            pf.touched_projects(
                ["README.md", "docs/install.rst", ".github/workflows/x.yml"]
            ),
            [],
        )


class BuildFactsTest(unittest.TestCase):
    def _facts(self, repo, files, pr=None):
        with mock.patch.object(
            pf, "default_test_selection", return_value=(["hipblaslt"], "stub")
        ):
            return pf.build_facts(
                repo, files, SKIP_PATTERNS, "skip-ci patterns", pr, ["ci:asan"]
            )

    def test_all_skippable_docs_change(self):
        facts = self._facts("ROCm/rocm-libraries", ["projects/rocsparse/README.md"])
        self.assertTrue(facts["all_skippable"])
        self.assertEqual(
            facts["files"][0],
            {"path": "projects/rocsparse/README.md", "skippable": True},
        )

    def test_code_change_lists_projects_and_default_tests(self):
        facts = self._facts(
            "ROCm/rocm-libraries",
            ["projects/hipblaslt/clients/tests/src/matmul_gtest.cpp"],
            pr={
                "number": 1,
                "isDraft": True,
                "labels": [{"name": "ci:asan"}, {"name": "project: hipblaslt"}],
            },
        )
        self.assertFalse(facts["all_skippable"])
        self.assertEqual(facts["projects"], ["projects/hipblaslt"])
        self.assertEqual(facts["default_tests"], ["hipblaslt"])
        self.assertEqual(facts["labels"], ["ci:asan"])
        self.assertTrue(facts["draft"])

    def test_therock_pr_has_no_project_selection(self):
        facts = self._facts("ROCm/TheRock", ["build_tools/fetch_sources.py"])
        self.assertEqual(facts["projects"], [])
        self.assertIsNone(facts["default_tests"])

    def test_unknown_file_list_is_flagged(self):
        facts = self._facts("ROCm/rocm-libraries", None)
        self.assertFalse(facts["files_known"])
        self.assertFalse(facts["all_skippable"])


class DefaultTestSelectionTest(unittest.TestCase):
    def test_no_projects(self):
        self.assertEqual(
            pf.default_test_selection([]), (None, "no project paths changed")
        )

    def test_selector_failure_is_reported(self):
        failed = subprocess.CompletedProcess([], 1, stdout="", stderr="boom")
        with mock.patch.object(pf.subprocess, "run", return_value=failed):
            tests, note = pf.default_test_selection(["projects/hipblaslt"])
        self.assertIsNone(tests)
        self.assertIn("boom", note)

    def test_real_selector_includes_the_changed_project(self):
        # Uses TheRock's committed consumer graph, next to this skill.
        tests, _ = pf.default_test_selection(["projects/hipblaslt"])
        self.assertIn("hipblaslt", tests)


class PresubmitFamiliesTest(unittest.TestCase):
    def test_reads_this_therock_checkouts_family_matrix(self):
        families = pf.presubmit_families()
        self.assertEqual(set(families), {"build_and_test", "build_only"})
        # gfx94x on Linux is TheRock's primary PR test family.
        self.assertIn("gfx94x (linux)", families["build_and_test"])


class GroupRepoLabelsTest(unittest.TestCase):
    def test_groups_drop_implied_prefixes(self):
        groups = pf.group_repo_labels(
            [
                "ci:asan",
                "ci:gfx950-dcgpu",
                "ci:build:gfx950",
                "ci:test:gfx950",
                "test:hipblaslt",
                "test_filter:quick",
            ]
        )
        self.assertEqual(groups["control"], ["ci:asan"])
        self.assertEqual(groups["ci:gfx..."], ["ci:gfx950-dcgpu"])
        self.assertEqual(groups["ci:build:..."], ["gfx950"])
        self.assertEqual(groups["ci:test:..."], ["gfx950"])
        self.assertEqual(groups["test:..."], ["hipblaslt"])
        self.assertEqual(groups["test_filter:..."], ["quick"])


class RepoCiLabelsTest(unittest.TestCase):
    def test_keeps_only_ci_labels(self):
        out = json.dumps(
            [{"name": "ci:asan"}, {"name": "bug"}, {"name": "test:hipblaslt"}]
        )
        with mock.patch.object(pf.cl, "run", return_value=(0, out, "")):
            self.assertEqual(pf.repo_ci_labels("o/r"), ["ci:asan", "test:hipblaslt"])

    def test_failure_is_none(self):
        with mock.patch.object(pf.cl, "run", return_value=(1, "", "err")):
            self.assertIsNone(pf.repo_ci_labels("o/r"))


if __name__ == "__main__":
    unittest.main()
