#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for configure_external_repo_ci.py."""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.fspath(Path(__file__).parent.parent / "github_actions"))

from configure_external_repo_ci import (
    CI_RELEVANT_NON_SUBTREE_PREFIXES,
    ConfigureResult,
    RepoEntry,
    configure,
    find_matched_subtrees,
    get_unclassified_paths,
    get_valid_prefixes,
    has_non_skippable,
    is_skippable,
    load_repo_config,
    matches_patterns,
)


class IsSkippableTest(unittest.TestCase):
    """Tests for is_skippable()."""

    def test_markdown_files_are_skippable(self):
        self.assertTrue(is_skippable("README.md"))
        self.assertTrue(is_skippable("docs/guide.md"))

    def test_rst_files_are_skippable(self):
        self.assertTrue(is_skippable("index.rst"))

    def test_docs_directory_is_skippable(self):
        self.assertTrue(is_skippable("docs/api.txt"))
        self.assertTrue(is_skippable("projects/rocblas/docs/readme.md"))

    def test_source_files_are_not_skippable(self):
        self.assertFalse(is_skippable("src/main.cpp"))
        self.assertFalse(is_skippable("projects/rocblas/src/blas.cpp"))
        self.assertFalse(is_skippable("CMakeLists.txt"))


class HasNonSkippableTest(unittest.TestCase):
    """Tests for has_non_skippable()."""

    def test_all_skippable_returns_false(self):
        paths = ["README.md", "docs/guide.rst", "CHANGELOG.md"]
        self.assertFalse(has_non_skippable(paths))

    def test_mixed_returns_true(self):
        paths = ["README.md", "src/main.cpp"]
        self.assertTrue(has_non_skippable(paths))

    def test_all_non_skippable_returns_true(self):
        paths = ["src/a.cpp", "src/b.cpp"]
        self.assertTrue(has_non_skippable(paths))


class MatchesPatternsTest(unittest.TestCase):
    """Tests for matches_patterns()."""

    def test_matches_workflow_pattern(self):
        paths = [".github/workflows/therock-ci.yml"]
        patterns = [".github/workflows/therock*"]
        self.assertTrue(matches_patterns(paths, patterns))

    def test_no_match_returns_false(self):
        paths = ["src/main.cpp"]
        patterns = [".github/workflows/therock*"]
        self.assertFalse(matches_patterns(paths, patterns))

    def test_empty_paths_returns_false(self):
        self.assertFalse(matches_patterns([], ["*.md"]))


class FindMatchedSubtreesTest(unittest.TestCase):
    """Tests for find_matched_subtrees()."""

    def test_finds_valid_prefixes(self):
        files = ["projects/rocblas/src/main.cpp", "projects/hipblas/CMakeLists.txt"]
        prefixes = {"projects/rocblas", "projects/hipblas", "projects/rocfft"}
        result = find_matched_subtrees(files, prefixes)
        self.assertEqual(result, ["projects/hipblas", "projects/rocblas"])

    def test_ignores_invalid_prefixes(self):
        files = ["projects/unknown/file.cpp", "random/file.txt"]
        prefixes = {"projects/rocblas"}
        result = find_matched_subtrees(files, prefixes)
        self.assertEqual(result, [])

    def test_handles_single_segment_paths(self):
        files = ["README.md"]
        prefixes = {"projects/rocblas"}
        result = find_matched_subtrees(files, prefixes)
        self.assertEqual(result, [])

    def test_nested_subtree_wins_over_parent(self):
        # A file inside a registered nested subtree (e.g. hipblaslt/tensilelite)
        # must match the longer, more specific prefix, not collapse to its
        # 2-segment parent -- a fixed-length truncation would make the two
        # indistinguishable and silently lose the more specific match.
        files = ["projects/hipblaslt/tensilelite/Tensile/KernelWriter.py"]
        prefixes = {"projects/hipblaslt", "projects/hipblaslt/tensilelite"}
        result = find_matched_subtrees(files, prefixes)
        self.assertEqual(result, ["projects/hipblaslt/tensilelite"])

    def test_parent_only_change_does_not_match_nested_subtree(self):
        # A change outside the nested subtree still matches the parent, not the
        # (unrelated) nested prefix.
        files = ["projects/hipblaslt/library/src/Handle.cpp"]
        prefixes = {"projects/hipblaslt", "projects/hipblaslt/tensilelite"}
        result = find_matched_subtrees(files, prefixes)
        self.assertEqual(result, ["projects/hipblaslt"])

    def test_mixed_nested_and_parent_changes_match_both(self):
        # Changes to both areas in the same PR attribute independently: one
        # file matches the nested subtree, the other matches the parent.
        files = [
            "projects/hipblaslt/tensilelite/Tensile/KernelWriter.py",
            "projects/hipblaslt/library/src/Handle.cpp",
        ]
        prefixes = {"projects/hipblaslt", "projects/hipblaslt/tensilelite"}
        result = find_matched_subtrees(files, prefixes)
        self.assertEqual(
            result, ["projects/hipblaslt", "projects/hipblaslt/tensilelite"]
        )


class GetValidPrefixesTest(unittest.TestCase):
    """Tests for get_valid_prefixes()."""

    def test_extracts_prefixes(self):
        config = [
            RepoEntry(name="rocblas", url="", branch="", category="projects"),
            RepoEntry(name="hipblas", url="", branch="", category="projects"),
        ]
        result = get_valid_prefixes(config)
        self.assertEqual(result, {"projects/rocblas", "projects/hipblas"})


class LoadRepoConfigTest(unittest.TestCase):
    """Tests for load_repo_config()."""

    def test_handles_missing_file(self):
        result = load_repo_config("/nonexistent/path.json")
        self.assertEqual(result, [])

    def test_ignores_unknown_fields(self):
        """Unknown fields in config should be ignored, not cause TypeError."""
        import tempfile
        import json

        config_data = {
            "repositories": [
                {
                    "name": "rocblas",
                    "url": "https://github.com/ROCm/rocBLAS",
                    "branch": "develop",
                    "category": "projects",
                    "future_unknown_field": "should be ignored",
                }
            ]
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            result = load_repo_config(temp_path)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].name, "rocblas")
        finally:
            os.unlink(temp_path)


class ConfigureTest(unittest.TestCase):
    """Tests for configure() main logic."""

    def test_schedule_event_runs_all_tests(self):
        result = configure(
            event_name="schedule",
            github_repo="ROCm/rocm-libraries",
            base_sha=None,
            head_sha=None,
            config_path="",
        )
        self.assertEqual(result.run_all_tests, True)
        self.assertEqual(result.skip_tests, False)

    def test_workflow_dispatch_runs_all_tests(self):
        result = configure(
            event_name="workflow_dispatch",
            github_repo="ROCm/rocm-libraries",
            base_sha=None,
            head_sha=None,
            config_path="",
        )
        self.assertEqual(result.run_all_tests, True)
        self.assertEqual(result.skip_tests, False)

    @patch("configure_external_repo_ci.get_modified_paths_api")
    def test_only_docs_changed_skips_tests(self, mock_api):
        mock_api.return_value = {"README.md", "docs/guide.md"}
        result = configure(
            event_name="pull_request",
            github_repo="ROCm/rocm-libraries",
            base_sha="abc123",
            head_sha="def456",
            config_path="",
        )
        self.assertEqual(result.skip_tests, True)
        self.assertEqual(result.run_all_tests, False)

    @patch("configure_external_repo_ci.get_modified_paths_api")
    def test_ci_workflow_changed_runs_all_tests(self, mock_api):
        mock_api.return_value = {".github/workflows/therock-ci.yml"}
        result = configure(
            event_name="pull_request",
            github_repo="ROCm/rocm-libraries",
            base_sha="abc123",
            head_sha="def456",
            config_path="",
        )
        self.assertEqual(result.run_all_tests, True)
        self.assertEqual(result.skip_tests, False)

    @patch("configure_external_repo_ci.get_modified_paths_api")
    @patch("configure_external_repo_ci.load_repo_config")
    def test_project_change_returns_changed_projects(self, mock_config, mock_api):
        mock_api.return_value = {"projects/rocblas/src/main.cpp"}
        mock_config.return_value = [
            RepoEntry(name="rocblas", url="", branch="", category="projects"),
        ]
        result = configure(
            event_name="pull_request",
            github_repo="ROCm/rocm-libraries",
            base_sha="abc123",
            head_sha="def456",
            config_path=".github/repos-config.json",
        )
        self.assertEqual(result.changed_projects, "projects/rocblas")
        self.assertEqual(result.run_all_tests, False)
        self.assertEqual(result.skip_tests, False)

    @patch("configure_external_repo_ci.get_modified_paths_api")
    def test_truncated_api_response_runs_all_tests(self, mock_api):
        mock_api.return_value = None  # Signals truncated response
        result = configure(
            event_name="pull_request",
            github_repo="ROCm/rocm-libraries",
            base_sha="abc123",
            head_sha="def456",
            config_path="",
        )
        self.assertEqual(result.run_all_tests, True)
        self.assertEqual(result.skip_tests, False)

    def test_no_shas_provided_runs_all_tests(self):
        result = configure(
            event_name="pull_request",
            github_repo="ROCm/rocm-libraries",
            base_sha=None,
            head_sha=None,
            config_path="",
        )
        self.assertEqual(result.run_all_tests, True)

    def test_push_without_base_sha_runs_all_tests(self):
        """Push events without base_sha should run all tests."""
        result = configure(
            event_name="push",
            github_repo="ROCm/rocm-libraries",
            base_sha=None,
            head_sha="def456",
            config_path="",
        )
        self.assertEqual(result.run_all_tests, True)


class GetUnclassifiedPathsTest(unittest.TestCase):
    """Tests for get_unclassified_paths()."""

    def test_unmapped_nonskippable_is_unclassified(self):
        valid = {"projects/rocblas"}
        self.assertEqual(
            get_unclassified_paths(["tools/build/x.py"], valid),
            ["tools/build/x.py"],
        )

    def test_recognized_and_skippable_are_not_unclassified(self):
        valid = {"projects/rocblas"}
        self.assertEqual(
            get_unclassified_paths(["projects/rocblas/src/a.cpp", "README.md"], valid),
            [],
        )


class ConfigureNonSubtreeTest(unittest.TestCase):
    """Surfacing of non-subtree shared/* + emulation/* paths and the
    unclassified-change fallback (rocm-systems multi-arch gap)."""

    def _configure(self, paths, config=None):
        with patch(
            "configure_external_repo_ci.get_modified_paths_api",
            return_value=set(paths),
        ), patch(
            "configure_external_repo_ci.load_repo_config",
            return_value=config
            or [RepoEntry(name="rocm-core", url="", branch="", category="projects")],
        ):
            return configure(
                event_name="pull_request",
                github_repo="ROCm/rocm-systems",
                base_sha="abc123",
                head_sha="def456",
                config_path=".github/repos-config.json",
            )

    def test_shared_component_is_surfaced(self):
        r = self._configure(["shared/amdgpu-windows-interop/pal/x.cpp"])
        self.assertEqual(r.changed_projects, "shared/amdgpu-windows-interop")
        self.assertFalse(r.run_all_tests)
        self.assertFalse(r.skip_tests)

    def test_emulation_components_are_surfaced(self):
        r = self._configure(["emulation/mirage/a.cpp", "emulation/rocjitsu/b.cpp"])
        self.assertEqual(
            sorted(r.changed_projects.split(",")),
            ["emulation/mirage", "emulation/rocjitsu"],
        )

    def test_ctest_harness_triggers_full_run(self):
        r = self._configure(["shared/ctest/TestCategories.cmake"])
        self.assertTrue(r.run_all_tests)
        self.assertEqual(r.changed_projects, "")

    def test_mixed_recognized_and_unclassified_runs_all(self):
        r = self._configure(
            ["projects/rocm-core/src/x.cpp", "tools/rocm-build/helper.py"]
        )
        self.assertTrue(r.run_all_tests)
        self.assertEqual(r.changed_projects, "")

    def test_recognized_plus_skippable_still_narrows(self):
        r = self._configure(["projects/rocm-core/src/x.cpp", "README.md"])
        self.assertFalse(r.run_all_tests)
        self.assertEqual(r.changed_projects, "projects/rocm-core")

    def test_declared_prefixes_are_wellformed(self):
        for prefix in CI_RELEVANT_NON_SUBTREE_PREFIXES:
            self.assertEqual(len(prefix.split("/")), 2, prefix)


if __name__ == "__main__":
    unittest.main()
