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
    ConfigureResult,
    RepoEntry,
    configure,
    find_matched_subtrees,
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


if __name__ == "__main__":
    unittest.main()
