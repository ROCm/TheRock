#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for configure_external_repo_ci.py.

That module is on its way out: it no longer decides test depth, only project
scope. Coverage here is deliberately limited to the scoping logic still in use.
Dead surface (has_non_skippable) and its duplicated CI-plumbing pattern lists
are not tested -- the live copy of those patterns is EXTERNAL_REPO_CI_PATTERNS
in configure_ci_path_filters.py, covered by configure_ci_path_filters_test.py.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.fspath(Path(__file__).parent.parent / "github_actions"))

from configure_ci_path_filters import (
    CI_RELEVANT_NON_SUBTREE_PREFIXES,
    find_matched_subtrees,
)
from configure_external_repo_ci import (
    RepoEntry,
    configure,
    get_valid_prefixes,
    load_repo_config,
)


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
    """Tests for configure() main logic.

    An empty return value means "could not narrow safely", which downstream
    means every project is tested.
    """

    @patch("configure_external_repo_ci.get_modified_paths_api")
    @patch("configure_external_repo_ci.load_repo_config")
    def test_only_docs_changed_no_projects(self, mock_config, mock_api):
        """Docs-only changes return no changed_projects (skip handled by TheRock)."""
        mock_api.return_value = {"README.md", "docs/guide.md"}
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
        self.assertEqual(result, "")

    @patch("configure_external_repo_ci.get_modified_paths_api")
    @patch("configure_external_repo_ci.load_repo_config")
    def test_only_codeowners_changed_no_projects(self, mock_config, mock_api):
        """CODEOWNERS-only changes return no changed_projects (skip handled by TheRock)."""
        mock_api.return_value = {".github/CODEOWNERS"}
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
        self.assertEqual(result, "")

    @patch("configure_external_repo_ci.get_modified_paths_api")
    def test_missing_repos_config_leaves_nothing_to_scope_to(self, mock_api):
        """Without the subtree registry there are no prefixes to attribute to."""
        mock_api.return_value = {".github/workflows/therock-ci.yml"}
        result = configure(
            event_name="pull_request",
            github_repo="ROCm/rocm-libraries",
            base_sha="abc123",
            head_sha="def456",
            config_path="",
        )
        self.assertEqual(result, "")

    @patch("configure_external_repo_ci.get_modified_paths_api")
    @patch("configure_external_repo_ci.load_repo_config")
    def test_ci_workflow_changed_scopes_to_the_callers_own_projects(
        self, mock_config, mock_api
    ):
        """A workflow edit cannot reach a project outside its own repo.

        The companion to the test above: given a registry to attribute against,
        the same diff narrows instead of widening to every project TheRock
        builds. Keeps downstream suites of other repos out of the run.
        """
        mock_api.return_value = {".github/workflows/therock-ci.yml"}
        mock_config.return_value = [
            RepoEntry(name="rocblas", url="", branch="", category="projects"),
            RepoEntry(name="hipblas", url="", branch="", category="projects"),
        ]
        result = configure(
            event_name="pull_request",
            github_repo="ROCm/rocm-libraries",
            base_sha="abc123",
            head_sha="def456",
            config_path=".github/repos-config.json",
        )
        self.assertEqual(result, "projects/hipblas,projects/rocblas")

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
        self.assertEqual(result, "projects/rocblas")

    @patch("configure_external_repo_ci.get_modified_paths_api")
    def test_truncated_api_response_is_not_scoped(self, mock_api):
        mock_api.return_value = None  # Signals truncated response
        result = configure(
            event_name="pull_request",
            github_repo="ROCm/rocm-libraries",
            base_sha="abc123",
            head_sha="def456",
            config_path="",
        )
        self.assertEqual(result, "")

    def test_no_shas_provided_is_not_scoped(self):
        result = configure(
            event_name="pull_request",
            github_repo="ROCm/rocm-libraries",
            base_sha=None,
            head_sha=None,
            config_path="",
        )
        self.assertEqual(result, "")

    def test_push_without_base_sha_is_not_scoped(self):
        """Push events without base_sha have no diff to narrow against."""
        result = configure(
            event_name="push",
            github_repo="ROCm/rocm-libraries",
            base_sha=None,
            head_sha="def456",
            config_path="",
        )
        self.assertEqual(result, "")


class ConfigureNonSubtreeTest(unittest.TestCase):
    """Surfacing of non-subtree paths and the unclassified-change fallback."""

    def _configure(self, paths, config=None, repo="ROCm/rocm-systems"):
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
                github_repo=repo,
                base_sha="abc123",
                head_sha="def456",
                config_path=".github/repos-config.json",
            )

    def test_shared_component_is_surfaced(self):
        r = self._configure(["shared/amdgpu-windows-interop/pal/x.cpp"])
        self.assertEqual(r, "shared/amdgpu-windows-interop")

    def test_emulation_components_are_surfaced(self):
        r = self._configure(["emulation/mirage/a.cpp", "emulation/rocjitsu/b.cpp"])
        self.assertEqual(
            sorted(r.split(",")),
            ["emulation/mirage", "emulation/rocjitsu"],
        )

    def test_ctest_harness_is_not_scoped(self):
        r = self._configure(["shared/ctest/TestCategories.cmake"])
        self.assertEqual(r, "")

    def test_dnn_provider_cmake_surfaces_prefix(self):
        r = self._configure(
            ["dnn-providers/cmake/Tests.cmake"],
            repo="ROCm/rocm-libraries",
        )
        self.assertEqual(r, "dnn-providers/cmake")

    def test_hipccl_monorepo_dir_surfaces_prefix(self):
        # projects/hipccl is monorepo-native (not subtree-synced); edits there
        # must classify to the projects/hipccl prefix, not fall back to the
        # full test suite.
        r = self._configure(
            ["projects/hipccl/hipccl3/rocprim/foo.hpp"],
            repo="ROCm/rocm-libraries",
        )
        self.assertEqual(r, "projects/hipccl")

    def test_mixed_recognized_and_unclassified_is_not_scoped(self):
        r = self._configure(
            ["projects/rocm-core/src/x.cpp", "tools/rocm-build/helper.py"]
        )
        self.assertEqual(r, "")

    def test_recognized_plus_skippable_still_narrows(self):
        r = self._configure(["projects/rocm-core/src/x.cpp", "README.md"])
        self.assertEqual(r, "projects/rocm-core")

    def test_declared_prefixes_are_wellformed(self):
        for prefix in CI_RELEVANT_NON_SUBTREE_PREFIXES:
            self.assertEqual(len(prefix.split("/")), 2, prefix)


if __name__ == "__main__":
    unittest.main()
