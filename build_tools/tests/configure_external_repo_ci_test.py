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

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, os.fspath(Path(__file__).parent.parent / "github_actions"))

from configure_external_repo_ci import (
    CI_RELEVANT_NON_SUBTREE_PREFIXES,
    RepoEntry,
    configure,
    main,
    find_matched_subtrees,
    get_modified_paths_api,
    get_pr_number_from_event,
    get_unclassified_paths,
    get_valid_prefixes,
    is_skippable,
    load_repo_config,
)


class IsSkippableTest(unittest.TestCase):
    """Tests for is_skippable()."""

    def test_markdown_files_are_skippable(self):
        self.assertTrue(is_skippable("README.md"))
        self.assertTrue(is_skippable("docs/guide.md"))

    def test_rst_files_are_skippable(self):
        self.assertTrue(is_skippable("index.rst"))

    def test_codeowners_files_are_skippable(self):
        self.assertTrue(is_skippable(".github/CODEOWNERS"))
        self.assertTrue(is_skippable("projects/rocblas/.github/CODEOWNERS"))

    def test_docs_directory_is_skippable(self):
        self.assertTrue(is_skippable("docs/api.txt"))
        self.assertTrue(is_skippable("projects/rocblas/docs/readme.md"))

    def test_source_files_are_not_skippable(self):
        self.assertFalse(is_skippable("src/main.cpp"))
        self.assertFalse(is_skippable("projects/rocblas/src/blas.cpp"))
        self.assertFalse(is_skippable("CMakeLists.txt"))


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
    def test_ci_workflow_changed_is_not_scoped(self, mock_api):
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

    @patch("configure_external_repo_ci.get_modified_paths_api")
    def test_pr_number_is_forwarded_to_api(self, mock_api):
        mock_api.return_value = None
        configure(
            event_name="pull_request",
            github_repo="ROCm/rocm-libraries",
            base_sha="abc123",
            head_sha="def456",
            config_path="",
            pr_number=15610,
        )
        mock_api.assert_called_once_with(
            "ROCm/rocm-libraries", "abc123", "def456", pr_number=15610
        )

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


class GetModifiedPathsApiTest(unittest.TestCase):
    """The PR diff must come from the paginated PR files endpoint."""

    @staticmethod
    def _gh_pages(total, per_page=100):
        """Fake `gh api` serving pulls/N/files in pages, like GitHub does."""
        names = [f"projects/hipccl/hipccl3/f{i}.hpp" for i in range(total)]

        def fake_run(cmd, **kwargs):
            url = cmd[-1]
            page = int(url.split("&page=")[1])
            chunk = names[(page - 1) * per_page : page * per_page]
            out = json.dumps([{"filename": n} for n in chunk])
            return Mock(stdout=out)

        return names, fake_run

    def test_pr_with_more_than_300_files_returns_every_path(self):
        names, fake_run = self._gh_pages(716)
        with patch(
            "configure_ci_path_filters.subprocess.run", side_effect=fake_run
        ) as run:
            paths = get_modified_paths_api(
                "ROCm/rocm-libraries", "base", "head", pr_number=15610
            )
        self.assertEqual(paths, set(names))
        urls = [c.args[0][-1] for c in run.call_args_list]
        self.assertEqual(len(urls), 8)  # 7 full pages, 1 short page
        self.assertTrue(all("pulls/15610/files?per_page=100" in u for u in urls))

    def test_pr_with_3000_files_is_truncated(self):
        _, fake_run = self._gh_pages(3000)
        with patch("configure_ci_path_filters.subprocess.run", side_effect=fake_run):
            paths = get_modified_paths_api(
                "ROCm/rocm-libraries", "base", "head", pr_number=1
            )
        self.assertIsNone(paths)

    def test_without_pr_number_uses_compare_and_caps_at_300(self):
        files = [{"filename": f"f{i}"} for i in range(300)]
        out = Mock(stdout=json.dumps({"files": files}))
        with patch(
            "configure_ci_path_filters.subprocess.run", return_value=out
        ) as run:
            paths = get_modified_paths_api("ROCm/rocm-libraries", "base", "head")
        self.assertIsNone(paths)
        self.assertIn("compare/base...head", run.call_args[0][0][-1])


class RenamedFilesTest(unittest.TestCase):
    """A renamed file changes both its old and its new path."""

    CONFIG = [
        RepoEntry(name="rocprim", url="", branch="", category="projects"),
        RepoEntry(name="hipcub", url="", branch="", category="projects"),
    ]
    MOVE = {
        "filename": "projects/hipcub/a.hpp",
        "previous_filename": "projects/rocprim/a.hpp",
        "status": "renamed",
    }

    def test_pr_files_include_previous_filename(self):
        out = Mock(stdout=json.dumps([self.MOVE]))
        with patch("configure_ci_path_filters.subprocess.run", return_value=out):
            paths = get_modified_paths_api(
                "ROCm/rocm-libraries", "base", "head", pr_number=1
            )
        self.assertEqual(paths, {"projects/hipcub/a.hpp", "projects/rocprim/a.hpp"})

    def test_compare_files_include_previous_filename(self):
        out = Mock(stdout=json.dumps({"files": [self.MOVE]}))
        with patch("configure_ci_path_filters.subprocess.run", return_value=out):
            paths = get_modified_paths_api("ROCm/rocm-libraries", "base", "head")
        self.assertEqual(paths, {"projects/hipcub/a.hpp", "projects/rocprim/a.hpp"})

    def test_move_between_projects_selects_both_projects(self):
        out = Mock(stdout=json.dumps([self.MOVE]))
        with patch("configure_ci_path_filters.subprocess.run", return_value=out), patch(
            "configure_external_repo_ci.load_repo_config", return_value=self.CONFIG
        ):
            result = configure(
                event_name="pull_request",
                github_repo="ROCm/rocm-libraries",
                base_sha="base",
                head_sha="head",
                config_path="",
                pr_number=1,
            )
        self.assertEqual(result, "projects/hipcub,projects/rocprim")

    def test_truncation_limit_counts_files_not_paths(self):
        # 1600 renames list 3200 paths but are only 1600 of the API's 3000 files.
        moves = [
            {"filename": f"projects/hipcub/f{i}.hpp",
             "previous_filename": f"projects/rocprim/f{i}.hpp"}
            for i in range(1600)
        ]

        def fake_run(cmd, **kwargs):
            page = int(cmd[-1].split("&page=")[1])
            chunk = moves[(page - 1) * 100 : page * 100]
            return Mock(stdout=json.dumps(chunk))

        with patch("configure_ci_path_filters.subprocess.run", side_effect=fake_run):
            paths = get_modified_paths_api(
                "ROCm/rocm-libraries", "base", "head", pr_number=1
            )
        self.assertEqual(len(paths), 3200)


class LargePrEndToEndTest(unittest.TestCase):
    """configure()/main() with only the `gh` subprocess faked."""

    CONFIG = [
        RepoEntry(name="hipcub", url="", branch="", category="projects"),
        RepoEntry(name="rocprim", url="", branch="", category="projects"),
    ]

    @staticmethod
    def _pr_files_run(paths):
        def fake_run(cmd, **kwargs):
            page = int(cmd[-1].split("&page=")[1])
            chunk = paths[(page - 1) * 100 : page * 100]
            return Mock(stdout=json.dumps([{"filename": p} for p in chunk]))

        return fake_run

    def _paths(self):
        paths = [f"projects/hipccl/hipccl3/f{i}.hpp" for i in range(715)]
        return paths + ["projects/hipcub/hipcub/include/x.hpp"]

    def test_configure_scopes_a_716_file_pr(self):
        run = self._pr_files_run(self._paths())
        with patch("configure_ci_path_filters.subprocess.run", side_effect=run), patch(
            "configure_external_repo_ci.load_repo_config", return_value=self.CONFIG
        ):
            result = configure(
                event_name="pull_request",
                github_repo="ROCm/rocm-libraries",
                base_sha="base",
                head_sha="head",
                config_path="",
                pr_number=15610,
            )
        self.assertEqual(result, "projects/hipccl,projects/hipcub")

    def test_main_reads_pr_number_from_event_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            event = Path(tmp) / "event.json"
            event.write_text(json.dumps({"pull_request": {"number": 15610}}))
            env = {"GITHUB_EVENT_PATH": os.fspath(event)}
            with patch.dict(os.environ, env, clear=True), patch(
                "configure_ci_path_filters.subprocess.run",
                side_effect=self._pr_files_run(self._paths()),
            ) as run, patch(
                "configure_external_repo_ci.load_repo_config",
                return_value=self.CONFIG,
            ), patch(
                "configure_external_repo_ci.set_github_output"
            ) as output:
                rc = main(
                    [
                        "--event-name", "pull_request",
                        "--github-repo", "ROCm/rocm-libraries",
                        "--base-sha", "base",
                        "--head-sha", "head",
                    ]
                )
        self.assertEqual(rc, 0)
        self.assertIn("pulls/15610/files", run.call_args_list[0].args[0][-1])
        output.assert_called_once_with({"changed_projects": "projects/hipccl,projects/hipcub"})


class GetPrNumberFromEventTest(unittest.TestCase):
    def _number(self, payload=None, raw=None, env=True):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "event.json"
            if raw is not None:
                path.write_text(raw, encoding="utf-8")
            elif payload is not None:
                path.write_text(json.dumps(payload), encoding="utf-8")
            environ = {"GITHUB_EVENT_PATH": os.fspath(path)} if env else {}
            with patch.dict(os.environ, environ, clear=True):
                return get_pr_number_from_event()

    def test_reads_pull_request_number(self):
        self.assertEqual(self._number({"pull_request": {"number": 15610}}), 15610)

    def test_push_event_has_no_number(self):
        self.assertIsNone(self._number({"ref": "refs/heads/develop"}))

    def test_unset_env_returns_none(self):
        self.assertIsNone(self._number({}, env=False))

    def test_missing_file_returns_none(self):
        with patch.dict(os.environ, {"GITHUB_EVENT_PATH": "/nonexistent.json"}):
            self.assertIsNone(get_pr_number_from_event())

    def test_malformed_json_returns_none(self):
        self.assertIsNone(self._number(raw="{not json"))


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
