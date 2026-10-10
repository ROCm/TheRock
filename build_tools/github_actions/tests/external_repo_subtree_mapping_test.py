# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the external-repo change-to-subtree mapping.

Covers the subtree matching itself and the lock-step between
CI_RELEVANT_NON_SUBTREE_PREFIXES here and the test selector's mapping in
determine_rocm_test_dependencies: every non-subtree prefix listed here must
resolve to a graph key, otherwise the selector hard-fails on it.
"""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).resolve().parent.parent))
import configure_external_repo_ci as cerc

sys.path.insert(0, os.fspath(Path(__file__).resolve().parents[3] / "test_tools"))
import determine_rocm_test_dependencies as drtd


class FindMatchedSubtreesTest(unittest.TestCase):
    def test_longest_prefix_wins(self):
        prefixes = {"projects/hipblaslt", "projects/hipblaslt/tensilelite"}
        matched = cerc.find_matched_subtrees(
            ["projects/hipblaslt/tensilelite/foo.cpp"], prefixes
        )
        self.assertEqual(matched, ["projects/hipblaslt/tensilelite"])

    def test_parent_match_when_no_nested_prefix(self):
        prefixes = {"projects/hipblaslt", "projects/hipblaslt/tensilelite"}
        matched = cerc.find_matched_subtrees(
            ["projects/hipblaslt/src/foo.cpp"], prefixes
        )
        self.assertEqual(matched, ["projects/hipblaslt"])

    def test_each_file_attributed_to_one_subtree(self):
        prefixes = {"projects/rdc", "projects/hipblaslt"}
        matched = cerc.find_matched_subtrees(
            ["projects/rdc/a.cpp", "projects/hipblaslt/b.cpp", "projects/rdc/c.cpp"],
            prefixes,
        )
        self.assertEqual(matched, ["projects/hipblaslt", "projects/rdc"])

    def test_no_match_returns_empty(self):
        matched = cerc.find_matched_subtrees(["README.md"], {"projects/rdc"})
        self.assertEqual(matched, [])


class GetValidPrefixesTest(unittest.TestCase):
    def test_builds_category_slash_name(self):
        config = [
            cerc.RepoEntry(name="rdc", url="u", branch="b", category="projects"),
            cerc.RepoEntry(name="rccl", url="u", branch="b", category="projects"),
        ]
        self.assertEqual(
            cerc.get_valid_prefixes(config), {"projects/rdc", "projects/rccl"}
        )


class GetUnclassifiedPathsTest(unittest.TestCase):
    def test_flags_unmapped_non_skippable_path(self):
        prefixes = {"projects/rdc"}
        unclassified = cerc.get_unclassified_paths(
            ["projects/rdc/a.cpp", "top_level_change.cmake"], prefixes
        )
        self.assertEqual(unclassified, ["top_level_change.cmake"])

    def test_ignores_skippable_paths(self):
        unclassified = cerc.get_unclassified_paths(
            ["docs/overview.md", "README.rst"], set()
        )
        self.assertEqual(unclassified, [])

    def test_ci_relevant_prefix_is_classified(self):
        # A path under a CI-relevant non-subtree dir must NOT be unclassified when
        # that prefix is in the valid set (as configure() builds it).
        prefixes = {"projects/rdc"} | cerc.CI_RELEVANT_NON_SUBTREE_PREFIXES
        unclassified = cerc.get_unclassified_paths(["shared/kpack/build.py"], prefixes)
        self.assertEqual(unclassified, [])

    def test_unknown_non_subtree_dir_is_unclassified(self):
        # A shared/* dir absent from both repos-config and the CI-relevant set is
        # unclassified, so the caller falls back to a full run.
        prefixes = {"projects/rdc"} | cerc.CI_RELEVANT_NON_SUBTREE_PREFIXES
        unclassified = cerc.get_unclassified_paths(
            ["shared/not-a-real-dir/x.cpp"], prefixes
        )
        self.assertEqual(unclassified, ["shared/not-a-real-dir/x.cpp"])


class ExternalRepoToSelectorChainTest(unittest.TestCase):
    """The S1 -> selector link: a matched non-subtree prefix must resolve to a
    graph key in determine_rocm_test_dependencies, with no hard-fail."""

    def test_non_subtree_change_resolves_to_graph_keys(self):
        prefixes = {"projects/rdc"} | cerc.CI_RELEVANT_NON_SUBTREE_PREFIXES
        matched = cerc.find_matched_subtrees(
            ["shared/amdgpu-windows-interop/interop.cpp"], prefixes
        )
        self.assertEqual(matched, ["shared/amdgpu-windows-interop"])
        self.assertEqual(drtd._normalize_changed_project(matched[0]), ["hip-clr"])

    def test_every_ci_relevant_prefix_resolves_in_selector(self):
        # The comment on CI_RELEVANT_NON_SUBTREE_PREFIXES promises each entry maps
        # in TheRock; the selector hard-fails on an unmapped shared/dnn-providers/
        # emulation path. This keeps the two modules in lock-step.
        for prefix in sorted(cerc.CI_RELEVANT_NON_SUBTREE_PREFIXES):
            with self.subTest(prefix=prefix):
                keys = drtd._normalize_changed_project(prefix)
                self.assertTrue(keys, f"{prefix} resolved to nothing")


if __name__ == "__main__":
    unittest.main()
