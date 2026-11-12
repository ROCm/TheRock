# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Every test-matrix job must be reachable from consumer-graph test selection.

fetch_test_configurations.py schedules a job when its selection key is in
PROJECTS_TO_TEST, or when that value is the "*" wildcard. A job that no component
can ever select therefore runs only under "*" -- that is, only when selection is
disabled entirely. Such a job looks healthy (it runs on wildcard builds) right up
until a caller starts passing a narrowed project list, at which point it silently
stops running.

These tests close that gap from both ends: every job must be selectable by some
component, and every alias must name a job that exists.
"""

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "build_tools" / "github_actions"))
sys.path.insert(0, str(_REPO_ROOT / "test_tools"))

import determine_rocm_test_dependencies as test_deps  # noqa: E402
from fetch_test_configurations import test_matrix  # noqa: E402

# Sanity is scheduled unconditionally by fetch_test_configurations.py (it gates
# the rest of the matrix), so it is exempt from needing a selecting component.
_ALWAYS_SCHEDULED = {"sanity"}


def _selection_keys() -> set[str]:
    """Return the keys fetch_test_configurations.py matches against selection.

    Split jobs (e.g. the multi-GPU variants) declare a "base_project" and are
    selected through it rather than their own matrix key.
    """
    return {
        config.get("base_project", key) for key, config in test_matrix.items()
    } - _ALWAYS_SCHEDULED


def _reachable_selectors() -> set[str]:
    """Return every selector any single component change can produce."""
    reachable: set[str] = set()
    for component in test_deps.list_subprojects():
        selected = test_deps.get_subprojects_to_test([component])
        reachable |= test_deps._to_ci_test_selectors(selected)
    return reachable


class TestMatrixReachability(unittest.TestCase):
    def test_every_job_is_selectable_by_some_component(self) -> None:
        unreachable = sorted(_selection_keys() - _reachable_selectors())

        self.assertEqual(
            unreachable,
            [],
            "test-matrix job(s) no component can select, so they run only under "
            f"the '*' wildcard: {unreachable}. Give each one a consumer-graph "
            "component by adding it to _CI_TEST_SELECTOR_ALIASES (or a "
            "test_include in test_policies.toml) in test_tools/.",
        )

    def test_selector_aliases_name_real_jobs(self) -> None:
        # The converse guard: an alias pointing at a renamed or deleted job
        # silently selects nothing, which no other test would notice.
        alias_targets = {
            target
            for targets in test_deps._CI_TEST_SELECTOR_ALIASES.values()
            for target in targets
        }
        dangling = sorted(alias_targets - _selection_keys() - _ALWAYS_SCHEDULED)

        self.assertEqual(
            dangling,
            [],
            f"_CI_TEST_SELECTOR_ALIASES names job(s) not in the test matrix: "
            f"{dangling}",
        )


if __name__ == "__main__":
    unittest.main()
