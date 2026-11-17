# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Guard that every test-matrix job is selectable by some component change.

A job whose selection key no component resolves to is not "rarely run" -- it is
run *only* when selection falls back to ``PROJECTS_TO_TEST=*``. That fallback
hides the gap today because a submodule bump passes no changed projects and
selects everything. As selection is scoped, any such job silently stops running
instead of failing loudly, so this pins reachability at the point the job is
declared.

The fix for an unreachable job is normally a ``base_project`` naming the
component it exercises (see ``selection_key`` in fetch_test_configurations), or
a ``_CI_TEST_SELECTOR_ALIASES`` entry when one component fans out to several
differently-named jobs.
"""

from pathlib import Path
import os
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
sys.path.insert(0, os.fspath(REPO_ROOT / "test_tools"))

import fetch_test_configurations
import determine_rocm_test_dependencies as test_deps

# "sanity" is matched by name in fetch_test_configurations rather than through a
# selector, so it always runs and is legitimately unreachable from the graph.
ALWAYS_RUN_JOBS = {"sanity"}


def _reachable_selectors() -> set[str]:
    """Union of CI selectors produced by changing each component on its own."""
    graph = test_deps._load_consumer_graph()
    reachable: set[str] = set()
    for component in graph:
        reachable |= test_deps._to_ci_test_selectors(
            test_deps.get_subprojects_to_test([component])
        )
    return reachable


class TestSelectorReachabilityTest(unittest.TestCase):
    def test_every_job_is_reachable_from_some_component(self) -> None:
        test_matrix = fetch_test_configurations.test_matrix
        reachable = _reachable_selectors()

        unreachable = sorted(
            key
            for key, entry in test_matrix.items()
            if key not in ALWAYS_RUN_JOBS
            and entry.get("base_project", key) not in reachable
        )

        self.assertEqual(
            unreachable,
            [],
            "test-matrix jobs are unreachable from every component and would run "
            f"only under PROJECTS_TO_TEST=*: {unreachable}. Add a 'base_project' "
            "naming the component each exercises, or a _CI_TEST_SELECTOR_ALIASES "
            "entry in test_tools/determine_rocm_test_dependencies.py.",
        )

    def test_base_project_targets_are_selectable(self) -> None:
        """A base_project must name something selection can actually produce.

        A typo'd or stale base_project silently makes the job *less* reachable
        than leaving it off, so check the targets resolve rather than only
        checking the jobs as a set.
        """
        test_matrix = fetch_test_configurations.test_matrix
        reachable = _reachable_selectors()

        bad = sorted(
            {
                entry["base_project"]
                for entry in test_matrix.values()
                if "base_project" in entry and entry["base_project"] not in reachable
            }
        )

        self.assertEqual(
            bad, [], f"base_project values no component selection produces: {bad}"
        )

    def test_reachability_guard_detects_an_unreachable_job(self) -> None:
        """The guard must fail on a job nothing selects, not just pass today.

        Without this, the guard would keep passing if the reachability
        computation were ever broken into vacuity.
        """
        test_matrix = fetch_test_configurations.test_matrix
        probe = "unreachable-probe"
        self.assertNotIn(probe, test_matrix)

        test_matrix[probe] = {"job_name": probe, "platform": ["linux"]}
        try:
            with self.assertRaises(AssertionError):
                self.test_every_job_is_reachable_from_some_component()
        finally:
            del test_matrix[probe]

        # The suite must be left exactly as found, or later tests see the probe.
        self.test_every_job_is_reachable_from_some_component()


if __name__ == "__main__":
    unittest.main()
