# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests that `quartz_tracking_id` is threaded through the release graph.

The multi-arch release entry point `multi_arch_release.yml` enables Quartz
release-lineage tracking. When on, it derives a `quartz_tracking_id` and must
propagate it to every workflow it triggers so any of them can report lineage
back to Quartz. This propagation is easy to break silently: a new reusable
workflow, a new `uses:` job, or a new `benc-uk/workflow-dispatch` step can
forget to declare or forward the input, and nothing at parse or actionlint time
catches it. See the review discussion at
https://github.com/ROCm/TheRock/pull/7443#discussion_r3807753206.

Walking the call graph from the root (the same way
`configure_ci_path_filters_test.py` walks it from `multi_arch_ci.yml`, but
also following `benc-uk/workflow-dispatch` edges), these tests assert that:

  1. every reachable non-root workflow declares `quartz_tracking_id` on the
     trigger used to reach it (`workflow_call` for reusable edges,
     `workflow_dispatch` for dispatch edges); and
  2. every edge forwards `quartz_tracking_id` to its target.
"""

import unittest

from workflow_utils import (
    WORKFLOWS_DIR,
    get_transitive_workflow_graph,
    get_workflow_call_inputs,
    get_workflow_dispatch_inputs,
    load_workflow,
)

QUARTZ_INPUT = "quartz_tracking_id"
ROOT_WORKFLOW = "multi_arch_release.yml"

_NODES, _EDGES = get_transitive_workflow_graph([ROOT_WORKFLOW])


class QuartzTrackingIdPropagationTest(unittest.TestCase):
    """Verifies quartz_tracking_id reaches every workflow in the release graph."""

    def test_reachable_workflows_declare_quartz_tracking_id(self):
        """Every reachable non-root workflow declares the input on its trigger."""
        errors = []
        for target, kinds in sorted(_NODES.items()):
            if target == ROOT_WORKFLOW:
                continue
            target_path = WORKFLOWS_DIR / target
            if not target_path.exists():
                errors.append(f"'{target}' is referenced but does not exist")
                continue
            workflow = load_workflow(target_path)
            if "reusable" in kinds and QUARTZ_INPUT not in get_workflow_call_inputs(
                workflow
            ):
                errors.append(
                    f"'{target}' is used as a reusable workflow but does not "
                    f"declare '{QUARTZ_INPUT}' under on.workflow_call.inputs"
                )
            if (
                "dispatch" in kinds
                and QUARTZ_INPUT not in get_workflow_dispatch_inputs(workflow)
            ):
                errors.append(
                    f"'{target}' is triggered via workflow-dispatch but does not "
                    f"declare '{QUARTZ_INPUT}' under on.workflow_dispatch.inputs"
                )
        if errors:
            self.fail("\n".join(errors))

    def test_edges_forward_quartz_tracking_id(self):
        """Every reusable and dispatch edge forwards the input to its target."""
        errors = []
        for edge in _EDGES:
            if QUARTZ_INPUT not in edge.passed_inputs:
                errors.append(
                    f"{edge.kind} edge '{edge.source}' -> '{edge.target}' "
                    f"(via '{edge.label}') does not forward '{QUARTZ_INPUT}'"
                )
        if errors:
            self.fail("\n".join(errors))


if __name__ == "__main__":
    unittest.main()
