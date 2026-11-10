# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the wiring of multi_arch_ci_host_asan.yml.

The host-asan workflow selects its behavior through inputs passed to
setup_multi_arch.yml. These tests check those inputs, so the label policy is
not only covered by the unit tests of configure_multi_arch_ci.py.
"""

import unittest

from workflow_utils import WORKFLOWS_DIR, get_workflow_job, load_workflow

HOST_ASAN_WORKFLOW = "multi_arch_ci_host_asan.yml"
SETUP_WORKFLOW = "setup_multi_arch.yml"


def _get_triggers(workflow: dict) -> dict:
    # PyYAML parses the unquoted YAML key `on:` as boolean True.
    return workflow.get("on") or workflow.get(True)


class HostAsanWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.workflow = load_workflow(WORKFLOWS_DIR / HOST_ASAN_WORKFLOW)
        self.setup_with = get_workflow_job(self.workflow, "setup")["with"]

    def test_setup_job_calls_setup_multi_arch(self):
        job = get_workflow_job(self.workflow, "setup")
        self.assertEqual(job["uses"], f"./.github/workflows/{SETUP_WORKFLOW}")

    def test_setup_passes_host_asan_policy_inputs(self):
        self.assertEqual(self.setup_with["build_variant"], "host-asan")
        self.assertEqual(self.setup_with["pull_request_opt_in_label"], "ci:host-asan")
        self.assertIs(self.setup_with["pull_request_tests"], True)

    def test_linux_families_default_to_gfx90a(self):
        """An empty family input would fall back to the presubmit families.

        On a pull request that builds gfx94X-dcgpu and gfx125X-dcgpu instead of
        gfx90a, so the workflow must supply gfx90a itself.
        """
        self.assertEqual(
            self.setup_with["linux_amdgpu_families"],
            "${{ inputs.linux_amdgpu_families || 'gfx90a' }}",
        )

    def test_does_not_set_opt_out_label(self):
        self.assertNotIn("pull_request_opt_out_label", self.setup_with)

    def test_only_manual_and_reusable_triggers_until_old_policy_is_removed(self):
        """Enabling pull_request/push/schedule needs the asan rewrite removed.

        See the TODO in the workflow. When enabling pull_request, include
        `types: [opened, synchronize, reopened, labeled]`.
        """
        self.assertEqual(
            set(_get_triggers(self.workflow)), {"workflow_dispatch", "workflow_call"}
        )

    def test_setup_inputs_are_declared_by_setup_multi_arch(self):
        setup = load_workflow(WORKFLOWS_DIR / SETUP_WORKFLOW)
        declared = _get_triggers(setup)["workflow_call"]["inputs"]
        undeclared = set(self.setup_with) - set(declared)
        self.assertFalse(undeclared, f"inputs not declared by setup: {undeclared}")

    def test_pull_request_tests_is_boolean_defaulting_to_false(self):
        setup = load_workflow(WORKFLOWS_DIR / SETUP_WORKFLOW)
        declared = _get_triggers(setup)["workflow_call"]["inputs"]["pull_request_tests"]
        self.assertEqual(declared["type"], "boolean")
        self.assertIs(declared["default"], False)


if __name__ == "__main__":
    unittest.main()
