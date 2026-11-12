# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the pull request label and test inputs of setup_multi_arch.yml.

Callers pass these inputs from their own per-variant workflows. The tests check
that setup declares them with the types and defaults the callers rely on.
"""

import unittest

from workflow_utils import WORKFLOWS_DIR, load_workflow

SETUP_WORKFLOW = "setup_multi_arch.yml"


def _get_triggers(workflow: dict) -> dict:
    # PyYAML parses the unquoted YAML key `on:` as boolean True.
    return workflow.get("on") or workflow.get(True)


class SetupMultiArchInputsTest(unittest.TestCase):
    def setUp(self):
        setup = load_workflow(WORKFLOWS_DIR / SETUP_WORKFLOW)
        self.inputs = _get_triggers(setup)["workflow_call"]["inputs"]

    def test_label_inputs_are_strings_defaulting_to_empty(self):
        for name in ("pull_request_opt_in_label", "pull_request_opt_out_label"):
            with self.subTest(name=name):
                self.assertEqual(self.inputs[name]["type"], "string")
                self.assertEqual(self.inputs[name]["default"], "")

    def test_pull_request_tests_is_boolean_defaulting_to_false(self):
        declared = self.inputs["pull_request_tests"]
        self.assertEqual(declared["type"], "boolean")
        self.assertIs(declared["default"], False)

    def test_inputs_are_optional(self):
        for name in (
            "pull_request_opt_in_label",
            "pull_request_opt_out_label",
            "pull_request_tests",
        ):
            with self.subTest(name=name):
                self.assertFalse(self.inputs[name].get("required", False))


if __name__ == "__main__":
    unittest.main()
