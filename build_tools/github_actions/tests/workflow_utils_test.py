# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the shared workflow YAML helpers in workflow_utils.

These feed small in-memory workflow dicts to each helper so the parsing quirks
are locked down in isolation, independent of the real .github/workflows/ tree.
The most important quirk: PyYAML parses the unquoted YAML key `on:` as the
Python boolean `True`, so the helpers look up both `"on"` and `True`.
"""

import unittest

from workflow_utils import (
    WorkflowEdge,
    get_choice_options,
    get_matrix_references,
    get_required_workflow_dispatch_inputs,
    get_transitive_workflow_graph,
    get_workflow_call_inputs,
    get_workflow_dispatch_inputs,
    get_workflow_edges,
    get_workflow_job,
)


class GetWorkflowDispatchInputsTest(unittest.TestCase):
    def test_reads_dispatch_inputs_under_boolean_on_key(self):
        # PyYAML parses `on:` as the boolean True.
        workflow = {True: {"workflow_dispatch": {"inputs": {"a": {}, "b": {}}}}}
        self.assertEqual(get_workflow_dispatch_inputs(workflow), {"a", "b"})

    def test_reads_dispatch_inputs_under_string_on_key(self):
        workflow = {"on": {"workflow_dispatch": {"inputs": {"a": {}}}}}
        self.assertEqual(get_workflow_dispatch_inputs(workflow), {"a"})

    def test_missing_dispatch_block_returns_empty_set(self):
        self.assertEqual(get_workflow_dispatch_inputs({"on": {}}), set())
        self.assertEqual(get_workflow_dispatch_inputs({}), set())

    def test_dispatch_without_inputs_returns_empty_set(self):
        workflow = {"on": {"workflow_dispatch": None}}
        self.assertEqual(get_workflow_dispatch_inputs(workflow), set())


class GetRequiredWorkflowDispatchInputsTest(unittest.TestCase):
    def test_required_without_default_only(self):
        workflow = {
            "on": {
                "workflow_dispatch": {
                    "inputs": {
                        "needs_value": {"required": True},
                        "has_default": {"required": True, "default": "dev"},
                        "optional": {"required": False},
                        "bare": {},
                    }
                }
            }
        }
        self.assertEqual(
            get_required_workflow_dispatch_inputs(workflow), {"needs_value"}
        )


class GetWorkflowCallInputsTest(unittest.TestCase):
    def test_reads_call_inputs_under_boolean_on_key(self):
        workflow = {
            True: {"workflow_call": {"inputs": {"build_config": {}, "ref": {}}}}
        }
        self.assertEqual(get_workflow_call_inputs(workflow), {"build_config", "ref"})

    def test_missing_call_block_returns_empty_set(self):
        self.assertEqual(get_workflow_call_inputs({"on": {}}), set())

    def test_call_without_inputs_returns_empty_set(self):
        workflow = {"on": {"workflow_call": {}}}
        self.assertEqual(get_workflow_call_inputs(workflow), set())


class GetChoiceOptionsTest(unittest.TestCase):
    def _workflow(self, input_def):
        return {"on": {"workflow_dispatch": {"inputs": {"family": input_def}}}}

    def test_returns_options_for_choice_input(self):
        workflow = self._workflow(
            {"type": "choice", "options": ["gfx94X-dcgpu", "gfx110X-all"]}
        )
        self.assertEqual(
            get_choice_options(workflow, "family"), ["gfx94X-dcgpu", "gfx110X-all"]
        )

    def test_non_choice_input_returns_none(self):
        workflow = self._workflow({"type": "string"})
        self.assertIsNone(get_choice_options(workflow, "family"))

    def test_missing_input_returns_none(self):
        workflow = self._workflow({"type": "choice", "options": ["a"]})
        self.assertIsNone(get_choice_options(workflow, "other"))


class GetMatrixReferencesTest(unittest.TestCase):
    def test_extracts_top_level_matrix_keys_from_nested_value(self):
        value = {
            "python_version": "${{ matrix.python_version }}",
            "url": "${{ format('{0}', matrix.family_info.amdgpu_family) }}",
            "list": ["${{ matrix.test_runs_on }}"],
        }
        self.assertEqual(
            get_matrix_references(value),
            {"python_version", "family_info", "test_runs_on"},
        )

    def test_no_references_returns_empty_set(self):
        self.assertEqual(get_matrix_references("static"), set())


class GetWorkflowEdgesTest(unittest.TestCase):
    def test_reusable_edge_captures_with_keys(self):
        workflow = {
            "jobs": {
                "setup": {
                    "uses": "./.github/workflows/setup.yml",
                    "with": {"a": 1, "quartz_tracking_id": "x"},
                }
            }
        }
        edges = get_workflow_edges(workflow, "root.yml")
        self.assertEqual(len(edges), 1)
        edge = edges[0]
        self.assertEqual(edge.kind, "reusable")
        self.assertEqual(edge.source, "root.yml")
        self.assertEqual(edge.label, "setup")
        self.assertEqual(edge.target, "setup.yml")
        self.assertEqual(edge.passed_inputs, {"a", "quartz_tracking_id"})

    def test_reusable_edge_without_with_block_has_no_inputs(self):
        workflow = {"jobs": {"j": {"uses": "./.github/workflows/x.yml"}}}
        edges = get_workflow_edges(workflow, "root.yml")
        self.assertEqual(edges[0].passed_inputs, set())

    def test_dispatch_edge_parses_json_inputs(self):
        workflow = {
            "jobs": {
                "trigger": {
                    "steps": [
                        {
                            "name": "kick",
                            "uses": "benc-uk/workflow-dispatch@v1.2.4",
                            "with": {
                                "workflow": "test.yml",
                                "inputs": '{ "a": "1", "quartz_tracking_id": "x" }',
                            },
                        }
                    ]
                }
            }
        }
        edges = get_workflow_edges(workflow, "root.yml")
        self.assertEqual(len(edges), 1)
        edge = edges[0]
        self.assertEqual(edge.kind, "dispatch")
        self.assertEqual(edge.label, "kick")
        self.assertEqual(edge.target, "test.yml")
        self.assertEqual(edge.passed_inputs, {"a", "quartz_tracking_id"})

    def test_dispatch_edge_without_inputs_has_no_inputs(self):
        workflow = {
            "jobs": {
                "trigger": {
                    "steps": [
                        {
                            "uses": "benc-uk/workflow-dispatch@v1.2.4",
                            "with": {"workflow": "test.yml"},
                        }
                    ]
                }
            }
        }
        edges = get_workflow_edges(workflow, "root.yml")
        self.assertEqual(edges[0].passed_inputs, set())
        self.assertEqual(edges[0].label, "(unnamed)")

    def test_ignores_non_local_uses_and_other_steps(self):
        workflow = {
            "jobs": {
                "external": {"uses": "some/action@v1"},
                "regular": {"steps": [{"uses": "actions/checkout@v4"}]},
            }
        }
        self.assertEqual(get_workflow_edges(workflow, "root.yml"), [])

    def test_no_jobs_returns_empty_list(self):
        self.assertEqual(get_workflow_edges({}, "root.yml"), [])
        self.assertEqual(get_workflow_edges({"jobs": None}, "root.yml"), [])


class GetTransitiveWorkflowGraphTest(unittest.TestCase):
    """Integration checks against the real .github/workflows/ tree."""

    def test_walks_release_graph_from_root(self):
        root = "multi_arch_release.yml"
        nodes, edges = get_transitive_workflow_graph([root])

        # Root is present with no incoming edge kinds.
        self.assertIn(root, nodes)
        self.assertEqual(nodes[root], set())

        # The walk reaches a non-trivial set via both edge kinds.
        self.assertGreater(len(nodes), 1)
        kinds = {edge.kind for edge in edges}
        self.assertEqual(kinds, {"reusable", "dispatch"})

        # Every non-root node is reached by at least one known edge kind, and
        # every edge is a WorkflowEdge pointing at a recorded node.
        for filename, reach_kinds in nodes.items():
            if filename == root:
                continue
            self.assertTrue(reach_kinds <= {"reusable", "dispatch"})
            self.assertTrue(reach_kinds)
        for edge in edges:
            self.assertIsInstance(edge, WorkflowEdge)
            self.assertIn(edge.target, nodes)

    def test_missing_root_yields_only_the_root_node(self):
        nodes, edges = get_transitive_workflow_graph(["does_not_exist.yml"])
        self.assertEqual(nodes, {"does_not_exist.yml": set()})
        self.assertEqual(edges, [])


class GetWorkflowJobTest(unittest.TestCase):
    def test_returns_named_job(self):
        workflow = {"jobs": {"build": {"uses": "x"}}}
        self.assertEqual(get_workflow_job(workflow, "build"), {"uses": "x"})

    def test_missing_jobs_block_raises(self):
        with self.assertRaises(KeyError):
            get_workflow_job({}, "build")


if __name__ == "__main__":
    unittest.main()
