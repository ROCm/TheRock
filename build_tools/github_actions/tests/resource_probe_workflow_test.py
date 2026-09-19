# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression tests for opt-in resource-probe workflow wiring."""

import unittest

from workflow_utils import WORKFLOWS_DIR, load_workflow


PROBE_TARGETS_INPUT = "resource_probe_targets"
PROBE_INTERVAL_INPUT = "resource_probe_interval_seconds"


def _trigger_inputs(workflow: dict, trigger: str) -> dict:
    # PyYAML 1.1 parses an unquoted `on` key as True.
    on_block = workflow.get("on") or workflow.get(True)
    return on_block[trigger]["inputs"]


class ResourceProbeWorkflowTest(unittest.TestCase):
    def test_resource_probe_defaults_are_disabled(self):
        expected = {
            ("multi_arch_ci.yml", "workflow_dispatch"),
            ("multi_arch_ci_linux.yml", "workflow_call"),
            ("multi_arch_build_portable_linux.yml", "workflow_call"),
            ("multi_arch_release.yml", "workflow_call"),
            ("multi_arch_release.yml", "workflow_dispatch"),
            ("multi_arch_release_linux.yml", "workflow_call"),
        }

        for filename, trigger in expected:
            with self.subTest(filename=filename, trigger=trigger):
                workflow = load_workflow(WORKFLOWS_DIR / filename)
                inputs = _trigger_inputs(workflow, trigger)
                self.assertEqual(inputs[PROBE_TARGETS_INPUT]["default"], "[]")
                self.assertEqual(inputs[PROBE_INTERVAL_INPUT]["default"], 5)

        artifact_workflow = load_workflow(
            WORKFLOWS_DIR / "multi_arch_build_portable_linux_artifacts.yml"
        )
        artifact_inputs = _trigger_inputs(artifact_workflow, "workflow_call")
        self.assertIs(artifact_inputs["enable_resource_probe"]["default"], False)
        self.assertEqual(artifact_inputs[PROBE_INTERVAL_INPUT]["default"], 5)

    def test_entry_paths_forward_probe_configuration(self):
        edges = (
            ("multi_arch_ci.yml", "linux_build_and_test"),
            ("multi_arch_ci_linux.yml", "build_multi_arch_stages"),
            ("multi_arch_release.yml", "linux_release"),
            ("multi_arch_release_linux.yml", "build_artifacts"),
        )

        for filename, job_name in edges:
            with self.subTest(filename=filename, job=job_name):
                workflow = load_workflow(WORKFLOWS_DIR / filename)
                passed_inputs = workflow["jobs"][job_name]["with"]
                self.assertIn(PROBE_TARGETS_INPUT, passed_inputs)
                self.assertIn(PROBE_INTERVAL_INPUT, passed_inputs)

    def test_stage_selection_uses_json_array_membership(self):
        workflow = load_workflow(WORKFLOWS_DIR / "multi_arch_build_portable_linux.yml")
        expected_jobs = {
            "compiler-runtime",
            "emulation",
            "runtime-tests",
            "math-libs",
            "comm-libs",
            "storage-libs",
            "debug-tools",
            "dctools-core",
            "profiler-apps",
            "cv-libs",
            "media-libs",
        }

        selected_jobs = set()
        for job_name, job in workflow["jobs"].items():
            if not str(job.get("uses", "")).endswith(
                "multi_arch_build_portable_linux_artifacts.yml"
            ):
                continue
            selected_jobs.add(job_name)
            passed_inputs = job["with"]
            selector = passed_inputs["enable_resource_probe"]
            self.assertIn(
                "contains(fromJSON(inputs.resource_probe_targets || '[]'),", selector
            )
            self.assertEqual(
                passed_inputs[PROBE_INTERVAL_INPUT],
                "${{ inputs.resource_probe_interval_seconds }}",
            )

        self.assertEqual(selected_jobs, expected_jobs)
        math_selector = workflow["jobs"]["math-libs"]["with"]["enable_resource_probe"]
        self.assertIn("format('math-libs:{0}'", math_selector)

    def test_build_step_uses_optional_probe_prefix(self):
        workflow = load_workflow(
            WORKFLOWS_DIR / "multi_arch_build_portable_linux_artifacts.yml"
        )
        build_step = next(
            step
            for step in workflow["jobs"]["build_stage"]["steps"]
            if step.get("name") == "Build stage"
        )
        script = build_step["run"]

        self.assertIn("PROBE=()", script)
        self.assertIn('if [[ "${ENABLE_RESOURCE_PROBE}" == "true" ]]', script)
        self.assertIn('"${PROBE[@]}" cmake --build', script)
        self.assertIn('--output-dir "${BUILD_DIR}/logs/resource-probe"', script)


if __name__ == "__main__":
    unittest.main()
