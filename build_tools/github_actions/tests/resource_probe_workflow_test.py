# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression tests for opt-in resource-probe workflow wiring."""

import unittest

from workflow_utils import WORKFLOWS_DIR, load_workflow


PROBE_TARGETS_INPUT = "resource_probe_targets"
PROBE_INTERVAL_INPUT = "resource_probe_interval_seconds"
PROBE_PHASES_INPUT = "resource_probe_phases"
PROBE_DETAIL_INPUT = "resource_probe_detail_profile"


def _trigger_inputs(workflow: dict, trigger: str) -> dict:
    # PyYAML 1.1 parses an unquoted `on` key as True.
    on_block = workflow.get("on") or workflow.get(True)
    return on_block[trigger]["inputs"]


class ResourceProbeWorkflowTest(unittest.TestCase):
    def test_manual_release_entry_stays_within_github_input_limit(self):
        workflow = load_workflow(WORKFLOWS_DIR / "multi_arch_release.yml")
        inputs = _trigger_inputs(workflow, "workflow_dispatch")
        self.assertLessEqual(len(inputs), 25)

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
                # The top-level manual entry uses the reusable workflow's safe
                # five-second fallback instead of exposing another input.
                if (filename, trigger) != (
                    "multi_arch_release.yml",
                    "workflow_dispatch",
                ):
                    self.assertEqual(inputs[PROBE_INTERVAL_INPUT]["default"], 5)

        artifact_workflow = load_workflow(
            WORKFLOWS_DIR / "multi_arch_build_portable_linux_artifacts.yml"
        )
        artifact_inputs = _trigger_inputs(artifact_workflow, "workflow_call")
        self.assertIs(artifact_inputs["enable_resource_probe"]["default"], False)
        self.assertEqual(artifact_inputs[PROBE_INTERVAL_INPUT]["default"], 5)
        self.assertEqual(artifact_inputs[PROBE_DETAIL_INPUT]["default"], "basic")
        self.assertEqual(
            artifact_inputs[PROBE_PHASES_INPUT]["default"],
            '["fetch-inbound-artifacts","fetch-sources","dependency-install",'
            '"configure","build-stage",'
            '"build-tests","packaging-tests","artifact-push","log-upload"]',
        )

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

    def test_full_stage_path_uses_explicit_probe_phase_wrapper(self):
        workflow = load_workflow(
            WORKFLOWS_DIR / "multi_arch_build_portable_linux_artifacts.yml"
        )
        job_environment = workflow["jobs"]["build_stage"]["env"]
        self.assertIn(
            "contains(fromJSON(inputs.resource_probe_phases), 'build-stage')",
            job_environment["THEROCK_RESOURCE_SPANS_FILE"],
        )
        expected = {
            "Install python deps": "dependency-install",
            "Fetch inbound artifacts": "fetch-inbound-artifacts",
            "Fetch sources": "fetch-sources",
            "Install stage python deps": "dependency-install",
            "Configure": "configure",
            "Build stage": "build-stage",
            "Run build tests (Comgr, blocking)": "build-tests",
            "Run build tests (emulation, non-blocking)": "build-tests",
            "Test Packaging": "packaging-tests",
            "Push stage artifacts": "artifact-push",
            "Upload stage logs": "log-upload",
        }
        steps = {
            step.get("name"): step for step in workflow["jobs"]["build_stage"]["steps"]
        }
        for name, phase in expected.items():
            with self.subTest(step=name):
                script = steps[name]["run"]
                self.assertIn("build_tools/run_resource_probe_phase.py", script)
                self.assertIn(f"--phase {phase}", script)
                self.assertIn('--selected-phases "${RESOURCE_PROBE_PHASES}"', script)
                self.assertIn(
                    '--detail-profile "${RESOURCE_PROBE_DETAIL_PROFILE}"', script
                )

        self.assertIn(
            '"${RESOURCE_PROBE_OUTPUT_DIR}/bootstrap"',
            steps["Install python deps"]["run"],
        )
        self.assertIn(
            '"${RESOURCE_PROBE_OUTPUT_DIR}/stage"',
            steps["Install stage python deps"]["run"],
        )

        self.assertIn(
            "RESOURCE_PROBE_POST_UPLOAD_DIR",
            steps["Upload stage logs"]["run"],
        )
        telemetry_upload = steps["Upload resource telemetry"]
        self.assertIn("always()", str(telemetry_upload["if"]))
        self.assertIn("inputs.enable_resource_probe", str(telemetry_upload["if"]))

    def test_manual_canary_is_paired_default_off_and_non_publishing(self):
        workflow = load_workflow(WORKFLOWS_DIR / "resource_probe_canary.yml")
        inputs = _trigger_inputs(workflow, "workflow_dispatch")
        self.assertGreaterEqual(inputs["pair_count"]["default"], 3)
        self.assertTrue(inputs["source_run_id"]["required"])

        serialized = str(workflow)
        for selector in (
            "compiler-runtime",
            "comm-libs",
            "math-libs:gfx110X-all",
            "math-libs:gfx94X-dcgpu",
        ):
            self.assertIn(selector, serialized)
        self.assertIn("baseline-probe", serialized)
        self.assertIn("probe-baseline", serialized)
        self.assertIn("artifact_equivalence_manifest.py", serialized)
        self.assertIn("fetch-sources", serialized)
        self.assertIn("artifact-push-local-equivalent", serialized)
        self.assertIn("log-upload-local-equivalent", serialized)
        self.assertIn("--detail-profile diagnostic", serialized)
        self.assertNotIn("artifact_manager.py push", serialized)
        self.assertNotIn("publish", serialized.lower())


if __name__ == "__main__":
    unittest.main()
