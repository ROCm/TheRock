# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contracts for opt-in telemetry in release packaging workflows."""

import unittest

from workflow_utils import WORKFLOWS_DIR, load_workflow


INPUT_DEFAULTS = {
    "enable_resource_probe": False,
    "resource_probe_interval_seconds": 5,
    "resource_probe_detail_profile": "basic",
}


def _trigger_inputs(workflow: dict, trigger: str) -> dict:
    # PyYAML 1.1 parses an unquoted `on` key as True.
    on_block = workflow.get("on") or workflow.get(True)
    return on_block[trigger]["inputs"]


def _steps(workflow: dict, job: str) -> dict:
    return {step.get("name"): step for step in workflow["jobs"][job]["steps"]}


class PackagingResourceProbeWorkflowTest(unittest.TestCase):
    def test_public_inputs_are_opt_in(self):
        workflows = {
            "multi_arch_build_tarballs.yml": ("workflow_call",),
            "build_portable_linux_python_packages.yml": ("workflow_call",),
            "multi_arch_build_native_linux_packages.yml": ("workflow_call",),
        }
        for filename, triggers in workflows.items():
            workflow = load_workflow(WORKFLOWS_DIR / filename)
            for trigger in triggers:
                with self.subTest(filename=filename, trigger=trigger):
                    inputs = _trigger_inputs(workflow, trigger)
                    for name, expected in INPUT_DEFAULTS.items():
                        self.assertEqual(inputs[name]["default"], expected)
                    self.assertIn("resource_probe_phases", inputs)

    def test_existing_standalone_dispatches_keep_probe_disabled(self):
        for filename, job_name in (
            ("multi_arch_build_tarballs.yml", "build_tarballs"),
            ("build_portable_linux_python_packages.yml", "build_rocm_wheels"),
        ):
            with self.subTest(filename=filename):
                workflow = load_workflow(WORKFLOWS_DIR / filename)
                dispatch_inputs = _trigger_inputs(workflow, "workflow_dispatch")
                self.assertNotIn("enable_resource_probe", dispatch_inputs)
                environment = workflow["jobs"][job_name]["env"]
                self.assertIn("|| false", environment["ENABLE_RESOURCE_PROBE"])
                self.assertIn("|| 5", environment["RESOURCE_PROBE_INTERVAL_SECONDS"])
                self.assertIn(
                    "|| 'basic'", environment["RESOURCE_PROBE_DETAIL_PROFILE"]
                )

    def test_tarball_phases_are_explicit_and_evidence_upload_is_always_run(self):
        workflow = load_workflow(WORKFLOWS_DIR / "multi_arch_build_tarballs.yml")
        steps = _steps(workflow, "build_tarballs")
        expected = {
            "Install Python requirements": "dependency-install",
            "Build tarballs": "package-build",
            "Upload tarballs": "package-upload",
        }
        self._assert_wrapped_phases(steps, expected)
        self._assert_always_uploads_evidence(steps)

    def test_python_package_phases_are_explicit_and_evidence_upload_is_always_run(self):
        workflow = load_workflow(
            WORKFLOWS_DIR / "build_portable_linux_python_packages.yml"
        )
        steps = _steps(workflow, "build_rocm_wheels")
        expected = {
            "Install Python requirements": "dependency-install",
            "Fetch artifacts": "package-fetch",
            "Build Python packages": "package-build",
            "Upload Python packages": "package-upload",
        }
        self._assert_wrapped_phases(steps, expected)
        self._assert_always_uploads_evidence(steps)

    def test_native_package_phases_are_explicit_and_evidence_upload_is_always_run(self):
        workflow = load_workflow(
            WORKFLOWS_DIR / "multi_arch_build_native_linux_packages.yml"
        )
        steps = _steps(workflow, "build_native_packages")
        expected = {
            "Install Python requirements": "dependency-install",
            "Install System requirements": "dependency-install",
            "Fetch Artifacts for all GPU families": "package-fetch",
            "Build Packages": "package-build",
            "Simulated install Test": "package-test",
            "Build Package repository": "package-repository-build",
            "Upload Package repo to S3": "package-upload",
        }
        self._assert_wrapped_phases(steps, expected)
        self.assertIn(
            '"${RESOURCE_PROBE_OUTPUT_DIR}/python"',
            steps["Install Python requirements"]["run"],
        )
        self.assertIn(
            '"${RESOURCE_PROBE_OUTPUT_DIR}/system"',
            steps["Install System requirements"]["run"],
        )
        self._assert_always_uploads_evidence(steps)

    def _assert_wrapped_phases(self, steps: dict, expected: dict) -> None:
        for step_name, phase in expected.items():
            with self.subTest(step=step_name):
                script = steps[step_name]["run"]
                self.assertIn("build_tools/run_resource_probe_phase.py", script)
                self.assertIn(f"--phase {phase}", script)
                self.assertIn('--selected-phases "${RESOURCE_PROBE_PHASES}"', script)
                self.assertIn(
                    '--detail-profile "${RESOURCE_PROBE_DETAIL_PROFILE}"', script
                )
                self.assertIn('--storage-path "${GITHUB_WORKSPACE}"', script)

    def _assert_always_uploads_evidence(self, steps: dict) -> None:
        upload = steps["Upload resource telemetry"]
        self.assertIn("always()", str(upload["if"]))
        self.assertIn("inputs.enable_resource_probe", str(upload["if"]))
        self.assertEqual(upload["with"]["if-no-files-found"], "warn")
        self.assertEqual(upload["with"]["retention-days"], 14)


if __name__ == "__main__":
    unittest.main()
