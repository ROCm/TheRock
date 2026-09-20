# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import unittest

from workflow_utils import WORKFLOWS_DIR, load_workflow


def trigger_inputs(workflow, trigger):
    on_block = workflow.get("on") or workflow.get(True)
    return on_block[trigger]["inputs"]


class WindowsResourceProbeWorkflowTest(unittest.TestCase):
    def test_public_inputs_default_off(self):
        for filename, trigger in (
            ("multi_arch_release_windows.yml", "workflow_call"),
            ("multi_arch_build_windows.yml", "workflow_call"),
            ("build_windows_python_packages.yml", "workflow_call"),
        ):
            with self.subTest(filename=filename):
                inputs = trigger_inputs(
                    load_workflow(WORKFLOWS_DIR / filename), trigger
                )
                self.assertEqual(
                    inputs["windows_resource_probe_targets"]["default"], "[]"
                )
                self.assertEqual(
                    inputs["windows_resource_probe_interval_seconds"]["default"], 5
                )

        dispatch_inputs = trigger_inputs(
            load_workflow(WORKFLOWS_DIR / "build_windows_python_packages.yml"),
            "workflow_dispatch",
        )
        self.assertNotIn("windows_resource_probe_targets", dispatch_inputs)
        self.assertNotIn("windows_resource_probe_interval_seconds", dispatch_inputs)
        self.assertNotIn("resource_probe_detail_profile", dispatch_inputs)

    def test_windows_stage_selectors_are_source_declared(self):
        workflow = load_workflow(WORKFLOWS_DIR / "multi_arch_build_windows.yml")
        expected = {
            "compiler-runtime",
            "runtime-tests",
            "math-libs",
            "debug-tools",
            "media-libs",
        }
        selected = set()
        for name, job in workflow["jobs"].items():
            if not str(job.get("uses", "")).endswith(
                "multi_arch_build_windows_artifacts.yml"
            ):
                continue
            selected.add(name)
            self.assertIn("enable_resource_probe", job["with"])
            self.assertEqual(
                job["with"]["resource_probe_detail_profile"],
                "${{ inputs.resource_probe_detail_profile }}",
            )
        self.assertEqual(selected, expected)
        math = workflow["jobs"]["math-libs"]["with"]["enable_resource_probe"]
        self.assertIn("'math-libs:*'", math)
        self.assertIn("format('math-libs:{0}'", math)

    def test_artifact_workflow_wraps_all_native_phases(self):
        workflow = load_workflow(
            WORKFLOWS_DIR / "multi_arch_build_windows_artifacts.yml"
        )
        job = workflow["jobs"]["build_stage"]
        scripts = "\n".join(str(step.get("run", "")) for step in job["steps"])
        for phase in (
            "windows-fetch-inbound-artifacts",
            "windows-fetch-sources",
            "windows-configure",
            "windows-stage-${STAGE_NAME}",
            "windows-build-tests",
            "windows-packaging-tests",
            "windows-artifact-push",
            "windows-log-upload",
        ):
            self.assertIn(
                f"--phase {phase}" if "${" not in phase else f'--phase "{phase}"',
                scripts,
            )
        self.assertIn("windows_resource_probe.py", scripts)
        upload = next(
            step
            for step in job["steps"]
            if step.get("name") == "Upload resource telemetry"
        )
        self.assertIn("always()", upload["if"])
        self.assertIn("resource_probe_canary_mode", job["env"]["CCACHE_DISABLE"])

    def test_python_package_phases_and_publication_are_wrapped(self):
        packages = load_workflow(WORKFLOWS_DIR / "build_windows_python_packages.yml")
        scripts = "\n".join(
            str(step.get("run", ""))
            for step in packages["jobs"]["build_rocm_wheels"]["steps"]
        )
        for selector in (
            "python-packages:dependencies",
            "python-packages:fetch",
            "python-packages:build",
            "python-packages:upload",
        ):
            self.assertIn(selector, scripts)
        release = load_workflow(WORKFLOWS_DIR / "multi_arch_release_windows.yml")
        publish = release["jobs"]["publish_to_release_buckets"]
        publish_script = next(
            step["run"]
            for step in publish["steps"]
            if step.get("name") == "Copy to release buckets"
        )
        self.assertIn("publish-release-buckets", publish_script)
        self.assertIn("${PUBLISH_DRY_RUN}", publish_script)
        self.assertIn("--dry-run", publish["env"]["PUBLISH_DRY_RUN"])

    def test_manual_canary_is_pinned_and_safe(self):
        workflow = load_workflow(WORKFLOWS_DIR / "windows_resource_probe_canary.yml")
        inputs = trigger_inputs(workflow, "workflow_dispatch")
        self.assertEqual(inputs["variant"]["options"], ["baseline", "probed"])
        self.assertEqual(inputs["pair_index"]["options"], ["1", "2", "3"])
        job = workflow["jobs"]["release_canary"]
        self.assertEqual(job["with"]["ref"], "${{ github.sha }}")
        self.assertIs(job["with"]["windows_resource_probe_canary_mode"], True)
        self.assertEqual(job["with"]["linux_amdgpu_families"], "none")
        self.assertIs(job["with"]["build_pytorch"], False)
        called_inputs = trigger_inputs(
            load_workflow(WORKFLOWS_DIR / "multi_arch_release.yml"),
            "workflow_call",
        )
        self.assertLessEqual(set(job["with"]), set(called_inputs))
        targets = job["with"]["windows_resource_probe_targets"]
        for selector in (
            "compiler-runtime",
            "runtime-tests",
            "math-libs:*",
            "debug-tools",
            "media-libs",
            "release-tarballs:windows",
            "python-packages:dependencies",
            "python-packages:fetch",
            "python-packages:build",
            "python-packages:upload",
            "release-publish:windows",
        ):
            self.assertIn(selector, targets)

        release = load_workflow(WORKFLOWS_DIR / "multi_arch_release_windows.yml")
        runner = release["jobs"]["build_artifacts"]["with"]["build_runs_on"]
        self.assertIn("azure-windows-scale-rocm", runner)

        branch_entry = load_workflow(WORKFLOWS_DIR / "multi_arch_release.yml")
        dispatch_inputs = trigger_inputs(branch_entry, "workflow_dispatch")
        for name in (
            "windows_resource_probe_targets",
            "windows_resource_probe_canary_mode",
            "windows_resource_probe_canary_pair",
            "windows_resource_probe_canary_variant",
        ):
            self.assertIn(name, dispatch_inputs)

    def test_pair_comparison_enforces_counterbalanced_order(self):
        workflow = load_workflow(WORKFLOWS_DIR / "windows_resource_probe_compare.yml")
        scripts = "\n".join(
            str(step.get("run", "")) for step in workflow["jobs"]["compare"]["steps"]
        )
        self.assertIn('"${baseline_pair}" == "2"', scripts)
        self.assertIn("pair 2 must use BA order", scripts)
        self.assertIn("pairs 1 and 3 must use AB order", scripts)
        self.assertIn("compare_artifact_equivalence_manifests.py", scripts)
        self.assertIn("run_conclusion", scripts)
        self.assertIn("canary label validation failed", scripts)
        self.assertIn("windows-canary-manifest-*-${baseline_attempt}", scripts)
        self.assertIn("EXPECTED_HEAD_SHA", scripts)


if __name__ == "__main__":
    unittest.main()
