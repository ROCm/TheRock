# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Keep emulation-only builds independent of compiler-runtime scheduling."""

import unittest

from workflow_utils import WORKFLOWS_DIR, load_workflow


class EmulationStageTest(unittest.TestCase):
    def test_emulation_has_no_upstream_jobs(self):
        workflow = load_workflow(WORKFLOWS_DIR / "multi_arch_build_portable_linux.yml")
        job = workflow["jobs"]["emulation"]
        self.assertFalse(job.get("needs"))
        self.assertEqual(job["with"]["stage_name"], "emulation")

    def test_comm_libs_still_waits_for_emulation(self):
        workflow = load_workflow(WORKFLOWS_DIR / "multi_arch_build_portable_linux.yml")
        self.assertIn("emulation", workflow["jobs"]["comm-libs"]["needs"])

    def test_stage_build_does_not_build_all_artifacts(self):
        workflow = load_workflow(
            WORKFLOWS_DIR / "multi_arch_build_portable_linux_artifacts.yml"
        )
        step = next(
            step
            for step in workflow["jobs"]["build_stage"]["steps"]
            if step.get("name") == "Build emulation stage"
        )
        self.assertIn("--target stage-${STAGE_NAME}", step["run"])
        self.assertNotIn("therock-artifacts", step["run"])

    def test_other_stages_keep_full_artifact_validation(self):
        workflow = load_workflow(
            WORKFLOWS_DIR / "multi_arch_build_portable_linux_artifacts.yml"
        )
        steps = {s["name"]: s for s in workflow["jobs"]["build_stage"]["steps"]}
        self.assertIn("therock-artifacts", steps["Build stage"]["run"])
        self.assertIn("!= 'emulation'", steps["Build stage"]["if"])
        self.assertIn("!= 'emulation'", steps["Test Packaging"]["if"])
        packaging = steps["Test emulation packaging"]
        self.assertIn("== 'emulation'", packaging["if"])
        self.assertIn("librocjitsu", packaging["run"])
        self.assertIn("libhsa_hotswap_rocjitsu", packaging["run"])
        self.assertIn("--no-tests=error", packaging["run"])
