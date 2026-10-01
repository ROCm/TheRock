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
