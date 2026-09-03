# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests the stage-reuse wiring that makes pull request CI skip build stages.

Both settings under test are workflow expressions with no coverage anywhere
else: nothing at parse or actionlint time notices if they are edited back to
the defaults, and the only symptom would be pull requests quietly rebuilding
stages they do not affect again.

* `multi_arch_ci.yml` selects `reuse-stage` for pull requests. Pushes stay on
  `dry-run`, because a push that reused stages would not produce the artifacts
  a later pull request reuses.
* `setup_multi_arch.yml` establishes commit ancestry from the pull request base
  commit. A pull request checks out a merge commit that is not on the baseline
  branch, so `validate_commit_compatibility` cannot place it in the branch
  history and rejects every candidate baseline as `unknown`.
"""

import unittest

from workflow_utils import WORKFLOWS_DIR, get_workflow_job, load_workflow

CI_WORKFLOW = "multi_arch_ci.yml"
SETUP_WORKFLOW = "setup_multi_arch.yml"


class StageReuseModeTest(unittest.TestCase):
    """Verifies the stage_reuse_mode passed from multi_arch_ci.yml to setup."""

    def setUp(self):
        workflow = load_workflow(WORKFLOWS_DIR / CI_WORKFLOW)
        self.expression = get_workflow_job(workflow, "setup")["with"][
            "stage_reuse_mode"
        ]

    def test_pull_requests_select_reuse_stage(self):
        self.assertIn("pull_request", self.expression)
        self.assertIn("reuse-stage", self.expression)

    def test_pushes_keep_building_every_stage(self):
        self.assertIn("dry-run", self.expression)

    def test_workflow_dispatch_input_still_wins(self):
        self.assertTrue(
            self.expression.strip().startswith("${{ inputs.stage_reuse_mode ||"),
            f"dispatch input no longer overrides the default: {self.expression!r}",
        )


class StageReuseCurrentShaTest(unittest.TestCase):
    """Verifies ancestry uses a commit that is on the baseline branch."""

    def setUp(self):
        workflow = load_workflow(WORKFLOWS_DIR / SETUP_WORKFLOW)
        steps = get_workflow_job(workflow, "setup")["steps"]
        configure = next(step for step in steps if step.get("id") == "configure")
        self.expression = configure["env"]["STAGE_REUSE_CURRENT_SHA"]

    def test_pull_requests_use_the_base_commit(self):
        self.assertIn("github.event.pull_request.base.sha", self.expression)

    def test_other_events_use_the_checkout_commit(self):
        self.assertIn("steps.checkout.outputs.commit", self.expression)


if __name__ == "__main__":
    unittest.main()
