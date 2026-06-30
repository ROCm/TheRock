#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
import os
import sys
import unittest

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import baseline_runs
import reuse_plan
from github_actions.stage_impact import StageImpactResult


class ReusePlanTest(unittest.TestCase):
    def test_create_reuse_plan(self):
        stage_impact = StageImpactResult(
            changed_inputs=("rocm-libraries",),
            matched_source_sets=("rocm-libraries",),
            impacted_artifact_groups=("math-libs",),
            rebuild_stages=("math-libs",),
            copy_stages=("compiler-runtime", "debug-tools", "media-libs"),
            full_rebuild_required=False,
            reasons=(),
            unmatched_inputs=(),
        )

        source_ref = baseline_runs.WorkflowRunSummary(
            repository="ROCm/TheRock",
            branch="main",
            commit="sha-123",
            workflow="multi_arch_ci.yml",
            run_id="123",
            status="completed",
            conclusion="success",
            timestamp="2026-06-17T20:00:00Z",
            html_url="https://github.com/ROCm/TheRock/actions/runs/123",
        )

        baseline = baseline_runs.BaselineRun(
            source_ref=source_ref,
            platform="linux",
            job_health=baseline_runs.WorkflowJobHealth(
                required_name_substrings=("Build",),
                matched_job_names=("Build",),
                failed_job_names=(),
                missing_name_substrings=(),
            ),
            artifact_availability=baseline_runs.ArtifactAvailability(
                required_artifacts=(baseline_runs.RequiredArtifact("base", "generic"),),
                matched_filenames=("base_lib_generic.tar.zst",),
                missing_artifacts=(),
            ),
        )

        plan = reuse_plan.create_reuse_plan(
            stage_impact=stage_impact,
            baseline=baseline,
        )

        self.assertEqual(plan.baseline_run_id, "123")
        self.assertEqual(plan.rebuild_stages, ("math-libs",))
        self.assertEqual(
            plan.reuse_stages,
            ("compiler-runtime", "debug-tools", "media-libs"),
        )
        self.assertFalse(plan.full_rebuild_required)

    def test_reuse_plan_to_dict(self):
        stage_impact = StageImpactResult(
            changed_inputs=("rocm-libraries",),
            matched_source_sets=("rocm-libraries",),
            impacted_artifact_groups=("math-libs",),
            rebuild_stages=("math-libs",),
            copy_stages=("compiler-runtime",),
            full_rebuild_required=False,
            reasons=(),
            unmatched_inputs=(),
        )

        source_ref = baseline_runs.WorkflowRunSummary(
            repository="ROCm/TheRock",
            branch="main",
            commit="sha-123",
            workflow="multi_arch_ci.yml",
            run_id="123",
            status="completed",
            conclusion="success",
            timestamp="2026-06-17T20:00:00Z",
            html_url="https://github.com/ROCm/TheRock/actions/runs/123",
        )

        baseline = baseline_runs.BaselineRun(
            source_ref=source_ref,
            platform="linux",
            job_health=baseline_runs.WorkflowJobHealth(
                required_name_substrings=("Build",),
                matched_job_names=("Build",),
                failed_job_names=(),
                missing_name_substrings=(),
            ),
            artifact_availability=baseline_runs.ArtifactAvailability(
                required_artifacts=(baseline_runs.RequiredArtifact("base", "generic"),),
                matched_filenames=("base_lib_generic.tar.zst",),
                missing_artifacts=(),
            ),
        )

        plan = reuse_plan.create_reuse_plan(
            stage_impact=stage_impact,
            baseline=baseline,
        )

        payload = plan.to_dict()
        self.assertEqual(
            set(payload.keys()),
            {
                "baseline_run_id",
                "baseline_source_ref",
                "stage_impact",
                "rebuild_stages",
                "reuse_stages",
                "full_rebuild_required",
            },
        )
        self.assertEqual(payload["baseline_run_id"], "123")
        self.assertEqual(payload["baseline_source_ref"]["run_id"], "123")
        self.assertEqual(payload["rebuild_stages"], ("math-libs",))
