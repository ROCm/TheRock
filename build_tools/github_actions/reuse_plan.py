#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Reuse-plan helpers for targeted CI dry-run reporting."""

from __future__ import annotations

from dataclasses import dataclass

from baseline_runs import BaselineRun
from github_actions.stage_impact import StageImpactResult


@dataclass(frozen=True)
class ReusePlan:
    """Stable rebuild-vs-reuse plan for a selected baseline."""

    baseline_run_id: str
    baseline_source_ref: object
    stage_impact: StageImpactResult
    rebuild_stages: tuple[str, ...]
    reuse_stages: tuple[str, ...]
    full_rebuild_required: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline_run_id": self.baseline_run_id,
            "baseline_source_ref": self.baseline_source_ref.to_dict(),
            "stage_impact": self.stage_impact.to_dict(),
            "rebuild_stages": self.rebuild_stages,
            "reuse_stages": self.reuse_stages,
            "full_rebuild_required": self.full_rebuild_required,
        }


def create_reuse_plan(
    *,
    stage_impact: StageImpactResult,
    baseline: BaselineRun,
) -> ReusePlan:
    """Combine stage impact and baseline selection into one reuse plan."""
    return ReusePlan(
        baseline_run_id=baseline.run_id,
        baseline_source_ref=baseline.source_ref,
        stage_impact=stage_impact,
        rebuild_stages=stage_impact.rebuild_stages,
        reuse_stages=stage_impact.copy_stages,
        full_rebuild_required=stage_impact.full_rebuild_required,
    )
