#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Configures CI matrix and stage decisions for multi-arch workflows.

This script is a pipeline of data transformations:

    1. Parse Inputs    — read GitHub event context → CIInputs
    2. Check Skip CI   — gate: should we skip CI entirely?
    3. Select Targets  — trigger type + labels → GPU families
    4. Decide Stages   — changed files + topology → rebuild/prebuilt per stage
    5. Expand Matrix   — families × variant → matrix entries
    6. Write Outputs   — JSON → GITHUB_OUTPUT + GITHUB_STEP_SUMMARY

Each step (except 1 and 6) is a pure function of typed dataclasses,
independently testable without environment variables or filesystem access.

Inputs:
    GITHUB_EVENT_NAME   : push, pull_request, schedule, workflow_dispatch
    GITHUB_EVENT_PATH   : JSON file with event payload (inputs, PR labels, etc.)
    GITHUB_REF_NAME     : Branch name
    GITHUB_OUTPUT       : Path to write workflow output variables
    GITHUB_STEP_SUMMARY : Path to write workflow summary
    BUILD_VARIANT       : Build variant (workflow_call input, not in event payload)

Outputs (written to GITHUB_OUTPUT):
    linux_variants      : JSON array of matrix entries
    windows_variants    : JSON array of matrix entries
    linux_test_labels   : JSON array of test label strings
    windows_test_labels : JSON array of test label strings
    enable_build_jobs   : "true" or "false"
    test_type           : "smoke" or "full"
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Dataclasses — the typed interfaces between pipeline steps
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CIInputs:
    """All external inputs to the CI configuration pipeline.

    Constructed once from the GitHub Actions environment. Every downstream
    function takes this (or a subset) as a plain argument — no environment
    access needed.
    """

    event_name: str  # push, pull_request, schedule, workflow_dispatch
    branch_name: str
    base_ref: str  # Git ref for diffing (PR base or HEAD^1)
    build_variant: str  # release, asan, tsan

    # PR labels (from event payload for pull_request events)
    pr_labels: list[str] = field(default_factory=list)

    # Per-platform workflow_dispatch overrides
    linux_amdgpu_families: str = ""
    windows_amdgpu_families: str = ""
    linux_test_labels: str = ""
    windows_test_labels: str = ""

    # Prebuilt configuration (from workflow_dispatch)
    prebuilt_stages: str = ""
    baseline_run_id: str = ""

    @property
    def is_pull_request(self) -> bool:
        return self.event_name == "pull_request"

    @property
    def is_push(self) -> bool:
        return self.event_name == "push"

    @property
    def is_schedule(self) -> bool:
        return self.event_name == "schedule"

    @property
    def is_workflow_dispatch(self) -> bool:
        return self.event_name == "workflow_dispatch"

    @staticmethod
    def from_environ() -> CIInputs:
        """Parse from GitHub Actions environment.

        Reads GITHUB_EVENT_PATH for the event payload and a few standard
        env vars. This is the only function in the pipeline that touches
        external state.
        """
        event_name = os.environ.get("GITHUB_EVENT_NAME", "")
        branch_name = os.environ.get("GITHUB_REF_NAME", "")
        if not branch_name:
            print(
                "[ERROR] GITHUB_REF_NAME is not set. Exiting.",
                file=sys.stderr,
            )
            sys.exit(1)

        # Read the full event payload
        event_path = os.environ.get("GITHUB_EVENT_PATH", "")
        if event_path and Path(event_path).exists():
            with open(event_path) as f:
                event = json.load(f)
        else:
            event = {}

        # Extract fields based on event type
        inputs = event.get("inputs") or {}
        pr_labels: list[str] = []
        base_ref = "HEAD^1"

        if event_name == "pull_request":
            pr_obj = event.get("pull_request", {})
            pr_labels = [label["name"] for label in pr_obj.get("labels", [])]
            # The merge commit's first parent is the PR base
            base_ref = "HEAD^"
        elif event_name == "push":
            base_ref = event.get("before", "HEAD^1")

        # BUILD_VARIANT comes from workflow_call inputs, not the event payload
        build_variant = os.environ.get("BUILD_VARIANT", "release")

        return CIInputs(
            event_name=event_name,
            branch_name=branch_name,
            base_ref=base_ref,
            build_variant=build_variant,
            pr_labels=pr_labels,
            linux_amdgpu_families=inputs.get("linux_amdgpu_families", ""),
            windows_amdgpu_families=inputs.get("windows_amdgpu_families", ""),
            linux_test_labels=inputs.get("linux_test_labels", ""),
            windows_test_labels=inputs.get("windows_test_labels", ""),
            prebuilt_stages=inputs.get("prebuilt_stages", ""),
            baseline_run_id=inputs.get("baseline_run_id", ""),
        )


@dataclass(frozen=True)
class SkipDecision:
    """Whether to skip CI entirely."""

    skip: bool
    reason: str  # e.g. "skip-ci label", "only .md files changed", ""


@dataclass(frozen=True)
class TargetSelection:
    """Which GPU families to build/test, per platform."""

    linux_families: list[str] = field(default_factory=list)
    windows_families: list[str] = field(default_factory=list)
    test_names: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class StageDecision:
    """Decision for a single build stage."""

    action: Literal["rebuild", "prebuilt"]
    reason: str


@dataclass(frozen=True)
class StageDecisions:
    """Per-stage build/prebuilt decisions and test type."""

    decisions: dict[str, StageDecision] = field(default_factory=dict)
    test_type: str = "smoke"
    test_type_reason: str = "default"

    @property
    def prebuilt_stages(self) -> list[str]:
        return [name for name, d in self.decisions.items() if d.action == "prebuilt"]

    @property
    def rebuild_stages(self) -> list[str]:
        return [name for name, d in self.decisions.items() if d.action == "rebuild"]


@dataclass(frozen=True)
class MatrixEntry:
    """One row of the GitHub Actions build matrix."""

    matrix_per_family_json: str  # JSON array of per-family info
    dist_amdgpu_families: str  # Semicolon-separated
    artifact_group: str
    build_variant_label: str
    build_variant_suffix: str
    build_variant_cmake_preset: str
    expect_failure: bool
    build_pytorch: bool

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return {
            "matrix_per_family_json": self.matrix_per_family_json,
            "dist_amdgpu_families": self.dist_amdgpu_families,
            "artifact_group": self.artifact_group,
            "build_variant_label": self.build_variant_label,
            "build_variant_suffix": self.build_variant_suffix,
            "build_variant_cmake_preset": self.build_variant_cmake_preset,
            "expect_failure": self.expect_failure,
            "build_pytorch": self.build_pytorch,
        }


@dataclass(frozen=True)
class CIOutputs:
    """All outputs from the CI configuration pipeline."""

    linux_variants: list[MatrixEntry] = field(default_factory=list)
    windows_variants: list[MatrixEntry] = field(default_factory=list)
    linux_test_labels: list[str] = field(default_factory=list)
    windows_test_labels: list[str] = field(default_factory=list)
    enable_build_jobs: bool = True
    test_type: str = "smoke"
    # Stage decisions (feeds into prebuilt workflow plumbing)
    prebuilt_stages: list[str] = field(default_factory=list)
    rebuild_stages: list[str] = field(default_factory=list)

    @staticmethod
    def skipped(reason: str) -> CIOutputs:
        """Produce empty outputs when CI is skipped."""
        return CIOutputs(enable_build_jobs=False)


# ---------------------------------------------------------------------------
# Step 2: Check Skip CI
# ---------------------------------------------------------------------------


def check_skip_ci(
    inputs: CIInputs,
    changed_files: list[str] | None,
) -> SkipDecision:
    """Determine whether CI should be skipped entirely.

    Returns SkipDecision(skip=True) for:
    - 'skip-ci' PR label
    - Only skippable files changed (docs, .md, etc.)
    - No files changed
    """
    # TODO: Implement — check skip-ci label, call is_ci_run_required()
    return SkipDecision(skip=False, reason="")


# ---------------------------------------------------------------------------
# Step 3: Select Targets
# ---------------------------------------------------------------------------


def select_targets(inputs: CIInputs) -> TargetSelection:
    """Determine GPU families and test names based on trigger type and inputs.

    Handles:
    - workflow_dispatch: parse explicit family/test inputs
    - pull_request: presubmit+postsubmit defaults, PR label opt-ins
    - push: presubmit+postsubmit defaults
    - schedule: all families
    """
    # TODO: Implement — trigger dispatch, label parsing, family validation
    return TargetSelection()


# ---------------------------------------------------------------------------
# Step 4: Decide Stages
# ---------------------------------------------------------------------------


def decide_stages(
    inputs: CIInputs,
    targets: TargetSelection,
    changed_files: list[str] | None,
) -> StageDecisions:
    """Determine per-stage rebuild/prebuilt decisions and test type.

    Currently returns "rebuild all" — source-set-aware logic comes in Phase 4.
    """
    # TODO: Implement — topology parsing, source-set analysis, propagation
    return StageDecisions(test_type="smoke", test_type_reason="default (stub)")


# ---------------------------------------------------------------------------
# Step 5: Expand Matrix
# ---------------------------------------------------------------------------


def expand_matrix(
    families: list[str],
    platform: str,
    build_variant: str,
) -> list[MatrixEntry]:
    """Expand families into multi-arch matrix entries for one platform.

    Groups all families into one entry per build variant (the multi-arch
    format), rather than one entry per family (single-arch format).
    """
    # TODO: Implement — port generate_multi_arch_matrix logic
    return []


# ---------------------------------------------------------------------------
# Step 6: Format and Write Outputs
# ---------------------------------------------------------------------------


def format_summary(outputs: CIOutputs) -> str:
    """Generate human-readable markdown summary. Pure function."""
    # TODO: Implement — structured markdown with families, stages, reasons
    lines = ["## Multi-Arch CI Configuration"]
    lines.append("")
    lines.append(f"* `enable_build_jobs`: {outputs.enable_build_jobs}")
    lines.append(f"* `test_type`: {outputs.test_type}")
    return "\n".join(lines)


def write_outputs(outputs: CIOutputs) -> None:
    """Write results to GITHUB_OUTPUT and GITHUB_STEP_SUMMARY.

    This is the only function with side effects (besides from_environ).
    """
    from github_actions_utils import gha_set_output, gha_append_step_summary

    output_vars = {
        "linux_variants": json.dumps(
            [entry.to_dict() for entry in outputs.linux_variants]
        ),
        "linux_test_labels": json.dumps(outputs.linux_test_labels),
        "windows_variants": json.dumps(
            [entry.to_dict() for entry in outputs.windows_variants]
        ),
        "windows_test_labels": json.dumps(outputs.windows_test_labels),
        "enable_build_jobs": json.dumps(outputs.enable_build_jobs),
        "test_type": outputs.test_type,
    }
    gha_set_output(output_vars)
    gha_append_step_summary(format_summary(outputs))


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


def configure(inputs: CIInputs) -> CIOutputs:
    """Main pipeline. Each step feeds the next.

    This function is the primary entry point for testing — construct a
    CIInputs and assert on the returned CIOutputs.
    """
    # Step 1 already done — inputs is the parsed CIInputs.

    # Step 2: Gate — should we skip CI entirely?
    # For schedule and workflow_dispatch, always proceed.
    changed_files: list[str] | None = None
    if inputs.is_pull_request or inputs.is_push:
        from configure_ci_path_filters import get_git_modified_paths

        changed_files = get_git_modified_paths(inputs.base_ref)

    skip = check_skip_ci(inputs, changed_files)
    if skip.skip:
        print(f"Skipping CI: {skip.reason}")
        return CIOutputs.skipped(skip.reason)

    # Step 3: Select targets
    targets = select_targets(inputs)

    # Step 4: Decide stages (stub: rebuild all)
    stage_decisions = decide_stages(inputs, targets, changed_files)

    # Step 5: Expand matrix
    linux_matrix = expand_matrix(
        targets.linux_families,
        "linux",
        inputs.build_variant,
    )
    windows_matrix = expand_matrix(
        targets.windows_families,
        "windows",
        inputs.build_variant,
    )

    return CIOutputs(
        linux_variants=linux_matrix,
        windows_variants=windows_matrix,
        linux_test_labels=targets.test_names,
        windows_test_labels=targets.test_names,
        enable_build_jobs=True,
        test_type=stage_decisions.test_type,
        prebuilt_stages=stage_decisions.prebuilt_stages,
        rebuild_stages=stage_decisions.rebuild_stages,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    inputs = CIInputs.from_environ()

    print("Multi-arch CI configuration")
    print(f"  event: {inputs.event_name}")
    print(f"  branch: {inputs.branch_name}")
    print(f"  variant: {inputs.build_variant}")
    if inputs.pr_labels:
        print(f"  pr_labels: {inputs.pr_labels}")
    print()

    outputs = configure(inputs)
    write_outputs(outputs)


if __name__ == "__main__":
    main()
