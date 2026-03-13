#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Configures CI matrix and job decisions for multi-arch workflows.

This script is a pipeline of data transformations:

    1. Parse Inputs    — read GitHub event context → CIInputs
    2. Check Skip CI   — gate: should we skip CI entirely?
    3. Select Targets  — trigger type + labels → GPU families
    4. Decide Jobs     — changed files + topology → per-job-group decisions
    5. Expand Matrix   — families × variant → matrix entries
    6. Write Outputs   — JSON → GITHUB_OUTPUT + GITHUB_STEP_SUMMARY

Each step (except 1 and 6) is a pure function of typed dataclasses,
independently testable without environment variables or filesystem access.

The CI pipeline is a DAG of job groups:

    build-rocm → test-rocm
               → build-rocm-python → build-pytorch → test-pytorch
                                   → build-jax     → test-jax (future)

Step 4 determines which job groups to run, skip, or satisfy with prebuilt
artifacts. Within build-rocm, per-stage rebuild/prebuilt granularity is
available. Test details (which tests to run, smoke vs full) are decided
per test job group.

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
    enable_build_jobs   : "true" or "false"
    test_type           : "smoke" or "full"
"""

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from configure_ci_path_filters import get_git_modified_paths
from github_actions_utils import gha_append_step_summary, gha_set_output

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
    def from_environ() -> "CIInputs":
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


# ---------------------------------------------------------------------------
# Job decisions — the CI pipeline as a DAG of job groups
#
#   build-rocm → test-rocm
#              → build-rocm-python → build-pytorch → test-pytorch
#                                  → build-jax     → test-jax (future)
#
# Each node gets a JobGroupDecision (run/prebuilt/skip). Subclasses add
# group-specific details (per-stage granularity, test type, etc.).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JobGroupDecision:
    """Decision for one node in the CI job graph."""

    action: Literal["run", "prebuilt", "skip"]
    reason: str


@dataclass(frozen=True)
class StageDecision:
    """Decision for a single build stage within build-rocm."""

    action: Literal["rebuild", "prebuilt"]
    reason: str


@dataclass(frozen=True)
class BuildRocmDecision(JobGroupDecision):
    """Build-rocm job group with per-stage granularity."""

    stage_decisions: dict[str, StageDecision] = field(default_factory=dict)

    @property
    def prebuilt_stages(self) -> list[str]:
        return [
            name for name, d in self.stage_decisions.items() if d.action == "prebuilt"
        ]

    @property
    def rebuild_stages(self) -> list[str]:
        return [
            name for name, d in self.stage_decisions.items() if d.action == "rebuild"
        ]


@dataclass(frozen=True)
class TestRocmDecision(JobGroupDecision):
    """Test-rocm job group with test filtering details."""

    test_type: str = "smoke"  # smoke or full
    test_type_reason: str = "default"


@dataclass(frozen=True)
class JobDecisions:
    """Decisions for the entire CI job graph.

    Each field corresponds to a node in the job DAG. The field types show
    which groups have extra decision logic beyond run/skip/prebuilt.
    """

    build_rocm: BuildRocmDecision
    test_rocm: TestRocmDecision
    build_rocm_python: JobGroupDecision
    build_pytorch: JobGroupDecision
    test_pytorch: JobGroupDecision


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

    is_ci_enabled: bool = True
    linux_variants: list[MatrixEntry] = field(default_factory=list)
    windows_variants: list[MatrixEntry] = field(default_factory=list)
    jobs: JobDecisions | None = None

    @staticmethod
    def skipped(reason: str) -> "CIOutputs":
        """Produce empty outputs when CI is skipped."""
        return CIOutputs(is_ci_enabled=False)


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
# Step 4: Decide Jobs
# ---------------------------------------------------------------------------


def decide_jobs(
    inputs: CIInputs,
    changed_files: list[str] | None,
) -> JobDecisions:
    """Determine which job groups to run, skip, or satisfy with prebuilt files.

    Currently returns "run everything, rebuild all stages" — subgraph
    selection based on changed files comes later.
    """
    # TODO: Implement — classify changed files, find entry point in job DAG,
    # propagate forward through reachable nodes, mark unreachable as skip
    return JobDecisions(
        build_rocm=BuildRocmDecision(action="run", reason="default (stub)"),
        test_rocm=TestRocmDecision(action="run", reason="default (stub)"),
        build_rocm_python=JobGroupDecision(action="run", reason="default (stub)"),
        build_pytorch=JobGroupDecision(action="run", reason="default (stub)"),
        test_pytorch=JobGroupDecision(action="run", reason="default (stub)"),
    )


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
    lines.append(f"* `is_ci_enabled`: {outputs.is_ci_enabled}")
    if outputs.jobs:
        lines.append(f"* `test_type`: {outputs.jobs.test_rocm.test_type}")
        for name in (
            "build_rocm",
            "test_rocm",
            "build_rocm_python",
            "build_pytorch",
            "test_pytorch",
        ):
            decision = getattr(outputs.jobs, name)
            lines.append(f"* `{name}`: {decision.action} — {decision.reason}")
    return "\n".join(lines)


def write_outputs(outputs: CIOutputs) -> None:
    """Write results to GITHUB_OUTPUT and GITHUB_STEP_SUMMARY.

    This is the only function with side effects (besides from_environ).
    """
    test_type = outputs.jobs.test_rocm.test_type if outputs.jobs else "smoke"
    output_vars = {
        "linux_variants": json.dumps(
            [entry.to_dict() for entry in outputs.linux_variants]
        ),
        "windows_variants": json.dumps(
            [entry.to_dict() for entry in outputs.windows_variants]
        ),
        # Workflow YAML references this as 'enable_build_jobs'
        "enable_build_jobs": json.dumps(outputs.is_ci_enabled),
        "test_type": test_type,
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
        changed_files = get_git_modified_paths(inputs.base_ref)

    skip = check_skip_ci(inputs=inputs, changed_files=changed_files)
    if skip.skip:
        print(f"Skipping CI: {skip.reason}")
        return CIOutputs.skipped(skip.reason)

    # Steps 3 and 4 are independent: target selection (which GPU families)
    # and job decisions (which job groups run) are orthogonal concerns.
    targets = select_targets(inputs)
    jobs = decide_jobs(inputs=inputs, changed_files=changed_files)

    # Step 5: Expand matrix
    linux_matrix = expand_matrix(
        families=targets.linux_families,
        platform="linux",
        build_variant=inputs.build_variant,
    )
    windows_matrix = expand_matrix(
        families=targets.windows_families,
        platform="windows",
        build_variant=inputs.build_variant,
    )

    return CIOutputs(
        is_ci_enabled=True,
        linux_variants=linux_matrix,
        windows_variants=windows_matrix,
        jobs=jobs,
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
