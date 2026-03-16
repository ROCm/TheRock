#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Configures CI matrix and job decisions for multi-arch workflows.

This script is a pipeline of data transformations:

    1. Parse Inputs    — read GitHub event context → CIInputs
    2. Check Skip CI   — gate: should we skip CI entirely?
    3. Decide Jobs     — changed files + topology → per-job-group decisions
    4. Select Targets  — trigger type + labels → per-platform GPU families
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

from amdgpu_family_matrix import all_build_variants, get_all_families_for_trigger_types
from configure_ci_path_filters import get_git_modified_paths
from github_actions_utils import gha_append_step_summary, gha_set_output

# ---------------------------------------------------------------------------
# Input parsing helpers
# ---------------------------------------------------------------------------


def _parse_comma_list(raw: str) -> list[str]:
    """Parse a comma-separated string into a list of stripped, non-empty names.

    Example: "gfx94X, gfx120X" → ["gfx94X", "gfx120X"]
    """
    return [name.strip() for name in raw.split(",") if name.strip()]


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

    # Per-platform workflow_dispatch overrides (parsed from comma-separated input)
    linux_amdgpu_families: list[str] = field(default_factory=list)
    windows_amdgpu_families: list[str] = field(default_factory=list)
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
            linux_amdgpu_families=_parse_comma_list(
                inputs.get("linux_amdgpu_families", "")
            ),
            windows_amdgpu_families=_parse_comma_list(
                inputs.get("windows_amdgpu_families", "")
            ),
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
# Step 3: Decide Jobs
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
# Step 4: Select Targets
# ---------------------------------------------------------------------------


def _validate_family_names(
    names: list[str],
    known: dict[str, dict],
) -> None:
    """Raise ValueError if any family name is not in the known matrix."""
    unknown = [name for name in names if name not in known]
    if unknown:
        raise ValueError(
            f"Unknown GPU families: {unknown}. "
            f"Known families: {sorted(known.keys())}"
        )


def _filter_families_by_platform(
    family_names: list[str],
    platform: str,
    lookup_matrix: dict[str, dict],
) -> list[str]:
    """Return only the family names that have an entry for the given platform."""
    return [
        name
        for name in family_names
        if name in lookup_matrix and platform in lookup_matrix[name]
    ]


def select_targets(inputs: CIInputs) -> TargetSelection:
    """Determine GPU families per platform based on trigger type and inputs.

    Trigger types run progressively larger sets of builds and tests:

    - pull_request: Smallest default set (presubmit families). Designed for
      fast feedback on proposed changes. PR labels can opt in to additional
      families (gfx* labels) or the full set (run-all-archs-ci).
    - push: Broader coverage (presubmit + postsubmit families). Runs on
      code that has landed, so we want more thorough validation than PRs
      without paying the full nightly cost.
    - schedule: Full coverage (all families including nightly-only). Catches
      regressions on targets that are too slow or expensive for every push.
    - workflow_dispatch: Full manual control. Per-platform family inputs are
      taken directly from the workflow inputs, giving the caller the ability
      to either replicate what CI does on PRs/push or build/test a narrow
      set of targets for investigation.

    Returns per-platform family lists, filtered to only include families
    that have a platform entry in amdgpu_family_matrix.py.
    """
    all_families = get_all_families_for_trigger_types(
        ["presubmit", "postsubmit", "nightly"]
    )

    # Select family names per platform based on trigger type.
    # Ordered from most-specific (workflow_dispatch) to broadest (schedule).
    if inputs.is_workflow_dispatch:
        # Manual trigger: caller specifies exact families per platform.
        # Empty input means "no families for that platform" — the caller
        # has full control over what runs.
        linux_names = list(inputs.linux_amdgpu_families)
        windows_names = list(inputs.windows_amdgpu_families)
    elif inputs.is_pull_request:
        # Smallest default set for fast PR feedback. PR labels can extend
        # the set below (gfx* for individual families, run-all-archs-ci
        # for everything).
        defaults = list(get_all_families_for_trigger_types(["presubmit"]).keys())
        linux_names = list(defaults)
        windows_names = list(defaults)
    elif inputs.is_push:
        # Broader than PR: presubmit + postsubmit. Code has landed, so
        # we validate on more targets (e.g. gfx950) without paying full
        # nightly cost.
        defaults = list(
            get_all_families_for_trigger_types(["presubmit", "postsubmit"]).keys()
        )
        linux_names = list(defaults)
        windows_names = list(defaults)
    elif inputs.is_schedule:
        # Full nightly coverage: every known family, including targets
        # that are too slow or expensive for per-push CI.
        linux_names = list(all_families.keys())
        windows_names = list(all_families.keys())
    else:
        raise ValueError(f"Unsupported event type: {inputs.event_name!r}")

    # PR labels can extend the family set (both platforms)
    if inputs.is_pull_request:
        for label in inputs.pr_labels:
            if label == "run-all-archs-ci":
                # Override to all families.
                linux_names = list(all_families.keys())
                windows_names = list(all_families.keys())
                print("  Label 'run-all-archs-ci' -> all families")
                break
            if label.startswith("gfx"):
                target = label.split("-")[0]
                linux_names.append(target)
                windows_names.append(target)
                print(f"  Label '{label}' -> adding target {target}")

    # De-dup, validate, then filter by platform availability.
    linux_names = list(dict.fromkeys(linux_names))
    windows_names = list(dict.fromkeys(windows_names))
    _validate_family_names(linux_names, all_families)
    _validate_family_names(windows_names, all_families)
    # TODO: For workflow_dispatch, a family requested for a specific platform
    # but not available there (e.g. gfx94x on windows) is silently dropped.
    # Consider validating per-platform and reporting the mismatch.
    # We could also filter per-platform in get_all_families_for_trigger_types.
    linux_names = _filter_families_by_platform(linux_names, "linux", all_families)
    windows_names = _filter_families_by_platform(windows_names, "windows", all_families)

    return TargetSelection(
        linux_families=linux_names,
        windows_families=windows_names,
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

    In multi-arch mode, all families that support the requested build variant
    are grouped into a single matrix entry. This produces one entry per
    build variant (typically just one — "release"), containing a JSON array
    of per-family info that downstream jobs matrix-expand over for
    per-architecture stages.

    The per-family info includes:
    - amdgpu_family: family name for THEROCK_AMDGPU_FAMILIES
    - amdgpu_targets: comma-separated gfx targets for split artifact fetching
    - test-runs-on: runner label for testing (empty = no test runner available)
    - sanity_check_only_for_family: whether to limit test scope
    """
    all_families = get_all_families_for_trigger_types(
        ["presubmit", "postsubmit", "nightly"]
    )
    platform_build_variants = all_build_variants.get(platform, {})

    # Collect per-family info, grouped by build variant. Each family may
    # support multiple variants (e.g. gfx94x supports release + asan + tsan),
    # but we only keep families that match the requested build_variant.
    variant_family_info: dict[str, list[dict]] = {}
    variant_config: dict[str, dict] = {}

    for family_name in families:
        family_entry = all_families.get(family_name)
        if not family_entry or platform not in family_entry:
            continue
        platform_info = family_entry[platform]

        for supported_variant in platform_info.get("build_variants", []):
            if supported_variant != build_variant:
                continue

            if supported_variant not in variant_family_info:
                variant_family_info[supported_variant] = []
                variant_config[supported_variant] = platform_build_variants.get(
                    supported_variant, {}
                )

            # De-dup by family name (a family can appear once per variant).
            existing = [
                f["amdgpu_family"] for f in variant_family_info[supported_variant]
            ]
            amdgpu_family = platform_info["family"]
            if amdgpu_family in existing:
                continue

            fetch_gfx_targets = platform_info.get("fetch-gfx-targets", [])
            variant_family_info[supported_variant].append(
                {
                    "amdgpu_family": amdgpu_family,
                    "amdgpu_targets": ",".join(fetch_gfx_targets),
                    "test-runs-on": platform_info.get("test-runs-on", ""),
                    "sanity_check_only_for_family": platform_info.get(
                        "sanity_check_only_for_family", False
                    ),
                }
            )

    # Create one MatrixEntry per build variant.
    entries: list[MatrixEntry] = []
    for variant_name, family_info_list in variant_family_info.items():
        config = variant_config[variant_name]
        if not config:
            continue

        family_names = [f["amdgpu_family"] for f in family_info_list]
        expect_failure = config.get("expect_failure", False)
        expect_pytorch_failure = config.get("expect_pytorch_failure", False)
        suffix = config.get("build_variant_suffix", "")

        entries.append(
            MatrixEntry(
                matrix_per_family_json=json.dumps(family_info_list),
                dist_amdgpu_families=";".join(family_names),
                artifact_group=f"multi-arch-{suffix or 'release'}",
                build_variant_label=config["build_variant_label"],
                build_variant_suffix=suffix,
                build_variant_cmake_preset=config["build_variant_cmake_preset"],
                expect_failure=expect_failure,
                build_pytorch=not expect_failure and not expect_pytorch_failure,
            )
        )

    return entries


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

    # Steps 3 and 4 are independent: job decisions (which job groups run)
    # and target selection (which GPU families) are orthogonal concerns.
    jobs = decide_jobs(inputs=inputs, changed_files=changed_files)
    targets = select_targets(inputs)

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
