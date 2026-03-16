#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Configures CI matrix and job decisions for multi-arch workflows.

This script is a pipeline of data transformations:

    1. Parse Inputs    — read GitHub event context → CIInputs
    2. Check Skip CI   — gate: should we skip CI entirely?
    3. Decide Jobs     — changed files + topology → per-job-group decisions
    4. Select Targets  — trigger type + labels → per-platform GPU families
    5. Build Configs   — families × variant → per-platform build configs
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
    linux_build_config    : JSON object with build config, or "" if skipped
    windows_build_config  : JSON object with build config, or "" if skipped
    linux_build_enabled   : "true" or "false"
    windows_build_enabled : "true" or "false"
    enable_build_jobs     : "true" or "false"
    test_type             : "smoke" or "full"
"""

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from amdgpu_family_matrix import all_build_variants, get_all_families_for_trigger_types
from configure_ci_path_filters import (
    get_git_modified_paths,
    get_git_submodule_paths,
    is_ci_run_required,
)
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

    def log(self) -> None:
        """Log parsed inputs for CI diagnostics."""
        print("CIInputs:")
        print(f"  event: {self.event_name}")
        print(f"  branch: {self.branch_name}")
        print(f"  variant: {self.build_variant}")
        if self.pr_labels:
            print(f"  pr_labels: {self.pr_labels}")
        if self.linux_amdgpu_families:
            print(f"  linux_amdgpu_families: {self.linux_amdgpu_families}")
        if self.windows_amdgpu_families:
            print(f"  windows_amdgpu_families: {self.windows_amdgpu_families}")
        if self.linux_test_labels:
            print(f"  linux_test_labels: {self.linux_test_labels}")
        if self.windows_test_labels:
            print(f"  windows_test_labels: {self.windows_test_labels}")
        if self.prebuilt_stages:
            print(f"  prebuilt_stages: {self.prebuilt_stages}")

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
class GitContext:
    """Git-derived data for the current commit/PR.

    Separated from CIInputs because these require git operations to compute,
    while CIInputs is parsed from the GitHub Actions environment. Tests
    construct GitContext directly without touching git.
    """

    changed_files: list[str] | None = None
    submodule_paths: list[str] | None = None

    @staticmethod
    def from_repo(base_ref: str) -> "GitContext":
        """Compute from the actual repo. Only called from main()."""
        changed_files = get_git_modified_paths(base_ref)
        submodule_paths = list(get_git_submodule_paths() or [])
        return GitContext(
            changed_files=changed_files,
            submodule_paths=submodule_paths,
        )

    @staticmethod
    def empty() -> "GitContext":
        """No git data (schedule/workflow_dispatch)."""
        return GitContext()

    def log(self) -> None:
        """Log git context for CI diagnostics."""
        if self.changed_files is None:
            print("GitContext: no changed files (schedule/workflow_dispatch)")
            return
        print(f"GitContext: {len(self.changed_files)} changed file(s)")
        for path in self.changed_files[:20]:
            print(f"  {path}")
        if len(self.changed_files) > 20:
            print(f"  ... and {len(self.changed_files) - 20} more")


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

    def log(self) -> None:
        """Log selected targets for CI diagnostics."""
        print("TargetSelection:")
        print(f"  linux: {self.linux_families}")
        print(f"  windows: {self.windows_families}")


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
    """Test-rocm job group with test filtering details.

    test_type levels (from least to most testing):
    - "quick"         — default for PRs and push
    - "standard"      — via test_filter:standard PR label
    - "comprehensive" — schedule/nightly
    - "full"          — submodule changes, test:* labels, or test_filter:full
    """

    test_type: str = "quick"
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

    def log(self) -> None:
        """Log job decisions for CI diagnostics."""
        print("JobDecisions:")
        print(
            f"  test_type: {self.test_rocm.test_type} "
            f"({self.test_rocm.test_type_reason})"
        )
        print(f"  build_rocm: {self.build_rocm.action}")
        print(f"  test_rocm: {self.test_rocm.action}")
        print(f"  build_rocm_python: {self.build_rocm_python.action}")
        print(f"  build_pytorch: {self.build_pytorch.action}")
        print(f"  test_pytorch: {self.test_pytorch.action}")


@dataclass(frozen=True)
class BuildConfig:
    """Build configuration for one platform.

    Produced by expand_matrices, one per platform. Contains per-family info
    for downstream per-architecture job expansion and variant metadata.
    """

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
class BuildConfigs:
    """Build configurations for both platforms, produced by expand_build_configs."""

    linux: BuildConfig | None = None
    windows: BuildConfig | None = None

    def _log_platform(self, name: str, config: BuildConfig | None) -> None:
        if config is None:
            print(f"  {name}: skipped")
        else:
            print(
                f"  {name}: {config.artifact_group} "
                f"families={config.dist_amdgpu_families}"
            )

    def log(self) -> None:
        """Log build configs for CI diagnostics."""
        print("BuildConfigs:")
        self._log_platform("linux", self.linux)
        self._log_platform("windows", self.windows)


@dataclass(frozen=True)
class CIOutputs:
    """All outputs from the CI configuration pipeline."""

    is_ci_enabled: bool = True
    builds: BuildConfigs = field(default_factory=BuildConfigs)
    jobs: JobDecisions | None = None

    @staticmethod
    def skipped(reason: str) -> "CIOutputs":
        """Produce empty outputs when CI is skipped."""
        return CIOutputs(is_ci_enabled=False)


# ---------------------------------------------------------------------------
# Step 2: Check Skip CI
# ---------------------------------------------------------------------------


def check_skip_ci(
    ci_inputs: CIInputs,
    git_context: GitContext,
) -> SkipDecision:
    """Determine whether CI should be skipped entirely.

    Returns SkipDecision(skip=True) for:
    - 'skip-ci' PR label
    - Only skippable files changed (docs, .md, etc.)
    - No files changed

    schedule and workflow_dispatch always proceed (changed_files is None
    for those triggers, and they have no PR labels).
    """
    if "skip-ci" in ci_inputs.pr_labels:
        return SkipDecision(skip=True, reason="skip-ci label")

    # changed_files is None for schedule/workflow_dispatch — always proceed.
    if git_context.changed_files is not None and not is_ci_run_required(
        git_context.changed_files
    ):
        return SkipDecision(skip=True, reason="no CI-relevant files changed")

    return SkipDecision(skip=False, reason="")


# ---------------------------------------------------------------------------
# Step 3: Decide Jobs
# ---------------------------------------------------------------------------


_VALID_TEST_FILTER_TYPES = {"quick", "standard", "comprehensive", "full"}


def _has_test_labels(ci_inputs: CIInputs) -> bool:
    """Check whether any test labels were specified (workflow_dispatch or PR)."""
    if ci_inputs.linux_test_labels or ci_inputs.windows_test_labels:
        return True
    return any(label.startswith("test:") for label in ci_inputs.pr_labels)


def _determine_test_type(
    ci_inputs: CIInputs,
    git_context: GitContext,
) -> tuple[str, str]:
    """Determine test_type and reason based on trigger, labels, and changed files.

    Test types from least to most testing:

    - "quick": Fast sanity checks. Default for PRs and push where only
      build infra or non-submodule files changed. Keeps CI fast for
      routine changes that are unlikely to break GPU-specific behavior.
    - "standard": More thorough than quick, but not full nightly coverage.
      Only available via explicit test_filter:standard PR label.
    - "comprehensive": Full nightly test suite. Used for scheduled runs
      to catch regressions across all components without requiring a
      submodule change to trigger it.
    - "full": Everything, including tests for specific components named
      by test:* labels. Triggered when a submodule changes (the actual
      GPU libraries changed, so we need thorough validation) or when
      test labels explicitly request specific component tests.

    The test_filter: PR label can override any of the above, giving
    developers manual control (e.g. test_filter:comprehensive on a PR
    to get nightly-level coverage before merge).

    Returns (test_type, reason). Checked in priority order — highest
    priority overrides win and return early.
    """
    # Priority 1: test_filter: PR label is an explicit manual override.
    # This is the escape hatch: run comprehensive on a PR before merge,
    # or downgrade to quick if you know the change is safe.
    for label in ci_inputs.pr_labels:
        if not label.startswith("test_filter:"):
            continue
        filter_type = label.split(":")[1]
        if filter_type not in _VALID_TEST_FILTER_TYPES:
            raise ValueError(
                f"Unrecognized test_filter value: {filter_type!r}. "
                f"Valid values: {sorted(_VALID_TEST_FILTER_TYPES)}"
            )
        return filter_type, f"test_filter label: {label}"

    # Priority 2: test:* labels request specific component tests (e.g.
    # test:rocprim). When someone explicitly asks for tests, run the full
    # suite — they're investigating something specific.
    if _has_test_labels(ci_inputs):
        return "full", "test labels specified"

    # Priority 3: schedule runs the full nightly suite — comprehensive
    # coverage on a cadence, catching regressions that quick tests miss.
    if ci_inputs.is_schedule:
        return "comprehensive", "scheduled run"

    # Priority 4: a submodule change means actual library code changed
    # (e.g. rocBLAS, MIOpen). These need full testing since the change
    # could affect any downstream consumer.
    if (
        git_context.changed_files is not None
        and git_context.submodule_paths is not None
    ):
        matching = set(git_context.submodule_paths) & set(git_context.changed_files)
        if matching:
            return "full", f"submodule(s) changed: {sorted(matching)}"

    # Default: quick tests for fast CI feedback.
    return "quick", "default"


def decide_jobs(
    ci_inputs: CIInputs,
    git_context: GitContext,
) -> JobDecisions:
    """Determine which job groups to run, skip, or satisfy with prebuilt files.

    All job groups currently run unconditionally. test_type is determined
    based on trigger type, labels, and changed files.

    TODO(#3399): Use changed files and BUILD_TOPOLOGY.toml to set per-stage
    prebuilt decisions in BuildRocmDecision.stage_decisions, and skip job
    groups that aren't reachable from the changed files.
    """
    test_type, test_type_reason = _determine_test_type(
        ci_inputs=ci_inputs,
        git_context=git_context,
    )

    return JobDecisions(
        build_rocm=BuildRocmDecision(action="run", reason="default"),
        test_rocm=TestRocmDecision(
            action="run",
            reason="default",
            test_type=test_type,
            test_type_reason=test_type_reason,
        ),
        build_rocm_python=JobGroupDecision(action="run", reason="default"),
        build_pytorch=JobGroupDecision(action="run", reason="default"),
        test_pytorch=JobGroupDecision(action="run", reason="default"),
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


def select_targets(ci_inputs: CIInputs) -> TargetSelection:
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
    if ci_inputs.is_workflow_dispatch:
        # Manual trigger: caller specifies exact families per platform.
        # Empty input means "no families for that platform" — the caller
        # has full control over what runs.
        linux_names = list(ci_inputs.linux_amdgpu_families)
        windows_names = list(ci_inputs.windows_amdgpu_families)
    elif ci_inputs.is_pull_request:
        # Smallest default set for fast PR feedback. PR labels can extend
        # the set below (gfx* for individual families, run-all-archs-ci
        # for everything).
        defaults = list(get_all_families_for_trigger_types(["presubmit"]).keys())
        linux_names = list(defaults)
        windows_names = list(defaults)
    elif ci_inputs.is_push:
        # Broader than PR: presubmit + postsubmit. Code has landed, so
        # we validate on more targets (e.g. gfx950) without paying full
        # nightly cost.
        defaults = list(
            get_all_families_for_trigger_types(["presubmit", "postsubmit"]).keys()
        )
        linux_names = list(defaults)
        windows_names = list(defaults)
    elif ci_inputs.is_schedule:
        # Full nightly coverage: every known family, including targets
        # that are too slow or expensive for per-push CI.
        linux_names = list(all_families.keys())
        windows_names = list(all_families.keys())
    else:
        raise ValueError(f"Unsupported event type: {ci_inputs.event_name!r}")

    # PR labels can extend the family set (both platforms)
    if ci_inputs.is_pull_request:
        for label in ci_inputs.pr_labels:
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
# Step 5: Build Configs
# ---------------------------------------------------------------------------


def _expand_build_config_for_platform(
    families: list[str],
    platform: str,
    build_variant: str,
    all_families: dict[str, dict],
    variant_config: dict,
) -> BuildConfig | None:
    """Build a BuildConfig for one platform, or None if no families match.

    Collects per-family info for all families that support the requested
    build variant on this platform, then bundles them into a BuildConfig.

    Per-family info fields:
    - amdgpu_family: family name for THEROCK_AMDGPU_FAMILIES
    - amdgpu_targets: comma-separated gfx targets for split artifact fetching
    - test-runs-on: runner label for testing (empty = no test runner available)
    - sanity_check_only_for_family: whether to limit test scope
    """
    per_family_info: list[dict] = []

    for family_name in families:
        # select_targets already validates family names and filters by
        # platform availability. Family name uniqueness is validated by
        # amdgpu_family_matrix_test.py. We can index directly here.
        platform_info = all_families[family_name][platform]

        # Filter out families missing the build variant (e.g. 'asan').
        if build_variant not in platform_info["build_variants"]:
            print(
                f"  Family {family_name} does not support variant "
                f"{build_variant} on {platform}, skipping"
            )
            continue

        per_family_info.append(
            {
                "amdgpu_family": platform_info["family"],
                "amdgpu_targets": ",".join(platform_info["fetch-gfx-targets"]),
                "test-runs-on": platform_info["test-runs-on"],
                "sanity_check_only_for_family": platform_info.get(
                    "sanity_check_only_for_family", False
                ),
            }
        )

    if not per_family_info:
        return None

    family_names = [f["amdgpu_family"] for f in per_family_info]
    expect_failure = variant_config.get("expect_failure", False)
    expect_pytorch_failure = variant_config.get("expect_pytorch_failure", False)
    suffix = variant_config.get("build_variant_suffix", "")

    return BuildConfig(
        matrix_per_family_json=json.dumps(per_family_info),
        dist_amdgpu_families=";".join(family_names),
        artifact_group=f"multi-arch-{suffix or 'release'}",
        build_variant_label=variant_config["build_variant_label"],
        build_variant_suffix=suffix,
        build_variant_cmake_preset=variant_config["build_variant_cmake_preset"],
        expect_failure=expect_failure,
        build_pytorch=not expect_failure and not expect_pytorch_failure,
    )


def expand_build_configs(
    targets: TargetSelection,
    build_variant: str,
) -> BuildConfigs:
    """Build a BuildConfig for each platform that supports the variant.

    Returns BuildConfigs with a BuildConfig per platform, or None for
    platforms where the variant isn't available or no families match.
    """
    all_families = get_all_families_for_trigger_types(
        ["presubmit", "postsubmit", "nightly"]
    )

    linux_config: BuildConfig | None = None
    windows_config: BuildConfig | None = None

    for platform, families in [
        ("linux", targets.linux_families),
        ("windows", targets.windows_families),
    ]:
        variant_config = all_build_variants.get(platform, {}).get(build_variant)
        if not variant_config:
            print(
                f"  Platform {platform} has no config for build variant "
                f"{build_variant}, skipping"
            )
            continue
        config = _expand_build_config_for_platform(
            families=families,
            platform=platform,
            build_variant=build_variant,
            all_families=all_families,
            variant_config=variant_config,
        )
        if platform == "linux":
            linux_config = config
        else:
            windows_config = config

    return BuildConfigs(
        linux=linux_config,
        windows=windows_config,
    )


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
        jobs = outputs.jobs
        lines.append(f"* `test_type`: {jobs.test_rocm.test_type}")
        lines.append(
            f"* `build_rocm`: {jobs.build_rocm.action} — {jobs.build_rocm.reason}"
        )
        lines.append(
            f"* `test_rocm`: {jobs.test_rocm.action} — {jobs.test_rocm.reason}"
        )
        lines.append(
            f"* `build_rocm_python`: {jobs.build_rocm_python.action} — {jobs.build_rocm_python.reason}"
        )
        lines.append(
            f"* `build_pytorch`: {jobs.build_pytorch.action} — {jobs.build_pytorch.reason}"
        )
        lines.append(
            f"* `test_pytorch`: {jobs.test_pytorch.action} — {jobs.test_pytorch.reason}"
        )
    return "\n".join(lines)


def write_outputs(outputs: CIOutputs) -> None:
    """Write results to GITHUB_OUTPUT and GITHUB_STEP_SUMMARY.

    This is the only function with side effects (besides from_environ).
    """
    test_type = outputs.jobs.test_rocm.test_type if outputs.jobs else "quick"
    linux = outputs.builds.linux
    windows = outputs.builds.windows
    output_vars = {
        "linux_build_config": json.dumps(linux.to_dict()) if linux else "",
        "windows_build_config": json.dumps(windows.to_dict()) if windows else "",
        "linux_build_enabled": json.dumps(linux is not None),
        "windows_build_enabled": json.dumps(windows is not None),
        # Workflow YAML references this as 'enable_build_jobs'
        "enable_build_jobs": json.dumps(outputs.is_ci_enabled),
        "test_type": test_type,
    }
    gha_set_output(output_vars)
    gha_append_step_summary(format_summary(outputs))


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


def configure(ci_inputs: CIInputs, git_context: GitContext) -> CIOutputs:
    """Main pipeline. Each step feeds the next.

    This function is the primary entry point for testing — construct
    CIInputs and GitContext directly and assert on the returned CIOutputs.
    No git operations or environment access needed.
    """
    ci_inputs.log()
    git_context.log()

    # Step 2: Gate — should we skip CI entirely?
    skip_decision = check_skip_ci(ci_inputs=ci_inputs, git_context=git_context)
    if skip_decision.skip:
        print(f"Skipping CI: {skip_decision.reason}")
        return CIOutputs.skipped(skip_decision.reason)

    # Steps 3 and 4 are independent: job decisions (which job groups run)
    # and target selection (which GPU families) are orthogonal concerns.
    jobs = decide_jobs(ci_inputs=ci_inputs, git_context=git_context)
    jobs.log()
    targets = select_targets(ci_inputs)
    targets.log()

    # Step 5: Build configs per platform
    builds = expand_build_configs(
        targets=targets,
        build_variant=ci_inputs.build_variant,
    )
    builds.log()

    return CIOutputs(
        is_ci_enabled=True,
        builds=builds,
        jobs=jobs,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    ci_inputs = CIInputs.from_environ()

    # Build git context for push/PR triggers (need changed files for
    # skip-ci and test_type decisions). Schedule/workflow_dispatch don't
    # need git data.
    if ci_inputs.is_pull_request or ci_inputs.is_push:
        git_context = GitContext.from_repo(base_ref=ci_inputs.base_ref)
    else:
        git_context = GitContext.empty()

    outputs = configure(ci_inputs, git_context)
    write_outputs(outputs)


if __name__ == "__main__":
    main()
