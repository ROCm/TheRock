#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Configures CI matrix and job decisions for multi-arch workflows.

This script is a pipeline of data transformations:

    1. Parse Inputs    — read GitHub event context → CIInputs, GitContext
    2. Check Skip CI   — gate: should we skip CI entirely?
    3. Decide Jobs     — changed files + topology → per-job-group decisions
    4. Select Targets  — trigger type + labels → per-platform GPU families
    5. Build Configs   — inputs × jobs × families → per-platform build configs
    6. Write Outputs   — JSON → GITHUB_OUTPUT + GITHUB_STEP_SUMMARY

Each step (except 1 and 6) is a pure function of typed dataclasses,
independently testable without environment variables or filesystem access.

The CI pipeline is a DAG of job groups:

    build-rocm → test-rocm
               → build-rocm-python → build-pytorch → test-pytorch
                                   → build-jax     → test-jax (future)
               → build-native-linux   → test-native-linux   (future)
               → build-native-windows → test-native-windows (future)

Step 4 determines which job groups to run, skip, or satisfy with prebuilt
artifacts. Within build-rocm, per-stage rebuild/prebuilt granularity is
available. Test details (which tests to run, quick vs full) are decided
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
    enable_build_jobs     : "true" or "false"
    test_type             : "quick", "standard", "comprehensive", or "full"
"""

import enum
import json
import os
import sys
import random
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

# Add parent directory to path for _therock_utils imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _therock_utils.build_topology import get_topology

from amdgpu_family_matrix import all_build_variants, get_all_families_for_trigger_types
from configure_ci_path_filters import (
    get_git_modified_paths,
    get_git_submodule_paths,
    is_ci_run_required,
)
from github_actions_api import gha_append_step_summary, gha_set_output

# ---------------------------------------------------------------------------
# Input parsing helpers
# ---------------------------------------------------------------------------


def _get_all_build_stages() -> list[str]:
    """Get all build stage names from BUILD_TOPOLOGY.toml.

    Returns:
        List of stage names in the order they appear in the TOML file
    """
    topology = get_topology()
    return list(topology.build_stages.keys())


def _parse_comma_list(raw: str) -> list[str]:
    """Parse a comma-separated string into a list of stripped, lowercased, non-empty names.

    Example: "gfx94X, gfx120X" → ["gfx94x", "gfx120x"]
    """
    return [name.strip().lower() for name in raw.split(",") if name.strip()]


def _parse_prebuilt_stages(raw: str) -> list[str]:
    """Parse prebuilt_stages input, expanding 'all' to all available stages.

    Reads stage names from BUILD_TOPOLOGY.toml when 'all' is specified.

    Example:
        "foundation,compiler-runtime" → ["foundation", "compiler-runtime"]
        "all" → ["foundation", "compiler-runtime", "math-libs", ...]
    """
    stages = _parse_comma_list(raw)
    if "all" in stages:
        return _get_all_build_stages()
    return stages


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

    run_id: str  # GITHUB_RUN_ID value
    event_name: str  # GITHUB_EVENT_NAME value (e.g. "push", "pull_request", "schedule", "workflow_dispatch")
    commit_ref: str  # GITHUB_REF_NAME value
    base_ref: str  # Git ref for the workflow run (PR base or HEAD^1, used for diffing)
    build_variant: str  # Build variant label, e.g. "release", "asan", "tsan"
    release_type: str = ""  # "" for CI, or "dev", "nightly", "prerelease" for releases

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
        for f in fields(self):
            print(f"  {f.name}: {getattr(self, f.name)!r}")

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
        """Parse from GitHub Actions environment."""
        run_id = os.environ["GITHUB_RUN_ID"]
        event_name = os.environ["GITHUB_EVENT_NAME"]
        commit_ref = os.environ["GITHUB_REF_NAME"]

        # Read the full event webhook payload (common to all event triggers).
        event_path = os.environ["GITHUB_EVENT_PATH"]
        with open(event_path) as f:
            event = json.load(f)

        # Extract additional fields based on event type.

        # "inputs" are set for workflow_dispatch, empty otherwise.
        inputs = event.get("inputs") or {}

        # BUILD_VARIANT and RELEASE_TYPE come from workflow_call inputs, not
        # the event payload.
        build_variant = os.environ.get("BUILD_VARIANT", "release")
        release_type = os.environ.get("RELEASE_TYPE", "")

        pr_labels: list[str] = []
        base_ref = "HEAD^1"
        if event_name == "pull_request":
            # Extract label name strings from the event payload's label objects:
            #   Sample input:  [{"name": "ci:skip", "color": "fff", ...}, ...]
            #   Sample output: ["ci:skip", ...]
            pr_obj = event.get("pull_request", {})
            pr_labels = [label["name"].lower() for label in pr_obj.get("labels", [])]

            # The merge commit's first parent is the PR base.
            base_ref = "HEAD^"
        elif event_name == "push":
            base_ref = event.get("before", "HEAD^1")

        return CIInputs(
            run_id=run_id,
            event_name=event_name,
            commit_ref=commit_ref,
            base_ref=base_ref,
            build_variant=build_variant,
            release_type=release_type,
            pr_labels=pr_labels,
            linux_amdgpu_families=_parse_comma_list(
                inputs.get("linux_amdgpu_families", "")
            ),
            windows_amdgpu_families=_parse_comma_list(
                inputs.get("windows_amdgpu_families", "")
            ),
            linux_test_labels=_parse_comma_list(inputs.get("linux_test_labels", "")),
            windows_test_labels=_parse_comma_list(
                inputs.get("windows_test_labels", "")
            ),
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

    # List of relative file paths modified relative to a base ref
    changed_files: list[str] | None = None

    # List of paths of all git submodules in the repo
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
        """Empty context with no git data.

        This should typically be used for schedule/workflow_dispatch events
        where we don't want to diff against a prior commit.
        """
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
# The CI pipeline forms a DAG where each node is a job group:
#
#   build-rocm ──> test-rocm
#              └─> build-rocm-python ──> build-pytorch ──> test-pytorch
#
# Each node gets a JobAction: RUN, PREBUILT, or SKIP.
#   - RUN:      Build from source (or run tests).
#   - PREBUILT: Fetch artifacts from a prior successful run. Only valid for
#               build job groups (build-rocm, build-rocm-python).
#   - SKIP:     Don't run at all. Used when no downstream job needs this
#               node's outputs.
#
# Note: this is aspirational and not fully implemented yet.
#
# Example: a commit that only changes ROCm python packaging code:
#
#   [PREBUILT] build-rocm          (ROCm itself unchanged, reuse artifacts)
#       │
#       ├──> [SKIP] test-rocm      (ROCm unchanged, no need to re-test)
#       │
#       └──> [RUN] build-rocm-python ──> [RUN] build-pytorch
#              (packaging changed)            │
#                                             └──> [RUN] test-pytorch
#
# Subclasses of JobGroupDecision add group-specific details:
#   - BuildRocmDecision: per-stage rebuild/prebuilt granularity
#   - TestRocmDecision: test type (quick/standard/comprehensive/full)
# ---------------------------------------------------------------------------


class JobAction(enum.Enum):
    """Action for a node in the CI job graph or a build stage."""

    RUN = "run"
    PREBUILT = "prebuilt"
    SKIP = "skip"


@dataclass(frozen=True)
class JobGroupDecision:
    """Decision for one node in the CI job graph.

    Nodes may inherit from this base class to add additional options."""

    action: JobAction


@dataclass(frozen=True)
class BuildRocmDecision(JobGroupDecision):
    """Build-rocm job group with per-stage granularity."""

    stage_decisions: dict[str, JobAction] = field(default_factory=dict)
    # Run ID to fetch prebuilt stage artifacts from. Currently passed through
    # from workflow_dispatch input; TODO(#3399): derive automatically from
    # the current commit's parent workflow run.
    baseline_run_id: str = ""

    @property
    def prebuilt_stages(self) -> list[str]:
        return [
            name
            for name, action in self.stage_decisions.items()
            if action == JobAction.PREBUILT
        ]

    @property
    def rebuild_stages(self) -> list[str]:
        return [
            name
            for name, action in self.stage_decisions.items()
            if action == JobAction.RUN
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
    # TODO: Consolidate test_type, test labels, and run_functional_tests
    # (from the single-arch pipeline) into a per-platform test config object
    # (e.g. linux_test_config JSON) instead of separate top-level outputs.


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
        print(f"  build_rocm: {self.build_rocm.action.value}")
        print(f"  test_rocm: {self.test_rocm.action.value}")
        print(f"  build_rocm_python: {self.build_rocm_python.action.value}")
        print(f"  build_pytorch: {self.build_pytorch.action.value}")
        print(f"  test_pytorch: {self.test_pytorch.action.value}")


@dataclass(frozen=True)
class BuildConfig:
    """Build configuration for one platform.

    Produced by expand_matrices, one per platform. Contains per-family info
    for downstream per-architecture job expansion and variant metadata.
    """

    per_family_info: list[dict]  # Per-family metadata for test/artifact jobs
    dist_amdgpu_families: str  # Semicolon-separated
    artifact_group: str
    build_variant_label: str
    build_variant_suffix: str
    build_variant_cmake_preset: str
    build_pytorch: bool
    # Prebuilt stage configuration — set by configure() from JobDecisions.
    prebuilt_stages: list[str] = field(default_factory=list)
    baseline_run_id: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["prebuilt_stages"] = ",".join(self.prebuilt_stages)
        return d


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
    # Test labels pass through from inputs to outputs for downstream workflows.
    linux_test_labels: str = ""
    windows_test_labels: str = ""

    @staticmethod
    def skipped() -> "CIOutputs":
        """Produce empty outputs when CI is skipped."""
        return CIOutputs(is_ci_enabled=False)


# ---------------------------------------------------------------------------
# Step 2: Check Skip CI
# ---------------------------------------------------------------------------


def should_skip_ci(
    ci_inputs: CIInputs,
    git_context: GitContext,
) -> bool:
    """Determine whether CI should be skipped entirely.

    Returns True for:
    - 'ci:skip' PR label
    - Only skippable files changed (docs, .md, etc.)
    - No files changed
    """
    if "ci:skip" in ci_inputs.pr_labels:
        print("  Skipping: 'ci:skip' PR label")
        return True

    # If we have a list of changed files (push/pull_request events), check if
    # CI should run for that set of changed files. For example: if only .md
    # files are changed, skip CI.
    if git_context.changed_files is not None:
        print(
            f"  Checking {len(git_context.changed_files)} changed file(s) "
            f"against path filters..."
        )
        if not is_ci_run_required(git_context.changed_files):
            print("  Skipping: no CI-relevant files changed")
            return True
        else:
            print("  CI-relevant files changed, running CI")

    return False


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

    This code implements the policies from docs/development/test_filtering.md
    and docs/development/ci_behavior_manipulation.md:

    * Available filter types: ["quick", "standard", "comprehensive", "full"]
    * Workflow runs choose a filter type automatically but PRs can override
      with labels like `test_filter:comprehensive`

    Returns (test_type, reason).
    """

    # Check in priority order - highest priority returns early.

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
    """Determine which job groups to run, skip, or satisfy with prebuilt files."""

    # Build ROCm.
    # TODO(#3399): Use changed files and build_topology.py to:
    #   1. set per-stage prebuilt decisions
    #   2. skip job groups that aren't reachable from the changed files
    # Parse explicit prebuilt stages from workflow_dispatch input.
    stage_decisions: dict[str, JobAction] = {}
    if ci_inputs.prebuilt_stages:
        for stage in _parse_prebuilt_stages(ci_inputs.prebuilt_stages):
            stage_decisions[stage] = JobAction.PREBUILT
    build_rocm = BuildRocmDecision(
        action=JobAction.RUN,
        stage_decisions=stage_decisions,
        baseline_run_id=ci_inputs.baseline_run_id,
    )

    # Test ROCm.
    test_type, test_type_reason = _determine_test_type(
        ci_inputs=ci_inputs,
        git_context=git_context,
    )
    test_rocm = TestRocmDecision(
        action=JobAction.RUN,
        test_type=test_type,
        test_type_reason=test_type_reason,
    )

    # Other jobs run unconditionally with no configuration.
    # TODO: job pruning: skip pytorch if only JAX has been edited, etc.

    return JobDecisions(
        build_rocm=build_rocm,
        test_rocm=test_rocm,
        build_rocm_python=JobGroupDecision(action=JobAction.RUN),
        build_pytorch=JobGroupDecision(action=JobAction.RUN),
        test_pytorch=JobGroupDecision(action=JobAction.RUN),
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
    all_families: dict[str, dict],
) -> list[str]:
    """Return only the family names that have an entry for the given platform."""
    return [
        name
        for name in family_names
        if name in all_families and platform in all_families[name]
    ]


def select_targets(ci_inputs: CIInputs) -> TargetSelection:
    """Determine GPU families per platform based on trigger type and inputs.

    Trigger types run progressively larger sets of builds and tests:

    - pull_request: Smallest default set (presubmit families). Designed for
      fast feedback on proposed changes. PR labels can opt in to additional
      families (gfx* labels) or the full set (ci:run-all-archs).
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
        # For CI dispatches, empty input means "no families for that
        # platform" — the caller has full control over what runs.
        # For release dispatches, empty input defaults to all families
        # so that release workflows don't need to enumerate every family.
        linux_names = list(ci_inputs.linux_amdgpu_families)
        windows_names = list(ci_inputs.windows_amdgpu_families)
        if ci_inputs.release_type and not linux_names and not windows_names:
            linux_names = list(all_families.keys())
            windows_names = list(all_families.keys())
            print(
                f"  Release type {ci_inputs.release_type!r} with no "
                f"explicit families -> all families"
            )
    elif ci_inputs.is_pull_request:
        # Smallest default set for fast PR feedback. PR labels can extend
        # the set below (gfx* for individual families, ci:run-all-archs
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
            if label == "ci:run-all-archs":
                # Override to all families.
                linux_names = list(all_families.keys())
                windows_names = list(all_families.keys())
                print("  Label 'ci:run-all-archs' -> all families")
                break
            if label.startswith("gfx"):
                # Trim suffixes from labels since amdgpu_family_matrix.py
                # specifies families with no suffix (e.g. `gfx94x`) but
                # we have some labels like `gfx94X-dcgpu` or `gfx103X-linux`.
                # Note: labels are normalized to lowercase during parsing.
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
    ci_inputs: CIInputs,
    jobs: JobDecisions,
    platform: str,
    variant_config: dict,
    all_families: dict[str, dict],
    families: list[str],
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
    build_variant = ci_inputs.build_variant

    # Extract kernel type from test_runner:<kernel> PR label (e.g. "oem").
    # Selects kernel-specific test runners for families that support them.
    test_runner_kernel = ""
    for label in ci_inputs.pr_labels:
        if label.startswith("test_runner:"):
            test_runner_kernel = label.split(":")[1]
            break

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

        # Determine test runner label.
        test_runs_on = platform_info["test-runs-on"]

        # Handle dual-label configuration with weighted random selection.
        # Some families (e.g. gfx94x) have multiple runner labels available.
        if "test-runs-on-alternate" in platform_info:
            alternate_label = platform_info["test-runs-on-alternate"]
            alternate_weight = platform_info.get("test-runs-on-alternate-weight", 0.5)
            if random.random() < alternate_weight:
                test_runs_on = alternate_label
                print(
                    f"  {family_name}: selected alternate runner (weight={alternate_weight}): "
                    f"{test_runs_on}"
                )
            else:
                print(
                    f"  {family_name}: selected primary runner (weight={1-alternate_weight}): "
                    f"{test_runs_on}"
                )

        # When a test_runner:<kernel> label is set, use the
        # kernel-specific runner if available, otherwise disable testing for
        # this family (the default runner may not have the right kernel).
        if test_runner_kernel:
            kernel_runners = platform_info.get("test-runs-on-kernel", {})
            if test_runner_kernel in kernel_runners:
                test_runs_on = kernel_runners[test_runner_kernel]
                print(
                    f"  {family_name}: using {test_runner_kernel} kernel "
                    f"runner: {test_runs_on}"
                )
            else:
                test_runs_on = ""
                print(
                    f"  {family_name}: no {test_runner_kernel} kernel "
                    f"runner available, disabling tests"
                )

        # TODO(#3433): Remove sandbox logic once ASAN tests are passing
        # For ASAN builds, use sandbox runner to avoid impacting production
        if build_variant == "asan":
            if "test-runs-on-sandbox" in platform_info:
                test_runs_on = platform_info["test-runs-on-sandbox"]
                print(f"  {family_name}: using ASAN sandbox runner: {test_runs_on}")
            else:
                test_runs_on = ""
                print(
                    f"  {family_name}: no ASAN sandbox runner available, "
                    f"disabling tests"
                )

        per_family_info.append(
            {
                "amdgpu_family": platform_info["family"],
                "amdgpu_targets": ",".join(platform_info["fetch-gfx-targets"]),
                "test-runs-on": test_runs_on,
                "sanity_check_only_for_family": platform_info.get(
                    "sanity_check_only_for_family", False
                ),
                "expect_pytorch_failure": platform_info.get(
                    "expect_pytorch_failure", False
                ),
            }
        )

    if not per_family_info:
        return None

    family_names = [f["amdgpu_family"] for f in per_family_info]
    suffix = variant_config.get("build_variant_suffix", "")

    return BuildConfig(
        per_family_info=per_family_info,
        dist_amdgpu_families=";".join(family_names),
        artifact_group=f"multi-arch-{suffix or 'release'}",
        build_variant_label=variant_config["build_variant_label"],
        build_variant_suffix=suffix,
        build_variant_cmake_preset=variant_config["build_variant_cmake_preset"],
        build_pytorch=jobs.build_pytorch.action == JobAction.RUN,
        prebuilt_stages=jobs.build_rocm.prebuilt_stages or [],
        baseline_run_id=jobs.build_rocm.baseline_run_id,
    )


def expand_build_configs(
    ci_inputs: CIInputs,
    jobs: JobDecisions,
    targets: TargetSelection,
) -> BuildConfigs:
    """Build a BuildConfig for each platform that supports the variant.

    Returns BuildConfigs with a BuildConfig per platform, or None for
    platforms where the variant isn't available or no families match.
    """
    all_families = get_all_families_for_trigger_types(
        ["presubmit", "postsubmit", "nightly"]
    )
    build_variant = ci_inputs.build_variant

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
            ci_inputs=ci_inputs,
            jobs=jobs,
            platform=platform,
            variant_config=variant_config,
            all_families=all_families,
            families=families,
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


def write_outputs(
    ci_inputs: CIInputs,
    outputs: CIOutputs,
) -> None:
    """Write results to GITHUB_OUTPUT and GITHUB_STEP_SUMMARY.

    This is the only function with side effects (besides from_environ).
    """
    linux = outputs.builds.linux
    windows = outputs.builds.windows
    test_type = outputs.jobs.test_rocm.test_type if outputs.is_ci_enabled else ""
    output_vars = {
        # Workflow YAML references this as 'enable_build_jobs'
        "enable_build_jobs": json.dumps(outputs.is_ci_enabled),
        "linux_build_config": json.dumps(linux.to_dict()) if linux else "",
        "windows_build_config": json.dumps(windows.to_dict()) if windows else "",
        "test_type": test_type,
        "linux_test_labels": outputs.linux_test_labels,
        "windows_test_labels": outputs.windows_test_labels,
    }
    gha_set_output(output_vars)

    # Lazy import: configure_multi_arch_ci_summary imports types from this
    # module, so importing it at the top level would create a circular import.
    from configure_multi_arch_ci_summary import format_summary

    gha_append_step_summary(
        format_summary(
            ci_inputs=ci_inputs,
            outputs=outputs,
        )
    )


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


def configure(ci_inputs: CIInputs, git_context: GitContext) -> CIOutputs:
    """Main pipeline. Each step feeds the next.

    This function is the primary entry point for testing — construct
    CIInputs and GitContext directly and assert on the returned CIOutputs.
    No git operations or environment access needed.
    """
    print("=== Inputs ===")
    ci_inputs.log()
    git_context.log()

    print("\n=== Checking if CI should run ===")
    if should_skip_ci(ci_inputs=ci_inputs, git_context=git_context):
        return CIOutputs.skipped()
    print("Result: CI will run")

    print("\n=== Deciding job configuration ===")
    jobs = decide_jobs(ci_inputs=ci_inputs, git_context=git_context)
    jobs.log()

    print("\n=== Selecting GPU target families ===")
    targets = select_targets(ci_inputs)
    targets.log()

    print("\n=== Building per-platform configs ===")
    builds = expand_build_configs(
        ci_inputs=ci_inputs,
        jobs=jobs,
        targets=targets,
    )
    builds.log()

    return CIOutputs(
        is_ci_enabled=True,
        builds=builds,
        jobs=jobs,
        linux_test_labels=ci_inputs.linux_test_labels,
        windows_test_labels=ci_inputs.windows_test_labels,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    ci_inputs = CIInputs.from_environ()

    if ci_inputs.is_pull_request or ci_inputs.is_push:
        # 'pull_request' and 'push' events can use the list of changed files
        # compared to the "prior commit" to affect job selections/options.
        git_context = GitContext.from_repo(base_ref=ci_inputs.base_ref)
    else:
        # 'workflow_dispatch' and 'schedule' events don't have as natural
        # a "prior commit" to compare against.
        git_context = GitContext.empty()

    outputs = configure(ci_inputs, git_context)
    write_outputs(ci_inputs=ci_inputs, outputs=outputs)


if __name__ == "__main__":
    main()
