#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Auto-compute per-stage rebuild/reuse decisions for multi-arch CI.

This module decides, per build stage, whether the stage can be satisfied with
prebuilt artifacts (reuse) instead of being rebuilt. It reuses the existing
``stage_impact`` and ``baseline_runs`` tooling and adds two gates before a stage
is reported reusable:

* Impact gate (``stage_impact.analyze_stage_impact``) - the change does not
  affect the stage, so it is a *candidate* for reuse.
* Availability gate (``baseline_runs.select_baseline_run``) - a single healthy,
  commit-compatible baseline run actually contains the artifacts that stage
  would produce, verified independently for every platform being built.

Mode switch
-----------
The module is wired into CI behind a two-way ``STAGE_REUSE_MODE`` switch so it
can be observed before it changes anything:

* ``dry-run``    - DEFAULT. Compute the analysis and LOG, for each stage that
                   *would* be reused, a line to the console + step summary, but
                   return NO auto stages, so ``prebuilt_stages`` is unchanged
                   and every stage still builds exactly.
* ``reuse-stage`` - Compute the analysis and actually return the reuse stages so
                     the orchestrator copies their artifacts and skips the build.

Note: this ``STAGE_REUSE_MODE`` switch drives the *automatic* detection layer.
It is orthogonal to the ``prebuilt_stages`` workflow input, which is the
explicit, manual list of stages to reuse and is always honored regardless of
mode.

Environment Variables
--------------------
``_default_baseline_selector`` reads the following environment variables. These
are set by ``setup_multi_arch.yml`` and form the stage-reuse interface:

* ``STAGE_REUSE_MODE``           - ``dry-run`` (default) or ``reuse-stage``.
* ``GITHUB_REPOSITORY``          - ``owner/repo`` (default ``ROCm/TheRock``).
* ``STAGE_REUSE_BASELINE_REPOSITORY`` - repository to query for baseline runs
                                   (default: uses ``GITHUB_REPOSITORY``). Set
                                   this to ``ROCm/TheRock`` when external repos
                                   want to reuse TheRock's prebuilt artifacts.
* ``STAGE_REUSE_BASELINE_BRANCH``  - baseline branch to search (default ``main``).
* ``STAGE_REUSE_BASELINE_WORKFLOW`` - baseline workflow file
                                   (default ``multi_arch_ci.yml``).
* ``STAGE_REUSE_CURRENT_SHA``    - current commit SHA; enables the
                                   commit-compatibility rule when set.
* ``STAGE_REUSE_MAX_AGE_HOURS``  - recency window in hours; disables the recency
                                   rule when unset.
* ``STAGE_REUSE_COMMIT_HISTORY`` - number of branch commits to fetch for
                                   ancestry (default ``50``).
* ``STAGE_REUSE_EXTERNAL_REBUILD_STAGES`` - comma-separated list of stages that
                                   the external repo affects (e.g., ``math-libs``).
                                   When set, all other stages become reuse candidates
                                   without requiring changed-file analysis. Use this
                                   for external repos where path analysis isn't available.
"""

import enum
import os
import logging
import functools
import sys
import baseline_runs
import github_actions_api
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

# Add build_tools to the path so sibling CI modules and _therock_utils import
# cleanly regardless of the current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _therock_utils.artifact_backend import ARTIFACT_EXTENSIONS
from _therock_utils.build_topology import BuildTopology, get_topology
from _therock_utils.cmake_amdgpu_targets import amdgpu_family_map, expand_families
from artifact_manager import ARTIFACT_COMPONENTS
from baseline_runs import BaselineRun, RequiredArtifact
from github_actions_api import GitHubAPIError
from stage_impact import analyze_stage_impact

logger = logging.getLogger(__name__)

LOG_PREFIX = "[STAGE-REUSE]"
GENERIC_FAMILY = "generic"


class StageReuseMode(enum.Enum):
    DRY_RUN = "dry-run"
    REUSE_STAGE = "reuse-stage"

    @staticmethod
    def from_environ(default: "StageReuseMode | None" = None) -> "StageReuseMode":
        """Read STAGE_REUSE_MODE; default to dry-run when unset/invalid."""
        default = default or StageReuseMode.DRY_RUN
        raw = (os.environ.get("STAGE_REUSE_MODE", "") or "").strip().lower()
        for mode in StageReuseMode:
            if raw == mode.value:
                return mode
        return default


BaselineSelector = Callable[[Sequence[RequiredArtifact]], BaselineRun | None]


@dataclass(frozen=True)
class StageReusePlan:
    """Result of the *planning* step: which stages are reuse candidates.
    This is a pure function of the changed files and topology -- no baseline
    selection, artifact verification, or reporting. Keeping it separate lets
    callers reuse the impact decision without triggering any network/API work.
    """

    candidate_stages: tuple[str, ...]
    rebuild_stages: tuple[str, ...]
    full_rebuild_required: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class AutoStageReuse:
    """Result of the auto stage-reuse analysis (planning + verification)."""

    mode: StageReuseMode
    candidate_stages: tuple[str, ...]
    rebuild_stages: tuple[str, ...]
    full_rebuild_required: bool
    baseline_run_id: str | None
    baseline_html_url: str | None
    available_stages: tuple[str, ...]
    unavailable_stages: tuple[str, ...]
    applied_reuse_stages: tuple[str, ...]
    reasons: tuple[str, ...]
    report_lines: tuple[str, ...] = field(default_factory=tuple)
    platform_available: dict[str, tuple[str, ...]] = field(default_factory=dict)


def _target_families(
    linux_amdgpu_families: Sequence[str],
    windows_amdgpu_families: Sequence[str],
) -> tuple[str, ...]:
    """GFX targets whose artifacts must be verified for stage reuse.

    Expands family names (like 'gfx94X-dcgpu') to their constituent GFX targets
    (like 'gfx942') since artifact archives are named with specific targets.
    De-duplicates and always appends the generic pseudo-family.
    """
    families = list(dict.fromkeys([*linux_amdgpu_families, *windows_amdgpu_families]))

    # Expand family names to actual GFX targets
    # Artifacts are named with targets (e.g., blas_lib_gfx942.tar.zst), not families
    try:
        family_map = amdgpu_family_map()
        targets = expand_families(families, family_map, strict=False)
    except Exception as exc:
        logger.warning(
            "%s could not expand families to targets (%s); using family names as-is",
            LOG_PREFIX,
            exc,
        )
        targets = families

    if GENERIC_FAMILY not in targets:
        targets.append(GENERIC_FAMILY)
    return tuple(targets)


def _required_artifacts_for_stages(
    topology: BuildTopology,
    stage_names: Sequence[str],
    target_families: Sequence[str],
) -> list[RequiredArtifact]:
    """Artifact/family pairs the given stages produce."""

    artifacts_by_group = topology.get_artifact_group_to_artifacts()
    required: list[RequiredArtifact] = []
    seen: set[RequiredArtifact] = set()
    for stage_name in stage_names:
        stage = topology.build_stages.get(stage_name)
        if stage is None:
            continue
        for group_name in stage.artifact_groups:
            for artifact_name in artifacts_by_group.get(group_name, []):
                for family in target_families:
                    req = RequiredArtifact(name=artifact_name, target_family=family)
                    if req not in seen:
                        seen.add(req)
                        required.append(req)
    return required


def _matched_filenames(baseline: BaselineRun | None) -> set[str]:
    if baseline is None:
        return set()
    return set(baseline.artifact_availability.matched_filenames)


def _stage_artifacts_available(
    topology: BuildTopology,
    stage_name: str,
    target_families: Sequence[str],
    available_filenames: set[str],
) -> bool:
    """True when every artifact this stage produces has an archive present."""

    artifacts_by_group = topology.get_artifact_group_to_artifacts()
    stage = topology.build_stages.get(stage_name)
    if stage is None:
        return False
    for group_name in stage.artifact_groups:
        for artifact_name in artifacts_by_group.get(group_name, []):
            for family in target_families:
                found = False
                for component in ARTIFACT_COMPONENTS:
                    for extension in ARTIFACT_EXTENSIONS:
                        filename = f"{artifact_name}_{component}_{family}{extension}"
                        if filename in available_filenames:
                            found = True
                            break
                    if found:
                        break
                if not found:
                    return False
    return True


def plan_stage_reuse(
    *,
    changed_files: Sequence[str] | None,
    platform: str | None = None,
    topology: BuildTopology | None = None,
    external_rebuild_stages: Sequence[str] | None = None,
) -> StageReusePlan:
    """Planning step: which stages are unaffected reuse candidates.

    Pure decision logic -- no baseline selection, artifact verification, or
    reporting -- so it can be reused independently of the CI plumbing.

    Args:
        changed_files: List of changed file paths for impact analysis.
        platform: Platform for filtering (linux/windows).
        topology: Build topology (loaded if not provided).
        external_rebuild_stages: For external repos without changed_files,
            specify which stages must rebuild. All other stages become candidates.
    """

    if topology is None:
        topology = get_topology()

    # External repo mode: explicit rebuild stages, everything else is reusable
    if external_rebuild_stages is not None:
        all_stages = set(topology.build_stages.keys())
        rebuild_set = set(external_rebuild_stages)
        # Validate that specified stages exist
        invalid = rebuild_set - all_stages
        if invalid:
            logger.warning(
                "%s invalid external_rebuild_stages: %s (ignoring)",
                LOG_PREFIX,
                ", ".join(sorted(invalid)),
            )
            rebuild_set = rebuild_set & all_stages
        candidate_stages = tuple(sorted(all_stages - rebuild_set))
        return StageReusePlan(
            candidate_stages=candidate_stages,
            rebuild_stages=tuple(sorted(rebuild_set)),
            full_rebuild_required=False,
            reasons=(f"external repo rebuild stages: {', '.join(sorted(rebuild_set))}",),
        )

    if changed_files is None:
        return StageReusePlan(
            candidate_stages=(),
            rebuild_stages=(),
            full_rebuild_required=True,
            reasons=("no changed-file list available",),
        )

    if topology is None:
        topology = get_topology()

    # TODO(#3399): thread build flags/variant through here for superrepos so a
    # baseline built with different flags is not considered reusable. Not needed
    # for the current single-variant multi-arch CI, but required before enabling
    # reuse across superrepo builds that vary build configuration.
    impact = analyze_stage_impact(
        changed_inputs=list(changed_files),
        topology=topology,
        platform=platform,
    )
    return StageReusePlan(
        candidate_stages=tuple(impact.copy_stages),
        rebuild_stages=tuple(impact.rebuild_stages),
        full_rebuild_required=impact.full_rebuild_required,
        reasons=tuple(impact.reasons),
    )


def compute_auto_stage_reuse(
    *,
    changed_files: Sequence[str] | None,
    mode: StageReuseMode,
    linux_amdgpu_families: Sequence[str] = (),
    windows_amdgpu_families: Sequence[str] = (),
    topology: BuildTopology | None = None,
    baseline_selector: BaselineSelector | None = None,
    baseline_selector_factory: Callable[[str], BaselineSelector] | None = None,
    external_rebuild_stages: Sequence[str] | None = None,
) -> AutoStageReuse:
    """Compute auto stage-reuse decisions, verified against a baseline run.
    A stage is only reusable when it is unaffected by the change AND its
    artifacts are present in a healthy baseline run for *every* platform being
    built. The platforms are derived from which family lists are non-empty:
    ``linux_amdgpu_families`` implies the ``linux`` platform, and
    ``windows_amdgpu_families`` implies ``windows``. This guards against the
    case where a stage available only in the Linux baseline is skipped on
    Windows. The report lines are logged before returning.

    For external repos where changed_files is unavailable, set
    ``external_rebuild_stages`` to the list of stages that must rebuild
    (e.g., ["math-libs"]). All other stages become reuse candidates.
    """
    platforms = _build_platforms(linux_amdgpu_families, windows_amdgpu_families)
    if not platforms:
        return _log_and_return(
            _empty_result(
                mode,
                full_rebuild_required=True,
                reasons=("no build platforms selected",),
                report_lines=(
                    f"{LOG_PREFIX} no build platforms selected; "
                    f"automatic stage reuse disabled.",
                ),
            )
        )

    # Read external rebuild stages from env var if not provided
    if external_rebuild_stages is None:
        external_rebuild_stages_raw = os.environ.get(
            "STAGE_REUSE_EXTERNAL_REBUILD_STAGES", ""
        )
        if external_rebuild_stages_raw.strip():
            external_rebuild_stages = [
                s.strip()
                for s in external_rebuild_stages_raw.split(",")
                if s.strip()
            ]
            logger.info(
                "%s external repo mode: rebuild stages from env: %s",
                LOG_PREFIX,
                ", ".join(external_rebuild_stages),
            )

    if topology is None:
        topology = get_topology()

    families = _target_families(linux_amdgpu_families, windows_amdgpu_families)

    plan = plan_stage_reuse(
        changed_files=changed_files,
        platform=platforms[0],
        topology=topology,
        external_rebuild_stages=external_rebuild_stages,
    )
    candidates = plan.candidate_stages
    rebuild = plan.rebuild_stages

    # External repo mode with explicit rebuild stages bypasses the changed_files check
    if changed_files is None and external_rebuild_stages is None:
        return _log_and_return(
            _empty_result(
                mode,
                full_rebuild_required=True,
                reasons=plan.reasons,
                report_lines=(
                    f"{LOG_PREFIX} mode={mode.value}; no changed-file list. "
                    f"Conservatively rebuilding all stages.",
                ),
            )
        )

    if plan.full_rebuild_required or not candidates:
        lines = _format_report(
            mode=mode,
            candidates=candidates,
            rebuild=rebuild,
            full_rebuild_required=plan.full_rebuild_required,
            reasons=plan.reasons,
            baseline_run_id=None,
            available=(),
            unavailable=candidates,
        )
        return _log_and_return(
            AutoStageReuse(
                mode=mode,
                candidate_stages=candidates,
                rebuild_stages=rebuild,
                full_rebuild_required=plan.full_rebuild_required,
                baseline_run_id=None,
                baseline_html_url=None,
                available_stages=(),
                unavailable_stages=candidates,
                applied_reuse_stages=(),
                reasons=plan.reasons,
                report_lines=lines,
            )
        )

    required = _required_artifacts_for_stages(topology, candidates, families)
    # Verify artifact availability independently for each platform. A single
    # ``baseline_selector`` (used by tests) applies to all platforms; otherwise
    # a per-platform selector is built so each platform is checked against a
    # baseline that actually produced that platform's artifacts.
    platform_baseline_run_ids: dict[str, str | None] = {}
    platform_baseline_urls: dict[str, str | None] = {}
    per_platform_available: dict[str, tuple[str, ...]] = {}
    baseline_error: str | None = None

    for platform in platforms:
        if baseline_selector is not None:
            selector = baseline_selector
        elif baseline_selector_factory is not None:
            selector = baseline_selector_factory(platform)
        else:
            selector = _default_baseline_selector(platform=platform)

        # Only transient GitHub API / network failures are tolerated here: a
        # failed baseline lookup falls back to a full rebuild, which is safe.
        # Configuration errors (e.g. a bad required-artifacts request) indicate
        # a bug and must surface, so they are left to propagate.
        try:
            baseline = selector(required)
        except GitHubAPIError as exc:
            baseline_error = str(exc)
            baseline = None

        platform_baseline_run_ids[platform] = (
            baseline.run_id if baseline is not None else None
        )
        platform_baseline_urls[platform] = (
            baseline.html_url if baseline is not None else None
        )

        available_filenames = _matched_filenames(baseline)
        available_here: list[str] = []
        if baseline is not None:
            for stage_name in candidates:
                if _stage_artifacts_available(
                    topology, stage_name, families, available_filenames
                ):
                    available_here.append(stage_name)
        per_platform_available[platform] = tuple(available_here)

    selected_run_ids = {
        run_id for run_id in platform_baseline_run_ids.values() if run_id is not None
    }

    # The workflow can only copy artifacts from a single baseline_run_id, so if
    # platforms resolve to different baseline runs we cannot safely reuse.

    if len(selected_run_ids) > 1:
        return _log_and_return(
            _empty_result(
                mode,
                full_rebuild_required=True,
                reasons=(
                    "automatic reuse resolved different baseline runs per platform",
                ),
                report_lines=(
                    f"{LOG_PREFIX} multiple baseline runs selected across "
                    f"platforms; disabling automatic reuse.",
                ),
            )
        )

    reported_baseline_run_id = next(iter(selected_run_ids), None)
    reported_baseline_url = next(
        (url for url in platform_baseline_urls.values() if url), None
    )

    # A stage is available only when present on EVERY platform being built.
    available: list[str] = []
    unavailable: list[str] = []

    for stage_name in candidates:
        if all(
            stage_name in per_platform_available.get(platform, ())
            for platform in platforms
        ):
            available.append(stage_name)
        else:
            unavailable.append(stage_name)

    available_t = tuple(available)
    unavailable_t = tuple(unavailable)
    applied = available_t if mode is StageReuseMode.REUSE_STAGE else ()

    lines = _format_report(
        mode=mode,
        candidates=candidates,
        rebuild=rebuild,
        full_rebuild_required=False,
        reasons=plan.reasons,
        baseline_run_id=reported_baseline_run_id,
        available=available_t,
        unavailable=unavailable_t,
        baseline_error=baseline_error,
        platforms=platforms,
        platform_available=per_platform_available,
    )
    return _log_and_return(
        AutoStageReuse(
            mode=mode,
            candidate_stages=candidates,
            rebuild_stages=rebuild,
            full_rebuild_required=False,
            baseline_run_id=reported_baseline_run_id,
            baseline_html_url=reported_baseline_url,
            available_stages=available_t,
            unavailable_stages=unavailable_t,
            applied_reuse_stages=applied,
            reasons=plan.reasons,
            report_lines=lines,
            platform_available=per_platform_available,
        )
    )


def _build_platforms(
    linux_amdgpu_families: Sequence[str],
    windows_amdgpu_families: Sequence[str],
) -> tuple[str, ...]:
    """Platforms whose baselines must contain a stage's artifacts to reuse it.

    Automatic reuse merges into the prebuilt_stages that flow to BOTH the Linux
    and Windows build configs, so a stage may only be reused when its artifacts
    are available for every platform actually being built.
    """
    platforms: list[str] = []
    if linux_amdgpu_families:
        platforms.append("linux")
    if windows_amdgpu_families:
        platforms.append("windows")
    return tuple(platforms)


def _default_baseline_selector(*, platform: str) -> BaselineSelector:
    """Build a selector bound to baseline_runs.select_baseline_run.
    ``select_baseline_run`` already requires each candidate run to have healthy
    build jobs (``required_successful_job_name_substrings=("Build",)``) AND to
    contain all requested artifacts. A run with no artifacts (e.g. a docs-only
    change) therefore fails the availability gate and is never selected, so no
    extra "passing build" check is needed here.

    For external repos (rocm-libraries, rocm-systems), set
    ``STAGE_REUSE_BASELINE_REPOSITORY`` to ``ROCm/TheRock`` to query TheRock's
    workflow runs for prebuilt artifacts instead of the external repo's own runs.
    """

    # Default to current repo, but allow override for external repos to use
    # TheRock's baseline runs.
    default_repository = os.environ.get("GITHUB_REPOSITORY", "ROCm/TheRock")
    baseline_repository = os.environ.get("STAGE_REUSE_BASELINE_REPOSITORY", "")
    github_repository = baseline_repository or default_repository
    is_cross_repo = baseline_repository and baseline_repository != default_repository

    branch = os.environ.get("STAGE_REUSE_BASELINE_BRANCH", "main")
    workflow_name = os.environ.get("STAGE_REUSE_BASELINE_WORKFLOW", "multi_arch_ci.yml")
    current_commit_sha = os.environ.get("STAGE_REUSE_CURRENT_SHA") or None
    max_age_hours_raw = os.environ.get("STAGE_REUSE_MAX_AGE_HOURS")
    max_age_hours = float(max_age_hours_raw) if max_age_hours_raw else None
    history_count_raw = os.environ.get("STAGE_REUSE_COMMIT_HISTORY", "50")
    try:
        history_count = max(1, int(history_count_raw))
    except ValueError:
        history_count = 50

    # For cross-repo baseline selection (e.g., rocm-libraries using TheRock's
    # baseline), commit compatibility checking is disabled because the baseline
    # repo's commit history is unrelated to the current repo's commits. We rely
    # solely on recency and artifact availability gates.
    if is_cross_repo:
        logger.info(
            "%s cross-repo baseline: querying %s (current repo: %s); "
            "commit-compatibility rule disabled.",
            LOG_PREFIX,
            github_repository,
            default_repository,
        )
        effective_commit_sha = None
        ordered_commit_shas = None
    else:
        # The commit-compatibility rule needs the branch history (newest-first) to
        # establish ancestry. select_baseline_run only accepts a candidate whose
        # head_sha is `same` or `ancestor` of current_commit_sha; with an EMPTY
        # window every candidate resolves to `unknown` and is rejected, so reuse
        # never activates. Fetch the real history here. If the SHA is set but the
        # history fetch fails (or returns empty), disable the commit rule (pass both
        # as None) rather than enabling it with an empty window -- recency and
        # artifact availability still gate the selection.
        ordered_commit_shas = None
        effective_commit_sha = current_commit_sha
        if current_commit_sha is not None:
            try:
                ordered_commit_shas = github_actions_api.gha_query_recent_branch_commits(
                    github_repository_name=github_repository,
                    branch=branch,
                    max_count=history_count,
                )
            except GitHubAPIError as exc:
                logger.warning(
                    "%s could not fetch branch history (%s); "
                    "skipping commit-compatibility rule.",
                    LOG_PREFIX,
                    exc,
                )
                ordered_commit_shas = None
            if not ordered_commit_shas:
                effective_commit_sha = None
                ordered_commit_shas = None

    # A functools.partial binds the resolved configuration to
    # select_baseline_run; the only free argument is required_artifacts, which
    # matches the BaselineSelector signature.
    return functools.partial(
        _invoke_select_baseline_run,
        github_repository=github_repository,
        workflow_name=workflow_name,
        branch=branch,
        platform=platform,
        current_commit_sha=effective_commit_sha,
        ordered_commit_shas=ordered_commit_shas,
        max_age_hours=max_age_hours,
    )


def _invoke_select_baseline_run(
    required: Sequence[RequiredArtifact], **kwargs
) -> BaselineRun | None:
    """Adapter so a partial can present the BaselineSelector(required) shape.

    Calls through the ``baseline_runs`` module attribute (rather than a bound
    reference) so tests can monkeypatch ``select_baseline_run``.
    """
    return baseline_runs.select_baseline_run(required_artifacts=required, **kwargs)


def _empty_result(
    mode: StageReuseMode,
    *,
    full_rebuild_required: bool = False,
    reasons: tuple[str, ...] = (),
    report_lines: tuple[str, ...] = (),
) -> AutoStageReuse:
    return AutoStageReuse(
        mode=mode,
        candidate_stages=(),
        rebuild_stages=(),
        full_rebuild_required=full_rebuild_required,
        baseline_run_id=None,
        baseline_html_url=None,
        available_stages=(),
        unavailable_stages=(),
        applied_reuse_stages=(),
        reasons=reasons,
        report_lines=report_lines,
    )


def _format_report(
    *,
    mode: StageReuseMode,
    candidates: Sequence[str],
    rebuild: Sequence[str],
    full_rebuild_required: bool,
    reasons: Sequence[str],
    baseline_run_id: str | None,
    available: Sequence[str],
    unavailable: Sequence[str],
    baseline_error: str | None = None,
    platforms: Sequence[str] = (),
    platform_available: dict[str, tuple[str, ...]] | None = None,
) -> tuple[str, ...]:
    platform_available = platform_available or {}
    lines: list[str] = [f"{LOG_PREFIX} mode={mode.value}"]
    if platforms:
        lines.append(f"{LOG_PREFIX} platforms verified: {', '.join(platforms)}")
    if full_rebuild_required:
        lines.append(
            f"{LOG_PREFIX} conservative full rebuild: no stages eligible for reuse."
        )
        for reason in reasons:
            lines.append(f"{LOG_PREFIX}   reason: {reason}")
        return tuple(lines)
    if not candidates:
        lines.append(f"{LOG_PREFIX} no unaffected stages; all stages rebuild.")
        return tuple(lines)
    if baseline_error:
        lines.append(
            f"{LOG_PREFIX} baseline lookup failed ({baseline_error}); "
            f"cannot verify artifacts, rebuilding all candidates."
        )
    elif baseline_run_id:
        lines.append(f"{LOG_PREFIX} baseline run for artifact check: {baseline_run_id}")
    else:
        lines.append(
            f"{LOG_PREFIX} no baseline run contains artifacts for all "
            f"candidate stages; rebuilding all candidates."
        )
    verb = (
        "WILL be skipped" if mode is StageReuseMode.REUSE_STAGE else "WOULD be skipped"
    )
    for stage in available:
        lines.append(
            f"{LOG_PREFIX} stage '{stage}' unaffected AND available in "
            f"baseline on all platforms -> {verb}"
        )
    for stage in unavailable:
        # Call out WHICH platforms are missing the artifacts so the mismatch
        # (e.g. present on linux, absent on windows) is visible in the log.
        if len(platforms) > 1:
            missing = [
                platform
                for platform in platforms
                if stage not in platform_available.get(platform, ())
            ]
            where = f" (missing on: {', '.join(missing)})" if missing else ""
        else:
            where = ""
        lines.append(
            f"{LOG_PREFIX} stage '{stage}' unaffected but artifacts "
            f"NOT available -> rebuild{where}"
        )
    if rebuild:
        lines.append(f"{LOG_PREFIX} stages rebuilding (impacted): {', '.join(rebuild)}")
    if mode is StageReuseMode.DRY_RUN and available:
        lines.append(
            f"{LOG_PREFIX} dry-run: prebuilt_stages NOT modified; all stages "
            f"still build."
        )
    return tuple(lines)


def _format_stage_list(stages: Sequence[str]) -> str:
    """Render a tuple of stage names as backticked, comma-separated markdown."""
    if not stages:
        return "_none_"
    return ", ".join(f"`{stage}`" for stage in stages)


def render_step_summary(result: AutoStageReuse) -> str:
    """Render a GitHub step-summary markdown block for the analysis."""
    baseline = f"`{result.baseline_run_id}`" if result.baseline_run_id else "_none_"
    candidates = _format_stage_list(result.candidate_stages)
    available = _format_stage_list(result.available_stages)
    applied = _format_stage_list(result.applied_reuse_stages)

    out = ["### Stage reuse analysis", ""]
    out.append(f"- mode: `{result.mode.value}`")
    out.append(f"- full rebuild required: `{result.full_rebuild_required}`")
    out.append(f"- baseline run checked: {baseline}")
    out.append(f"- unaffected candidates: {candidates}")
    out.append(f"- available in baseline: {available}")
    out.append(f"- applied: {applied}")
    if result.platform_available:
        out.append("- available per platform:")
        for platform, stages in result.platform_available.items():
            out.append(f"  - {platform}: {_format_stage_list(stages)}")
    if result.reasons:
        out.append("- reasons:")
        for reason in result.reasons:
            out.append(f"  - {reason}")
    if result.mode is StageReuseMode.DRY_RUN and result.available_stages:
        out.append("")
        out.append(
            "> Dry-run only: no build steps were skipped. Artifacts were "
            "verified against the baseline run above. Set "
            "`STAGE_REUSE_MODE=reuse-stage` after review to enable skipping."
        )
    return "\n".join(out)


def _log_and_return(result: AutoStageReuse) -> AutoStageReuse:
    """Emit the analysis report lines to the module logger and return result."""
    for line in result.report_lines:
        logger.info(line)
    return result
