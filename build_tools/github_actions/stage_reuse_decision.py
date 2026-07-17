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
* ``STAGE_REUSE_BASELINE_BRANCH``  - baseline branch to search (default ``main``).
* ``STAGE_REUSE_BASELINE_WORKFLOW`` - baseline workflow file
                                   (default ``multi_arch_ci.yml``).
* ``STAGE_REUSE_CURRENT_SHA``    - current commit SHA; enables the
                                   commit-compatibility rule when set.
* ``STAGE_REUSE_MAX_AGE_HOURS``  - recency window in hours; disables the recency
                                   rule when unset.
* ``STAGE_REUSE_COMMIT_HISTORY`` - number of branch commits to fetch for
                                   ancestry (default ``50``).
"""

from __future__ import annotations

import enum
import os
import logging
import functools
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence


logger = logging.getLogger(__name__)


class StageReuseMode(enum.Enum):
    DRY_RUN = "dry-run"
    ENFORCE = "reuse-stage"

    @staticmethod
    def from_environ(default: "StageReuseMode" = None) -> "StageReuseMode":
        """Read STAGE_REUSE_MODE; default to dry-run when unset/invalid."""
        default = default or StageReuseMode.DRY_RUN
        raw = (os.environ.get("STAGE_REUSE_MODE", "") or "").strip().lower()
        return _STAGE_REUSE_MODE_ALIASES.get(raw, default)


# Accepted spellings for each mode. ``skip-stage`` is the legacy alias that
# predates the ``reuse-stage`` rename and is kept for backwards compatibility
# with any callers still passing the old value.
_STAGE_REUSE_MODE_ALIASES = {
    "dry-run": StageReuseMode.DRY_RUN,
    "reuse-stage": StageReuseMode.ENFORCE,
    "skip-stage": StageReuseMode.ENFORCE,
}


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


BaselineSelector = Callable[[Sequence["object"]], Optional["object"]]


@dataclass(frozen=True)
class AutoStageReuse:
    """Result of the auto stage-reuse analysis (planning + verification)."""

    mode: StageReuseMode
    candidate_stages: tuple[str, ...]
    rebuild_stages: tuple[str, ...]
    full_rebuild_required: bool
    baseline_run_id: Optional[str]
    baseline_html_url: Optional[str]
    available_stages: tuple[str, ...]
    unavailable_stages: tuple[str, ...]
    applied_reuse_stages: tuple[str, ...]
    reasons: tuple[str, ...]
    report_lines: tuple[str, ...] = field(default_factory=tuple)
    platform_available: dict[str, tuple[str, ...]] = field(default_factory=dict)
    stage_artifacts: dict[str, tuple[str, ...]] = field(default_factory=dict)


LOG_PREFIX = "[STAGE-REUSE]"


def stage_reuse_target_families(
    linux_amdgpu_families: Sequence[str] = (),
    windows_amdgpu_families: Sequence[str] = (),
) -> list[str]:
    """Family list whose artifacts must be verified for stage reuse.
    Lives here (next to the reuse implementation) rather than in the caller,
    since it only exists to prepare inputs for ``compute_auto_stage_reuse``.
    Always includes ``generic`` because every stage produces a generic archive.
    """
    families = list(dict.fromkeys([*linux_amdgpu_families, *windows_amdgpu_families]))
    if "generic" not in families:
        families.append("generic")
    return families


def required_artifacts_for_stages(topology, stage_names, target_families):
    """Artifact/family pairs the given stages produce."""
    from baseline_runs import RequiredArtifact

    artifacts_by_group = topology.get_artifact_group_to_artifacts()
    required: list = []
    seen: set = set()
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


def _matched_filenames(baseline) -> set:
    if baseline is None:
        return set()
    return set(baseline.artifact_availability.matched_filenames)


def _stage_artifacts_available(
    topology, stage_name, target_families, available_filenames
):
    """True when every artifact this stage produces has an archive present."""
    from artifact_manager import ARTIFACT_COMPONENTS
    from _therock_utils.artifact_backend import ARTIFACT_EXTENSIONS
    artifacts_by_group = topology.get_artifact_group_to_artifacts()
    stage = topology.build_stages.get(stage_name)
    if stage is None:
        return False
    for group_name in stage.artifact_groups:
        for artifact_name in artifacts_by_group.get(group_name, []):
            for family in target_families:
                found = False
                for artifact in ARTIFACT_COMPONENTS:
                    for extension in ARTIFACT_EXTENSIONS:
                        filename = f"{artifact_name}_{artifact}_{family}{extension}"
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
    topology=None,
) -> StageReusePlan:
    """Planning step: which stages are unaffected reuse candidates.

    Pure decision logic -- no baseline selection, artifact verification, or
    reporting -- so it can be reused independently of the CI plumbing.
    """

    if changed_files is None:
        return StageReusePlan(
            candidate_stages=(),
            rebuild_stages=(),
            full_rebuild_required=True,
            reasons=("no changed-file list available",),
        )

    if topology is None:
        from _therock_utils.build_topology import get_topology

        topology = get_topology()

    from stage_impact import analyze_stage_impact

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
    platform: str | None = None,
    platforms: Sequence[str] | None = None,
    target_families: Sequence[str] = (),
    topology=None,
    baseline_selector: BaselineSelector | None = None,
    baseline_selector_factory: Callable[[str], BaselineSelector] | None = None,
) -> AutoStageReuse:
    """Compute auto stage-reuse decisions, verified against a baseline run.
    A stage is only reusable when it is unaffected by the change AND its
    artifacts are present in a healthy baseline run for *every* platform in
    ``platforms``. This guards against the case flagged in review where the
    resulting ``prebuilt_stages`` are applied to both Linux and Windows build
    configs: a stage available only in the Linux baseline must not be skipped
    on Windows. ``platform`` remains accepted as a single-platform shorthand
    for backwards compatibility.
    """
    if platforms is not None and len(platforms) == 0 and platform is None:
        return _empty_result(
            mode,
            full_rebuild_required=True,
            reasons=("no build platforms selected",),
            report_lines=(
                f"{LOG_PREFIX} no build platforms selected; automatic stage reuse disabled.",
            ),
        )

    # Resolve the platform set. ``platforms`` wins; fall back to the legacy
    # single ``platform`` arg; default to linux only when neither is given.
    resolved_platforms = _resolve_platforms(platforms, platform)

    if topology is None and changed_files is not None:
        from _therock_utils.build_topology import get_topology

        topology = get_topology()

    plan = plan_stage_reuse(
        changed_files=changed_files,
        platform=resolved_platforms[0],
        topology=topology,
    )
    candidates = plan.candidate_stages
    rebuild = plan.rebuild_stages
    families = tuple(target_families) or ("generic",)
    stage_artifacts = (
        {stage: _stage_artifact_names(topology, stage) for stage in candidates}
        if topology is not None
        else {}
    )

    if changed_files is None:
        return _empty_result(
            mode,
            full_rebuild_required=True,
            reasons=plan.reasons,
            report_lines=(
                f"{LOG_PREFIX} mode={mode.value}; no changed-file list. "
                f"Conservatively rebuilding all stages.",
            ),
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
            stage_artifacts=stage_artifacts,
        )
        return AutoStageReuse(
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
            stage_artifacts=stage_artifacts,
        )

    required = required_artifacts_for_stages(topology, candidates, families)
    # Verify artifact availability independently for each platform. A single
    # ``baseline_selector`` (used by tests) applies to all platforms; otherwise
    # a per-platform selector is built so each platform is checked against a
    # baseline that actually produced that platform's artifacts.
    baseline_error: Optional[str] = None
    per_platform_available: dict[str, tuple[str, ...]] = {}
    # The reported baseline run id/url comes from the first platform's baseline
    # (they share commit/recency gates); per-platform detail lives in
    # platform_available.
    reported_baseline_run_id: Optional[str] = None
    reported_baseline_url: Optional[str] = None

    platform_baseline_run_ids: dict[str, Optional[str]] = {}
    platform_baseline_urls: dict[str, Optional[str]] = {}
    per_platform_available: dict[str, tuple[str, ...]] = {}
    baseline_error: Optional[str] = None

    for plat in resolved_platforms:
        baseline = None
        try:
            if baseline_selector is not None:
                selector = baseline_selector
            elif baseline_selector_factory is not None:
                selector = baseline_selector_factory(plat)
            else:
                selector = _default_baseline_selector(platform=plat)
            baseline = selector(required)
        except Exception as exc:
            baseline_error = str(exc)

        platform_baseline_run_ids[plat] = (
            baseline.run_id if baseline is not None else None
        )
        platform_baseline_urls[plat] = (
            baseline.html_url if baseline is not None else None
        )

        available_filenames = _matched_filenames(baseline)
        avail_here: list[str] = []
        if baseline is not None:
            for stage_name in candidates:
                if _stage_artifacts_available(
                    topology, stage_name, families, available_filenames
                ):
                    avail_here.append(stage_name)
        per_platform_available[plat] = tuple(avail_here)

    selected_run_ids = {
        run_id for run_id in platform_baseline_run_ids.values() if run_id is not None
    }

    if len(selected_run_ids) > 1:
        return _empty_result(
            mode,
            full_rebuild_required=True,
            reasons=("automatic reuse resolved different baseline runs per platform",),
            report_lines=(
                f"{LOG_PREFIX} multiple baseline runs selected across platforms; disabling automatic reuse.",
            ),
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
            stage_name in per_platform_available.get(plat, ())
            for plat in resolved_platforms
        ):
            available.append(stage_name)
        else:
            unavailable.append(stage_name)

    available_t = tuple(available)
    unavailable_t = tuple(unavailable)
    applied = available_t if mode is StageReuseMode.ENFORCE else ()

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
        platforms=resolved_platforms,
        platform_available=per_platform_available,
        stage_artifacts=stage_artifacts,
    )
    return AutoStageReuse(
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
        stage_artifacts=stage_artifacts,
    )


def _resolve_platforms(
    platforms: Sequence[str] | None, platform: str | None
) -> tuple[str, ...]:
    """Normalize the platform inputs to a de-duplicated, ordered tuple."""
    if platforms:
        resolved = tuple(dict.fromkeys(platforms))
    elif platform:
        resolved = (platform,)
    else:
        resolved = ("linux",)
    return resolved


def _default_baseline_selector(*, platform: str | None) -> BaselineSelector:
    """Build a selector bound to baseline_runs.select_baseline_run.
    ``select_baseline_run`` already requires each candidate run to have
    healthy build jobs (``required_successful_job_name_substrings=("Build",)``)
    AND to contain all requested artifacts. A run with no artifacts (e.g. a
    docs-only change) therefore fails the availability gate and is never
    selected, so no extra "passing build" check is needed here.
    """
    from baseline_runs import select_baseline_run

    github_repository = os.environ.get("GITHUB_REPOSITORY", "ROCm/TheRock")
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
            from github_actions_api import gha_query_recent_branch_commits

            ordered_commit_shas = gha_query_recent_branch_commits(
                github_repository_name=github_repository,
                branch=branch,
                max_count=history_count,
            )
            if not ordered_commit_shas:
                effective_commit_sha = None
                ordered_commit_shas = None
        except Exception as exc:
            logger.warning(
                "%s could not fetch branch history (%s); "
                "skipping commit-compatibility rule.",
                LOG_PREFIX,
                exc,
            )
            effective_commit_sha = None
            ordered_commit_shas = None

    # A functools.partial binds the resolved configuration to
    # select_baseline_run; the only free argument is required_artifacts, which
    # matches the BaselineSelector signature.
    return functools.partial(
        _invoke_select_baseline_run,
        select_baseline_run=select_baseline_run,
        github_repository=github_repository,
        workflow_name=workflow_name,
        branch=branch,
        platform=platform or "linux",
        current_commit_sha=effective_commit_sha,
        ordered_commit_shas=ordered_commit_shas,
        max_age_hours=max_age_hours,
    )


def _invoke_select_baseline_run(required, *, select_baseline_run, **kwargs):
    """Adapter so a partial can present the BaselineSelector(required) shape."""
    return select_baseline_run(required_artifacts=required, **kwargs)


def _empty_result(mode, *, full_rebuild_required=False, reasons=(), report_lines=()):
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
    mode,
    candidates,
    rebuild,
    full_rebuild_required,
    reasons,
    baseline_run_id,
    available,
    unavailable,
    baseline_error=None,
    platforms: Sequence[str] = (),
    platform_available: dict[str, tuple[str, ...]] | None = None,
    stage_artifacts: dict[str, tuple[str, ...]] | None = None,
):
    platform_available = platform_available or {}
    stage_artifacts = stage_artifacts or {}
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

    verb = "WILL be skipped" if mode is StageReuseMode.ENFORCE else "WOULD be skipped"

    for plat in platforms or ("linux", "windows"):
        plat_available = platform_available.get(plat, ())
        plat_unavailable = tuple(
            stage for stage in candidates if stage not in plat_available
        )

        lines.append(f"{LOG_PREFIX} platform={plat}")
        for stage in plat_available:
            lines.append(
                f"{LOG_PREFIX}   stage '{stage}' unaffected AND available in "
                f"baseline -> {verb}"
            )
        for stage in plat_unavailable:
            lines.append(
                f"{LOG_PREFIX}   stage '{stage}' unaffected but artifacts "
                f"NOT available -> rebuild"
            )

    if stage_artifacts:
        for stage in candidates:
            artifacts = stage_artifacts.get(stage, ())
            if not artifacts:
                continue
            lines.append(
                f"{LOG_PREFIX}   stage '{stage}' builds: "
                f"{', '.join(f'`{artifact}`' for artifact in artifacts)}"
            )

    if rebuild:
        lines.append(f"{LOG_PREFIX} stages rebuilding (impacted): {', '.join(rebuild)}")
    if mode is StageReuseMode.DRY_RUN and available:
        lines.append(
            f"{LOG_PREFIX} dry-run: prebuilt_stages NOT modified; all stages "
            f"still build."
        )
    return tuple(lines)


def _format_stage_list(stages: tuple[str, ...]) -> str:
    """Render a tuple of stage names as backticked, comma-separated markdown."""
    if not stages:
        return "_none_"
    return ", ".join(f"`{stage}`" for stage in stages)


def render_step_summary(result: AutoStageReuse) -> str:
    """Render a GitHub step-summary markdown block for the analysis."""
    platforms = tuple(result.platform_available) or ("linux", "windows")

    lines: list[str] = []
    for plat in platforms:
        platform_stages = result.platform_available.get(plat, ())
        baseline = f"`{result.baseline_run_id}`" if result.baseline_run_id else "_none_"
        candidates = _format_stage_list(result.candidate_stages)
        available = _format_stage_list(platform_stages)
        unavailable = _format_stage_list(
            tuple(
                stage
                for stage in result.candidate_stages
                if stage not in platform_stages
            )
        )
        applied = _format_stage_list(
            tuple(
                stage
                for stage in result.applied_reuse_stages
                if stage in platform_stages
            )
        )

        lines.append(f"### Stage reuse analysis ({plat})")
        lines.append("")
        lines.append(f"- mode: `{result.mode.value}`")
        lines.append(f"- full rebuild required: `{result.full_rebuild_required}`")
        lines.append(f"- baseline run checked: {baseline}")
        lines.append(f"- unaffected candidates: {candidates}")
        lines.append(f"- available in baseline: {available}")
        lines.append(f"- not available on this platform: {unavailable}")
        lines.append(f"- applied: {applied}")
        if result.stage_artifacts:
            lines.append("- stage artifacts:")
            for stage in result.candidate_stages:
                artifacts = result.stage_artifacts.get(stage, ())
                if not artifacts:
                    continue
                artifact_list = ", ".join(f"`{artifact}`" for artifact in artifacts)
                lines.append(f"  - `{stage}`: {artifact_list} ")
        if result.reasons:
            lines.append("- reasons:")
            for reason in result.reasons:
                lines.append(f"  - {reason}")
        if result.mode is StageReuseMode.DRY_RUN and result.available_stages:
            lines.append("")
            lines.append(
                "> Dry-run only: no build steps were skipped. Artifacts were "
                "verified against the baseline run above. Set "
                "`STAGE_REUSE_MODE=reuse-stage` after review to enable skipping."
            )
        lines.append("")

    return "\n".join(lines).rstrip()


def _stage_artifact_names(topology, stage_name: str) -> tuple[str, ...]:
    """Return the artifact names produced by a stage."""
    stage = topology.build_stages.get(stage_name)
    if stage is None:
        return ()

    artifacts_by_group = topology.get_artifact_group_to_artifacts()
    names: list[str] = []
    seen: set[str] = set()

    for group_name in stage.artifact_groups:
        for artifact_name in artifacts_by_group.get(group_name, []):
            if artifact_name not in seen:
                seen.add(artifact_name)
                names.append(artifact_name)

    return tuple(names)


def log_report(result: AutoStageReuse) -> None:
    """Emit the analysis report lines to the module logger."""
    for line in result.report_lines:
        logger.info(line)
