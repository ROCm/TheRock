#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Auto-compute per-stage rebuild/reuse decisions for multi-arch CI.

This script decides per build stage, whether the stage can be satisfied with prebuilt
artifacts (reuse) instead of being rebuilt.

---------------------------------
This module is wired into CI behind a two-way mode switch so it can be
observed before it changes anything:
* ``dry-run``  - DEFAULT. Compute the analysis and PRINT, for each stage that
                 *would* be skipped, a line to the console + step summary, but
                 return NO auto stages, so ``prebuilt_stages`` is unchanged and
                 every stage still builds exactly.
* ``skip-stage``  - Compute the analysis and actually return the reuse stages so
                    the orchestrator copies their artifacts and skips the build.

"""

from __future__ import annotations

import enum
import os
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence


class StageReuseMode(enum.Enum):
    DRY_RUN = "dry-run"
    ENFORCE = "skip-stage"

    @staticmethod
    def from_environ(default: "StageReuseMode" = None) -> "StageReuseMode":
        """Read STAGE_REUSE_MODE; default to dry-run when unset/invalid."""
        default = default or StageReuseMode.DRY_RUN
        raw = (os.environ.get("STAGE_REUSE_MODE", "") or "").strip().lower()
        for mode in StageReuseMode:
            if raw == mode.value:
                return mode
        return default


BaselineSelector = Callable[[Sequence["object"]], Optional["object"]]


@dataclass(frozen=True)
class AutoStageReuse:
    """Result of the auto stage-reuse analysis."""

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


LOG_PREFIX = "[STAGE-REUSE]"


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


def _stage_artifacts_available(topology, stage_name, target_families, available_filenames):
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


def compute_auto_stage_reuse(
    *,
    changed_files: Sequence[str] | None,
    mode: StageReuseMode,
    platform: str | None = None,
    target_families: Sequence[str] = (),
    topology=None,
    baseline_selector: BaselineSelector | None = None,
) -> AutoStageReuse:
    """Compute auto stage-reuse decisions, verified against a baseline run."""
    if changed_files is None:
        return _empty_result(
            mode,
            full_rebuild_required=True,
            reasons=("no changed-file list available",),
            report_lines=(
                f"{LOG_PREFIX} mode={mode.value}; no changed-file list "
                f"Conservatively rebuilding all stages.",
            ),
        )

    if topology is None:
        from _therock_utils.build_topology import get_topology

        topology = get_topology()

    from stage_impact import analyze_stage_impact

    impact = analyze_stage_impact(
        changed_inputs=list(changed_files),
        topology=topology,
        platform=platform,
    )
    candidates = tuple(impact.copy_stages)
    rebuild = tuple(impact.rebuild_stages)
    families = tuple(target_families) or ("generic",)

    if impact.full_rebuild_required or not candidates:
        lines = _format_report(
            mode=mode, candidates=candidates, rebuild=rebuild,
            full_rebuild_required=impact.full_rebuild_required,
            reasons=tuple(impact.reasons), baseline_run_id=None,
            available=(), unavailable=candidates,
        )
        return AutoStageReuse(
            mode=mode, candidate_stages=candidates, rebuild_stages=rebuild,
            full_rebuild_required=impact.full_rebuild_required,
            baseline_run_id=None, baseline_html_url=None,
            available_stages=(), unavailable_stages=candidates,
            applied_reuse_stages=(), reasons=tuple(impact.reasons),
            report_lines=lines,
        )

    required = required_artifacts_for_stages(topology, candidates, families)
    baseline = None
    baseline_error: Optional[str] = None
    try:
        if baseline_selector is None:
            baseline_selector = _default_baseline_selector(platform=platform)
        baseline = baseline_selector(required)
    except Exception as exc:
        baseline_error = str(exc)

    available_filenames = _matched_filenames(baseline)
    available: list[str] = []
    unavailable: list[str] = []
    if baseline is not None:
        for stage_name in candidates:
            if _stage_artifacts_available(topology, stage_name, families, available_filenames):
                available.append(stage_name)
            else:
                unavailable.append(stage_name)
    else:
        unavailable = list(candidates)

    available_t = tuple(available)
    unavailable_t = tuple(unavailable)
    applied = available_t if mode is StageReuseMode.ENFORCE else ()
    baseline_run_id = baseline.run_id if baseline is not None else None
    baseline_url = baseline.html_url if baseline is not None else None

    lines = _format_report(
        mode=mode, candidates=candidates, rebuild=rebuild,
        full_rebuild_required=False, reasons=tuple(impact.reasons),
        baseline_run_id=baseline_run_id, available=available_t,
        unavailable=unavailable_t, baseline_error=baseline_error,
    )
    return AutoStageReuse(
        mode=mode, candidate_stages=candidates, rebuild_stages=rebuild,
        full_rebuild_required=False, baseline_run_id=baseline_run_id,
        baseline_html_url=baseline_url, available_stages=available_t,
        unavailable_stages=unavailable_t, applied_reuse_stages=applied,
        reasons=tuple(impact.reasons), report_lines=lines,
    )


def _default_baseline_selector(*, platform: str | None) -> BaselineSelector:
    """Build a selector bound to baseline_runs.select_baseline_run."""
    from baseline_runs import select_baseline_run

    github_repository = os.environ.get("GITHUB_REPOSITORY", "ROCm/TheRock")
    branch = os.environ.get("STAGE_REUSE_BASELINE_BRANCH", "main")
    workflow_name = os.environ.get("STAGE_REUSE_BASELINE_WORKFLOW", "multi_arch_ci.yml")
    current_commit_sha = os.environ.get("STAGE_REUSE_CURRENT_SHA") or None
    max_age_hours_raw = os.environ.get("STAGE_REUSE_MAX_AGE_HOURS")
    max_age_hours = float(max_age_hours_raw) if max_age_hours_raw else None

    def _select(required):
        return select_baseline_run(
            required_artifacts=required,
            github_repository=github_repository,
            workflow_name=workflow_name,
            branch=branch,
            platform=platform or "linux",
            current_commit_sha=current_commit_sha,
            ordered_commit_shas=None if current_commit_sha is None else [],
            max_age_hours=max_age_hours,
        )

    return _select


def _empty_result(mode, *, full_rebuild_required=False, reasons=(), report_lines=()):
    return AutoStageReuse(
        mode=mode, candidate_stages=(), rebuild_stages=(),
        full_rebuild_required=full_rebuild_required,
        baseline_run_id=None, baseline_html_url=None,
        available_stages=(), unavailable_stages=(),
        applied_reuse_stages=(), reasons=reasons, report_lines=report_lines,
    )


def _format_report(*, mode, candidates, rebuild, full_rebuild_required,
                   reasons, baseline_run_id, available, unavailable, baseline_error=None):
    lines: list[str] = [f"{LOG_PREFIX} mode={mode.value}"]
    if full_rebuild_required:
        lines.append(f"{LOG_PREFIX} conservative full rebuild: no stages eligible for reuse.")
        for reason in reasons:
            lines.append(f"{LOG_PREFIX}   reason: {reason}")
        return tuple(lines)
    if not candidates:
        lines.append(f"{LOG_PREFIX} no unaffected stages; all stages rebuild.")
        return tuple(lines)
    if baseline_error:
        lines.append(f"{LOG_PREFIX} baseline lookup failed ({baseline_error}); "
                     f"cannot verify artifacts, rebuilding all candidates.")
    elif baseline_run_id:
        lines.append(f"{LOG_PREFIX} baseline run for artifact check: {baseline_run_id}")
    else:
        lines.append(f"{LOG_PREFIX} no baseline run contains artifacts for all "
                     f"candidate stages; rebuilding all candidates.")
    verb = ("WILL be skipped" if mode is StageReuseMode.ENFORCE
            else "WOULD be skipped")
    for stage in available:
        lines.append(f"{LOG_PREFIX} stage '{stage}' unaffected AND available in baseline -> {verb}")
    for stage in unavailable:
        lines.append(f"{LOG_PREFIX} stage '{stage}' unaffected but artifacts NOT available -> rebuild")
    if rebuild:
        lines.append(f"{LOG_PREFIX} stages rebuilding (impacted): {', '.join(rebuild)}")
    if mode is StageReuseMode.DRY_RUN and available:
        lines.append(f"{LOG_PREFIX} dry-run: prebuilt_stages NOT modified; all stages "
                     f"still build.")
    return tuple(lines)


def render_step_summary(result: AutoStageReuse) -> str:
    """Render a GitHub step-summary markdown block for the analysis."""
    out = ["### Stage reuse analysis", ""]
    out.append(f"- mode: `{result.mode.value}`")
    out.append(f"- full rebuild required: `{result.full_rebuild_required}`")
    out.append("- baseline run checked: "
                (f"`{result.baseline_run_id}`" if result.baseline_run_id else "_none_"))
    out.append("- unaffected candidates: "
                (", ".join(f"`{s}`" for s in result.candidate_stages) or "_none_"))
    out.append("- available in baseline: "
                (", ".join(f"`{s}`" for s in result.available_stages) or "_none_"))
    out.append("- applied: "
                (", ".join(f"`{s}`" for s in result.applied_reuse_stages) or "_none_"))
    if result.reasons:
        out.append("- reasons:")
        for reason in result.reasons:
            out.append(f"  - {reason}")
    if result.mode is StageReuseMode.DRY_RUN and result.available_stages:
        out.append("")
        out.append("> Dry-run only: no build steps were skipped. Artifacts were "
                   "verified against the baseline run above. Set "
                   "`STAGE_REUSE_MODE=skip-stage`.")
    return "\n".join(out)