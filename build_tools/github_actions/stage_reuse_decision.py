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
from typing import Sequence


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


@dataclass(frozen=True)
class AutoStageReuse:
    """Result of the auto stage-reuse analysis.

    Attributes:
        mode: The mode this analysis ran under.
        would_reuse_stages: Stages stage-impact says are unaffected and could
            be satisfied by prebuilt artifacts.
        rebuild_stages: Stages that must rebuild.
        full_rebuild_required: True when the conservative fallback fired
            (e.g. build tooling / workflow / topology change).
        reasons:  Reasons from the impact analysis.
        applied_reuse_stages: Stages actually returned for application. Empty
            unless mode == ENFORCE.
        report_lines: Console / step-summary lines describing the decision.
    """

    mode: StageReuseMode
    would_reuse_stages: tuple[str, ...]
    rebuild_stages: tuple[str, ...]
    full_rebuild_required: bool
    reasons: tuple[str, ...]
    applied_reuse_stages: tuple[str, ...]
    report_lines: tuple[str, ...] = field(default_factory=tuple)

LOG_PREFIX = "[STAGE-REUSE]"


def compute_auto_stage_reuse(
    *,
    changed_files: Sequence[str] | None,
    mode: StageReuseMode,
    platform: str | None = None,
    topology=None,
) -> AutoStageReuse:
    """Compute auto stage-reuse decisions from changed files.

    Args:
        changed_files: Repo-relative changed paths. ``None`` (schedule /
            workflow_dispatch with no diff) is treated as "cannot narrow scope"
            -> conservative full rebuild, nothing reused.
        mode: The stage-reuse mode switch.
        platform: Optional platform filter passed to stage-impact.
        topology: Optional BuildTopology (injected for testing).

    Returns:
        An AutoStageReuse. ``applied_reuse_stages`` is non-empty only when
        ``mode == ENFORCE``.
    """
    # No diff available -> cannot safely narrow scope -> rebuild everything.
    if changed_files is None:
        lines = (
            f"{LOG_PREFIX} mode={mode.value}; no changed-file list available "
            f"(schedule/dispatch). Conservatively rebuilding all stages.",
        )
        return AutoStageReuse(
            mode=mode,
            would_reuse_stages=(),
            rebuild_stages=(),
            full_rebuild_required=True,
            reasons=("no changed-file list available",),
            applied_reuse_stages=(),
            report_lines=lines,
        )

    # Import here so the rest of the module don't require the
    # topology stack to be importable.
    from stage_impact import analyze_stage_impact

    impact = analyze_stage_impact(
        changed_inputs=list(changed_files),
        topology=topology,
        platform=platform,
    )

    would_reuse = tuple(impact.copy_stages)
    rebuild = tuple(impact.rebuild_stages)

    applied = would_reuse if mode is StageReuseMode.ENFORCE else ()

    report_lines = _format_report(
        mode=mode,
        would_reuse=would_reuse,
        rebuild=rebuild,
        full_rebuild_required=impact.full_rebuild_required,
        reasons=tuple(impact.reasons),
        applied=applied,
    )

    return AutoStageReuse(
        mode=mode,
        would_reuse_stages=would_reuse,
        rebuild_stages=rebuild,
        full_rebuild_required=impact.full_rebuild_required,
        reasons=tuple(impact.reasons),
        applied_reuse_stages=applied,
        report_lines=report_lines,
    )


def _format_report(
    *,
    mode: StageReuseMode,
    would_reuse: tuple[str, ...],
    rebuild: tuple[str, ...],
    full_rebuild_required: bool,
    reasons: tuple[str, ...],
    applied: tuple[str, ...],
) -> tuple[str, ...]:
    lines: list[str] = []
    verb = "WILL be skipped (enforced)" if mode is StageReuseMode.ENFORCE else "WOULD be skipped (dry-run)"
    lines.append(f"{LOG_PREFIX} mode={mode.value}")
    if full_rebuild_required:
        lines.append(
            f"{LOG_PREFIX} conservative full rebuild: no stages eligible for reuse."
        )
        for reason in reasons:
            lines.append(f"{LOG_PREFIX}   reason: {reason}")
        return tuple(lines)

    if not would_reuse:
        lines.append(f"{LOG_PREFIX} no stages eligible for reuse; all stages rebuild.")
        return tuple(lines)

    for stage in would_reuse:
        lines.append(f"{LOG_PREFIX} stage '{stage}' {verb}")
    if rebuild:
        lines.append(f"{LOG_PREFIX} stages rebuilding: {', '.join(rebuild)}")
    if mode is StageReuseMode.DRY_RUN:
        lines.append(
            f"{LOG_PREFIX} dry-run: prebuilt_stages NOT modified; "
            f"all stages still build. Set STAGE_REUSE_MODE=skip-stage to apply."
        )
    return tuple(lines)


def render_step_summary(result: AutoStageReuse) -> str:
    """Render a GitHub step-summary markdown block for the analysis."""
    out = ["### Stage reuse analysis", ""]
    out.append(f"- mode: `{result.mode.value}`")
    out.append(f"- full rebuild required: `{result.full_rebuild_required}`")
    out.append(
        f"- would reuse: "
        + (", ".join(f"`{s}`" for s in result.would_reuse_stages) or "_none_")
    )
    out.append(
        f"- applied: "
        + (", ".join(f"`{s}`" for s in result.applied_reuse_stages) or "_none_")
    )
    if result.reasons:
        out.append("- reasons:")
        for reason in result.reasons:
            out.append(f"  - {reason}")
    if result.mode is StageReuseMode.DRY_RUN and result.would_reuse_stages:
        out.append("")
        out.append(
            "> Dry-run only: no build steps were skipped. "
            "Set `STAGE_REUSE_MODE=skip-stage` to enable skipping."
        )
    return "\n".join(out)
