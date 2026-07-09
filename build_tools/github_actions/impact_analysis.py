#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Impact analysis for multi-arch CI.

This module translates build-stage impact into a test-component impact plan.

The intent is conservative:
- if the change is ambiguous or a full rebuild is already required, keep the
  test matrix broad;
- otherwise, derive a dry-run filter that explains which test components appear
  affected and which look safe to skip.

The workplan expects this to start as dry-run/reporting only, with opt-in
filtering later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

ALL_TEST_COMPONENTS: tuple[str, ...] = (
    "hip-tests",
    "rocrtst",
    "rocprofiler-sdk",
    "rocprofiler-compute",
    "rocprofiler-systems",
    "aqlprofile",
    "rocgdb",
    "rocr-debug-agent",
    "rocdecode",
    "rocjpeg",
    "rccl",
)

# Conservative, name-based rules. These intentionally over-approximate.
_STAGE_COMPONENT_RULES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (
        ("compiler-runtime", "hipify"),
        ("hip-tests", "rocrtst", "rocgdb", "rocr-debug-agent"),
    ),
    (("debug-tools", "rocgdb"), ("rocgdb", "rocr-debug-agent")),
    (
        ("profiler", "rocprofiler"),
        ("rocprofiler-sdk", "rocprofiler-compute", "rocprofiler-systems", "aqlprofile"),
    ),
    (("media-libs", "rocdecode", "rocjpeg"), ("rocdecode", "rocjpeg")),
    (("comm-libs", "rccl"), ("rccl",)),
    (
        ("math-libs", "blas", "rocblas", "hipblas"),
        ("rocprofiler-sdk", "rocprofiler-compute", "aqlprofile"),
    ),
    (("storage-libs",), ()),
)

_ARTIFACT_COMPONENT_RULES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (
        ("base", "amd-llvm", "hipify", "core-runtime", "core-hip", "core-amdsmi"),
        ("hip-tests", "rocrtst", "rocgdb", "rocr-debug-agent"),
    ),
    (
        ("amd-dbgapi", "rocgdb", "rocr-debug-agent"),
        ("rocgdb", "rocr-debug-agent"),
    ),
    (
        ("rocprofiler-sdk", "rocprofiler-compute", "rocprofiler-systems", "aqlprofile"),
        ("rocprofiler-sdk", "rocprofiler-compute", "rocprofiler-systems", "aqlprofile"),
    ),
    (("rocdecode", "rocjpeg"), ("rocdecode", "rocjpeg")),
    (("rccl", "rocshmem"), ("rccl",)),
    (
        ("blas", "rocblas", "hipblas", "math-libs", "miopen"),
        ("rocprofiler-sdk", "rocprofiler-compute", "aqlprofile"),
    ),
)

_SOURCE_SET_RULES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("core", "runtime", "compiler"), ("hip-tests", "rocrtst")),
    (("debug", "dbg"), ("rocgdb", "rocr-debug-agent")),
    (
        ("profiler", "trace"),
        ("rocprofiler-sdk", "rocprofiler-compute", "rocprofiler-systems", "aqlprofile"),
    ),
    (("media",), ("rocdecode", "rocjpeg")),
    (("comm", "communication"), ("rccl",)),
    (
        ("math", "blas", "rocblas", "hipblas"),
        ("rocprofiler-sdk", "rocprofiler-compute", "aqlprofile"),
    ),
)


@dataclass(frozen=True)
class ImpactAnalysisPlan:
    changed_paths: tuple[str, ...]
    affected_source_sets: tuple[str, ...]
    affected_test_components: tuple[str, ...]
    skipped_test_components: tuple[str, ...]
    selected_test_components: tuple[str, ...]
    full_rebuild_required: bool
    reasons: tuple[str, ...]
    report_lines: tuple[str, ...] = field(default_factory=tuple)


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _unique_ordered(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return tuple(out)


def _matches_any(name: str, needles: Sequence[str]) -> bool:
    normalized = _normalize_name(name)
    return any(_normalize_name(needle) in normalized for needle in needles)


def _rule_components(
    name: str, rules: Sequence[tuple[tuple[str, ...], tuple[str, ...]]]
) -> tuple[str, ...]:
    matched: list[str] = []
    for patterns, components in rules:
        if _matches_any(name, patterns):
            matched.extend(components)
    return _unique_ordered(matched)


def _stage_artifact_names(topology, stage_name: str) -> tuple[str, ...]:
    stage = getattr(topology, "build_stages", {}).get(stage_name)
    if stage is None:
        return ()

    names: list[str] = []
    seen: set[str] = set()

    # Preferred path: explicit stage -> artifact mapping.
    if hasattr(topology, "get_produced_artifacts"):
        try:
            produced = topology.get_produced_artifacts(stage_name)
        except Exception:
            produced = ()
        for artifact_name in produced or ():
            if artifact_name not in seen:
                seen.add(artifact_name)
                names.append(artifact_name)

    # Secondary path: artifact groups -> artifacts.
    if hasattr(stage, "artifact_groups") and hasattr(
        topology, "get_artifact_group_to_artifacts"
    ):
        try:
            artifacts_by_group = topology.get_artifact_group_to_artifacts()
        except Exception:
            artifacts_by_group = {}
        for group_name in getattr(stage, "artifact_groups", ()):
            for artifact_name in artifacts_by_group.get(group_name, []):
                if artifact_name not in seen:
                    seen.add(artifact_name)
                    names.append(artifact_name)

    return tuple(names)


def _source_set_name(obj) -> str | None:
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj
    name = getattr(obj, "name", None)
    if isinstance(name, str) and name:
        return name
    return None


def _changed_paths_to_source_sets(
    topology, changed_paths: Sequence[str]
) -> tuple[str, ...]:
    source_sets: list[str] = []
    seen: set[str] = set()

    for path in changed_paths:
        source_set = None

        if hasattr(topology, "get_source_set_for_path"):
            try:
                source_set = topology.get_source_set_for_path(path, platform=None)
            except TypeError:
                try:
                    source_set = topology.get_source_set_for_path(path)
                except Exception:
                    source_set = None
            except Exception:
                source_set = None

        name = _source_set_name(source_set)
        if name is None and hasattr(topology, "get_source_set_for_submodule"):
            # Best-effort fallback for submodule-root changes.
            first_part = path.split("/", 1)[0]
            try:
                source_set = topology.get_source_set_for_submodule(
                    first_part, platform=None
                )
            except TypeError:
                try:
                    source_set = topology.get_source_set_for_submodule(first_part)
                except Exception:
                    source_set = None
            except Exception:
                source_set = None
            name = _source_set_name(source_set)

        if name and name not in seen:
            seen.add(name)
            source_sets.append(name)

    return tuple(source_sets)


def map_stages_to_test_components(
    topology, stage_names: Sequence[str]
) -> tuple[str, ...]:
    """Map impacted build stages to test components conservatively."""
    components: list[str] = []

    for stage_name in stage_names:
        components.extend(_rule_components(stage_name, _STAGE_COMPONENT_RULES))

        # Also inspect the artifacts produced by the stage.
        artifact_names = _stage_artifact_names(topology, stage_name)
        if artifact_names:
            components.extend(
                map_artifacts_to_test_components(topology, artifact_names)
            )

        # Optional topology hook if available in future.
        for hook_name in (
            "get_test_components_for_stage",
            "get_test_components_for_build_stage",
        ):
            if hasattr(topology, hook_name):
                try:
                    hook_value = getattr(topology, hook_name)(stage_name)
                except Exception:
                    hook_value = ()
                if hook_value:
                    if isinstance(hook_value, str):
                        components.append(hook_value)
                    else:
                        components.extend(str(item) for item in hook_value)

    return _unique_ordered(comp for comp in components if comp in ALL_TEST_COMPONENTS)


def map_artifacts_to_test_components(
    topology, artifact_names: Sequence[str]
) -> tuple[str, ...]:
    """Map impacted artifact names to test components conservatively."""
    components: list[str] = []

    for artifact_name in artifact_names:
        components.extend(_rule_components(artifact_name, _ARTIFACT_COMPONENT_RULES))

        # Optional topology hook if available in future.
        for hook_name in (
            "get_test_components_for_artifact",
            "get_test_components_for_artifacts",
        ):
            if hasattr(topology, hook_name):
                try:
                    hook_value = getattr(topology, hook_name)(artifact_name)
                except Exception:
                    hook_value = ()
                if hook_value:
                    if isinstance(hook_value, str):
                        components.append(hook_value)
                    else:
                        components.extend(str(item) for item in hook_value)

    return _unique_ordered(comp for comp in components if comp in ALL_TEST_COMPONENTS)


def _source_sets_to_test_components(source_sets: Sequence[str]) -> tuple[str, ...]:
    components: list[str] = []
    for source_set in source_sets:
        components.extend(_rule_components(source_set, _SOURCE_SET_RULES))
    return _unique_ordered(comp for comp in components if comp in ALL_TEST_COMPONENTS)


def compute_test_matrix_filter(
    *,
    changed_paths: Sequence[str] | None,
    stage_impact_result,
    topology,
    dry_run: bool = True,
) -> ImpactAnalysisPlan:
    """
    Compute a conservative test matrix filter.

    For now this stays report-only by default:
    - dry_run=True keeps all tests selected but explains what could be skipped;
    - dry_run=False selects only the affected tests.
    """
    if changed_paths is None:
        return ImpactAnalysisPlan(
            changed_paths=(),
            affected_source_sets=(),
            affected_test_components=ALL_TEST_COMPONENTS,
            skipped_test_components=(),
            selected_test_components=ALL_TEST_COMPONENTS,
            full_rebuild_required=True,
            reasons=("no changed-path list available",),
            report_lines=(
                "[TEST-IMPACT] no changed-path list; keeping the full test matrix.",
            ),
        )

    changed_paths_t = tuple(changed_paths)
    affected_source_sets = _changed_paths_to_source_sets(topology, changed_paths_t)

    full_rebuild_required = bool(
        getattr(stage_impact_result, "full_rebuild_required", False)
    )
    reasons = tuple(getattr(stage_impact_result, "reasons", ()))

    rebuild_stages = tuple(getattr(stage_impact_result, "rebuild_stages", ()))
    copy_stages = tuple(getattr(stage_impact_result, "copy_stages", ()))

    affected_from_stages = map_stages_to_test_components(topology, rebuild_stages)
    affected_from_copy = map_stages_to_test_components(topology, copy_stages)

    affected_from_source_sets = _source_sets_to_test_components(affected_source_sets)

    # Artifact names from impacted stages can refine the mapping a little more.
    impacted_artifacts: list[str] = []
    for stage_name in rebuild_stages:
        impacted_artifacts.extend(_stage_artifact_names(topology, stage_name))
    affected_from_artifacts = map_artifacts_to_test_components(
        topology, impacted_artifacts
    )

    affected_components = _unique_ordered(
        [
            *affected_from_stages,
            *affected_from_copy,
            *affected_from_source_sets,
            *affected_from_artifacts,
        ]
    )

    if full_rebuild_required:
        selected_test_components = ALL_TEST_COMPONENTS
        skipped_test_components = ()
    else:
        if dry_run:
            selected_test_components = ALL_TEST_COMPONENTS
            skipped_test_components = tuple(
                comp for comp in ALL_TEST_COMPONENTS if comp not in affected_components
            )
        else:
            # Opt-in mode: only run the affected components.
            selected_test_components = affected_components or ALL_TEST_COMPONENTS
            skipped_test_components = tuple(
                comp
                for comp in ALL_TEST_COMPONENTS
                if comp not in selected_test_components
            )

    report_lines = [
        "[TEST-IMPACT] test impact analysis",
        f"[TEST-IMPACT] changed paths: {', '.join(changed_paths_t) if changed_paths_t else '_none_'}",
        f"[TEST-IMPACT] affected source sets: {', '.join(affected_source_sets) if affected_source_sets else '_none_'}",
        f"[TEST-IMPACT] rebuild stages: {', '.join(rebuild_stages) if rebuild_stages else '_none_'}",
        f"[TEST-IMPACT] copy stages: {', '.join(copy_stages) if copy_stages else '_none_'}",
        f"[TEST-IMPACT] affected test components: {', '.join(affected_components) if affected_components else '_none_'}",
        f"[TEST-IMPACT] selected test components: {', '.join(selected_test_components) if selected_test_components else '_none_'}",
    ]
    if skipped_test_components:
        report_lines.append(
            f"[TEST-IMPACT] skipped test components: {', '.join(skipped_test_components)}"
        )
    if full_rebuild_required:
        report_lines.append(
            "[TEST-IMPACT] full CI fallback: keeping the full test matrix."
        )
        for reason in reasons:
            report_lines.append(f"[TEST-IMPACT]   reason: {reason}")
    elif dry_run:
        report_lines.append(
            "[TEST-IMPACT] dry-run only: no tests were removed from the matrix."
        )

    return ImpactAnalysisPlan(
        changed_paths=changed_paths_t,
        affected_source_sets=affected_source_sets,
        affected_test_components=affected_components,
        skipped_test_components=skipped_test_components,
        selected_test_components=selected_test_components,
        full_rebuild_required=full_rebuild_required,
        reasons=reasons,
        report_lines=tuple(report_lines),
    )


def format_impact_analysis_summary(plan: ImpactAnalysisPlan) -> str:
    """Render a markdown summary for GitHub step summary output."""

    def fmt(values: Sequence[str]) -> str:
        if not values:
            return "_none_"
        return ", ".join(f"`{value}`" for value in values)

    out: list[str] = ["### Test impact analysis", ""]
    out.append(f"- full rebuild required: `{plan.full_rebuild_required}`")
    out.append(f"- changed paths: {fmt(plan.changed_paths)}")
    out.append(f"- affected source sets: {fmt(plan.affected_source_sets)}")
    out.append(f"- affected test components: {fmt(plan.affected_test_components)}")
    out.append(f"- selected test components: {fmt(plan.selected_test_components)}")
    out.append(f"- skipped test components: {fmt(plan.skipped_test_components)}")
    if plan.reasons:
        out.append("- reasons:")
        for reason in plan.reasons:
            out.append(f"  - {reason}")
    return "\n".join(out)
