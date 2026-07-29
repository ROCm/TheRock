# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Report-only mapping from semantic stage impact to test components."""


from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from _therock_utils.build_topology import (
    BuildTopology,
    artifact_may_apply_to_platform,
)

# Only names that cannot be resolved through BuildTopology's
# normal subproject/artifact aliases belong here.
TEST_COMPONENT_ARTIFACT_OVERRIDES: dict[
    str,
    tuple[str, ...],
] = {
    "hip-tests": ("core-hiptests",),
    "rocgdb-cpu": ("rocgdb",),
    "rocgdb-gpu": ("rocgdb",),
    "rocr-debug-agent": ("rocr-debug-agent-tests",),
}

DEFAULT_FORCED_COMPONENTS = ("sanity",)


@dataclass(frozen=True)
class StageToTestImpactResult:
    """Dry-run test-component recommendation for one platform."""

    platform: str
    rebuild_stages: tuple[str, ...]
    would_run: tuple[str, ...]
    would_skip: tuple[str, ...]
    not_applicable: tuple[str, ...]
    unmapped: tuple[str, ...]
    forced: tuple[str, ...]
    full_rebuild_required: bool
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "platform": self.platform,
            "rebuild_stages": self.rebuild_stages,
            "would_run": self.would_run,
            "would_skip": self.would_skip,
            "not_applicable": self.not_applicable,
            "unmapped": self.unmapped,
            "forced": self.forced,
            "full_rebuild_required": (self.full_rebuild_required),
            "reasons": self.reasons,
        }


def requested_test_components(
    labels: Sequence[str],
) -> tuple[str, ...]:
    """Normalize explicit test labels into component names."""

    components: list[str] = []

    for label in labels:
        normalized = label.strip().lower()

        if not normalized or normalized.startswith("test_filter:"):
            continue

        if normalized.startswith("test:"):
            normalized = normalized.removeprefix("test:")

        if normalized:
            components.append(normalized)

    return tuple(dict.fromkeys(components))


def _resolve_component_artifacts(
    *,
    topology: BuildTopology,
    component: str,
    overrides: Mapping[str, Sequence[str]],
) -> tuple[str, ...]:
    overridden = overrides.get(component)

    if overridden is not None:
        return tuple(overridden)

    artifact_name = topology.resolve_project_to_artifact(component)

    if artifact_name is None:
        return ()

    return (artifact_name,)


def compute_test_impact(
    *,
    topology: BuildTopology,
    platform: str,
    components: Sequence[str],
    rebuild_stages: Sequence[str],
    full_rebuild_required: bool,
    reasons: Sequence[str] = (),
    forced_components: Sequence[str] = (),
    component_artifact_overrides: Mapping[
        str,
        Sequence[str],
    ] = TEST_COMPONENT_ARTIFACT_OVERRIDES,
) -> StageToTestImpactResult:
    """Compute a report-only recommendation from semantic stage impact.

    This function never changes the caller's test labels or
    generated matrix. Unknown components, missing artifacts,
    and missing producer stages remain enabled conservatively.
    """

    rebuild_order = tuple(dict.fromkeys(rebuild_stages))
    rebuild_set = set(rebuild_order)

    producer_stages = topology.get_artifact_to_producer_stages()

    forced_order = tuple(
        dict.fromkeys(
            (
                *DEFAULT_FORCED_COMPONENTS,
                *forced_components,
            )
        )
    )
    forced_set = set(forced_order)

    component_order = tuple(
        dict.fromkeys(
            (
                *components,
                *forced_order,
            )
        )
    )

    would_run: list[str] = []
    would_skip: list[str] = []
    not_applicable: list[str] = []
    unmapped: list[str] = []
    forced: list[str] = []

    for component in component_order:
        if component in forced_set:
            forced.append(component)
            would_run.append(component)
            continue

        mapped_artifacts = _resolve_component_artifacts(
            topology=topology,
            component=component,
            overrides=(component_artifact_overrides),
        )

        if not mapped_artifacts:
            unmapped.append(component)
            would_run.append(component)
            continue

        applicable_artifacts: list[str] = []
        mapping_incomplete = False

        for artifact_name in mapped_artifacts:
            artifact = topology.artifacts.get(artifact_name)

            if artifact is None:
                mapping_incomplete = True
                continue

            if artifact_may_apply_to_platform(
                artifact,
                platform,
            ):
                applicable_artifacts.append(artifact_name)

        if mapping_incomplete:
            unmapped.append(component)
            would_run.append(component)
            continue

        if not applicable_artifacts:
            not_applicable.append(component)
            continue

        component_stages: set[str] = set()

        for artifact_name in applicable_artifacts:
            stages = producer_stages.get(
                artifact_name,
                (),
            )

            if not stages:
                mapping_incomplete = True
                break

            component_stages.update(stages)

        if mapping_incomplete:
            unmapped.append(component)
            would_run.append(component)
        elif full_rebuild_required or component_stages & rebuild_set:
            would_run.append(component)
        else:
            would_skip.append(component)

    return StageToTestImpactResult(
        platform=platform,
        rebuild_stages=rebuild_order,
        would_run=tuple(would_run),
        would_skip=tuple(would_skip),
        not_applicable=tuple(not_applicable),
        unmapped=tuple(unmapped),
        forced=tuple(forced),
        full_rebuild_required=(full_rebuild_required),
        reasons=tuple(reasons),
    )


def _format_names(values: Sequence[str]) -> str:
    if not values:
        return "_none_"

    formatted_values: list[str] = []

    for value in values:
        safe_value = value.replace("`", "'")
        formatted_values.append(f"`{safe_value}`")

    return ", ".join(formatted_values)


def _format_reason(reason: str) -> str:
    return reason.replace("\r", " ").replace("\n", " ")


def render_test_impact_summary(
    results: Sequence[StageToTestImpactResult],
) -> str:
    """Render platform results for GITHUB_STEP_SUMMARY."""

    lines = [
        "### Test impact analysis — dry-run",
        "",
        ("> Report only: test labels and generated " "test matrices are unchanged."),
        (
            "> This is semantic impact analysis before "
            "existing test-type and label filtering."
        ),
        (
            "> Extended functional and benchmark matrices "
            "are not evaluated by this prototype."
        ),
    ]

    ordered_results = sorted(
        results,
        key=lambda result: (
            result.platform != "linux",
            result.platform,
        ),
    )

    for result in ordered_results:
        lines.extend(
            [
                "",
                f"#### {result.platform.capitalize()}",
                "",
                ("- full rebuild required: " f"`{result.full_rebuild_required}`"),
                (
                    "- semantic rebuild stages: "
                    f"{_format_names(result.rebuild_stages)}"
                ),
                ("- would keep enabled: " f"{_format_names(result.would_run)}"),
                ("- would skip: " f"{_format_names(result.would_skip)}"),
                ("- unmapped and kept enabled: " f"{_format_names(result.unmapped)}"),
                ("- not applicable: " f"{_format_names(result.not_applicable)}"),
                ("- forced to remain enabled: " f"{_format_names(result.forced)}"),
            ]
        )

        if result.reasons:
            lines.append("- fallback reasons:")
            lines.extend(f"  - {_format_reason(reason)}" for reason in result.reasons)

    return "\n".join(lines)
