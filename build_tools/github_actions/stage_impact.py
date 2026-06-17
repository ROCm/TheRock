# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Stage-impact analysis for TheRock CI/CD pipeline.

This module takes a set of changed repository paths (or submodule roots) and
uses BUILD_TOPOLOGY metadata to determine which build stages are impacted.

It is intentionally conservative:
- Changes to build tooling, workflow files, or the topology file itself force
  a full CI fallback.
- Otherwise, changed paths are mapped to top-level submodule names, then to
  source sets, artifact groups, and build stages.

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Dict, List, Optional, Sequence, Set, Tuple

from _therock_utils.build_topology import BuildTopology, SourceSet


@dataclass(frozen=True)
class StageImpactRuleSet:
    """Rules that determine when to fall back to full CI.

    These rules are intentionally conservative. Any matching input should cause
    the analyzer to stop short of partial reuse unless a future caller explicitly
    opts into a broader policy.
    """

    full_ci_prefixes: Tuple[str, ...] = (
        ".github/",
        "build_tools/",
        "docs/",
        "scripts/",
    )
    full_ci_exact_paths: Tuple[str, ...] = (
        "BUILD_TOPOLOGY.toml",
        "CMakeLists.txt",
    )


@dataclass(frozen=True)
class StageImpactResult:
    """Result of stage-impact analysis.

    Fields:
        changed_inputs: Normalized changed paths or submodule names.
        matched_source_sets: Source sets matched from the changed inputs.
        impacted_artifact_groups: Artifact groups directly impacted.
        rebuild_stages: Stages that must rebuild.
        copy_stages: Stages that can be reused/copied.
        full_rebuild_required: Whether conservative fallback was triggered.
        reasons: Why full CI or a broader rebuild was selected.
        unmatched_inputs: Inputs that could not be mapped to a source set.
    """

    changed_inputs: Tuple[str, ...]
    matched_source_sets: Tuple[str, ...]
    impacted_artifact_groups: Tuple[str, ...]
    rebuild_stages: Tuple[str, ...]
    copy_stages: Tuple[str, ...]
    full_rebuild_required: bool
    reasons: Tuple[str, ...] = ()
    unmatched_inputs: Tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "changed_inputs": self.changed_inputs,
            "matched_source_sets": self.matched_source_sets,
            "impacted_artifact_groups": self.impacted_artifact_groups,
            "rebuild_stages": self.rebuild_stages,
            "copy_stages": self.copy_stages,
            "full_rebuild_required": self.full_rebuild_required,
            "reasons": self.reasons,
            "unmatched_inputs": self.unmatched_inputs,
        }


class StageImpactAnalyzer:
    """Analyze changed inputs and compute impacted build stages."""

    def __init__(
        self, topology: BuildTopology, rules: Optional[StageImpactRuleSet] = None
    ):
        self.topology = topology
        self.rules = rules or StageImpactRuleSet()

    def analyze(
        self, changed_inputs: Sequence[str], platform: Optional[str] = None
    ) -> StageImpactResult:
        """Analyze a set of changed paths or submodule roots.

        Args:
            changed_inputs: Repository-relative paths or submodule root names.
            platform: Optional platform filter (e.g. "linux", "windows").

        Returns:
            A deterministic summary of impacted source sets and build stages.
        """
        normalized_inputs = tuple(
            self._normalize_input(item)
            for item in changed_inputs
            if item and item.strip()
        )

        reasons: List[str] = []
        unmatched_inputs: List[str] = []
        full_rebuild_required = False

        for item in normalized_inputs:
            if self._requires_full_ci(item):
                full_rebuild_required = True
                reasons.append(f"'{item}' matches a conservative full-CI trigger")
                continue

        matched_source_sets = self._resolve_source_sets(
            normalized_inputs, platform=platform, unmatched_inputs=unmatched_inputs
        )

        if not matched_source_sets and not full_rebuild_required:
            full_rebuild_required = True
            reasons.append("No changed input could be mapped to a known source set")

        impacted_artifact_groups = self._resolve_artifact_groups(matched_source_sets)
        impacted_stages = self._resolve_stages(impacted_artifact_groups)

        # Conservative downstream expansion: if a stage is impacted, any stage
        # that consumes its produced artifact groups is also impacted.
        impacted_stages = self._expand_downstream_stages(impacted_stages)

        # If we cannot confidently narrow scope, prefer full CI.
        if full_rebuild_required:
            rebuild_stages = tuple(self.topology.build_stages.keys())
            copy_stages: Tuple[str, ...] = ()
        else:
            rebuild_set = set(impacted_stages)
            rebuild_stages = tuple(sorted(rebuild_set))
            copy_stages = tuple(
                stage_name
                for stage_name in sorted(self.topology.build_stages.keys())
                if stage_name not in rebuild_set
            )

        return StageImpactResult(
            changed_inputs=normalized_inputs,
            matched_source_sets=tuple(sorted(matched_source_sets)),
            impacted_artifact_groups=tuple(sorted(impacted_artifact_groups)),
            rebuild_stages=rebuild_stages,
            copy_stages=copy_stages,
            full_rebuild_required=full_rebuild_required,
            reasons=tuple(reasons),
            unmatched_inputs=tuple(sorted(set(unmatched_inputs))),
        )

    def _normalize_input(self, item: str) -> str:
        return item.strip().lstrip("./")

    def _requires_full_ci(self, item: str) -> bool:
        if item in self.rules.full_ci_exact_paths:
            return True

        return any(
            item == prefix.rstrip("/") or item.startswith(prefix)
            for prefix in self.rules.full_ci_prefixes
        )

    def _resolve_source_sets(
        self,
        changed_inputs: Sequence[str],
        platform: Optional[str],
        unmatched_inputs: List[str],
    ) -> Set[str]:
        matched_source_sets: Set[str] = set()

        for item in changed_inputs:
            if self._requires_full_ci(item):
                continue

            source_set = self._resolve_source_set(item, platform=platform)
            if source_set is None:
                unmatched_inputs.append(item)
                continue
            matched_source_sets.add(source_set.name)

        return matched_source_sets

    def _resolve_source_set(
        self, item: str, platform: Optional[str]
    ) -> Optional[SourceSet]:
        """Resolve a changed input to a source set.

        We support two forms:
        - explicit submodule root names (e.g. "rocm-libraries")
        - paths inside a submodule checkout (e.g. "rocm-libraries/projects/rocPRIM/foo.cpp")
        """
        # First try the whole item as a submodule root.
        source_set = self.topology.get_source_set_for_submodule(item, platform=platform)
        if source_set is not None:
            return source_set

        source_set = self.topology.get_source_set_for_path(item, platform=platform)
        if source_set is not None:
            return source_set

        # Then try each path component as a possible submodule root.
        for part in PurePosixPath(item).parts:
            source_set = self.topology.get_source_set_for_submodule(
                part,
                platform=platform,
            )
            if source_set is not None:
                return source_set

        return None

    def _resolve_artifact_groups(self, source_set_names: Set[str]) -> Set[str]:
        source_set_to_groups = self.topology.get_source_set_to_artifact_groups()
        groups: Set[str] = set()
        for source_set_name in source_set_names:
            groups.update(source_set_to_groups.get(source_set_name, []))
        return groups

    def _resolve_stages(self, artifact_groups: Set[str]) -> Set[str]:
        stages_by_group = self.topology.get_artifact_group_to_build_stages()
        impacted_stages: Set[str] = set()

        # Direct impact: any stage that builds a touched artifact group.
        for group_name in artifact_groups:
            impacted_stages.update(stages_by_group.get(group_name, []))

        return impacted_stages

    def _expand_downstream_stages(self, initial_stages: Set[str]) -> Set[str]:
        """Expand impacted stages through stage-level artifact dependencies.

        If a stage produces artifacts consumed by another stage, the consumer is
        conservatively considered impacted too.
        """
        if not initial_stages:
            return set()

        produced_by_stage = {
            stage_name: self.topology.get_produced_artifacts(stage_name)
            for stage_name in self.topology.build_stages
        }

        # Map artifact -> consumer stages.
        artifact_to_consumer_stages: Dict[str, Set[str]] = {}
        for stage_name in self.topology.build_stages:
            stage = self.topology.build_stages[stage_name]
            for group_name in stage.artifact_groups:
                for artifact in self.topology.get_artifacts_in_group(group_name):
                    for dep_name in artifact.artifact_deps:
                        artifact_to_consumer_stages.setdefault(dep_name, set()).add(
                            stage_name
                        )

        expanded = set(initial_stages)
        worklist = list(initial_stages)

        while worklist:
            stage_name = worklist.pop()
            for artifact_name in produced_by_stage.get(stage_name, set()):
                for consumer_stage in artifact_to_consumer_stages.get(
                    artifact_name, set()
                ):
                    if consumer_stage not in expanded:
                        expanded.add(consumer_stage)
                        worklist.append(consumer_stage)

        return expanded


def analyze_stage_impact(
    changed_inputs: Sequence[str],
    topology: Optional[BuildTopology] = None,
    platform: Optional[str] = None,
    rules: Optional[StageImpactRuleSet] = None,
) -> StageImpactResult:
    """Convenience helper for one-shot stage-impact analysis."""
    if topology is None:
        from _therock_utils.build_topology import get_topology

        topology = get_topology()

    analyzer = StageImpactAnalyzer(topology=topology, rules=rules)
    return analyzer.analyze(changed_inputs=changed_inputs, platform=platform)
