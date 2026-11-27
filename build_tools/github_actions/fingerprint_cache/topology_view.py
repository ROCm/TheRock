# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Topology adapters for the planner.

The planner only needs a small, structural view of the build graph:

* the stages in build order,
* the artifact units each stage produces (expanded across target families),
* which units a set of changed paths directly modified.

``TopologyView`` is that structural interface. ``BuildTopologyView`` adapts
TheRock's real ``_therock_utils.build_topology.BuildTopology`` to it, so the
planner can run against the committed ``BUILD_TOPOLOGY.toml``. ``DictTopologyView``
is a hand-built view for tests and small experiments, so the planner's logic can
be exercised without a full checkout.
"""


from typing import Protocol, Sequence

from .cache_key import ArtifactIdentity

# Component sentinel for an artifact-granularity reuse unit that bundles all of
# an artifact's component archives (lib/dev/run/...). Split / per-arch artifacts
# can instead use a concrete component in their ArtifactIdentity.
BUNDLE_COMPONENT = "bundle"


class TopologyView(Protocol):
    """Minimal structural view of the build graph used by the planner."""

    def build_order_stages(self) -> list[str]:
        """Stage names in dependency order."""
        ...

    def units_for_stage(
        self, stage: str, target_families: Sequence[str]
    ) -> list[ArtifactIdentity]:
        """Artifact units a stage produces, expanded across target families."""
        ...

    def changed_units(
        self, changed_paths: Sequence[str], target_families: Sequence[str]
    ) -> set[tuple[str, str]]:
        """``(artifact_name, target_family)`` pairs directly modified.

        A pair in this set is forced to rebuild (cache miss by construction).
        Returning target_family-agnostic names is fine; the planner matches on
        name and treats all its target families as changed.
        """
        ...


class DictTopologyView:
    """A simple in-memory topology for tests and small experiments.

    Args:
        stages: ordered list of ``(stage_name, artifact_spec)`` where
            ``artifact_spec`` is a list of ``(artifact_name, is_target_specific)``.
            Target-neutral artifacts (``is_target_specific=False``) always expand
            to the single ``generic`` family; target-specific ones expand to each
            requested target family.
        path_to_artifacts: maps a changed path to the artifact names it modifies.
            Any path not present (and not matching a prefix in ``global_paths``)
            modifies nothing. A path in ``global_paths`` forces a full rebuild
            (every unit changed).
    """

    def __init__(
        self,
        stages: list[tuple[str, list[tuple[str, bool]]]],
        path_to_artifacts: dict[str, list[str]] | None = None,
        global_paths: Sequence[str] = (),
    ):
        self._stages = stages
        self._path_to_artifacts = path_to_artifacts or {}
        self._global_paths = tuple(global_paths)

    def build_order_stages(self) -> list[str]:
        return [name for name, _ in self._stages]

    def _specs(self, stage: str) -> list[tuple[str, bool]]:
        for name, specs in self._stages:
            if name == stage:
                return specs
        return []

    def units_for_stage(
        self, stage: str, target_families: Sequence[str]
    ) -> list[ArtifactIdentity]:
        units: list[ArtifactIdentity] = []
        for artifact_name, target_specific in self._specs(stage):
            families = list(target_families) if target_specific else ["generic"]
            for family in families:
                units.append(ArtifactIdentity(artifact_name, BUNDLE_COMPONENT, family))
        return units

    def _all_artifact_names(self) -> set[str]:
        return {name for _, specs in self._stages for name, _ in specs}

    def changed_units(
        self, changed_paths: Sequence[str], target_families: Sequence[str]
    ) -> set[tuple[str, str]]:
        families = list(target_families) + ["generic"]
        changed_names: set[str] = set()
        for path in changed_paths:
            if any(path.startswith(g) for g in self._global_paths):
                # Conservative full rebuild.
                changed_names |= self._all_artifact_names()
            changed_names.update(self._path_to_artifacts.get(path, []))
        return {(name, fam) for name in changed_names for fam in families}


class BuildTopologyView:
    """Adapts TheRock's real ``BuildTopology`` to ``TopologyView``.

    Imports ``_therock_utils.build_topology`` lazily so the pure planner and its
    tests do not require a full repo checkout. Construct via
    :meth:`from_topology_path` from within the TheRock tree.
    """

    def __init__(self, topology):
        self._t = topology

    @classmethod
    def from_topology_path(cls, topology_path=None) -> "BuildTopologyView":
        from pathlib import Path
        from _therock_utils.build_topology import get_topology  # lazy import

        if topology_path is not None and not isinstance(topology_path, Path):
            topology_path = Path(topology_path)
        return cls(get_topology(topology_path))

    def build_order_stages(self) -> list[str]:
        return list(self._t.get_build_order())

    def units_for_stage(
        self, stage: str, target_families: Sequence[str]
    ) -> list[ArtifactIdentity]:
        units: list[ArtifactIdentity] = []
        produced = self._t.get_produced_artifacts(stage)
        artifacts_by_name = {a.name: a for a in self._t.get_artifacts()}
        for name in sorted(produced):
            artifact = artifacts_by_name.get(name)
            target_specific = bool(
                artifact and getattr(artifact, "type", "") == "target-specific"
            )
            families = list(target_families) if target_specific else ["generic"]
            for family in families:
                if artifact is not None and self._t.is_artifact_disabled_on_processor(
                    artifact, family
                ):
                    continue
                units.append(ArtifactIdentity(name, BUNDLE_COMPONENT, family))
        return units

    def changed_units(
        self, changed_paths: Sequence[str], target_families: Sequence[str]
    ) -> set[tuple[str, str]]:
        families = list(target_families) + ["generic"]
        changed_names: set[str] = set()
        for path in changed_paths:
            names = self._t.get_artifacts_for_path(path)
            if not names:
                # Unmapped path: conservative -- treat as changing everything so
                # the planner never claims a false reuse. Mirrors stage_impact's
                # full-CI fallback.
                for artifact in self._t.get_artifacts():
                    changed_names.add(artifact.name)
            else:
                changed_names.update(names)
        return {(name, fam) for name in changed_names for fam in families}
