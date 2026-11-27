# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The cache planner (flowchart 1).

Given the current source (changed paths + commit) and the target families to
build, the planner:

1. enumerates the required artifact *units* per build stage (from the topology),
2. computes a cache key for each unit,
3. classifies each unit as a hit or a miss with a single batched cache lookup,
4. decides, per stage, whether the whole stage can be skipped (all units hit)
   or which specific units must be built.

Units whose source was directly modified are forced misses -- they must be
rebuilt regardless of what the cache contains. Everything else is a direct
key lookup, with no GitHub API calls and no git-history traversal.

This is deliberately separate from *test* selection (RFC0013 / the consumer
graph): the planner decides how to obtain build outputs, not what to validate.
A cache hit here does not suppress any test.
"""


from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .cache_key import (
    ArtifactIdentity,
    CacheKey,
    compute_cache_key,
    derive_fallback_fingerprint,
)
from .cache_db import CacheDB, ReadPolicy
from .manifest import Manifest
from .topology_view import TopologyView


@dataclass(frozen=True)
class RequiredUnit:
    """An artifact unit a run needs, with its computed cache key."""

    artifact: ArtifactIdentity
    stage: str
    cache_key: CacheKey
    source_changed: bool

    @property
    def nt(self) -> tuple[str, str]:
        return (self.artifact.name, self.artifact.target_family)


@dataclass(frozen=True)
class PlanEntry:
    """Outcome for one required unit."""

    unit: RequiredUnit
    status: str  # "hit" | "miss"
    reason: str
    manifest: Manifest | None = None

    @property
    def is_hit(self) -> bool:
        return self.status == "hit"


@dataclass
class StagePlan:
    """Plan for a single build stage."""

    stage: str
    entries: list[PlanEntry] = field(default_factory=list)

    @property
    def can_skip(self) -> bool:
        """True when every required unit is a cache hit (skip the whole job)."""
        return bool(self.entries) and all(e.is_hit for e in self.entries)

    @property
    def units_to_build(self) -> list[ArtifactIdentity]:
        return [e.unit.artifact for e in self.entries if not e.is_hit]

    @property
    def hit_count(self) -> int:
        return sum(1 for e in self.entries if e.is_hit)


@dataclass
class CachePlan:
    """The full plan for a workflow run."""

    build_variant: str
    source_commit: str
    stage_plans: list[StagePlan] = field(default_factory=list)

    # -- aggregate metrics -------------------------------------------------
    @property
    def total_units(self) -> int:
        return sum(len(sp.entries) for sp in self.stage_plans)

    @property
    def total_hits(self) -> int:
        return sum(sp.hit_count for sp in self.stage_plans)

    @property
    def total_misses(self) -> int:
        return self.total_units - self.total_hits

    @property
    def skippable_stages(self) -> list[str]:
        return [sp.stage for sp in self.stage_plans if sp.can_skip]

    @property
    def hit_rate(self) -> float:
        return (self.total_hits / self.total_units) if self.total_units else 0.0

    def to_dict(self) -> dict:
        return {
            "build_variant": self.build_variant,
            "source_commit": self.source_commit,
            "total_units": self.total_units,
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "hit_rate": round(self.hit_rate, 4),
            "skippable_stages": self.skippable_stages,
            "stages": [
                {
                    "stage": sp.stage,
                    "can_skip": sp.can_skip,
                    "hits": sp.hit_count,
                    "units": len(sp.entries),
                    "units_to_build": [str(a) for a in sp.units_to_build],
                    "entries": [
                        {
                            "unit": str(e.unit.artifact),
                            "status": e.status,
                            "reason": e.reason,
                            "key": e.unit.cache_key.short,
                            "from_run": (
                                e.manifest.provenance.run_id if e.manifest else None
                            ),
                        }
                        for e in sp.entries
                    ],
                }
                for sp in self.stage_plans
            ],
        }

    def render(self) -> str:
        """Human-readable plan, echoing the flowchart's decisions."""
        lines = [
            f"Cache plan @ {self.source_commit[:12]} variant={self.build_variant}",
            f"  units={self.total_units} hits={self.total_hits} "
            f"misses={self.total_misses} hit_rate={self.hit_rate:.1%}",
            f"  skippable stages ({len(self.skippable_stages)}): "
            f"{', '.join(self.skippable_stages) or '(none)'}",
        ]
        for sp in self.stage_plans:
            verb = "SKIP" if sp.can_skip else "BUILD"
            lines.append(
                f"  [{verb}] {sp.stage}: {sp.hit_count}/{len(sp.entries)} hits"
            )
            if not sp.can_skip:
                for a in sp.units_to_build:
                    lines.append(f"        build {a}")
        return "\n".join(lines)


def plan_cache(
    *,
    topology: TopologyView,
    changed_paths: Sequence[str],
    target_families: Sequence[str],
    source_commit: str,
    db: CacheDB,
    build_variant: str = "release",
    read_policy: ReadPolicy | None = None,
    restrict_to_stages: Sequence[str] | None = None,
    dep_fingerprints: Mapping[tuple[str, str], Sequence[str]] | None = None,
    source_fingerprints: Mapping[tuple[str, str], str] | None = None,
) -> CachePlan:
    """Produce a :class:`CachePlan` for one workflow run.

    Args:
        topology: build-graph view (``BuildTopologyView`` or ``DictTopologyView``).
        changed_paths: paths modified by the PR/push, relative to repo root.
        target_families: GPU target families being built (e.g. ``["gfx94X-dcgpu"]``).
        source_commit: the commit the run is building (used for the fallback key).
        db: the cache database to query.
        build_variant: build variant that scopes the keys.
        read_policy: trust/validation/point-in-time filter for lookups.
        restrict_to_stages: optional subset of stages to plan (else all, in
            build order).
        dep_fingerprints: optional per-unit dependency fingerprints to fold into
            the key. Usually empty.
        source_fingerprints: optional per-unit source fingerprint override, keyed
            by ``(artifact_name, target_family)``. When present for a unit it is
            used directly instead of the fallback -- this is the hook
            where a real uploaded ``.fprint`` plugs in, and is what lets the
            replay harness model per-subproject (cross-commit) reuse.
    """
    dep_fingerprints = dep_fingerprints or {}
    source_fingerprints = source_fingerprints or {}
    stages = list(restrict_to_stages or topology.build_order_stages())
    changed = topology.changed_units(changed_paths, target_families)

    # Pass 1: enumerate units + keys, collect keys to look up in one batch.
    units_by_stage: list[tuple[str, list[RequiredUnit]]] = []
    keys_to_lookup: list[str] = []
    for stage in stages:
        stage_units: list[RequiredUnit] = []
        for artifact in topology.units_for_stage(stage, target_families):
            nt = (artifact.name, artifact.target_family)
            source_changed = nt in changed
            override = source_fingerprints.get(nt)
            if override is not None:
                fingerprint = override
            else:
                fingerprint = derive_fallback_fingerprint(
                    source_commit=source_commit,
                    artifact=artifact,
                    dep_fingerprints=dep_fingerprints.get(nt, ()),
                    build_variant=build_variant,
                )
            key = compute_cache_key(
                source_fingerprint=fingerprint, build_variant=build_variant
            )
            unit = RequiredUnit(
                artifact=artifact,
                stage=stage,
                cache_key=key,
                source_changed=source_changed,
            )
            stage_units.append(unit)
            if not source_changed:
                keys_to_lookup.append(key.value)
        units_by_stage.append((stage, stage_units))

    # Single batched lookup for every non-changed unit across all stages.
    found = db.lookup_batch(keys_to_lookup, read_policy)

    # Pass 2: classify and assemble.
    plan = CachePlan(build_variant=build_variant, source_commit=source_commit)
    for stage, stage_units in units_by_stage:
        sp = StagePlan(stage=stage)
        for unit in stage_units:
            if unit.source_changed:
                sp.entries.append(
                    PlanEntry(unit, "miss", "source changed -> must rebuild")
                )
                continue
            manifest = found.get(unit.cache_key.value)
            if manifest is not None:
                sp.entries.append(
                    PlanEntry(
                        unit,
                        "hit",
                        f"cache hit from run {manifest.provenance.run_id}",
                        manifest,
                    )
                )
            else:
                sp.entries.append(PlanEntry(unit, "miss", "no compatible cache entry"))
        plan.stage_plans.append(sp)
    return plan
