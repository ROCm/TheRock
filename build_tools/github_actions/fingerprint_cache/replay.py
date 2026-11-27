# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Point-in-time replay harness (the evidence experiment).

Replays a chronological sequence of workflow runs against a cache that is built
up as the replay proceeds, so each run only ever sees entries produced by
*earlier* runs. For each run it records what the fingerprint cache would have
reused (hit rate, units reused, stages skippable), and compares against a naive
baseline that models the current approach: reuse is possible only when an
earlier run built the exact same commit.

This quantifies the two questions from the design notes -- "how many artifacts
are being rebuilt that don't need to be?" and "how would the new system compare
to what we have?" -- while being honest about the fallback key's coarseness (it keys
on whole-repo commit, so it understates achievable reuse).

The harness operates on ``ReplayRun`` scenarios so it can run fully offline in
tests. Building scenarios from real history (run metadata + artifact listings)
is a thin wrapper around ``history_ingest`` helpers; see the CLI.
"""


from dataclasses import dataclass, field
from typing import Sequence

from .cache_key import ArtifactIdentity, derive_fallback_fingerprint
from .cache_db import CacheDB, ReadPolicy
from .manifest import OutputObject, Provenance, TrustLevel, ValidationState
from .planner import plan_cache
from .publish import publish_build_outputs
from .topology_view import BUNDLE_COMPONENT, TopologyView


@dataclass(frozen=True)
class ReplayRun:
    """One run in the replay timeline."""

    run_id: str
    commit: str
    created_at: str  # ISO 8601, used only for ordering/provenance
    branch: str = "main"
    event: str = "push"
    changed_paths: tuple[str, ...] = ()
    # Artifact units this run produced, as (name, target_family). Their objects
    # are synthesized; for real data these come from the artifact listing.
    produced_units: tuple[tuple[str, str], ...] = ()
    # Optional per-unit source fingerprints keyed by (name, target_family). This
    # models what real per-subproject ``.fprint`` values provide: an unchanged
    # subproject keeps the same fingerprint across commits, enabling cross-commit
    # reuse. When empty, the coarse whole-commit fallback is used instead.
    unit_fingerprints: dict[tuple[str, str], str] = field(default_factory=dict)


@dataclass
class ReplayRunResult:
    run_id: str
    commit: str
    units: int
    cache_hits: int
    baseline_hits: int  # naive same-commit model
    skippable_stages: int

    @property
    def cache_hit_rate(self) -> float:
        return (self.cache_hits / self.units) if self.units else 0.0

    @property
    def baseline_hit_rate(self) -> float:
        return (self.baseline_hits / self.units) if self.units else 0.0


@dataclass
class ReplayReport:
    results: list[ReplayRunResult] = field(default_factory=list)

    @property
    def total_units(self) -> int:
        return sum(r.units for r in self.results)

    @property
    def total_cache_hits(self) -> int:
        return sum(r.cache_hits for r in self.results)

    @property
    def total_baseline_hits(self) -> int:
        return sum(r.baseline_hits for r in self.results)

    @property
    def cache_hit_rate(self) -> float:
        return (self.total_cache_hits / self.total_units) if self.total_units else 0.0

    @property
    def baseline_hit_rate(self) -> float:
        return self.total_baseline_hits / self.total_units if self.total_units else 0.0

    def to_dict(self) -> dict:
        return {
            "runs": len(self.results),
            "total_units": self.total_units,
            "cache_hits": self.total_cache_hits,
            "baseline_hits": self.total_baseline_hits,
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "baseline_hit_rate": round(self.baseline_hit_rate, 4),
            "improvement_ratio": round(
                (
                    (self.cache_hit_rate / self.baseline_hit_rate)
                    if self.baseline_hit_rate
                    else float("inf")
                ),
                3,
            ),
            "per_run": [
                {
                    "run_id": r.run_id,
                    "commit": r.commit[:12],
                    "units": r.units,
                    "cache_hits": r.cache_hits,
                    "baseline_hits": r.baseline_hits,
                    "cache_hit_rate": round(r.cache_hit_rate, 4),
                    "skippable_stages": r.skippable_stages,
                }
                for r in self.results
            ],
        }

    def render(self) -> str:
        lines = [
            f"Replay over {len(self.results)} runs, {self.total_units} units:",
            f"  fingerprint-cache hit rate: {self.cache_hit_rate:.1%} "
            f"({self.total_cache_hits}/{self.total_units})",
            f"  naive same-commit baseline: {self.baseline_hit_rate:.1%} "
            f"({self.total_baseline_hits}/{self.total_units})",
        ]
        return "\n".join(lines)


def _synthesize_objects(
    run: ReplayRun, name: str, target_family: str
) -> list[OutputObject]:
    key = f"{name}_lib_{target_family}.tar.zst"
    return [OutputObject(key=key, uri=f"s3://ci-artifacts/{run.run_id}/{key}")]


def replay(
    scenarios: Sequence[ReplayRun],
    *,
    topology: TopologyView,
    target_families: Sequence[str],
    build_variant: str = "release",
    read_policy: ReadPolicy | None = None,
) -> ReplayReport:
    """Replay runs chronologically, planning before ingesting each run.

    For each run (in the given order): plan against the cache as it stands
    (seeing only earlier runs), record metrics, then publish the run's produced
    units so later runs can reuse them.
    """
    report = ReplayReport()
    seen_commits: set[str] = set()
    # Loosen default policy for replay so untrusted history is visible; callers
    # can pass a stricter policy to model trusted-only reuse.
    policy = read_policy or ReadPolicy(
        min_trust=TrustLevel.UNTRUSTED, min_validation=ValidationState.BUILD_PASSED
    )
    with CacheDB() as db:
        for run in scenarios:
            plan = plan_cache(
                topology=topology,
                changed_paths=run.changed_paths,
                target_families=target_families,
                source_commit=run.commit,
                db=db,
                build_variant=build_variant,
                read_policy=policy,
                source_fingerprints=run.unit_fingerprints,
            )
            # Naive baseline: every non-changed unit is reusable iff this exact
            # commit was already built by an earlier run.
            baseline_possible = run.commit in seen_commits
            baseline_hits = 0
            if baseline_possible:
                baseline_hits = sum(
                    1
                    for sp in plan.stage_plans
                    for e in sp.entries
                    if not e.unit.source_changed
                )
            report.results.append(
                ReplayRunResult(
                    run_id=run.run_id,
                    commit=run.commit,
                    units=plan.total_units,
                    cache_hits=plan.total_hits,
                    baseline_hits=baseline_hits,
                    skippable_stages=len(plan.skippable_stages),
                )
            )

            # Ingest this run's produced units for future reuse.
            provenance = Provenance(
                repository="replay",
                run_id=run.run_id,
                branch=run.branch,
                commit=run.commit,
                workflow="replay",
                created_at=run.created_at,
            )
            trust = (
                TrustLevel.TRUSTED
                if run.event in ("push", "schedule") and run.branch == "main"
                else TrustLevel.UNTRUSTED
            )
            for name, target_family in run.produced_units:
                identity = ArtifactIdentity(name, BUNDLE_COMPONENT, target_family)
                fingerprint = run.unit_fingerprints.get(
                    (name, target_family)
                ) or derive_fallback_fingerprint(
                    source_commit=run.commit,
                    artifact=identity,
                    build_variant=build_variant,
                )
                publish_build_outputs(
                    db=db,
                    artifact=identity,
                    outputs=_synthesize_objects(run, name, target_family),
                    provenance=provenance,
                    source_fingerprint=fingerprint,
                    build_variant=build_variant,
                    trust=trust,
                    validation=ValidationState.BUILD_PASSED,
                )
            seen_commits.add(run.commit)
    return report
