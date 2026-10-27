# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Plan from current CMake keys. A database candidate is not a verified hit."""

from dataclasses import dataclass, field

from .manifest import ArtifactKey, LocalManifest
from .inventory import parse_identity
from .diagnostics import trace, traced


@dataclass(frozen=True)
class RequiredUnit:
    artifact_identity: str
    stage: str
    cache_key: ArtifactKey | None
    reason: str = ""
    components: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlanEntry:
    unit: RequiredUnit
    status: str  # candidate, miss, uncacheable; verification happens on restore
    reason: str
    manifest: LocalManifest | None = None

    @property
    def is_hit(self):
        return False  # Metadata lookup alone never establishes reusable bytes.


@dataclass
class StagePlan:
    stage: str
    entries: list[PlanEntry] = field(default_factory=list)

    @property
    def can_skip(self):
        return False  # Scheduler integration and verified completeness are separate.

    @property
    def all_requested_candidates(self):
        return bool(self.entries) and all(e.status == "candidate" for e in self.entries)


@dataclass
class CachePlan:
    stage_plans: list[StagePlan]
    inventory_scope: str = "observed_outputs"
    dependency_impact: dict | None = None

    @property
    def entries(self):
        return [e for stage in self.stage_plans for e in stage.entries]

    def to_dict(self):
        entries = self.entries
        return {
            "operation": "plan", "inventory_scope": self.inventory_scope,
            "dependency_impact": self.dependency_impact,
            "payloads_verified": False, "skippable_stages": [],
            "summary": {status: sum(e.status == status for e in entries)
                        for status in ("candidate", "miss", "uncacheable")},
            "stages": [{"stage": sp.stage, "can_skip": False,
                        "all_requested_candidates": sp.all_requested_candidates,
                        "entries": [{"artifact_identity": e.unit.artifact_identity,
                                     "status": e.status, "reason": e.reason,
                                     "components": list(e.unit.components),
                                     "fingerprint": e.unit.cache_key.fingerprint if e.unit.cache_key else None,
                                     "epoch": e.unit.cache_key.epoch if e.unit.cache_key else None,
                                     "cache_entry_id": e.manifest.cache_entry_id if e.manifest else None}
                                    for e in sp.entries]} for sp in self.stage_plans],
        }

    def render(self):
        import json
        return json.dumps(self.to_dict(), indent=2)


@traced("PLAN")
def plan_cache(*, db, records, topology=None, read_policy=None,
               changed_paths=(), inventory_scope="observed_outputs", dependency_impact=None):
    """Use (collector record, ArtifactKey-or-None) pairs from current configure.

    Real CMake fingerprints include dependency propagation. Changed paths are
    an optional conservative forced rebuild, never a replacement for keys.
    Unmapped outputs remain visible; inferred inventory cannot skip a stage.
    """
    if changed_paths and topology is None:
        raise ValueError("changed paths require a topology")
    changed = topology.changed_names(changed_paths) if topology else set()
    trace("PLAN_SCOPE", inventory_scope=inventory_scope, topology=topology is not None,
          changed_paths=list(changed_paths), forced_artifacts=sorted(changed))
    units, keys, seen = [], [], set()
    for record, key in records:
        name = record["artifact_identity"]
        if name in seen:
            raise ValueError(f"duplicate output identity: {name}")
        seen.add(name)
        if key is not None and (not isinstance(key, ArtifactKey) or key.artifact_identity != name
                                or key.components != tuple(record.get("components", ()))):
            raise ValueError("planner key does not match the requested output identity")
        parsed = parse_identity(name)
        forced = parsed is not None and parsed[0] in changed
        stage = (topology.stage_for(name) if topology else None) or "unmapped"
        unit = RequiredUnit(name, stage, key, "changed path" if forced else record.get("detail") or "",
                            tuple(record.get("components", ())))
        units.append((unit, forced))
        if key is not None and not forced:
            keys.append(key)
    found = db.lookup_batch(keys, read_policy)
    stages = {}
    for unit, forced in units:
        manifest = found.get(unit.cache_key) if not forced else None
        status = "uncacheable" if unit.cache_key is None else "candidate" if manifest else "miss"
        reason = unit.reason or ("verify stored payloads before reuse" if manifest else "no eligible production")
        trace("PLAN_ARTIFACT", artifact=unit.artifact_identity, stage=unit.stage,
              status=status, reason=reason, entry=manifest.cache_entry_id if manifest else None,
              payloads_verified=False, can_skip_stage=False)
        stages.setdefault(unit.stage, StagePlan(unit.stage)).entries.append(
            PlanEntry(unit, status, reason, manifest))
    order = {s: i for i, s in enumerate(topology.stages)} if topology else {}
    return CachePlan(sorted(stages.values(), key=lambda s: order.get(s.stage, len(order))), inventory_scope, dependency_impact)
