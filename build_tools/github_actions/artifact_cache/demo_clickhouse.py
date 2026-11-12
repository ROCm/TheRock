# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""End-to-end fingerprint-cache demo against a shared ClickHouse index.

Tells the whole story with the real planner and the real ClickHouse backend:

1. a first build publishes every artifact unit's fingerprint + outputs to the
   shared cache (``register_local_outputs``);
2. an unchanged rerun re-fingerprints to the same keys -> every unit is a cache
   candidate -> every stage can be skipped;
3. a run where one subproject's source changed re-fingerprints only that unit to
   a new key that was never published -> that unit misses and its stage must
   build, while every other stage still skips.

The planner (``plan_cache``) and the cache contract (``ArtifactKey`` /
``LocalManifest`` / ``ReadPolicy``) are exactly the ones the package ships; only
the DB backend is swapped to ClickHouse via ``CLICKHOUSE_URL``. Fingerprints
stand in for ``THEROCK_FPRINT``: a real build would read them from
``build/artifacts/*.fprint`` instead of deriving them here.
"""

import argparse
import hashlib
import sys

from .manifest import ArtifactFile, ArtifactKey, TrustLevel
from .cache_db import ReadPolicy
from .planner import plan_cache
from .clickhouse_cache_db import ClickHouseCacheDB

# (stage, artifact_identity) in build order. Mirrors a small slice of TheRock.
UNITS = [
    ("base", "sysdeps_lib_generic"),
    ("base", "hipcc_lib_generic"),
    ("core", "rocr_lib_gfx94X-dcgpu"),
    ("core", "hip_lib_gfx94X-dcgpu"),
    ("math-libs", "blas_lib_gfx94X-dcgpu"),
    ("math-libs", "fft_lib_gfx94X-dcgpu"),
]
STAGES = ["base", "core", "math-libs"]
EPOCH = 1


class DemoTopology:
    """Minimal topology matching what plan_cache calls."""

    def __init__(self, unit_to_stage: dict[str, str]):
        self._unit_to_stage = unit_to_stage
        self.stages = STAGES

    def stage_for(self, identity: str) -> str | None:
        return self._unit_to_stage.get(identity)

    def changed_names(self, changed_paths) -> set[str]:
        return set(changed_paths)  # demo passes names directly


def fingerprint(identity: str, revision: str) -> str:
    """Stand-in for THEROCK_FPRINT: stable hex per (identity, source revision)."""
    return hashlib.sha256(f"{identity}@{revision}".encode()).hexdigest()


def key_for(identity: str, revision: str) -> ArtifactKey:
    return ArtifactKey(identity, fingerprint(identity, revision), epoch=EPOCH)


def records_and_keys(revisions: dict[str, str]):
    """Build (collector-record, ArtifactKey) pairs for every required unit."""
    pairs = []
    for _stage, identity in UNITS:
        record = {"artifact_identity": identity, "components": (), "detail": ""}
        pairs.append((record, key_for(identity, revisions[identity])))
    return pairs


def publish_build(db: ClickHouseCacheDB, revisions: dict[str, str]) -> int:
    for _stage, identity in UNITS:
        key = key_for(identity, revisions[identity])
        db.register_local_outputs(
            key,
            local_directory=f"/cache/obj/{identity}",
            files=[ArtifactFile(f"lib/{identity}.so", "d" * 64, 4096)],
            validation_state="build_passed",
            producer_label="demo-build",
            producer_commit=revisions[identity],
            trust=TrustLevel.TRUSTED,
        )
    return db.count()


def run_plan(db: ClickHouseCacheDB, revisions: dict[str, str], title: str) -> None:
    topology = DemoTopology({identity: stage for stage, identity in UNITS})
    plan = plan_cache(
        db=db,
        records=records_and_keys(revisions),
        topology=topology,
        read_policy=ReadPolicy(),
    )
    print(f"\n=== {title} ===")
    skipped = built = 0
    for sp in plan.stage_plans:
        hits = sum(e.status == "candidate" for e in sp.entries)
        total = len(sp.entries)
        # Verified-skip decision (the CI-integration layer the planner defers):
        # every required unit is a policy-eligible candidate -> skip the stage.
        can_skip = hits == total and total > 0
        verb = "SKIP " if can_skip else "BUILD"
        skipped += can_skip
        built += not can_skip
        misses = [
            e.unit.artifact_identity for e in sp.entries if e.status != "candidate"
        ]
        detail = "" if can_skip else "  (build: " + ", ".join(misses) + ")"
        print(f"  [{verb}] {sp.stage:<10} {hits}/{total} units cached{detail}")
    print(f"  -> {skipped}/{len(STAGES)} stages skipped, {built} built")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="demo_clickhouse", description=__doc__)
    parser.add_argument(
        "--reset", action="store_true", help="truncate the cache table first"
    )
    parser.add_argument(
        "--changed",
        default="blas_lib_gfx94X-dcgpu",
        help="artifact identity whose source 'changed' in the third scenario",
    )
    args = parser.parse_args(argv)

    db = ClickHouseCacheDB()
    if args.reset:
        db._execute(f"TRUNCATE TABLE {db._cfg.database}.cache_entries")

    base_rev = "commit-A"
    revisions = {identity: base_rev for _stage, identity in UNITS}

    print(f"Shared ClickHouse cache index: {db._cfg.url}")
    print(f"First build at {base_rev}: publishing {len(UNITS)} artifact units...")
    print(f"  cache now holds {publish_build(db, revisions)} ready entries")

    run_plan(db, revisions, "Scenario 1: unchanged rerun (same commit)")

    changed = args.changed
    if changed not in revisions:
        print(f"\nunknown --changed identity: {changed}", file=sys.stderr)
        return 2
    revisions2 = dict(revisions)
    revisions2[changed] = "commit-B"  # only this subproject's source changed
    run_plan(
        db,
        revisions2,
        f"Scenario 2: only {changed} changed (new fingerprint, never published)",
    )
    print(
        "\nResult: the changed unit misses and its stage rebuilds; every other "
        "stage is served from the fingerprint cache."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
