# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CI entry points for the fingerprint cache against a shared ClickHouse index.

Three subcommands a CI job runs as separate steps/jobs, so publish and plan
cross process and run boundaries exactly as they do across real CI runs:

* ``ping``    - preflight connectivity check (fails fast with a clear message).
* ``plan``    - look every required unit up in the shared cache and report, per
                stage, whether it can be skipped. Writes a GitHub step summary
                and ``$GITHUB_OUTPUT`` values. This is the reuse decision.
* ``publish`` - record this build's units in the shared cache so later runs hit.

Units come from either ``--build DIR`` (real ``build/artifacts/*.fprint`` from a
configured TheRock build; the filename is the artifact identity and the file
holds the fingerprint) or ``--synthetic`` (a fixed small stage graph that needs
no build, for demonstrating the loop on a GitHub-hosted runner).
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from .manifest import ArtifactFile, ArtifactKey, TrustLevel
from .cache_db import ReadPolicy
from .planner import plan_cache
from .clickhouse_cache_db import ClickHouseCacheDB

EPOCH = 1

# Synthetic stage graph (stage, artifact_identity) in build order.
_SYNTHETIC_UNITS = [
    ("base", "sysdeps_lib_generic"),
    ("base", "hipcc_lib_generic"),
    ("core", "rocr_lib_gfx94X-dcgpu"),
    ("core", "hip_lib_gfx94X-dcgpu"),
    ("math-libs", "blas_lib_gfx94X-dcgpu"),
    ("math-libs", "fft_lib_gfx94X-dcgpu"),
]


class _Topology:
    """Minimal topology matching what plan_cache calls (changed_names/stage_for/stages)."""

    def __init__(self, unit_to_stage, stages):
        self._unit_to_stage = unit_to_stage
        self.stages = stages

    def stage_for(self, identity):
        return self._unit_to_stage.get(identity, "build")

    def changed_names(self, changed_paths):
        return set(changed_paths)


def _synthetic_units(commit):
    units = []
    for stage, identity in _SYNTHETIC_UNITS:
        fp = hashlib.sha256(f"{identity}@{commit}".encode()).hexdigest()
        units.append((stage, identity, fp))
    return units


def _build_units(build_dir):
    """Read real fingerprints from ``<build>/artifacts/*.fprint``."""
    artifacts = Path(build_dir) / "artifacts"
    if not artifacts.is_dir():
        artifacts = Path(build_dir)
    units = []
    for path in sorted(artifacts.glob("*.fprint")):
        identity = path.name[: -len(".fprint")]
        fingerprint = path.read_text().strip()
        if len(fingerprint) != 64 or any(
            c not in "0123456789abcdef" for c in fingerprint
        ):
            continue  # dirty-tree / invalid fingerprint is uncacheable; skip it
        units.append(("build", identity, fingerprint))
    return units


def _load_units(args):
    if args.build:
        units = _build_units(args.build)
        if not units:
            print(f"no valid .fprint files under {args.build}", file=sys.stderr)
    else:
        units = _synthetic_units(args.commit)
    # Simulate a source change: give the named units a fresh fingerprint that was
    # never published, so they become cache misses (for dispatch-time testing).
    changed = set(getattr(args, "changed", None) or [])
    if changed:
        units = [
            (
                stage,
                identity,
                (
                    hashlib.sha256((fp + ":changed").encode()).hexdigest()
                    if identity in changed
                    else fp
                ),
            )
            for stage, identity, fp in units
        ]
    return units


def _db():
    return ClickHouseCacheDB()


def _topology(units):
    unit_to_stage = {identity: stage for stage, identity, _fp in units}
    stages = []
    for stage, _identity, _fp in units:
        if stage not in stages:
            stages.append(stage)
    return _Topology(unit_to_stage, stages)


def _emit_github(summary_lines, outputs):
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        Path(summary).open("a", encoding="utf-8").write("\n".join(summary_lines) + "\n")
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with Path(out).open("a", encoding="utf-8") as fh:
            for key, value in outputs.items():
                fh.write(f"{key}={value}\n")


def cmd_ping(args):
    print("ClickHouse version:", _db().ping())
    return 0


def cmd_reset(args):
    db = _db()
    db._execute(f"TRUNCATE TABLE {db._cfg.database}.cache_entries")
    print("cache table truncated")
    return 0


def cmd_publish(args):
    db = _db()
    units = _load_units(args)
    if not units:
        return 1
    for _stage, identity, fp in units:
        key = ArtifactKey(identity, fp, epoch=EPOCH)
        db.register_local_outputs(
            key,
            local_directory=f"/cache/obj/{identity}",
            files=[ArtifactFile(f"lib/{identity}.so", "d" * 64, 4096)],
            validation_state="build_passed",
            producer_label=args.run_label,
            producer_commit=args.commit,
            trust=TrustLevel.TRUSTED,
        )
    print(f"published {len(units)} units; cache now holds {db.count()} ready entries")
    _emit_github(
        [f"### Fingerprint cache publish", f"- published **{len(units)}** units"],
        {"published": len(units)},
    )
    return 0


def cmd_plan(args):
    db = _db()
    units = _load_units(args)
    if not units:
        return 1
    records = [
        (
            {"artifact_identity": identity, "components": (), "detail": ""},
            ArtifactKey(identity, fp, epoch=EPOCH),
        )
        for _stage, identity, fp in units
    ]
    plan = plan_cache(
        db=db, records=records, topology=_topology(units), read_policy=ReadPolicy()
    )
    summary = [
        "### Fingerprint cache plan",
        "",
        "| stage | cached | decision |",
        "|---|---|---|",
    ]
    skipped = built = total_hits = 0
    for sp in plan.stage_plans:
        hits = sum(e.status == "candidate" for e in sp.entries)
        total = len(sp.entries)
        total_hits += hits
        can_skip = total > 0 and hits == total
        skipped += can_skip
        built += not can_skip
        verb = "SKIP" if can_skip else "BUILD"
        print(f"  [{verb:<5}] {sp.stage:<12} {hits}/{total} cached")
        summary.append(f"| {sp.stage} | {hits}/{total} | {verb} |")
        for e in sp.entries:
            mark = "HIT " if e.status == "candidate" else "MISS"
            print(f"         {mark}  {e.unit.artifact_identity}  ({e.reason})")
            summary.append(
                f"| &nbsp;&nbsp;{mark} | {e.unit.artifact_identity} | {e.reason} |"
            )
    n_units = len(units)
    print(
        f"  -> {skipped} stage(s) skippable, {built} to build; {total_hits}/{n_units} units cached"
    )
    summary.append("")
    summary.append(
        f"**{skipped} stage(s) skippable, {total_hits}/{n_units} units cached.**"
    )
    _emit_github(
        summary,
        {
            "skippable_stages": skipped,
            "units_cached": total_hits,
            "units_total": n_units,
        },
    )
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ci_cache", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn, helptext in [
        ("ping", cmd_ping, "check ClickHouse connectivity"),
        ("plan", cmd_plan, "report per-stage skip decision from the cache"),
        ("publish", cmd_publish, "publish this build's units to the cache"),
        ("reset", cmd_reset, "truncate the shared cache table"),
    ]:
        p = sub.add_parser(name, help=helptext)
        if name != "ping":
            src = p.add_mutually_exclusive_group()
            src.add_argument(
                "--build", default=None, help="build dir with artifacts/*.fprint"
            )
            src.add_argument(
                "--synthetic",
                action="store_true",
                help="use a fixed demo stage graph (no build required)",
            )
            p.add_argument(
                "--changed",
                action="append",
                default=[],
                help="artifact identity whose source 'changed' (fresh fingerprint -> miss); repeatable",
            )
            p.add_argument("--commit", default=os.environ.get("GITHUB_SHA", "local"))
            p.add_argument(
                "--run-label", default=os.environ.get("GITHUB_RUN_ID", "local")
            )
        p.set_defaults(func=fn)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
