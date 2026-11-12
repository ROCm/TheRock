# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fingerprint-driven stage reuse for multi-arch CI (replaces baseline selection).

The baseline mechanism reuses a stage by finding one prior run whose archives
cover it. This computes the same decision from fingerprints instead: it reads
the real ``THEROCK_FPRINT`` sidecars a configure produced, looks each artifact
unit up in the shared ClickHouse cache, and marks a stage reusable when every
one of its units is a cache hit. The producer run recorded on the hit manifests
supplies the ``baseline_run_id`` the existing "Copy Prebuilt Stages" step uses,
so only the *decision* changes, not the fetch.

Because real fingerprints already encode source and dependency changes, a
changed subproject gets a new fingerprint, misses, and forces its stage to
rebuild -- no separate changed-path analysis is needed here.

Every ClickHouse call is timed (per-call latency, call count, key count) so a CI
run also produces database timing/API-call data, which is a goal of the POC.
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

from .cache_key import current_cache_epoch
from .manifest import ArtifactKey
from .inventory import parse_identity

FINGERPRINT_SUFFIX = ".fprint"
_HEX = set("0123456789abcdef")


@dataclass
class ClickHouseTiming:
    """Per-call ClickHouse timing, for the POC's DB latency/API-call report."""

    ops: list = field(default_factory=list)  # (label, ms, n_keys)

    def add(self, label: str, ms: float, n_keys: int) -> None:
        self.ops.append((label, ms, n_keys))

    @property
    def calls(self) -> int:
        return len(self.ops)

    @property
    def total_ms(self) -> float:
        return sum(ms for _l, ms, _n in self.ops)

    @property
    def total_keys(self) -> int:
        return sum(n for _l, _ms, n in self.ops)

    def summary(self) -> dict:
        times = [ms for _l, ms, _n in self.ops]
        return {
            "clickhouse_calls": self.calls,
            "keys_looked_up": self.total_keys,
            "total_ms": round(self.total_ms, 2),
            "mean_ms": round(sum(times) / len(times), 2) if times else 0.0,
            "max_ms": round(max(times), 2) if times else 0.0,
            "min_ms": round(min(times), 2) if times else 0.0,
            # First call pays TLS/connection (and serverless wake); split it out.
            "first_call_ms": round(times[0], 2) if times else 0.0,
        }

    def render(self) -> str:
        s = self.summary()
        return (
            f"ClickHouse: {s['clickhouse_calls']} calls, {s['keys_looked_up']} keys, "
            f"total {s['total_ms']}ms, mean {s['mean_ms']}ms, "
            f"first {s['first_call_ms']}ms, max {s['max_ms']}ms"
        )


@dataclass
class FingerprintReuseResult:
    reusable_stages: list
    rebuild_stages: list
    baseline_run_id: str | None
    stage_hits: dict  # stage -> (hits, total)
    timing: ClickHouseTiming
    report_lines: list


def read_fingerprints(build_dir) -> list:
    """Real THEROCK_FPRINT sidecars: ``(artifact_identity, fingerprint)`` pairs.

    Dirty-tree / invalid fingerprints (not 64 lowercase hex) are skipped -- they
    are uncacheable, so their stage will rebuild.
    """
    artifacts = Path(build_dir) / "artifacts"
    if not artifacts.is_dir():
        artifacts = Path(build_dir)
    pairs = []
    for path in sorted(artifacts.glob("*" + FINGERPRINT_SUFFIX)):
        identity = path.name[: -len(FINGERPRINT_SUFFIX)]
        fingerprint = path.read_text().strip()
        if len(fingerprint) == 64 and set(fingerprint) <= _HEX:
            pairs.append((identity, fingerprint))
    return pairs


def stage_index_from_topology(view) -> dict:
    """Map artifact name -> stage from a BuildTopologyView (or its topology)."""
    topology = getattr(view, "_t", view)
    index = {}
    for stage in topology.get_build_order():
        for name in topology.get_produced_artifacts(stage):
            index[name] = stage
    return index


def _stage_for(name_index: dict, identity: str) -> str:
    parsed = parse_identity(identity)
    name = parsed[0] if parsed else identity
    return name_index.get(name, "unmapped")


def compute_fingerprint_reuse(
    *,
    build_dir,
    stage_for,
    db=None,
    read_policy=None,
    epoch=None,
) -> FingerprintReuseResult:
    """Decide reusable stages from fingerprints in ``build_dir`` via ClickHouse.

    ``stage_for`` maps an artifact identity to its stage. ``db`` is a
    ClickHouseCacheDB (constructed from the environment if omitted). One batched
    lookup per stage keeps the API-call count low and the timing legible.
    """
    if db is None:
        from .clickhouse_cache_db import ClickHouseCacheDB

        db = ClickHouseCacheDB()
    epoch = current_cache_epoch() if epoch is None else epoch

    units = read_fingerprints(build_dir)
    stage_units: dict = {}
    for identity, fingerprint in units:
        stage = stage_for(identity) or "unmapped"
        key = ArtifactKey(identity, fingerprint, epoch=epoch)
        stage_units.setdefault(stage, []).append((identity, key))

    timing = ClickHouseTiming()
    reusable, rebuild = [], []
    stage_hits: dict = {}
    run_votes: dict = {}
    for stage, items in sorted(stage_units.items()):
        keys = [key for _identity, key in items]
        started = perf_counter()
        found = db.lookup_batch(keys, read_policy)
        timing.add(f"lookup:{stage}", (perf_counter() - started) * 1000.0, len(keys))
        hit_manifests = [found.get(key) for _identity, key in items]
        hits = sum(m is not None for m in hit_manifests)
        stage_hits[stage] = (hits, len(items))
        if items and hits == len(items):
            reusable.append(stage)
            for manifest in hit_manifests:
                label = manifest.producer_label
                run_votes[label] = run_votes.get(label, 0) + 1
        else:
            rebuild.append(stage)

    baseline_run_id = max(run_votes, key=run_votes.get) if run_votes else None
    report = [
        "### Fingerprint stage reuse",
        "",
        f"{len(reusable)} stage(s) reusable, {len(rebuild)} to rebuild "
        f"(baseline_run_id={baseline_run_id or '-'})",
        "",
        "| stage | units cached | decision |",
        "|---|---|---|",
    ]
    for stage in sorted(stage_hits):
        hits, total = stage_hits[stage]
        verb = "REUSE" if stage in reusable else "REBUILD"
        report.append(f"| {stage} | {hits}/{total} | {verb} |")
    report += ["", "**" + timing.render() + "**"]
    return FingerprintReuseResult(
        reusable_stages=reusable,
        rebuild_stages=rebuild,
        baseline_run_id=baseline_run_id,
        stage_hits=stage_hits,
        timing=timing,
        report_lines=report,
    )


def emit_timing_artifact(result: FingerprintReuseResult) -> None:
    """Write the ClickHouse timing summary to $CLICKHOUSE_TIMING_JSON if set."""
    path = os.environ.get("CLICKHOUSE_TIMING_JSON")
    if not path:
        return
    payload = {
        "reusable_stages": result.reusable_stages,
        "rebuild_stages": result.rebuild_stages,
        "baseline_run_id": result.baseline_run_id,
        "clickhouse_timing": result.timing.summary(),
        "per_call_ms": [
            {"label": label, "ms": round(ms, 3), "keys": n}
            for label, ms, n in result.timing.ops
        ],
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Standalone stage entry point: run locally or as a separate CI job.
# ---------------------------------------------------------------------------


def _write_output(name: str, value: str) -> None:
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(f"{name}={value}\n")


def _append_summary(text: str) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")


def _load_stage_for(topology_path):
    """Map identity -> stage from the real BuildTopology; degrade gracefully."""
    try:
        from .topology_view import BuildTopologyView

        view = BuildTopologyView.from_topology_path(topology_path)
        index = stage_index_from_topology(view)
        return lambda identity: _stage_for(index, identity)
    except Exception as exc:  # topology optional: keep the stage runnable anywhere
        print(
            f"[fingerprint-stage] topology unavailable ({exc}); "
            "using artifact name as stage"
        )
        return lambda identity: (parse_identity(identity) or (identity,))[0]


def _publish(build_dir, pairs, db) -> None:
    from .manifest import ArtifactFile, ArtifactKey, TrustLevel

    run_label = os.environ.get("GITHUB_RUN_ID", "local")
    commit = os.environ.get("GITHUB_SHA", "")
    for identity, fingerprint in pairs:
        db.register_local_outputs(
            ArtifactKey(identity, fingerprint),
            local_directory=f"artifacts/{identity}",
            files=[ArtifactFile(f"{identity}.fprint", "d" * 64, 64)],
            validation_state="build_passed",
            producer_label=run_label,
            producer_commit=commit,
            trust=TrustLevel.TRUSTED,
        )
    print(f"[fingerprint-stage] published {len(pairs)} units as run {run_label}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="fingerprint-stage",
        description="Fingerprint-driven stage reuse decision via the ClickHouse cache.",
    )
    parser.add_argument(
        "--build",
        default=os.environ.get("STAGE_REUSE_FPRINT_BUILD_DIR", "build"),
        help="build dir containing artifacts/*.fprint",
    )
    parser.add_argument("--topology", default=None, help="BUILD_TOPOLOGY.toml path")
    parser.add_argument(
        "--publish",
        action="store_true",
        help="also publish this build's fingerprints for later runs to reuse",
    )
    args = parser.parse_args(argv)

    pairs = read_fingerprints(args.build)
    if not pairs:
        print(f"[fingerprint-stage] no .fprint under {args.build}; no reuse computed")
        _write_output("prebuilt_stages", "")
        _write_output("baseline_run_id", "")
        return 0

    from .clickhouse_cache_db import ClickHouseCacheDB

    db = ClickHouseCacheDB()
    result = compute_fingerprint_reuse(
        build_dir=args.build, stage_for=_load_stage_for(args.topology), db=db
    )
    emit_timing_artifact(result)
    _append_summary("\n".join(result.report_lines))
    prebuilt = ",".join(result.reusable_stages)
    _write_output("prebuilt_stages", prebuilt)
    _write_output("baseline_run_id", result.baseline_run_id or "")
    print(result.timing.render())
    print(f"prebuilt_stages={prebuilt}  baseline_run_id={result.baseline_run_id}")
    if args.publish:
        _publish(args.build, pairs, db)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
