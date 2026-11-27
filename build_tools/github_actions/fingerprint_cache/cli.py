#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Command-line entry point for the fingerprint cache.

Subcommands::

    python -m fingerprint_cache ingest  --db cache.sqlite --max-runs 50
    python -m fingerprint_cache plan     --db cache.sqlite --commit <sha> \
        --targets gfx94X-dcgpu --changed math-libs/BLAS/rocBLAS
    python -m fingerprint_cache stats    --db cache.sqlite
    python -m fingerprint_cache lookup   --db cache.sqlite --key <cache-key>

``ingest`` reaches the real GitHub API and (for artifact listing) the CI S3
buckets, so it needs the usual TheRock auth. ``plan`` reads the committed
``BUILD_TOPOLOGY.toml`` and the local SQLite cache only.
"""


import argparse
import json
import sys

from .cache_db import CacheDB, ReadPolicy
from .manifest import TrustLevel, ValidationState
from .planner import plan_cache


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def cmd_ingest(args: argparse.Namespace) -> int:
    from .history_ingest import ingest_runs, make_s3_artifact_lister, query_recent_runs

    runs = query_recent_runs(
        github_repository=args.repo,
        workflow_name=args.workflow,
        branch=args.branch,
        max_runs=args.max_runs,
    )
    lister = make_s3_artifact_lister(repository=args.repo, platform=args.platform)
    with CacheDB(args.db) as db:
        stats = ingest_runs(
            db=db,
            runs=runs,
            artifact_lister=lister,
            repository=args.repo,
            build_variant=args.variant,
            require_trusted=args.require_trusted,
        )
        print(json.dumps({"ingest": stats.to_dict(), "cache": db.stats()}, indent=2))
    return 0


def _read_policy(args: argparse.Namespace) -> ReadPolicy:
    return ReadPolicy(
        min_trust=TrustLevel(args.min_trust),
        min_validation=ValidationState(args.min_validation),
    )


def cmd_plan(args: argparse.Namespace) -> int:
    from .topology_view import BuildTopologyView

    # Make TheRock's _therock_utils importable for the topology parser.
    from pathlib import Path

    here = Path(__file__).resolve()
    sys.path.insert(0, str(here.parents[2]))  # build_tools/

    topology = BuildTopologyView.from_topology_path(args.topology)
    with CacheDB(args.db) as db:
        plan = plan_cache(
            topology=topology,
            changed_paths=_split_csv(args.changed),
            target_families=_split_csv(args.targets),
            source_commit=args.commit,
            db=db,
            build_variant=args.variant,
            read_policy=_read_policy(args),
            restrict_to_stages=_split_csv(args.stages) or None,
        )
    if args.json:
        print(json.dumps(plan.to_dict(), indent=2))
    else:
        print(plan.render())
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    with CacheDB(args.db) as db:
        print(json.dumps(db.stats(), indent=2))
    return 0


def cmd_lookup(args: argparse.Namespace) -> int:
    with CacheDB(args.db) as db:
        manifest = db.lookup(args.key, ReadPolicy(min_trust=TrustLevel.UNTRUSTED))
    if manifest is None:
        print(json.dumps({"key": args.key, "result": "miss"}, indent=2))
        return 1
    print(json.dumps(manifest.to_dict(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fingerprint_cache", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="ingest real run history into the cache")
    p_ingest.add_argument("--db", required=True)
    p_ingest.add_argument("--repo", default="ROCm/TheRock")
    p_ingest.add_argument("--workflow", default="multi_arch_ci.yml")
    p_ingest.add_argument("--branch", default="main")
    p_ingest.add_argument("--max-runs", type=int, default=20)
    p_ingest.add_argument("--platform", default="linux")
    p_ingest.add_argument("--variant", default="release")
    p_ingest.add_argument("--require-trusted", action="store_true")
    p_ingest.set_defaults(func=cmd_ingest)

    p_plan = sub.add_parser("plan", help="plan cache hits/misses for a run")
    p_plan.add_argument("--db", required=True)
    p_plan.add_argument("--commit", required=True)
    p_plan.add_argument("--changed", default="", help="comma-separated changed paths")
    p_plan.add_argument("--targets", default="", help="comma-separated target families")
    p_plan.add_argument("--stages", default="", help="restrict to these stages")
    p_plan.add_argument("--variant", default="release")
    p_plan.add_argument("--topology", default=None, help="path to BUILD_TOPOLOGY.toml")
    p_plan.add_argument(
        "--min-trust", default="trusted", choices=["trusted", "untrusted"]
    )
    p_plan.add_argument(
        "--min-validation",
        default="build_passed",
        choices=["unvalidated", "build_passed", "tests_passed"],
    )
    p_plan.add_argument("--json", action="store_true")
    p_plan.set_defaults(func=cmd_plan)

    p_stats = sub.add_parser("stats", help="print cache stats")
    p_stats.add_argument("--db", required=True)
    p_stats.set_defaults(func=cmd_stats)

    p_lookup = sub.add_parser("lookup", help="look up a single cache key")
    p_lookup.add_argument("--db", required=True)
    p_lookup.add_argument("--key", required=True)
    p_lookup.set_defaults(func=cmd_lookup)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
