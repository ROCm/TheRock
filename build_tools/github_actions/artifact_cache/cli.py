#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Relational artifact cache: planning, managed publication, restore and history."""

import argparse

from .diagnostics import add_logging_args, run_logged, trace
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

from .cache_db import CacheDB, ReadPolicy
from .manifest import ArtifactKey, OutputObject, TrustLevel, ValidationState
from .reporting import log_report


def _policy(args):
    return ReadPolicy(min_trust=TrustLevel(args.min_trust),
                      min_validation=ValidationState(args.min_validation),
                      created_before=args.created_before)


def _policy_args(parser):
    parser.add_argument("--min-trust", choices=[v.value for v in TrustLevel], default="untrusted")
    parser.add_argument("--min-validation", choices=[v.value for v in ValidationState], default="build_passed")
    parser.add_argument("--created-before", help="ISO timestamp with timezone; filters publication/validation time")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest="command", required=True)
    for name in ("stats", "lookup", "plan", "ingest", "replay", "evict", "publish", "restore"):
        s = subs.add_parser(name)
        add_logging_args(s)
        s.add_argument("--db", type=Path, required=True, help="existing relational cache.db")
        if name in ("lookup", "plan", "replay", "restore"):
            _policy_args(s)
        if name in ("plan", "publish", "restore"):
            s.add_argument("--build", type=Path, required=True)
            s.add_argument("--expected", type=Path, required=name != "plan")
            s.add_argument("--granularity", choices=["bundle", "component"], default="bundle")
        if name == "lookup":
            s.add_argument("--artifact", required=True, help="full component or artifact/bundle/target identity")
            s.add_argument("--components", default="", help="comma-separated component identities for a bundle key")
            s.add_argument("--fingerprint", required=True, help="raw 64-hex CMake fingerprint")
        elif name == "plan":
            s.add_argument("--topology", type=Path)
            s.add_argument("--changed", default="", help="optional comma-separated paths forcing rebuild")
            s.add_argument("--changed-subproject", action="append", default=[],
                           help="CMake target whose configured downstream impact to report; repeatable")
            s.add_argument("--targets", default="", help="optional emitted family suffixes; generic is included")
        elif name == "ingest":
            s.add_argument("--runs-json", type=Path, help="offline run list with artifacts and optional source_fingerprints")
            s.add_argument("--repo", default="ROCm/TheRock")
            s.add_argument("--workflow", default="multi_arch_ci.yml")
            s.add_argument("--branch", default="main")
            s.add_argument("--max-runs", type=int, default=20)
            s.add_argument("--platform", default="linux")
            s.add_argument("--variant", default="release")
            s.add_argument("--require-trusted", action="store_true")
        elif name == "replay":
            s.add_argument("--repo")
            s.add_argument("--topology", type=Path)
        elif name == "evict":
            group = s.add_mutually_exclusive_group(required=True)
            group.add_argument("--before", help="retire entries created before this zoned ISO time")
            group.add_argument("--missing", action="store_true", help="retire entries with missing local objects")
        elif name in ("publish", "restore"):
            s.add_argument("--cache", type=Path, required=True)
            if name == "publish":
                s.add_argument("--label", default="managed-publication")
                s.add_argument("--jobs", type=int)
                s.add_argument("--producer-trust", choices=[v.value for v in TrustLevel], default="untrusted",
                               help="local policy label, not an authentication mechanism")
            else:
                s.add_argument("--destination", type=Path, required=True)
    return p


def execute(args):
    if args.command in ("publish", "restore"):
        common = dict(database=args.db, expected_manifest=args.expected, build_dir=args.build,
                      cache_root=args.cache, granularity=args.granularity)
        if args.command == "publish":
            from .publish import build_and_publish
            return build_and_publish(**common, label=args.label, jobs=args.jobs,
                                     producer_trust=TrustLevel(args.producer_trust)), 0
        from .restore import restore_local_outputs
        report = restore_local_outputs(**common, destination=args.destination, policy=_policy(args))
        return report, int(report["needs_build"] > 0)
    with CacheDB(args.db) as db:
        if args.command == "stats":
            return {**db.stats(), "history_runs": len(db.history())}, 0
        if args.command == "lookup":
            key = ArtifactKey(args.artifact, args.fingerprint,
                              components=tuple(sorted(args.components.split(","))) if args.components else ())
            result = db.lookup(key, _policy(args))
            return (result.to_dict() if result else {"result": "miss", "artifact_identity": args.artifact}), int(result is None)
        if args.command == "plan":
            from .inventory import TopologyAdapter, parse_identity
            from .local_cache import prepared_scan
            from .planner import plan_cache
            topology = TopologyAdapter.from_path(args.topology) if args.topology else None
            with prepared_scan(args.build, args.expected, granularity=args.granularity) as records:
                impact = None
                if args.changed_subproject:
                    from .dependency_impact import configured_impact
                    impact = configured_impact(args.build, args.changed_subproject)
                if args.targets:
                    requested = set(args.targets.split(",")) | {"generic"}
                    records = [(r, key) for r, key in records
                               if not parse_identity(r["artifact_identity"])
                               or parse_identity(r["artifact_identity"])[2] in requested]
                    present = {parse_identity(r["artifact_identity"])[2] for r, _ in records
                               if parse_identity(r["artifact_identity"])}
                    if missing := requested - {"generic"} - present:
                        raise ValueError("no configured outputs for targets: " + ", ".join(sorted(missing)))
                if impact is not None:
                    affected = {item["artifact"] for item in impact["affected_artifacts"]}
                    impact["affected_requested_outputs"] = sorted(
                        r["artifact_identity"] for r, _ in records
                        if parse_identity(r["artifact_identity"])
                        and parse_identity(r["artifact_identity"])[0] in affected
                    )
                return plan_cache(db=db, records=records, topology=topology, read_policy=_policy(args),
                                  dependency_impact=impact,
                                  changed_paths=[v for v in args.changed.split(",") if v],
                                  inventory_scope=("explicit_outputs" if args.expected else
                                                   "configured_ninja_and_observed_outputs" if args.granularity == "bundle"
                                                   else "observed_outputs")).to_dict(), 0
        if args.command == "ingest":
            from .history_ingest import ingest_runs, make_s3_artifact_lister, query_recent_runs
            if args.runs_json:
                runs = json.loads(args.runs_json.read_text(encoding="utf-8"))
                if not isinstance(runs, list):
                    raise ValueError("--runs-json must contain an array")
                def lister(run):
                    return [OutputObject.from_dict(o) for o in run.get("artifacts", [])]
            else:
                if args.max_runs < 1:
                    raise ValueError("--max-runs must be positive")
                runs = query_recent_runs(github_repository=args.repo, workflow_name=args.workflow,
                                         branch=args.branch, max_runs=args.max_runs)
                lister = make_s3_artifact_lister(repository=args.repo, platform=args.platform)
            return ingest_runs(db=db, runs=runs, artifact_lister=lister, repository=args.repo,
                               build_variant=args.variant, platform=args.platform,
                               require_trusted=args.require_trusted).to_dict(), 0
        if args.command == "replay":
            from .inventory import TopologyAdapter
            from .replay import replay
            topology = TopologyAdapter.from_path(args.topology) if args.topology else None
            return replay(db.history(args.repo), read_policy=_policy(args), topology=topology).to_dict(), 0
        if args.command == "evict":
            count = db.evict_before(args.before) if args.before else db.evict_where_missing(lambda p: Path(p).is_file())
            return {"retired_entries": count, "payloads_deleted": False}, 0
    raise ValueError("unknown command")


def main(argv=None):
    args = build_parser().parse_args(argv)
    return run_logged(args, _execute_logged)


def _execute_logged(args):
    try:
        report, code = execute(args)
        print(json.dumps(report, indent=2, allow_nan=False))
        log_report(report)
        return code
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, sqlite3.Error, subprocess.SubprocessError) as exc:
        trace("OPERATION_FAILED", error_type=type(exc).__name__, reason=str(exc))
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
