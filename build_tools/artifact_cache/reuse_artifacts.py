#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Turn an artifact-cache lookup plan into artifact-level reuse inputs.

`artifact_cache.py lookup` decides which artifacts a build could reuse and writes
a plan (``populate`` when every requested artifact matches a prior run, else
``build``). This script reads that plan and emits the ``reusable_artifacts`` /
``rebuild_artifacts`` / ``baseline_run_id`` values that the multi-arch build
already consumes for *granular, artifact-level* reuse within stages (the
``copy_prebuilt_stages`` job restores ``--artifacts`` from one run; each stage
build configures only ``rebuild_artifacts``).

This follows RFC0017: the unit of reuse is a **restore group** -- one artifact
name and target, with all of its components -- not a whole build stage. A lookup
hit is a complete restore group (the lookup requires every component to match a
single fingerprint), so each hit is reused and everything else is rebuilt. That
is exactly RFC0017 milestone 2: reuse ``amd-llvm`` from the cache while the rest
of the compiler/runtime stage keeps building.

Safety:
  * Only a ``populate`` plan with at least one hit produces reuse.
  * Every hit must come from a single producer run, because the copy step
    restores from one ``baseline_run_id``; a plan spanning runs yields no reuse
    (safe fallback to a full build).
  * ``rebuild_artifacts`` is the complement of the hits over the whole build
    graph, so no reused artifact is also rebuilt.

Used behind the ``ci:reuse-fprint`` label, so the default CI path is unchanged.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def load_topology(topology_path):
    """Load TheRock's BuildTopology (lazy import so tests can inject their own)."""
    from _therock_utils.build_topology import get_topology

    return get_topology(Path(topology_path) if topology_path else None)


def plan_reuse(plan: dict, topology) -> tuple[list[str], list[str], str | None]:
    """Return (reusable_artifacts, rebuild_artifacts, baseline_run_id).

    Empty reusable / None baseline means "no reuse": the caller falls back to a
    normal build. ``rebuild_artifacts`` is only meaningful alongside a non-empty
    ``reusable_artifacts``.
    """
    if plan.get("action") != "populate":
        return [], [], None
    hits = [a for a in plan.get("artifacts", []) if a.get("status") == "hit"]
    if not hits:
        return [], [], None
    run_ids = {a["provenance"]["run_id"] for a in hits}
    if len(run_ids) != 1:
        # copy restores from a single run; refuse a plan that spans runs.
        return [], [], None
    baseline_run_id = next(iter(run_ids))

    # A hit is a complete restore group (lookup verified all components against
    # one fingerprint), so reuse by artifact name.
    reusable = {a["name"] for a in hits}

    # Rebuild everything else the build graph would produce. The complement keeps
    # a reused artifact from also being rebuilt, and lets reused artifacts act as
    # dependencies of the artifacts that still build.
    produced: set[str] = set()
    for stage in topology.get_build_order():
        produced |= set(topology.get_produced_artifacts(stage))
    rebuild = produced - reusable

    return sorted(reusable), sorted(rebuild), baseline_run_id


def _write_output(
    reusable: list[str], rebuild: list[str], baseline_run_id: str | None
) -> None:
    values = {
        "reusable_artifacts": ",".join(reusable),
        "rebuild_artifacts": ",".join(rebuild),
        "baseline_run_id": baseline_run_id or "",
    }
    for key, value in values.items():
        print(f"{key}={value}")
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as handle:
            for key, value in values.items():
                handle.write(f"{key}={value}\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        required=True,
        help="lookup plan JSON (from artifact_cache lookup)",
    )
    parser.add_argument(
        "--topology",
        type=Path,
        default=None,
        help="BUILD_TOPOLOGY.toml (default: repo root)",
    )
    args = parser.parse_args(argv)

    if not args.plan.is_file():
        # No plan (lookup found nothing selectable) -> no reuse, not an error.
        _write_output([], [], None)
        return 0
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    reusable, rebuild, baseline_run_id = plan_reuse(plan, load_topology(args.topology))
    _write_output(reusable, rebuild, baseline_run_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
