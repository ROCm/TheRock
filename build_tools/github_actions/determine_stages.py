#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import tomllib
from collections import defaultdict, deque


TOPOLOGY_FILE = "BUILD_TOPOLOGY.toml"

def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        text=True,
    ).strip()


def changed_files(base: str, head: str) -> list[str]:
    diff = git("diff", "--name-only", f"{base}...{head}")
    return [x for x in diff.splitlines() if x.strip()]


def load_topology():
    with open(TOPOLOGY_FILE, "rb") as f:
        return tomllib.load(f)


def build_group_graph(topology):
    groups = topology["artifact_groups"]

    deps = defaultdict(set)
    reverse = defaultdict(set)

    for name, meta in groups.items():
        for d in meta.get("artifact_group_deps", []):
            deps[name].add(d)
            reverse[d].add(name)

    return deps, reverse


def build_stage_map(topology):
    stages = topology["build_stages"]

    group_to_stage = {}
    stage_to_groups = {}

    for stage, meta in stages.items():
        stage_to_groups[stage] = set(meta["artifact_groups"])
        for g in meta["artifact_groups"]:
            group_to_stage[g] = stage

    return group_to_stage, stage_to_groups


# ----------------------------
# naive path → group mapping
# ----------------------------

def map_path_to_groups(path: str) -> set[str]:
    p = path.lower()

    if "llvm-project" in p or "hipify" in p:
        return {"compiler"}

    if "rocm-libraries" in p:
        return {"math-libs", "ml-libs"}

    if "rocm-systems" in p:
        # affects almost everything
        return {
            "base",
            "core-runtime",
            "hip-runtime",
            "opencl-runtime",
            "runtime-tests",
            "profiler-core",
            "comm-libs",
            "storage-libs",
            "debug-tools",
            "dctools-core",
            "profiler-apps",
            "media-libs",
            "rocjitsu",
        }

    return set()


# ----------------------------
# propagate dependencies
# ----------------------------

def closure(seed: set[str], reverse_deps: dict[str, set[str]]) -> set[str]:
    q = deque(seed)
    out = set(seed)

    while q:
        g = q.popleft()
        for dep in reverse_deps.get(g, []):
            if dep not in out:
                out.add(dep)
                q.append(dep)

    return out


# ----------------------------
# main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--base", required=True)
    ap.add_argument("--head", required=True)

    args = ap.parse_args()

    topo = load_topology()

    group_deps, reverse_deps = build_group_graph(topo)
    group_to_stage, stage_to_groups = build_stage_map(topo)

    files = changed_files(args.base, args.head)

    if not files:
        print(json.dumps({"stages": []}, indent=2))
        return

    impacted_groups = set()

    for f in files:
        impacted_groups |= map_path_to_groups(f)

    # propagate group-level dependencies
    impacted_groups = closure(impacted_groups, reverse_deps)

    # map to stages
    impacted_stages = set()

    for g in impacted_groups:
        if g in group_to_stage:
            impacted_stages.add(group_to_stage[g])

    if not impacted_stages:
        impacted_stages = set(stage_to_groups.keys())

    print(json.dumps({
        "changed_files": files,
        "impacted_groups": sorted(impacted_groups),
        "impacted_stages": sorted(impacted_stages),
    }, indent=2))


if __name__ == "__main__":
    main()