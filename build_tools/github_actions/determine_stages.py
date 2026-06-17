#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import tomllib
from collections import defaultdict, deque


TOPOLOGY_FILE = "BUILD_TOPOLOGY.toml"

def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def changed_files(base: str, head: str) -> list[str]:
    out = git("diff", "--name-only", f"{base}...{head}")
    return [x for x in out.splitlines() if x.strip()]

def load_topology():
    with open(TOPOLOGY_FILE, "rb") as f:
        return tomllib.load(f)

def build_source_set_index(topo):
    """
    submodule -> source_set mapping
    """
    index = {}

    for set_name, meta in topo["source_sets"].items():
        for sm in meta.get("submodules", []):
            index[sm] = set_name

    return index


def build_source_set_to_groups(topo):
    """
    source_set -> artifact_groups
    """
    mapping = defaultdict(set)

    for gname, gmeta in topo["artifact_groups"].items():
        for ss in gmeta.get("source_sets", []):
            mapping[ss].add(gname)

    return mapping


def build_group_deps(topo):
    deps = defaultdict(set)
    reverse = defaultdict(set)

    for g, meta in topo["artifact_groups"].items():
        for d in meta.get("artifact_group_deps", []):
            deps[g].add(d)
            reverse[d].add(g)

    return deps, reverse


def build_stage_map(topo):
    group_to_stage = {}
    stage_to_groups = {}

    for stage, meta in topo["build_stages"].items():
        stage_to_groups[stage] = set(meta["artifact_groups"])
        for g in meta["artifact_groups"]:
            group_to_stage[g] = stage

    return group_to_stage, stage_to_groups


def extract_submodule(path: str) -> str | None:
    parts = path.split("/")
    return parts[0] if parts else None

def closure(seed: set[str], reverse_deps: dict[str, set[str]]) -> set[str]:
    q = deque(seed)
    out = set(seed)

    while q:
        x = q.popleft()
        for n in reverse_deps.get(x, []):
            if n not in out:
                out.add(n)
                q.append(n)

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--head", required=True)
    args = ap.parse_args()

    topo = load_topology()

    submodule_to_source_set = build_source_set_index(topo)
    source_set_to_groups = build_source_set_to_groups(topo)
    _, group_reverse_deps = build_group_deps(topo)
    group_to_stage, stage_to_groups = build_stage_map(topo)

    files = changed_files(args.base, args.head)

    impacted_groups = set()

    for f in files:
        sm = extract_submodule(f)
        if not sm:
            continue

        source_set = submodule_to_source_set.get(sm)
        if not source_set:
            continue

        impacted_groups |= source_set_to_groups.get(source_set, set())

    impacted_groups = closure(impacted_groups, group_reverse_deps)
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