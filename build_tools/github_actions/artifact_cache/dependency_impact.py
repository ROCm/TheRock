# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Explain configured dependency impact; this is not a build-skip decision."""

from collections import deque
import json
from pathlib import Path


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate graph field: {key}")
        result[key] = value
    return result


def _read(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"cannot read configured dependency metadata {path}: {exc}") from exc


def _names(value):
    return isinstance(value, list) and all(isinstance(n, str) and n.strip() for n in value)


def configured_impact(build, changed_subprojects):
    """Return one shortest dependency path per root and affected artifact.

    CMake emits lowercase graph names but preserves target spelling in the
    artifact map. Resolve that spelling difference, rejecting graph collisions.
    Both files must come from the same successful configuration.
    """
    build = Path(build)
    graph = _read(build / "therock_consumer_graph.json")
    artifacts = _read(build / "artifact_subprojects.json")
    if not isinstance(graph, dict) or not isinstance(artifacts, dict):
        raise ValueError("configured dependency metadata must contain JSON objects")
    edges = {}
    for name, node in graph.items():
        if not name.strip() or not isinstance(node, dict) or not _names(node.get("consumers")):
            raise ValueError(f"invalid consumer graph node: {name}")
        canonical = name.casefold()
        if canonical in edges:
            raise ValueError(f"ambiguous consumer graph target: {name}")
        edges[canonical] = sorted(set(n.casefold() for n in node["consumers"]))
    for consumers in list(edges.values()):
        for consumer in consumers:
            if consumer not in edges:
                raise ValueError(f"consumer missing from graph: {consumer}")
    owners = {}
    for artifact, projects in artifacts.items():
        if not artifact.strip() or not _names(projects):
            raise ValueError(f"invalid artifact mapping: {artifact}")
        owners[artifact] = sorted(set(p.casefold() for p in projects))
        missing = set(owners[artifact]) - edges.keys()
        if missing:
            raise ValueError(f"artifact {artifact} references unknown targets: {sorted(missing)}")
    roots = sorted(set(n.casefold() for n in changed_subprojects))
    if not roots or any(root not in edges for root in roots):
        raise ValueError(f"unknown or empty changed subprojects: {roots}")
    reasons = {}
    affected = set()
    for root in roots:
        paths = {root: [root]}
        queue = deque([root])
        while queue:
            current = queue.popleft()
            for consumer in edges[current]:
                if consumer not in paths:
                    paths[consumer] = paths[current] + [consumer]
                    queue.append(consumer)
        affected.update(paths)
        for artifact, projects in owners.items():
            matches = [paths[p] for p in projects if p in paths]
            if matches:
                path = min(matches, key=lambda p: (len(p), p))
                reasons.setdefault(artifact, []).append({
                    "changed_subproject": root,
                    "contributing_subproject": path[-1],
                    "dependency_path": path,
                })
    return {
        "source": "configured_cmake_graph",
        "scope": "dependency_impact_only",
        "changed_subprojects": roots,
        "affected_subprojects": sorted(affected),
        "affected_artifacts": [
            {"artifact": name, "reasons": reasons[name]} for name in sorted(reasons)
        ],
        "unmapped_subprojects": sorted(affected - {p for ps in owners.values() for p in ps}),
        "build_selection_changed": False,
    }
