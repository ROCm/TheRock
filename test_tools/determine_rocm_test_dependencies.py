# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Compute subproject test dependencies from the consumer graph.

The consumer graph is a flat reverse-dependency map,
{ subproject: { consumers: [...] } }, produced by therock_emit_consumer_graph()
and read from disk by this tool. Each subproject's `consumers` are the projects
that directly depend on it.

Algorithm
---------
1. Load the consumer graph.
2. For each changed subproject, walk its `consumers` edges to a depth set by the
   component's policy level (see the level ladder below).
3. UNION the per-subproject walk results across all changed subprojects.
4. Apply include/exclude policy: union in `test_include`, then subtract
   `test_exclude` last so the result is order-independent.

Level ladder (walk depth over consumer edges)
---------------------------------------------
* Level 5 -> depth 0: self only.
* Level 4 -> depth 1: self + direct consumers (one hop). The default.
* Level 3 -> unbounded: self + transitive consumers (full BFS closure).

The per-component level, plus test_include/test_exclude, come from the
hand-maintained `test_tools/test_policies.toml`. A component with no table there
uses the default level 4 (depth 1) with no include/exclude. See the schema
reference in docs/rfcs/RFC0013-Consumer-Based-Test-Selection.md.

Example
-------
$ python test_tools/determine_rocm_test_dependencies.py --changed-projects rocSPARSE
["hipsparse", "rocsolver", "rocsparse"]
"""

import argparse
import json
import sys
import tomllib
from collections import deque
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "build_tools" / "github_actions")
)
from github_actions_api import gha_set_output

# Level -> BFS walk depth over `consumers` edges:
#   depth 0    = self only
#   depth 1    = self + direct consumers
#   unbounded  = self + transitive consumers (full closure)
# Levels 1 and 2 currently alias to level 3 (transitive).
# TODO: distinguish them via a per-project test-tier output (e.g. full-QA/nightly).
_DEFAULT_LEVEL = 4
_LEVEL_TO_DEPTH: dict[int, int | None] = {5: 0, 4: 1, 3: None, 2: None, 1: None}

# Walk depth -> (label, human description), used to render --explain.
_DEPTH_DESCRIPTION: dict[int | None, tuple[str, str]] = {
    0: ("0", "self only"),
    1: ("1", "self + direct consumers"),
    None: ("unbounded", "self + transitive consumers"),
}

_CONSUMER_GRAPH_NAME = "therock_consumer_graph.json"
_TEST_POLICIES_NAME = "test_policies.toml"

# The committed graph and policy live next to this tool in test_tools/. Reading
# them script-relative lets selection work from a clean checkout with no build/.
_TEST_TOOLS_DIR = Path(__file__).resolve().parent

_EXTERNAL_SUBTREE_ALIASES = {
    "shared/rocroller": "rocroller",
    "dnn-providers/hipblaslt-provider": "hipblasltprovider",
    "dnn-providers/hip-kernel-provider": "hipkernelprovider",
    "dnn-providers/miopen-provider": "miopenprovider",
}


def _test_tools_file(name: str, therock_dir: Path | None) -> Path:
    """Resolve a committed file under test_tools/.

    Defaults to the copy committed next to this tool. A `--therock-dir` override
    points at that directory's test_tools/ copy instead, so callers can select
    against an alternate checkout.
    """
    if therock_dir is None:
        return _TEST_TOOLS_DIR / name
    return Path(therock_dir) / "test_tools" / name


def _consumer_graph_path(therock_dir: Path | None) -> Path:
    """Resolve the committed consumer graph path (test_tools/<graph>.json)."""
    return _test_tools_file(_CONSUMER_GRAPH_NAME, therock_dir)


def _load_consumer_graph(therock_dir: Path | None = None) -> dict:
    """Load the committed consumer graph JSON.

    Read directly from test_tools/ — no configure, no source fetch. Freshness is
    enforced by the test_consumer_graph_drift.yml CI job.
    """
    graph_path = _consumer_graph_path(therock_dir)
    if not graph_path.exists():
        raise FileNotFoundError(
            f"Consumer graph not found at {graph_path}.\n"
            "It is a committed, generated artifact. If it is missing, regenerate "
            "it with a configure (`cmake -B build -GNinja -DTHEROCK_ENABLE_ALL=ON "
            "...`) and copy build/therock_consumer_graph.json to "
            "test_tools/therock_consumer_graph.json."
        )
    return json.loads(graph_path.read_text())


def _test_policies_path(therock_dir: Path | None) -> Path:
    """Resolve the hand-maintained test-policy path (test_tools/<policies>.toml)."""
    return _test_tools_file(_TEST_POLICIES_NAME, therock_dir)


def _load_policies(therock_dir: Path | None = None) -> dict[str, dict]:
    """Return per-component policy overrides { subproject -> policy }.

    Reads test_tools/test_policies.toml. Each `[component.<name>]` table becomes a
    normalized, lowercased policy
    { "level": int, "test_include": [...], "test_exclude": [...] } (missing level
    -> 4, missing lists -> []). The file is committed and must be present; a
    missing file is an error. Schema: RFC0013-Consumer-Based-Test-Selection.md.
    """
    policies_path = _test_policies_path(therock_dir)
    if not policies_path.exists():
        raise FileNotFoundError(
            f"Test policy file not found at {policies_path}.\n"
            "It is a committed file expected to be present in every checkout."
        )

    raw = tomllib.loads(policies_path.read_text())
    components = raw.get("component", {})
    if not isinstance(components, dict):
        raise ValueError(
            "test_policies.toml: [component] must be a table of "
            "[component.<name>] entries, not "
            f"{type(components).__name__}"
        )

    policies: dict[str, dict] = {}
    for name, body in components.items():
        level = body.get("level", _DEFAULT_LEVEL)
        # Reject non-int levels (bool is an int subclass, so exclude it too).
        if not isinstance(level, int) or isinstance(level, bool):
            raise ValueError(
                f"test_policies.toml: component '{name}' has non-integer level "
                f"{level!r}; level must be an integer in [1, 5]"
            )
        policies[name.lower()] = {
            "level": level,
            "test_include": [c.lower() for c in body.get("test_include", [])],
            "test_exclude": [c.lower() for c in body.get("test_exclude", [])],
        }
    return policies


def _level_for(policies: dict[str, dict], proj: str) -> int:
    """Return the walk level for a component, defaulting to level 4."""
    return policies.get(proj, {}).get("level", _DEFAULT_LEVEL)


def _walk_consumers(graph: dict, start: str, max_depth: int | None) -> set[str]:
    """BFS over `consumers` edges from `start`, bounded by `max_depth`.

    depth 0 = {start}; depth 1 = start + direct consumers; max_depth None =
    unbounded transitive closure. Always includes `start` itself (so max_depth 0
    yields {start} via the loop's depth guard). Build stage is never consulted.
    """
    seen = {start}
    queue: deque[tuple[str, int]] = deque([(start, 0)])
    while queue:
        node, depth = queue.popleft()
        if max_depth is not None and depth >= max_depth:
            continue
        for consumer in graph.get(node, {}).get("consumers", []):
            if consumer not in seen:
                seen.add(consumer)
                queue.append((consumer, depth + 1))
    return seen


def _apply_policies(
    result: set[str], changed: list[str], policies: dict[str, dict]
) -> set[str]:
    """Apply include (union) then exclude (subtract) for all changed projects.

    Include/exclude are keyed on the changed subproject. Excludes are applied in
    a second pass, after all includes, so exclude is order-independent: an
    exclude for one changed project is not silently undone by another changed
    project unioning the same consumer.
    """
    for proj in changed:
        result.update(policies.get(proj, {}).get("test_include", []))
    for proj in changed:
        result.difference_update(policies.get(proj, {}).get("test_exclude", []))
    return result


def get_subprojects_to_test(
    changed_subprojects: list[str],
    therock_dir: Path | None = None,
    level: int | None = None,
) -> set[str]:
    """Return the set of subproject names (lowercase) to test.

    For each changed subproject, walk its `consumers` edges by hop distance to
    the depth set by its policy level, union across changed subprojects, then
    apply test_include (union) and test_exclude (subtract, last).

    `level` overrides the per-component level for ALL changed subprojects (used
    for CLI/testing). When None, each component's level comes from its policy in
    test_policies.toml (default 4).

    When `therock_dir` is None the committed graph next to this tool is used, so
    selection works from a clean checkout with no build/ directory.

    Inputs and outputs are consumer-graph keys (the lowercased subproject names in
    therock_consumer_graph.json). This function only lowercases; the CLI handles
    the external-repo subtree identifiers it receives from GitHub Actions by
    stripping a leading `projects/` and mapping known non-project subtree paths
    to graph keys. Callers holding other identifiers must map them to graph keys
    first — notably, subtree paths like `projects/clr` (-> `hip-clr`) and the
    hyphenated CI test-matrix keys (`hipdnn-integration-tests` vs the graph's
    `hipdnn_integration_tests`). That mapping is tracked separately.
    """
    graph = _load_consumer_graph(therock_dir)
    policies = _load_policies(therock_dir)

    changed_lower = [p.lower() for p in changed_subprojects]

    # Warn on unrecognized projects (typo guard).
    known = set(graph.keys())
    unknown = [p for p in changed_lower if p not in known]
    if unknown:
        print(
            f"Warning: unrecognized project(s) {unknown}; "
            "no tests will be selected for them",
            file=sys.stderr,
        )

    result: set[str] = set()
    for proj in changed_lower:
        proj_level = level if level is not None else _level_for(policies, proj)
        max_depth = _LEVEL_TO_DEPTH.get(proj_level, _LEVEL_TO_DEPTH[_DEFAULT_LEVEL])
        result |= _walk_consumers(graph, proj, max_depth)

    return _apply_policies(result, changed_lower, policies)


def explain_component(component: str, therock_dir: Path | None = None) -> str:
    """Return a human-readable derivation of one component's test selection.

    Shows, for `component`, the fully resolved selection and how it was reached:
    the effective policy `level` and resulting walk depth, the graph-walk (BFS)
    result at that depth, the `test_include` values unioned in (marking any that
    are test-only, i.e. not graph keys), the `test_exclude` values subtracted,
    and the final selection set. Output is sorted/deterministic.

    The final set is computed by calling the real selection function
    (`get_subprojects_to_test`) so `--explain` can never drift from what selection
    actually computes. The graph-walk line is shown separately for insight into
    how that final set was reached.
    """
    graph = _load_consumer_graph(therock_dir)
    policies = _load_policies(therock_dir)

    comp = component.lower()
    known = set(graph.keys())

    level = _level_for(policies, comp)
    max_depth = _LEVEL_TO_DEPTH.get(level, _LEVEL_TO_DEPTH[_DEFAULT_LEVEL])
    depth_label, depth_desc = _DEPTH_DESCRIPTION[max_depth]

    include = policies.get(comp, {}).get("test_include", [])
    exclude = policies.get(comp, {}).get("test_exclude", [])

    walk = _walk_consumers(graph, comp, max_depth)
    # Call the real selection so the final set cannot drift from CI.
    final = get_subprojects_to_test([comp], therock_dir)

    lines: list[str] = []
    lines.append(f"Explain: {comp}")
    if comp not in known:
        lines.append(
            f"\tWARNING: '{comp}' is not a consumer-graph key; walk is self-only."
        )
    lines.append(f"\tlevel:\t{level} (walk depth {depth_label}: {depth_desc})")
    lines.append("\tgraph walk:\t" + ", ".join(sorted(walk)))
    if include:
        marked = [f"{v} (test-only)" if v not in known else v for v in sorted(include)]
        lines.append("\ttest_include:\t" + ", ".join(marked))
    else:
        lines.append("\ttest_include:\t(none)")
    if exclude:
        lines.append("\ttest_exclude:\t" + ", ".join(sorted(exclude)))
    else:
        lines.append("\ttest_exclude:\t(none)")
    lines.append("\tfinal:\t" + ", ".join(sorted(final)))
    return "\n".join(lines)


def validate_policies(therock_dir: Path | None = None) -> tuple[bool, list[str]]:
    """Validate test_policies.toml against the committed consumer graph.

    Referential-integrity check (RFC0013 "CI integration"): needs only the two
    committed files, no configure/fetch. Returns (ok, messages).

    FAILS (ok is False) if:
      * any `[component.<name>]` KEY is not a consumer-graph key — a stale /
        renamed / removed component that could never fire the selection. Each
        offending key is listed.
      * any `level` is not an implemented level (a key of `_LEVEL_TO_DEPTH`).
        The range is derived from the engine, so validation can never bless a
        level the walk would silently coerce to the default.

    Does NOT fail on `test_include` / `test_exclude` VALUES absent from the graph:
    those are legitimately test-only targets (ctest suites / tool targets like
    rocgdb-cpu, hipinfo) the graph cannot express. Such values are collected into
    an informational note only (never a failure).
    """
    graph = _load_consumer_graph(therock_dir)
    policies = _load_policies(therock_dir)
    graph_keys = set(graph.keys())

    errors: list[str] = []
    stale_keys = sorted(k for k in policies if k not in graph_keys)
    for key in stale_keys:
        errors.append(
            f"stale component key '{key}': not a consumer-graph key "
            "(renamed/removed component? it can never trigger selection)"
        )

    valid_levels = sorted(_LEVEL_TO_DEPTH)
    lo, hi = valid_levels[0], valid_levels[-1]
    bad_levels: list[str] = []
    test_only_values: set[str] = set()
    for name, body in sorted(policies.items()):
        level = body.get("level", _DEFAULT_LEVEL)
        if level not in _LEVEL_TO_DEPTH:
            bad_levels.append(
                f"component '{name}': level {level} is outside [{lo}, {hi}]"
            )
        for value in body.get("test_include", []) + body.get("test_exclude", []):
            if value not in graph_keys:
                test_only_values.add(value)
    errors.extend(bad_levels)

    messages: list[str] = []
    if errors:
        messages.append("::error::test_policies.toml validation failed:")
        messages.extend(f"  - {e}" for e in errors)
        return False, messages

    note = ""
    if test_only_values:
        note = (
            f"; {len(test_only_values)} test-only value(s) noted (not graph keys): "
            + ", ".join(sorted(test_only_values))
        )
    messages.append(
        f"OK: {len(policies)} component key(s) validated against "
        f"{len(graph_keys)} graph node(s){note}"
    )
    return True, messages


def list_subprojects(therock_dir: Path | None = None, show_deps: bool = False):
    """List all subprojects known to the consumer graph.

    When `therock_dir` is None the committed graph next to this tool is read, so
    this works from a clean checkout with no build/ directory (finding #8).
    """
    graph = _load_consumer_graph(therock_dir)

    if show_deps:
        return {
            name: (entry["consumers"] if entry.get("consumers") else "empty")
            for name, entry in sorted(graph.items())
        }
    return sorted(graph.keys())


def _normalize_changed_project(project: str) -> str:
    """Normalize CI subtree identifiers to consumer-graph keys for the CLI."""
    project_lower = project.lower()
    if project_lower in _EXTERNAL_SUBTREE_ALIASES:
        return _EXTERNAL_SUBTREE_ALIASES[project_lower]
    return project_lower.removeprefix("projects/")


def main():
    parser = argparse.ArgumentParser(
        description="Compute subproject test dependencies from the consumer graph"
    )
    parser.add_argument(
        "--therock-dir",
        type=str,
        default=None,
        help="TheRock directory. When omitted, the committed consumer graph next "
        "to this tool (test_tools/therock_consumer_graph.json) is used, so this "
        "works from a clean checkout with no build/ directory.",
    )
    parser.add_argument(
        "--changed-projects",
        type=str,
        nargs="*",
        metavar="PROJECT",
        help="Project(s) that changed. Accepts space- or comma-separated list. "
        "Supports graph keys, 'projects/<name>', and known external subtree "
        "paths such as 'shared/rocroller'.",
    )
    parser.add_argument(
        "--level",
        type=int,
        choices=sorted(_LEVEL_TO_DEPTH),
        help="Override the walk level for all changed projects (default: each "
        "component's policy level, or 4). 5=self, 4=direct, 3=transitive.",
    )
    parser.add_argument(
        "--explain",
        type=str,
        metavar="COMPONENT",
        help="Print the fully resolved test selection for COMPONENT and how it "
        "was derived (level, graph walk, includes/excludes, final set), using "
        "the same resolution code path as real selection.",
    )
    parser.add_argument(
        "--validate-policies",
        action="store_true",
        help="Validate test_policies.toml against the committed consumer graph "
        "(each component key must be a graph key; levels must be in [1, 5]). "
        "Needs only the committed files; exits non-zero on any violation.",
    )
    parser.add_argument(
        "--list-subprojects", action="store_true", help="List all known subprojects"
    )
    parser.add_argument(
        "--show-deps",
        action="store_true",
        help="With --list-subprojects, show consumers (or 'empty' if none)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "list"],
        default="json",
        help="Output format: json (default) or list (newline-separated)",
    )
    parser.add_argument(
        "--gha-output",
        action="store_true",
        help="Write projects_to_test to GITHUB_OUTPUT",
    )

    args = parser.parse_args()
    therock_dir = Path(args.therock_dir).resolve() if args.therock_dir else None

    if args.validate_policies:
        ok, messages = validate_policies(therock_dir)
        if not ok:
            raise SystemExit("\n".join(messages))
        for line in messages:
            print(line)
        return

    if args.explain:
        print(explain_component(args.explain, therock_dir))
        return

    if args.list_subprojects:
        result = list_subprojects(therock_dir, show_deps=args.show_deps)
        print(json.dumps(result, indent=2))
        return

    # Parse and normalize changed_projects
    changed = args.changed_projects
    if changed:
        flattened = []
        for item in changed:
            flattened.extend(p.strip() for p in item.split(",") if p.strip())
        changed = flattened

    if changed:
        changed = [_normalize_changed_project(p) for p in changed]

    # No projects specified → all tests
    if not changed:
        if args.gha_output:
            gha_set_output({"projects_to_test": "*"})
        else:
            print("*")
        return

    result = get_subprojects_to_test(changed, therock_dir, level=args.level)
    projects_to_test = ",".join(sorted(result))

    if args.gha_output:
        gha_set_output({"projects_to_test": projects_to_test})
    elif args.format == "json":
        print(json.dumps(sorted(result)))
    else:
        for item in sorted(result):
            print(item)


if __name__ == "__main__":
    main()
