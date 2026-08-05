---
author: Abhilash Reddy Endurthi (endurthiabhilash)
created: 2026-07-28
modified: 2026-08-04
status: draft
---

# Derive Test-Subproject Selection From the Consumer Graph

This RFC proposes replacing the partially-populated, hand-maintained
`TEST_SUBPROJECTS` lists in component `CMakeLists.txt` files with a **consumer
graph** generated from the build's own dependency edges and committed to the repo
(kept honest by a CI drift check), combined with a small set of explicit
test-policy declarations in a dedicated file. The goal is to make change-driven
test selection ("when subproject X changes, which test suites must run?") derive
from the dependency edges the build already declares, rather than from a parallel
list that must be manually kept in sync.

## Background: gating levels

This work exists to implement per-PR **gating** across TheRock's components: a
per-component tier that fixes *how much* every PR must build and test, sized to the
component's downstream blast radius and regression history. The levels below are a
public restatement of the internal gating scheme; the mechanism in this RFC does
not depend on the specific names or numbering (see the naming note in Open
questions).

The levels are points on a single axis — **how far to walk the dependency graph
from a changed component** — plus a test-tier dimension at the extremes:

| Level | What every PR must build & test             | Graph reach                       | Intended for                               |
| ----- | ------------------------------------------- | --------------------------------- | ------------------------------------------ |
| **1** | Full QA integration + nightly suites        | everything + nightly dispatch     | components actively destabilizing releases |
| **2** | Framework-facing consumers built and tested | a named frameworks group          | high downstream blast radius               |
| **3** | All downstream consumers (transitive)       | **transitive** closure over edges | broad-dependency / regression history      |
| **4** | The component + its **direct** consumers    | **one hop** over consumer edges   | well-tested, contained surface             |
| **5** | Fast unit tests only                        | `{self}`, unit tier only          | proven track record + strong local tests   |

The motivating failure mode is exactly what gating names: a component change
passes its own local tests, then surfaces a regression days later during a
downstream module bump. Gating pulls that discovery left: a PR must exercise the
components it can break *before* it merges, not after.

Lower numbers are stricter (walk more of the graph); higher numbers are cheaper.
The rollout is staged: **Phase 1 starts every component at level 4**, and gating is
tightened (4 → 3 → 2 → 1) only for components that accumulate regression history.
The long-term goal moves the other way: promote proven components *up* to level 5
(fast unit tests only) so the mainline moves fast where it safely can. Each level
is sized against a per-PR test-time target defined by the CI/gating program; the
level-4 default (direct consumers only) is what keeps a typical PR proportional to
the change rather than triggering whole-tree runs.

Level 4 — the component plus its **directly-dependent** consumers — is precisely
the selection this RFC automates as the default: for a changed subproject, run it
plus the projects that directly depend on it (one hop along the reverse edges), and
**not** their indirect downstream (that is level 3). For example, a change to
`rocRoller` selects `hipBLASLt` (which directly depends on it) but not `MIOpen`
(which depends on `hipBLASLt`, so only *indirectly* on `rocRoller`). Doing that for
*every* component requires a reliable, low-maintenance answer to "what directly
depends on X?" across the whole tree. The hand-maintained `TEST_SUBPROJECTS` lists
were that answer in principle but not in practice (below); the consumer graph makes
the level-4 baseline derivable rather than curated, and the per-component level
becomes the knob (how many hops to walk) for moving a component off that baseline.

## Motivation

TheRock CI selects which GPU test suites to run based on which projects changed in
a PR. That mapping is expressed with a `TEST_SUBPROJECTS` argument on
`therock_cmake_subproject_declare()`. The key exists today, but only a subset of
subprojects populate it. Most declarations carry no `TEST_SUBPROJECTS` at all, so
a change to those projects selects nothing beyond itself. The mechanism is sound;
the coverage is incomplete, and completing it by hand has proven impractical:

- **Manual duplication.** `TEST_SUBPROJECTS` restates dependency relationships
  that are *already* declared via `BUILD_DEPS` / `RUNTIME_DEPS`. When a new
  consumer of a library is added, the library's `TEST_SUBPROJECTS` must be edited
  too, or its tests silently stop running on relevant changes.
- **Drift and false greens.** A stale or missing entry produces a passing CI run
  that never exercised the affected downstream suite: the most dangerous class of
  gap because it is invisible.
- **Distributed ownership.** The lists live across component `CMakeLists.txt`
  files in `math-libs`, `ml-libs`, `debug-tools`, `media-libs`, and `profiler/`,
  with no single place to reason about the full change→test mapping.

Two prior attempts to close the gap stalled:

1. **Populate `TEST_SUBPROJECTS` by hand for every project.** A PR that added the
   missing component names across all declarations was opened but never merged:
   the list was large, error-prone to review, and would need re-editing on every
   dependency change.
1. **Commit a generated consumer graph alongside a static `overrides.json`.** A
   follow-up explored emitting the graph, checking it into the repo, and layering
   a committed `test_subprojects_overrides.json`. This was also never merged.

The dependency information needed to answer "what runs when X changes?" already
exists: every subproject declares its build and runtime dependencies. This RFC
makes test selection a *derived* property of those declarations, so no exhaustive
hand-maintained list is required.

## Proposal

### Two files with opposite lifecycles

Test selection is split across two files, deliberately kept separate:

1. **`therock_consumer_graph.json` — generated, CI-owned.** The full reverse
   dependency graph, emitted from the build system. Never hand-edited; a CI drift
   check regenerates it and fails on any difference (see CI integration).
1. **`test_policies.toml` — hand-maintained, human-reviewed.** The small set of
   escape hatches the graph cannot derive: each component's gating level (how far
   to walk the graph) and any `test_include` / `test_exclude`.

They are separate files because they have **opposite lifecycles**. The graph is
machine-generated and must never be hand-edited — edits would be clobbered on
regeneration, and the whole point is that it cannot silently drift from the build
definitions. The policy file is human intent that CI cannot derive. Combining them
would create a file that is half-generated and half-authored, which the drift check
cannot cleanly "regenerate and diff": it would have to diff *around* the
hand-authored regions. Keeping them apart lets the graph be fully CI-owned
(regenerate-and-fail-on-diff) and the policy be fully human-owned (reviewed like
code). This mirrors ScottTodd's "generated from build-system knowledge, with a few
escape hatches" framing: the graph is the generated knowledge, the policy file is
the escape hatches.

The tradeoff is two places to look and cross-file referential integrity to
maintain; both are addressed in CI integration and Testing below.

### Overview

1. **Register consumers at declare time.** Every
   `therock_cmake_subproject_declare()` appends the subproject to a global
   registry and records a reverse edge (`consumer`) for each of its
   `BUILD_DEPS` / `RUNTIME_DEPS`.
1. **Emit and commit the consumer graph.** At the end of the top-level configure,
   `therock_emit_consumer_graph()` serializes the registry to
   `therock_consumer_graph.json`, which is **committed to the repo**. A CI check
   regenerates it and fails if the committed copy is stale, so the change-detection
   path reads a checked-in file with no configure on the hot path (see CI
   integration).
1. **Derive test subprojects in Python.** `determine_rocm_test_dependencies.py`
   loads the graph and, for each changed subproject, walks its consumer edges to
   the depth set by that component's level — one hop for the level-4 default
   (direct consumers), transitive for level 3.
1. **Apply explicit test policy from a dedicated file.** Couplings the graph
   cannot express (test-only projects with no link-time edge) and each component's
   gating level are declared in a standalone test-policy file
   (`test_include` / `test_exclude` and the per-component level), separate from
   `BUILD_TOPOLOGY.toml`.

### The consumer graph

The graph carries only consumer (reverse-dependency) edges:

```json
{
  "<subproject-lowercase>": { "consumers": ["<consumer-lowercase>", ...] },
  ...
}
```

CMake populates it during declaration:

```cmake
set_property(GLOBAL APPEND PROPERTY THEROCK_ALL_SUBPROJECTS "${target_name}")
foreach(_dep IN LISTS ARG_BUILD_DEPS ARG_RUNTIME_DEPS)
  set_property(GLOBAL APPEND PROPERTY "THEROCK_CONSUMERS_OF_${_dep}" "${target_name}")
endforeach()
```

The graph carries only the reverse edges. Nothing else about a subproject
(build stage, artifact) is duplicated into it, so the graph stays minimal and each
other fact keeps its existing single source of truth.

### Selection: walk the graph by hop-distance

For a changed subproject, selection is a bounded breadth-first walk over its
consumer edges. The number of hops is set by the component's gating level:

- **Level 4 (default) — direct consumers, one hop.** Select the changed
  subproject plus every project that *directly* depends on it. This is the
  "local + directly-dependent" tier. A change to `rocRoller` selects `hipBLASLt`
  (direct) but not `MIOpen` (indirect). A change to `amdsmi` selects `rdc` and
  its other direct consumers.
- **Level 3 — transitive consumers.** Walk all reachable consumer edges (BFS
  closure): "test everything downstream." For a foundational dep like `hip-clr`
  or `ROCR-Runtime` this is most of the tree — which is the point of level 3, and
  why it is opt-in per component rather than the default.

A declared `BUILD_DEPS` / `RUNTIME_DEPS` edge is deliberate, so a consumer that
depends on the changed project — directly or transitively — genuinely *can* be
broken by it. The graph is therefore the source of truth for "what could this
change affect"; the gating level chooses **how far along those edges to walk**,
trading cost against blast-radius coverage. The default stops at one hop to keep
per-PR test cost proportional to the change; widening a component to level 3 is an
explicit, reviewable change to its level, not a property of the graph's shape.

> **Why not a build-stage boundary?** An earlier draft cut selection at the
> `BUILD_TOPOLOGY` build-stage boundary (select only same-stage consumers). That
> was wrong: cross-stage dependencies are commonplace and intentional (nearly
> everything depends on the compiler `amd-hip`, the runtime `hip-clr`, and the
> profiler `rocprofiler-sdk`), so a stage cut discards real edges. Concretely, it
> would *miss* the amdsmi → rdc breakage class (they are in different stages,
> `compiler-runtime` vs `dctools-core`, so the cut drops `rdc` despite the direct
> edge), and it selects *nothing* for stage-less foundational deps like `hip-clr`.
> Direct-vs-indirect hop distance is the correct boundary; build stage is not.

### Overrides

Two optional keys layer explicit selection on top of the derived graph, for the
couplings the reverse edges cannot express:

- `test_include` — extra subprojects to test when this component changes, beyond
  its graph consumers. Covers test-only projects with no link-time edge
  (e.g. `hip-tests`, `rocgdb-cpu`) and real reverse edges not present as forward
  `BUILD_DEPS` / `RUNTIME_DEPS` consumers.
- `test_exclude` — consumers to prune even though they appear in the graph.
  Applied **last** so it can drop a specific consumer of an otherwise-selected
  set.

Note that `test_fanout_all` from the earlier draft is **removed**: it existed only
to escape the same-stage cut for foundational deps, and "select all consumers" is
now simply level 3 (transitive walk). A foundational dep that genuinely needs
broad testing is set to level 3 rather than carrying a bespoke fanout flag.

These declarations, together with each component's gating level, live in a
**dedicated test-policy file** rather than in `BUILD_TOPOLOGY.toml`. Test policy
(what to test, how far to walk) is a different concern from build topology (how
artifacts are grouped and staged); keeping it separate avoids overloading the
topology tables and gives one clear place to answer "what testing does a change to
this component trigger?". A sketch (exact schema is an [open
question](#open-questions)):

```toml
# test_policies.toml
[component.hipblaslt]
level = 4                      # default; may be omitted
test_include = ["hipsparselt"]

[component.rocm-core]
level = 3                      # foundational dep: walk transitive consumers
```

Keying by component (subproject) name sidesteps the multi-subproject-artifact
problem the earlier `BUILD_TOPOLOGY.toml` draft had, where one artifact packaging
several subprojects needed per-subproject sub-tables to avoid over-applying an
override to siblings.

Because the policy file references components by name, its keys and
`test_include` / `test_exclude` values must stay in sync with the graph: an entry
naming a renamed or removed component would otherwise sit stale and unnoticed. A
validation step checks every policy key/value against the current graph and fails
if any names an unknown component (see CI integration).

**Resolution order.** With a per-component level *and* include/exclude layered on
the generated graph, results must be deterministic: walk the graph to the
component's level, then apply `test_include` (union), then `test_exclude`
(subtract) last. `determine_rocm_test_dependencies.py` implements this order and a
`--explain <component>` mode prints the resolved selection (graph walk + policy
applied) so the composed result of the two files can be inspected without composing
it by hand.

### CI integration

The graph is emitted by a configure (`cmake -DTHEROCK_ENABLE_ALL=ON`), which
requires all subproject sources present. Doing that configure *inside the
per-PR change-detection job* costs ~6 minutes on a GitHub-hosted runner (~4m40s
checkout + ~1m40s configure) — too expensive to pay before every PR can even
shard its test jobs. An early draft that regenerated the graph dynamically on
every run was rejected for exactly this reason.

Instead, the emitted `therock_consumer_graph.json` is **committed to the repo**,
and the per-PR change-detection job reads the checked-in file directly — no
`cmake`, no source fetch on the hot path. A **drift check** keeps the committed
copy honest: a CI job (triggered only when files that affect the graph change —
`CMakeLists.txt`, the emit/registration cmake, the committed graph) regenerates
the graph from a configure and fails if it differs from the committed copy,
prompting the contributor to regenerate and re-commit. This confines the ~6-minute
configure cost to the low-frequency drift job instead of every PR, and makes the
graph reviewable in-repo. `BUILD_TOPOLOGY.toml` is committed under the same
contract, so this follows existing precedent.

A second, cheap check validates `test_policies.toml` against the graph: every
component key and every `test_include` / `test_exclude` value must name a component
that exists in the current graph, so a renamed or removed component cannot leave a
stale policy entry behind. This runs on every PR (it needs only the committed graph,
no configure).

Committing the graph also helps **external-repo CI**. Repos that run on multi-arch
CI (rocm-libraries, rocm-systems) can consume TheRock's committed graph directly
rather than each regenerating its own — they do not have the full source tree
present at configure time anyway, so a checked-in graph is strictly better for
them.

A later refinement: each build stage could emit the slice of the graph it knows
and a downstream step could merge them, avoiding the single `ENABLE_ALL` full-tree
configure in the drift job. Sequenced after the committed-graph + drift-check
lands.

## Generalizing to the full level ladder

The default is level 4, but level 4 is not special: it is one hop-distance over
the consumer graph. The levels form a monotone ladder of **graph reach**, so
supporting per-component levels is a matter of choosing the walk depth per changed
subproject. The graph and the `test_include` / `test_exclude` overrides are reused
unchanged; the level only decides how far to walk before overrides layer on top.

### The level ladder as walk depth

Each level is a different reach over the same consumer edges:

| Level | Reach over the graph                              | Notes                                   |
| ----- | ------------------------------------------------- | --------------------------------------- |
| **5** | `{self}` only, unit tier                          | seed set; needs the test-tier dimension |
| **4** | `{self}` + **direct** consumers (one hop)         | the default this RFC implements         |
| **3** | `{self}` + **transitive** consumers (BFS closure) | "test everything downstream"            |
| **2** | `{self}` + a named **frameworks** group           | frameworks group + overrides            |
| **1** | everything (`*`) + full-QA / nightly dispatch     | the existing `projects_to_test = "*"`   |

Levels 3, 4, and (by radius) 5 are expressible with the selection engine directly:
they are just BFS with depth 0, 1, and ∞. Level 2 adds a named group; level 1 adds
a workflow dispatch.

### What the generalization adds

1. **A per-component level** (default 4 when unset — "Phase 1 starts every
   component at level 4"). Its home is an [open question](#open-questions); the
   value is a single scalar per component either way.
1. **A depth-parameterized walk** in `get_subprojects_to_test()`: the selection
   becomes a BFS over `consumers` whose max depth is the level (4 → depth 1,
   3 → unbounded), replacing the special-cased branches of the earlier draft.
1. **A `frameworks` group** for level 2 (the PyTorch / JAX / hipDNN-facing
   consumers).

With this in place, setting a component's gating level is a one-line change with
no code edit: tightening a repeat offender (4 → 3 → 2 → 1) or promoting a proven
component (4 → 5) becomes a reviewed, auditable, single-line change.

### The tier dimension (why walk depth alone is not enough)

The selection engine decides *which subprojects* to test. Two levels are
statements about *which test tier within a subproject* runs, which walk depth
cannot express:

- **Level 5** is "fast unit tests only." Selecting `{self}` alone still runs
  *all* of self's tests, so without a tier filter level 5 collapses to a narrow
  level 4.
- **Level 1** is "full QA integration + nightly suites," a different workflow
  dispatch, not merely `projects_to_test = "*"`.

Honestly scoped: depth-based selection delivers clean levels **3, 4, and a rough
5** on top of what this RFC ships. True 1 / 2 / 5 need a **second output**
alongside `projects_to_test`: a per-project `test_tier` (unit / full / nightly)
that the test-runner job honors (a ctest label filter for unit-only, a workflow
dispatch for nightly). That tier plumbing touches the test-*execution* job, not
just selection, and is the genuinely new work; the depth ladder is a refactor of
the engine already built here. The recommended sequencing is to land the
level-key + depth-walk refactor first (immediate per-project 3/4/5-by-depth),
then add the tier dimension as a follow-up.

## Alternatives considered

### A: Fully populate `TEST_SUBPROJECTS` by hand (attempted)

Complete the existing mechanism by adding the missing component names to every
`therock_cmake_subproject_declare()`. A PR doing this was opened but never merged:
the list was large, hard to review for correctness, and would require re-editing on
every dependency change: the same drift/duplication problem, just fully expanded.

### B: Regenerate the graph dynamically on every run (rejected)

Emit the graph fresh in each per-PR change-detection job and never commit it, so it
can never go stale. Rejected on cost: the emit requires a full
`THEROCK_ENABLE_ALL=ON` configure (~6 min including source checkout) on the hot
path of every PR before test jobs can even be sharded. This RFC instead commits the
graph and guards it with a drift check (see CI integration), which confines that
configure cost to a low-frequency job and makes the graph reviewable in-repo. The
committed-graph choice does add a regenerate-and-recommit step on dependency edits,
but the drift check turns that from *silent* staleness into a *loud* CI failure —
the same contract `BUILD_TOPOLOGY.toml` already carries.

### C: Cut selection at the build-stage boundary

An earlier draft of this RFC selected only consumers in the *same*
`BUILD_TOPOLOGY` build stage as the changed subproject, treating cross-stage
consumers as "universal" and cutting them. Rejected: cross-stage dependencies are
commonplace and intentional (compiler, runtime, profiler), so the cut discards
real edges — it misses the amdsmi → rdc breakage class (different stages) and
selects nothing for stage-less foundational deps like `hip-clr`. Direct-vs-indirect
hop distance is the correct bound; build stage is not. (This also removes the need
to derive or encode a subproject → stage map for selection at all.)

### D: Always select all transitive consumers

Make transitive closure the *default* rather than a level. Rejected as a default:
it explodes into whole-tree test runs for any change to a foundational dependency,
blowing the per-PR SLA. Transitive closure is retained, but as the opt-in **level
3** for components that warrant it, not the baseline.

## Impact and migration

- **Removed:** the `TEST_SUBPROJECTS` parameter from
  `therock_cmake_subproject_declare()` and its (partial) uses across component
  `CMakeLists.txt` files.
- **Added:** `cmake/therock_emit_consumer_graph.cmake`; consumer registration in
  `cmake/therock_subproject.cmake`; the emit call in the top-level
  `CMakeLists.txt`; the committed `therock_consumer_graph.json`;
  `test_policies.toml` (`test_include` / `test_exclude` and per-component levels);
  graph-driven selection plus an `--explain` mode in
  `determine_rocm_test_dependencies.py`.
- **CI:** the per-PR change-detection job reads the committed graph directly and
  validates `test_policies.toml` against it; a low-frequency drift-check job
  regenerates the graph from a configure and fails on mismatch.
- **Behavioral parity:** existing couplings (e.g. rocGDB → rocgdb-cpu/gpu,
  hipCUB/rocThrust → rocPRIM, amdsmi → hip-tests/rocrtst) are preserved — as
  direct graph consumers or migrated `test_include` overrides — so selection
  matches prior behavior for known changes while additionally covering
  newly-added consumers automatically. Because the default now walks *direct*
  consumers regardless of stage, changes like amdsmi → rdc that the earlier
  stage-cut draft dropped are now selected.

## Testing

`test_tools/tests/determine_rocm_test_dependencies_test.py` is expanded to cover
the direct-consumer (one-hop) default, the transitive (level-3) walk, each
test-policy key, the per-component level, and the resolution order (walk → include
→ exclude). It also covers the policy-validation check (an override naming an
unknown component fails) and the `--explain` output.

## Open questions

- **Test-policy file schema and location.** Policy lives in a dedicated,
  hand-maintained file separate from the generated graph (settling the "topology is
  the wrong home" concern and the generated-vs-authored split). Still open: keyed
  strictly by subproject name, or by path glob so a directory of code can map to a
  level? And where the file lives — repo root vs. `test_tools/` (ScottTodd leaned
  toward more of this logic living in `test_tools/`).
- **Per-PR test-time targets.** Each level should be sized against the CI/gating
  program's per-PR test-time targets. Those targets exist but are maintained
  outside this repo; this RFC should cite them once they can be linked, and the
  level definitions may need tuning to match.
- **Universal deps at level 4.** Direct-consumers-only for a runtime like
  `hip-clr` is still ~42 projects — its true blast radius. Is that the intended
  level-4 cost for foundational deps, or should specific universal deps carry a
  narrower explicit list?
- **Drift-check trigger scope.** The drift job regenerates the graph only when
  graph-affecting files change. Is the trigger path set (`**/CMakeLists.txt`, the
  emit/registration cmake, the committed graph) sufficient, or can an edge change
  slip past it?
- **Cross-repo selection.** How should the graph interact with external-repo CI
  (rocm-systems / rocm-libraries) where only a subset of source is present at
  configure time?
- **Test tiers for levels 1/2/5.** The depth ladder covers levels 3-5 by
  selection radius, but true level 5 (unit-only) and level 1 (full QA / nightly)
  need a per-project `test_tier` output the test-runner job honors. Should that
  tier plumbing be part of this feature or a dedicated follow-up RFC?
- **Level naming in open source.** The "DEFCON" label is internal. Should the
  public RFC and code use a different name for the gating levels? (Pending input
  from @ScottTodd / @astyrrian1.)
