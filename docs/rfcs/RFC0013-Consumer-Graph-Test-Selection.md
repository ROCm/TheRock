---
author: Abhilash Reddy Endurthi (endurthiabhilash)
created: 2026-07-28
modified: 2026-07-28
status: draft
---

# Derive Test-Subproject Selection From the Consumer Graph

This RFC proposes replacing the partially-populated, hand-maintained
`TEST_SUBPROJECTS` lists in component `CMakeLists.txt` files with a **consumer
graph** emitted automatically at CMake configure time, combined with a small set
of explicit overrides declared in `BUILD_TOPOLOGY.toml`. The goal is to make
change-driven test selection ("when subproject X changes, which test suites must
run?") derive from the dependency edges the build already declares, rather than
from a parallel list that must be manually kept in sync.

## Background: DEFCON gating levels

This work exists to implement **DEFCON 4** per-PR gating across TheRock's
components. DEFCON is the ROCm enablement-gating scheme defined in
[DEFCON Enablement: Gating Levels][defcon] (MLSE space): a per-component tier
that fixes *how much* every PR must build and test, sized to the component's
downstream blast radius and regression history.

| Level        | What every PR must build & test                                     | Intended for                                             |
| ------------ | ------------------------------------------------------------------- | -------------------------------------------------------- |
| **DEFCON 1** | Full QA integration + nightly suites                                | Components actively destabilizing releases               |
| **DEFCON 2** | Frameworks built and their tests run                                | High downstream blast radius (e.g. framework-facing libs)|
| **DEFCON 3** | All rocm-libraries projects/tests + rest of rocm-systems (rccl etc.)| Broad-dependency components with regression history      |
| **DEFCON 4** | Local + intimately related project builds/tests only                | Well-tested components with contained surface            |
| **DEFCON 5** | Fast unit tests only                                                | Proven track record + strong local test coverage         |

The motivating failure mode is exactly the one DEFCON names: a component change
passes its own local tests, then surfaces a regression days later during a
downstream module bump. Gating pulls that discovery left: a PR must exercise the
components it can break *before* it merges, not after.

Lower numbers are stricter (broader fan-out); higher numbers are cheaper. The
rollout is staged: **Phase 1 starts every component at DEFCON 4**, and gating is
tightened (4 → 3 → 2 → 1) only for components that accumulate regression history.
The long-term goal moves the other way: promote proven components *up* to
DEFCON 5 (fast unit tests only) so the mainline moves fast where it safely can. A
per-PR test-time SLA of **10–15 minutes** bounds the cost.

DEFCON 4, "local **and intimately related** project builds/tests", is precisely
the selection this RFC automates: for a changed subproject, run it plus the
projects directly downstream of it. Doing that for *every* component requires a
reliable, low-maintenance answer to "what is directly downstream of X?" across the
whole tree. The hand-maintained `TEST_SUBPROJECTS` lists were that answer in
principle but not in practice (below); the consumer graph makes the DEFCON 4
baseline derivable rather than curated, and the override keys become the knobs for
moving a component off that baseline.

[defcon]: https://amd.atlassian.net/wiki/spaces/MLSE/pages/1795331309/DEFCON+Enablement+Gating+Levels

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
2. **Commit a generated consumer graph alongside a static `overrides.json`.** A
   follow-up explored emitting the graph, checking it into the repo, and layering
   a committed `test_subprojects_overrides.json`. This was also never merged.

The dependency information needed to answer "what runs when X changes?" already
exists: every subproject declares its build and runtime dependencies. This RFC
makes test selection a *derived* property of those declarations, so no exhaustive
hand-maintained list is required.

## Proposal

### Overview

1. **Register consumers at declare time.** Every
   `therock_cmake_subproject_declare()` appends the subproject to a global
   registry and records a reverse edge (`consumer`) for each of its
   `BUILD_DEPS` / `RUNTIME_DEPS`.
2. **Emit a consumer graph.** At the end of the top-level configure,
   `therock_emit_consumer_graph()` serializes the registry to
   `build/therock_consumer_graph.json`, a configure-time side effect analogous to
   `compile_commands.json`. It is **dynamic and never committed**.
3. **Derive test subprojects in Python.** `determine_rocm_test_dependencies.py`
   loads the graph, computes each subproject's build stage from the committed
   `artifact-*.toml` descriptors + `BUILD_TOPOLOGY.toml`, and for each changed
   subproject selects its **same-stage** direct consumers.
4. **Apply explicit overrides from `BUILD_TOPOLOGY.toml`.** Couplings the graph
   cannot express (test-only projects with no link edge, cross-stage couplings,
   and stage-less foundational deps) are declared with `test_include` /
   `test_exclude` / `test_fanout_all` keys on the relevant artifact.

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

The subproject→build-stage mapping needed for the same-stage cut is intentionally
**not** duplicated in the graph. It is derived separately by the Python tooling
from the committed artifact descriptors, so the graph stays minimal and the stage
logic has a single source of truth.

### The "same-stage cut"

Selecting *all* transitive consumers of a foundational dependency (e.g.
`hip-clr`, `ROCR-Runtime`, the compiler) would pull in nearly the entire test
tree: every math lib, ML lib, comm lib, etc. To keep selection bounded, the tool
only selects consumers **in the same BUILD_TOPOLOGY build stage** as the changed
subproject. Cross-stage consumers are treated as "universal" and cut.

Where a specific cross-stage or test-only coupling genuinely must run, it is
declared explicitly (see Overrides). This makes the expensive, broad selections
opt-in and auditable rather than accidental.

The same-stage cut is what keeps the default at **DEFCON 4** ("local + intimately
related") rather than sliding into DEFCON 3 ("all rocm-libraries / rocm-systems").
Selecting every transitive consumer of a foundational dep is effectively DEFCON 3
behavior on every touch of that dep; the cut confines the automatic selection to
the directly-related, same-stage neighborhood and keeps runs inside the 10–15 min
SLA. Deliberately widening a component toward DEFCON 3/2 is then an explicit,
reviewable act (`test_fanout_all` / `test_include`) rather than an accident of the
graph's shape.

### Overrides in BUILD_TOPOLOGY.toml

Three optional keys layer explicit selection on top of the derived graph, keeping
all test-selection metadata alongside the artifact/stage definitions it depends on
(rather than in a separate, standalone file):

- `test_include` — extra subprojects to test when this artifact changes, beyond
  same-stage consumers. Covers test-only projects with no link-time edge
  (e.g. `hip-tests`, `rocgdb-cpu`) and same-stage reverse edges not present as
  forward consumers.
- `test_exclude` — consumers to prune even though they appear in the graph.
  Applied **last** (after include and after fanout) so it can drop a specific
  consumer of a fanned-out foundational dep.
- `test_fanout_all` — when true, a change to this artifact selects **all** of its
  graph consumers, bypassing the same-stage cut. This is the deliberate opt-in for
  stage-less foundational deps (e.g. artifact `base`: `rocm-core`, `rocm-cmake`,
  `rocm-half`, `rocprofiler-register`) whose consumers span every stage and would
  otherwise select nothing.

When one artifact packages several subprojects with differing couplings, a
per-subproject sub-table is used instead of the flat keys:

```toml
[artifacts.blas.test_overrides.hipblaslt]
test_include = ["hipsparselt"]
```

### CI integration

The change-detection job in `test_artifacts.yml` gains three steps before it
determines test dependencies:

1. `pip install -r requirements.txt`
2. `fetch_sources.py` — all subproject source trees must be present so every
   `therock_cmake_subproject_declare()` runs and the emitted graph is complete.
3. A configure-only `cmake -B build -DTHEROCK_ENABLE_ALL=ON ...` that emits
   `build/therock_consumer_graph.json` as a side effect.

`determine_rocm_test_dependencies.py` then reads the freshly emitted graph. The
graph is never committed, so it cannot drift from the build definitions.

## Alternatives considered

### A: Fully populate `TEST_SUBPROJECTS` by hand (attempted)
Complete the existing mechanism by adding the missing component names to every
`therock_cmake_subproject_declare()`. A PR doing this was opened but never merged:
the list was large, hard to review for correctness, and would require re-editing on
every dependency change: the same drift/duplication problem, just fully expanded.

### B: Commit a generated consumer graph plus a static `overrides.json` (attempted)
Emit the graph, check the committed copy into the repo, and layer a committed
`test_subprojects_overrides.json` for the couplings the graph cannot express. This
was explored but never merged. Committing the graph reintroduces a file that can go
stale and adds a regeneration ritual to every dependency change; the separate
overrides file also split test-selection metadata away from the artifact/stage
definitions it depends on. This RFC keeps the graph dynamic (regenerated on demand
in CI, never committed) and folds the overrides into `BUILD_TOPOLOGY.toml` instead.
*(The dynamic-graph choice is reversible: if configure cost becomes a concern, a
committed-with-drift-lint model can be revisited.)*

### C: Encode stages in the graph itself
Have CMake write each subproject's build stage into the graph JSON. Rejected: the
stage is already derivable from committed artifact descriptors, and duplicating it
creates a second source of truth that can disagree with `BUILD_TOPOLOGY.toml`.

### D: Select all transitive consumers (no same-stage cut)
Simplest selection rule, but explodes into whole-tree test runs for any change to
a foundational dependency. Rejected; the same-stage cut plus explicit
`test_fanout_all` gives bounded, auditable behavior.

## Impact and migration

- **Removed:** the `TEST_SUBPROJECTS` parameter from
  `therock_cmake_subproject_declare()` and its (partial) uses across component
  `CMakeLists.txt` files.
- **Added:** `cmake/therock_emit_consumer_graph.cmake`; consumer registration in
  `cmake/therock_subproject.cmake`; the emit call in the top-level
  `CMakeLists.txt`; `test_include` / `test_exclude` / `test_fanout_all` (and the
  `test_overrides` sub-tables) in `BUILD_TOPOLOGY.toml`; graph-driven logic in
  `determine_rocm_test_dependencies.py`.
- **CI:** `test_artifacts.yml` performs a configure-only step to emit the graph.
- **Behavioral parity:** existing couplings (e.g. rocGDB → rocgdb-cpu/gpu,
  hipCUB/rocThrust → rocPRIM, amdsmi → hip-tests/rocrtst) are preserved via the
  migrated overrides, so selection results should match the prior behavior for
  known changes while additionally covering newly-added consumers automatically.

## Testing

`test_tools/tests/determine_rocm_test_dependencies_test.py` is expanded to cover
the same-stage cut, each override key, per-subproject sub-tables, fanout, and the
stage-derivation path. The `TEST_OVERRIDE_CHANGED_PROJECTS` env override in the
external-repo CI configuration allows forcing a specific `changed_projects` value
to exercise the selection end-to-end in a real CI run without an actual source
change.

## Open questions

- **Configure cost in CI.** The change-detection job now runs a full
  `THEROCK_ENABLE_ALL=ON` configure. Is configure-only latency acceptable at the
  current subproject count, or should the graph be cached/keyed on a manifest
  hash?
- **Multi-subproject artifacts.** The `test_overrides.<subproject>` sub-table
  handles artifacts that package several subprojects; are there remaining cases
  where the flat keys over-apply across siblings?
- **Cross-repo selection.** How should the graph interact with external-repo CI
  (rocm-systems / rocm-libraries) where only a subset of source is present at
  configure time?
