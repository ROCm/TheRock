# Static CMake consumer graph

Generates TheRock's consumer graph by statically parsing the super-project CMake
— no `cmake` configure, no submodule checkout. It unions both branches of every
conditional, producing a conservative **may-depend** graph: it may report edges
no single configure activates, but never omits one a real configure produces.

CI (`unit_tests.yml`) runs it **report-only**, comparing the result against the
committed `test_tools/therock_consumer_graph.json`. See
[RFC0013](../../docs/rfcs/RFC0013-Consumer-Based-Test-Selection.md) for design
and rationale.

## Output

- **consumer graph** — reverse edges (`dependency -> consumer`), the schema of
  `test_tools/therock_consumer_graph.json`.
- **`subtree_map.json`** — each external source subtree (`projects/clr`,
  `shared/rocroller`, …) mapped to the graph key(s) built from it.
- **comparison + inventory** — the diff against the committed graph and the
  parsed / reached / skipped file lists.

## How it works

- Reads Git-tracked `CMakeLists.txt` / `*.cmake` (submodules are single gitlink
  paths, so the file set is stable regardless of checkout) and walks from the
  root listfile.
- For each `therock_cmake_subproject_declare()`, records the name, `BUILD_DEPS`,
  `RUNTIME_DEPS`, and `COMPILER_TOOLCHAIN`.
- Evaluates the dependency-relevant CMake subset: `set`/`unset`,
  `list(APPEND/PREPEND/REMOVE_ITEM)`, `${var}` and `;`-list expansion, `if()`
  branch union, `foreach()` (incl. `IN LISTS`/`IN ITEMS`/`RANGE`), and
  `add_subdirectory()` child scope vs. `include()` caller scope.
- Fails on an unreached declaration file, a dependency that names no known
  subproject, or a reference node/edge the static graph misses. Conservative-only
  extras are reported, not failed.

## Running

From the repo root, with test requirements installed:

```bash
pip install -r requirements-test.txt
PYTHONPATH=build_tools/consumer_graph python -m cmake_consumer_graph.cli \
  --therock-dir . --output-dir build/consumer_graph_report
```

Tests (from this directory):

```bash
python -m pytest tests/
```

Synthetic unit tests always run. The real-tree checks — the superset invariant
and subtree-map coverage — are opt-in via `CONSUMER_GRAPH_ENFORCE` so they don't
gate unrelated PRs; CI runs them in a non-blocking step:

```bash
CONSUMER_GRAPH_ENFORCE=1 python -m pytest tests/analyzer_test.py
```

## Known limitations

- Correlations between separate conditions are ignored, so combinations that can
  never co-occur may still contribute edges.
- Only the dependency-relevant CMake subset is modeled; unsupported dependency
  expressions should fail loudly and gain a test.
- Only literal `therock_cmake_subproject_declare()` / `add_subdirectory()` calls
  are seen — those reached through a wrapping function or macro are invisible, and
  the completeness check cannot catch them. The superset invariant is the backstop.
- A *defined-but-empty* dependency variable expands to nothing without being
  flagged, silently dropping its edge (`analyzer.py` `_expand_token`); a unit test
  pins this.
- Sources wired through `therock_enable_external_source()` don't resolve
  statically and get no `subtree_map` entry.
- The subtree map assumes the default `rocm-libraries` / `rocm-systems` layout and
  ignores a `THEROCK_ROCM_*_SOURCE_DIR` cache override.
