# Static CMake consumer graph prototype

This prototype tests whether TheRock's consumer graph can be generated from its
super-project CMake files without initializing submodules or running CMake.

It intentionally computes a conservative **may-depend** graph: both sides of
conditional CMake code are analyzed and their possible dependency values are
unioned. The generated graph may contain relationships that are not active in
one particular configure, but it must not omit any relationship from a real
configure.

## Current results

Results were captured on 2026-08-31 from TheRock commit
`3c4ec014374d4d32243334ef229093fe7b38c5b2`.

| Measurement                                      |    Result |
| ------------------------------------------------ | --------: |
| Super-project-owned CMake files tracked by Git   |       140 |
| Files successfully parsed by `cmake-parser`      |       140 |
| Files reached from the root listfile             |        78 |
| Tracked files containing subproject declarations |        56 |
| Declaration files missed by traversal            |         0 |
| Declaration calls reached                        |       122 |
| Unique subproject names                          |       121 |
| Reference nodes reproduced                       | 117 / 117 |
| Reference edges reproduced                       | 421 / 421 |
| Conservative-only nodes                          |         4 |
| Conservative-only edges                          |        11 |
| Unresolved dependency names                      |         0 |

The additional nodes are conditionally declared projects that were absent from
the Linux `gfx94X-dcgpu` configure used to produce the committed reference:

- `hipinfo`
- `hrx`
- `therock-openmpi`
- `therock-wsl-rocdxg`

The full generated graph, analysis inventory, and direct comparison are under
[`results/`](results/).

## Step 1: characterize `cmake-parser`

`cmake-parser==0.9.2` provides useful structured nodes for `if()` blocks,
commands, source lines, quoted values, and variable tokens. It parsed all 140
tracked TheRock CMake files successfully.

One important sharp edge was found and is covered by a regression test:
`parse_tree(..., skip_comments=True)` removes standalone comment nodes but
retains comments inside command parentheses as `Token(kind="COMMENT")`. Those
tokens must be explicitly filtered or comments can become false dependency or
toolchain values.

The parser also correctly rejects structurally malformed input with a source
location instead of returning a partial tree.

## Step 2: discover TheRock-owned CMake files

The prototype asks Git for tracked files and keeps only `CMakeLists.txt` and
`*.cmake`. A gitlink is tracked as one path, not as the files inside its checked
out submodule, so this boundary is stable whether submodules are initialized or
empty.

Starting at TheRock's root `CMakeLists.txt`, the analyzer follows tracked
`add_subdirectory()` calls through both sides of conditionals. It also follows
resolvable tracked `include()` calls. Paths that resolve to built-in CMake
modules, generated files, external source trees, or submodules are recorded as
skipped diagnostics.

As a completeness check, the analyzer independently parses the entire tracked
inventory and reports any declaration-bearing file that the traversal missed.
The current result is zero missed files.

## Step 3: extract declarations and dependencies

For each `therock_cmake_subproject_declare()` call, the prototype extracts:

- the literal subproject name;
- `BUILD_DEPS`;
- `RUNTIME_DEPS`; and
- `COMPILER_TOOLCHAIN`.

The abstract environment currently implements the constructs used by those
arguments in TheRock:

- `set()` and `unset()`;
- `list(APPEND)`, `list(PREPEND)`, and `list(REMOVE_ITEM)`;
- `${variable}` expansion and semicolon-separated lists;
- directory-scope environment inheritance;
- conservative branch union for `if()`; and
- one conservative pass through `foreach()` bodies.

An unresolved variable in a dependency-bearing argument is a fatal error. It is
never silently dropped.

The `amd-hip -> hip-clr` and `amd-llvm -> amd-llvm` toolchain mappings are
temporarily duplicated in Python. A replacement design should make that mapping
declarative in CMake so both implementations consume one definition.

## Step 4: generate and compare graphs

The generated JSON uses the same reverse-edge schema as
`test_tools/therock_consumer_graph.json`. Comparison is performed on normalized
node names and `dependency -> consumer` edges.

The command fails if:

- a tracked declaration file was not reached;
- a dependency does not resolve to a discovered subproject; or
- the static graph omits a node or edge from the reference graph.

Conservative-only nodes and edges are reported but do not fail the command.

## Running the prototype

Create an isolated environment outside the repository and install the pinned
requirements. For example, from this directory in PowerShell:

```powershell
py -3.12 -m venv D:\scratch\cmake-consumer-graph-venv
D:\scratch\cmake-consumer-graph-venv\Scripts\python.exe -m pip install `
  -r requirements.txt
```

Run the unit tests with cache and temporary files outside the checkout:

```powershell
D:\scratch\cmake-consumer-graph-venv\Scripts\python.exe -m pytest `
  --override-ini=cache_dir=D:/scratch/cmake-consumer-graph-pytest-cache `
  --basetemp=D:/scratch/cmake-consumer-graph-pytest-tmp `
  -q
```

Generate and compare the graph:

```powershell
D:\scratch\cmake-consumer-graph-venv\Scripts\python.exe `
  -m cmake_consumer_graph.cli `
  --therock-dir ..\..\..\TheRock `
  --output-dir results
```

The CLI does not depend on the current working directory except that Python
must be able to import the prototype package, such as when run from this folder.

The synthetic (`tmp_path`) unit tests always run. The real-tree enforcement
tests — the committed-graph superset invariant and the subtree-map coverage
cross-check — are opt-in: they only run when `CONSUMER_GRAPH_ENFORCE=1` is set,
so an unmodeled CMake construct in an unrelated PR does not block the default
unit-test job. CI runs them in a dedicated `continue-on-error` step; run them
locally with `CONSUMER_GRAPH_ENFORCE=1 python -m pytest tests/analyzer_test.py`.

## Known limitations and next experiments

- The analyzer deliberately ignores correlations between conditions, so it can
  add edges that no single configuration activates. This is the selected safety
  tradeoff, but graph fanout should be measured before integration.
- Include/module resolution is best-effort. It is currently sufficient because
  all declaration-bearing files are reached through `add_subdirectory()`.
- The evaluator supports the dependency-related CMake subset present today, not
  arbitrary CMake execution. New unsupported dependency expressions should fail
  loudly and gain focused tests.
- Embedded variable expansion is modeled as a set product and escaped-semicolon
  behavior has not yet been characterized.
- The committed reference represents one Linux configuration. A later
  differential test should compare the static graph against multiple Linux and
  Windows CMake configurations, requiring each configured graph to be a subset.
- Only literal `therock_cmake_subproject_declare()` / `add_subdirectory()` calls are
  modeled. A declaration or subdirectory reached only through a wrapping function or
  macro is not seen, and the completeness check (which flags tracked files with a
  *literal* declaration missed by traversal) cannot catch it — a potential
  under-selection source the differential-configure superset test is meant to guard.
- A dependency variable that is *defined but empty* expands to nothing without being
  flagged unresolved, so its edge is silently dropped (`analyzer.py` `_expand_token`).
  A unit test characterizes this; the differential test is the real guard.

## Alternatives considered

- A metadata-only CMake mode would retain CMake as the interpreter, but requires
  separating declaration from activation throughout the super-project.
- Moving dependency metadata into `BUILD_TOPOLOGY.toml` would avoid parsing, but
  is a larger source-of-truth migration.
- Scanning every filesystem CMake file was rejected because initialized
  submodules would change the input set. Git's tracked-file boundary is both
  cheaper and deterministic.
