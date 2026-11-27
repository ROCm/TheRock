# Onboarding a New Component into TheRock

This guide documents everything required to add a new ROCm component
(a "sub-project") to TheRock build system: where the source lives, the CMake
wiring, the artifact/packaging registration, CI, and tests.

It complements two existing docs and pulls their pieces into one end-to-end
checklist:

- [Adding Sub-Projects](./build_system.md#adding-sub-projects) — the CMake
  mechanics.
- [Git Chores: Adding a new submodule](./git_chores.md#adding-a-new-submodule)
  — populating the source tree.

> [!NOTE]
> The worked example throughout is the **RPP** computer-vision library, which
> was onboarded in three PRs and is a faithful template for a new component:
>
> - build (1/3): [#7079](https://github.com/ROCm/TheRock/pull/7079)
> - packaging (2/3): [#7080](https://github.com/ROCm/TheRock/pull/7080)
> - inclusion/test (3/3): [#5708](https://github.com/ROCm/TheRock/pull/5708)

______________________________________________________________________

## Mental model

TheRock is a CMake super-project that assembles many independent CMake
sub-projects into a single distribution. Onboarding a component means wiring it
into four layers:

1. **Source** — the component's code, checked out via a git submodule.
1. **Build** — a CMake sub-project declaration + an artifact definition.
1. **Packaging** — how the built files flow into OS packages and Python wheels.
1. **CI & tests** — how the component builds and is tested in GitHub Actions.

The single most important fact: **almost nothing is auto-discovered.** Every
sub-project and artifact is explicitly declared. The build/test CI matrices are
_generated_ from [`BUILD_TOPOLOGY.toml`](/BUILD_TOPOLOGY.toml), so you rarely
edit workflow YAML — but you must register the component in the topology,
CMake, and the artifact/test registries by hand.

### The three source-of-truth registries

| Registry            | File                                                                                    | Role                                                                                                                             |
| ------------------- | --------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Build topology      | [`BUILD_TOPOLOGY.toml`](/BUILD_TOPOLOGY.toml)                                           | Declares the artifact, its group, deps, feature flag, platform gating. Generates the `THEROCK_ENABLE_*` flags and the CI matrix. |
| Artifact descriptor | `<family>/artifact-<name>.toml`                                                         | Declares which staged files land in each component slice (`lib`/`dev`/`dbg`/`doc`/`run`/`test`).                                 |
| CMake registration  | `<family>/CMakeLists.txt` (+ root [`CMakeLists.txt`](/CMakeLists.txt) for a new family) | `therock_cmake_subproject_declare` → … → `therock_provide_artifact`.                                                             |

______________________________________________________________________

## Step 0 — Decide where the source lives

There are two source layouts. Pick based on where the upstream code is:

### A. Inside a super-repo monorepo (the common case)

Most ROCm libraries now live inside one of two large monorepo submodules:

- [`rocm-libraries`](https://github.com/ROCm/rocm-libraries) → checked out at
  `rocm-libraries/`, sources at `rocm-libraries/projects/<name>` (BLAS, RAND,
  PRIM, FFT, MIOpen, RPP, …).
- [`rocm-systems`](https://github.com/ROCm/rocm-systems) → checked out at
  `rocm-systems/`, sources at `rocm-systems/projects/<name>` (runtime,
  profiler, RCCL, rocSHMEM, rocDecode, …).

For these, the component's source lands **in the super-repo first** (a separate
PR to `rocm-libraries`/`rocm-systems`), and TheRock references it via
`EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_LIBRARIES_SOURCE_DIR}/projects/<name>"`
(or `${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/...`). Onboarding into TheRock is then
a **submodule pointer bump** plus the wiring below — no new `.gitmodules` entry.

The in-tree family directories (`math-libs/`, `ml-libs/`, `comm-libs/`,
`cv-libs/`, …) contain **only the TheRock build glue** — a `CMakeLists.txt`,
`artifact-*.toml` descriptors, and optional `pre_hook_*.cmake` /
`post_hook_*.cmake` patch hooks. They do **not** contain component source.

### B. As a standalone git submodule (rarely)

Small, truly-independent projects get their own submodule (see the
`half`/`libhipcxx`/`rocgdb` entries in [`.gitmodules`](/.gitmodules)). Follow
[Git Chores: Adding a new submodule](./git_chores.md#adding-a-new-submodule):

```bash
git submodule add --name <Name> -b <branch> \
   https://github.com/ROCm/<Name>.git <family>/<Name>
```

Then add the submodule name to the appropriate project list in
[`build_tools/fetch_sources.py`](/build_tools/fetch_sources.py) and run
`python build_tools/fetch_sources.py` to initialize it.

______________________________________________________________________

## Step 1 — Land / point at the source

- **Super-repo component:** bump the `rocm-libraries` or `rocm-systems`
  submodule pointer to a commit that contains `projects/<name>`:
  ```bash
  git -C rocm-libraries fetch origin develop && git -C rocm-libraries checkout <sha>
  git add rocm-libraries
  ```
- **New submodule:** complete the `git submodule add` + `.gitmodules` +
  `fetch_sources.py` steps from Step 0B.

Verify the tree populates:

```bash
python build_tools/fetch_sources.py
ls <family>/<Name>   # or rocm-libraries/projects/<name>
```

______________________________________________________________________

## Step 2 — CMake sub-project wiring

Edit the family's `CMakeLists.txt` (e.g. [`cv-libs/CMakeLists.txt`](/cv-libs/CMakeLists.txt),
[`math-libs/CMakeLists.txt`](/math-libs/CMakeLists.txt)). The uniform recipe,
wrapped in an `if(THEROCK_ENABLE_<FEATURE>)` guard:

```cmake
if(THEROCK_ENABLE_RPP)
  therock_cmake_subproject_declare(rpp
    USE_DIST_AMDGPU_TARGETS
    BACKGROUND_BUILD
    EXCLUDE_FROM_ALL
    EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_LIBRARIES_SOURCE_DIR}/projects/rpp"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/rpp"
    COMPILER_TOOLCHAIN amd-hip
    BUILD_DEPS
      rocm-half
    RUNTIME_DEPS
      hip-clr
    CMAKE_ARGS
      -DROCM_PATH=
      -DBACKEND=HIP
    CMAKE_INCLUDES
      therock_explicit_finders.cmake
    IGNORE_PACKAGES
      rpp
    INTERFACE_LINK_DIRS
      "lib"
  )
  therock_cmake_subproject_glob_c_sources(rpp SUBDIRS .)
  therock_cmake_subproject_provide_package(rpp rpp lib/cmake/rpp)
  therock_cmake_subproject_activate(rpp)

  therock_test_validate_shared_lib(
    PATH "${CMAKE_CURRENT_BINARY_DIR}/rpp/dist/lib"
    LIB_NAMES librpp.so
  )

  therock_provide_artifact(rpp
    TARGET_NEUTRAL
    DESCRIPTOR artifact-rpp.toml
    COMPONENTS dbg dev doc lib test
    SUBPROJECT_DEPS rpp
  )
endif()
```

The five core calls (all defined in
[`cmake/therock_subproject.cmake`](/cmake/therock_subproject.cmake) and
[`cmake/therock_artifacts.cmake`](/cmake/therock_artifacts.cmake)):

| Call                                                               | Purpose                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `therock_cmake_subproject_declare(<name> …)`                       | Registers the sub-project + `<name>+configure/+build/+stage/+dist` targets. Key args: `EXTERNAL_SOURCE_DIR`, `COMPILER_TOOLCHAIN` (`amd-hip`/`amd-llvm`), `BUILD_DEPS` (build+provide first, non-transitive), `RUNTIME_DEPS` (build first **and** ship in the unified tree, transitive), `CMAKE_ARGS`, `CMAKE_INCLUDES`, `IGNORE_PACKAGES` (force `find_package` to fall through to the system), `INTERFACE_LINK_DIRS`, and the `USE_DIST_AMDGPU_TARGETS`/`BACKGROUND_BUILD`/`EXCLUDE_FROM_ALL` flags. |
| `therock_cmake_subproject_glob_c_sources(<name> SUBDIRS …)`        | `CONFIGURE_DEPENDS` glob so source edits re-trigger the build phase.                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `therock_cmake_subproject_provide_package(<name> <Pkg> <relpath>)` | Declares a `find_package(<Pkg>)` config the sub-project installs (e.g. `lib/cmake/rpp`), so dependents resolve to the sibling build.                                                                                                                                                                                                                                                                                                                                                                   |
| `therock_cmake_subproject_activate(<name>)`                        | Finalizes the sub-project (analogous to `FetchContent_MakeAvailable`). Must come last.                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `therock_provide_artifact(<name> …)`                               | Binds the sub-project's staged output to an artifact slice. `DESCRIPTOR` = the `artifact-<name>.toml`, `COMPONENTS` = which slices, `SUBPROJECT_DEPS` = feeding sub-projects, `TARGET_NEUTRAL` = single generic bundle vs per-arch. **Fails fast if `<name>` is not in `BUILD_TOPOLOGY.toml`** (Step 4).                                                                                                                                                                                               |

**Source patching (optional):** drop `pre_hook_<Target>.cmake` or
`post_hook_<Target>.cmake` in the family dir; they are auto-detected by filename
(see `math-libs/pre_hook_rocRAND.cmake`, `comm-libs/pre_hook_rccl.cmake`).

**New family only:** if you are creating a brand-new top-level directory
(not adding to an existing family), also add it to root
[`CMakeLists.txt`](/CMakeLists.txt):

- an `add_subdirectory(<dir>)` at the correct dependency-DAG position (the
  ordered block near lines 498–522), and
- a `THEROCK_ENABLE_<GROUP>` option in the feature-group block (lines 221–241).

______________________________________________________________________

## Step 3 — Artifact descriptor (`artifact-<name>.toml`)

Create `<family>/artifact-<name>.toml`. It maps each component **slice** to file
globs relative to each sub-project's `stage/` directory. Empty tables mean
"take the default fileset for that slice"; `include`/`exclude` refine it.

Example ([`cv-libs/artifact-rpp.toml`](/cv-libs/artifact-rpp.toml)):

```toml
# rpp
[components.lib."cv-libs/rpp/stage"]
[components.dbg."cv-libs/rpp/stage"]
[components.dev."cv-libs/rpp/stage"]
[components.doc."cv-libs/rpp/stage"]
[components.test."cv-libs/rpp/stage"]
include = [
  "share/rpp/**",
]
```

Slices are `lib` (runtime libs), `dev` (headers/cmake), `dbg` (debug info),
`doc`, `run` (runtime tools), `test`. The descriptor is consumed by
`build_tools/fileset_tool.py artifact`, invoked from `therock_provide_artifact`.

______________________________________________________________________

## Step 4 — Register in `BUILD_TOPOLOGY.toml`

[`BUILD_TOPOLOGY.toml`](/BUILD_TOPOLOGY.toml) is the source of truth for which
artifacts exist, their grouping, deps, and feature flags. Add an
`[artifacts.<name>]` block:

```toml
[artifacts.rpp]
artifact_group = "cv-libs"
type = "target-neutral"           # or "target-specific" for per-arch code
artifact_deps = ["core-runtime", "core-hip", "base", "sysdeps"]
feature_group = "CV_LIBS"
# disable_platforms = ["windows"]  # hard-off per platform, if applicable
```

At configure time, [`build_tools/topology_to_cmake.py`](/build_tools/topology_to_cmake.py)
reads this and generates a `therock_add_feature()` call, which creates the
`THEROCK_ENABLE_<FEATURE>` cache variable your `if()` guard tests. The feature
name defaults to the uppercased artifact name (`rpp` → `THEROCK_ENABLE_RPP`);
the group default comes from `feature_group` (`CV_LIBS` → `THEROCK_ENABLE_CV_LIBS`).

**Conditional availability** (see the table in
[`build_system.md`](./build_system.md#conditional-availability)):

- `disable_platforms = ["windows"]` — hard disable, cannot be overridden.
- `disable_processors = ["ppc64le"]` — soft default-off, overridable.

**New family only:** also add `[build_stages.<stage>]`, `[artifact_groups.<group>]`
(with `artifact_group_deps` and `source_sets`), and — if the source is a new
submodule — a `[source_sets.*]` entry listing the submodules the group needs.
CI sharding and partial submodule checkouts are driven from these.

Validate the topology before building:

```bash
python build_tools/topology_to_cmake.py --validate-only
python build_tools/topology_to_cmake.py --print-graph   # dependency graph as JSON
```

______________________________________________________________________

## Step 5 — Packaging

Two packaging targets consume artifacts. A normal library flows into Python
wheels **automatically** by component slice (no wheel-template edit), but OS
packaging and the test-side fetch flags are explicit.

- **OS packages (Linux):** add the component's files to
  [`build_tools/packaging/linux/package.json`](/build_tools/packaging/linux/package.json)
  (this was the bulk of RPP packaging PR [#7080](https://github.com/ROCm/TheRock/pull/7080)).
- **Artifact fetch flag:** add a `--<name>` flag to the `artifacts_group`
  argparse block in
  [`build_tools/install_rocm_from_artifacts.py`](/build_tools/install_rocm_from_artifacts.py).
  This is what the test matrix's `fetch_artifact_args` references.
- **Python packaging touch-points seen in the RPP PR:**
  [`build_tools/build_python_packages.py`](/build_tools/build_python_packages.py)
  and `build_tools/packaging/templates/rocm/src/rocm_sdk/_dist_info.py`.
- **Artifact→subproject map:** `build_tools/artifact_subprojects.json` (a small
  entry was added for RPP).

If the component's build or tests need a new host Python dependency, prefer the
targeted knobs — per-artifact `python_requires` in `BUILD_TOPOLOGY.toml` or a
test's `additional_requirements_files` — over the global
[`requirements.txt`](/requirements.txt).

______________________________________________________________________

## Step 6 — Tests

- **Test matrix (primary registry):** add an entry to the `test_matrix` dict in
  [`build_tools/github_actions/fetch_test_configurations.py`](/build_tools/github_actions/fetch_test_configurations.py).
  Set `job_name`, `fetch_artifact_args` (e.g. `--rpp --tests`), `test_script`,
  `platform`, `total_shards_dict`, and optional `include_family`/`exclude_family`/
  `multi_gpu`/`container_image`/`additional_requirements_files`. This entry feeds
  the reusable `test_component.yml` job.
- **Test script:** either reuse the generic
  `build_tools/github_actions/test_executable_scripts/test_runner.py` (and add a
  `job_name → ctest dir` entry to its `COMPONENT_DIR_MAPPING`), or add a bespoke
  `test_executable_scripts/test_<name>.py` (RPP added `test_rpp.py`).
- **Test-selection policy (optional):** add a `[component.<name>]` block to
  [`test_tools/test_policies.toml`](/test_tools/test_policies.toml) only for
  couplings the auto-generated consumer graph can't express (see
  [RFC0013](/docs/rfcs/RFC0013-Consumer-Based-Test-Selection.md)).
- **Artifact-structure test:** update
  [`tests/test_artifact_structure.py`](/tests/test_artifact_structure.py) so the
  new artifact's expected layout is asserted.

______________________________________________________________________

## Step 7 — CI (mostly automatated) and assisted via docs

- **Build/test matrices:** generated from `BUILD_TOPOLOGY.toml` — **no workflow
  YAML edit** for a plain new component.
- **New CI workflow file only:** if you add a new `.github/workflows/*.yml` that
  `multi_arch_ci.yml` calls, add its filename to the manually-maintained
  allowlist `_GITHUB_WORKFLOWS_CI_FILENAMES` in
  [`build_tools/github_actions/configure_ci_path_filters.py`](/build_tools/github_actions/configure_ci_path_filters.py)
  (a unit test enforces this).
- **New GPU family (not a new library):** edit the matrices in
  [`build_tools/github_actions/amdgpu_family_matrix.py`](/build_tools/github_actions/amdgpu_family_matrix.py).
  Not needed for a plain library.
- **Docs & housekeeping:** the RPP build PR also touched `CLAUDE.md` (component
  listing), `README.md`, `<family>/README.md`, `build_tools/analyze_build_times.py`,
  `build_tools/github_actions/build_configure.py`,
  `build_tools/detail/linux_portable_build_in_container.sh`, and several docs
  under `docs/development/`. Grep for an analogous existing component and mirror
  its footprint.

______________________________________________________________________

## Past Working example: the RPP three-PR footprint

Exact files changed, as a concrete template:

### 1/3 — build ([#7079](https://github.com/ROCm/TheRock/pull/7079))

```
.github/workflows/multi_arch_build_portable_linux.yml
BUILD_TOPOLOGY.toml                       # [artifacts.rpp], group, feature
CMakeLists.txt                            # add_subdirectory(cv-libs) + option (new family)
cv-libs/CMakeLists.txt                    # declare → activate → provide_artifact
cv-libs/artifact-rpp.toml                 # artifact descriptor
cv-libs/README.md
CLAUDE.md, README.md
build_tools/artifact_subprojects.json
build_tools/analyze_build_times.py
build_tools/github_actions/build_configure.py
build_tools/detail/linux_portable_build_in_container.sh
docs/development/{artifacts,build_system,ci_behavior_manipulation,ci_overview,windows_support}.md
tests/test_artifact_structure.py
```

### 2/3 — packaging ([#7080](https://github.com/ROCm/TheRock/pull/7080))

```
build_tools/packaging/linux/package.json          # OS packaging (bulk)
build_tools/install_rocm_from_artifacts.py         # --rpp fetch flag
build_tools/build_python_packages.py
build_tools/packaging/templates/rocm/src/rocm_sdk/_dist_info.py
build_tools/packaging/tests/install_rocm_from_artifacts_test.py
docs/development/installing_artifacts.md
tests/test_artifact_structure.py
```

### 3/3 — inclusion/test ([#5708](https://github.com/ROCm/TheRock/pull/5708))

```
build_tools/github_actions/fetch_test_configurations.py   # test matrix entry
build_tools/github_actions/test_executable_scripts/test_rpp.py
build_tools/install_rocm_from_artifacts.py
```

______________________________________________________________________

## Onboarding checklist

### New component in an existing family

- [ ] Bump the `rocm-libraries`/`rocm-systems` submodule to a commit containing `projects/<name>`.
- [ ] `BUILD_TOPOLOGY.toml` — add `[artifacts.<name>]` (group, deps, type, feature_group).
- [ ] `<family>/artifact-<name>.toml` — new artifact descriptor.
- [ ] `<family>/CMakeLists.txt` — `declare → glob_c_sources → provide_package → activate → provide_artifact`, guarded by `if(THEROCK_ENABLE_<FEATURE>)`.
- [ ] `build_tools/install_rocm_from_artifacts.py` — add `--<name>` fetch flag (if fetched by tests).
- [ ] `build_tools/github_actions/fetch_test_configurations.py` — add `test_matrix` entry (if it has tests).
- [ ] Test script: `test_runner.py` `COMPONENT_DIR_MAPPING` entry, or a new `test_<name>.py`.
- [ ] `build_tools/packaging/linux/package.json` — OS packaging entries.
- [ ] `tests/test_artifact_structure.py` — expected-layout assertions.
- [ ] `test_tools/test_policies.toml` — optional test-coupling overrides.

### Additionally, for a new family or a new submodule

- [ ] Root `CMakeLists.txt` — `add_subdirectory(<family>)` + `THEROCK_ENABLE_<GROUP>` option.
- [ ] `BUILD_TOPOLOGY.toml` — `[build_stages.<stage>]`, `[artifact_groups.<group>]`, `[source_sets.*]`.
- [ ] `.gitmodules` + `build_tools/fetch_sources.py` — register a standalone submodule.
- [ ] `configure_ci_path_filters.py` `_GITHUB_WORKFLOWS_CI_FILENAMES` — only if you add a new CI workflow file.

### Validation

```bash
python build_tools/topology_to_cmake.py --validate-only
python build_tools/fetch_sources.py
# configure + build just this component:
cmake -B build -GNinja . -DTHEROCK_ENABLE_<FEATURE>=ON
ninja -C build <family>/<name>/all
python -m pytest tests/test_artifact_structure.py
```

______________________________________________________________________

## Reference

- CMake sub-project machinery: [`cmake/therock_subproject.cmake`](/cmake/therock_subproject.cmake)
- Artifact machinery: [`cmake/therock_artifacts.cmake`](/cmake/therock_artifacts.cmake)
- Topology parser/generator: [`build_tools/topology_to_cmake.py`](/build_tools/topology_to_cmake.py), [`build_tools/_therock_utils/build_topology.py`](/build_tools/_therock_utils/build_topology.py)
- Build system manual: [`docs/development/build_system.md`](./build_system.md)
- Artifacts: [`docs/development/artifacts.md`](./artifacts.md)
- Adding tests: [`docs/development/adding_tests.md`](./adding_tests.md)
- Submodule chores: [`docs/development/git_chores.md`](./git_chores.md)
- CI overview: [`docs/development/ci_overview.md`](./ci_overview.md)
