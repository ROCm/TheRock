# TheRock Build System Manual

The ROCm distribution consists of many component projects that form a DAG
consisting of build and runtime dependencies between projects. Each individual
sub-project is CMake based and dependencies between them are generally
resolved via `find_package`. This means that it is possible to build each
project in isolation.

However, working with the individual pieces is not well suited for many tasks,
most notably CI and full-stack development workflows. TheRock provides a CMake
base super-project and monorepo like organization of the source and build to
make building and testing more of a one stop shop.

## Terminology

*Super-Project*

- This refers to `TheRock` project itself as the container of sub-projects.

*Sub-Project*

- Each individual piece or standalone dependency of the ROCm system is
  referred to as a sub-project.

*Build Phases*

- Each sub-project is built in several phases: `configure`, `build`, `stage`,
  and `dist`.
- Inter-project dependencies are taken at a build phase
  granularity, allowing a degree of parallelism in the case of a more
  limited dependency.
- For each project in the tree, a specific phase can
  be built interactively by appending `+phase` to the sub-project's target
  name.

*Build Dependency*

- If a sub-project dependency is a build dependency, it is not required to be
  co-resident in a unified install tree in order to function.

*Runtime Dependency*

- If a sub-project is a runtime dependency that all/part of it must be
  co-resident in the unified install tree of the depending project in order
  to function.

*Utility Targets*

- Sub-projects may expose additional utility targets that can be accessed as
  `+utility`. Currently, this just includes `+expunge`, which removes all
  configured/built files related to the sub-project.

*Stamps*

- All Build Phases depend on a set of stamp files and produce a stamp file
  named `phase.stamp`. This produces a worst-case ordering DAG between
  phases. If a stamp file is removed (manually or via `clean`), the phase
  and all of its dependencies will re-run, regardless of whether any sources
  changed.

## Directory Layout

While the namespace of sub-projects is flat (every sub-project is a CMake target
name and therefore must be globally unique), the repository itself is organized
into a hierarchy with individual sub-projects at the leaves. Since the `ninja`
build system has good scoping of parts of the build by sub-directory (via the
implicit `all` pseudo-target), this provides some developer ergonomics, allowing
easy partial builds/cleans or deletions of parts of the build tree while
working.

Sub-project directories:

- [`base/`](/base/) : Utility sub-projects that are dependency free or minimally
  co-dependent, providing core dependencies for the rest of the system.
- [`compiler/`](/compiler/) : Compiler sub-projects, most notably the
  [AMD-LLVM](https://github.com/ROCm/llvm-project) build, `hipcc`, etc.
- [`core/`](/core/) : Core runtime sub-projects, including the low-level
  [ROCR-Runtime](https://github.com/ROCm/ROCR-Runtime) and
  higher level HIP runtimes.
- [`comm-libs/`](/comm-libs/) : Communication library sub-projects, including
  [rccl](https://github.com/ROCm/rccl)
- [`math-libs/`](/math-libs/) : Math library sub-projects, including `math-libs/BLAS` for
  [Basic Linear Algebra Subprogram](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms)
  projects like [hipBLASLt](https://github.com/ROCm/hipBLASLt).
- [`ml-libs/`](/ml-libs/) : Machine learning library sub-projects like
  [MIOpen](https://github.com/ROCm/MIOpen).
- [`profiler/`](/profiler/) : Profiler sub-projects

Note that there is nothing in the build system which ensures naming consistency,
however, we try to name leaf directories after their global sub-project
target name (i.e. `ROCR-Runtime` or `amd-llvm`) and if possible, the `project()`
name of the sub-project itself, if that is something we control. Consistency
just makes everyone's lives easier, even at the expense of sometimes not
having a naming/capitalization convention that is uniform.

## Build Directory Layout

Each sub-project, by default, uses a standard directory layout for its build:

- `build/` : The `CMAKE_BINARY_DIR` for the project, containing the
  `CMakeCache.txt`, etc. Once an initial super-project build is performed,
  developers can interact directly with the CMake build here if they prefer to
  work on a single component. The `configure` phase populates this directory
  by running the initial `cmake` configure. The `build` phase compiles the
  `all` target.
- `stage/` : Local staging install directory for the sub-project. The
  sub-projects install components (or just default install) will populate
  this directory during the `stage` build phase. Note that for sub-projects
  with runtime deps, this will be a "torn" directory in that it may contain
  shared libraries whose dependencies cannot be found because their RPATH is
  configured for the standard directory layout, which presumes that the project
  is bundled with its runtime deps.
- `dist/` : This is populated by the `dist` build phase by hard linking (or
  copying if hard-links are not possible) the cone of all runtime dep project's
  `stage/` and this project's `stage/` directory contents. In this way, each
  individual project's `dist/` directory should consist of a self-contained
  slice that is relocatable and usable (for testing, etc). Top-level
  distributions are created in this same way by having runtime deps on all
  relevant sub-projects. Keeping sub-projects isolated in this way aids in
  testing, especially to ensure that dependencies are declared and layered
  properly.
- `_init.cmake` : Generated as part of the super-project to set all necessary
  CMake settings to build the sub-project. This is injected at configure time
  via the
  [`CMAKE_PROJECT_TOP_LEVEL_INCLUDES`](https://cmake.org/cmake/help/latest/variable/CMAKE_PROJECT_TOP_LEVEL_INCLUDES.html)
  facility and serves some other ancillary functions as well:
  - Loads sub-project specific `pre` and `post` CMake file for further
    sub-project customization needed as part of the integrated whole.
  - Installs a [CMake dependency provider](https://cmake.org/cmake/help/latest/command/cmake_language.html#dependency-providers)
    which rewrites `find_package` calls for any packages provided as part
    of super-project deps appropriately.
- `_toolchain.cmake` : Generated as part of the super-project to configure the
  toolchain used to build the sub-project. This is injected at configure time
  via the
  [`CMAKE_TOOLCHAIN_FILE`](https://cmake.org/cmake/help/latest/variable/CMAKE_TOOLCHAIN_FILE.html)
  facility and allows for sub-projects to use compilers and settings which
  differ from the host toolchain, e.g. building with the AMD version of `clang`
  instead of the host `clang` on Linux or the host MSVC on Windows.
  - Note that a number of sub-projects provide their own toolchain files, one
    per platform, like
    [`rocPRIM/toolchain-linux.cmake`](https://github.com/ROCm/rocPRIM/blob/develop/toolchain-linux.cmake)
    and
    [`rocPRIM/toolchain-windows.cmake`](https://github.com/ROCm/rocPRIM/blob/develop/toolchain-windows.cmake).
    In TheRock we generate a single toolchain file dynamically that is used for
    the current platform.
- `stamp/` : Directory of `{phase}.stamp` files that are used to control
  build sequencing.

## CMake Configuration Options

TODO

## Developer Cookbook

TheRock aims to not just be a CI tool but to be a daily driver for developer
and end-users who wish to consume a source build of ROCm. This section contains
some advice that may help such users be more productive.

### Building part of the tree

Ninja's built-in directory scoping allows easy partial-builds:

Example:

```
ninja compiler/amd-llvm/all
```

Dependencies are tracked so if building a leaf project, it will build projects
it depends on. Note that there is an implicit `all` target at each directory
level, so the path can be arbitrarily fine-grained.

Similarly, cleaning the super-project and configure state can be done with:

```
ninja -t clean compiler/amd-llvm/all
```

Note that this will just clean the stamp files and byproducts that the
super-project knows about, causing a subsequent build to perform all build
phases regardless of whether any contents changed. Notably at present, this
does not clean the built files (this may change in the future), so a subsequent
build will still use CMake's own cache to avoid a full rebuild.

Sometimes a sub-project just needs to be removed entirely and rebuilt from
scratch. For this, you can do it the manual way via
`rm -Rf compiler/amd-llvm; cmake .` (note that if using a big hammer like this,
you need to regenerate the super-project build system because you just deleted
part of it). Or you can invoke the expunge sub-target:

```
ninja amd-llvm+expunge
```

## Build Topology

TheRock uses a declarative topology file (`BUILD_TOPOLOGY.toml` in the repository
root) to define the relationships between build artifacts. This file serves as
the single source of truth for:

1. **CMake Feature Generation**: During CMake configure, the topology is parsed
   to automatically generate `THEROCK_ENABLE_*` feature flags for each artifact.
   This replaces manual `therock_add_feature()` calls for artifact-mapped features.

1. **CI/CD Pipeline Sharding**: The topology defines which artifacts belong to
   which build stages, enabling the CI system to fetch only the required
   dependencies from artifact storage before each build stage.

### Topology Structure

The topology has a three-level hierarchy:

- **Build Stages**: CI/CD pipeline jobs (e.g., `compiler-runtime`, `math-libs`)
- **Artifact Groups**: Logical groupings with shared dependencies (e.g., `hip-runtime`)
- **Artifacts**: Individual build outputs (e.g., `core-hip`, `blas`, `miopen`)

### Key Files

- `BUILD_TOPOLOGY.toml` - The topology definition (see inline documentation)
- `build_tools/_therock_utils/build_topology.py` - Python parser and utilities
- `build_tools/topology_to_cmake.py` - Generates CMake includes from topology

### Naming Conventions

| Field             | Convention                 | Example           |
| ----------------- | -------------------------- | ----------------- |
| Entity names      | lowercase-with-hyphens     | `core-runtime`    |
| `feature_name`    | UPPERCASE_WITH_UNDERSCORES | `CORE_RUNTIME`    |
| `feature_group`   | UPPERCASE_WITH_UNDERSCORES | `CORE`            |
| `type` values     | lowercase                  | `target-specific` |
| `platform` values | lowercase                  | `windows`         |

### Validation

Run validation without generating output:

```bash
python build_tools/topology_to_cmake.py --validate-only
```

Print the dependency graph as JSON:

```bash
python build_tools/topology_to_cmake.py --print-graph
```

## Adding Sub-Projects

The entire sub-project facility is defined in
[`cmake/therock_subproject.cmake`](/cmake/therock_subproject.cmake)
and it may be useful to refer to that if doing anything advanced. This section
attempts to document the basics.

> [!TIP]
> These instructions assume the source directory is already populated.<br>
> You may want to first follow
> [Git Maintenance Chores: Adding a new submodule](./git_chores.md#adding-a-new-submodule).

Consider an example that is typical. This is taken from the tree but annotated
with comments, describing what is going on.

```cmake
# Create a CMake target named `ROCR-Runtime` and set it up as a sub-project.
# This will also cause phase specific convenience targets like
# `ROCR-Runtime+build` to be created. Additional calls are needed to further
# set up the sub-project, and the sequence must terminate with a call to
# `therock_cmake_subproject_activate()`
# Specific settings used here:
#   * EXTERNAL_SOURCE_DIR: Tells the system that the sources are located
#     somewhere else (in this case within the same subdirectory).
#   * CMAKE_ARGS: Additional arguments to pass to CMake. This is in addition
#     to a number of default arguments.
#   * BUILD_DEPS: Sub-projects that must be built and staged before this
#     project's configure phase can run.
#   * RUNTIME_DEPS: Sub-projects that should be considered a BUILD_DEP and
#     also are required to be in a unified distribution tree at runtime.
therock_cmake_subproject_declare(ROCR-Runtime
  EXTERNAL_SOURCE_DIR "ROCR-Runtime"
  CMAKE_ARGS
    "-DBUILD_SHARED_LIBS=ON"
  BUILD_DEPS
    amd-llvm
  RUNTIME_DEPS
    rocprofiler-register
)

# By default, the super-project treats the sources for the sub-project like a
# black box. This means that if you change the sources and rebuild, nothing
# will happen (unless if you clean or invoke the sub-project build directly).
# This directive tells the super-project that it should rebuild if C source
# files in any of the given sub-directories are modified.
therock_cmake_subproject_glob_c_sources(ROCR-Runtime
  SUBDIRS
    libhsakmt
    runtime
)

# Tells the build system that this sub-project is expected to produce two
# `find_package` packages for consumers. The path is relative to the unified
# install directory layout.
therock_cmake_subproject_provide_package(ROCR-Runtime hsakmt lib/cmake/hsakmt)
therock_cmake_subproject_provide_package(ROCR-Runtime hsa-runtime64 lib/cmake/hsa-runtime64)

# Activates the sub-project once all customization is done. This is analogous
# to `FetchContent_MakeAvailable()` for that facility.
therock_cmake_subproject_activate(ROCR-Runtime)
```
