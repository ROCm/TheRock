# Build Flags

Build flags are typed, system-wide controls that affect how TheRock subprojects
are configured and compiled. Each flag creates a `THEROCK_FLAG_{NAME}` CMake
cache variable, is persisted in distribution metadata, and is made available to
participating ROCm projects through a versioned provider-state file.

Flags support `BOOL` and `INTEGER` values. A ROCm project can consume the same
flag in two modes:

- In a TheRock build, TheRock's value is authoritative.
- In a standalone build, the project's own cache variable and default are used.

## Flags vs Features

| Concept                                 | Purpose                                             | Naming                  |
| --------------------------------------- | --------------------------------------------------- | ----------------------- |
| **Features** (`therock_features.cmake`) | Control which subprojects are included in the build | `THEROCK_ENABLE_{NAME}` |
| **Flags** (`FLAGS.cmake`)               | Control *how* included subprojects are configured   | `THEROCK_FLAG_{NAME}`   |

Features are about "what to build". Flags are about "how to build it".

## Architecture

```
FLAGS.cmake              Central declarations (project root)
  └── therock_declare_flag()   →  THEROCK_FLAG_{NAME} cache var
  └── BRANCH_FLAGS.cmake       →  Legacy per-branch default overrides
  └── BRANCH_CONFIG.json       →  Per-branch defaults and optional sources
  └── therock_finalize_flags() →  Propagation data + typed state and JSON
  └── therock_report_flags()   →  Status output

cmake/therock_flag_utils.cmake   Processing functions
cmake/ROCMBuildFlags.cmake       Standalone/integrated consumer API
build_tools/topology_to_cmake.py Generated branch config CMake helpers
cmake/therock_subproject.cmake   Provider-state injection via project_init.cmake
base/aux-overlay/                Canonical consuming project example
```

TheRock's `ROCMBuildFlags.cmake` is the canonical source. Consumer
super-repositories must carry a verbatim copy. The module intentionally has no
external dependencies and remains compatible with CMake 3.7.

### Provider State

`therock_finalize_flags()` generates
`rocm_build_flags_state.cmake` in the build tree. The state is data expressed
using basic CMake `set()` calls so old standalone project CMake versions do not
need a JSON parser:

```cmake
set(ROCM_BUILD_FLAGS_PROTOCOL_VERSION 1)
set(ROCM_BUILD_FLAGS_PROVIDER "TheRock")
set(ROCM_BUILD_FLAGS_NAMES EXAMPLE_BOOL EXAMPLE_INTEGER)
set(ROCM_BUILD_FLAG_EXAMPLE_BOOL_TYPE "BOOL")
set(ROCM_BUILD_FLAG_EXAMPLE_BOOL_VALUE "1")
set(ROCM_BUILD_FLAG_EXAMPLE_INTEGER_TYPE "INTEGER")
set(ROCM_BUILD_FLAG_EXAMPLE_INTEGER_VALUE "-17")
set(ROCM_BUILD_FLAGS_STATE_COMPLETE 1)
```

The completion marker must be last. Consumers fail during configure on a
missing or incomplete file, an unsupported protocol, duplicate or missing
flags, malformed values, and type mismatches. TheRock injects only the absolute
`ROCM_BUILD_FLAGS_STATE_FILE` path into each subproject's generated
`project_init.cmake`; it does not inject the helper or create an exported
target. The state and helper participate in subproject configure fingerprints,
so changing a central value reconfigures consumers.

### Legacy Propagation

Flag effects are injected into subprojects via the generated
`project_init.cmake` files (the same mechanism used for
`THEROCK_DEFAULT_CMAKE_VARS`):

- **GLOBAL_PROPAGATE_FLAG**: Mirrors `THEROCK_FLAG_{NAME}` to **all**
  subprojects, regardless of whether the flag is enabled or disabled.
- **GLOBAL_CMAKE_VARS**: `VAR=VALUE` pairs set in the super-project and
  propagated to **all** subprojects when the flag is enabled.
- **GLOBAL_CPP_DEFINES**: Preprocessor defines added to **all** subprojects
  when the flag is enabled via `add_compile_definitions()` in project_init.cmake.
- **CMAKE_VARS**: `VAR=VALUE` pairs injected only into the listed
  **SUB_PROJECTS** when the flag is enabled.
- **CPP_DEFINES**: Preprocessor defines added only to the listed
  **SUB_PROJECTS** when the flag is enabled via `add_compile_definitions()`.

The enabled-only variable and compiler-definition mechanisms are retained for
existing BOOL users. They are not supported for INTEGER flags. New C or C++
consumers should use `ROCMBuildFlags.cmake` and a private generated header
instead of relying on directory-wide definitions.

Structural concerns (conditional subproject inclusion, runtime dependency
wiring) remain as explicit conditionals in the consuming CMakeLists.txt files.
Flags do not auto-include subprojects.

## Declaring a Flag

All flags are declared in `FLAGS.cmake` at the project root:

```cmake
therock_declare_flag(
  NAME EXAMPLE_INTEGER
  TYPE INTEGER
  DEFAULT_VALUE -17
  VALID_VALUES -17 0 5
  DESCRIPTION "Example integer build control"
)
```

### Parameters

| Parameter               | Required | Description                                                                      |
| ----------------------- | -------- | -------------------------------------------------------------------------------- |
| `NAME`                  | Yes      | Unique identifier. Creates `THEROCK_FLAG_{NAME}` cache variable.                 |
| `TYPE`                  | No       | `BOOL` or `INTEGER`; defaults to `BOOL`.                                         |
| `DEFAULT_VALUE`         | Yes      | Value matching `TYPE`.                                                           |
| `DESCRIPTION`           | Yes      | Short description shown in CMake cache UI.                                       |
| `VALID_VALUES`          | No       | Allowed values for an `INTEGER` flag.                                            |
| `ISSUE`                 | No       | Tracking issue URL.                                                              |
| `GLOBAL_PROPAGATE_FLAG` | No       | Mirror `THEROCK_FLAG_{NAME}` to all subprojects whether enabled or disabled.     |
| `GLOBAL_CMAKE_VARS`     | No       | `VAR=VALUE` pairs for all subprojects when enabled.                              |
| `GLOBAL_CPP_DEFINES`    | No       | Preprocessor defines for all subprojects when enabled.                           |
| `CMAKE_VARS`            | No       | `VAR=VALUE` pairs for listed `SUB_PROJECTS` only when enabled.                   |
| `CPP_DEFINES`           | No       | Preprocessor defines for listed `SUB_PROJECTS` when enabled.                     |
| `SUB_PROJECTS`          | No\*     | Target names for scoped `CMAKE_VARS`/`CPP_DEFINES`. \*Required if either is set. |

BOOL values accept `ON/OFF`, `TRUE/FALSE`, `YES/NO`, `Y/N`, and `1/0`, and are
normalized to `0` or `1` in provider state. INTEGER values must use canonical
signed base-10 spelling: `0` or `-?[1-9][0-9]*`. Hexadecimal values, leading
plus signs, leading zeroes, suffixes, and C++ expressions are rejected.

### Using a Flag in CMakeLists.txt

TheRock itself uses its typed cache variables directly for structural choices:

```cmake
if(THEROCK_FLAG_KPACK_SPLIT_ARTIFACTS)
  # Conditional subproject inclusion, dependency wiring, etc.
endif()
```

### Using a Flag in a ROCm Project

Include the helper from the containing super-repository and resolve each flag:

```cmake
include("${ROCM_SUPER_REPO_ROOT}/cmake/ROCMBuildFlags.cmake")

rocm_resolve_build_flag(
  NAME EXAMPLE_INTEGER
  TYPE INTEGER
  CACHE_VARIABLE MY_PROJECT_EXAMPLE_INTEGER
  DEFAULT_VALUE 5
  VALID_VALUES -17 0 5
  DESCRIPTION "Standalone value for the example control"
  OUTPUT_VARIABLE _example_integer
)
```

When `ROCM_BUILD_FLAGS_STATE_FILE` is present, the provider value is
authoritative. Defining `MY_PROJECT_EXAMPLE_INTEGER` in the cache is then an
error, even if it has the same value. This prevents an old project `option()` or
packaging argument from silently shadowing TheRock. Without provider state,
the project cache variable is created normally and an explicit standalone
`-D` value takes precedence over `DEFAULT_VALUE`.

Use the resolved values to configure a project-specific, private header. The
header should provide this fail-closed accessor:

```c
#define ROCM_BUILD_FLAG_CAT_IMPL(a, b) a##b
#define ROCM_BUILD_FLAG_CAT(a, b) ROCM_BUILD_FLAG_CAT_IMPL(a, b)
#define ROCM_BUILD_FLAG(name) \
  (ROCM_BUILD_FLAG_CAT(ROCM_BUILD_FLAG_INTERNAL_, name)())

#define ROCM_BUILD_FLAG_INTERNAL_EXAMPLE_INTEGER() @_example_integer@
```

The function-like expansion is deliberate. An unknown or misspelled name
produces an invalid expression in both normal C/C++ code and `#if`, instead of
the preprocessor's usual undefined-identifier-to-zero behavior.

The complete canonical example is
[`base/aux-overlay`](../../base/aux-overlay/CMakeLists.txt). It resolves
ordinary flags declared in `FLAGS.cmake`, configures a private header, and
unconditionally compiles C and C++ object sources containing `#if`,
`_Static_assert`, and `static_assert` checks. Its three canary declarations are
permanent infrastructure checks and must not be removed or repurposed.

### Distribution Boundary

The helper and state file are source/build-time inputs only:

- Do not install `ROCMBuildFlags.cmake`, provider state, or a generated private
  flag header.
- Do not link an exported target to a build-flag helper target.
- Do not include a private flag header from an installed public header.
- Do not add a build-flag `find_dependency()` to an installed package config.

Consequently, an application that uses `find_package()` on an installed ROCm
library does not need TheRock, a ROCm super-repository, or the flag helper. A
future flag that must affect a public header or ABI needs a separate,
project-owned installed configuration contract.

## Branch Configuration

Integration branches can change flag defaults and request optional source sets
by creating a `BRANCH_CONFIG.json` file in the project root:

```json
{
  "flags": {
    "INCLUDE_HRX": "ON",
    "EXAMPLE_INTEGER": -17
  },
  "source_sets": ["optional-hrx"],
  "artifact_groups": {
    "core-runtime": {
      "source_sets": ["optional-hrx"]
    }
  }
}
```

At configure time, `build_tools/topology_to_cmake.py` reads
`BRANCH_CONFIG.json` and generates a `therock_apply_branch_config_flags()`
macro that calls `therock_override_flag_default()` for each entry in `flags`.
`FLAGS.cmake` invokes that generated macro before `therock_finalize_flags()`.

Flag values may be strings, JSON booleans, or JSON integers. Explicit `-D`
flags on the CMake command line always take precedence over branch defaults.

### Optional Source Sets

`BRANCH_CONFIG.json` also controls optional source fetching:

- Top-level `"source_sets"` are fetched by the default
  `build_tools/fetch_sources.py` invocation when no `--stage` is specified.
- `"artifact_groups"` source sets are fetched when `fetch_sources.py --stage`
  selects a stage containing that artifact group.
- `fetch_sources.py --source-sets <name>` can force extra source sets for any
  invocation.
- `fetch_sources.py --list-source-sets` lists available source sets, including
  optional external git checkouts.

Optional external git sources are declared in `BUILD_TOPOLOGY.toml` source sets
with `external_git_sources` entries and are fetched under the ignored
`optional-sources/` directory. For example:

```toml
[source_sets.optional-hrx]
description = "Optional HRX source checkout"
external_git_sources = [
  { name = "hrx", origin = "https://github.com/ROCm/hrx.git", commit = "e642a13425f46bcf909078459dd4e07df0723a0d", path = "optional-sources/hrx" },
]
```

### Legacy Branch Flags

Existing branches can still change flag defaults by creating a
`BRANCH_FLAGS.cmake` file in the project root:

```cmake
# BRANCH_FLAGS.cmake
# Override defaults for the kpack-integration branch.
therock_override_flag_default(KPACK_SPLIT_ARTIFACTS ON)
```

`BRANCH_FLAGS.cmake` remains supported for compatibility. When both files are
present, `BRANCH_CONFIG.json` flag defaults are applied after
`BRANCH_FLAGS.cmake`.

## Manifest Integration

Flag states are recorded in the TheRock manifest (`share/therock/therock_manifest.json`)
under a `"flags"` key:

```json
{
  "the_rock_commit": "abc123...",
  "submodules": [...],
  "flags": {
    "KPACK_SPLIT_ARTIFACTS": false,
    "EXAMPLE_INTEGER": -17
  }
}
```

This is generated automatically: `therock_finalize_flags()` writes
`flag_settings.json` to the build directory, which is passed to
`generate_therock_manifest.py` via the aux-overlay subproject.

## Adding a New Flag

1. Add a `therock_declare_flag()` call in `FLAGS.cmake`.
1. Use `THEROCK_FLAG_{NAME}` in the relevant CMakeLists.txt files for
   structural decisions (conditional subproject inclusion, dependency wiring).
1. For a new ROCm C or C++ consumer, follow the aux-overlay
   `rocm_resolve_build_flag()` and private-header pattern. Use legacy
   propagation only when maintaining an existing BOOL integration.
1. Verify standalone defaults and `-D` overrides, integrated provider
   resolution, generated state, manifest types, and C/C++ compilation.

## Alternatives Considered

### Plumbing individual flags to subprojects via CMAKE_ARGS

Before the flag system, each flag's effects were manually forwarded to
subprojects in their `therock_cmake_subproject_declare()` calls. For example,
`THEROCK_KPACK_SPLIT_ARTIFACTS` required manual `-DROCM_KPACK_ENABLED=ON`
forwarding to hip-clr. This approach doesn't scale and is error-prone: adding a
new flag requires modifying multiple declaration sites.

### Plumbing flags to the manifest generator individually

For manifest integration, each flag could be passed as its own CMake variable to
the aux-overlay subproject, then read by `generate_therock_manifest.py`. This
was rejected in favor of generating a single `flag_settings.json` file that is
splat into the manifest, avoiding per-flag plumbing.

### Merging flags into the feature system

Flags could be added as a new mode in `therock_features.cmake`. However,
features and flags serve fundamentally different purposes (inclusion vs
configuration), and mixing them would complicate the feature dependency
resolution logic.

### Installing a common build-flags package

An installed CMake package or exported interface target would cause downstream
`find_package()` consumers to acquire a build-system implementation dependency.
Build flags are resolved while building a library and compiled into its private
implementation instead.

### Using undefined preprocessor macros as false

Plain `#if SOME_FLAG` silently treats a misspelled or unavailable macro as zero.
The function-like `ROCM_BUILD_FLAG(name)` spelling makes unavailable names
syntactically invalid and therefore fails during compilation.
