# Build Flags

Build flags are system-wide controls that affect how TheRock subprojects are
configured. Each flag creates a `THEROCK_FLAG_{NAME}` CMake cache variable and
can optionally propagate CMake variables and C preprocessor defines to all or
specific subprojects.

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
  └── BRANCH_FLAGS.cmake       →  Optional per-branch default overrides
  └── therock_finalize_flags() →  Propagation data + flag_settings.json
  └── therock_report_flags()   →  Status output

cmake/therock_flag_utils.cmake   Processing functions
cmake/therock_subproject.cmake   Injection via project_init.cmake
```

### Propagation Mechanism

When a flag is enabled, its effects are injected into subprojects via the
generated `project_init.cmake` files (the same mechanism used for
`THEROCK_DEFAULT_CMAKE_VARS`):

- **GLOBAL_CMAKE_VARS**: `VAR=VALUE` pairs set in the super-project and
  propagated to **all** subprojects via `THEROCK_DEFAULT_CMAKE_VARS`.
- **GLOBAL_CPP_DEFINES**: Preprocessor defines added to **all** subprojects
  via `add_compile_definitions()` in project_init.cmake.
- **CMAKE_VARS**: `VAR=VALUE` pairs injected only into the listed
  **SUB_PROJECTS** via `set(VAR VALUE CACHE STRING "" FORCE)`.
- **CPP_DEFINES**: Preprocessor defines added only to the listed
  **SUB_PROJECTS** via `add_compile_definitions()`.

Structural concerns (conditional subproject inclusion, runtime dependency
wiring) remain as explicit conditionals in the consuming CMakeLists.txt files.
Flags do not auto-include subprojects.

## Declaring a Flag

All flags are declared in `FLAGS.cmake` at the project root:

```cmake
therock_declare_flag(
  NAME KPACK_SPLIT_ARTIFACTS
  DEFAULT_VALUE OFF
  DESCRIPTION "Split target-specific artifacts into generic and arch-specific components"
  ISSUE "https://github.com/ROCm/TheRock/issues/3448"
  CMAKE_VARS
    ROCM_KPACK_ENABLED=ON
  SUB_PROJECTS
    hip-clr
)
```

### Parameters

| Parameter            | Required | Description                                                                      |
| -------------------- | -------- | -------------------------------------------------------------------------------- |
| `NAME`               | Yes      | Unique identifier. Creates `THEROCK_FLAG_{NAME}` cache variable.                 |
| `DEFAULT_VALUE`      | Yes      | `ON` or `OFF`.                                                                   |
| `DESCRIPTION`        | Yes      | Short description shown in CMake cache UI.                                       |
| `ISSUE`              | No       | Tracking issue URL.                                                              |
| `GLOBAL_CMAKE_VARS`  | No       | `VAR=VALUE` pairs for all subprojects.                                           |
| `GLOBAL_CPP_DEFINES` | No       | Preprocessor defines for all subprojects.                                        |
| `CMAKE_VARS`         | No       | `VAR=VALUE` pairs for listed `SUB_PROJECTS` only.                                |
| `CPP_DEFINES`        | No       | Preprocessor defines for listed `SUB_PROJECTS` only.                             |
| `SUB_PROJECTS`       | No\*     | Target names for scoped `CMAKE_VARS`/`CPP_DEFINES`. \*Required if either is set. |

### Using a Flag in CMakeLists.txt

Flags are regular CMake cache variables, so consuming code uses them directly:

```cmake
if(THEROCK_FLAG_KPACK_SPLIT_ARTIFACTS)
  # Conditional subproject inclusion, dependency wiring, etc.
endif()
```

## Branch Flag Overrides

Integration branches can change flag defaults by creating a
`BRANCH_FLAGS.cmake` file in the project root:

```cmake
# BRANCH_FLAGS.cmake
# Override defaults for the kpack-integration branch.
therock_override_flag_default(KPACK_SPLIT_ARTIFACTS ON)
```

`BRANCH_FLAGS.cmake` is `.gitignore`d on main but can be committed on
integration branches. Overrides are logged to the configure output so they are
visible in CI.

Explicit `-D` flags on the cmake command line always take precedence over
branch overrides.

## Manifest Integration

Flag states are recorded in the TheRock manifest (`share/therock/therock_manifest.json`)
under a `"flags"` key:

```json
{
  "the_rock_commit": "abc123...",
  "submodules": [...],
  "flags": {
    "KPACK_SPLIT_ARTIFACTS": false
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
1. If the flag needs to set variables or defines in subprojects, use the
   `CMAKE_VARS`, `CPP_DEFINES`, `GLOBAL_CMAKE_VARS`, or `GLOBAL_CPP_DEFINES`
   parameters to automate propagation.
1. Run cmake configure and verify the flag report output and, if applicable,
   inspect the generated `project_init.cmake` files.

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
