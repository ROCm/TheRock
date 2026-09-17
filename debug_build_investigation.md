# TheRock Debug Build Investigation

## Summary

Investigation into enabling a global `Debug` build type in TheRock, the failures
encountered, root causes, and the final fix applied.

---

## Background

TheRock is a CMake super-project for building HIP and ROCm from source. By default
it builds with `Release`. The goal was to enable `Debug` globally so all components
produce binaries with full debug symbols (`-g -O0`) for GDB-based debugging.

### CMake Build Types

| Build Type | Compiler Flags | Use Case |
|------------|---------------|----------|
| `Debug` | `-g -O0` | Full debugging, no optimization |
| `Release` | `-O3 -DNDEBUG` | Production, max speed |
| `RelWithDebInfo` | `-O2 -g -DNDEBUG` | Debug info + moderate optimization |
| `MinSizeRel` | `-Os -DNDEBUG` | Smallest binary size |

---

## Approach

### Change Made (`CMakeLists.txt`)

Changed the default build type from `Release` to `Debug` in the root
`CMakeLists.txt`, with per-component overrides for components that cannot
build correctly under `-O0`:

```cmake
# Set the default build type to Debug if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# When building Debug globally, certain components must be overridden:
#   - amdsmi: uses _FORTIFY_SOURCE=2 which requires at least -O1
#   - therock-host-blas / therock-host-blas64: OpenBLAS installs to a
#     config-specific subdir (e.g. lib/Debug/) in Debug mode, breaking the
#     expected path lib/host-math/lib/librocm-openblas.so
#   - therock-SuiteSparse: third-party lib with no debug build support
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  set(amdsmi_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type for amdsmi" FORCE)
  set(therock-host-blas_BUILD_TYPE Release CACHE STRING "Build type for host-blas" FORCE)
  set(therock-SuiteSparse_BUILD_TYPE Release CACHE STRING "Build type for SuiteSparse" FORCE)
endif()
```

### How Per-Component Override Works

`cmake/therock_subproject.cmake` checks for a `{component}_BUILD_TYPE` variable
before falling back to the global `CMAKE_BUILD_TYPE`:

```cmake
set(_cmake_build_type "${${target_name}_BUILD_TYPE}")
if(NOT _cmake_build_type)
  set(_cmake_build_type "${CMAKE_BUILD_TYPE}")
endif()
```

So `amdsmi_BUILD_TYPE=RelWithDebInfo` overrides only `amdsmi`, while all other
components inherit `Debug`.

---

## Failures At a Glance

| Component | Failed Job | Error Snippet |
|-----------|-----------|---------------|
| `amdsmi` | [job/102149094367](https://github.com/ROCm/rockrel/actions/runs/34199215189/job/102149094367) | `error: _FORTIFY_SOURCE requires compiling with optimization (-O) [-Werror,-W#warnings]` |
| `therock-host-blas` | [job/104901696635](https://github.com/ROCm/rockrel/actions/runs/35127952788/job/104901696635) | `OSError: librocm-openblas.so: cannot open shared object file: No such file or directory` |
| `therock-host-blas64` | [job/104901696635](https://github.com/ROCm/rockrel/actions/runs/35127952788/job/104901696635) | `OSError: librocm-openblas64.so: cannot open shared object file: No such file or directory` |

---

## Failures Encountered

### Failure 1 — `amdsmi` build error

**CI Run:** https://github.com/ROCm/rockrel/actions/runs/34199215189/job/102149094367

**Error:**
```
[amdsmi] /usr/include/features.h:381:4: error: _FORTIFY_SOURCE requires compiling
with optimization (-O) [-Werror,-W#warnings]
1 error generated.
```

**Root Cause:**

`amdsmi` sets `-D_FORTIFY_SOURCE=2` in its compile flags. This Linux/glibc
hardening macro requires at least `-O1` to function. When the global build type
is `Debug` (`-O0`), the compiler (with `-Werror`) rejects the combination.

**Fix:** Pin `amdsmi` to `RelWithDebInfo` when the global type is `Debug`.

```cmake
set(amdsmi_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type for amdsmi" FORCE)
```

---

### Failure 2 — `librocm-openblas.so` not found

**CI Run:** https://github.com/ROCm/rockrel/actions/runs/35127952788/job/104901696635

**Error:**
```
OSError: .../build/third-party/host-blas/host-blas/dist/lib/host-math/lib/
librocm-openblas.so: cannot open shared object file: No such file or directory

OSError: .../build/third-party/host-blas/host-blas64/dist/lib/host-math/lib/
librocm-openblas64.so: cannot open shared object file: No such file or directory
```

**Root Cause:**

OpenBLAS (used by `therock-host-blas`) changes its install layout based on
`CMAKE_BUILD_TYPE`. In `Debug` mode it installs libraries to a config-specific
subdirectory (e.g. `lib/Debug/`) instead of the expected flat path
`lib/host-math/lib/`. The validation test `validate_shared_library.py` looks
for the `.so` at the hardcoded path `host-blas/dist/lib/host-math/lib/` and
fails when the file is not there.

**Fix:** Pin `therock-host-blas` and `therock-SuiteSparse` to `Release` when
the global type is `Debug` — matching the same overrides used in all existing
`CMakePresets.json` presets.

```cmake
set(therock-host-blas_BUILD_TYPE Release CACHE STRING "Build type for host-blas" FORCE)
set(therock-SuiteSparse_BUILD_TYPE Release CACHE STRING "Build type for SuiteSparse" FORCE)
```

---

## Final Component Build Type Summary

With the fix in place, the effective build types per component are:

| Component | Build Type | Reason |
|-----------|-----------|--------|
| All components (default) | `Debug` (`-g -O0`) | Global default — full debug symbols |
| `amdsmi` | `RelWithDebInfo` | `_FORTIFY_SOURCE=2` requires `-O1+` |
| `therock-host-blas` | `Release` | OpenBLAS install path breaks in Debug mode |
| `therock-SuiteSparse` | `Release` | Third-party lib, no Debug build support |
| `amd-llvm` | `Release` | Large compiler toolchain, impractical to debug-build |

> **Note:** `amd-llvm` was already `Release` before this change via existing
> presets. It is not affected by the `CMakeLists.txt` change since this change
> only adds overrides when `CMAKE_BUILD_TYPE=Debug` is the active default.

---

## Existing CI Build Type Context (Before This Change)

The nightly release CI (`linux-release-package` preset) uses `RelWithDebInfo`
globally — not `Debug`. Components already pinned to `Release` in all presets:

| Component | Variable |
|-----------|---------|
| `amd-llvm` | `amd-llvm_BUILD_TYPE=Release` |
| `therock-host-blas` | `therock-host-blas_BUILD_TYPE=Release` |
| `therock-SuiteSparse` | `therock-SuiteSparse_BUILD_TYPE=Release` |

---

## How to Verify the Debug Build is Working

### 1. CMakeCache.txt
```bash
grep "CMAKE_BUILD_TYPE" build/CMakeCache.txt
# Expected: CMAKE_BUILD_TYPE:STRING=Debug

grep "BUILD_TYPE" build/CMakeCache.txt
# Expected: amdsmi_BUILD_TYPE=RelWithDebInfo, therock-host-blas_BUILD_TYPE=Release
```

### 2. Check compiler flags used
```bash
grep "CMAKE_CXX_FLAGS_DEBUG" build/rocblas/build/CMakeCache.txt
# Should contain: -g -O0
```

### 3. Check debug symbols in binary
```bash
file build/dist/rocm/lib/librocblas.so
# Debug: ELF ... not stripped

readelf --debug-dump=info build/dist/rocm/lib/librocblas.so | head -20
# Should show DW_AT_comp_dir, DW_AT_name etc.
```

### 4. Check CI logs
Search the compiler-runtime stage logs for:
```
PROJECT SPECIFIC CMAKE_BUILD_TYPE=RelWithDebInfo   ← amdsmi override active
CMAKE_BUILD_TYPE=Debug                             ← global applied to others
```

### 5. Binary size sanity check
```bash
ls -lh build/dist/rocm/lib/librocblas.so
# Debug binaries are significantly larger than Release
```

---

## CI Build Links

| Run | Status | Notes |
|-----|--------|-------|
| https://github.com/ROCm/rockrel/actions/runs/34171922937 | Passed | Baseline nightly before change (`RelWithDebInfo`) |
| https://github.com/ROCm/rockrel/actions/runs/34199215189/job/102149094367 | Failed | First debug attempt — `amdsmi` `_FORTIFY_SOURCE` error |
| https://github.com/ROCm/rockrel/actions/runs/35127952788/job/104901696635 | Failed | Second attempt — `librocm-openblas.so` not found |

---

## Files Changed

| File | Change |
|------|--------|
| `CMakeLists.txt` | Changed default build type to `Debug`; added per-component overrides for `amdsmi`, `therock-host-blas`, `therock-SuiteSparse` |
