# Building ROCm with Sanitizers

## Basic Usage

Sanitizers can be enabled via the `THEROCK_SANITIZER` variable. We will be extending this to support all sanitizers. Presently supported sanitizers are:

- `ASAN` : Enables full ASAN with both host-side (`-fsanitize=address`) and device-side instrumentation (xnack+ GPU targets for gfx942, gfx950). Requires xnack-capable hardware at runtime.
- `HOST_ASAN` : Enables host-side ASAN only (`-fsanitize=address` compiler flags) without device-side instrumentation. Use this when:
  - You don't have xnack-capable hardware (gfx942, gfx950)
  - You want faster builds (no xnack+ kernel variants)
  - You only need to catch host-side memory errors
- `OFF` : Explicitly disable sanitizers.
- `CSAN` : Enables host and AMDGPU ConcurrencySanitizer (`-fsanitize=concurrency`) on Linux x86-64. GPU instrumentation uses the HSA runtime and does not require xnack.

The sanitizer selection can be controlled per project by using a variable of the form `{subproject}_SANITIZER={VALUE}`. This is most commonly used to disable santiziers for specific projects once enabled globally.

Because ROCm includes a compiler and uses multiple toolchains to build, there are only certain configurations of the project that support ASAN: generally, we allow ASAN to be enabled for any component that is built with the ROCm version of LLVM. This ensures that we only link to a single ASAN support library (each process can have only one) and can setup RPATH entries and other settings so that anything so compiled will function without further flags, preloads, path settings, etc. We are in the process of switching more of the project to bootstrap off of the built-in compiler, which will make more of the project capable of being instrumented out of the box with a sanitizer.

### CMake Preset

In order to simplify use, the following presets are available for setting up specific sanitizer strategies:

- `--preset linux-release-asan`: Full ASAN build with both host and device instrumentation. Enables ASAN globally and selectively disables it for the compiler and certain system libraries that are not yet ready for generic sanitizer builds. Requires xnack-capable hardware (gfx942, gfx950) at runtime.
- `--preset linux-release-host-asan`: Host-only ASAN build without device-side instrumentation. Same as above but GPU_TARGETS are not modified to include xnack+ variants. Can run on any GPU hardware.
- `--preset linux-release-csan`: ConcurrencySanitizer with optimized code and debug locations. Uses the built ROCm compiler and shared sanitizer runtime; bootstrap compiler tools and the same unsupported system components remain uninstrumented.
- TODO: compiler-asan preset: We will enable a build mode such that the compiler and base libraries can also be instrumented. We will use this for qualifying compiler builds but not generally for *using* the compiler.

## ConcurrencySanitizer

Configure a separate build tree using the pinned compiler submodules:

```bash
cmake --preset linux-release-csan -B build-csan -DTHEROCK_AMDGPU_FAMILIES=gfx1100
cmake --build build-csan
```

The development branch pins LLVM to `641fded1f5684d68f0f678d84ac46437936b0d42`,
SPIRV-LLVM-Translator to `88b7c5b2c6f14a54f307cc620cea236524965fc9`, and hipify
to `d364ef407b2c3e20c6ceb3575bd852f0c7cebb8c`. The LLVM AMDGPU runtime cache at
this revision includes `csan`. Keep the LLVM monorepo's `libc` sources available:
the GPU runtime and host HSA interceptors use its common utility headers.
The existing compiler artifact includes the Clang resource directory containing
the host shared runtime, GPU runtime, and offload initialization archives.

Compile and link consuming applications with `-fsanitize=concurrency -shared-libsan`
and `-g -O1` or higher optimization using the same compiler. CSan cannot be combined
with ASan or TSan. It samples accesses, so a clean run does not prove freedom from
races. See [Clang's ConcurrencySanitizer documentation](https://clang.llvm.org/docs/ConcurrencySanitizer.html)
for `CSAN_OPTIONS`, detection behavior, and supported platforms.

### CI artifacts and build measurements

The manual [ConcurrencySanitizer workflow](../../.github/workflows/ci_csan.yml)
uses the same setup, stage dependency graph, per-family build matrices, and S3
artifact handoffs as the ASAN portable Linux build. Each stage uses
`linux-release-csan` and enables build metrics. The `linux_amdgpu_families` input
accepts comma-separated family keys (`gfx94x,gfx950`); its default is `gfx94x`,
and `all` selects all CSAN-enabled CI families. This CI allowlist follows ASAN;
local CSAN builds do not require xnack-capable GPUs.

Stage reuse is disabled so measurements come from this run's builds. The workflow
runs stage build checks and packaging validation; GPU execution tests, framework
builds, native packages, and release publication are separate from this workflow.
The WSL ROCDXG stage keeps its existing Windows/WSL runner and manylinux container.
The initial sanity checks cover the Linux CI container and workflow configuration;
a complete CSAN build and execution of the Windows/WSL stage remain to be validated.

Download each stage's outputs from the GitHub Actions run's **Artifacts** section:

- `csan-artifacts-<stage>-<family>-<attempt>` contains a tar archive of component
  artifacts and manifests, preserving permissions and symlinks.
- `csan-metrics-<stage>-<family>-<attempt>` contains metrics, Ninja logs, component
  logs, source revisions, and configure diagnostics. The WSL stage bundles these
  files in `csan-stage-metrics.tar.gz` for transfer to the Windows upload action.

Generic stages use `generic` in place of the family. Uploads also run on failure
to retain any partial results available; these downloads do not replace the S3
artifacts used by dependent stages.

To collect the same measurements locally, configure with
`-DTHEROCK_BUILD_METRICS=ON`, build, then run:

```bash
python build_tools/build_metrics.py report --build-dir build-csan --output-dir build-csan/metrics
```

The report combines component stage measurements with Ninja logs and binary
sizes. Ninja command durations measure elapsed time, not CPU consumption;
resource measurements are collected while each component stage runs. CPU percent
uses one core as 100% and can exceed 100%. RAM is sampled aggregate process-tree
RSS: shared pages can be counted more than once and short-lived peaks can be
missed. The separate maximum-child RSS field is not an aggregate peak. Binary
sizes are staged ELF/archive file sizes, excluding symlinks and merged dist
copies; per-component totals are in `binary_totals.csv`. Records from incremental
builds are retained, so use fresh build directories for comparisons. A full
build is required to obtain representative measurements.

## Sanitizer Aware Project Development

### GFX Target Munging (ASAN only)

When using full `ASAN` mode (not `HOST_ASAN`), certain GFX targets have special hardware support for device-side ASAN. This is enabled transparently where available by changing the `GPU_TARGETS` variable propagated to sub-projects:

- `gfx942` -> `gfx942:xnack+`
- `gfx950` -> `gfx950:xnack+`

Some sub-projects have strict gfx target checks that do not allow these extends target specifiers. In general, sub-projects should take targets as given vs trying to guess whether some should be ASAN enabled.

### CMake Variables

When a project is configured for sanitizers, it will have certain variables injected into it. While it is often possible to not require projects to have any special knowledge of what sanitizer they were compiled for, some do need to know. For these cases, it can be necessary to use these special variables so injected.

- `THEROCK_SANITIZER={ASAN|HOST_ASAN|TSAN|CSAN}` : Set if a sanitizer is active for the project and indicates which one. `ASAN` enables both host and device ASAN; `HOST_ASAN` enables host-only ASAN; `CSAN` enables host and device concurrency instrumentation.
- `THEROCK_SANITIZER_LAUNCHER` : If invoking certain tools at build time that dynamically link to a shared library compiled with a sanitizer, you need to prefix it with this value as `${THEROCK_SANITIZER_LAUNCHER}` (not surrounded in quotes so it can expand to multiple terms). This is most commonly needed for invoking system-python and importing native extensions that were built in the project with ASAN. This is set for both `ASAN` and `HOST_ASAN` modes.

You are recommended to code defensively with patterns like:

```cmake
if(NOT DEFINED THEROCK_SANITIZER)
  set(THEROCK_SANITIZER)
endif()
if(NOT DEFINED THEROCK_SANITIZER_LAUNCHER)
  set(THEROCK_SANITIZER_LAUNCHER)
endif()
```

Note that you may need to add additional environment variable special casing to support all launch modes. For example, when using system python, it is common to disable the leak sanitizer because non-debug Python (by design) does not deallocate on exit.

Example:

```cmake
if(NOT DEFINED THEROCK_SANITIZER_LAUNCHER)
  set(THEROCK_SANITIZER_LAUNCHER)
endif()
set(PYTHON_LAUNCHER ${THEROCK_SANITIZER_LAUNCHER} "${Python3_EXECUTABLE}")
# Disable leak detector for any ASAN mode (both ASAN and HOST_ASAN)
if(THEROCK_SANITIZER STREQUAL "ASAN" OR THEROCK_SANITIZER STREQUAL "HOST_ASAN")
  list(PREPEND PYTHON_LAUNCHER "${CMAKE_COMMAND}" -E ASAN_OPTIONS=detect_leaks=0 --)
endif()

# Then use ${PYTHON_LAUNCHER} (without quotes) in command lists.
```

We may add more built-in helpers if we see a lot of common usages like this. However, if history is a guide, projects that need to special case, often need to do so in bespoke ways that are hard to make common.

### Compile Time Checks

If needing to activate specific codepaths in ROCm libraries specifically for ASAN, CMake plumbing can be done to set ad-hoc defines, but it is highly recommended to instead use built-in compile definitions for this. For example:

```c
#if defined(__has_feature)
# if __has_feature(address_sanitizer)
    // Code that builds only when AddressSanitizer is enabled
#  define ASAN_ENABLED
# endif
#endif

// Later in your code:
#ifdef ASAN_ENABLED
    // Perform AddressSanitizer-specific actions
#endif
```

Note that this only works for clang so must not be done in public/API headers that may be used by any consumer.

## Implementation Details

Handling sanitizer builds can be very complicated because of the need to fit all details together. We do it in several places:

- During toolchain selection, routines are called in `therock_sanitizers.cmake` to compute and set variables/options for the sub-project.
- In `therock_subproject_dep_provider.cmake` (TODO: rename to `therock_subproject_init.cmake` to signify what it has become), we do initialization within the project. This includes:
  - Handling the special `THEROCK_INCLUDE_CLANG_RESOURCE_DIR_RPATH` injected variable which tells us to query `clang` for its resource directory and setup both BUILD and INSTALL RPATH extensions so that we can always find the sanitizer runtime.
  - This augments the lists that are used in `therock_global_post_subproject.cmake` to walk through all targets and setup RPATHs.
    - `THEROCK_PRIVATE_INSTALL_RPATH_DIRS`
    - `THEROCK_PRIVATE_INSTALL_RPATH_DIRS`
  - Setup `THEROCK_SANITIZER_LAUNCHER`

## Using ASan-Instrumented Libraries

Once you have ASan-instrumented ROCm libraries installed (via
[native packages or tarball](../../RELEASES.md#installing-asan-instrumented-libraries)),
set `ROCM_ASAN_PATH` to the root of your ASan install tree, then configure the
remaining environment variables in the shell from which you launch your
application. Your production ROCm installation is left untouched; unset the
variables to return to normal operation.

```bash
# DEB / RPM install:
export ROCM_ASAN_PATH=/opt/rocm/core
# Tarball install:
export ROCM_ASAN_PATH=<EXTRACT_PATH>
```

### Step 1: Enable XNACK

Required for device-side GPU instrumentation:

```bash
export HSA_XNACK=1
```

Without this, only host-side (CPU) errors will be detected.

### Step 2: Locate the ASan runtime directory

The host ASan runtime is used here only to resolve the directory where all ASan
runtime libraries are installed:

```bash
ASAN_LIB_PATH=$(amdclang --print-file-name=libclang_rt.asan-x86_64.so)
# ${ASAN_LIB_PATH%/*} strips the filename, giving the containing directory
```

### Step 3: Set library paths

Choose **one** of the following options:

**Option 1 — `LD_LIBRARY_PATH` (recommended)**

Prepend the instrumented library directories so the loader finds them ahead of
the default ROCm libraries:

```bash
export LD_LIBRARY_PATH="${ASAN_LIB_PATH%/*}:${ROCM_ASAN_PATH}/lib:${ROCM_ASAN_PATH}/lib/llvm/lib:${ROCM_ASAN_PATH}/lib/rocm_sysdeps/lib/:$LD_LIBRARY_PATH"
```

- `${ASAN_LIB_PATH%/*}` — ASan runtime directory, so the runtime is resolved
  without a separate `LD_PRELOAD`
- `${ROCM_ASAN_PATH}/lib` — instrumented ROCm libraries (e.g. `libamdhip64.so`,
  `libhsa-runtime64.so`)
- `${ROCM_ASAN_PATH}/lib/llvm/lib` — LLVM support libraries
- `${ROCM_ASAN_PATH}/lib/rocm_sysdeps/lib/` — ROCm system dependency libraries

**Option 2 — `LD_PRELOAD` + `LD_LIBRARY_PATH`**

Use this when the application resolves ROCm libraries through a hardcoded
`RPATH`/`RUNPATH` that `LD_LIBRARY_PATH` cannot override. `LD_PRELOAD` forces
the ASan runtime and the specified instrumented libraries to be loaded before
anything else, bypassing RPATH lookups:

```bash
ASAN_LIB_NAME=$(amdclang --print-file-name=libclang_rt.asan-gfx90a.so)
export LD_PRELOAD="${ASAN_LIB_PATH%/*}/$ASAN_LIB_NAME:${ROCM_ASAN_PATH}/lib/libamdhip64.so:${ROCM_ASAN_PATH}/lib/libhsa-runtime64.so"
```

> [!NOTE]
> `ASAN_LIB_NAME` holds only the filename of the GPU-specific ASan runtime
> (e.g. `libclang_rt.asan-gfx90a.so`). Replace `gfx90a` with your target GPU
> architecture. The GPU-specific `--print-file-name` returns only a filename,
> not a full path — that is why `ASAN_LIB_PATH` (derived from the host variant
> in Step 2) is used to supply the directory. Add or remove libraries from
> `LD_PRELOAD` depending on which ROCm components you need to instrument.

## Troubleshooting

TODO: Add troubleshooting tips here.
