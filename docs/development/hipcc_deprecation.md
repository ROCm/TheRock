# hipcc Deprecation Guide

`hipcc` is a compiler wrapper that historically injected HIP-specific flags
(include paths, device library paths, GPU target flags, etc.) before invoking
`clang++`. `hipcc` is being deprecated in favor of calling `amdclang++`
directly. `amdclang++` is a user-facing binary (`amdllvm`) that provides the
same HIP-capable compiler as `clang++` but under the AMD-branded name, with the
HIP toolchain configured by CMake.

This document explains why the change is being made, what the equivalents are,
and how to migrate your project.

## Why deprecate hipcc?

- **Redundant indirection.** CMake toolchains can inject `--hip-path`,
  `--hip-device-lib-path`, and all other necessary flags via
  `CMAKE_CXX_FLAGS_INIT`. hipcc was doing the same thing via shell script logic,
  creating two overlapping mechanisms for the same job.
- **CMake-native HIP support.** CMake has had first-class HIP language support
  since CMake 3.21. Projects using `enable_language(HIP)` or
  `set_source_files_properties(... PROPERTIES LANGUAGE HIP)` work correctly with
  `CMAKE_HIP_COMPILER` pointing to `amdclang++` directly — no wrapper needed.
- **Transparency.** Direct `amdclang++` invocations produce cleaner build logs,
  better IDE integration (via `compile_commands.json`), and easier debugging of
  compiler flags.
- **hipconfig.** `hipconfig` is the companion introspection tool bundled with
  `hipcc`. Callers that used `hipconfig --version`, `--cxxflags`, or `--ldflags`
  should migrate to CMake-native equivalents (see below).

## Flag equivalency table

The following table maps hipcc's implicit behavior to the explicit flags or
CMake variables that replace it when using `amdclang++` directly.

| hipcc behavior                                       | amdclang++ / CMake equivalent                                                                                                                                                 |
| ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Injects `-I<hip>/include`                            | `--hip-path=<path>`                                                                                                                                                           |
| Injects `--hip-device-lib-path=<path>`               | `--hip-device-lib-path=<path>` (usually derivable from `--hip-path`)                                                                                                          |
| Sets HIP compiler in CMake                           | `CMAKE_HIP_COMPILER` set to `clang++` (user-facing as `amdclang++`) in toolchain                                                                                              |
| Auto-detects GPU targets via `rocm_agent_enumerator` | `--offload-arch=native` (build machine's GPUs) or explicit `--offload-arch=gfxNNNN`, or `CMAKE_HIP_ARCHITECTURES` in CMake                                                    |
| Defines `-D__HIP_PLATFORM_AMD__`                     | Added by `hip-config.cmake` target properties via `find_package(hip)`                                                                                                         |

## Migrating a CMake project

### Compiler

Replace any explicit `CMAKE_CXX_COMPILER=hipcc` with `CMAKE_CXX_COMPILER=amdclang++`:

```cmake
# Before
cmake -DCMAKE_CXX_COMPILER=hipcc ...

# After
cmake -DCMAKE_CXX_COMPILER=amdclang++ ...
```

If your project uses CMake's native HIP language support, set
`CMAKE_HIP_COMPILER` instead of (or in addition to) `CMAKE_CXX_COMPILER`:

```cmake
cmake -DCMAKE_HIP_COMPILER=amdclang++ ...
```

### HIP_HIPCC_EXECUTABLE

Some projects (including older versions of libhipcxx) use the
`HIP_HIPCC_EXECUTABLE` CMake variable to locate hipcc. Replace this with the
direct path to `amdclang++`:

```cmake
# Before
-DHIP_HIPCC_EXECUTABLE=/path/to/rocm/bin/hipcc

# After
# Not needed — use CMAKE_HIP_COMPILER or CMAKE_CXX_COMPILER instead
```

### hipconfig introspection

Projects that called `hipconfig` to retrieve version or path information should
migrate to CMake-native equivalents:

| hipconfig invocation      | CMake equivalent                                            |
| ------------------------- | ----------------------------------------------------------- |
| `hipconfig --version`     | `hip_VERSION` (set by `find_package(hip REQUIRED)`)         |
| `hipconfig --hip-version` | `hip_VERSION`                                               |
| `hipconfig --cxxflags`    | Use `hip::host` and `hip::device` imported targets          |
| `hipconfig --ldflags`     | Use `hip::host` and `hip::device` imported targets          |
| `hipconfig --hippath`     | `HIP_PATH` environment variable or `hip_DIR` CMake variable |
| `hipconfig --rocmpath`    | `ROCM_PATH` environment variable                            |
| `hipconfig --compiler`    | `CMAKE_HIP_COMPILER`                                        |

Example of migrating a version check in CMake:

```cmake
# Before — calls hipconfig at configure time
find_program(HIPCONFIG_EXEC hipconfig REQUIRED)
execute_process(
  COMMAND ${HIPCONFIG_EXEC} --version
  OUTPUT_VARIABLE HIP_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# After — version already available from find_package
find_package(hip REQUIRED)
set(HIP_VERSION ${hip_VERSION})
```

## Migrating a Makefile or shell-based project

For projects that invoke `hipcc` directly in a Makefile or shell script,
replace it with `amdclang++` and add the required flags explicitly:

```bash
# Before
hipcc -o my_kernel my_kernel.cpp --offload-arch=gfx1100

# After
amdclang++ \
  --hip-path=${ROCM_PATH} \
  --hip-device-lib-path=${ROCM_PATH}/lib/llvm/amdgcn/bitcode \  # usually derivable from --hip-path; explicit here for clarity
  -x hip \
  --offload-arch=gfx1100 \  # or --offload-arch=native to target the build machine's GPUs
  -o my_kernel my_kernel.cpp
```

The `--hip-path` and `--hip-device-lib-path` values are available from the
`hip-config.cmake` installed in `<rocm_install>/lib/cmake/hip/`.
