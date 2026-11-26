# Code Coverage

TheRock can build a ROCm library with code coverage instrumentation when the
library implements a `<PROJECT>_ENABLE_COVERAGE` option.

## Enabling coverage for a local build

```bash
cmake -B build -GNinja . \
  -DTHEROCK_AMDGPU_FAMILIES=gfx94X-dcgpu \
  -DTHEROCK_ENABLE_ALL=OFF \
  -DTHEROCK_ENABLE_HIPDNN=ON \
  -DHIPDNN_ENABLE_COVERAGE=ON
```

TheRock passes `-D<PROJECT>_ENABLE_COVERAGE` to the matching subproject
unchanged. `<PROJECT>` is the subproject name in upper case: `hipDNN` reads
`HIPDNN_ENABLE_COVERAGE`. The subproject implements the option itself; TheRock
adds no compiler or linker flags. A subproject that doesn't read the option
builds uninstrumented, and CMake reports the variable as unused at the end of
that subproject's configure.

Most rocm-libraries projects still use an older option name, such as
`BUILD_CODE_COVERAGE`, and build uninstrumented until their CMake moves to
`<PROJECT>_ENABLE_COVERAGE`. Of the projects in the group lists below, only
hipDNN and hipCCL read the new name so far.

### Enabling a whole group

| Option                                | Instruments                                                   |
| ------------------------------------- | ------------------------------------------------------------- |
| `THEROCK_COVERAGE_ROCM_LIBRARIES_ALL` | the subprojects in `THEROCK_COVERAGE_ROCM_LIBRARIES_PROJECTS` |
| `THEROCK_COVERAGE_ROCM_SYSTEMS_ALL`   | the subprojects in `THEROCK_COVERAGE_ROCM_SYSTEMS_PROJECTS`   |
| `THEROCK_COVERAGE_ALL`                | both of the above                                             |

Both lists are in the top-level `CMakeLists.txt`. Each group option sets
`<PROJECT>_ENABLE_COVERAGE=ON` for the subprojects in its list. An explicit
`-D<PROJECT>_ENABLE_COVERAGE=OFF` wins over a group option. These options
default to `OFF` and have no effect on a normal build.

hipCCL's instrumented build doesn't link on the CI build image yet, so add
`-DHIPCCL_ENABLE_COVERAGE=OFF` when using `THEROCK_COVERAGE_ROCM_LIBRARIES_ALL`
or `THEROCK_COVERAGE_ALL`.

## What a project's coverage option must do

TheRock passes only the flag, so `<PROJECT>_ENABLE_COVERAGE` must be complete in
the project itself:

- **LLVM source-based coverage** (`-fprofile-instr-generate -fcoverage-mapping`),
  not gcov. gcov writes `.gcda` files rather than the `.profraw` files the
  coverage tooling merges.
- **Host code only.** For HIP sources, qualify the flags with `-Xarch_host`.
  Otherwise the HIP driver forwards them to device compilation: the runtime
  aborts looking up missing profiling symbols in the device image, and when
  readback succeeds it writes a profile named after the device target (colons
  included, which breaks CI artifact upload).
- **The profile runtime's link dependencies.** `libclang_rt.profile_rocm.a`
  needs libdl and libpthread, which the compiler driver doesn't link. On the
  manylinux_2_28 build image (glibc 2.28) they are separate from libc, so link
  `${CMAKE_DL_LIBS}` and `Threads::Threads` when coverage is on.
