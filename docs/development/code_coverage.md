# Code Coverage

TheRock can build ROCm libraries with LLVM source-based coverage
instrumentation.

## Enabling coverage for a local build

```bash
cmake -B build -GNinja . \
  -DTHEROCK_AMDGPU_FAMILIES=gfx94X-dcgpu \
  -DHIPDNN_ENABLE_COVERAGE=ON
```

TheRock passes `-D<PROJECT>_ENABLE_COVERAGE` to the matching subproject
unchanged. `<PROJECT>` is the subproject name in upper case: `hipDNN` reads
`HIPDNN_ENABLE_COVERAGE`. The subproject implements the option itself; TheRock
adds no compiler or linker flags. A subproject that doesn't read the option
builds uninstrumented, and CMake reports the variable as unused at the end of
that subproject's configure.

A subproject that still uses a legacy option name, such as
TheRock passes `-D<PROJECT>_ENABLE_COVERAGE` to the matching subproject
`BUILD_CODE_COVERAGE`, will be changed to -D<PROJECT>_ENABLE_COVERAGE in subprojects CMakeLists.txt to instrument for code coverage.

### Enabling a whole group

| Option                                | Instruments                                                   |
| ------------------------------------- | ------------------------------------------------------------- |
| `THEROCK_COVERAGE_ROCM_LIBRARIES_ALL` | the subprojects in `THEROCK_COVERAGE_ROCM_LIBRARIES_PROJECTS` |
| `THEROCK_COVERAGE_ROCM_SYSTEMS_ALL`   | the subprojects in `THEROCK_COVERAGE_ROCM_SYSTEMS_PROJECTS`   |
| `THEROCK_COVERAGE_ALL`                | both of the above                                             |

Both lists are in the top-level `CMakeLists.txt`. Each group option sets
`<PROJECT>_ENABLE_COVERAGE=ON` for the subprojects in its list. An explicit
`-D<PROJECT>_ENABLE_COVERAGE=OFF` wins over a group option. These options
default to `OFF` and have no effect on a normal build. Test subprojects are not
in the lists, so pass their arguments as shown above.

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

