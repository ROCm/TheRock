# Code Coverage

TheRock can build ROCm libraries with LLVM source-based coverage
instrumentation. Coverage is opt-in per project — instrumentation slows a
library substantially, and a report is only worth producing when the test suite
gives a meaningful signal.

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
`BUILD_CODE_COVERAGE`, can be instrumented through its per-subproject arguments:

```bash
  -DhipRAND_CMAKE_ARGS=-DBUILD_CODE_COVERAGE=ON
```

### Test subprojects

`hipCCL_tests` is built from hipCCL's source, so it reads
`HIPCCL_ENABLE_COVERAGE`, not `HIPCCL_TESTS_ENABLE_COVERAGE`. The prim libraries
are header-only, so their coverage comes from the tests build. Pass hipCCL's
option to it directly:

```bash
  -DHIPCCL_ENABLE_COVERAGE=ON \
  -DhipCCL_tests_CMAKE_ARGS=-DHIPCCL_ENABLE_COVERAGE=ON
```

With `-DTHEROCK_FLAG_BUILD_HIPCCL=OFF`, the same applies to `rocPRIM_tests` and
`ROCPRIM_ENABLE_COVERAGE`.

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
- **The profile runtime's link dependencies.** `libclang_rt.profile_rocm.a`
  needs `-ldl` and `-lpthread`, which the driver does not supply. On the
  manylinux base these are not in libc, so an instrumented shared library gets
  undefined references and any executable linking it fails
  `--no-allow-shlib-undefined`. Link `${CMAKE_DL_LIBS}` and `Threads::Threads`
  when coverage is on.

## Adding a project

1. Implement `<PROJECT>_ENABLE_COVERAGE` upstream as described above. Keep the
   project's old option name as a deprecated alias for a while.
1. Bump the submodule in TheRock.
1. Add the subproject to `THEROCK_COVERAGE_ROCM_LIBRARIES_PROJECTS` or
   `THEROCK_COVERAGE_ROCM_SYSTEMS_PROJECTS` in `CMakeLists.txt`.
1. Verify that a local instrumented build produces a non-empty report.

## Blocked projects

These are left out of the group lists because their instrumented builds fail.

`hipBLASLt` — its coverage branch links against bare `rocroller` while the rest
of its CMakeLists uses `roc::rocroller`. Under TheRock rocRoller is a separate
subproject; the bare name reaches the linker as `-lrocroller` with no `-L`.

`hipTensor` — upstream applies coverage as a blanket `add_compile_options` on a
composable_kernel-based library. The coverage mapping data exceeds 2GB, failing
the link with out-of-range `R_X86_64_PC32` relocations. A larger code model is
the likely fix.
