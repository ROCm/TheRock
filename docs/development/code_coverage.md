# Code Coverage

TheRock builds ROCm libraries with LLVM source-based coverage and reports to
[Codecov](https://app.codecov.io/gh/ROCm/TheRock). Coverage is opt-in per
project — instrumentation slows a library substantially, and a report is only
worth producing when the test suite gives a meaningful signal.

## Enabling coverage for a local build

```bash
cmake -B build -GNinja . \
  -DTHEROCK_AMDGPU_FAMILIES=gfx94X-dcgpu \
  -DHIPRAND_ENABLE_COVERAGE=ON
```

`<PROJECT>_ENABLE_COVERAGE` is TheRock's own flag; upstream option names are
not standardised (hipRAND: `BUILD_CODE_COVERAGE`, rocRAND: `CODE_COVERAGE`,
RCCL: `ENABLE_CODE_COVERAGE`), so `therock_subproject.cmake` translates it to
whichever name `COVERAGE_PROJECTS` registers for that project. Passing the flag
for an unregistered project is a configure error, not a silent no-op.

### Enabling a whole group

| Option                                | Instruments                                  |
| ------------------------------------- | -------------------------------------------- |
| `THEROCK_COVERAGE_ROCM_LIBRARIES_ALL` | all coverage-enabled rocm-libraries projects |
| `THEROCK_COVERAGE_ROCM_SYSTEMS_ALL`   | all coverage-enabled rocm-systems projects   |
| `THEROCK_COVERAGE_ALL`                | both of the above                            |

Each expands to individual `<PROJECT>_ENABLE_COVERAGE` flags. An explicit
`-D<PROJECT>_ENABLE_COVERAGE=OFF` wins over a group flag. Membership is
generated from `COVERAGE_PROJECTS`; these options default `OFF` and have no
effect on a normal build.

### Host code only

Upstream coverage flags are unqualified and the HIP driver forwards them to
device compilation. Two failures result: the runtime aborts looking up missing
profiling symbols in the device image, and when readback succeeds it writes a
profile named after the device target (colons included, breaking CI artifact
upload). `therock_subproject.cmake` appends the negation scoped to device
compilation:

```
-Xarch_device -fno-profile-instr-generate -Xarch_device -fno-coverage-mapping
```

The later of a `-f`/`-fno-` pair wins; the project's own flags arrive after
`CMAKE_<LANG>_FLAGS`, so the negation is appended to
`CMAKE_<LANG>_COMPILE_OBJECT` via a generated file passed as
`CMAKE_PROJECT_INCLUDE`.

### The profile runtime's link dependencies

`libclang_rt.profile_rocm.a` needs `-ldl` and `-lpthread`, which the driver
does not supply. On the manylinux base these are not in libc, so an instrumented
shared library gets undefined references and any executable linking it fails
`--no-allow-shlib-undefined`. `therock_subproject.cmake` adds both libraries
automatically.

## Adding a project

Add an entry to `COVERAGE_PROJECTS` in `configure_coverage_ci.py`.

`coverage_option` must name the CMake option the project implements, and it must
select LLVM instrumentation (not gcov, which writes `.gcda` files this pipeline
can't read). Names vary: most use `BUILD_CODE_COVERAGE` or `CODE_COVERAGE`;
RCCL uses `ENABLE_CODE_COVERAGE`.

Verify a local instrumented build produces a non-empty report before committing
the entry. Set `source_repo` (`ROCM_LIBRARIES` or `ROCM_SYSTEMS`) to route the
project into the correct group alias; membership lists are generated at configure
time, so no CMake edit is needed. `object_globs` must match what the project
actually installs — a miss fails the report job rather than publishing zero.

## Blocked projects

A `blocked_reason` alongside `coverage_option` means the instrumented build is
broken. The project is dropped from the default and group aliases (including
`--emit-cmake` output) but can still be named explicitly, printing the reason.
A stage is built as a unit, so one blocked project takes the whole stage down.
Drop `blocked_reason` once the upstream fix lands.

`hipblaslt` — its coverage branch links against bare `rocroller` while the rest
of its CMakeLists uses `roc::rocroller`. Under TheRock rocRoller is a separate
subproject; the bare name reaches the linker as `-lrocroller` with no `-L`.

`hiptensor` — upstream applies coverage as a blanket `add_compile_options` on a
composable_kernel-based library. The coverage mapping data exceeds 2GB, failing
the link with out-of-range `R_X86_64_PC32` relocations. A larger code model is
the likely fix.
