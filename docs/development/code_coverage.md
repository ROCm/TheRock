# Code Coverage

TheRock builds ROCm libraries with LLVM's source based code coverage
instrumentation and reports the results to
[Codecov](https://app.codecov.io/gh/ROCm/TheRock). Coverage is opt-in per
project: instrumentation slows a library down substantially, and the report is
only worth producing for projects whose test suites are thorough enough to give
a meaningful signal.

## Enabling coverage for a local build

Pass `-D<PROJECT>_ENABLE_COVERAGE=ON` at the top level configure. The project
name must be upper case, matching the usual CMake convention for definitions:

```bash
cmake -B build -GNinja . \
  -DTHEROCK_AMDGPU_FAMILIES=gfx94X-dcgpu \
  -DHIPRAND_ENABLE_COVERAGE=ON \
  -DCOMPILER_RT_BUILD_PROFILE_ROCM=ON
```

`therock_subproject.cmake` forwards the flag to the matching subproject, which
is responsible for translating it into compiler flags (usually
`-fprofile-instr-generate -fcoverage-mapping`). Passing the flag for a project
that does not implement it has no effect.

`COMPILER_RT_BUILD_PROFILE_ROCM=ON` builds `libclang_rt.profile_rocm`, the
device side profiling runtime. Without it, instrumented device code cannot write
profiles, and you will only see host coverage.

> [!NOTE]
> Coverage builds are incompatible with split kernel packaging. The CI
> workflows pass `-DTHEROCK_FLAG_KPACK_SPLIT_ARTIFACTS=OFF`; do the same if you
> build with a target family that enables kernel packing by default.

## Producing a report locally

Instrumented binaries write one `.profraw` file per process, named by the
`LLVM_PROFILE_FILE` environment variable. The `%p` (pid) and `%m` (binary
signature) substitutions keep concurrent processes from clobbering each other:

```bash
export LLVM_PROFILE_FILE="$PWD/coverage-report/profraw/%p-%m.profraw"
ctest --test-dir build/math-libs/hiprand
```

`merge_coverage_report.py` then merges the profiles and exports lcov:

```bash
python build_tools/github_actions/merge_coverage_report.py \
  --profraw-dir coverage-report/profraw \
  --rocm-dir build/dist/rocm \
  --object-globs "lib/libhiprand.so*"
```

The `llvm-profdata` and `llvm-cov` binaries must come from the same compiler
that built the instrumented objects, so the script prefers the copies under
`<rocm-dir>/lib/llvm/bin`. A version mismatch surfaces as an unhelpfully generic
"malformed instrumentation profile data" error.

## Coverage CI

Two workflows produce the reports that land in Codecov. Both take their project
list from `build_tools/github_actions/configure_coverage_ci.py`, which is the
registry of coverage-enabled projects and everything the pipeline needs to know
about each one: the CMake target, the build stage it lives in, its test
component, the object globs handed to `llvm-cov`, and its Codecov flag.

| Workflow                             | Trigger                                         | Build strategy                                                                                             |
| ------------------------------------ | ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `multi_arch_ci_coverage.yml`         | `ci:coverage` label on a PR, or manual dispatch | Builds the instrumented compiler-runtime once, then rebuilds only the target project for each matrix entry |
| `multi_arch_ci_coverage_nightly.yml` | Nightly schedule, or manual dispatch            | Builds the whole instrumented stack once, then tests and reports on each project against that single build |

Both delegate to `multi_arch_ci_coverage_linux.yml`, which owns the per-project
build, test, and aggregation sequence for one GPU family. Keeping the nesting at
three levels matters: GitHub refuses to run reusable workflows nested more than
four deep.

The test jobs run through the same `test_component.yml` as regular CI, with
`coverage_enabled: true` adding an `LLVM_PROFILE_FILE` pointing into the
workspace and uploading the resulting profiles as an artifact. The aggregation
job downloads the profiles from every shard, merges them, and uploads the lcov
report.

### Adding a project

Add an entry to `COVERAGE_PROJECTS` in `configure_coverage_ci.py`. Before doing
so, confirm that the project implements `<PROJECT>_ENABLE_COVERAGE` in its own
CMake, and that a local instrumented build produces a non-empty report. A
project whose tests never load the instrumented library will build and test
cleanly but report zero coverage.
