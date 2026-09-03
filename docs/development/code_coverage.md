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
  -DHIPRAND_ENABLE_COVERAGE=ON
```

`therock_subproject.cmake` forwards the flag to the matching subproject, which
is responsible for translating it into compiler flags (usually
`-fprofile-instr-generate -fcoverage-mapping`). Passing the flag for a project
that does not implement it has no effect.

Device side profiling relies on the ROCm profiling runtime
(`libclang_rt.profile_rocm`) being present in the compiler build. TheRock does
not configure that from the top level; it comes from how amd-llvm's compiler-rt
is built. If a report shows host coverage only, check that the runtime is in the
compiler you built against.

> [!NOTE]
> Coverage builds are incompatible with split kernel packaging. The CI
> workflows pass `-DTHEROCK_FLAG_KPACK_SPLIT_ARTIFACTS=OFF`; do the same if you
> build with a target family that enables kernel packing by default.

### Enabling a whole group

Three aggregate options turn on coverage for a whole component group instead of
naming each project. All default `OFF`:

| Option                                | Instruments                                                         |
| ------------------------------------- | ------------------------------------------------------------------- |
| `THEROCK_COVERAGE_ROCM_LIBRARIES_ALL` | all coverage-enabled rocm-libraries projects (math-libs, ml-libs …) |
| `THEROCK_COVERAGE_ROCM_SYSTEMS_ALL`   | all coverage-enabled rocm-systems projects                          |
| `THEROCK_COVERAGE_ALL`                | both of the above                                                   |

```bash
cmake -B build -GNinja . \
  -DTHEROCK_AMDGPU_FAMILIES=gfx94X-dcgpu \
  -DTHEROCK_COVERAGE_ROCM_LIBRARIES_ALL=ON
```

Each option expands to the individual `<PROJECT>_ENABLE_COVERAGE` flags above, so
the same device-runtime and kernel-packing caveats apply. An explicit
`-D<PROJECT>_ENABLE_COVERAGE=OFF` on the command line wins over the group flag.
Group membership is generated from the coverage registry (`COVERAGE_PROJECTS`),
so a group only ever contains onboarded projects; selecting a group with none
(currently rocm-systems) configures nothing and prints a warning.

> [!NOTE]
> These options default `OFF` and have no effect on a normal, non-coverage build
> — no `<PROJECT>_ENABLE_COVERAGE` flag is set. The membership list is
> regenerated from `COVERAGE_PROJECTS` on every configure (a small scripted step;
> it also means editing `configure_coverage_ci.py` triggers a reconfigure), but
> it only changes the build when one of the options is `ON`.

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

Coverage reports are produced nightly by
`multi_arch_ci_coverage_nightly.yml`, which builds the whole instrumented stack
once and then tests and reports on each project against that single build. It
delegates to `multi_arch_ci_coverage_linux.yml` for the per-project test and
aggregation sequence on one GPU family.

Both take their project list from
`build_tools/github_actions/configure_coverage_ci.py`, which is the registry of
coverage-enabled projects and everything the pipeline needs to know about each
one: the CMake target, the build stage it lives in, its test component, the
object globs handed to `llvm-cov`, and its Codecov flag.

The instrumented build calls `multi_arch_build_portable_linux_artifacts.yml` —
the same per-stage build workflow regular CI uses — one job per stage, rather
than going through the full `multi_arch_build_portable_linux.yml` pipeline.
Coverage only ever needs compiler-runtime plus the stages owning the projects
under test, so naming those stages directly keeps the shared pipeline free of
coverage-specific stage filtering. `extra_cmake_options` on that workflow is the
only build-side hook coverage adds.

The test jobs run through the same `test_component.yml` as regular CI, with
`coverage_enabled: true` adding an `LLVM_PROFILE_FILE` pointing into the
workspace and uploading the resulting profiles as an artifact. The aggregation
job downloads the profiles from every shard, merges them, and uploads the lcov
report.

### Scheduling

The regular nightly dispatches `multi_arch_ci_coverage_nightly.yml` once its
build stage finishes, handing over its own run id as `baseline_run_id`. Coverage
therefore has a run id of its own: a coverage failure does not colour the
nightly's status, and the nightly's test jobs — which rarely all pass — do not
hold coverage up. It waits on the build alone because that is the part producing
the artifacts coverage needs.

The dispatch is gated on the `COVERAGE_NIGHTLY_ENABLED` repository variable
being `"true"`, so a repository or fork that does not want nightly coverage
simply never sets it. Dispatching the workflow by hand and pasting in a
`baseline_run_id` from a recent nightly does the same thing, which is the
supported way to reproduce or re-run a nightly report.

Nightly coverage currently runs every onboarded project on one architecture
(gfx942, the `gfx94X-dcgpu` family). Running only what changed, and running more
architectures, come later.

#### Why there are two run ids

The nightly instruments the whole stack in one build, because building each
project separately would mean rebuilding its dependencies each time. Reporting,
though, has to stay per project: a coverage report for hipRAND should not shift
because rocRAND changed, and an instrumented rocRAND under an instrumented
hipRAND also writes its own profiles into the same run.

Each test job therefore assembles a mostly non-instrumented install:

1. `setup_test_environment` installs the **baseline** run in full — every
   library non-instrumented, as the regular nightly built it.
1. `overlay_coverage_artifacts.py` fetches the artifact holding the project
   under test from the **coverage** run and copies only that project's files
   over the baseline ones.

The baseline is read from the channel that published it (`baseline_release_type`,
normally `nightly`) rather than from the coverage run's own `ci` channel, since
artifacts are bucketed per channel.

Step 2 is per project rather than per artifact because TheRock's artifacts are
grouped: `rand` carries both rocRAND and hipRAND. Each project's files sit under
the subproject stage directory they were built in (`math-libs/hipRAND/stage`),
which is what `artifact_relpaths` in the coverage registry names. Nothing is
renamed along the way — the two runs write to different S3 directories, so the
instrumented `rand_lib_gfx942.tar.xz` and the regular one never collide.

If the overlay finds none of the project's files, the job fails rather than
testing the baseline build and reporting coverage for binaries that were never
instrumented.

### Selecting projects to run

A nightly run measures every onboarded project. Dispatching the workflow by hand
narrows that down: the `projects_to_test` input takes a comma-separated list of
project names, and also accepts three case-insensitive **group aliases** that
expand to a whole component group: `rocm_libraries_all`, `rocm_systems_all`, and
`all`. They may be mixed with explicit names, an empty input still means "every
project", and selecting a group with no onboarded projects fails the run with a
clear error rather than launching an empty matrix.

The aliases are expanded to concrete project names inside
`configure_coverage_ci.py` before the job matrix is built, so nothing downstream
ever sees them: the coverage pipeline hands `fetch_test_configurations.py`
(shared with regular CI) an already-resolved per-project `test_component`, never
an alias. The `projects_to_test` input and these aliases exist only on the
coverage workflow, so non-coverage test selection is unaffected.

### Adding a project

Add an entry to `COVERAGE_PROJECTS` in `configure_coverage_ci.py`. Before doing
so, confirm that the project implements `<PROJECT>_ENABLE_COVERAGE` in its own
CMake, and that a local instrumented build produces a non-empty report. A
project whose tests never load the instrumented library will build and test
cleanly but report zero coverage.

Set the entry's `source_repo` (`ROCM_LIBRARIES` by default, or `ROCM_SYSTEMS`);
that is what routes the project into the correct group alias and
`THEROCK_COVERAGE_*_ALL` option. No CMake edit is needed — the group membership
lists are generated from `COVERAGE_PROJECTS` at configure time, so both the local
aggregate flags and the CI aliases pick up the new project automatically.

`artifact_names` and `artifact_relpaths` are what the nightly overlay uses, so
they need to name the artifact the project ships in and the subproject stage
directory holding its files. Getting `artifact_relpaths` wrong fails the nightly
test job outright rather than quietly reporting against a non-instrumented
build.

A project whose stage is not yet built by `multi_arch_ci_coverage_nightly.yml`
also needs that stage's build job added there; today it builds compiler-runtime
and math-libs.
