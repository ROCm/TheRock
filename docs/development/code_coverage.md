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

That runtime calls `dlsym`, `dladdr` and `pthread_once`, and the driver does not
put `-ldl` or `-lpthread` on the link line for it. On the manylinux base those
symbols are still in `libdl` and `libpthread` rather than in `libc`, so an
instrumented shared library would be left with undefined references, and any
executable linking against it would fail `--no-allow-shlib-undefined`.
`therock_subproject.cmake` adds the two libraries whenever it enables coverage
for a subproject, so this is handled; it is worth knowing about if you
instrument a project outside the normal flow.

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

For a step-by-step walkthrough of a nightly run, from the dispatch through to
the Codecov upload, see
[Nightly Code Coverage Flow](code_coverage_flows.md). The rest of this section
is the summary.

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

`multi_arch_ci_coverage_nightly.yml` is dispatched by hand. Pick a recent
nightly run, pass its id as `baseline_run_id`, and leave `baseline_release_type`
at `nightly`. There is no cron trigger, and the regular nightly does not know
about this workflow.

Coverage therefore has a run id of its own, which is what keeps a coverage
failure from colouring the nightly's status and keeps the nightly's test jobs —
which rarely all pass — from holding coverage up.

Having the regular nightly dispatch this workflow automatically once its build
finishes, handing over its own run id, is a later change. Nothing else has to
move for it: `baseline_run_id` is already an input, so that change only adds the
dispatching job.

Nightly coverage runs the projects named by `projects_to_test` on one
architecture (gfx942, the `gfx94X-dcgpu` family). Running only what changed, and
running more architectures, come later.

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

The baseline is read from the channel that published it
(`baseline_release_type`, normally `nightly`) rather than from the coverage
run's own `ci` channel, since artifacts are bucketed per channel.

Step 2 is per project rather than per artifact because TheRock's artifacts are
grouped: `rand` carries both rocRAND and hipRAND. Each project's files sit under
the subproject stage directory they were built in (`math-libs/hipRAND/stage`),
which is what `artifact_relpaths` in the coverage registry names. Nothing is
renamed along the way — the two runs write to different S3 directories, so the
instrumented `rand_lib_gfx942.tar.xz` and the regular one never collide.

If the overlay finds none of the project's files, the job fails rather than
testing the baseline build and reporting coverage for binaries that were never
instrumented.

`object_globs` then scopes the report itself: `llvm-cov export` is handed only
that project's objects, so a selection covering several projects still yields
one report per project. The overlay and the globs solve different halves of the
same problem. The globs keep a sibling's whole functions out of the report; the
overlay keeps a sibling's instrumented inline and header code from merging into
this project's counters in the first place, which globs cannot undo because
`linkonce_odr` definitions land in every binary that uses them.

### Selecting projects to run

A nightly run measures every onboarded project the workflow can build.
Dispatching by hand narrows that down: the `projects_to_test` input takes a
comma-separated list of project names, and also accepts three case-insensitive
**group aliases** that expand to a whole component group: `rocm_libraries_all`,
`rocm_systems_all`, and `all`. They may be mixed with explicit names, and
selecting a group with no onboarded projects fails the run with a clear error
rather than launching an empty matrix.

An empty input means every project with a build job, which today is
`rocm_libraries_all`. `rccl` and `rocshmem` are onboarded but live in
`comm-libs`, which this workflow does not build, so they are excluded from the
default — and naming them, or `rocm_systems_all` or `all`, is rejected outright.
The narrowing applies only to the default; an explicit request is never silently
reduced. See [Adding a stage](#adding-a-stage).

`hipblaslt` is excluded from the default and the aliases too, for a different
reason: it is measurable, but its instrumented build currently fails to link.
Naming it is honoured with a warning rather than rejected, since rerunning it
is how the block gets noticed as fixed. See
[Blocked projects](#blocked-projects).

The aliases are expanded to concrete project names inside
`configure_coverage_ci.py` before the job matrix is built, so nothing downstream
ever sees them: the coverage pipeline hands `fetch_test_configurations.py`
(shared with regular CI) an already-resolved per-project `test_component`, never
an alias. The `projects_to_test` input and these aliases exist only on the
coverage workflow, so non-coverage test selection is unaffected.

### Adding a project

Add an entry to `COVERAGE_PROJECTS` in `configure_coverage_ci.py`.

Its `coverage_option` has to name the CMake option the project actually
implements, and that option has to select LLVM source-based instrumentation
(`-fprofile-instr-generate -fcoverage-mapping`). Neither can be assumed. The
names are not standardised — most projects use `BUILD_CODE_COVERAGE`, several
use `CODE_COVERAGE`, a few use `<PROJECT>_ENABLE_COVERAGE`, RCCL uses
`ENABLE_CODE_COVERAGE` — and some projects instrument with gcov
(`-fprofile-arcs`) instead, which writes `.gcda` files this pipeline cannot
read. `therock_subproject.cmake` translates TheRock's `<PROJECT>_ENABLE_COVERAGE`
into whichever name you register, per subproject, so a generic name does not
leak into the rest of the build.

A project that cannot be measured yet gets an `unsupported_reason` instead. It
stays in the registry so the gap is recorded, stays out of the group aliases,
and is rejected with that reason if someone names it.

### Blocked projects

A project that is measurable but whose instrumented build is broken gets a
`blocked_reason` alongside its `coverage_option`. Unlike `unsupported_reason`
this is expected to be temporary, so the project is only dropped from the
default selection and the group aliases — including the group lists
`--emit-cmake` writes, so a group build does not set a flag the Python side
just declined to set. Naming it still works and prints the reason as a warning.

`hipblaslt` is the current example. Its coverage-only branch of
`clients/CMakeLists.txt` links `hipblaslt-test` against a bare `rocroller`
while the surrounding lines use imported targets. rocRoller exports as
`roc::rocroller`, and under TheRock it is a separate subproject found through
its package config, so the bare name is not a target here and reaches the
linker as `-lrocroller` with no `-L` to resolve it. hipBLASLt's own build
avoids this by having rocRoller in-tree. The fix belongs upstream in
`ROCm/rocm-libraries`; once it lands, drop the `blocked_reason`.

Confirm a local instrumented build produces a non-empty report before relying on
a new entry. A project whose tests never load the instrumented library builds
and tests cleanly and then reports nothing.

Set the entry's `source_repo` (`ROCM_LIBRARIES` by default, or `ROCM_SYSTEMS`);
that is what routes the project into the correct group alias and
`THEROCK_COVERAGE_*_ALL` option. No CMake edit is needed — the group membership
lists are generated from `COVERAGE_PROJECTS` at configure time, so both the local
aggregate flags and the CI aliases pick up the new project automatically.

`object_globs` names the binaries the report is generated from, relative to the
install tree. For most projects that is the shared library
(`lib/librocblas.so*`); header-only projects have none, so they name their
installed test binaries instead. Getting it wrong fails the report job, which
refuses to run when no glob matches rather than publishing zero coverage.

### Adding a stage

`multi_arch_ci_coverage_nightly.yml` builds compiler-runtime and math-libs, and
`configure_coverage_ci.py` rejects any selection that reaches beyond them
(`BUILDABLE_STAGES`). That covers 17 of the 19 measurable projects.

Onboarding a project from another stage needs a build job for that stage added
to the workflow, and the stage added to `BUILDABLE_STAGES`. Order matters: a
stage consumes inbound artifacts from other stages, so anything it depends on
has to be built first. `comm-libs`, the stage `rccl` and `rocshmem` need, takes
`hipify` and `rocjitsu` from `emulation` — so it needs two jobs, emulation then
comm-libs, not one. Consult `build_topology` rather than assuming; a missing
inbound artifact does not surface until well into the build.
