# Code Coverage

TheRock can build ROCm components with LLVM source-based coverage
instrumentation and publish an lcov report per component. This implements the
phase 1 proof of concept from
[RFC0014](https://github.com/ROCm/TheRock/pull/6967): host-side coverage on a
single default architecture, driven by a standalone nightly workflow that takes
its baseline run id by hand. hipRAND is the component it is validated against.

This page covers the concepts and how to request coverage.
[Code Coverage Flow](code_coverage_flow.md) traces a nightly run end to end,
naming the file responsible at each step.

## Why coverage is requested per project

Instrumented code writes `.profraw` counter files whenever it runs. If a
component's dependencies are instrumented too, they emit counters during the
component's own test run, and those counters land in the same report. The
practical consequences are that the report covers code the component does not
own, and that its denominator moves whenever an unrelated upstream project
changes.

This is why a single shared option name does not work. rocm-libraries projects
define `BUILD_CODE_COVERAGE` (hipRAND) or `CODE_COVERAGE` (rocRAND) today, and
setting either across the super-build would instrument every project that reads
it. Coverage is therefore requested per project, and TheRock only passes the
option to the sub-project being measured.

Downstream components are excluded for a different reason: coverage of a
component is independent of its consumers. rocFFT coverage says nothing about
hipFFT coverage, and hipFFT is already exercised by the regular CI test suites.

## Requesting coverage

The option name for a project is its logical target name in upper case, so
`hipDNN` becomes `HIPDNN_ENABLE_COVERAGE`. There are three ways to ask for it.

### Per project

Most useful for local builds of a project you already know the name of:

```bash
cmake -B build -GNinja \
  -DTHEROCK_AMDGPU_FAMILIES=gfx942 \
  -DROCFFT_ENABLE_COVERAGE=ON \
  -DHIPBLASLT_ENABLE_COVERAGE=ON
```

An explicit `OFF` also opts a single project out of the group flags below.

### By project list

CI detects changed projects and gets their names in whatever casing the source
used, so this form accepts any casing and does the conversion internally.
Separate names with commas or semicolons; `all` and `none` are also accepted:

```bash
cmake -B build -GNinja \
  -DTHEROCK_AMDGPU_FAMILIES=gfx942 \
  -DTHEROCK_COVERAGE_PROJECTS="hiprand,rocFFT,HIPBLASLT"
```

### By monorepo

For nightly runs that instrument a whole stack at once:

| Flag                                  | Instruments                                                       |
| ------------------------------------- | ----------------------------------------------------------------- |
| `THEROCK_COVERAGE_ROCM_LIBRARIES_ALL` | Every rocm-libraries component (math-libs, ml-libs, cv-libs, ...) |
| `THEROCK_COVERAGE_ROCM_SYSTEMS_ALL`   | Every rocm-systems component (base, core, profiler, ...)          |
| `THEROCK_COVERAGE_ALL`                | Both of the above                                                 |

The two monorepo flags are independent so a nightly run can instrument the
libraries without paying to instrument the entire ROCm system stack. Neither
flag touches in-tree or third-party sub-projects such as `amd-llvm` or
googletest, which contribute nothing to a component report.

A sub-project's monorepo is determined from its `EXTERNAL_SOURCE_DIR`, so a
component moving between monorepos needs no bookkeeping here.

### CMake preset

`--preset linux-release-coverage` sets up an instrumented build: it turns on
`THEROCK_COVERAGE_ROCM_LIBRARIES_ALL` and builds `RelWithDebInfo`, since
source-based coverage needs debug info to map counters back to lines. Narrow
the scope by adding a project list on top of the preset:

```bash
cmake -B build -GNinja --preset linux-release-coverage \
  -DTHEROCK_AMDGPU_FAMILIES=gfx942 \
  -DTHEROCK_COVERAGE_ROCM_LIBRARIES_ALL=OFF \
  -DTHEROCK_COVERAGE_PROJECTS=hiprand
```

The group flag has to be switched back off explicitly, or it keeps instrumenting
everything regardless of the project list.

## Nightly coverage workflow

`.github/workflows/coverage_nightly.yml` runs the pipeline end to end. It is a
separate workflow rather than an extension of the regular nightly because
instrumented binaries are not what anyone wants to ship or benchmark, and
because coverage is slow enough (instrumented rocPRIM tests take around 3 hours
against 20 minutes uninstrumented) that it needs to be able to fail or be
retried without holding up the regular nightly. A separate workflow also means
a separate run id, which keeps coverage artifacts in their own S3 namespace.

### Inputs

| Input               | Default                    | Purpose                                                         |
| ------------------- | -------------------------- | --------------------------------------------------------------- |
| `baseline_run_id`   | required                   | Regular nightly run to take non-instrumented dependencies from  |
| `amdgpu_family`     | `gfx94x`                   | Single family to build and test on                              |
| `coverage_projects` | `hiprand`                  | Projects to instrument; empty instruments all of rocm-libraries |
| `test_labels`       | `hiprand`                  | Components to run coverage tests for; empty runs all available  |
| `prebuilt_stages`   | everything but `math-libs` | Stages copied from the baseline instead of built instrumented   |

### How the hybrid artifact set is assembled

Only the projects under test should be instrumented, but the test jobs need a
complete ROCm stack to run against. Rather than unpacking a full stack and then
overwriting individual files from a grouped coverage tarball, the workflow makes
its own run id hold exactly the right mix:

1. `copy_baseline_stages` copies the baseline nightly's stages into this run's
   artifact namespace, non-instrumented and unmodified.
1. `build_instrumented_stack` builds the remaining stage (`math-libs` by
   default) with the coverage preset and pushes it to the same namespace.
1. `test_coverage` fetches everything from this run id, so it gets a stack in
   which only the projects under test are instrumented.

The test jobs therefore need no special artifact handling, and there is no
`-coverage` artifact suffix to maintain: the separate run id already isolates
these artifacts from the regular nightly's.

### Collecting and merging profraw

`test_component.yml` gains a `coverage_enabled` input. When set, each test shard
points `LLVM_PROFILE_FILE` at a per-shard, per-process path and uploads its
`.profraw` files unmerged. A `coverage_report` job then runs once per component,
after the whole shard matrix, and calls
`build_tools/github_actions/coverage_report.py` to merge them and export lcov.

Merging across shards is what makes the number meaningful. Each shard runs only
its slice of the suite, so a per-shard report would claim roughly
100/*shard count* percent coverage no matter how complete the suite is.

The report job needs no GPU. It reads profraw files and the instrumented
binaries, using `llvm-profdata` and `llvm-cov` from the same build's artifacts,
since a coverage profile is only readable by a tool at least as new as the
compiler that produced it.

Objects are discovered rather than configured:

- A component that installs `lib/lib<component>.so` is reported against that
  library, which keeps test binaries out of the numbers automatically.
- A header-only component (rocPRIM, hipCUB) has no library to point at because
  its code is compiled into the test binaries, so those binaries become the
  objects. `llvm-cov`'s `-object` flag takes one path and does not expand globs,
  so each binary is passed separately. Test sources are then filtered out with
  `--ignore-filename-regex` so that testing code does not count as covered
  product code.

Missing profraw files fail the job. Instrumentation that produced no counters,
or an `LLVM_PROFILE_FILE` pointing somewhere else, would otherwise be published
as 0% rather than reported as a problem.

Each component's report is uploaded as a `coverage-report-<component>-<family>`
workflow artifact containing `coverage.info` (lcov) and `coverage.txt`.
Forwarding those to a coverage service is deliberately left out; RFC0014 places
the choice of service and its configuration outside its scope.

## Scope

This is the RFC0014 phase 1 proof of concept and nothing beyond it. The
following are all deliberate omissions rather than oversights, and each is a
later phase in the RFC:

- **Host-side coverage only.** Device-side counters need the compiler-rt
  profile runtime compiled into instrumented device code. That is out of scope
  here; hipRAND's own code is host code, which is enough to validate the
  pipeline end to end.
- **Single architecture.** One default family, `gfx94x`/`gfx942`.
  Architecture-specific coverage needs arch-specific change detection first.
- **`baseline_run_id` is manual.** RFC0014 option B1: the run id of a
  successful regular nightly is typed in at dispatch time. Manual coordination
  is acceptable for initial validation, and it is what keeps the production
  nightly unmodified.
- **No changes to the regular nightly.** The coverage workflow is standalone
  and nothing triggers it automatically, so it can fail, be retried, or be
  abandoned without any effect on production CI.

Two implementation details will also change as components adopt the RFC:

- **Components still use the legacy option names.** No component accepts
  `<PROJECT>_ENABLE_COVERAGE` yet, so TheRock passes it alongside
  `BUILD_CODE_COVERAGE` and `CODE_COVERAGE`. Passing those is safe here because
  they only reach the configure command of the sub-project being measured, which
  is exactly the scoping a super-build-wide `-DBUILD_CODE_COVERAGE` would lose.
  They can be dropped once components take the project-specific option.
- **Hyphenated target names produce awkward option names.** `rocprofiler-sdk`
  maps to `ROCPROFILER-SDK_ENABLE_COVERAGE`, following the RFC's naming rule.
  It is harmless (an unused CMake variable) and only reachable via
  `THEROCK_COVERAGE_ROCM_SYSTEMS_ALL`, which is not part of phase 1.
