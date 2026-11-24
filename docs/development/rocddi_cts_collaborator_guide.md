# Rocddi CTS and HIP collaborator guide

Use [ROCm/TheRock:shared/rocddi-cts-dev](https://github.com/ROCm/TheRock/tree/shared/rocddi-cts-dev)
to test a branch, tag or commit from ROCm/rocm-systems against an explicit GPU
matrix. `rocm_systems_ref` defaults to `shared/rocddi-cts-dev`. It is resolved
once to a commit shared by every matrix entry, covering runtime, CTS, CLR and
HIP-test source. Personal forks are not CI source dependencies.

Execution currently uses native Linux x86-64 build and GPU runners. Select any
GPU family/target supported by TheRock and provide matching hardware through
its runner configuration or an explicit runner override. A request to test an
architecture does not assert that rocddi already supports it. Failures are
expected results: retain them, diagnose, fix rocddi and rerun the same selection.

## Run a selected revision on a GPU matrix

An authenticated GitHub CLI and workflow-dispatch permission in ROCm/TheRock
are required. A maintainer can dispatch for collaborators with read access.
No local checkout, compiler or GPU is needed for this route.

```bash
gh auth status
gh workflow run rocddi_cts_dev.yml --repo ROCm/TheRock \
  --ref shared/rocddi-cts-dev \
  -f rocm_systems_ref=users/example/fix-loader \
  -f arch_matrix='["gfx90a","gfx950-dcgpu"]' \
  -f suites=both
```

Replace the example branch and families with the desired source and hardware.
A full commit SHA or tag also works. `--ref` selects the TheRock workflow;
`rocm_systems_ref` independently selects the ROCm/rocm-systems revision.
TheRock's committed runtime gitlink is unchanged by these disposable CI overrides.

`suites=cts` runs runtime host tests and independent CTS. `suites=hip` builds
and runs the upstream HIP suite with rocddi HSA. `suites=both` does both.
Shared-branch pushes automatically run host-only CTS against the default runtime
branch; native matrices are explicitly dispatched. No GPU is selected implicitly.

For a host-only run:

```bash
gh workflow run rocddi_cts_dev.yml --repo ROCm/TheRock \
  --ref shared/rocddi-cts-dev -f suites=cts -f arch_matrix='[]'
```

HIP execution requires a nonempty architecture matrix. The former separate
`rocddi_hip_dev.yml` vector-add workflow has been replaced by this entry point.

## Specify architectures and runners

`arch_matrix` is a JSON array. String entries name TheRock families, for example
`["gfx94X-dcgpu","gfx110X-all"]`. Each family expands through TheRock's CMake
GPU target definitions and uses the configured Linux test runner. Missing runner
configuration fails preflight instead of silently dropping the architecture.

Object entries allow an explicit target subset or custom runner:

```json
[
  {"amdgpu_family": "gfx950-dcgpu"},
  {
    "amdgpu_family": "gfx110X-all",
    "amdgpu_targets": ["gfx1100"],
    "test_runs_on": ["self-hosted", "my-gfx1100-runner"]
  }
]
```

Targets must belong to the specified family. A single target, such as `gfx908`,
can itself be the family when supplying a suitable runner. `test_runs_on` accepts
a label string or an array of required labels. Arrange matching devices and
permissions on custom runners; the workflow exposes `/dev/kfd` and `/dev/dri`
in its native test container.
Native jobs reuse TheRock's standard container settings for that family,
including device groups, GPU-isolation environment and backend-specific user
handling. Custom runners need the same prepared device access and
`/etc/podinfo/gha-gpu-isolation-settings` file (empty if isolation is unnecessary).
Each entry builds for its own selected targets,
so architecture-dependent test configuration stays specific to that entry.
Matrix failures do not cancel other entries. CTS and HIP execution are separate
jobs; a failing suite does not suppress the other completed test package.

## Select upstream HIP tests

The workload is the existing
[projects/hip-tests/catch](https://github.com/ROCm/rocm-systems/tree/shared/rocddi-cts-dev/projects/hip-tests/catch)
suite, built through TheRock's `hip-tests` target against candidate CLR/HIP.
Both runtime flags are enabled and the ROCr HSA runtime is omitted. Build and
execution checks compare the selected HSA image with the candidate provider.
The complete `hip-tests/dist` closure is packaged, including compiler/headers
needed by tests, rather than a hand-picked vector-add payload.

Inputs:

| Input                | Default       | Meaning                                                             |
| -------------------- | ------------- | ------------------------------------------------------------------- |
| `hip_test_label`     | `smoke`       | CTest label regex; `all` disables label filtering                   |
| `hip_test_filter`    | empty         | Additional CTest test-name regex                                    |
| `hip_test_level`     | `level_0`     | Upstream workload parameters, independently of selection            |
| `hip_gdb_on_failure` | `false`       | Collect supplementary GDB observations of failed executed HIP cases |
| `baseline_run_id`    | `43537534946` | Explicit prebuilt LLVM/base/sysdeps dependency run                  |

For a focused subset of existing cases:

```bash
gh workflow run rocddi_cts_dev.yml --repo ROCm/TheRock \
  --ref shared/rocddi-cts-dev \
  -f rocm_systems_ref=users/example/fix-loader \
  -f arch_matrix='["gfx950-dcgpu"]' -f suites=hip \
  -f hip_test_label=all \
  -f 'hip_test_filter=Unit_hip(Malloc|Memcpy|StreamCreate|EventCreate)' \
  -f hip_test_level=level_0
```

Use the explicit `all` value for an unfiltered workflow run: an empty workflow
input can restore its default. With the runner CLI directly, `--label-regex ''`
disables label filtering.

An empty selection or a selection with every case skipped fails. CTest continues through selected cases after a failure,
returns a failing status and retains JUnit/CTest diagnostics. The existing HIP
runner's architecture exclusions and upstream disabled tests remain in effect;
inspect the recorded inventory when assessing coverage. No rocddi-specific
failure exclusions are added. `level_0` changes workload parameters; it does
not by itself select fewer tests. Build failures and runtime failures remain
separately visible.

For supplementary crash diagnostics, dispatch with `-f hip_gdb_on_failure=true`
or add `--gdb-on-failure` to the runner CLI with `--junit`. GDB is installed only
for the opted-in HIP job. The original CTest exit status, JUnit and diagnostics
remain authoritative. `hip-tests-gdb/report.json` and per-case logs appear in
the HIP diagnostic artifact. Only failed, executed cases from that original
JUnit report are eligible; passing, skipped and disabled cases are not rerun.
Each eligible case runs at most once under batch GDB, within eight-case,
120-second-per-case, 600-second debugger budget and 2 MiB-per-log limits. The
CTest JSON inventory has a separate 30-second bound. The report records
timeouts, exhausted budgets and unavailable observations.

These are observations of the top-level CTest command, with its recorded
arguments, working directory and environment properties. Direct Linux ELF
executables inside the installed `catch_tests` tree are supported; scripts,
fixtures, resource allocation, affinity and other unsupported CTest properties
are reported as unavailable. CTest JSON flattens launcher commands, so these
observations do not qualify launcher or descendant-process behavior. GDB changes
timing; a rerun cannot establish the original failure's cause or reproducibility.
Logs request the first SIGSEGV stop's thread stacks, shared-library mappings,
registers and program-counter instruction even when binaries are stripped.

## CTS coverage and execution subjects

| Suite                       | Subject                                                                 | Native requirements      |
| --------------------------- | ----------------------------------------------------------------------- | ------------------------ |
| Upstream runtime host tests | Rust tests, build helpers and C smoke/layout tests                      | Build host; no GPU       |
| `amdf-host`                 | Public AMDF negotiation/lifetime contracts loading the selected runtime | No GPU                   |
| `ddi-host`                  | Host allocation/lifetime contracts in rocddi linked into its executable | No runtime prefix or GPU |
| `kfd`                       | Real version-query ioctl and repeat-query consistency                   | Usable `/dev/kfd`        |

With `suites=cts` or `suites=both`, host CTS runs in the build environment and
every requested native profile also runs the three installed CTS suites.
KFD version CTS does not submit GPU work;
these initial cases do not establish full runtime or driver conformance. There
is no independent HSA CTS family yet. The
[candidate contract catalog](https://gist.github.com/zjgarvey/d2194330b6f252bbd177314774b070cc)
is a source of cases to implement, not completed coverage.

## Inspect runs and retain evidence

```bash
gh run list --repo ROCm/TheRock --workflow rocddi_cts_dev.yml \
  --branch shared/rocddi-cts-dev \
  --json databaseId,displayTitle,createdAt,event,headSha,status,conclusion,url
read -r -p 'Workflow run ID: ' RUN_ID
gh run watch "$RUN_ID" --repo ROCm/TheRock --exit-status
gh run view "$RUN_ID" --repo ROCm/TheRock --log-failed
gh run download "$RUN_ID" --repo ROCm/TheRock \
  --pattern 'rocddi-*-diagnostics-*' --dir run-diagnostics
```

Run titles show the requested runtime ref, suites and first matrix family, with
`+ more` for additional entries or `host` for an empty matrix. `headSha` identifies
TheRock, so before/after runtime runs can share it. For each downloaded build
diagnostic artifact, inspect `SOURCE.json`'s `rocm_systems_commit` for the resolved
immutable runtime SHA and `rocm_systems_ref` for the requested ref. For example:

```bash
cat run-diagnostics/rocddi-build-diagnostics-0-gfx950-dcgpu/SOURCE.json
```

The same manifest is included in completed CTS and HIP packages. The configure
job's `Runtime source and matrix` summary records the resolved SHA and full matrix
before runtime builds begin, including when a later build fails. Its `Resolve runtime revision and architecture matrix` log also records the resolved SHA.

Packages and diagnostics are named by matrix row index and family, for example
`rocddi-cts-0-gfx950-dcgpu` and `rocddi-hip-tests-0-gfx950-dcgpu`. Artifacts are
retained for seven days. `SOURCE.json` records TheRock's commit, the requested
runtime ref and resolved SHA, dependency baseline, selected targets and HIP
filters/level. Runtime branches need no gitlink update to test a new commit:
rerun with the new ref/SHA. Normal TheRock builds outside this workflow still
follow the committed gitlink.

## Replay completed packages locally

Choose a run with a successful package upload; native test failure or queueing
does not invalidate its completed package. Use a compatible Linux x86-64 host.
CTS needs CMake/CTest 3.25+; HIP replay also uses Python 3 and the checked-out
TheRock runner script. Native execution needs the devices/permissions of the
selected profile. No system ROCm installation is needed.

```bash
read -r -p 'CTS artifact name: ' CTS_ARTIFACT
gh run download "$RUN_ID" --repo ROCm/TheRock \
  --name "$CTS_ARTIFACT" --dir cts-replay
cd cts-replay
tar -xzf rocddi-cts.tar.gz
cat SOURCE.json
export ROCDDI_RESULTS="$(mktemp -d "$PWD/results.XXXXXX")"
ctest -S cts-prefix/cmake/RunTests.cmake -V \
  -DTEST_SUITE:STRING=amdf-host -DRUNTIME_PREFIX:PATH="$PWD/cts-runtime" \
  -DRESULTS_DIRECTORY:PATH="$ROCDDI_RESULTS/amdf-host"
ctest -S cts-prefix/cmake/RunTests.cmake -V \
  -DTEST_SUITE:STRING=ddi-host \
  -DRESULTS_DIRECTORY:PATH="$ROCDDI_RESULTS/ddi-host"
ctest -S cts-prefix/cmake/RunTests.cmake -V \
  -DTEST_SUITE:STRING=kfd \
  -DRESULTS_DIRECTORY:PATH="$ROCDDI_RESULTS/kfd"
```

Run suites separately and use fresh absolute result directories each time.
AMDF consumes the included runtime prefix. DDI's implementation is linked into
its binary; changing runtime libraries does not replace it. Keep `run.txt`,
`build.txt`, JUnit and `Testing/` records.

For HIP, from a checkout of the matching TheRock branch:

```bash
read -r -p 'HIP artifact name: ' HIP_ARTIFACT
gh run download "$RUN_ID" --repo ROCm/TheRock \
  --name "$HIP_ARTIFACT" --dir hip-replay
mkdir hip-replay/candidate hip-replay/diagnostics
tar -xzf hip-replay/rocddi-hip-tests.tar.gz -C hip-replay/candidate
cat hip-replay/candidate/SOURCE.json
(cd hip-replay/candidate && sha256sum -c HSA.sha256)
export AMDGPU_FAMILIES="$(python3 -c 'import json; print(json.load(open("hip-replay/candidate/SOURCE.json"))["amdgpu_family"])')"
export PATH="$PWD/hip-replay/candidate/bin:$PWD/hip-replay/candidate/lib/llvm/bin:$PATH"
THEROCK_BIN_DIR="$PWD/hip-replay/candidate/bin" AMD_LOG_LEVEL=1 \
  HIP_TEST_LEVEL=level_0 \
  python3 build_tools/github_actions/test_executable_scripts/test_hiptests.py \
    --label-regex smoke --junit "$PWD/hip-replay/diagnostics/hip-tests.xml"
```

Use a clean library environment and the same filters, level and compatible
hardware as the original profile. Preserve failures and logs.

## Local source development

```bash
git clone --branch shared/rocddi-cts-dev https://github.com/ROCm/TheRock.git
cd TheRock
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python build_tools/fetch_sources.py --stage runtime-tests --jobs 12 --depth 1
```

Provision TheRock's Linux build dependencies and the selected runtime's
`rust-toolchain.toml`; explicitly fetch its locked Cargo dependencies. Source
fetching resets submodules and reapplies patches: preserve edits before running
it again. Ordinary local fetching uses the committed runtime pin. For another
revision, use the existing
[EXTERNAL_REPO_CONFIG source override](https://github.com/ROCm/TheRock/blob/shared/rocddi-cts-dev/build_tools/fetch_sources.py)
with a full SHA from ROCm/rocm-systems, as this workflow does.

The build recipe is
[rocddi_dev_build.sh](https://github.com/ROCm/TheRock/blob/shared/rocddi-cts-dev/build_tools/github_actions/rocddi_dev_build.sh)
and its environment/step order is in
[rocddi_dev_test.yml](https://github.com/ROCm/TheRock/blob/shared/rocddi-cts-dev/.github/workflows/rocddi_dev_test.yml).
The helpers use assembled TheRock LLVM and explicit selected targets. The
[runtime component instructions](https://github.com/ROCm/TheRock/blob/shared/rocddi-cts-dev/core/rocm-runtimes/README.md)
and [independent CTS instructions](https://github.com/ROCm/rocm-systems/blob/shared/rocddi-cts-dev/runtimes/testing/README.md)
cover direct build/test commands.

Add CTS cases under `rocm-systems/runtimes/testing/` and rebuild/reinstall tests
when their linked implementation changes. Follow the repository's
[contribution branch policy](../../CONTRIBUTING.md#branch-creation-policy) when
choosing where to host focused PRs. This development workflow selects runtime
sources from ROCm/rocm-systems; shared integration branches remain in ROCm-owned
repositories. Include source identities, matrix, filters, kernel/GPU identity,
exit statuses and diagnostics when reporting a failure.
