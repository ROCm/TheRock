# Simulator Testing (rocjitsu)

This page covers TheRock's CPU-only path for running real ROCm applications
against an emulated AMD GPU via the
[rocjitsu](https://github.com/ROCm/rocm-systems/tree/develop/emulation/rocjitsu)
simulator.

## When to use it

- You want to exercise a ROCm library end-to-end without booting a runner that
  has a physical AMD GPU.
- You need a regression signal for the rocjitsu interposer itself or for an
  upstream LLVM/HIP change that would only show up under emulation.

The simulator is much slower than real hardware (PDES on CPU) so the framework
intentionally runs a narrow filter of fast tests by default. Heavier presets
exist for nightly use.

## What lives where

| Path                                                                               | Purpose                                                                                                                                                                                                                                                      |
| ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `build_tools/github_actions/test_executable_scripts/simulator_runner.py`           | Wrapper that preloads the rocjitsu interposer and delegates to the existing per-component test driver.                                                                                                                                                       |
| `build_tools/github_actions/test_executable_scripts/simulator_runner_filters.yaml` | GTest allow/skip patterns per component and preset.                                                                                                                                                                                                          |
| `build_tools/github_actions/fetch_test_configurations.py`                          | CI matrix; entry `rocrand-sim` plugs the runner into the standard test pipeline.                                                                                                                                                                             |
| `.github/workflows/test-simulator.yml`                                             | Dual-mode simulator test workflow: (1) consume prebuilt artifacts via `workflow_call` (Release pipeline) or `workflow_dispatch source_run_id` (fast loop), or (2) build from source via `workflow_dispatch build_from_source=true`. gfx94X-dcgpu only today. |
| `build_tools/install_rocm_from_artifacts.py`                                       | `--rocjitsu` flag (added upstream in #5106) pulls the rocjitsu artifact into a dist for downstream test jobs.                                                                                                                                                |
| `emulation/CMakeLists.txt` + `emulation/artifact-rocjitsu.toml`                    | Where the rocjitsu artifact is built and packaged in TheRock.                                                                                                                                                                                                |

## Running locally

You have two paths: **(A) consume prebuilt artifacts** (fast, recommended) or
**(B) build from source** (slow, needed when iterating on rocjitsu C++ source
or compiler internals locally).

### A. Consume prebuilt artifacts (~10-15 min, no LLVM build)

The simulator only needs rocjitsu + rocRAND test binaries + the base ROCm
runtime layer. All of these are already produced by TheRock's
`compiler-runtime` and `math-libs` stages on every multi-arch build run, and
they are published to S3 by `artifact_manager.py push`. The
[install_rocm_from_artifacts.py](../../build_tools/install_rocm_from_artifacts.py)
helper assembles them into a working dist with a single command.

```bash
# 1) Pull rocjitsu + rocRAND tests + base runtime layer into build/dist/rocm/.
#    Only `--run-id <CI_run_id>` fetches per-component artifacts that include
#    rocjitsu; `--release` / `--latest-release` ship the flattened dev tarball
#    layout which does NOT carry the rocjitsu component, so they cannot drive
#    this workflow. Pick a recent green TheRock CI run that built the families
#    you want (gfx94X-dcgpu today).
python3 build_tools/install_rocm_from_artifacts.py \
    --run-id <CI_run_id> \
    --amdgpu-family gfx94X-dcgpu \
    --output-dir build/dist/rocm \
    --rocjitsu --rand --tests

# 2) Run the simulator. Same convention as every other per-component driver:
#    THEROCK_BIN_DIR is `<rocm_root>/bin`; simulator_runner.py derives
#    <rocm_root> via Path(THEROCK_BIN_DIR).parent to find librocjitsu_kmd.so
#    and the rocjitsu config / schema files.
export THEROCK_BIN_DIR=$PWD/build/dist/rocm/bin
export AMDGPU_FAMILIES=gfx94X-dcgpu
export TEST_COMPONENT=rocrand
export TEST_TYPE=full
python3 build_tools/github_actions/test_executable_scripts/simulator_runner.py \
    --component rocrand --filter-preset basic
```

Skew note: if you have local rocjitsu source changes you want to validate, the
prebuilt artifacts will NOT contain them - the simulator runs against whatever
was built upstream. Use path B in that case.

### B. Build from source (~4-5 h, full LLVM rebuild)

Use this when iterating on rocjitsu C++ source, compiler/amd-llvm patches, or
anything that would not yet be in a published artifact.

```bash
# 1) Build TheRock with the simulator enabled.
cmake -B build -GNinja \
  -DTHEROCK_AMDGPU_FAMILIES=gfx94X-dcgpu \
  -DTHEROCK_ENABLE_EMULATION=ON \
  -DTHEROCK_ENABLE_ROCJITSU=ON \
  -DTHEROCK_BUILD_TESTING=ON
ninja -C build therock-artifacts therock-dist

# 2) Safety net: therock-dist *should* flatten the rand_test component into
#    build/dist/rocm/bin/rocRAND/, but if not, merge it in defensively. We use
#    a `tar | tar` pipe (not rsync) because rsync is not always installed in
#    the manylinux build container CI uses.
for d in build/artifacts/rand_test build/artifacts/rand_test_*; do
  [ -d "$d" ] && \
    ( cd "$d" && tar --exclude=artifact_manifest.txt -cf - . ) \
    | tar -C build/dist/rocm -xf -
done

# 3) Run the simulator (same as path A step 2).
export THEROCK_BIN_DIR=$PWD/build/dist/rocm/bin
export AMDGPU_FAMILIES=gfx94X-dcgpu
export TEST_COMPONENT=rocrand
export TEST_TYPE=full
python3 build_tools/github_actions/test_executable_scripts/simulator_runner.py \
    --component rocrand --filter-preset basic
```

The wrapper composes (with `<root> = Path(THEROCK_BIN_DIR).parent`):

- `LD_PRELOAD=<root>/lib/librocjitsu_kmd.so`
- `RJ_CONFIG=<root>/share/rocjitsu/configs/amdgpu_<cdna3|cdna4>_kmd.json` (picked from `AMDGPU_FAMILIES`)
- `RJ_SCHEMA=<root>/share/rocjitsu/schemas/simulation_config.fbs`
- `HSA_ENABLE_SDMA=1`
- `GTEST_FILTER` from the chosen preset and the per-component skip list.

It then `subprocess.run`s the existing per-component driver (e.g.
`test_rocrand.py`). The same driver is used by the real-GPU CI lane, so
behavioral differences are exclusively the result of running on the simulator.
After the driver returns, the wrapper runs the post-run guards (see below)
and returns a non-zero exit code if the driver failed *or* a guard tripped.

## Filter presets

`simulator_runner_filters.yaml` keeps three presets per component:

- `basic` - small allow-list intended for PR-time signal (~minutes). For
  rocrand today this is two CPU-only tests (`rocrand_get_version_test`,
  `rocrand_generator_test`) plus exactly one parameterized create/destroy
  cycle for `PHILOX4_32_10` (the lightest RNG). When picking what goes here,
  favor tests that exercise the simulator's GPU init path but avoid
  parameterized sweeps - one rocrand `*basic_tests*` glob, for example,
  expands to 22+ cases that each take minutes under PDES.
- `quick` - the existing `QUICK_TESTS` set from the component's test driver,
  expected to take 30-90 min under emulation.
- `full` - the entire ctest set; nightly only.

Each component has a `skip` list of gtest patterns. Every entry should carry a
short comment explaining why the test cannot pass under the simulator yet.

### Preset schema

Each preset is a mapping with three fields (a flat list of patterns is also
accepted as a back-compat shorthand for `allow:` only):

| Field         | Type        | Default | Purpose                                                                                                                                                                                          |
| ------------- | ----------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `allow`       | list[str]   | -       | gtest filter globs joined with `:` to build the positive half of `GTEST_FILTER`.                                                                                                                 |
| `ctest_regex` | str (regex) | `.*`    | `ctest -R <regex>` narrowing: only ctest binaries whose name matches are run. The same regex defines which binaries count as "in scope" for the post-run guards.                                 |
| `min_gtests`  | int         | `1`     | Minimum total number of gtest cases the in-scope binaries must actually execute. The post-run guard fails the workflow step if the observed total is lower (catches over-narrow filter strings). |

In addition, each component declares a top-level `ctest_dir:` (e.g.
`rocRAND`) pointing to the directory under `THEROCK_BIN_DIR` that holds
`CTestTestfile.cmake`. The wrapper uses it to locate
`Testing/Temporary/LastTest.log` for the post-run guards.

### Post-run guards

`simulator_runner.py` parses `<ctest_dir>/Testing/Temporary/LastTest.log`
after the driver returns and enforces two guards against silent green runs:

1. **No empty-set passes.** GoogleTest exits 0 (and ctest reports `Passed`)
   when `GTEST_FILTER` matches no tests inside a binary; it prints
   `WARNING: filter "..." did not match any test; no tests were run`. We
   grep `LastTest.log` for that sentinel and fail the step if **any** binary
   matching `ctest_regex` produced it. Without this, a typo or stale filter
   entry would produce a green CI run with zero simulator coverage (the
   exact failure mode `Rocjitsu_005` documented).
1. **Minimum coverage floor.** We sum the gtest counts reported by every
   in-scope binary (`[==========] N tests from M test suites ran.`) and fail
   if the total is below the preset's `min_gtests`.

Both guards run after the driver finishes, regardless of whether the driver
succeeded. A driver failure always wins (its exit code is propagated as-is);
a guard failure produces a non-zero exit only when the driver succeeded.
Bypass the guards with `SIMULATOR_RUNNER_SKIP_GUARDS=1` if you are bisecting
the wrapper itself.

### Driver-side knobs (`SIMULATOR_*` env vars)

`simulator_runner.py` exports the following env vars consumed by the wrapped
per-component driver. All are no-ops on the real-GPU CI lane where the vars
are unset, so adding them does not change on-device behavior.

| Env var                         | Set by       | Effect in `test_rocrand.py`                                                                      |
| ------------------------------- | ------------ | ------------------------------------------------------------------------------------------------ |
| `SIMULATOR_CTEST_DIR`           | wrapper      | Overrides the historical `<THEROCK_BIN_DIR>/rocRAND` ctest test-dir.                             |
| `SIMULATOR_CTEST_INCLUDE_REGEX` | wrapper      | Adds `ctest -R <regex>` when non-empty and not `.*`.                                             |
| `SIMULATOR_NO_RETRY`            | wrapper (=1) | Drops `ctest --repeat until-pass:3`. Retries on the deterministic simulator only hide real bugs. |

### Per-test wall-clock cap

### Per-test wall-clock cap

`simulator_runner.py` exports `CTEST_TEST_TIMEOUT=600` (10 minutes per ctest
case) by default. This caps any single test so a stall fails fast with a
clear ctest log instead of consuming the whole workflow step budget. Override
it for nightly `full` runs by setting `CTEST_TEST_TIMEOUT` in the environment
before invoking `simulator_runner.py`.

## Adding a new component

1. Add a driver (or reuse the existing one) under
   `build_tools/github_actions/test_executable_scripts/test_<comp>.py`. Have it
   honor `SIMULATOR_CTEST_DIR`, `SIMULATOR_CTEST_INCLUDE_REGEX`, and
   `SIMULATOR_NO_RETRY` if you want the post-run guards to be effective.
1. Map the component in `COMPONENT_DRIVERS` inside `simulator_runner.py`.
1. Add a `<comp>:` entry to `simulator_runner_filters.yaml` with:
   - `ctest_dir:` pointing at the directory under `THEROCK_BIN_DIR` that
     holds `CTestTestfile.cmake` for this component.
   - `presets:` with `basic`/`quick`/`full` entries (each using the
     `allow:` + `ctest_regex:` + `min_gtests:` schema above).
   - `skip:` for the component-wide deny list.
1. Add a `<comp>-sim` entry in `fetch_test_configurations.py` that points at
   `simulator_runner.py --component <comp> --filter-preset basic`.
1. Verify locally with the steps above. Confirm the post-run guard summary
   line (`[simulator_runner] guards: in_scope_binaries=... gtests_ran=...`)
   reports the gtest count you expected for the preset, then add the matrix
   entry to whatever workflow you want to wire it into (start by extending
   `test-simulator.yml`).

## CI

`test-simulator.yml` has two jobs guarded by `if:` conditions, both running on
the `azure-linux-scale-rocm` runner pool (CPU-only - rocjitsu has no GPU
dependency).

### Mode 1: `consume_and_test` (default, fast - ~10-15 min)

Downloads `--rocjitsu` + `--rand --tests` artifacts that were already produced
by the `compiler-runtime` and `math-libs` stages of the calling TheRock build
and runs the simulator against them. No LLVM rebuild.

Triggered by:

- **`workflow_call`** from the `simulator-test` sibling job inside
  [`.github/workflows/multi_arch_build_portable_linux.yml`](../../.github/workflows/multi_arch_build_portable_linux.yml).
  That sibling has `needs: [compiler-runtime, math-libs]`, so the simulator
  starts as soon as those two stages finish - it does **not** wait for the
  rest of the build DAG. The caller passes its own `github.run_id` as
  `artifact_run_id`, so the simulator always tests the artifacts from the
  exact run that triggered it. The sibling is gated by a new
  `run_simulator` input that defaults to `false`; only the Linux Release
  pipeline ([`multi_arch_release_linux.yml`](../../.github/workflows/multi_arch_release_linux.yml))
  opts in today. PR CI (`multi_arch_ci_linux.yml`, `multi_arch_ci_asan.yml`)
  is unchanged.
- **`workflow_dispatch`** with `source_run_id=<id>` pins to a specific
  TheRock CI run (useful for bisecting or for the fast-iteration loop
  below).

Removed (do not look for them): `workflow_run` on `Multi-Arch Build (Linux)`,
`pull_request`, and `source_release` / `--latest-release`. The
`install_rocm_from_artifacts.py --release` / `--latest-release` code paths
ship the flattened dev tarball layout, which does not carry the rocjitsu
component, so they cannot install a working simulator. Only `--run-id` mode
fetches the per-component tree the simulator needs.

### Fast iteration loop (manual dispatch)

For rapid iteration on `simulator_runner.py`, `simulator_runner_filters.yaml`,
or `test_rocrand.py` without paying for a full release build:

1. Find a recent green TheRock CI run id for the family you want (e.g.
   gfx94X-dcgpu) via the Actions tab.
1. Push your wrapper / filter changes to a branch.
1. Trigger the `Test Simulator` workflow manually via the Actions tab on that
   branch, passing `source_run_id=<the_id_from_step_1>`. The workflow will
   download the already-published artifacts from that run and exercise them
   with the simulator framework code from your branch.

This bypasses the entire build DAG, so each iteration is just the ~10-15 min
of the consume-artifacts path. Use Mode 2 (build_from_source) instead when
the change is in rocjitsu C++ source or `compiler/amd-llvm/**` and you need
the rebuilt artifact.

### Mode 2: `build_and_test` (escape hatch, slow - ~4-5 h)

Builds TheRock from source with `THEROCK_ENABLE_EMULATION=ON` /
`THEROCK_ENABLE_ROCJITSU=ON` / `THEROCK_BUILD_TESTING=ON`, then runs the
simulator. Use this when a PR changes rocjitsu C++ source or
`compiler/amd-llvm/**` and the consumed artifacts would be stale.

Triggered only by **`workflow_dispatch`** with `build_from_source=true`.

### Common to both modes

- The `Run rocrand under rocjitsu simulator` step is
  `continue-on-error: true` until the pass rate stabilizes, so a flaky test
  does not block the workflow's success signal. The post-run guards in
  `simulator_runner.py` (see [Post-run guards](#post-run-guards) above)
  still cause that step's outcome to flip to `failure` when empty-set silent
  passes occur, so the `Report` step's `Simulator test result:` line
  reflects guard failures even while `continue-on-error` keeps the job
  green.
- Both modes upload `test-logs/` (ctest output + any `rocjitsu_*.log`) as a
  workflow artifact for offline triage.
- Mode 1 also includes a `Skew guard` step that logs the consumed artifact
  manifest SHAs alongside the current `HEAD` and the `rocm-systems` submodule
  SHA, so reviewers can spot when prebuilt artifacts pre-date the source on
  the PR branch.

## Bring-up ladder

Roll the framework forward conservatively:

1. PR-1: `basic` preset stays as the only thing in CI, soft-fail.
1. PR-2: bump `rocrand-sim` to `quick` once three consecutive PR-1 runs are
   green, grow the skip list as needed, then promote to required.
1. PR-3: add `hiprand-sim` reusing the same wrapper (`--component hiprand`).

## Known limitations

- Only `gfx94X-dcgpu` is supported today - rocjitsu ships CDNA3/CDNA4
  topologies (gfx942/gfx950). Other architectures will land as upstream JSON
  configs become available.
- The KFD interposer is Linux-only, so the workflow does not run on Windows.
- Performance benchmarking under the simulator is not meaningful - the
  framework is functional-only.
