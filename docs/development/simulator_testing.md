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

| Path                                                                               | Purpose                                                                                                       |
| ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `build_tools/github_actions/test_executable_scripts/simulator_runner.py`           | Wrapper that preloads the rocjitsu interposer and delegates to the existing per-component test driver.        |
| `build_tools/github_actions/test_executable_scripts/simulator_runner_filters.yaml` | GTest allow/skip patterns per component and preset.                                                           |
| `build_tools/github_actions/fetch_test_configurations.py`                          | CI matrix; entry `rocrand-sim` plugs the runner into the standard test pipeline.                              |
| `.github/workflows/test-simulator.yml`                                             | Dual-mode simulator test workflow: (1) consume prebuilt artifacts from a Multi-Arch Build (Linux) run, or (2) build from source. gfx94X-dcgpu only today. |
| `build_tools/install_rocm_from_artifacts.py`                                       | `--rocjitsu` flag (added upstream in #5106) pulls the rocjitsu artifact into a dist for downstream test jobs. |
| `emulation/CMakeLists.txt` + `emulation/artifact-rocjitsu.toml`                    | Where the rocjitsu artifact is built and packaged in TheRock.                                                 |

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
#    Pick ONE of:
#      --run-id <CI_run_id>       (e.g. the run that produced the artifacts you want to test)
#      --release dev-tarball       (latest dev tarball release)
#      --release nightly-tarball   (latest nightly release)
python3 build_tools/install_rocm_from_artifacts.py \
    --release dev-tarball \
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
- `RJ_CONFIG=<root>/share/rocjitsu/configs/amdgpu_cdna4_kmd.json`
- `RJ_SCHEMA=<root>/share/rocjitsu/schemas/simulation_config.fbs`
- `HSA_ENABLE_SDMA=1`
- `GTEST_FILTER` from the chosen preset and the per-component skip list.

It then `execvpe`s into the existing per-component driver (e.g.
`test_rocrand.py`). The same driver is used by the real-GPU CI lane, so
behavioral differences are exclusively the result of running on the simulator.

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

### Per-test wall-clock cap

`simulator_runner.py` exports `CTEST_TEST_TIMEOUT=600` (10 minutes per ctest
case) by default. This caps any single test so a stall fails fast with a
clear ctest log instead of consuming the whole workflow step budget. Override
it for nightly `full` runs by setting `CTEST_TEST_TIMEOUT` in the environment
before invoking `simulator_runner.py`.

## Adding a new component

1. Add a driver (or reuse the existing one) under
   `build_tools/github_actions/test_executable_scripts/test_<comp>.py`.
1. Map the component in `COMPONENT_DRIVERS` inside `simulator_runner.py`.
1. Add a `<comp>:` entry with `presets:` and `skip:` to
   `simulator_runner_filters.yaml`.
1. Add a `<comp>-sim` entry in `fetch_test_configurations.py` that points at
   `simulator_runner.py --component <comp> --filter-preset basic`.
1. Verify locally with the steps above, then add the matrix entry to whatever
   workflow you want to wire it into (start by extending
   `test-simulator.yml`).

## CI

`test-simulator.yml` has two jobs guarded by `if:` conditions, both running on
the `azure-linux-scale-rocm` runner pool (CPU-only - rocjitsu has no GPU
dependency).

### Mode 1: `consume_and_test` (default, fast - ~10-15 min)

Downloads `--rocjitsu` + `--rand --tests` artifacts that were already produced
by the `compiler-runtime` and `math-libs` stages of a prior TheRock build and
runs the simulator against them. No LLVM rebuild.

Triggered by:

- **`workflow_run`** on the `Multi-Arch Build (Linux)` workflow completing.
  Consumes that exact run's artifacts via `github.event.workflow_run.id`. This
  is how the simulator gets automatic per-release-build coverage without
  paying for an LLVM rebuild.
- **`workflow_dispatch`** with `source_run_id=<id>` pins to a specific TheRock
  CI run (useful for bisecting).
- **`workflow_dispatch`** with `source_release=<tag>` (default `dev-tarball`)
  consumes a release tag's artifacts.
- **`pull_request`** when only the simulator framework files (or this
  workflow itself) change. Defaults to the latest `dev-tarball`.

### Mode 2: `build_and_test` (escape hatch, slow - ~4-5 h)

Builds TheRock from source with `THEROCK_ENABLE_EMULATION=ON` /
`THEROCK_ENABLE_ROCJITSU=ON` / `THEROCK_BUILD_TESTING=ON`, then runs the
simulator. Use this when a PR changes rocjitsu C++ source or
`compiler/amd-llvm/**` and the consumed artifacts would be stale.

Triggered only by **`workflow_dispatch`** with `build_from_source=true`.

### Common to both modes

- The `Run rocrand under rocjitsu simulator` step is
  `continue-on-error: true` until the pass rate stabilizes, so a flaky test
  does not block the workflow's success signal.
- Both modes upload `test-logs/` (ctest output + any `rocjitsu_*.log`) as a
  workflow artifact for offline triage.
- Mode 1 also includes a `Skew guard` step that logs the consumed artifact
  manifest SHAs alongside the current `HEAD` and the `rocm-systems` submodule
  SHA, so reviewers can spot when prebuilt artifacts pre-date the source on
  the PR branch.

### Phase 2 (planned follow-up): native `simulator-tests` stage in Linux::Release

Today the workflow is standalone. A future change will promote it to a real
`simulator-tests` stage inside
[`.github/workflows/multi_arch_build_portable_linux.yml`](../../.github/workflows/multi_arch_build_portable_linux.yml)
that `needs: [compiler-runtime, math-libs]` and consumes the same-run
artifacts via `artifact_manager.py fetch`. This will make simulator coverage
part of every release build automatically. Mode 1 today is the runtime-only
wiring that will be reused by that stage.

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
