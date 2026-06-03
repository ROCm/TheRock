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
| `.github/workflows/test-simulator.yml`                                             | Self-contained build + simulator-test workflow (gfx94X-dcgpu only today).                                     |
| `build_tools/install_rocm_from_artifacts.py`                                       | `--rocjitsu` flag (added upstream in #5106) pulls the rocjitsu artifact into a dist for downstream test jobs. |
| `emulation/CMakeLists.txt` + `emulation/artifact-rocjitsu.toml`                    | Where the rocjitsu artifact is built and packaged in TheRock.                                                 |

## Running locally

Prerequisites: a build host capable of building TheRock and Python with PyYAML
in your venv.

```bash
# 1) Build TheRock with the simulator enabled.
cmake -B build -GNinja \
  -DTHEROCK_AMDGPU_FAMILIES=gfx94X-dcgpu \
  -DTHEROCK_ENABLE_EMULATION=ON \
  -DTHEROCK_ENABLE_ROCJITSU=ON \
  -DTHEROCK_BUILD_TESTING=ON
# Upstream replaced therock-archives with therock-artifacts in #4771: it
# produces per-component directories under build/artifacts/ (e.g.
# build/artifacts/rand_test/) instead of tarballs.
ninja -C build therock-artifacts therock-dist

# 2) Make sure the rocRAND test layout is in the dist. therock-dist should
#    already flatten the rand_test component into build/dist/rocm/bin/rocRAND
#    via artifact-flatten, but merge defensively in case it does not. We use
#    a `tar | tar` pipe (not rsync) because rsync is not always installed in
#    the manylinux build container CI uses.
for d in build/artifacts/rand_test build/artifacts/rand_test_*; do
  [ -d "$d" ] && \
    ( cd "$d" && tar --exclude=artifact_manifest.txt -cf - . ) \
    | tar -C build/dist/rocm -xf -
done

# 3) Run a real ROCm library under the simulator. Follow the project-wide
#    convention: THEROCK_BIN_DIR is `<rocm_root>/bin`; per-component drivers
#    derive `<rocm_root>` via Path(THEROCK_BIN_DIR).parent, and so does
#    simulator_runner.py when it locates librocjitsu_kmd.so + configs.
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

`test-simulator.yml` runs on:

- Pull requests that touch `compiler/amd-llvm/**`, `rocm-systems/emulation/rocjitsu/**`,
  `emulation/**`, or any of the simulator framework files in `build_tools/`.
- `workflow_dispatch` for ad-hoc runs (override `amdgpu_families` and
  `filter_preset` via the UI).

The job builds the dist with `THEROCK_ENABLE_EMULATION=ON` /
`THEROCK_ENABLE_ROCJITSU=ON` / `THEROCK_BUILD_TESTING=ON` and then runs the
`basic` rocrand preset. The test step is `continue-on-error: true` until the
pass rate stabilizes; the workflow uploads ctest logs and the rocjitsu logs as
artifacts so failures can be triaged offline.

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
