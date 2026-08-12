# Build JAX with ROCm support

This directory provides tooling for building JAX with ROCm Python wheels.

> [!TIP]
> If you want to install our prebuilt JAX packages instead of building them
> from source, see [RELEASES.md](/RELEASES.md#installing-jax-python-packages) instead.

Table of contents:

- [Support status](#support-status)
- [Build instructions](#build-instructions)
- [Test instructions](#test-instructions)
- [Nightly releases](#nightly-releases)

For JAX development references, see:

- [jax-ml/jax](https://github.com/jax-ml/jax) - upstream JAX
- [ROCm/jax](https://github.com/ROCm/jax) - ROCm's downstream JAX fork, used to
  build the ROCm JAX wheels
- [ROCm/rocm-jax](https://github.com/ROCm/rocm-jax) - infrastructure (e.g.
  Dockerfiles), no longer used to build the JAX wheels
- [JAX developer documentation](https://docs.jax.dev/en/latest/developer.html)

## Support status

### Project and feature support status

| Project / feature        | Linux support | Windows support  |
| ------------------------ | ------------- | ---------------- |
| jaxlib                   | ✅ Supported  | ❌ Not supported |
| `jax_rocm<major>_pjrt`   | ✅ Supported  | ❌ Not supported |
| `jax_rocm<major>_plugin` | ✅ Supported  | ❌ Not supported |

> [!IMPORTANT]
> The plugin and PJRT package names embed the major version of the ROCm release
> they were built against, so the name changes across ROCm releases:
>
> | ROCm release | Package names                          |
> | ------------ | -------------------------------------- |
> | 7.x          | `jax_rocm7_plugin`, `jax_rocm7_pjrt`   |
> | 10.x         | `jax_rocm10_plugin`, `jax_rocm10_pjrt` |
>
> Wheels published from earlier ROCm releases keep their original name, so look
> for `jax_rocm7_*` when installing a historical ROCm 7.x release.
>
> This document writes `jax_rocm<major>_*` where either spelling applies.

### Supported JAX versions

Support for JAX is provided via stable release branches from
[ROCm/jax](https://github.com/ROCm/jax), built with the manylinux flow.

| JAX version | Linux                                                                                                   | Windows          |
| ----------- | ------------------------------------------------------------------------------------------------------- | ---------------- |
| 0.11.0      | ✅ Supported via [ROCm/jax `rocm-jaxlib-v0.11.0`](https://github.com/ROCm/jax/tree/rocm-jaxlib-v0.11.0) | ❌ Not supported |
| 0.10.2      | ✅ Supported via [ROCm/jax `rocm-jaxlib-v0.10.2`](https://github.com/ROCm/jax/tree/rocm-jaxlib-v0.10.2) | ❌ Not supported |
| 0.10.1      | ✅ Supported via [ROCm/jax `rocm-jaxlib-v0.10.1`](https://github.com/ROCm/jax/tree/rocm-jaxlib-v0.10.1) | ❌ Not supported |
| 0.10.0      | ✅ Supported via [ROCm/jax `rocm-jaxlib-v0.10.0`](https://github.com/ROCm/jax/tree/rocm-jaxlib-v0.10.0) | ❌ Not supported |

> [!NOTE]
> Python 3.11 is not supported for JAX 0.11.0 and later (dropped upstream).

See also:

- Workflow source code:
  [`multi_arch_build_linux_jax_wheels.yml`](/.github/workflows/multi_arch_build_linux_jax_wheels.yml)

## Build instructions

This repository builds the following ROCm-enabled JAX artifacts:

- **`jax_rocm<major>_pjrt`** (PJRT runtime for ROCm)
- **`jax_rocm<major>_plugin`** (JAX runtime plugin for ROCm)

> [!NOTE]
> jaxlib is **not built** for supported JAX versions (0.10.0+); it is installed
> from upstream PyPI (e.g. `pip install jaxlib==0.11.0`). Only
> **`jax_rocm<major>_pjrt`** and **`jax_rocm<major>_plugin`** are built.

### How building with TheRock differs from upstream

The [downstream ROCm/jax](https://github.com/ROCm/jax) build instructions
assume that a stable ROCm version is already installed on the system.

Supported JAX versions (0.10.0+) build against ROCm Python packages installed
from the TheRock multi-arch Python package index (the manylinux flow).

### Prerequisites

- **OS**: Linux (supported distributions with ROCm)
- **Python**: 3.12 recommended (Python 3.11 is not supported for JAX 0.11.0+)
- **Compiler**: Clang provided by the manylinux build environment
- **ROCm**: ROCm Python packages from the TheRock multi-arch package index

### Steps

1. Checkout the source repository for your JAX version (replace the ref with the
   version you want to build):

   ```bash
   git clone https://github.com/ROCm/jax.git

   pushd jax
   git checkout rocm-jaxlib-v0.11.0
   popd
   ```

1. Choose your configuration:

   - **JAX version**: e.g. `0.10.0`, `0.10.1`, `0.10.2`, or `0.11.0`
   - **Python version**: e.g. `3.12`
   - **Package index**: the TheRock multi-arch Python package index.

1. Build the JAX wheels (multi-arch package flow).

   From the `ROCm/jax` checkout, build the ROCm plugin and PJRT wheels:

   ```bash
   python build/build.py build --wheels=jax-rocm-plugin,jax-rocm-pjrt \
     --python_version=3.12 \
     --bazel_startup_options=--bazelrc=build/rocm/rocm.bazelrc \
     --bazel_options=--config=rocm_release_wheel \
     --bazel_options=--repo_env=ROCM_PATH=$(rocm-sdk path --root) \
     --bazel_options=--repo_env=ML_WHEEL_TYPE=release \
     --bazel_options=--//jaxlib/tools:jaxlib_git_hash=$(git rev-parse HEAD) \
     --verbose --detailed_timestamped_log --output_path=$(pwd)/dist
   ```

   This is the same flow the GitHub Actions workflow uses:

   - `.github/workflows/multi_arch_build_linux_jax_wheels.yml`

   The workflow installs ROCm Python packages from the configured TheRock
   multi-arch package index before building `jax_rocm<major>_plugin` and
   `jax_rocm<major>_pjrt`, where `<major>` is the ROCm major version being built
   against.

1. Locate built wheels.

   After a successful build, wheels will be available in:

   ```text
   jax/dist/
   ```

For more detailed build options, see the `ROCm/jax` repository and the
`.github/workflows/multi_arch_build_linux_jax_wheels.yml` workflow in TheRock.

## Test instructions

### Prerequisites

- AMD GPU matching the target `amdgpu_family`
- Python environment with pip
- Access to the JAX wheel package index

### Testing JAX wheels

1. Checkout the JAX test repo:

   ```bash
   git clone https://github.com/ROCm/jax.git jax_tests
   pushd jax_tests
   git checkout rocm-jaxlib-v<JAX_VERSION>
   popd
   ```

1. Create a virtual environment:

   ```bash
   python3 -m venv jax_test_env
   source jax_test_env/bin/activate
   ```

1. Install ROCm Python packages:

   ```bash
   pip install \
   --index-url <package_index_url> \
   "rocm[libraries,device-<gfx_arch>]==<rocm_version>"
   ```

1. Install JAX wheels from the package index:

   ```bash
   # Replace <rocm_major> with the ROCm major version of the index you install
   # from, e.g. 7 for a ROCm 7.x release or 10 for a ROCm 10.x release.
   python external-builds/jax/install_jax_wheels.py \
     --index-url <package_index_url> \
     --plugin-package jax_rocm<rocm_major>_plugin \
     --pjrt-package jax_rocm<rocm_major>_pjrt \
     --plugin-version <JAX_VERSION> \
     --pjrt-version <JAX_VERSION> \
     --jax-version <JAX_VERSION>
   ```

1. Install the test requirements, which come from the checkout so that they match
   the release under test:

   ```bash
   python external-builds/jax/install_jax_test_requirements.py \
     --jax-dir jax_tests \
     --python-version 3.12
   ```

1. Run the JAX tests:

   ```bash
   python external-builds/jax/run_jax_tests.py \
     --jax-dir jax_tests \
     --jax-version <JAX_VERSION> \
     --amdgpu-family <amdgpu_family>
   ```

### How the test runner works

[`run_jax_tests.py`](./run_jax_tests.py) is what CI runs, so the command above
reproduces a CI job locally. The suite script in the `ROCm/jax` checkout
(`ci/run_pytest_rocm.sh`) stays the source of truth for how the tests run, and
the runner adds only what is ours:

- The ROCm runtime workarounds this repository needs. Everything else about the
  environment, such as the allocator and the XLA flags, is read back out of the
  suite script so the retry pass below matches the run it checks. A version that
  changes those values, including behind a conditional, needs no change here. If
  the script's environment section cannot be read, because it was renamed, the
  job fails with an error naming what to update, rather than retrying under an
  environment that does not match the run.
- The known-bad tests for the version and GPU family under test, as a pytest
  `-k` expression. These live in [`skip_tests/`](./skip_tests/README.md), which
  also documents how to inspect the expression or run only the skipped tests.
- Two layers of retry, which are not interchangeable:
  - **In-process reruns** (`--in-process-reruns`, default 2) are
    pytest-rerunfailures. They repeat a failed test inside the same worker,
    recovering a crashed worker or a one-off runtime error.
  - **A fresh-process retry** (`--fresh-process-retry`, on by default) is a
    separate pytest run over the tests the suite reported as failed. Some
    failures are sticky: the first one leaves the process in a state where
    every later test fails the same way, which no rerun inside that process can
    clear. Only a new process tells a real failure apart from a poisoned worker,
    which is why this pass decides the result. It only decides it when the
    suite's own pytest sessions ran to the end, which their reported exit codes
    are what say: a session that stopped early leaves the suite failure
    standing, since its report lists only the tests it reached.
- The CPUs the run may use. A container shares the host's kernel, so a job
  holding 25 cores of a shared 8-GPU box still reads 192 cores and sizes every
  thread pool from that: one worker holds 1021 threads, against 16 workers at a
  time. On CI the allocation arrives in `KUBE_CPU_REQUEST`; anywhere else
  `--cpus` says the same thing. The run is pinned to that many CPUs, which
  takes a worker to 107 threads.

Useful flags while debugging:

```bash
# Print the environment and the commands without running anything.
python external-builds/jax/run_jax_tests.py --jax-dir jax_tests --dry-run

# Run only the tests the skip lists would skip.
python external-builds/jax/run_jax_tests.py --jax-dir jax_tests \
  --jax-version 0.10.2 --amdgpu-family gfx94X-dcgpu --debug

# Run the suite alone, with both retry layers off.
python external-builds/jax/run_jax_tests.py --jax-dir jax_tests --no-retries

# Hold the run to 25 cores, as a CI pod would be.
python external-builds/jax/run_jax_tests.py --jax-dir jax_tests --cpus 25
```

### Teaching an installed JAX about a new ROCm major version

The plugin wheels carry the ROCm major version in their package name
(`jax_rocm7_plugin`, `jax_rocm10_plugin`), and `jaxlib` looks that name up from a
list of majors that existed when it was released. A newly bumped major is not
found, so the GPU kernel modules resolve to `None` and `jax.devices()` fails with
`'NoneType' object has no attribute ...`.

[`patch_installed_jax_rocm_plugin_names.py`](./patch_installed_jax_rocm_plugin_names.py)
adds the name to the installed `jaxlib`, and to the PJRT shim on wheels that
predate the fix in ROCm/jax:

```bash
python external-builds/jax/patch_installed_jax_rocm_plugin_names.py \
    --plugin-package jax_rocm10_plugin
```

Run it between installing the wheels and running anything that imports `jax`. It
edits the installed files in place, is idempotent, and detects the fixes that
obsolete it, so it becomes a no-op once a `jaxlib` carrying
[jax-ml/jax#39634](https://github.com/jax-ml/jax/pull/39634) is installed. The
ROCm 7 wheels need none of this.

## Nightly releases

### Gating releases with JAX tests

Successful builds publish JAX wheels to the nightly multi-arch Python package
index:

<https://rocm.nightlies.amd.com/whl-multi-arch/>

The published wheels are validated by the JAX test workflow as part of the
nightly release process before being made available for use.
