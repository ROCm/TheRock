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

| Project / feature | Linux support | Windows support  |
| ----------------- | ------------- | ---------------- |
| jaxlib            | ✅ Supported  | ❌ Not supported |
| jax_rocm7_pjrt    | ✅ Supported  | ❌ Not supported |
| jax_rocm7_plugin  | ✅ Supported  | ❌ Not supported |

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

- **jax_rocm7_pjrt** (PJRT runtime for ROCm)
- **jax_rocm7_plugin** (JAX runtime plugin for ROCm)

> [!NOTE]
> jaxlib is **not built** for supported JAX versions (0.10.0+); it is installed
> from upstream PyPI (e.g. `pip install jaxlib==0.11.0`). Only
> **jax_rocm7_pjrt** and **jax_rocm7_plugin** are built.

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
   multi-arch package index before building `jax_rocm7_plugin` and
   `jax_rocm7_pjrt`.

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

1. Install requirements:

   ```bash
   cd jax
   pip install -r build/test-requirements.txt
   pip install pytest-html pytest-csv uv pytest-json-report
   ```

1. Install ROCm Python packages:

   ```bash
   pip install \
   --index-url <package_index_url> \
   "rocm[libraries,device-<gfx_arch>]==<rocm_version>"
   ```

1. Install JAX wheels from the package index:

   ```bash
   pip install \
   --index-url <package_index_url> \
   jax_rocm7_plugin \
   jax_rocm7_pjrt

   # Install jax from PyPI to match the version
   pip install jax==<JAX_VERSION>
   ```

1. Run JAX tests:

   ```bash
   cd jax
   # Create a dist directory (required to run the pytest-rocm.sh script).
   mkdir -p dist
   sh ci/run_pytest_rocm.sh
   ```

## Nightly releases

### Gating releases with JAX tests

Successful builds publish JAX wheels to the nightly multi-arch Python package
index:

<https://rocm.nightlies.amd.com/whl-multi-arch/>

The published wheels are validated by the JAX test workflow as part of the
nightly release process before being made available for use.
