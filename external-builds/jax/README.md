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

For upstream JAX development references, see:

- [ROCm/rocm-jax BUILDING.md](https://github.com/ROCm/rocm-jax/blob/master/BUILDING.md)
- [JAX developer documentation](https://docs.jax.dev/en/latest/developer.html)

## Support status

### Project and feature support status

| Project / feature | Linux support | Windows support  |
| ----------------- | ------------- | ---------------- |
| jaxlib            | ✅ Supported  | ❌ Not supported |
| jax_rocm7_pjrt    | ✅ Supported  | ❌ Not supported |
| jax_rocm7_plugin  | ✅ Supported  | ❌ Not supported |

### Supported JAX versions

Support for JAX is provided via stable release branches.

JAX 0.9.1 uses release branch from [ROCm/rocm-jax](https://github.com/ROCm/rocm-jax).
Starting with JAX 0.10.0, build support uses the [ROCm/jax](https://github.com/ROCm/jax) repository.

| JAX version | Linux                                                                                                           | Windows          |
| ----------- | --------------------------------------------------------------------------------------------------------------- | ---------------- |
| 0.10.0      | ✅ Supported via [ROCm/jax `rocm-jaxlib-v0.10.0`](https://github.com/ROCm/jax/tree/rocm-jaxlib-v0.10.0)         | ❌ Not supported |
| 0.9.1       | ✅ Supported via [ROCm/rocm-jax `rocm-jaxlib-v0.9.1`](https://github.com/ROCm/rocm-jax/tree/rocm-jaxlib-v0.9.1) | ❌ Not supported |

See also:

- Workflow source code:
  [`multi_arch_build_linux_jax_wheels.yml`](/.github/workflows/multi_arch_build_linux_jax_wheels.yml)

## Build instructions

This repository builds the following ROCm-enabled JAX artifacts:

- **jaxlib** (ROCm) - built for JAX ≤ 0.9.0 only
- **jax_rocm7_pjrt** (PJRT runtime for ROCm)
- **jax_rocm7_plugin** (JAX runtime plugin for ROCm)

> [!NOTE]
> Starting with JAX 0.9.1, jaxlib is **not built** - it is used from upstream
> PyPI (`pip install jaxlib==0.9.1`). Only **jax_rocm7_pjrt** and
> **jax_rocm7_plugin** are built.

### How building with TheRock differs from upstream

The upstream [rocm-jax build instructions](https://github.com/ROCm/rocm-jax/blob/master/BUILDING.md)
assume that a stable ROCm version is already installed on the system.

TheRock currently supports two build paths depending on the JAX release branch:

- **JAX 0.9.1** uses the legacy tarball-based build flow via
  `build/ci_build --therock-path`.
- **JAX 0.10.0** builds against ROCm Python packages installed from the
  TheRock multi-arch Python package index.

### Prerequisites

- **OS**: Linux (supported distributions with ROCm)
- **Python**: 3.12 recommended
- **Compiler**:
  - JAX 0.9.1: Clang provided by the TheRock tarball
  - JAX 0.10.0: Clang provided by the manylinux build environment
- **ROCm**:
  - JAX 0.9.1: TheRock tarball
  - JAX 0.10.0: ROCm Python packages from the TheRock multi-arch package index

### Steps

1. Checkout the source repository for your JAX version.

   **JAX 0.9.1**

   ```bash
   git clone https://github.com/ROCm/rocm-jax.git
   git clone https://github.com/ROCm/jax.git

   pushd rocm-jax
   git checkout rocm-jaxlib-v0.9.1
   popd

   pushd jax
   git checkout rocm-jaxlib-v0.9.1
   popd
   ```

   **JAX 0.10.0**

   ```bash
   git clone https://github.com/ROCm/jax.git

   pushd jax
   git checkout rocm-jaxlib-v0.10.0
   popd
   ```

1. Choose your configuration:

   - **JAX version**: e.g. `0.9.1` or `0.10.0`
   - **Python version**: e.g. `3.12`

   For **JAX 0.9.1**:

   - TheRock tarball URL, local tarball, or extracted ROCm installation.

   For **JAX 0.10.0**:

   - TheRock multi-arch Python package index.

1. Build JAX 0.9.1 (legacy tarball flow)

   ```bash
   pushd rocm-jax
      PYTHON_VERSION=<python versions, comma separated>
      ROCM_VERSION=<rocm_version>

      python3 build/ci_build --therock-path "<path_to_tarball_or_rocm_dir>"
      --python-versions="$PYTHON_VERSION"
      --rocm-version="$ROCM_VERSION"
      dist_wheels
   popd
   ```

   > [!NOTE]
   > The `--jax-source-dir` flag is required when building jaxlib from source
   > (JAX \<= 0.9.0) and points to the cloned `jax` repository directory.
   > For JAX >= 0.9.1, jaxlib is installed from upstream PyPI, so this flag
   > can be omitted.

1. Build JAX 0.10.0 (multi-arch package flow)

   JAX 0.10.0 builds are performed by the GitHub Actions workflow:

   - `.github/workflows/multi_arch_build_linux_jax_wheels.yml`

   The workflow installs ROCm Python packages from the configured TheRock
   multi-arch package index before building `jax_rocm7_plugin` and
   `jax_rocm7_pjrt`.

1. Locate built wheels:

   **JAX 0.9.1**

   After a successful build, wheels will be available in:

   ```text
   rocm-jax/jax_rocm_plugin/wheelhouse/
   ```

   **JAX 0.10.0**

   After a successful build, wheels will be available in:

   ```text
   jax/dist/
   ```

For more detailed build options, see the build instructions for the JAX
release branch you are using.

- JAX 0.9.1: [ROCm/rocm-jax BUILDING.md](https://github.com/ROCm/rocm-jax/blob/master/BUILDING.md#building)
- JAX 0.10.0: See the `ROCm/jax` repository and the
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
   pytest jax_tests/tests/multi_device_test.py -q --log-cli-level=INFO
   pytest jax_tests/tests/core_test.py -q --log-cli-level=INFO
   pytest jax_tests/tests/util_test.py -q --log-cli-level=INFO
   pytest jax_tests/tests/scipy_stats_test.py -q --log-cli-level=INFO
   ```

## Nightly releases

### Gating releases with JAX tests

Successful builds publish JAX wheels to the nightly multi-arch Python package
index:

<https://rocm.nightlies.amd.com/whl-multi-arch/>

The published wheels are validated by the JAX test workflow as part of the
nightly release process before being made available for use.
