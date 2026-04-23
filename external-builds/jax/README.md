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

Support for JAX is provided via stable release branches of
[ROCm/rocm-jax](https://github.com/ROCm/rocm-jax).

| JAX version | Linux                                                                                                           | Windows          |
| ----------- | --------------------------------------------------------------------------------------------------------------- | ---------------- |
| 0.9.1       | ✅ Supported via [ROCm/rocm-jax `rocm-jaxlib-v0.9.1`](https://github.com/ROCm/rocm-jax/tree/rocm-jaxlib-v0.9.1) | ❌ Not supported |
| 0.9.0       | ✅ Supported via [ROCm/rocm-jax `rocm-jaxlib-v0.9.0`](https://github.com/ROCm/rocm-jax/tree/rocm-jaxlib-v0.9.0) | ❌ Not supported |
| 0.8.2       | ✅ Supported via [ROCm/rocm-jax `rocm-jaxlib-v0.8.2`](https://github.com/ROCm/rocm-jax/tree/rocm-jaxlib-v0.8.2) | ❌ Not supported |

See also:

- Workflow source code:
  [`build_linux_jax_wheels.yml`](/.github/workflows/build_linux_jax_wheels.yml)

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
assume that a stable ROCm version is already installed on the system. When
building through TheRock, we instead build ROCm from the latest sources and
provide it via **tarballs** with arbitrary install locations.

> [!IMPORTANT]
> JAX currently requires a **tarball-based** ROCm installation. TheRock Python
> wheel packages (`rocm[libraries,devel]`) are **not yet supported** for
> building JAX. This is because the JAX build system expects a traditional
> filesystem-based ROCm installation layout (e.g. `/opt/rocm-<version>/`) that
> tarballs provide but Python wheel packages do not.

### Prerequisites

- **OS**: Linux (supported distributions with ROCm)
- **Python**: 3.12 recommended
- **Compiler**: Clang (provided via the TheRock tarball)
- **ROCm**: Provided via a TheRock tarball (see
  [RELEASES.md](/RELEASES.md#installing-from-tarballs) for tarball download
  locations and installation instructions)

### Steps

1. Checkout rocm-jax and jax:

   ```bash
   git clone https://github.com/ROCm/rocm-jax.git
   git clone https://github.com/ROCm/jax.git
   pushd rocm-jax
   git checkout rocm-jaxlib-v<JAX_VERSION>
   popd
   pushd jax
   git checkout rocm-jaxlib-v<JAX_VERSION>
   popd
   ```

1. Choose your configuration:

   - **JAX version**: e.g. `0.9.1`, `0.9.0`, or `0.8.2`
   - **Python version**: e.g. `3.12`
   - **TheRock tarball**: A tarball URL, a local tarball file path, or a
     directory containing a ROCm installation. Nightly tarballs are available
     at <https://rocm.nightlies.amd.com/tarball/>.

1. Build all wheels:

   ```bash
   pushd rocm-jax
   python3 build/ci_build \
     --compiler=clang \
     --python-versions="3.12" \
     --rocm-version="<rocm_version>" \
     --therock-path="<path_to_tarball_or_rocm_dir>" \
     --jax-source-dir="<path_to_jax_directory>" \
     dist_wheels
   popd
   ```

   > [!NOTE]
   > The `--jax-source-dir` flag is required when building jaxlib from source
   > (JAX \<= 0.9.0) and points to the cloned `jax` repository directory.
   > For JAX >= 0.9.1, jaxlib is installed from upstream PyPI, so this flag
   > can be omitted.

1. Locate built wheels:

   After a successful build, wheels will be available in:

   ```text
   rocm-jax/jax_rocm_plugin/wheelhouse/
   ```

For more detailed build options (including building `jax_rocm7_plugin` and
`jax_rocm7_pjrt` wheels individually), see
[ROCm/rocm-jax BUILDING.md](https://github.com/ROCm/rocm-jax/blob/master/BUILDING.md#building).

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
   pip install -r external-builds/jax/requirements-jax.txt
   ```

1. Install ROCm from TheRock tarball:

   ```bash
   python build_tools/install_rocm_from_artifacts.py \
     --release "<rocm_version>" \
     --artifact-group "<amdgpu_family>" \
     --output-dir "/opt/rocm-<rocm_version>"
   ```

   For detailed instructions and example usage, see the
   [TheRock RELEASES.md](https://github.com/ROCm/TheRock/blob/main/RELEASES.md#automated-tarball-extraction).

1. Install JAX wheels from the package index:

   ```bash
   # Install jaxlib, jax_rocm7_plugin, and jax_rocm7_pjrt from the GPU-family index
   pip install --index-url "https://rocm.nightlies.amd.com/v2/<amdgpu_family>/" \
     jaxlib jax_rocm7_plugin jax_rocm7_pjrt

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

We are planning to expand our test coverage and update the testing workflow.
Upcoming changes will include running smoke tests, unit tests, and multi-GPU
tests using the `pip install` packaging method for improved reliability and
consistency. Tracking issue:
[ROCm/TheRock#2592](https://github.com/ROCm/TheRock/issues/2592)

## Nightly releases

### Gating releases with JAX tests

With passing builds we upload `jaxlib`, `jax_rocm7_plugin`, and
`jax_rocm7_pjrt` wheels to subfolders of the `v2-staging` directory in the
nightly release S3 bucket with a public URL at
<https://rocm.nightlies.amd.com/v2-staging/>.

Only with passing JAX tests we promote passed wheels to the `v2` directory in
the nightly release S3 bucket with a public URL at
<https://rocm.nightlies.amd.com/v2/>.
