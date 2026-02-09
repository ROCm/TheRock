# Build JAX with ROCm support

This directory provides tooling for building JAX with ROCm Python wheels.

> [!TIP]
> If you want to install our prebuilt JAX packages instead of building them
> from source, see [RELEASES.md](/RELEASES.md) instead.

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

| JAX version | Linux                                                                                                          | Windows          |
| ----------- | -------------------------------------------------------------------------------------------------------------- | ---------------- |
| 0.8.0       | ✅ Supported via [ROCm/rocm-jax `rocm-jaxlib-v0.8.0`](https://github.com/ROCm/rocm-jax/tree/rocm-jaxlib-v0.8.0) | ❌ Not supported |

See also:

- Workflow source code:
  [`build_linux_jax_wheels.yml`](/.github/workflows/build_linux_jax_wheels.yml)

## Build instructions

This repository builds the following ROCm-enabled JAX artifacts:

- **jaxlib** (ROCm)
- **jax_rocm7_pjrt** (PJRT runtime for ROCm)
- **jax_rocm7_plugin** (JAX runtime plugin for ROCm)

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

1. Checkout rocm-jax:

   ```bash
   git clone https://github.com/ROCm/rocm-jax.git
   pushd rocm-jax
   ```

2. Choose your configuration:

   - **Python version**: e.g. `3.12`
   - **TheRock tarball**: A tarball URL, a local tarball file path, or a
     directory containing a ROCm installation. Nightly tarballs are available
     at <https://rocm.nightlies.amd.com/tarball/>.

3. Build all wheels:

   ```bash
   python3 build/ci_build \
     --compiler=clang \
     --python-versions="3.12" \
     --rocm-version="<rocm_version>" \
     --therock-path="<path_to_tarball_or_rocm_dir>" \
     dist_wheels
   ```

4. Locate built wheels:

   After a successful build, wheels will be available in:

   ```text
   jax_rocm_plugin/wheelhouse/
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
   git clone https://github.com/rocm/jax.git jax_tests
   pushd jax_tests
   git checkout rocm-jaxlib-v<JAX_VERSION>
   popd
   ```

2. Create a virtual environment:

   ```bash
   python3 -m venv jax_test_env
   source jax_test_env/bin/activate
   ```

3. Install requirements:

   ```bash
   pip install -r external-builds/jax/requirements-jax.txt
   ```

4. Install ROCm from TheRock tarball:

   ```bash
   python build_tools/install_rocm_from_artifacts.py \
     --release "<rocm_version>" \
     --artifact-group "<amdgpu_family>" \
     --output-dir "/opt/rocm-<rocm_version>"
   ```

   For detailed instructions and example usage, see the
   [TheRock RELEASES.md](https://github.com/ROCm/TheRock/blob/main/RELEASES.md#installing-tarballs-using-install_rocm_from_artifactspy).

5. Install JAX wheels from the package index:

   ```bash
   # Install jaxlib, jax_rocm7_plugin, and jax_rocm7_pjrt from the GPU-family index
   pip install --index-url "https://rocm.nightlies.amd.com/v2/<amdgpu_family>/" \
     jaxlib jax_rocm7_plugin jax_rocm7_pjrt

   # Install jax from PyPI to match the version
   pip install jax==<jax_version>
   ```

6. Run JAX tests:

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
