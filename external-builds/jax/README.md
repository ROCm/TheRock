# Build JAX with ROCm support

## Table of Contents

- [Support status](#support-status)
- [Build instructions](#build-instructions)
- [Build jax_rocmX_plugin and jax_rocmX_pjrt wheels instructions](#build-jax_rocmx_plugin-and-jax_rocmx_pjrt-wheels-instructions)
- [Developer Setup](#developer-setup)
- [Running/testing JAX](#runningtesting-jax)
- [Nightly releases](#nightly-releases)
- [Advanced build instructions](#advanced-build-instructions)

These build procedures are meant to run as part of ROCm CI and development flows
and thus leave less room for interpretation than in upstream repositories.

## Support status

### Project and feature support status

| Project / feature | Linux support | Windows support  |
| ----------------- | ------------- | ---------------- |
| jaxlib            | ✅ Supported  | ❌ Not supported |
| jax_rocm7_pjrt    | ✅ Supported  | ❌ Not supported |
| jax_rocm7_plugin  | ✅ Supported  | ❌ Not supported |

### Supported JAX versions

We support building various Jax versions compatible with the latest ROCm
sources and release packages.

Support for JAX is provided via the stable release branch of [ROCm/rocm-jax](https://github.com/ROCm/rocm-jax). Can build using the stable release branch (for example, `rocm-jaxlib-v0.8.0`) to suit your requirements.

See the following table for supported version:

| JAX version | Linux                                                                                                                                   | Windows          |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| 0.8.0       | ✅ Supported<br><ul><li>[ROCm/rocm-jax `rocm-jaxlib-v0.8.0` branch](https://github.com/ROCm/rocm-jax/tree/rocm-jaxlib-v0.8.0)</li></ul> | ❌ Not supported |

## Build instructions

This repository builds the ROCm-enabled JAX artifacts:

- jaxlib (ROCm)
- jax_rocm7_pjrt (PJRT runtime for ROCm)
- jax_rocm7_plugin (JAX runtime plugin for ROCm)

TheRock provides a streamlined workflow for building JAX with ROCm support, leveraging TheRock tarballs for ROCm installation.

### Prerequisites

- **OS**: Linux (supported distributions with ROCm)
- **Python**: 3.12 recommended
- **Compiler**: clang-6.0 (provided via TheRock tarball or system)

### Steps

- Checkout rocm-jax

  ```bash
  git clone https://github.com/ROCm/rocm-jax.git
  cd rocm-jax
  ```

- Choose versions and TheRock Source

  - Pick Python version
  - Pick a TheRock tarball URL, a local tarball file path, or a directory containing a ROCm installation (nightly tarballs are available at `https://rocm.nightlies.amd.com/tarball/`)

- Build all wheels using a tarball URL:

  ```bash
  python3 build/ci_build \
  --compiler=clang \
  --python-versions="3.12" \
  --rocm-version="<rocm_version>" \
  --therock-path="<tar.gz path>" \
  dist_wheels
  ```

- Locate built wheels
  After successfull build, wheel will be available in:
  `jax_rocm_plugin/wheelhouse/`

## Test instructions

### Prerequisites

- AMD GPU matching the target `amdgpu_family`
- Python environment with pip
- Access to the JAX wheel package index (staging)

### Local Testing with Built Wheels Steps

After building wheels locally, you can test them:


- Checkout jax test repo

  ```bash
  git clone https://github.com/rocm/jax.git jax_tests
  cd jax_tests
  git checkout rocm-jaxlib-v<JAX_VERSION>
  ```

- Create a virtual environment

  ```bash
  cd ..
  python3 -m venv jax_test_env
  source jax_test_env/bin/activate
  python3 build_tools/setup_venv.py jax_test_env/.bin
  ```

- Install requriements

  ```bash
  pip install -r external-builds/jax/requirements-jax.txt
  ```

- Install ROCm from TheRock tarball

  ```bash
  python build_tools/install_rocm_from_artifacts.py \
    --release "<rocm_version>" \
    --artifact-group "<amdgpu_family>" \
    --output-dir "/opt/rocm-<rocm_version>"
  ```

> Example: 
> For detailed instructions and example usage, see the [official TheRock documentation](https://github.com/ROCm/TheRock/blob/main/RELEASES.md#installing-tarballs-using-install_rocm_from_artifactspy).


- Install JAX wheels

  ```bash
  # Install the built wheels
  pip install jax/jax_rocm_plugin/wheelhouse/*.whl

  # Install JAX from PyPI to match the version
  pip install --extra-index-url https://pypi.org/simple jax==<jax_version>
  ```

- Run Test jax wheels

  ```bash
  pytest jax_tests/tests/multi_device_test.py -q --log-cli-level=INFO
  pytest jax_tests/tests/core_test.py -q --log-cli-level=INFO
  pytest jax_tests/tests/util_test.py -q --log-cli-level=INFO
  pytest jax_tests/tests/scipy_stats_test.py -q --log-cli-level=INFO
  ```

> [!NOTE]
> We are planning to expand our test coverage and update the testing workflow. Upcoming changes will include running smoke tests, unit tests, and multi-GPU tests using the `pip install` packaging method for improved reliability and consistency.
> Tracking issue: [ROCm/TheRock#2592](https://github.com/ROCm/TheRock/issues/2592)
