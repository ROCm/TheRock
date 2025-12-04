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
| jax_rocmX_pjrt    | ✅ Supported  | ❌ Not supported |
| jax_rocmX_plugin  | ✅ Supported  | ❌ Not supported |

### Supported JAX versions

We support building various Jax versions compatible with the latest ROCm
sources and release packages.

Support for JAX is provided via the stable `rocm-jaxlib-v0.8.0` release branch of [ROCm/rocm-jax](https://github.com/ROCm/rocm-jax). Developers can build using the `rocm-jaxlib-v0.8.0` branch to suit their requirements.

See the following table for supported version:

| JAX version | Linux                                                                                                                                   | Windows          |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| 0.8.0       | ✅ Supported<br><ul><li>[ROCm/rocm-jax `rocm-jaxlib-v0.8.0` branch](https://github.com/ROCm/rocm-jax/tree/rocm-jaxlib-v0.8.0)</li></ul> | ❌ Not supported |

## Build instructions

This repository builds the ROCm-enabled JAX artifacts:

- jaxlib (ROCm)
- jax_rocmX_pjrt (PJRT runtime for ROCm)
- jax_rocmX_plugin (JAX runtime plugin for ROCm)

We support simple flow. The path uses TheRock tarballs for ROCm install.

### Prerequisites

- **OS**: Linux (supported distributions with ROCm)
- **Python**: 3.12 recommended
- **Compiler**: clang (provided via TheRock tarball or system)

### Steps

- Checkout rocm-jax

  - `git clone https://github.com/ROCm/rocm-jax.git`
  - `cd rocm-jax`

- Choose versions and TheRock Source

  - Pick Python version
  - Pick a TheRock tarball URL, a local tarball file path or a directory containing ROCm installation
  - the TAR url paths for nightly `https://rocm.nightlies.amd.com/tarball/`

- Build all wheels using a tarball URL:

  ```bash
  python3 build/ci_build \
  --compiler=clang \
  --python-versions="3.12" \
  --rocm-version="<rocm_version>" \
  --therock-path="<tar.gz path>" \
  dist_wheels
  ```

  example for `therock-path=https://rocm.nightlies.amd.com/tarball/therock-dist-linux-gfx94X-dcgpu-7.10.0a20251120.tar.gz`
  `rocm_version=7.10.0a20251120`

> [!NOTE]
> We are planning to expand our test coverage and update the testing workflow. Upcoming changes will include running smoke tests, unit tests, and multi-GPU tests using the `pip install` packaging method for improved reliability and consistency.
