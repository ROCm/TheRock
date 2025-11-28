# Build JAX with ROCm support

## Table of Contents

- [Support status](#support-status)
- [Supported JAX versions](#supported-jax-versions)
- [Build instructions](#build-instructions)
- [Build jax_rocmX_plugin and jax_rocmX_pjrt wheels instructions](#build-jax_rocmx_plugin-and-jax_rocmx_pjrt-wheels-instructions)
- [Developer Setup](#developer-setup)
- [Running/testing JAX](#runningtesting-jax)
- [Nightly releases](#nightly-releases)
- [Advanced build instructions](#advanced-build-instructions)

These build procedures are meant to run as part of ROCm CI and development flows
and thus leave less room for interpretation than in upstream repositories. Some
of this tooling is being migrated upstream as part of
[[RFC] Enable native Windows CI/CD on ROCm](https://github.com/pytorch/pytorch/issues/159520).

This incorporates advice from:

- https://github.com/pytorch/pytorch#from-source
- `.ci/manywheel/build_rocm.sh` and friends

## Support status

### Project and feature support status

| Project / feature | Linux support                                                                 | Windows support |
| ----------------- | ----------------------------------------------------------------------------- | --------------- |
| jaxlib            | ✅ Supported                                                                  | ❌ Not supported |
| jax_pjrt          | ✅ Supported                                                                  | ❌ Not supported |
| jax_plugin        | ✅ Supported                                                                  | ❌ Not supported |

### Supported JAX versions

We support building various Jax versions compatible with the latest ROCm
sources and release packages.

Support for the latest upstream JAX code is provided via the `master` branch of [ROCm/rocm-jax](https://github.com/ROCm/rocm-jax), as well as the stable `rocm-jaxlib-v0.7.1` release branch. Developers can build using either the `master` or `rocm-jaxlib-v0.7.1` branches to suit their requirements.


Each JAX version uses a combination of:

- Git repository URLs for each project
- Git "repo hashtags" (branch names, tag names, or commit refs) for each project
- Optional patches to be applied on top of a git checkout

See the following table for how each version is supported:

| JAX version   | Linux                                                                                                                                                                                                                                  | Windows           |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- |
| master        | ✅ Supported<br><ul><li>[ROCm/rocm-jax `master` branch](https://github.com/ROCm/rocm-jax/tree/master)</li></ul>                                                                                                                        | ❌ Not supported  |
| 0.7.1         | ✅ Supported<br><ul><li>[ROCm/rocm-jax `rocm-jaxlib-v0.7.1` branch](https://github.com/ROCm/rocm-jax/tree/rocm-jaxlib-v0.7.1)</li></ul>                                                                                                | ❌ Not supported  |



## Build instructions

For the most up-to-date and detailed build steps, please refer to the [Quick Build instructions in the ROCm JAX repository](https://github.com/ROCm/rocm-jax/tree/master?tab=readme-ov-file#quickbuild). Follow those steps to ensure a successful build process.

## Build jax_rocmX_plugin and jax_rocmX_pjrt wheels instructions

To build `jax_rocmX_plugin` and `jax_rocmX_pjrt` wheels, please follow the official instructions provided in the ROCm JAX repository. The most accurate and up-to-date build steps are documented at [BUILDING.md#building](https://github.com/ROCm/rocm-jax/blob/master/BUILDING.md#building). Refer to this guide for environment setup, build commands, and troubleshooting tips.


## Developer Setup

For JAX development setup and the latest instructions, please refer to the [ROCm JAX DEVSETUP guide](https://github.com/ROCm/rocm-jax/blob/master/DEVSETUP.md). If you want to run JAX locally, follow the steps outlined in that document for environment preparation and build commands.

## Running/testing JAX

After building the wheels, you should verify their functionality by running the recommended tests. Please follow the instructions provided in the [ROCm JAX BUILDING.md - Running Tests section](https://github.com/ROCm/rocm-jax/blob/master/BUILDING.md#3-running-tests) to ensure your build passes all required checks. This guide covers environment setup, test commands, and troubleshooting tips for validating your wheel builds.

## Nightly releases

### Gating releases with JAX tests

With passing builds, we upload `jaxlib`, `jax_pjrt`, and `jax_plugin` wheels to subfolders of the "v2-staging" directory in the nightly release S3 bucket with a public URL at https://rocm.nightlies.amd.com/v2-staging/

Only after passing JAX tests do we promote validated wheels to the "v2" directory in the nightly release S3 bucket with a public URL at https://rocm.nightlies.amd.com/v2/

If no runner is available: Promotion is blocked by default. Set `bypass_tests_for_releases=true` for exceptional cases under [`amdgpu_family_matrix.py`](/build_tools/github_actions/amdgpu_family_matrix.py)

## Advanced build instructions

### Other ways to install the rocm packages

The `rocm[libraries,devel]` packages can be installed in multiple ways:

- (As above) during the `build_prod_wheels.py build` subcommand

- Using the more tightly scoped `build_prod_wheels.py install-rocm` subcommand:

  ```bash
  build_prod_wheels.py
      --index-url https://rocm.nightlies.amd.com/v2/gfx110X-dgpu/ \
      install-rocm
  ```

- Manually installing from a release index:

  ```bash
  # From therock-nightly-python
  python -m pip install \
    --index-url https://rocm.nightlies.amd.com/v2/gfx110X-dgpu/ \
    rocm[libraries,devel]

  # OR from therock-dev-python
  python -m pip install \
    --index-url https://d25kgig7rdsyks.cloudfront.net/v2/gfx110X-dgpu/ \
    rocm[libraries,devel]
  ```

> [!NOTE]
> We are planning to expand our test coverage and update the testing workflow. Upcoming changes will include running smoke tests, unit tests, and multi-GPU tests using the `pip install` packaging method for improved reliability and consistency.


