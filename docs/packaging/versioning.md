# TheRock package versioning

We build and distribute packages for a variety of projects across multiple
packaging systems, release channels, and operating systems.

This document describes the version schemes we use for those packages.

Table of contents:

- [Overview](#overview)
  - [Constraints and design guidelines](#constraints-and-design-guidelines)
  - [Distribution channels (dev, nightly, release)](#distribution-channels-dev-nightly-release)
- [Python package versions](#python-package-versions)
- [Native Linux package versions](#native-linux-package-versions)
- [Native Windows package versions](#native-windows-package-versions)

## Overview

Generally we use semantic versioning (SemVer) for most projects, e.g. `X.Y.Z`
where

- `X` is the "major version"
- `Y` is the "minor version"
- `Z` is the "patch version"

The [`version.json`](/version.json) file at the root of TheRock defines the
base version used for packages, while subprojects may have their own independent
library versions (for example `HIPBLASLT_PROJECT_VERSION` in
[`rocm-libraries/projects/hipblaslt/CMakeLists.txt`](https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipblaslt/CMakeLists.txt)).

<!-- TODO: touch on ABI versions in libraries (.so/.dll) -->

<!-- TODO: mention manifest files? (data about subproject commits used in builds) -->

### Constraints and design guidelines

We are limited by what each packaging system accepts as valid versions.

For Python packages see:

- https://packaging.python.org/en/latest/discussions/versioning/
- https://packaging.python.org/en/latest/specifications/version-specifiers/

For Debain packages see:

- https://www.debian.org/doc/debian-policy/ch-controlfields.html#version

For Fedora packages see:

- https://docs.fedoraproject.org/en-US/packaging-guidelines/Versioning/

### Distribution channels (dev, nightly, release)

Most users are expected to use stable releases, but several other distribution
channels are also available and may be of interest to project developers,
users who want early previews of upcoming releases, and QA/test team members.

| Distribution channel | Base URL                          | Source of builds                                                                                                                                                                                             |
| -------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| stable releases      | https://repo.amd.com/rocm/        | Manually promoted prereleases                                                                                                                                                                                |
| prereleases          | https://rocm.prereleases.amd.com/ | Manually triggered workflows in [rockrel](https://github.com/ROCm/rockrel)                                                                                                                                   |
| nightly releases     | https://rocm.nightlies.amd.com/   | Scheduled workflows in [TheRock](https://github.com/ROCm/TheRock)                                                                                                                                            |
| dev releases         | https://rocm.devreleases.amd.com/ | Manually triggered test workflows in [TheRock](https://github.com/ROCm/TheRock)                                                                                                                              |
| dev builds           | No central index                  | Local builds and per-commit workflows in [TheRock](https://github.com/ROCm/TheRock),<br>[rocm-libraries](https://github.com/ROCm/rocm-libraries), [rocm-systems](https://github.com/ROCm/rocm-systems), etc. |

## Python package versions

Python package versions are handled by scripts:

- [`build_tools/compute_rocm_package_version.py`](/build_tools/compute_rocm_package_version.py)
  - [`build_tools/tests/compute_rocm_package_version_test.py`](/build_tools/tests/compute_rocm_package_version_test.py)

The script produces these versions for each distribution channel:

| Distribution channel | Version format    | Version example                                                                                                                                                     |
| -------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| stable releases      | `X.Y.Z`           | `7.10.0`                                                                                                                                                            |
| prereleases          | `X.Y.ZrcN`        | `7.10.0rc0`<br>(The first release candidate for that stable release)                                                                                                |
| nightly releases     | `X.Y.ZaYYYYMMDD`  | `7.10.0a20251124`<br>(The nightly release on 2025-11-24)                                                                                                            |
| dev builds/releases  | `X.Y.Z.dev0+NNNN` | `7.10.0.dev0+efed3c3b10a5cce8578f58f8eb288582c26d18c4`<br>(For commit [`efed3c3`](https://github.com/ROCm/TheRock/commit/efed3c3b10a5cce8578f58f8eb288582c26d18c4)) |

Each distribution channel (and GPU family within that channel) is currently
hosted on a separate release index that can be passed to `pip` or `uv` via
`--index-url`. For example:

```bash
pip install --index-url=https://rocm.nightlies.amd.com/v2/gfx94X-dcgpu/ rocm`
```

See [RELEASES.md - Installing releases using pip](../../RELEASES.md#installing-releases-using-pip)
for details.

> [!NOTE]
> We plan on later providing a single multi-architecture index as part of
> multi-arch work, see
> [RFC0008-Multi-Arch-Packaging.md - Python Packaging](../rfcs/RFC0008-Multi-Arch-Packaging.md#python-packaging).

### External Python package versions

When we build external projects like
[PyTorch](https://github.com/pytorch/pytorch) we sometimes extend the base
package version with our own
[local version identifier](https://packaging.python.org/en/latest/specifications/version-specifiers/#local-version-identifiers).

For example, for torch version `2.9.0` built with ROCm version `7.9.0` we
generate a composite torch version `2.9.0+rocm7.9.0`.

<!-- TODO: Document [stable, rc, and dev] local version identifiers. -->

<!-- See https://github.com/ROCm/TheRock/pull/2392 which may change for dev -->

#### PyTorch versions

PyTorch packages versions are handled via scripts:

- [`build_tools/github_actions/determine_version.py`](/build_tools/github_actions/determine_version.py) (this generates e.g. `--version-suffix +rocm7.10.0`)
  - [`build_tools/github_actions/tests/determine_version_test.py`](/build_tools/github_actions/tests/determine_version_test.py)
- [`external-builds/pytorch/build_prod_wheels.py`](/external-builds/pytorch/build_prod_wheels.py) (this appends the version suffix to each build version)
- [`build_tools/github_actions/write_torch_versions.py`](/build_tools/github_actions/write_torch_versions.py) (this finds the versions in built packages)

The scripts produce these versions for each distribution channel:

| Package name        | Example release version | Example nightly version        |
| ------------------- | ----------------------- | ------------------------------ |
| torch               | `2.7.1+rocm7.9.0rc1`    | `2.10.0a0+rocm7.10.0a20251024` |
| torchaudio          | `2.7.1a0+rocm7.9.0rc1`  | `2.10.0a0+rocm7.10.0a20251024` |
| torchvision         | `0.22.1+rocm7.9.0rc1`   | `0.24.0+rocm7.11.0a20251124`   |
| pytorch-triton-rocm | `3.3.1+rocm7.9.0rc1`    | `3.5.1+rocm7.11.0a20251124`    |

#### JAX versions

TODO: fill this in together with:

- https://github.com/ROCm/TheRock/issues/1985
- https://github.com/ROCm/TheRock/issues/2091

<!--
- jax-rocm7-pjrt
- jax-rocm7-plugin
- jaxlib (no rocm code in here) -->

### Working with Python package versions

When working with versions please use these tools and avoid custom parsing
(such as regex) if possible:

- The `packaging.version` Python module: https://packaging.pypa.io/en/stable/version.html

  For example:

  ```python
  >>> from packaging.version import Version
  >>> v1 = Version("1.1.0")
  >>> v2 = Version("1.2.0+abc")
  >>> v2 > v1
  True
  >>> v2.base_version
  '1.2.0'
  ```

- Existing Python scripts:

  - [`build_tools/compute_rocm_package_version.py`](/build_tools/compute_rocm_package_version.py)
  - [`build_tools/github_actions/determine_version.py`](/build_tools/github_actions/determine_version.py)
  - [`build_tools/github_actions/write_torch_versions.py`](/build_tools/github_actions/write_torch_versions.py)

#### Tip - installing prereleases

Python package installers like pip ignore pre-releases by default if a final
release exists unless explicitly requested with e.g.
`pip install rocm==7.10.0rc0` or `pip install --pre rocm`. See also
[Python Packaging User Guide - Versioning](https://packaging.python.org/en/latest/discussions/versioning/).

#### Tip - Upgrading and force reinstalling

The `--upgrade` and `--force-reinstall` options can also be useful when
switching between version types to ensure that the expected package versions
are used. See the documentation for
[pip install](https://pip.pypa.io/en/stable/cli/pip_install/).

#### Tip - checking package versions

A few ways to look up the version of an installed package are:

- [`pip show`](https://pip.pypa.io/en/stable/cli/pip_show/):

  ```console
  $ pip show torch | grep Version
  Version: 2.10.0a0+rocm7.11.0a20251209
  ```

- [`pip list`](https://pip.pypa.io/en/stable/cli/pip_list/):

  ```console
  $ pip list | grep torch
  torch                          2.10.0a0+rocm7.11.0a2025120
  ```

- [`pip freeze`](https://pip.pypa.io/en/stable/cli/pip_freeze/):

  ```console
  $ pip freeze | grep torch
  torch==2.10.0a0+rocm7.11.0a20251209
  ```

- The `__version__` module attribute:

  ```console
  $ python -c "import torch; print(torch.__version__)"
  2.10.0a0+rocm7.11.0a20251209
  ```

## Native Linux package versions

TODO: fill this in together with https://github.com/ROCm/TheRock/pull/2159

## Native Windows package versions

TODO: fill this in together with https://github.com/ROCm/TheRock/pull/2159
