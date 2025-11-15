# TheRock package versioning

We build and distribute packages for a variety of projects across multiple
packaging systems, release channels, and operating systems.

This document describes the version schemes we use.

## Overview

Generally we use semantic versioning (SemVer) for most projects, e.g. `X.Y.Z`
where

- `X` is the "major version"
- `Y` is the "minor version"
- `Z` is the "patch version"

The [`version.json`](/version.json) file at the root of TheRock defines the
base version used for packages, while subprojects may have their own independent
library versions (for example `HIPBLASLT_PROJECT_VERSION` in
[`rocm-libraries/projects/hipblaslt/CMakeLists.txt`](/rocm-libraries/projects/hipblaslt/CMakeLists.txt)).

<!-- TODO: touch on ABI versions in libraries (.so/.dll) -->

<!-- TODO: mention manifest files? (data about subproject commits used in builds) -->

## Constraints and design guidelines

We are limited by what each packaging system accepts as valid versions.

For Python packages see:

- https://packaging.python.org/en/latest/discussions/versioning/
- https://packaging.python.org/en/latest/specifications/version-specifiers/

For Debain packages see:

- https://www.debian.org/doc/debian-policy/ch-controlfields.html#version

For Fedora packages see:

- https://docs.fedoraproject.org/en-US/packaging-guidelines/Versioning/

## Release channels (dev, nightly, release)

Most users are expected to use stable releases, but several other release
channels are also available and may be of interest to project developers,
users who want early previews of upcoming releases, and QA/test team members.

### Python package release channels

| Release channel | Version scheme    | Index URL                                                        | Source of builds                                                                                                                                                                                          |
| --------------- | ----------------- | ---------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| stable          | `X.Y.Z`           | https://repo.amd.com/rocm/whl/                                   | Manually promoted prereleases                                                                                                                                                                             |
| prerelease      | `X.Y.ZrcN`        | https://rocm.prereleases.amd.com/whl/                            | Manually triggered workflows in [rockrel](https://github.com/ROCm/rockrel)                                                                                                                                |
| nightly         | `X.Y.ZaYYYYMMDD`  | https://rocm.nightlies.amd.com/v2/                               | Scheduled workflows in [TheRock](https://github.com/ROCm/TheRock)                                                                                                                                         |
| dev releases    | `X.Y.Z.dev0+NNNN` | https://d25kgig7rdsyks.cloudfront.net/v2/<br>(Nicer URL pending) | Manually triggered test workflows in [TheRock](https://github.com/ROCm/TheRock)                                                                                                                           |
| dev builds      | `X.Y.Z.dev0+NNNN` | No central index                                                 | Local builds and per-commit workflows in [TheRock](https://github.com/ROCm/TheRock), [rocm-libraries](https://github.com/ROCm/rocm-libraries), [rocm-systems](https://github.com/ROCm/rocm-systems), etc. |

<!-- TODO: specific examples -->

<!-- TODO: mention v2-staging, whl/, whl-staging/ -->

<!-- TODO: mention wheelnext / wheel variants, link to tracking issue once filed -->

<!-- TODO: mention how versions compare to one another (stable > rc > a > dev) -->

<!--       stable takes precedence, exact versions always work, add `--pre` to allow rc/a/dev -->

## External project versions

When we build external projects like
[PyTorch](https://github.com/pytorch/pytorch) we sometimes extend the base
package version with our own
[local version identifier](https://packaging.python.org/en/latest/specifications/version-specifiers/#local-version-identifiers).

For example, for torch version `2.9.0` built with ROCm version `7.9.0` we use
generate a composite torch version `2.9.0+rocm7.9.0`.

### PyTorch versions

TODO: specific examples of torch versions, links to how this is done in code

- torch
- torchvision
- torchaudio
- pytorch-triton-rocm

### JAX versions

TODO: specific examples of torch versions, links to how this is done in code

- jax-rocm7-pjrt
- jax-rocm7-plugin
- jaxlib (no rocm code in here)

## Working with versions

When working with versions please use these tools and avoid custom parsing
(such as regex) if possible:

- The `packaging.version` Python module: https://packaging.pypa.io/en/stable/version.html
- Python scripts:
  - [`build_tools/compute_rocm_package_version.py`](/build_tools/compute_rocm_package_version.py)
  - [`build_tools/github_actions/determine_version.py`](build_tools/github_actions/determine_version.py)
  - [`build_tools/github_actions/write_torch_versions.py`](build_tools/github_actions/write_torch_versions.py)

<!-- TODO: code samples (e.g. how to compare python package versions) -->

<!-- TODO: how to check the version for each package/tool (`--version` commands, `pip show`) -->
