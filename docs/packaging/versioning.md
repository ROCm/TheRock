# TheRock package versioning

We build and distribute packages for a variety of projects across multiple
packaging systems, release channels, and operating systems.

This document describes the version schemes we use for those packages.

Table of contents:

- [Overview](#overview)
  - [Constraints and design guidelines](#constraints-and-design-guidelines)
  - [Distribution channels (dev, nightly, release)](#distribution-channels-dev-nightly-release)
- [Release branch metadata](#release-branch-metadata)
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
base version used for packages:

```json
{
  "rocm-version": "10.1.0"
}
```

> [!NOTE]
> Subprojects may have their own independent
> library versions (for example `HIPBLASLT_PROJECT_VERSION` in
> [`rocm-libraries/projects/hipblaslt/CMakeLists.txt`](https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipblaslt/CMakeLists.txt)):
>
> ```cmake
> set(HIPBLASLT_PROJECT_VERSION "1.4.1" CACHE STRING "Semantic version string.")
> ```

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
| nightly releases     | https://rocm.nightlies.amd.com/   | Scheduled workflows in [rockrel](https://github.com/ROCm/rockrel)                                                                                                                                            |
| BKC releases         | TBD                               | Workflows in [rockrel](https://github.com/ROCm/rockrel)                                                                                                                                                      |
| dev releases         | https://rocm.devreleases.amd.com/ | Manually triggered test workflows in [TheRock](https://github.com/ROCm/TheRock) and [rockrel](https://github.com/ROCm/rockrel)                                                                               |
| dev builds           | No central index                  | Local builds and per-commit workflows in [TheRock](https://github.com/ROCm/TheRock),<br>[rocm-libraries](https://github.com/ROCm/rocm-libraries), [rocm-systems](https://github.com/ROCm/rocm-systems), etc. |

Each distribution channel is currently hosted on a separate release index that
can be passed to package managers like `pip` (see
[RELEASES.md - Installing releases using pip](../../RELEASES.md#installing-releases-using-pip)
for details). For example:

```bash
pip install --index-url=https://rocm.nightlies.amd.com/whl-multi-arch/ rocm
pip install --index-url=https://rocm.devreleases.amd.com/whl-multi-arch/ rocm
```

With the exception of "dev releases", each distribution channel only contains
release artifacts of the matching release type. The "dev releases" channel
_can_ contain any type of release.

## Release branch metadata

The `version.json` file also contains a generic `release-metadata` object
hosting fields that release types can interpret as needed. These fields should
be empty on the base branch and may be populated by commits on release branches:

```jsonc
{
  "rocm-version": "10.1.0",
  "release-metadata": {
    // Empty on the base branch.
    "base-date": ""
    // A BKC release branch could set base-date to 20260811 for a package
    // version like 10.1.0a20260811+bkc.20260813 where
    //   * 20260811 is fixed to when the release branch forked from mainline
    //   * 20260813 is the current date
  }
}
```

## Python package versions

Python package versions are handled by scripts:

- [`build_tools/compute_rocm_package_version.py`](/build_tools/compute_rocm_package_version.py)
  - [`build_tools/tests/compute_rocm_package_version_test.py`](/build_tools/tests/compute_rocm_package_version_test.py)

The script produces these versions for each release type:

| Release type | Version format                             | Version example                                                                                                                                                     |
| ------------ | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| stable       | `X.Y.Z`                                    | `7.10.0`                                                                                                                                                            |
| prerelease   | `X.Y.ZrcN`                                 | `7.10.0rc0`<br>(The first release candidate for that stable release)                                                                                                |
| nightly      | `X.Y.ZaYYYYMMDD`                           | `7.10.0a20251124`<br>(The nightly release on 2025-11-24)                                                                                                            |
| nightly-bkc  | `X.Y.ZaBASEDATE+bkc.YYYYMMDD`              | `10.1.0a20260811+bkc.20260814`                                                                                                                                      |
| dev          | `X.Y.Z.dev0+NNNN`                          | `7.10.0.dev0+efed3c3b10a5cce8578f58f8eb288582c26d18c4`<br>(For commit [`efed3c3`](https://github.com/ROCm/TheRock/commit/efed3c3b10a5cce8578f58f8eb288582c26d18c4)) |
| dev-bkc      | `X.Y.Z.dev0+NNNN`<br>(same as regular dev) | `10.1.0.dev0+efed3c3b10a5cce8578f58f8eb288582c26d18c4`                                                                                                              |

### Post releases

Python package versions may add a PEP 440 post-release segment to an existing
stable version. For example, `7.14.0.post1` is newer than `7.14.0` but older
than `7.14.1`:

```text
7.14.0 < 7.14.0.post1 < 7.14.0.post2 < 7.14.1
```

Post releases let us publish a corrected Python distribution derived from a
stable release without changing the underlying ROCm release number. Suitable
uses include correcting package metadata, dependency declarations, or other
Python packaging and publication errors while retaining the stable release's
native payload. Published distributions are immutable, so a corrected artifact
must have a new version rather than replacing the file for `7.14.0`.

Post releases should not be used as a substitute for ROCm patch releases. If a
change fixes ROCm software, changes native binaries, or changes an interface or
ABI, prefer a new patch release such as `7.14.1`. This follows the
[Python packaging guidance for post releases](https://packaging.python.org/en/latest/specifications/version-specifiers/#post-releases),
which strongly discourages using them for maintenance releases containing
software bug fixes.

TheRock can construct post releases from stable release inputs in a few ways:

- Repackage an unchanged stable payload with corrected Python metadata.
- Publish a post release of a leaf or selector package when its compatibility
  constraints allow it to work with the base stable packages.
- Publish a coordinated post release of all packages in a tightly coupled set
  when their metadata requires exact matching versions.

Post releases are not currently a regular release type produced by
`compute_rocm_package_version.py`, and `promote_packages.py --dest-version` does
not currently accept `.postN`. Creating one therefore requires a purpose-built
repackaging flow or an extension to the promotion tooling. The resulting
artifacts must still go through the normal validation and publishing process.

Compatibility must be considered before deciding whether to post-release one
package or a coordinated set:

- An exact requirement such as `rocm-sdk-device-gfx1100==7.14.0` does not accept
  `7.14.0.post1`, and the reverse exact requirement does not accept `7.14.0`.
- Runtime checks based on string equality have the same problem and may reject
  an otherwise compatible mix of base, patch, and post releases.
- A major/minor constraint such as `==7.14.*` permits patch and post releases.
  Use it only at boundaries where that compatibility is an intentional
  contract. Keep exact pins where packages must be built, tested, and installed
  as one versioned set.
- Before publishing, test clean installs as well as upgrades from the base
  stable version. Check that dependency resolution does not downgrade another
  SDK package or produce a mixture that the runtime cannot use.

### BKC versions

For `nightly-bkc`, `BASEDATE` comes from `release-metadata.base-date` in
`version.json` and identifies the regular nightly used to create the BKC
branch. The date in the `bkc.YYYYMMDD` local version identifier is the BKC
build date. This makes a BKC build sort newer than its base nightly but older
than the next regular nightly:

```text
10.1.0a20260811 < 10.1.0a20260811+bkc.20260814 < 10.1.0a20260812
```

### External Python package versions

When we build external projects like
[PyTorch](https://github.com/pytorch/pytorch) we sometimes extend the base
package version with our own
[local version identifier](https://packaging.python.org/en/latest/specifications/version-specifiers/#local-version-identifiers).

For example, for torch version `2.9.0` built with ROCm version `7.10.0` we
generate a composite torch version `2.9.0+rocm7.10.0`. See this table for more
possible version combinations:

| ROCm release type | ROCm version example           | Composite torch version example                                                    |
| ----------------- | ------------------------------ | ---------------------------------------------------------------------------------- |
| stable            | `7.10.0`                       | `2.9.0+rocm7.10.0`                                                                 |
| nightly           | `7.10.0a20251124`              | `2.9.0+rocm7.10.0a20251124`                                                        |
| nightly-bkc       | `10.1.0a20260811+bkc.20260813` | `2.13.0+rocm10.1.0a20260811.bkc.20260813`                                          |
| dev               | `7.10.0.dev0+efed3c`           | `2.9.0+devrocm7.10.0.dev0-efed3c`<br>_(Note the `devrocm` and `-` instead of `+`)_ |

These local version identifiers are specially constructed such that the expected
version sorting of `stable > nightly > dev` is preserved. Note that per the
["Local version identifiers" specification](https://packaging.python.org/en/latest/specifications/version-specifiers/#local-version-identifiers),
comparison and ordering of local version identifiers goes segment by segment
with special rules _different from the rules used for base versions_. This
ordering can be tested like so:

```python
>>> from packaging.version import Version
>>> stable = Version("2.9.0+rocm7.10.0")
>>> nightly = Version("2.9.0+rocm7.10.0a20251124")
>>> dev = Version("2.9.0+devrocm7.10.0.dev0-efed3c")
>>> stable > nightly
True
>>> nightly > dev
True
```

> [!WARNING]
> Known issue: https://github.com/ROCm/TheRock/issues/7183.
>
> Package versions using ROCm version 10.0.0+ sort as _older_ than previous
> versions using this schema:
>
> ```python
> >>> from packaging.version import Version
> >>> Version("2.12.0+rocm10.0") > Version("2.12.0+rocm7.0")
> False
> ```
>
> We plan on publishing 10.0.0+ packages to a new index to avoid this issue for
> future installs.

#### PyTorch versions

PyTorch packages versions are handled via scripts:

- [`build_tools/github_actions/determine_version.py`](/build_tools/github_actions/determine_version.py) (this generates e.g. `--version-suffix +rocm7.10.0`)
  - [`build_tools/github_actions/tests/determine_version_test.py`](/build_tools/github_actions/tests/determine_version_test.py)
- [`external-builds/pytorch/build_prod_wheels.py`](/external-builds/pytorch/build_prod_wheels.py) (this appends the version suffix to each build version)
- [`build_tools/github_actions/write_torch_versions.py`](/build_tools/github_actions/write_torch_versions.py)
  (this finds the versions in built packages)
- [`build_tools/github_actions/generate_pytorch_source_manifest.py`](/build_tools/github_actions/generate_pytorch_source_manifest.py)
  (this computes expected PyTorch ecosystem package versions and records them
  with pinned source commits for checkout/build jobs)
- [`external-builds/pytorch/checkout_from_manifest.py`](/external-builds/pytorch/checkout_from_manifest.py)
  (this checks out the exact source commits recorded in the manifest)

The scripts produce these versions for each distribution channel:

| Package name | Example release version (stable x stable) | Example nightly version (nightly x nightly) |
| ------------ | ----------------------------------------- | ------------------------------------------- |
| torch        | `2.9.1+rocm7.10.0`                        | `2.10.0a0+rocm7.10.0a20251024`              |
| torchaudio   | `2.9.0+rocm7.10.0`                        | `2.10.0a0+rocm7.10.0a20251024`              |
| torchvision  | `0.24.0+rocm7.10.0`                       | `0.24.0+rocm7.11.0a20251124`                |
| triton       | `3.3.1+rocm7.10.0`                        | `3.5.1+rocm7.11.0a20251124`                 |

For manually dispatched `dev` PyTorch builds (`build_prod_wheels.py --release-type dev`), each wheel version is additionally tagged with its own 8-character source commit in the PEP 440 local version segment, e.g. `2.12.0a0+git1a2b3c4d.rocm7.10.0`. This applies to the `torch`, `torchaudio`, and `torchvision` wheels, so each records exactly which source commit produced it.

#### JAX versions

JAX packages versions are handled via scripts:

- [`build_tools/github_actions/determine_version.py`](/build_tools/github_actions/determine_version.py) (this generates e.g. `--version-suffix +rocm7.10.0`)
  - [`build_tools/github_actions/tests/determine_version_test.py`](/build_tools/github_actions/tests/determine_version_test.py)
- [`build_tools/github_actions/write_jax_versions.py`](/build_tools/github_actions/write_jax_versions.py)
  (this finds the versions in built packages)
- [`build_tools/github_actions/generate_jax_manifest.py`](/build_tools/github_actions/generate_jax_manifest.py)
  (this records versions into a manifest file)
  - [`build_tools/github_actions/tests/generate_jax_manifest_test.py`](/build_tools/github_actions/tests/generate_jax_manifest_test.py)
- In the [ROCm/rocm-jax repository](https://github.com/ROCm/rocm-jax): [`build/ci_build`](https://github.com/ROCm/rocm-jax/blob/rocm-jax-infra/build/ci_build) (see the `--rocm-version` and `--no-rocm-version-extra` flags)

Versions for each distribution channel:

| Package name     | Example release version (stable x stable) | Example nightly version (stable x nightly) |
| ---------------- | ----------------------------------------- | ------------------------------------------ |
| jax-rocm7-pjrt   | `0.10.0+rocm7.14.0`                       | `0.10.2+rocm7.15.0a20260712`               |
| jax-rocm7-plugin | `0.10.0+rocm7.14.0`                       | `0.10.2+rocm7.15.0a20260712`               |

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

- The `packaging.specifiers.SpecifierSet` utility:
  https://packaging.pypa.io/en/stable/specifiers.html

  Use `SpecifierSet` when code needs to apply the same compatibility rules as
  Python package metadata. A prefix match can express compatibility across a
  ROCm major/minor release while accepting stable patch and post releases:

  ```python
  >>> from packaging.specifiers import SpecifierSet
  >>> from packaging.version import Version
  >>> compatible_rocm = SpecifierSet("==7.14.*")
  >>> Version("7.14.0") in compatible_rocm
  True
  >>> Version("7.14.0.post1") in compatible_rocm
  True
  >>> Version("7.14.1") in compatible_rocm
  True
  >>> Version("7.15.0") in compatible_rocm
  False
  ```

  Exact equality has different semantics and excludes post releases:

  ```python
  >>> exact_release = SpecifierSet("==7.14.0")
  >>> Version("7.14.0") in exact_release
  True
  >>> Version("7.14.0.post1") in exact_release
  False
  ```

  `SpecifierSet("~=7.14.0")` is another way to accept `7.14.0` or newer
  versions in the `7.14` series. Prefer `==7.14.*` when the policy is simply
  "the major and minor components must match"; use `~=` when its lower bound is
  also part of the compatibility requirement. Pre-releases are excluded by
  default unless explicitly enabled or otherwise selected according to the
  standard Python packaging rules.

  The `packaging` project is available to TheRock build and release tooling,
  but it is not automatically a runtime dependency of every generated package.
  Do not add a `packaging` import to installed SDK code unless the corresponding
  distribution declares that dependency. Runtime code without that dependency
  should use a small, narrowly scoped comparison for the specific version
  policy it implements.

- Existing Python scripts:

  - [`build_tools/compute_rocm_package_version.py`](/build_tools/compute_rocm_package_version.py)
  - [`build_tools/github_actions/determine_version.py`](/build_tools/github_actions/determine_version.py)
  - [`build_tools/github_actions/write_torch_versions.py`](/build_tools/github_actions/write_torch_versions.py)
  - [`build_tools/github_actions/generate_pytorch_source_manifest.py`](/build_tools/github_actions/generate_pytorch_source_manifest.py)
  - [`external-builds/pytorch/checkout_from_manifest.py`](/external-builds/pytorch/checkout_from_manifest.py)

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

TheRock supports rpm and debian packages. Each has different versioning scheme as mentioned below.
Native package versions are handled by scripts:

- [`build_tools/compute_rocm_package_version.py`](/build_tools/compute_rocm_package_version.py)
  - [`build_tools/tests/compute_rocm_package_version_test.py`](/build_tools/tests/compute_rocm_package_version_test.py)

The script produces these versions for rpm packages for each release type:

| Release type | Version format                               | Version example                                                                                                                        |
| ------------ | -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| stable       | `X.Y.Z`                                      | `7.10.0`                                                                                                                               |
| prerelease   | `X.Y.Z~rcN`                                  | `7.10.0~rc0`<br>(The first release candidate for that stable release)                                                                  |
| nightly      | `X.Y.Z~YYYYMMDD`                             | `7.10.0~20251124`<br>(The nightly release on 2025-11-24)                                                                               |
| nightly-bkc  | `X.Y.Z~BASEDATE.bkc.YYYYMMDD`                | `10.1.0~20260811.bkc.20260814`                                                                                                         |
| dev          | `X.Y.Z~YYYYMMDDg<git-hash>`                  | `7.10.0~20251124gefed3c3`<br>(For commit [`efed3c3`](https://github.com/ROCm/TheRock/commit/efed3c3b10a5cce8578f58f8eb288582c26d18c4)) |
| dev-bkc      | `X.Y.Z~YYYYMMDDg<git-hash>`<br>(same as dev) | `10.1.0~20260814gefed3c3`                                                                                                              |

The script produces these versions for debian packages for each release type:

| Release type | Version format                               | Version example                                                        |
| ------------ | -------------------------------------------- | ---------------------------------------------------------------------- |
| stable       | `X.Y.Z`                                      | `7.10.0`                                                               |
| prerelease   | `X.Y.Z~preN`                                 | `7.10.0~pre0`<br>(The first release candidate for that stable release) |
| nightly      | `X.Y.Z~YYYYMMDD`                             | `7.10.0~20251124`<br>(The nightly release on 2025-11-24)               |
| nightly-bkc  | `X.Y.Z~BASEDATE.bkc.YYYYMMDD`                | `10.1.0~20260811.bkc.20260814`                                         |
| dev          | `X.Y.Z~devYYYYMMDD`                          | `7.10.0~dev20251124`<br>(For dev build on 2025-11-24)                  |
| dev-bkc      | `X.Y.Z~devYYYYMMDD`<br>(same as regular dev) | `10.1.0~dev20260814`                                                   |

## Native Windows package versions

TODO: fill this in together with https://github.com/ROCm/TheRock/pull/2159
