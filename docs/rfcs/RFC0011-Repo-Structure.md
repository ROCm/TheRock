---
author: Saad Rahim (saadrahim)
created: 2026-04-08
modified: 2026-05-26
status: draft
---

# ROCm software ecosystem package repository structure

## Overview

repo.amd.com's open source software release publications need standardization. In scope is the ROCm software ecosystem which spans the ROCm Core SDK, expansions like ROCm-DS, and standalone projects like RVS. Software packaging for this ecosystem needs a well defined hierarchy reflected in the package distribution folder structure. As software is published on repo.amd.com, the planned hierarchy must be extensible by other software ecosystem published by AMD. As a result, this proposal includes the ability to add the AMD GPU driver to this structure in the future.

## Definitions

- Repository Streams
  - nightly - nightly builds from the develop branch
  - stablerc - release candidate builds for the next stable release
  - stable - GA releases of ROCm with a short term support lifecycle, tagged as ROCm releases.
  - ltsrc - *(future)* release candidate builds for the next LTS release
  - lts - *(future)* long term stability (LTS) releases
- Products
  - Core SDK
  - expansions - SDK built with dependencies on the ROCm Core SDK
    - HPC SDK - a ROCm expansion released on its own cadence, pinned to a single ROCm version per release
  - extras - standalone components part of ROCm
- pyindex - top-level folder for the ROCm Python package indices. Contains
  two sibling indices (see "Python Indices" section):
  - `pyindex/multi-arch/` — multi-arch index requiring explicit device
    extras (e.g. `pip install rocm[device-gfx942]`)
  - `pyindex/multi-arch-compat/` — backward-compatible multi-arch index
    where `pip install rocm` pulls in all device extras automatically

## Repository Structure

`repo.amd.com` will have the following folder structure:

- **amdgpu** *(reserved for future use; follows the same stream structure as `rocm-platform`: **nightly**, **stablerc**, **stable**, **ltsrc**, **lts**)*
- **amdrepos**
  - packages
    - **Linux Distros [a–z]**
- **archives** *(unmaintained releasees, for reference only)*
- **rocm** (current rocm folder with non production releases, move to archives in 6 months)
- **rocm-platform**

  - **nightly** *(Retention policy: 30 dev, 120 nightly)*
    - **pyindex** *(see Python Indices section)*
      - **multi-arch** *(device extras specified manually)*
      - **multi-arch-compat** *(backward-compatible; `pip install rocm` pulls in all device extras)*
    - **core**
      - tarball
      - zip
      - installers
      - whl
      - packages
        - **Linux Distros [a–z]**
    - **windows**
      - MSI and EXE files for Windows
    - **expansions [a–z]** *(e.g. **hpc-sdk** — see HPC SDK Release Model section)*
      - tarball
      - whl
      - packages
    - **extras-[ROCm-major]** #projects released independently for each ROCm major version
      - **Decision:** per-project folder structure (each extra gets its own
        folder; allows S3 bucket permission granularity by group). Flat
        structure was considered and rejected.
      - **rvs**
        - tarball
        - packages
      - **rocoptiq**
        - tarball
        - whl
        - packages
      - **omnistat**
        - whl
    - **pytorch**
      - **nightly**
        - whl
      - **stablerc**
        - whl
      - **stable**
        - whl
    - **jax** *(follows the same stream and artifact rules as **pytorch**)*
    - **onnx-runtime** *(follows the same stream and artifact rules as **pytorch**)*

  - **stablerc** *(Retention policy: 2 years)*

    - Release candidate builds for the next stable release
    - Mirrors nightly folder structure
    - Tested by QA
    - Must match structure of `repo.amd.com`

  - **stable**

    - Current ROCm Core release from TheRock
    - **standard** (includes default build packages, asan build packages, default-debug symbol packages, and asan-debug system packages)
    - **rpath** (includes rpath variant of standard packages)

  - **ltsrc** — long term support release candidate *(future; Retention policy: 2 years)*

    - Release candidate builds for the next LTS release
    - Mirrors stable folder structure

  - **lts** — long term support *(future)*

    - `YYYYMM`
      - Mirrors stable folder structure

## Python Indices

ROCm publishes wheels through two parallel multi-arch indices, per the
direction in ROCm/TheRock#5289. Per-family indices were considered and
rejected — only the two multi-arch flavors below are in scope.

- **`pyindex/multi-arch/`** — multi-arch index where the user explicitly
  picks the device extras they need. Smaller installs, but the user
  must know their target architecture.
  ```
  pip install --index-url https://repo.amd.com/.../pyindex/multi-arch/ rocm[device-gfx942]
  ```

- **`pyindex/multi-arch-compat/`** — backward-compatible multi-arch
  index. `pip install rocm` (or `pip install torch`) pulls in **all**
  device extras automatically, matching the "it just works" behavior
  users expect from `pip install torch --index-url
  https://download.pytorch.org/whl/rocm7.2`. Larger download (~5.5 GB
  for the torch case), but no architecture knowledge required.
  ```
  pip install --index-url https://repo.amd.com/.../pyindex/multi-arch-compat/ rocm
  ```

Both indices ship under every stream that publishes wheels (`nightly`,
`stablerc`, `stable`; not `ltsrc`/`lts` until LTS exists), and both are
built from the same underlying wheel set — `multi-arch-compat/` simply
republishes the entry-point wheels (`rocm`, `torch`, `torchvision`, …)
with `device-all` added as an automatic requirement, plus links to the
unmodified device wheels in `multi-arch/` so storage is not duplicated.

Future direction: WheelNext (`uv pip install` with a wheel-variant
provider backed by `rocm-bootstrap`) is the long-term plan and will
eventually make `multi-arch-compat/` unnecessary. Until that lands and
is widely adopted, both indices must coexist.

## Third Party AI Forks

`pytorch`, `jax`, and `onnx-runtime` are **ROCm forks/builds of upstream
third-party AI frameworks**, not first-party AMD projects. They are
published on `repo.amd.com` so users can pick up ROCm-enabled wheels
without having to build them locally, but the upstream project owns the
source of truth and the release cadence for the framework itself.

Rules that apply to all third-party AI forks:

- **Upstream tracking:** each entry mirrors a specific upstream release
  (or upstream nightly), with ROCm patches applied on top. Metadata in
  every artifact records the upstream version and the ROCm version it
  was built against.
- **Streams:** published only under `nightly/`, `stablerc/`, and
  `stable/`. Not published under `ltsrc/` or `lts/` — long-term-support
  guarantees do not extend to third-party fork builds.
- **Artifact format:** `whl` only. No tarballs, no native distro
  packages — users install via `pip` from the matching ROCm wheel index.
- **Dependency rule:** framework wheels must depend **only on Python
  wheels of the ROCm Core SDK** (published under `core/whl/` and surfaced
  through `pyindex/`). They must not depend on system packages, native
  distro packages, or any non-wheel ROCm artifact. This keeps `pip
  install` of a framework wheel fully self-contained and reproducible
  across distros.
- **Versioning:** uses the upstream framework's own version string
  (e.g. PyTorch's `2.x.y+rocm<rocm-version>` convention), not the ROCm
  `YYYYMM`/`YYYY.MM` scheme.
- **Support model:** bug fixes for the ROCm-specific delta ship in the
  next stream promotion; we do not back-patch older third-party fork
  releases.
- **Coverage list:** `pytorch`, `jax`, `onnx-runtime`. New third-party
  forks added to `repo.amd.com` follow this same model by default.

## Dependency Closure for Expansions and Extras

Every package published under `expansions/` and `extras-[ROCm-major]/`
must declare a **complete dependency chain**. The goal is that a single
one-line install command for any expansion or extra pulls in every
required component automatically, with no manual follow-up.

Rules:

- **Native packages (rpm/deb):** `Requires:` / `Depends:` must list every
  ROCm Core SDK package, expansion, or extra the component needs at
  runtime, at the exact (or minimum-compatible) versions. A `yum install
  <pkg>` or `apt install <pkg>` must succeed without the user adding
  ROCm Core SDK packages by hand.
- **Python wheels:** `install_requires` / `Requires-Dist` must list every
  ROCm Core SDK wheel (and any other expansion wheel) the component
  needs. A `pip install <pkg> --index-url <pyindex>` must pull in the
  full chain.
- **Cross-format:** packages must not silently rely on a parallel
  artifact in a different format (e.g. an rpm that quietly needs a wheel
  to be installed, or vice versa). Each install path must be
  self-sufficient.
- **CI enforcement:** the publish pipeline runs a clean-environment
  install test for every expansion and extra in each stream
  (`nightly`/`stablerc`/`stable`) on every supported distro. Missing
  transitive dependencies fail the publish.
- **Meta-packages:** umbrella packages such as `rocm-hpc-YYYY.MM` (HPC
  SDK) inherit this rule and additionally declare their hard dependency
  on the pinned ROCm Core SDK version.

## HPC SDK Release Model

The HPC SDK is a ROCm expansion and is published under the `expansions [a–z]`
folder of the matching `rocm-platform` stream (nightly, stablerc, stable,
ltsrc, lts). Unlike other expansions, it is released on its own cadence — decoupled
from ROCm release cadence — and uses date-based versioning. This mirrors the
model used by NVIDIA's HPC SDK, which is published independently of CUDA
releases.

- **Cadence:** approximately 4 releases per year. Not every ROCm release will
  have a corresponding HPC SDK release.
- **Versioning:** date-based, `YYYY.MM` (e.g. `2026.06`). Versioning is
  intentionally independent of ROCm version numbers to make the decoupling
  explicit and avoid mix-and-match confusion.
- **ROCm pinning:** each HPC SDK release is pinned to exactly one ROCm Core
  SDK version. Pinning follows the stream:
  - stable HPC SDK pins to a stable ROCm Core SDK release.
  - LTS HPC SDK (when available) pins to an LTS ROCm Core SDK release.
  Both streams coexist once LTS exists; a stable HPC SDK release is always
  available regardless of LTS status.
- **Patch releases:** bug fixes are delivered as patch releases (`YYYY.MM.N`)
  against the pinned ROCm version, on the same model as LTS maintenance.
- **Streams (required in v1):** `nightly/`, `stablerc/`, and `stable/` are
  all required at launch. `ltsrc/` and `lts/` are reserved for future use,
  aligned with ROCm LTS.
  - `nightly/` — daily builds against the develop ROCm branch; retention
    matches the `rocm-platform` nightly policy (30 dev / 120 nightly).
  - `stablerc/` — release candidates for the next stable HPC SDK, tested
    by QA, mirrors `stable/` layout. Retention: 2 years.
  - `stable/` — GA HPC SDK releases, pinned to a stable ROCm version.
    Retention: forever.
  - `ltsrc/` *(future)* — release candidates for the next LTS HPC SDK.
  - `lts/` *(future)* — long-term-support HPC SDK releases, pinned to an
    LTS ROCm version.
- **Layout:** published as `expansions/hpc-sdk/<YYYY.MM>/` under the
  appropriate `rocm-platform` stream, with release metadata recording the
  pinned ROCm version. Nightly builds use a date-stamped subfolder
  (`expansions/hpc-sdk/nightly/<YYYYMMDD>/`).
- **First target:** the first HPC SDK stable release is pinned to ROCm 7.14.
- **Meta-package:** each release ships an installable umbrella package
  `rocm-hpc-YYYY.MM` (rpm and deb) that depends on every HPC SDK component
  at the versions in that release, plus a hard dependency on the pinned
  ROCm Core SDK version. Patch releases update the component pins without
  changing the meta-package name.
- **Components:** first release includes rocHPCG, rocALUTION, hipFort,
  rocHPL, and hipTensor. Second release adds miniHPL and miniHPCG.

## Repository Package

Allow users to install the all ROCm repositories via a convienient package. The package provides
all the repository files. Example, the rpm repo file will add files to /etc/yum.repos.d/ and
debian repo file will add to /etc/apt/sources.list.d/. The
repo file will also install the gpg key, ideally prompting the user to accept the key. Updating
the repo file will update the gpg key to the latest.

- amdrocm-repo.rpm
- amdrocm-repo-rpath.rpm # rpath is only available for rpm based releases today
- amdrocm-repo.deb
- amdrocm-repo-lts-YYYYMM.rpm #reserved for future LTS release streams

Repository stream selection must be implemented per package manager.
For rpm packages, install a package-manager variable file, for example
/etc/yum/vars/amdrocm_release_stream or /etc/dnf/vars/amdrocm_release_stream,
and use $amdrocm_release_stream in the repo baseurl.
For Debian-based systems, the repository package must not rely on shell-style
or yum-style variable expansion in APT source files. It should install explicit
deb822 .sources stanzas, or separate .sources files, for each supported stream,
using Enabled: yes/no to control which stream is active.

The repository package is to include the latest amdgpu driver folder from repo.radeon.com.
This is temporary until amdgpu is moved to repo.amd.com.

This the repo packages are published in the amd-repos folder in the repo.amd.com hierarchy.
The repo packages adds the amd-repos repository as well.

It is designed to work with the following commands

```
wget https://repo.amd.com/amd-repos/$OS/rocm-repo.rpm
rpm -ivh rocm-repo.rpm

or directly via package manager
yum install https://repo.amd.com/amd-repos/$OS/rocm-repo.rpm
```
