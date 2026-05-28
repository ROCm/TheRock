---
author: Saad Rahim (saadrahim)
created: 2026-04-08
modified: 2026-05-28
status: draft
---

# ROCm software ecosystem package repository structure

## Related RFCs

- **RFC0008** — multi-arch Python wheels and device extras (source of the
  `pyindex/multi-arch/` model used here).
- **RFC0009** — native packaging conventions (source of the
  `amdrocm<major>-<project>` package-naming family referenced for extras).
- **RFC0012** — end-user projects independent release lifecycle (defines
  the per-project release model for extras; this RFC defines where those
  artifacts land on `repo.amd.com`).

## Overview

repo.amd.com's open source software release publications need standardization. In scope is the ROCm software ecosystem which spans the ROCm Core SDK, expansions like ROCm-DS, and standalone projects like RVS. Software packaging for this ecosystem needs a well defined hierarchy reflected in the package distribution folder structure. As software is published on repo.amd.com, the planned hierarchy must be extensible by other software ecosystem published by AMD. As a result, this proposal includes the ability to add the AMD GPU driver to this structure in the future.

## Definitions

- Repository Streams
  - nightly - nightly builds from the develop branch
  - rc - release candidate builds for the next stable release
  - stable - GA releases of ROCm with a short term support lifecycle, tagged as ROCm releases.
  - ltsrc - *(future)* release candidate builds for the next LTS release
  - lts - *(future)* long term stability (LTS) releases

  > **Note on stream vocabulary:** RFC0012 does not define a strict
  > stream taxonomy for extras — it only states that extras may publish
  > "nightly / pre-release builds … to a staging repository for early
  > validation" alongside ordinary releases. The canonical streams on
  > `repo.amd.com` are the ones defined in this RFC
  > (`nightly`/`rc`/`stable`, with `ltsrc`/`lts` reserved).
  > Per-extra publishing maps into those streams as follows: nightly /
  > staging builds → `nightly/`, pre-release builds intended for QA →
  > `rc/`, GA releases → `stable/`. Each extra's release notes
  > record which stream a given build landed in.
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

## Stream Subdomains

Each release stream is hosted at its own subdomain of `repo.amd.com`,
using the pattern `<stream>.repo.amd.com`. The subdomain *is* the
stream selector — stream-scoped paths sit at the root of the subdomain
rather than under a `<stream>/` prefix on the parent domain.

| Stream    | Subdomain               | Status            |
| :-------- | :---------------------- | :---------------- |
| nightly   | `nightly.repo.amd.com`  | required at v1    |
| rc        | `rc.repo.amd.com`       | required at v1    |
| stable    | `stable.repo.amd.com`   | required at v1    |
| ltsrc     | `ltsrc.repo.amd.com`    | future (reserved) |
| lts       | `lts.repo.amd.com`      | future (reserved) |
| archives  | `archives.repo.amd.com` | required at v1    |

Two folders are **singletons** on the bare `repo.amd.com` domain and
do **not** follow the stream subdomain pattern:

- `amdrepos/` — the repo-packages folder (see Repository Package
  section). Hosted at `https://repo.amd.com/amdrepos/`. Serves all
  streams; stream selection happens inside the installed
  `amdrocm-repo` package via the active stream variable.
- `rocm/` — the existing non-production `rocm/` folder. Hosted at
  `https://repo.amd.com/rocm/`. Retained as-is for backward
  compatibility and scheduled to move to `archives.repo.amd.com` in
  6 months.

Rules:

- Each stream subdomain serves **only** the artifacts for its own
  stream. There is no cross-stream pathing on a single subdomain.
- The folder hierarchy under each stream subdomain matches the
  per-stream structure defined in the next section (e.g. `core/`,
  `expansions/`, `extras-[ROCm-major]/`, `pyindex/`, `pytorch/`, etc.).
- The bare `repo.amd.com` domain serves as a **navigation landing
  page** plus the two singleton folders (`amdrepos/` and `rocm/`). It
  must list and link to every stream subdomain so a user starting at
  `https://repo.amd.com/` can click through to any active stream.
  Beyond the landing page and the two singleton folders, it is **not**
  required to serve any artifact content directly —
  `repo.amd.com/<stream>/...` paths are not part of the contract, and
  canonical artifact URLs live exclusively on the stream subdomains.
- `amdrocm-repo` packages (see Repository Package section) point
  `baseurl` / APT sources at the stream subdomain selected by the
  user's active stream variable (e.g.
  `https://${amdrocm_release_stream}.repo.amd.com/...`). The repo
  packages themselves are downloaded from
  `https://repo.amd.com/amdrepos/`.
- TLS certificates must cover every stream subdomain plus the bare
  `repo.amd.com` (wildcard `*.repo.amd.com` plus the apex is
  acceptable).
- Reserved future subdomains (`ltsrc`, `lts`) must resolve before
  content is published, even if they initially serve an empty index,
  so that repo-package definitions referencing them do not break.
- **Concurrent stream installation:** users are allowed to install
  packages from different streams simultaneously on the same system —
  e.g. a `stable` ROCm Core SDK alongside a `nightly` expansion, or
  `stable` + `lts` side-by-side. Package naming, install prefixes, and
  repo definitions are designed to coexist without conflict.
  **Exception:** `rc` is **not** eligible for concurrent installation.
  An `rc` stream must be the **only** active stream on a system while
  it is enabled — `rc` packages may not be mixed with `nightly`,
  `stable`, `ltsrc`, or `lts` packages. This keeps QA signal on `rc`
  clean (a bug seen on `rc` is attributable to `rc` alone) and avoids
  release-candidate artifacts leaking into production-flavored
  installs. The `amdrocm-repo` package must enforce this by disabling
  all other stream sources whenever `rc` is the active stream.

## Repository Structure

Hosting splits into three layers, each with its own structure:

1. The **bare `repo.amd.com`** domain (landing page + singletons).
2. The **stream subdomains** `<stream>.repo.amd.com` for `<stream>` in
   `{nightly, rc, stable, ltsrc, lts}` — all share an identical folder
   tree.
3. The **`archives.repo.amd.com`** subdomain.

### Structure on `repo.amd.com` (bare domain)

The bare domain hosts only the navigation landing page and the two
singleton folders. No per-stream artifact content lives here.

- **(landing page)** — `https://repo.amd.com/` — links to every
  active stream subdomain and the archives subdomain.
- **amdrepos/** *(singleton — `https://repo.amd.com/amdrepos/`)*
  - packages
    - **Linux Distros [a–z]**
  - See Repository Package section for contents and the
    `amdrocm-repo` install commands.
- **rocm/** *(singleton — `https://repo.amd.com/rocm/`)* — current
  legacy `rocm/` folder with non-production releases. Retained for
  backward compatibility; to be moved to `archives.repo.amd.com` in
  6 months.

> **Name reuse note:** the singleton `rocm/` on the bare domain and the
> `rocm/` folder under each stream subdomain (see next section) share
> the same name but live on different hosts (`repo.amd.com/rocm/` vs
> `<stream>.repo.amd.com/rocm/`) and are unrelated. The bare-domain
> `rocm/` is the legacy folder slated for archive; the per-subdomain
> `rocm/` is the new ROCm platform tree.

### Structure on each `<stream>.repo.amd.com` subdomain

The folder tree below is **identical across every stream subdomain**
(`nightly`, `rc`, `stable`, `ltsrc`, `lts`). Stream-specific
specializations (per-stream content variants, retention) are captured
in *Per-stream specializations* further down.

- **amdgpu/** *(reserved for future use; same tree, future GPU driver
  artifacts)*
- **rocm/**
  - **pyindex/** — **central** PEP 503 simple index for the entire
    stream. A single `pyindex/` serves wheels from every wheel-
    producing area in the stream (`core/`, `expansions/`,
    wheel-producing extras, `pytorch/`, `jax/`, `onnx-runtime/`).
    There is no per-package `pyindex/`. Required sub-folders:
    - **one/** — single-arch variant (user picks device extras)
    - **all/** — all-arch variant (`pip install rocm` pulls in every
      device extra automatically)

    See Python Indices section.
  - **core/**
    - tarball
    - zip
    - installers
    - **whl** — must publish two variants: (1) single-arch wheels
      where the user picks device extras at install time, and (2)
      all-arch wheels where `pip install rocm` pulls in every device
      extra automatically. Internal folder layout, filenames, and
      sub-paths are implementation details left to the publish
      tooling — not consumed by humans.
    - packages
      - **Linux Distros [a–z]**
  - **windows/** — MSI and EXE files for Windows.
  - **expansions [a–z]/** *(e.g. **hpc-sdk** — see HPC SDK Release Model section)*
    - tarball
    - **whl** — same two-variant rule as `core/whl` (single-arch +
      all-arch). Internal layout is implementation-defined.
    - packages
  - **extras-[ROCm-major]/** — projects released independently for each
    ROCm major version.
    - **Decision:** per-project folder structure on `repo.amd.com` (each
      extra gets its own folder; allows S3 bucket permission granularity
      by group). A flat distribution structure was considered and rejected.
    - **Distribution vs install layout — these are separate concerns
      and do not conflict:**
      - *Distribution layout* (this RFC, on `repo.amd.com`): **per
        project**. Each extra has its own folder under
        `extras-[ROCm-major]/` (e.g.
        `stable.repo.amd.com/rocm/extras-7/rvs/`,
        `.../extras-7/rocoptiq/`). Chosen so S3 bucket permissions can
        be granted per project/group.
      - *Install layout* (RFC0012, on the user's disk): **flat**. After
        install, all extras for a given ROCm major share a single
        merged tree (e.g. `/opt/rocm/extras-7/bin/`,
        `/opt/rocm/extras-7/lib/`, ...) — binaries and libraries from
        different extras live side-by-side, not in per-project
        subdirectories.
      - The two layouts are independent: per-project folders on
        `repo.amd.com` are how artifacts are *published and
        permissioned*; the flat tree under `/opt/rocm/extras-7/` is how
        they are *installed and consumed*. A reader should not infer
        that the on-disk layout mirrors the publication folders, or
        vice versa. RFC0012 is the source of truth for the install
        layout.
      - **Name-collision note:** the string `extras-7` appears in both
        places (`<stream>.repo.amd.com/rocm/extras-7/` here and
        `/opt/rocm/extras-7/` in RFC0012) **because both are scoped to
        the ROCm major version**, not because one mirrors the other.
        The matching name is a coincidence of versioning, not a layout
        guarantee — distribution folders and install prefixes remain
        governed by their respective RFCs.
    - **Package naming:** native packages for extras follow the
      `amdrocm<major>-<project>` convention on `repo.amd.com` (e.g.
      `amdrocm7-rvs`, `amdrocm7-rocoptiq`); the **distro-native**
      equivalent shipped through distro repositories is
      `rocm<major>-<project>` (e.g. `rocm7-rvs`). Optional `-devel` /
      `-dev` sibling packages follow the same prefix when an extra
      exposes a public API. See RFC0009 and RFC0012 §5 for the full
      naming and split rules.
    - **Versioning:** extras use **semver**
      (`<project>-<major>.<minor>.<patch>`, e.g. `rvs-1.2.0`) per
      RFC0012 §2. The date-based `YYYY.MM` scheme used elsewhere in
      this RFC is **HPC-SDK-only** and must not be applied to extras.
    - **Python wheels for extras:** when an extra ships a wheel, the
      selector package is named `rocm<major>-<project>` (per RFC0012
      §5) and declares `Requires-Dist: rocm[core] >=X.0, <(X+1).0`.
      Wheels are published through the central `pyindex/one/` and
      `pyindex/all/` like all other ROCm wheels.
    - **Major-version compatibility:** each `extras-[ROCm-major]`
      directory is the compatibility boundary. An extra published under
      `extras-7` is guaranteed to install against any ROCm 7.x Core SDK
      release; cross-major compatibility is not promised. See RFC0012
      for the full compat contract.
    - **rvs/** — tarball, packages
    - **rocoptiq/** — tarball, whl, packages
    - **omnistat/** — whl
  - **pytorch/** — `whl` (two variants, same rule as `core/whl`).
  - **jax/** *(follows the same artifact rules as **pytorch**)*
  - **onnx-runtime/** *(follows the same artifact rules as **pytorch**)*

#### Per-stream specializations

The tree above is the same on every stream subdomain. The
stream-specific differences are:

- **`nightly.repo.amd.com`** — Retention: 30 dev, 120 nightly.
  Daily builds from the develop branch.
- **`rc.repo.amd.com`** — Retention: 2 years. Release-candidate builds
  for the next stable release; tested by QA. Must match `stable`
  layout.
- **`stable.repo.amd.com`** — Retention: forever. Current ROCm Core
  release from TheRock. `core/` ships in two variants:
  - **standard** — default build packages, asan build packages,
    default-debug symbol packages, asan-debug system packages.
  - **rpath** — rpath variant of standard packages.
- **`ltsrc.repo.amd.com`** *(future; Retention: 2 years)* —
  release-candidate builds for the next LTS release; mirrors
  `stable.repo.amd.com` layout.
- **`lts.repo.amd.com`** *(future)* — long-term-support releases.
  Adds a `YYYYMM/` subfolder under each artifact type, otherwise
  mirrors the `stable` layout.

### Structure on `archives.repo.amd.com`

Unmaintained releases, for reference only. Layout is preserved
historically and not constrained by this RFC; new content moved in
from the bare-domain `rocm/` folder (in 6 months) lands here under its
existing path.

## Python Indices

ROCm publishes wheels through two parallel multi-arch indices, per the
direction in ROCm/TheRock#5289. Per-family indices were considered and
rejected — only the two multi-arch flavors below are in scope.

**`pyindex/` is central** — a single PEP 503 simple index per stream
serves wheels from every wheel-producing area in that stream (`core/`,
`expansions/`, any extras that ship wheels, `pytorch/`, `jax/`,
`onnx-runtime/`). There is no per-package `pyindex/`. Centralizing the
index lets `pip install rocm` (and `pip install torch`, `pip install
jax`, etc.) resolve cross-package dependencies in one resolution pass
against a single `--index-url`.

The central index has **two required sub-folders**:

- **`pyindex/one/`** — single-arch variant. The user explicitly picks
  the device extras they need. Smaller installs, but the user must
  know their target architecture.
  ```
  pip install --index-url https://<stream>.repo.amd.com/pyindex/one/ rocm[device-gfx942]
  ```

- **`pyindex/all/`** — all-arch variant. `pip install rocm` (or `pip
  install torch`) pulls in **all** device extras automatically,
  matching the "it just works" behavior users expect from `pip install
  torch --index-url https://download.pytorch.org/whl/rocm7.2`. Larger
  download (~5.5 GB for the torch case), but no architecture knowledge
  required.
  ```
  pip install --index-url https://<stream>.repo.amd.com/pyindex/all/ rocm
  ```

> **Naming note:** the `one/` and `all/` sub-folders under `pyindex/`
> are required and stable — they are the entry points that
> `amdrocm-repo` templates into `--index-url` and are part of the
> public contract. The layout *under* each (sharding, filename
> conventions) and the layout of the underlying `whl/` folders that
> back them remain implementation details left to the publish tooling
> and are not required to be human-readable.

Both variants ship under every stream that publishes wheels
(`nightly`, `rc`, `stable`; not `ltsrc`/`lts` until LTS exists), and
both are built from the same underlying wheel set — `pyindex/all/`
simply republishes the entry-point wheels (`rocm`, `torch`,
`torchvision`, …) with `device-all` added as an automatic requirement,
plus links to the unmodified device wheels in `pyindex/one/` so
storage is not duplicated.

**Wheel-only ROCm dependency rule (applies to every wheel published on
`repo.amd.com`):** any package distributed as a Python wheel — Core
SDK wheels, expansion wheels, extras wheels, third-party AI fork
wheels — **must obtain its ROCm dependency exclusively through
`pyindex/`**. That is:

- Wheel `install_requires` / `Requires-Dist` entries that resolve a
  ROCm dependency must resolve to **other ROCm wheels served by
  `pyindex/one/` or `pyindex/all/`**.
- Wheels must not depend on, assume the presence of, or trigger the
  installation of ROCm via any **non-wheel** channel: rpm/deb native
  packages, tarballs, runfile installers, container base images,
  out-of-band scripts, etc. If a wheel needs HIP, a ROCm library, or
  any other ROCm component, that component must itself be available as
  a wheel through `pyindex/`.
- A `pip install <pkg> --index-url https://<stream>.repo.amd.com/pyindex/{one,all}/`
  must complete the entire ROCm install chain. The user must not be
  required to run `yum`, `apt`, a `.run` installer, or extract a
  tarball as a prerequisite.
- Native package installs (rpm/deb/tar/runfile) remain the supported
  path for users who want ROCm from native packages — but the **wheel
  path is fully self-contained** and never crosses over into the
  native-package world.

This rule keeps every `pip`-driven install fully resolvable from a
single `--index-url`, makes the wheel ecosystem reproducible across
distros (and inside container images that have no system package
manager), and ensures the central `pyindex/` is the single source of
truth for the wheel side of the ROCm ecosystem.

Future direction: WheelNext (`uv pip install` with a wheel-variant
provider backed by `rocm-bootstrap`) is the long-term plan and will
eventually make `pyindex/all/` unnecessary. Until that lands and is
widely adopted, both variants must coexist.

**Native multi-arch (parallel mechanism):** `pyindex/one/` and
`pyindex/all/` cover the **wheel** side of multi-arch. The **native-package**
side is covered by `rocm-kpack` per RFC0008 — host and device code are
split into separate packages, with device code shipped either as
per-architecture packages (`amdrocm-<library>-gfx<arch>`) or loaded at
runtime from kpack archives. These device packages and the
architecture-family meta-packages that group them (`rocm-gfx94X`,
`rocm-gfx90X`, etc.) are published under
`rocm/<stream>/core/packages/<distro>/` alongside the host
ROCm Core SDK packages — they are not a separate folder. Wheel
multi-arch (via the two `pyindex` variants) and native multi-arch (via
`rocm-kpack`) are sibling mechanisms: wheel users go through whichever
`--index-url` matches their workflow, native users install the
matching device or `rocm-gfx<family>X` package from `core/packages/`.
RFC0008 owns the full multi-arch contract for both sides.

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
- **Streams:** published only under `nightly/`, `rc/`, and
  `stable/`. Not published under `ltsrc/` or `lts/` — long-term-support
  guarantees do not extend to third-party fork builds.
- **Artifact format:** `whl` only. No tarballs, no native distro
  packages — users install via `pip` from the matching ROCm wheel index.
- **Dependency rule:** framework wheels must depend **only on Python
  wheels of the ROCm Core SDK** (published under `core/whl/` and
  surfaced through the central `pyindex/one/` and `pyindex/all/`).
  They must not depend on system packages, native distro packages, or
  any non-wheel ROCm artifact. This keeps `pip install` of a framework
  wheel fully self-contained and reproducible across distros.
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
- **Host-only dependencies for extras (transitive chains stop at host
  packages):** the **typical case** for extras (RFC0012 §5) is that
  they call ROCm host APIs and do not ship their own GPU device code.
  In that case, extras declare hard dependencies only on host-side ROCm
  Core SDK packages (compilers, runtimes, host libraries). The
  transitive dependency chain must terminate at host packages —
  consuming-only extras must never pull in a specific GPU architecture
  or any `amdrocm<major>-<library>-gfx*` device package as a hard
  requirement, directly or transitively. Device coverage is the user's
  choice at install time, selected via the multi-arch wheel index
  (`pyindex/multi-arch/` with explicit `device-gfxNNNN` extras) or by
  installing the matching device or architecture-family meta-packages
  (e.g. `rocm-gfx94X`) alongside the extra. Recommended/suggested deps
  (rpm `Recommends:`, deb `Recommends:`) are allowed for discoverability
  but must remain non-hard.

  - **Exception — extras that ship their own GPU kernels (RFC0012 §5,
    "rare case"):** when an extra produces its own pre-compiled device
    code, it must be split into per-architecture package variants named
    `amdrocm<major>-<project>-gfx<arch>` (e.g.
    `amdrocm7-mytool-gfx942`). Each architecture variant **may**
    declare a hard dependency on the matching ROCm library device
    package (e.g. `amdrocm-blas-gfx942`). The host-only rule applies
    only to the consuming case; kernel-shipping extras are explicitly
    exempt for their own device variants and only for them. The host
    (non-`-gfx*`) package of such an extra still follows the host-only
    rule.

  - **CI enforcement:** the dependency-closure check runs in a
    **host-only image** for every consuming extra; pulling in any
    device package as a hard dep fails the publish. Kernel-shipping
    `-gfx<arch>` variants are tested in an image with the matching
    device packages installed.

  This rule mirrors the device-extras model in RFC0008 and the
  end-user-project dependency rules in RFC0012 §5, and applies to both
  native packages and Python wheels.
- **CI enforcement:** the publish pipeline runs a clean-environment
  install test for every expansion and extra in each stream
  (`nightly`/`rc`/`stable`) on every supported distro. Missing
  transitive dependencies fail the publish.
- **Meta-packages:** umbrella packages such as `amdrocm-hpc-YYYY.MM` (HPC
  SDK) inherit this rule and additionally declare their hard dependency
  on the pinned ROCm Core SDK version.

## HPC SDK Release Model

The HPC SDK is a ROCm expansion and is published under the `expansions [a–z]`
folder of the matching `rocm` stream (nightly, rc, stable,
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
- **Streams (required in v1):** `nightly/`, `rc/`, and `stable/` are
  all required at launch. `ltsrc/` and `lts/` are reserved for future use,
  aligned with ROCm LTS.
  - `nightly/` — daily builds against the develop ROCm branch; retention
    matches the `rocm` nightly policy (30 dev / 120 nightly).
  - `rc/` — release candidates for the next stable HPC SDK, tested
    by QA, mirrors `stable/` layout. Retention: 2 years.
  - `stable/` — GA HPC SDK releases, pinned to a stable ROCm version.
    Retention: forever.
  - `ltsrc/` *(future)* — release candidates for the next LTS HPC SDK.
  - `lts/` *(future)* — long-term-support HPC SDK releases, pinned to an
    LTS ROCm version.
- **Layout:** published as `expansions/hpc-sdk/<YYYY.MM>/` under the
  appropriate `rocm` stream, with release metadata recording the
  pinned ROCm version. Nightly builds use a date-stamped subfolder
  (`expansions/hpc-sdk/nightly/<YYYYMMDD>/`).
- **First target:** the first HPC SDK stable release is pinned to ROCm 7.14.
- **Meta-package:** each release ships an installable umbrella package
  `amdrocm-hpc-YYYY.MM` (rpm and deb) that depends on every HPC SDK component
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

The repo packages are published in the **singleton** `amdrepos/` folder
hosted directly on the bare `repo.amd.com` domain (i.e.
`https://repo.amd.com/amdrepos/`). This folder is **not** replicated
under any stream subdomain — there is one canonical copy that serves
all streams. The repo packages add the `amdrepos` repository to the
user's package manager as well, so future updates to the repo packages
themselves come through the normal `yum update` / `apt upgrade` flow.

It is designed to work with the following commands:

```
wget https://repo.amd.com/amdrepos/$OS/amdrocm-repo.rpm
rpm -ivh amdrocm-repo.rpm

# or directly via package manager
yum install https://repo.amd.com/amdrepos/$OS/amdrocm-repo.rpm
```
