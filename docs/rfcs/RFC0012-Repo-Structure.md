---
author: Saad Rahim (saadrahim)
created: 2026-04-08
modified: 2026-05-28
status: draft
---

# ROCm software ecosystem package repository structure

## Related RFCs

- **RFC0008** — multi-arch Python wheels and device extras (source of
  the multi-arch wheel model; the two indices it describes are
  published here as `whl/` (backward-compatible, all-arch) and
  `whl-next/` (explicit device extras), following the PyTorch
  `https://download.pytorch.org/whl/` naming convention).
- **RFC0009** — native packaging conventions (source of the
  `amdrocm<major>-<project>` package-naming family referenced for extras).
- **RFC00XX** — end-user projects independent release lifecycle (defines
  the per-project release model for extras; this RFC defines where those
  artifacts land on `repo.amd.com`). Available as a PR at this stage.

## Overview

repo.amd.com's open source software release publications need standardization. In scope is the ROCm software ecosystem which spans the ROCm Core SDK, expansions like ROCm-DS, and standalone projects like RVS. Software packaging for this ecosystem needs a well defined hierarchy reflected in the package distribution folder structure. As software is published on repo.amd.com, the planned hierarchy must be extensible by other software ecosystem published by AMD. As a result, this proposal includes the ability to add the AMD GPU driver to this structure in the future.

## Definitions

- Repository Streams
  - dev - per-commit / pre-nightly developer builds, promoted in from
    the prior `rocm.devreleases.amd.com` host. Lowest bar; intended
    for developer-facing testing and infra dry-runs, not end users.
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
  > (`dev`/`nightly`/`rc`/`stable`, with `ltsrc`/`lts` reserved).
  > Per-extra publishing maps into those streams as follows: per-commit
  > / developer builds → `dev/`, nightly / staging builds → `nightly/`,
  > pre-release builds intended for QA → `rc/`, GA releases →
  > `stable/`. Each extra's release notes record which stream a given
  > build landed in.
- Products
  - Core SDK
  - expansions - SDK built with dependencies on the ROCm Core SDK
    - HPC SDK - a ROCm expansion released on its own cadence, pinned to a single ROCm version per release
  - extras - standalone components part of ROCm
  - HPC SDK - top-level peer to `core/`, `pytorch/`, `jax/`,
    `onnx-runtime/`; a ROCm expansion released on its own cadence,
    pinned to a single ROCm version per release. (Not a child of
    `expansions/`.)
- Python wheel indices — two **central** sibling PEP 503 indices per
  stream, sitting directly under each subdomain's `rocm/` tree (no
  `pyindex/` wrapper). See "Python Indices" section:
  - `whl/` — backward-compatible / all-arch index where
    `pip install rocm` pulls in every device extra automatically.
    Matches the PyTorch `download.pytorch.org/whl/` shape so users
    can swap index URLs without changing install habits.
  - `whl-next/` — explicit-device-extras index where the user picks
    the device they want (e.g. `pip install rocm[device-gfx942]`).
    Smaller installs; requires the user to know their target arch.

## Stream Subdomains

Each release stream is hosted at its own subdomain of `repo.amd.com`,
using the pattern `<stream>.repo.amd.com`. The subdomain *is* the
stream selector — stream-scoped paths sit at the root of the subdomain
rather than under a `<stream>/` prefix on the parent domain.

| Stream    | Subdomain               | Status            |
| :-------- | :---------------------- | :---------------- |
| dev       | `dev.repo.amd.com`      | required at v2    |
| nightly   | `nightly.repo.amd.com`  | required at v2    |
| rc        | `rc.repo.amd.com`       | required at v1    |
| stable    | `stable.repo.amd.com`   | required at v1    |
| ltsrc     | `ltsrc.repo.amd.com`    | future (reserved) |
| lts       | `lts.repo.amd.com`      | future (reserved) |
| archives  | `archives.repo.amd.com` | required at v2    |

The `dev` subdomain **replaces** the existing
`rocm.devreleases.amd.com` host. Once `dev.repo.amd.com` is live, the
old host is redirected to it and retired. This brings developer
pre-nightly builds under the same domain, certificate, and stream
contract as every other stream.

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
  `expansions/`, `extras-[ROCm-major]/`, `whl/`, `whl-next/`, `pytorch/`, etc.).
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
  e.g. a `stable` ROCm Core SDK alongside an `lts` ROCm Core SDK, or
  `stable` + `nightly` side-by-side. Package naming, install prefixes,
  and repo definitions are designed to coexist without conflict.
  - **Mechanism:** the `amdrocm-repo` package installs **one repo
    definition per stream** (rpm: distinct `.repo` files; deb: distinct
    deb822 `.sources` stanzas) — *not* a single repo file templated by
    `$amdrocm_release_stream`. Each stream's repo definition is
    independently `enabled`/`disabled`. The
    `$amdrocm_release_stream` variable selects the **default** stream
    for unqualified `yum install` / `apt install`; users opt into a
    second stream by enabling its repo (`dnf --enablerepo=...`) or
    qualifying the package version. This is the workflow that supports
    concurrent installs without forcing the user to flip a single
    global variable.
  - **Coexistence on disk:** every stream's Core SDK installs into a
    stream-distinct, version-scoped path under `/opt/rocm/core/` (see
    *Install Locations*), so `stable` 7.12 and `nightly` 20260602 do
    not collide.
  - **Compatibility caveat:** concurrent installs are supported by the
    *packaging and on-disk layout*, but cross-stream **dependency
    chains** (e.g. a `nightly` expansion linked against a `stable`
    Core SDK) are not guaranteed to resolve — `nightly` is typically
    one minor version ahead and may pin to ABIs that `stable` does
    not yet ship. The reliable concurrent patterns are: (a) full
    parallel stacks (`stable` Core + `stable` expansion next to
    `nightly` Core + `nightly` expansion), and (b) Python wheels,
    which carry explicit version constraints. Mixing native packages
    across streams in a single dependency chain is **not** a supported
    workflow.
  - **`rc` exception:** `rc` is **not** eligible for concurrent
    installation. An `rc` stream must be the **only** active stream on
    a system while it is enabled — `rc` packages may not be mixed with
    `dev`, `nightly`, `stable`, `ltsrc`, or `lts` packages. This keeps QA
    signal on `rc` clean (a bug seen on `rc` is attributable to `rc`
    alone) and avoids release-candidate artifacts leaking into
    production-flavored installs. The `amdrocm-repo` package must
    enforce this by disabling all other stream repo files whenever
    the `rc` repo is enabled.

## Repository Structure

Hosting splits into three layers, each with its own structure:

1. The **bare `repo.amd.com`** domain (landing page + singletons).
2. The **stream subdomains** `<stream>.repo.amd.com` for `<stream>` in
   `{dev, nightly, rc, stable, ltsrc, lts}` — all share an identical folder
   tree.
3. The **`archives.repo.amd.com`** subdomain.

### Structure on `repo.amd.com` (bare domain)

The bare domain hosts only the navigation landing page and the two
singleton folders. No per-stream artifact content lives here.

- **(landing page)** — `https://repo.amd.com/` — links to every
  active stream subdomain and the archives subdomain.
- **amdrepos/** *(singleton — `https://repo.amd.com/amdrepos/`)*
  - **per-distro subdirectories** — one folder per supported distro
    (e.g. `ubuntu2404/`, `ubuntu2204/`, `rhel9/`, `rhel10/`,
    `sles16/`, `azurelinux3/`). Each distro ships its own
    `amdrocm-repo` package because repo file install paths and URL
    suffixes are distro-family specific (rpm-family →
    `/etc/yum.repos.d/`, deb-family → `/etc/apt/sources.list.d/`).
  - **gpg/** — public signing keys served alongside the repo
    packages. The `amdrocm-repo` package references the key by URL so
    `dnf` / `apt-key` can fetch and pin it on install. Keys are
    rotated through repo-package upgrades; the previous key is kept
    for one release cycle to allow non-broken upgrades.
  - See Repository Package section for contents and the
    `amdrocm-repo` install commands.
- **rocm/** *(singleton — `https://repo.amd.com/rocm/`)* — current
  legacy `rocm/` folder with non-production releases. Retained for
  backward compatibility; to be moved to `archives.repo.amd.com` in
  6 months. While retained, the bare-domain `rocm/` also serves as a
  **navigation index** to the per-stream `rocm/` trees on the stream
  subdomains — each entry below is a folder-style link off
  `https://repo.amd.com/rocm/` that redirects to (or browses through
  to) the matching stream subdomain's `rocm/` root:
  - `dev/` → `https://dev.repo.amd.com/rocm/`
  - `nightly/` → `https://nightly.repo.amd.com/rocm/`
  - `rc/` → `https://rc.repo.amd.com/rocm/`
  - `stable/` → `https://stable.repo.amd.com/rocm/`
  - `ltsrc/` *(future)* → `https://ltsrc.repo.amd.com/rocm/`
  - `lts/` *(future)* → `https://lts.repo.amd.com/rocm/`

  These links are **navigation only** — canonical artifact URLs live
  on the stream subdomains, and package-manager `baseurl`s point
  directly at `<stream>.repo.amd.com`, never at
  `repo.amd.com/rocm/<stream>/`. The legacy non-production content
  underneath `https://repo.amd.com/rocm/` remains in place at its
  existing paths (untouched by these new per-stream link entries)
  until the 6-month archive migration.

> **Name reuse note:** the singleton `rocm/` on the bare domain and
> the `rocm/` folder under each stream subdomain (see next section)
> share the same name but live on different hosts
> (`repo.amd.com/rocm/` vs `<stream>.repo.amd.com/rocm/`). The
> bare-domain `rocm/` is the legacy folder slated for archive **plus**
> a navigation index linking out to each `<stream>.repo.amd.com/rocm/`
> tree; the per-subdomain `rocm/` is the new ROCm platform tree where
> all production artifacts actually live. The per-stream links under
> the bare-domain `rocm/` are not artifact paths — they only point a
> browsing user at the right subdomain.

### Structure on each `<stream>.repo.amd.com` subdomain

The folder tree below is **identical across every stream subdomain**
(`dev`, `nightly`, `rc`, `stable`, `ltsrc`, `lts`). Stream-specific
specializations (per-stream content variants, retention) are captured
in *Per-stream specializations* further down.

- **amdgpu/** *(reserved for future use; same tree, future GPU driver
  artifacts)*
- **rocm/**
  - **whl/** — **central** PEP 503 simple index for the entire
    stream — **backward-compatible / all-arch** variant. A single
    `whl/` serves wheels from every wheel-producing area in the
    stream (`core/`, `expansions/`, wheel-producing extras,
    `pytorch/`, `jax/`, `onnx-runtime/`, `hpc-sdk/`). `pip install
    rocm` (or `pip install torch`) against this index pulls in
    **all** device extras automatically. Matches the PyTorch
    `download.pytorch.org/whl/` shape.
  - **whl-next/** — **central** PEP 503 simple index for the entire
    stream — **explicit-device-extras** variant. The user picks the
    device they want (e.g. `pip install rocm[device-gfx942]`).
    Smaller installs; requires the user to know their target arch.
    Serves the same set of wheel-producing areas as `whl/`.

    See Python Indices section. There is no per-package central
    index; the per-component `whl/` and `whl-next/` folders listed
    below are storage buckets that the central indices reference.
  - **core/**
    - tarball — Linux archive format, **`.tar.gz` only** (other
      compression formats such as `.tar.xz` are not produced).
    - zip — **Windows only**; the `.zip` archive is the Windows
      equivalent of the Linux tarball and is not produced for Linux
      distros.
    - windows-installers — Windows installer artifacts (`.exe`,
      `.msi`) for setups that don't unpack the `zip` archive.
    - linux-installers — Linux installer artifacts, including the
      runfile installer (`.run` self-extracting installer for
      environments without a package manager).
    - **whl/** — backward-compatible / all-arch wheels (referenced
      by the central `whl/` index above).
    - **whl-next/** — explicit-device-extras wheels (referenced by
      the central `whl-next/` index above). Internal folder layout,
      filenames, and sub-paths are implementation details left to
      the publish tooling — not consumed by humans.
    - packages
      - **Linux Distros [a–z]** — one folder per supported distro
        identifier, **alphabetized**. Each distro ships in two
        package-type variants per the RFC0009 Repository Layout
        rule (`Package-type = standard, asan, future variant`):
        - `<distro>/` — **standard** packages (release builds; the
          default install for end users).
        - `<distro>-asan/` — **ASAN** packages (AddressSanitizer
          builds of the same component set, for memory-error
          debugging). ASAN packages are kept in a sibling folder
          rather than mixed into the standard distro folder so they
          remain invisible to package managers that have only added
          the standard repo, and so that `dnf install rocm` from
          `<distro>/` never accidentally pulls an ASAN build.
        - *Future variants* follow the same `<distro>-<variant>/`
          sibling-folder pattern. Reserved future variants include:
          - `<distro>-rpath/` — **rpath** packages (the `$ORIGIN`-
            based RPATH variant of the same component set, served
            via the `amdrocm-repo-rpath` package). Currently rpm-
            only per *Per-stream specializations*; reserved here so
            the folder name is fixed when the variant ships.
          - `<distro>-debug/` — debug-symbols / unstripped builds.
          Additional `-<variant>` siblings may be added later
          without changing this naming rule.
        - Example for RHEL 10:
          ```
          packages/
            rhel10/         # standard release packages
            rhel10-asan/    # ASAN-instrumented packages
          ```
        - The full set of distro identifiers (`debian12`,
          `ubuntu2204`, `ubuntu2404`, `rhel8`, `rhel9`, `rhel10`,
          `sles15`, `azl3`) is defined by RFC0009; every supported
          distro gets at minimum a `<distro>/` folder, and a
          `<distro>-asan/` sibling whenever ASAN builds are
          published for that distro.
  - *(no separate top-level `windows/` folder — Windows artifacts
    live alongside their Linux siblings inside each component, e.g.
    `core/zip`, `core/windows-installers`.)*
  - **expansions [a–z]/** — generic expansions that follow the ROCm
    release cadence. (HPC SDK is **not** here; it is a top-level peer
    — see `hpc-sdk/` below.)
    - tarball
    - **whl/** and **whl-next/** — same two-bucket rule as
      `core/whl/` + `core/whl-next/`. Internal layout is
      implementation-defined.
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
      Wheels are published through the central `whl/` and `whl-next/`
      indices like all other ROCm wheels.
    - **Major-version compatibility:** each `extras-[ROCm-major]`
      directory is the compatibility boundary. An extra published under
      `extras-7` is guaranteed to install against any ROCm 7.x Core SDK
      release; cross-major compatibility is not promised. See RFC0012
      for the full compat contract.
    - **rvs/** — tarball, packages
    - **rocoptiq/** — tarball, whl, whl-next, packages
    - **omnistat/** — whl, whl-next
  - **hpc-sdk/** *(top-level peer to `core/`, `pytorch/`, `jax/`,
    `onnx-runtime/`; see HPC SDK Release Model section)* — released on
    its own cadence with date-based `YYYY.MM` versioning, pinned to a
    single ROCm Core SDK version per release. Contains tarball,
    installers, `whl/` + `whl-next/` (same rule as
    `core/whl/` + `core/whl-next/`), and `packages/` per distro.
  - **pytorch/** — `whl/` + `whl-next/` (same rule as `core/`).
  - **jax/** *(follows the same artifact rules as **pytorch**)*
  - **onnx-runtime/** *(follows the same artifact rules as **pytorch**)*

#### Per-stream specializations

The tree above is the same on every stream subdomain. The
stream-specific differences are:

- **`dev.repo.amd.com`** — Retention: 30 days. Per-commit /
  pre-nightly developer builds, replacing the legacy
  `rocm.devreleases.amd.com` host. Lowest bar of all streams; no QA
  gate. Layout is a **flat package repository** under
  `dev.repo.amd.com/rocm/core/packages/<distro>/` (same shape as
  `stable`/`rc`). Retention is enforced by pruning old package
  versions from the flat tree, not by deleting date-stamped
  subfolders. Intended for developer-facing testing and infra
  dry-runs; **not** for end users. No `rc`-style exclusivity applies
  — `dev` is eligible for concurrent installation with other streams
  on the same machine (see *Concurrent stream installation*).
- **`nightly.repo.amd.com`** — Retention: 120 nightly. Promoted
  builds from the develop branch (`dev` builds that passed the
  promotion gate).
  - **Repo layout:** `nightly.repo.amd.com/rocm/core/packages/<distro>/`
    is a **flat package repository**, identical in structure to
    `stable` and `rc`. All retained nightly versions are co-resident in
    that single flat tree (multiple `amdrocm-core-<NNNN>` package
    versions side-by-side), so the generic `amdrocm-repo` package
    works without a date-stamped subpath. Retention is enforced by
    pruning *old package versions out of the flat tree*, not by
    deleting date-stamped subfolders.
  - **Relationship to `dev`:** the `30 dev / 120 nightly` retention
    in earlier drafts is now split across the two streams — `dev`
    keeps 30 days of per-commit builds; `nightly` keeps 120 days of
    promoted builds. Both are tagged in package metadata so users can
    filter via `dnf --showduplicates` and install a specific version.
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

### Install Locations (ROCm Core SDK)

Native packages install the ROCm Core SDK into a stream-scoped,
version-scoped directory directly under `/opt/rocm/`, following the
hyphenated `/opt/rocm/core-<ver>` naming convention defined in
RFC0009 (*TheRock Software Packaging Requirements* — Directory
Layout). The version component is **chosen per stream** so multiple
streams can coexist on the same machine without colliding (see
*Concurrent stream installation* in Stream Subdomains).

| Stream   | Variant    | Install location                              | Example                                     |
| :------- | :--------- | :-------------------------------------------- | :------------------------------------------ |
| `dev`    | standard   | `/opt/rocm/core-dev-<YYYYMMDD-sha>`           | `/opt/rocm/core-dev-20260602-a3f1c9`        |
| `dev`    | asan       | `/opt/rocm/core-dev-<YYYYMMDD-sha>-asan`      | `/opt/rocm/core-dev-20260602-a3f1c9-asan`   |
| `nightly`| standard   | `/opt/rocm/core-<YYYYMMDD>`                   | `/opt/rocm/core-20260602`                   |
| `nightly`| asan       | `/opt/rocm/core-<YYYYMMDD>-asan`              | `/opt/rocm/core-20260602-asan`              |
| `rc`     | standard   | `/opt/rocm/core-<X.Y>rc<N>`                   | `/opt/rocm/core-7.12rc1`                    |
| `rc`     | asan       | `/opt/rocm/core-<X.Y>rc<N>-asan`              | `/opt/rocm/core-7.12rc1-asan`               |
| `stable` | standard   | `/opt/rocm/core-<X.Y>`                        | `/opt/rocm/core-7.12`                       |
| `stable` | asan       | `/opt/rocm/core-<X.Y>-asan`                   | `/opt/rocm/core-7.12-asan`                  |
| `ltsrc`  | standard   | `/opt/rocm/core-<YYYY.MM>rc<N>`               | `/opt/rocm/core-2026.09rc1`                 |
| `ltsrc`  | asan       | `/opt/rocm/core-<YYYY.MM>rc<N>-asan`          | `/opt/rocm/core-2026.09rc1-asan`            |
| `lts`    | standard   | `/opt/rocm/core-<YYYY.MM>`                    | `/opt/rocm/core-2026.09`                    |
| `lts`    | asan       | `/opt/rocm/core-<YYYY.MM>-asan`               | `/opt/rocm/core-2026.09-asan`               |

The `Variant` column matches the repo-side split from RFC0009 and the
*Linux Distros* section above: `standard` packages come from
`packages/<distro>/`, `asan` packages come from
`packages/<distro>-asan/`. The two variants install to parallel sibling
directories with the `-asan` suffix appended to the standard path, so
they can coexist on disk. Future variants reserved by RFC0009 (e.g.
`rpath`, `debug`) follow the same `-<variant>` suffix pattern
(`/opt/rocm/core-<ver>-rpath`, `/opt/rocm/core-<ver>-debug`) when
introduced.

Rules:

- The trailing version component is **always present** — there is no
  unversioned install directory. `/opt/rocm/core/` (with trailing
  slash) and `/opt/rocm/core-<major>` are reserved by RFC0009 as
  **symlinks** to the latest installed core (e.g. `/opt/rocm/core/
  → /opt/rocm/core-7.12` and `/opt/rocm/core-7 →
  /opt/rocm/core-7.12`); they are not RFC0011's to define. This RFC
  only specifies the per-stream install directory name; the symlink
  policy stays owned by RFC0009.
- `dev`, `nightly`, and `rc` use distinct version syntaxes from
  `stable`/`lts` so a glob (`ls /opt/rocm/core-*`) makes the stream
  obvious without inspecting metadata.
- The `rc` and `ltsrc` directories are short-lived: they are removed
  by the stable/LTS package's post-install (or by the user's package
  manager when the matching final release supersedes them).
- Extras install under `/opt/rocm/extras-<ROCm-major>/` per RFC0012
  and are **not** stream-scoped on disk — the stream-scoped path
  applies to Core SDK only. (Extras coexist across streams through
  their own versioning per RFC0012 §2.)
- Patch releases land in-place under the same directory (`stable`
  `7.12.1` overwrites `/opt/rocm/core-7.12`); they do not create a
  new path. This matches RFC0009's "patch versions must be in place
  within the existing X.Y folder" rule.
- **HPC SDK** follows the same hyphenated convention from RFC0009:
  installs at `/opt/rocm/hpc-<YYYY.MM>` (e.g. `/opt/rocm/hpc-2026.06`)
  with `/opt/rocm/hpc/ → /opt/rocm/hpc-<latest>` as the convenience
  symlink. HPC SDK patch releases (`YYYY.MM.N`) land in-place inside
  the matching `hpc-YYYY.MM` directory.

> **Note — `dev` / `nightly` simplified layout (open option):** as an
> alternative to the date-stamped paths above, `dev` and `nightly`
> may instead install at `/opt/rocm/core-<X.Y>-dev`, where `<X.Y>` is
> the **next** ROCm Core release the develop branch is targeting
> (e.g. if the next stable is 7.14, develop-branch builds install at
> `/opt/rocm/core-7.14-dev`). Under this option there is **no plan
> to support coexistence of multiple `dev` or `nightly` builds on the
> same filesystem** — each new build overwrites the previous one
> in-place inside `core-<X.Y>-dev`, the same way patch releases
> overwrite a stable directory. Only one `-dev` directory exists per
> target version; if the develop branch retargets to a new `<X.Y>`,
> the previous `-dev` directory is removed by the package upgrade.
> This trades the "every nightly retained on disk" property for a
> simpler layout that matches the stable scheme exactly. The
> date-stamped variants in the table above remain the option for
> teams that need on-disk coexistence of multiple develop-branch
> snapshots.

## Python Indices

ROCm publishes wheels through two parallel multi-arch indices, per the
direction in [ROCm/TheRock#5289](https://github.com/ROCm/TheRock/pull/5289).
Per-family indices were considered and rejected — only the two
multi-arch flavors below are in scope.

> **Implementation details:** the build, sharding, and publish
> mechanics for `whl/` and `whl-next/` are tracked in
> [ROCm/TheRock#5289](https://github.com/ROCm/TheRock/pull/5289).
> That PR is the source of truth for the index generator, the wheel
> selector behavior, and the on-disk layout *under* `whl/` and
> `whl-next/` (which this RFC intentionally leaves as
> implementation-defined).

**`whl/` and `whl-next/` are central** — two PEP 503 simple indices
per stream, sitting directly under each subdomain's `rocm/` tree
(`<stream>.repo.amd.com/rocm/whl/` and
`<stream>.repo.amd.com/rocm/whl-next/`). Each index serves wheels
from every wheel-producing area in that stream (`core/`,
`expansions/`, any extras that ship wheels, `pytorch/`, `jax/`,
`onnx-runtime/`, `hpc-sdk/`). There is no per-package central index
and no `pyindex/` wrapper. Centralizing the indices lets
`pip install rocm` (and `pip install torch`, `pip install jax`, etc.)
resolve cross-package dependencies in one resolution pass against a
single `--index-url`.

- **`whl/`** — **backward-compatible / all-arch** variant.
  `pip install rocm` (or `pip install torch`) pulls in **all** device
  extras automatically, matching the "it just works" behavior users
  expect from `pip install torch --index-url
  https://download.pytorch.org/whl/rocm7.2`. Larger download (~5.5 GB
  for the torch case), but no architecture knowledge required. This
  index is the **default** for users coming from the PyTorch wheel
  ecosystem.
  ```
  pip install --index-url https://<stream>.repo.amd.com/rocm/whl/ rocm
  ```

- **`whl-next/`** — **explicit-device-extras** variant. The user
  picks the device extras they need (e.g. `rocm[device-gfx942]`).
  Smaller installs, but the user must know their target architecture.
  This is the forward-looking shape that will fold into WheelNext
  once `uv pip install` with wheel-variant providers ships.
  ```
  pip install --index-url https://<stream>.repo.amd.com/rocm/whl-next/ rocm[device-gfx942]
  ```

> **Naming note:** `whl/` and `whl-next/` are required and stable —
> they are the public entry points users pass directly to
> `pip install --index-url …/rocm/whl/` or `…/rocm/whl-next/`, and
> are part of the public contract. The layout *under* each (sharding,
> filename conventions) and the layout of the per-component `whl/` +
> `whl-next/` storage folders that back them remain implementation
> details left to the publish tooling and are not required to be
> human-readable.
>
> **No Python-side repo-setup package:** there is no `amdrocm-repo`
> Python package (or equivalent) and none is planned. The
> `amdrocm-repo` package is a **native (rpm/deb) repo-setup
> package only** — it configures `yum`/`apt` sources and the gpg
> key, and has no role in the Python wheel install path. Users wire
> up `pip` by passing `--index-url` themselves (or by adding the
> index to their `pip.conf` / `requirements.txt`).

Both variants ship under every stream that publishes wheels (`dev`,
`nightly`, `rc`, `stable`; not `ltsrc`/`lts` until LTS exists), and
both are built from the same underlying wheel set — `whl/` simply
republishes the entry-point wheels (`rocm`, `torch`, `torchvision`,
…) with `device-all` added as an automatic requirement, plus links
to the unmodified device wheels in `whl-next/` so storage is not
duplicated.

**Wheel-only ROCm dependency rule (applies to every wheel published on
`repo.amd.com`):** any package distributed as a Python wheel — Core
SDK wheels, expansion wheels, extras wheels, third-party AI fork
wheels — **must obtain its ROCm dependency exclusively through
`whl{,-next}/`**. That is:

- Wheel `install_requires` / `Requires-Dist` entries that resolve a
  ROCm dependency must resolve to **other ROCm wheels served by
  `whl/` or `whl-next/`**.
- Wheels must not depend on, assume the presence of, or trigger the
  installation of ROCm via any **non-wheel** channel: rpm/deb native
  packages, tarballs, runfile installers, container base images,
  out-of-band scripts, etc. If a wheel needs HIP, a ROCm library, or
  any other ROCm component, that component must itself be available as
  a wheel through one of the two central indices.
- A `pip install <pkg> --index-url https://<stream>.repo.amd.com/rocm/whl{,-next}/`
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
manager), and ensures the central indices are the single source of
truth for the wheel side of the ROCm ecosystem.

Future direction: WheelNext (`uv pip install` with a wheel-variant
provider backed by `rocm-bootstrap`) is the long-term plan and will
eventually make `whl/` unnecessary — `whl-next/` becomes the sole
index once WheelNext is widely adopted. Until that lands, both
variants must coexist.

**Native multi-arch (parallel mechanism):** `whl/` and `whl-next/`
cover the **wheel** side of ROCm. The **native-package**
side is covered by `rocm-kpack` per RFC0008 — host and device code are
split into separate packages, with device code shipped either as
per-architecture packages (`amdrocm-<library>-gfx<arch>`) or loaded at
runtime from kpack archives. These device packages and the
architecture-family meta-packages that group them (`rocm-gfx94X`,
`rocm-gfx90X`, etc.) are published under
`rocm/<stream>/core/packages/<distro>/` alongside the host
ROCm Core SDK packages — they are not a separate folder. Wheel
multi-arch (via `whl/` + `whl-next/`) and native multi-arch (via
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
  surfaced through the central `whl/` and `whl-next/`).
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
  needs. A `pip install <pkg> --index-url <whl{,-next}>` must pull in the
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
  (`whl-next/` with explicit `device-gfxNNNN` extras, or
  `whl/` for automatic device coverage) or by
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
  (`dev`/`nightly`/`rc`/`stable`) on every supported distro. Missing
  transitive dependencies fail the publish.
- **Meta-packages:** umbrella packages such as `amdrocm-hpc-YYYY.MM` (HPC
  SDK) inherit this rule and additionally declare their hard dependency
  on the pinned ROCm Core SDK version.

## HPC SDK

The HPC SDK is a **top-level peer** to `core/`, `pytorch/`, `jax/`,
and `onnx-runtime/` under each stream subdomain's `rocm/` tree — it
is **not** a child of `expansions/`. Its release cadence, date-based
versioning (`YYYY.MM`), ROCm pinning rule, meta-package
(`amdrocm-hpc-YYYY.MM`), and component list are defined in a
separate RFC — see **RFC00XX — AMD HPC SDK release model**. This RFC
covers only its placement on `repo.amd.com`; everything HPC-SDK
specific lives in that RFC.

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

> **Conflict with `amdgpu-install`:** the `amdgpu-install` package
> provides the equivalent repo-setup functionality on legacy ROCm
> releases (it writes the same `/etc/yum.repos.d/` and
> `/etc/apt/sources.list.d/` files and registers the AMD GPG key).
> `amdrocm-repo` supersedes that role for ROCm 7.14 and above and
> **conflicts** with `amdgpu-install` — the two cannot be installed
> simultaneously because they manage overlapping repo files and key
> entries. The `amdrocm-repo` packages declare `Conflicts:`/`Breaks:`
> against `amdgpu-install` so the package manager surfaces the
> collision at install time and the user removes one before adding
> the other.

**`amdrocm-repo-rpath` details:**

- Configures the **rpath variant** of the stable stream's `core/`
  packages (the `rpath` variant listed under
  `stable.repo.amd.com` per *Per-stream specializations*). The repo
  it adds points at a distinct subpath under the `stable` subdomain
  so that `dnf install rocm` resolves to rpath-built packages.
- **Coexistence with `amdrocm-repo`:** the two packages **may** be
  installed simultaneously. The rpath variant installs to a distinct
  versioned directory under `/opt/rocm/` (parallel to the standard
  install per the *Install Locations* section), so the two variants
  no longer collide on disk and do not need to declare mutual
  `Conflicts:`.
- Available as `.rpm` only (deb-family packaging does not currently
  ship an rpath variant).

Repository stream selection must be implemented per package manager.
For rpm packages, install a package-manager variable file, for example
/etc/yum/vars/amdrocm_release_stream or /etc/dnf/vars/amdrocm_release_stream,
and use $amdrocm_release_stream in the repo baseurl.
For Debian-based systems, the repository package must not rely on shell-style
or yum-style variable expansion in APT source files. It should install explicit
deb822 .sources stanzas, or separate .sources files, for each supported stream,
using Enabled: yes/no to control which stream is active.

The repository package is to include the amdgpu driver folder from
`repo.radeon.com`. **Temporarily**, `amdrocm-repo` includes links to
**all** AMD GPU driver folders that support ROCm 7.14 and above
(including the `latest/` folder), so users can install any
ROCm-compatible driver version through the same repo configuration.
Once the AMD GPU drivers are consolidated into `repo.amd.com`, this
reduces to a single link to all driver releases for a particular OS
hosted on `repo.amd.com`. Driver-version selection follows the stream:

- `stable` / `lts` → the GA amdgpu driver listed at
  `https://repo.radeon.com/amdgpu/latest/`. Pinned at repo-package
  build time; refreshed only by publishing a new `amdrocm-repo`
  package.
- `rc` / `ltsrc` → the **pre-GA amdgpu driver paired with the
  candidate**, not `latest/`. The repo package built for an `rc` line
  carries an explicit driver version corresponding to the ROCm
  candidate under QA. This prevents an in-flight `rc` from picking up
  a driver bump that the candidate has not been validated against.
- `nightly` → tracks the driver version the develop branch is
  currently built against; updated whenever the nightly pipeline
  rebuilds the repo package.
- `dev` → tracks the driver version the develop branch is currently
  built against, same as `nightly`. Refreshed on every dev repo-package
  rebuild (i.e. effectively per-commit). Intended for developer
  testing only; not a supported install target for end users.

In all cases the amdgpu URL is **pinned at repo-package build time**,
not resolved at install time, so a given installed `amdrocm-repo`
always points at one specific driver version until the user updates
the repo package itself.

The repo packages are published in the **singleton** `amdrepos/` folder
hosted directly on the bare `repo.amd.com` domain (i.e.
`https://repo.amd.com/amdrepos/`). This folder is **not** replicated
under any stream subdomain — there is one canonical copy that serves
all streams. The repo packages add the `amdrepos` repository to the
user's package manager as well, so future updates to the repo packages
themselves come through the normal `yum update` / `apt upgrade` flow.

The repo package itself is **stream-agnostic** — the install URL has no
stream subdomain because `amdrepos/` is the bare-domain singleton.
Stream selection happens **after** install via the
`$amdrocm_release_stream` package-manager variable, which the repo
package templates into each `baseurl` so the active stream subdomain
(`<stream>.repo.amd.com`) is selected at `yum`/`apt` time.

It is designed to work with the following commands:

```
wget https://repo.amd.com/amdrepos/$OS/amdrocm-repo.rpm
rpm -ivh amdrocm-repo.rpm

# or directly via package manager
yum install https://repo.amd.com/amdrepos/$OS/amdrocm-repo.rpm
```
