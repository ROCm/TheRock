---
author: Saad Rahim (saadrahim)
created: 2026-04-08
modified: 2026-06-09
status: Approved
---

# ROCm software ecosystem package repository structure

## Overview

repo.amd.com's open source software release publications need standardization. In scope is the ROCm software ecosystem which spans the ROCm Core SDK, expansions like ROCm-DS, and standalone projects like RVS. Software packaging for this ecosystem needs a well defined hierarchy reflected in the package distribution folder structure. As software is published on repo.amd.com, the planned hierarchy must be extensible by other software ecosystem published by AMD. As a result, this proposal includes the ability to add the AMD GPU driver to this structure in the future.

## Definitions

- Repository Streams

  - dev - per-commit / pre-nightly developer builds, promoted in from
    the prior `rocm.devreleases.amd.com` host. Lowest bar; intended
    for developer-facing testing and infra dry-runs, not end users.
  - nightly - nightly builds from the develop branch
  - weekly - weekly promotion of a nightly build, retained on a
    longer horizon than `nightly` for integration testing and as a
    stable target for downstream CI that does not need per-day churn.
    Cadence: one promotion per ISO calendar week.
  - rc - release candidate builds for the next stable release
  - stable - GA releases of ROCm with a short term support lifecycle, tagged as ROCm releases.
  - ltsrc - *(future)* release candidate builds for the next LTS release
  - lts - *(future)* long term stability (LTS) releases

  `archives` is **not** a release stream — it is a retention-only
  subdomain that hosts unmaintained content migrated from other
  sources. See *Stream Subdomains* and *Structure on
  `archives.repo.amd.com`* below.

  > **Note on stream vocabulary:** RFC0012 does not define a strict
  > stream taxonomy for extras — it only states that extras may publish
  > "nightly / pre-release builds … to a staging repository for early
  > validation" alongside ordinary releases. The canonical streams on
  > `repo.amd.com` are the ones defined in this RFC
  > (`dev`/`nightly`/`weekly`/`rc`/`stable`, with `ltsrc`/`lts`
  > reserved). Per-extra publishing maps into those streams as follows:
  > per-commit / developer builds → `dev/`, nightly / staging builds →
  > `nightly/`, weekly-promoted builds intended for downstream CI →
  > `weekly/`, pre-release builds intended for QA → `rc/`, GA releases →
  > `stable/`. Each extra's release notes record which stream a given
  > build landed in.

- Products

  - **Core SDK** — the ROCm Core SDK; published under `core/` on
    every stream subdomain.
  - **Expansions** — SDKs built with dependencies on the ROCm Core
    SDK (e.g. Computer Vision SDK, ROCm-DS, future
    cadenced SDKs). **Expansions are not a folder.** Each expansion
    is published as a **top-level peer to `core/`** under each
    stream subdomain's `rocm/` tree (e.g.
    `computer-vision-sdk/`, `rocm-ds/`), with its own cadence and
    ROCm pinning rule defined in its own RFC. There is no
    `expansions/` parent folder.
  - **Third-party AI forks** — `pytorch/`, `jax/`,
    `onnx-runtime/`. Also top-level peers to `core/`; AMD/ROCm forks
    of upstream third-party AI frameworks, published as ROCm-built
    wheels (see *Third Party AI Forks*).
  - **Extras** — standalone components part of ROCm, released
    independently per ROCm major version; published under a **single
    `extras/` folder** (no `-<ROCm-major>` suffix). For **native
    packages** the ROCm major lives in the package name
    (`amdrocm<major>-<project>`), not the folder path; for **Python
    wheels** the major is *not* in the name — it is carried by the
    `+rocm<major>` PEP 440 local-version tag and a matching
    `Requires-Dist` range (see *Structure on each
    `<stream>.repo.amd.com` subdomain* → *extras/*). Either way every
    major coexists in one tree.

- Python wheel indices — two **central** sibling PEP 503 indices per
  stream, sitting directly under each subdomain's `rocm/` tree (no
  `pyindex/` wrapper), per
  [ROCm/TheRock#5289](https://github.com/ROCm/TheRock/pull/5289).
  See "Python Indices" section:

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

Release streams (grouped by lifecycle):

| Stream  | Subdomain              | Status            |
| :------ | :--------------------- | :---------------- |
| stable  | `stable.repo.amd.com`  | required at v1    |
| rc      | `rc.repo.amd.com`      | required at v1    |
| weekly  | `weekly.repo.amd.com`  | required at v2    |
| nightly | `nightly.repo.amd.com` | required at v2    |
| dev     | `dev.repo.amd.com`     | required at v2    |
| ltsrc   | `ltsrc.repo.amd.com`   | future (reserved) |
| lts     | `lts.repo.amd.com`     | future (reserved) |

Retention / archive subdomain (not a release stream):

| Subdomain               | Status         | Purpose                         |
| :---------------------- | :------------- | :------------------------------ |
| `archives.repo.amd.com` | required at v2 | unmaintained content; read-only |

The `dev` subdomain **replaces** the existing
`rocm.devreleases.amd.com` host. Once `dev.repo.amd.com` is live, the
old host is redirected to it and retired. This brings developer
pre-nightly builds under the same domain, certificate, and stream
contract as every other stream.

Two folders are **singletons** on the bare `repo.amd.com` domain and
do **not** follow the stream subdomain pattern:

- `amdrepos/` — the repo-packages folder (see Repository Package
  section). Hosted at `https://repo.amd.com/amdrepos/`. Serves all
  streams; stream selection happens inside the installed tier
  package (`amdrocm-repo-stable`, `amdrocm-repo-stablerc`,
  `amdrocm-repo-nightly`, or `amdrocm-repo-dev`) via the active
  stream variable. Each tier package only exposes the streams it
  owns — see *Repository Package* for the tier table.
- `rocm/` — the existing non-production `rocm/` folder. Hosted at
  `https://repo.amd.com/rocm/`. Retained as-is for backward
  compatibility and scheduled to move to `archives.repo.amd.com` in
  6 months.

Rules:

- Each stream subdomain serves **only** the artifacts for its own
  stream. There is no cross-stream pathing on a single subdomain.
- The folder hierarchy under each stream subdomain matches the
  per-stream structure defined in the next section (e.g. `core/`,
  per-expansion folders such as `computer-vision-sdk/` /
  `rocm-ds/`, `extras/`, `whl/`, `whl-next/`,
  `pytorch/`, etc. — every expansion is a top-level peer; there is
  no `expansions/` grouping folder).
- The bare `repo.amd.com` domain serves as a **navigation landing
  page** plus the two singleton folders (`amdrepos/` and `rocm/`). It
  must list and link to every stream subdomain so a user starting at
  `https://repo.amd.com/` can click through to any active stream.
  Beyond the landing page and the two singleton folders, it is **not**
  required to serve any artifact content directly —
  `repo.amd.com/<stream>/...` paths are not part of the contract, and
  canonical artifact URLs live exclusively on the stream subdomains.
- The tier repo packages (`amdrocm-repo-stable`,
  `amdrocm-repo-stablerc`, `amdrocm-repo-nightly`, `amdrocm-repo-dev`;
  see *Repository Package* section) each install `baseurl` / APT
  sources pointing at the subset of stream subdomains they cover,
  selected by the user's active stream variable (e.g.
  `https://${amdrocm_release_stream}.repo.amd.com/...`). The repo
  packages themselves are downloaded from
  `https://repo.amd.com/amdrepos/`.
- TLS certificates must cover every stream subdomain plus the bare
  `repo.amd.com` (wildcard `*.repo.amd.com` plus the apex is
  acceptable).
- Reserved future subdomains (`ltsrc`, `lts`) must resolve before
  content is published, even if they initially serve an empty index,
  so that repo-package definitions referencing them do not break.

## Repository Structure

Hosting splits into three layers, each with its own structure:

1. The **bare `repo.amd.com`** domain (landing page + singletons).
1. The **stream subdomains** `<stream>.repo.amd.com` for `<stream>` in
   `{dev, nightly, weekly, rc, stable, ltsrc, lts}` — all share an
   identical folder tree.
1. The **`archives.repo.amd.com`** subdomain (retention only).

### Structure on `repo.amd.com` (bare domain)

The bare domain hosts only the navigation landing page and the two
singleton folders. No per-stream artifact content lives here.

- **(landing page)** — `https://repo.amd.com/` — links to every
  active stream subdomain and the archives subdomain.

- **amdrepos/** *(singleton — `https://repo.amd.com/amdrepos/`)*

  - **per-distro subdirectories** — one folder per supported distro
    (e.g. `ubuntu2404/`, `ubuntu2204/`, `rhel9/`, `rhel10/`,
    `sles16/`, `azurelinux3/`). Each distro ships its own copy of
    every tier package (`amdrocm-repo-stable`, `amdrocm-repo-stablerc`,
    `amdrocm-repo-nightly`, `amdrocm-repo-dev`) because repo file
    install paths and URL suffixes are distro-family specific
    (rpm-family → `/etc/yum.repos.d/`, deb-family →
    `/etc/apt/sources.list.d/`).
  - **gpg/** — public signing keys served alongside the repo
    packages. Each tier package references the key by URL so
    `dnf` / `apt-key` can fetch and pin it on install. Keys are
    rotated through repo-package upgrades; the previous key is kept
    for one release cycle to allow non-broken upgrades.
  - See Repository Package section for contents and the per-tier
    install commands.

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
  - `weekly/` → `https://weekly.repo.amd.com/rocm/`
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
(`dev`, `nightly`, `weekly`, `rc`, `stable`, `ltsrc`, `lts`).
Stream-specific specializations (per-stream content variants,
retention) are captured in *Per-stream specializations* further down.

- **amdgpu/** *(reserved for future use; same tree, future GPU driver
  artifacts)*
- **rocm/**
  - **whl/** — **central** PEP 503 simple index for the entire
    stream — **backward-compatible / all-arch** variant. A single
    `whl/` serves wheels from every wheel-producing area in the
    stream (`core/`, every top-level expansion, wheel-producing extras,
    `pytorch/`, `jax/`, `onnx-runtime/`). `pip install rocm` (or `pip install torch`) against this index pulls in
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

    - tarball — Linux archive format, **primarily `.tar.gz`**.

    - zip — **Windows only**; the `.zip` archive is the Windows
      equivalent of the Linux tarball and is not produced for Linux
      distros. A single Windows OS variant agnostic `.zip` is published in
      `zip/` as one build covers all supported Windows
      versions.

    - windows-installers — Windows installer artifacts (`.exe`,
      `.msi`) for setups that don't unpack the `zip` archive.
      Organized into **one subfolder per supported Windows OS
      version**, using the commonly used short names. Supported
      versions:

      - `win11/` — Windows 11 (client)
      - `server2022/` — Windows Server 2022
      - `server2025/` — Windows Server 2025

      Windows 10 is **not** supported and has no folder. Additional
      OS versions follow the same flat short-name convention
      (`winNN/` for client, `serverYYYY/` for server) and are added
      as they become supported. Each OS folder contains the
      `.exe` / `.msi` artifacts for that OS; per-OS variants (debug,
      etc.) follow the same `<os>-<variant>/` sibling-folder rule
      used for Linux distros (e.g. `win11-debug/`) if and when they
      ship.

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

      - **Layout under `packages/` is intentionally left to the
        publishing implementation.** This RFC does **not** fix whether
        packages are exposed as per-distro folders (`<distro>-*`) or
        grouped by package family (`deb/`, `rpm/`), nor whether the
        OS/distro profile is encoded in the directory tree at all or
        surfaced only through the aggregated package index. Any of
        these realizations is acceptable as long as the normative
        requirements below hold; the goal is to keep the on-disk scheme
        flexible so it can evolve with the packaging tooling.

        The only **normative** requirements are:

        - **Standard is the default.** A user who adds the standard
          repo gets release builds by default; no extra opt-in is
          needed for the ordinary install path.
        - **Variant separation.** Non-standard build variants — `asan`
          today, and reserved future `rpath` / `debug` variants — must
          be separable so that a package manager pointed only at the
          standard repo never accidentally resolves an ASAN or debug
          build (e.g. `dnf install rocm` must not pull an ASAN build).
          *How* that separation is realized — sibling folders, a
          distinct index, or a disabled-by-default repo stanza such as
          the `amdrocm-stable-rpath` stanza inside `amdrocm-repo-stable`
          — is left to the implementation.
        - **Defined variant and distro sets.** Build variants follow
          the RFC0009 Repository Layout rule
          (`Package-type = standard, asan, future variant`), and the
          supported distro identifiers (`debian12`, `ubuntu2204`,
          `ubuntu2404`, `rhel8`, `rhel9`, `rhel10`, `sles15`, `azl3`)
          are defined by RFC0009.

        *Illustrative only* — one valid per-distro realization for
        RHEL 10:

        ```
        packages/
          rhel10/         # standard release packages
          rhel10-asan/    # ASAN-instrumented packages
        ```

        A `deb/` + `rpm/` grouping that carries the per-OS profiles in
        the aggregated index is an equally acceptable realization. The
        example is not prescriptive.

  - *(no separate top-level `windows/` folder — Windows artifacts
    live alongside their Linux siblings inside each component, e.g.
    `core/zip`, `core/windows-installers`.)*

  - **pytorch/** — `whl/` + `whl-next/` (same rule as `core/`).

  - **jax/** *(follows the same artifact rules as **pytorch**)*

  - **onnx-runtime/** *(follows the same artifact rules as **pytorch**)*

  - **[expansion-name]/** *(top-level peer to `core/`,
    `pytorch/`, `jax/`, `onnx-runtime/`)* — each expansion (e.g.
    `rocm-ds/`, future SDKs built on the ROCm Core SDK) is published
    as its **own top-level folder** under each stream subdomain's
    `rocm/` tree, **not** grouped under an `expansions/` parent. This
    matches the placement rule already used by the
    third-party AI forks: every component with an independent
    release cadence gets a top-level peer so its cadence is visible
    in the URL and there is no intermediate namespace to navigate.

    - **No `expansions/` parent folder.** A grouping folder was
      considered and rejected because (a) it adds a path segment
      that carries no information, (b) it hides each expansion's
      cadence behind a shared name, and (c) it forces a directory
      rename if a component is later reclassified between
      "expansion" and something else.
    - **Folder contents per expansion:** the same artifact set as
      `core/` — `tarball`, `zip` (with per-OS subfolders mirroring
      `core/zip/` when needed), `windows-installers/` (with per-OS
      subfolders `win11/`, `server2022/`, `server2025/`, same rule
      as `core/windows-installers/`),
      `linux-installers`, `whl/` + `whl-next/` (per the central
      indices; internal layout implementation-defined), and
      `packages/` (same implementation-defined layout as
      `core/packages/` — see the `packages/` rules above).
    - **Cadence and pinning:** each expansion picks its own cadence
      and ROCm Core SDK pinning rule; that rule belongs in the
      per-expansion RFC, not here. The forthcoming Computer Vision SDK
      ([PR #5631](https://github.com/ROCm/TheRock/pull/5631)) is an
      example of an expansion that already owns its own RFC.
    - **Naming and alphabetization:** expansion folder names use the
      project's canonical short name (`rocm-ds/`,
      `computer-vision-sdk/`, …). Browsing the per-stream `rocm/`
      tree, expansions and forks sort alphabetically alongside
      `core/`; there is no enforced grouping by category.

  - **extras/** — projects released independently per ROCm major
    version. **Single folder, no `-<ROCm-major>` suffix** — the ROCm
    major lives in the *package name* (`amdrocm<major>-<project>`),
    not the folder path, so every major's packages coexist in one
    `extras/` tree with a single origin per package.

    - **rvs/** — tarball, packages

    - **rocoptiq/** — tarball, whl, whl-next, packages

    - **omnistat/** — whl, whl-next

    *Rationale (revised per PR #4414 review):*

    - **Decision (revised per
      [PR #4414 review](https://github.com/ROCm/TheRock/pull/4414#discussion_r3348478551)):**
      the earlier `extras-<ROCm-major>/` layout is dropped. Splitting
      by major would have forced the same logical project (e.g.
      `rvs`) to appear under multiple origins (`extras-7/rvs/`,
      `extras-8/rvs/`), which **breaks the aggregated index** — each
      package in that index may come from only one origin, and
      multi-origin support adds churn and filename-conflict risk.
      Because each major's artifacts are already distinguishable
      without the folder — native package names embed the major
      (`amdrocm7-rvs` vs `amdrocm8-rvs`), and Python wheels carry it
      in the `+rocm<major>` local-version tag on a single-named wheel
      (`rocm-rvs-…+rocm7` vs `rocm-rvs-…+rocm8`) — a single `extras/`
      folder holds every major's packages without collision, each
      with a single origin.

    - **Per-project subfolders retained.** Inside `extras/`, each
      project keeps its own folder (`extras/rvs/`,
      `extras/rocoptiq/`) so S3 bucket permissions can still be
      granted per project/group — that motivation is unchanged. Both
      majors' packages for a project live in that one project folder
      (e.g. `stable.repo.amd.com/rocm/extras/rvs/` holds
      `amdrocm7-rvs` and `amdrocm10-rvs`).

    - **Aggregated index — every major is included.** All extras in
      `extras/` feed the central `whl/` / `whl-next/` indices,
      regardless of ROCm major. Two options were weighed in review:
      (a) a single `extras/` folder with every major in the index, or
      (b) only the latest major in the index, older majors excluded.
      **Option (a) was chosen:** every major's packages are distinct
      index entries with no conflict, so **every major keeps the
      one-line install** resolving its full dependency chain from a
      single `--index-url`. The major is distinguished differently per
      format, but both coexist in the one index:

      - *Native:* the package name embeds the major
        (`dnf install amdrocm7-rvs`, `dnf install amdrocm10-rvs`),
        so each major is a distinct package.
      - *Python wheels:* the package name **drops** the major
        (`rocm-rvs`); every major is the *same* PEP 503 project page
        with version entries distinguished by the `+rocm<major>`
        local-version tag (`rocm-rvs-1.2.0+rocm7`,
        `rocm-rvs-1.5.0+rocm10`). `pip install rocm-rvs` resolves to
        the build whose `Requires-Dist: rocm[core]` range matches the
        installed ROCm Core; an explicit major is pinned with
        `pip install "rocm-rvs==1.2.0+rocm7"` or a constraints file.

      The exclusion fallback in option (b) is unnecessary.

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
  dry-runs; **not** for end users. No `rc`-style exclusivity applies.
- **`nightly.repo.amd.com`** — Retention: 120 nightly. Promoted
  builds from the develop branch (`dev` builds that passed the
  promotion gate).
  - **Repo layout:** `nightly.repo.amd.com/rocm/core/packages/<distro>/`
    is a **flat package repository**, identical in structure to
    `stable` and `rc`. All retained nightly versions are co-resident in
    that single flat tree (multiple `amdrocm-core-<NNNN>` package
    versions side-by-side), so the `amdrocm-repo-nightly` package
    works without a date-stamped subpath. Retention is enforced by
    pruning *old package versions out of the flat tree*, not by
    deleting date-stamped subfolders.
  - **Relationship to `dev`:** the `30 dev / 120 nightly` retention
    in earlier drafts is now split across the two streams — `dev`
    keeps 30 days of per-commit builds; `nightly` keeps 120 days of
    promoted builds. Both are tagged in package metadata so users can
    filter via `dnf --showduplicates` and install a specific version.
- **`weekly.repo.amd.com`** — Retention: 52 weeks (one calendar year of
  weekly promotions). Promotion source: the latest `nightly` build at
  the cut time. Cadence: **one promotion per ISO calendar week**, cut
  Monday 00:00 UTC.
  - **Repo layout:** `weekly.repo.amd.com/rocm/core/packages/<distro>/`
    is a **flat package repository**, identical in structure to
    `nightly`, `stable`, and `rc`. All retained weekly versions are
    co-resident in the flat tree. Retention is enforced by pruning
    old package versions out of the flat tree.
  - **Relationship to `nightly`:** every `weekly` artifact is a
    direct promotion of a specific `nightly`; the `nightly` source
    version is recorded in package metadata. No additional QA gate
    beyond the nightly promotion gate — `weekly` exists to give
    downstream CI a stable, longer-retention target with predictable
    cadence, not to add a quality bar.
  - **Intended consumers:** downstream CI pipelines, framework
    integrators, and partners who want one bump per week instead of
    one per night. Not a substitute for `rc`/`stable` for end users.
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
  `lts.repo.amd.com` layout.
- **`lts.repo.amd.com`** *(future)* — long-term-support releases.
  Adds a `YYYYMM/` subfolder under each artifact type, otherwise
  mirrors the `stable` layout.

### Structure on `archives.repo.amd.com`

`archives.repo.amd.com` is a **retention-only** subdomain, not a
release stream. Content is read-only: no new artifacts are published
directly here; everything arrives via migration from one of the
sources below.

**Sources of archived content:**

- **Legacy `repo.amd.com/rocm/`** — the non-production folder on the
  bare domain, migrated 6 months after `archives.repo.amd.com` goes
  live. Original URLs redirect to `https://archives.repo.amd.com/legacy-rocm/`
  preserving the existing path structure so historical documentation
  links continue to resolve.
- **`repo.radeon.com` (ROCm content)** — mirroring the historical
  ROCm content from `repo.radeon.com` into `archives.repo.amd.com`
  is **planned but deferred**. Scope (which subtrees), URL layout,
  redirect behavior, GPG key handling, and the migration timeline
  are **to be determined in a follow-up discussion** tied to the
  AMD GPU driver consolidation into `repo.amd.com` (see *Repository
  Package*). This RFC reserves the placement under
  `archives.repo.amd.com` for that content; the technical details
  are explicitly out of scope here.

`archives.repo.amd.com` holds **only pre-stream-model content** —
artifacts that predate the move to the per-stream subdomain release
model. Aged-out artifacts from the new `<stream>.repo.amd.com`
subdomains are **not** copied here; once a build falls outside its
stream's retention window it is removed. Reproducibility for the new
streams is handled by the per-stream retention windows themselves
(`stable` forever, `rc` 2 years, `nightly` 120 days, etc.), not by
the archive subdomain.

Layout is preserved historically per source and is not constrained by
this RFC's per-subdomain tree. The retention horizon is **indefinite**
— content is not removed once archived, only added.

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
per-expansion folders (`computer-vision-sdk/`,
`rocm-ds/`, …), any extras that ship wheels, `pytorch/`, `jax/`,
`onnx-runtime/`). There is no per-package central index
and no `pyindex/` wrapper. Centralizing the indices lets
`pip install rocm` (and `pip install torch`, `pip install jax`, etc.)
resolve cross-package dependencies in one resolution pass against a
single `--index-url`.

- **`whl/`** — **backward-compatible / all-arch** variant.
  `pip install rocm` (or `pip install torch`) pulls in **all** device
  extras automatically, matching the "it just works" behavior users
  expect from `pip install torch --index-url https://download.pytorch.org/whl/rocm7.2`. Larger download (~5.5 GB
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
> **No Python-side repo-setup package:** there is no
> `amdrocm-repo-*` Python package (or equivalent) and none is
> planned. The `amdrocm-repo-stable` / `amdrocm-repo-stablerc` /
> `amdrocm-repo-nightly` / `amdrocm-repo-dev` packages are
> **native (rpm/deb) repo-setup packages only** — they configure
> `yum` / `apt` sources and the gpg key, and have no role in the
> Python wheel install path. Users wire up `pip` by passing
> `--index-url` themselves (or by adding the index to their
> `pip.conf` / `requirements.txt`).

Both variants ship under every stream that publishes wheels (`dev`,
`nightly`, `weekly`, `rc`, `stable`; not `ltsrc`/`lts` until LTS
exists), and both are built from the same underlying wheel set —
`whl/` simply republishes the entry-point wheels (`rocm`, `torch`,
`torchvision`, …) with `device-all` added as an automatic requirement,
plus links to the unmodified device wheels in `whl-next/` so storage
is not duplicated.

Future direction: WheelNext (`uv pip install` with a wheel-variant
provider backed by `rocm-bootstrap`) is the long-term plan and will
eventually make `whl/` unnecessary — `whl-next/` becomes the sole
index once WheelNext is widely adopted. Until that lands, both
variants must coexist.

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
- **Streams:** published under `nightly/`, `weekly/`, `rc/`, and
  `stable/`. **Not** published under `dev/` (per-commit churn is not
  useful for fork consumers — the dev gate is too low and the build
  cost would be prohibitive). **Not** published under `ltsrc/` or
  `lts/` (long-term-support guarantees do not extend to third-party
  fork builds).
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
  Core SDK `<X.Y>` scheme and not the LTS `<YYYY.MM>` scheme.
- **Support model:** bug fixes for the ROCm-specific delta ship in the
  next stream promotion; we do not back-patch older third-party fork
  releases.
- **Coverage list:** `pytorch`, `jax`, `onnx-runtime`. New third-party
  forks added to `repo.amd.com` follow this same model by default.

## Repository Package

**Defined in a separate RFC** — see
`RFC00XX-Repository-Package.md`. That document is the source of
truth for the `amdrocm-repo-stable`, `amdrocm-repo-stablerc`,
`amdrocm-repo-nightly`, and `amdrocm-repo-dev` tier packages,
including their installed `.repo` / `.sources` filenames, default
enablement, GPG key handling, amdgpu driver pinning per stream,
the `Conflicts:` rule against legacy `amdgpu-install`, and the
deferred LTS placement.

The only repo-package facts this RFC needs to assert are
placement-related and already covered elsewhere in this document:

- **Publication folder.** The tier packages are published in the
  singleton `amdrepos/` folder on the bare domain
  (`https://repo.amd.com/amdrepos/<distro>/`) — see *Structure on
  `repo.amd.com` (bare domain)*. `amdrepos/` is not replicated
  under any stream subdomain; there is one canonical copy that
  serves all streams.
- **Stream-subdomain pointers.** Each tier package's installed
  `baseurl` / `URIs:` resolves to
  `https://<stream>.repo.amd.com/...` for the streams it covers
  — see *Stream Subdomains* for the subdomain contract.
- **On-disk coexistence.** Streams installed via different tier
  packages coexist because each Core SDK build lands in a
  stream-distinct, version-scoped path under `/opt/rocm/` — see
  *Install Locations (ROCm Core SDK)* in `RFC00XX-Repository-Package.md`.

All other repo-package mechanics (filename scheme, deb822 stanza
shape, default enablement table, rpath sibling, driver-pin churn
isolation, install commands) live in `RFC00XX-Repository-Package.md`
and are not duplicated here.
