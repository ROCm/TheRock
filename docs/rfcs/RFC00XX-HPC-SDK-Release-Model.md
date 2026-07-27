---
author: Saad Rahim (saadrahim)
created: 2026-06-08
modified: 2026-06-08
status: draft
---

# AMD HPC SDK release model (v2 — Core extra meta-package)

> **Scope change (2026-06-08).** The HPC SDK is no longer a separately
> released expansion with its own cadence, streams, and top-level
> `hpc-sdk/` folder. It is now built **along with ROCm Core**, shipped
> as an **extra meta-package** (`amdrocm-hpc`) that is **not** part of
> the default `amdrocm` meta-package. The independent-cadence model is
> preserved in `RFC00YY-HPC-SDK-Release-Model.md` for reference.

## Related RFCs

- **RFC0012** — ROCm software ecosystem package repository structure
  (stream subdomains, the `rocm/` tree layout, and the `amdrocm-repo`
  package family). The HPC SDK now rides on the Core SDK's existing
  placement rather than occupying its own peer folder.
- **RFC0009** — native packaging conventions (source of the
  `amdrocm<major>-<project>` naming family that HPC SDK component
  packages follow, the same as every other Core SDK component).

## Overview

The AMD HPC SDK is built and released **together with the ROCm Core
SDK**. Its components are built on the same release train, carry the
same version as Core, ship on the same streams, and install to the
same locations as any other Core SDK component.

The only thing that sets the HPC SDK apart from the rest of Core is
packaging: its components are grouped into a separate **`amdrocm-hpc`
meta-package** that is **not** pulled in by the default `amdrocm`
meta-package. Users who want the HPC SDK opt in by installing
`amdrocm-hpc`; a default ROCm install does not include it.

Repository hierarchy, stream subdomains, `amdrocm-repo` packaging,
install locations, and versioning are all inherited from the Core SDK
(RFC00XX / RFC0009) and are not redefined here.

## Versioning and cadence

- **Cadence:** the HPC SDK ships with every ROCm Core SDK release —
  it is built on the same train. There is no separate HPC SDK
  cadence, schedule, or promotion flow.
- **Versioning:** the HPC SDK and its components carry the **same
  version as the ROCm Core SDK release they are built in** (`<X.Y>`,
  e.g. `7.14`; patch releases `<X.Y>.N`, e.g. `7.14.1`). There is no
  separate HPC version number. This applies to packages only. The individual components still version their APIs and ABIs independently from the package version that follows the ROCm Core SDK.
- **Streams:** the HPC SDK is present on the same streams as Core
  (`dev`, `nightly`, `rc`, `stable`) by virtue of being built with
  Core. It does not define its own streams or retention rules.

## Repo placement

The HPC SDK is **not** a top-level peer and has **no** dedicated
`hpc-sdk/` folder or expansion path. Its component packages are
published with the rest of the Core SDK's packages under the Core
SDK's existing `core/` location on each stream subdomain. The
`amdrocm-repo-<stream>` packages that already configure Core also
surface the HPC SDK components — no additional repo wiring is needed.

## Install location

No special install location. HPC SDK components install to the **same
locations as every other ROCm Core SDK component** (the standard Core
SDK install prefix). There is no `hpc-` prefix or separate directory
tree.

## Meta-package

The HPC SDK ships a single meta-package, **`amdrocm-hpc`**, built and
versioned with the Core SDK release. It depends on every HPC SDK
component at the exact version shipped in that Core release.

- It is a **standalone, opt-in** meta-package — the default `amdrocm`
  meta-package does **not** depend on it, so a default ROCm install
  does not pull in the HPC SDK.
- Installing `amdrocm-hpc` adds the HPC SDK components on top of the
  matching Core SDK install:

  ```
  yum install amdrocm-hpc
  apt install amdrocm-hpc
  ```

- Because HPC components ship with Core, `amdrocm-hpc`'s component
  dependencies resolve against the Core SDK version already on the
  system (or are pulled from the same stream).
- Available in both rpm and deb formats for every supported distro.
- Component list is owned by the HPC SDK team and reviewed each
  release so new components are added to the meta-package
  automatically.

## Component versioning

HPC SDK component packages follow the **same versioning and naming as
all other Core SDK components** — there is no HPC-specific scheme.
Naming follows the Core SDK convention
(`amdrocm-<component>[-<target>]-<version>-<pkgrel>.<arch>.<ext>`,
e.g. `amdrocm-rochpcg-gfx1152-7.14.0-1.x86_64.rpm`), with `<version>`
matching the Core SDK release.

## Components

Initial release (ROCm 7.14):

- rocALUTION
- hipTensor

hipFort is planned for a **future** release. Other HPC components
(rocHPCG, rocHPL, miniHPL, miniHPCG) are **not in scope** for the
initial release.
