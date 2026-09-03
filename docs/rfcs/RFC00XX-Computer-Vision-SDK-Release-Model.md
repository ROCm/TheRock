---
author: Marco Grond (marco-grond)
created: 2026-06-04
modified: 2026-08-27
status: draft
---

# RFC00XX: AMD Computer Vision SDK release model

## Related RFCs

- **RFC0012** — ROCm software ecosystem package repository structure
  (defines the stream subdomains, the `rocm/` tree layout, and the
  `amdrocm-repo` package family that this RFC builds on). The Vision
  SDK is hosted as a top-level peer to `core/`, `pytorch/`, `jax/`,
  and `onnx-runtime/` under each stream subdomain's `rocm/` tree.
- **RFC0009** — native packaging conventions (source of the
  `amdrocm<major>-<project>` naming family that HPC SDK component
  packages follow).
- **RFC0014** - HPC SDK dependent on the ROCm Core SDK (similar
  ticket for the HPC SDK - the computer vision SDK should follow
  the same structure and requirements)

## Overview

The AMD Computer Vision SDK is a ROCm expansion which is built and
release ***together with the ROCm Core SDK and the HPC SDK**. Its
components are built on the release train, carry the same version
as the Core SDK, ship on the same streams, and installs to the same
locations as any other Core SDK component.

The only thing that sets the CV SDK apart from the rest of Core is
packaging: its components are grouped into a separate **`amdrocm-vision`
meta-package** that is **not** pulled in by the default `amdrocm`
meta-package. Users who want the CV SDK opt in by installing
`amdrocm-vision`; a default ROCm install does not include it.

Repository hierarchy, stream subdomains, `amdrocm-repo` packaging,
install locations, and versioning are all inherited from the Core SDK
(RFC0012 / RFC0009) and are not redefined here.

## Versioning and cadence

- **Cadence:** the CV SDK ships with every ROCm Core SDK release —
  it is built on the same train. There is no separate CV SDK
  cadence, schedule, or promotion flow.
- **Versioning:** the CV SDK and its components carry the **same
  version as the ROCm Core SDK release they are built in** (`<X.Y>`,
  e.g. `10.1`; patch releases `<X.Y>.N`, e.g. `10.1.1`). There is no
  separate CV version number. This applies to packages only. The
  individual components still version their APIs and ABIs independently
  from the package version that follows the ROCm Core SDK.
- **Streams:** the CV SDK is present on the same streams as Core
  (`dev`, `nightly`, `rc`, `stable`) by virtue of being built with
  Core. It does not define its own streams or retention rules.

## Repo placement

The CV SDK has an independent repository from `TheRock` under the
`rocm` GitHub organization, namely `rocm-vision`. Both its component
packages and its tarball are published under the Core SDK's existing
`core/` location on each stream subdomain, alongside the equivalent
Core SDK artifacts. The `amdrocm-repo-<stream>` packages that already
configure Core also surface the CV SDK components — no additional
repo wiring is needed.

## Install location

No special install location. CV SDK components install to the **same
locations as every other ROCm Core SDK component** (the standard Core
SDK install prefix). There is no `vision-` prefix or separate directory
tree.

## Meta-package

The CV SDK ships a single meta-package, **`amdrocm-vision`**, built
and versioned with the Core SDK release. It depends on every CV SDK
component at the exact version shipped in that Core release.

- It is a **standalone, opt-in** meta-package — the default `amdrocm`
  meta-package does **not** depend on it, so a default ROCm install
  does not pull in the CV SDK.

- Installing `amdrocm-vision` adds the CV SDK components on top of
  the matching Core SDK install:

  ```
  yum install amdrocm-vision
  apt install amdrocm-vision
  ```

- Because CV components ship with Core, `amdrocm-vision`'s component
  dependencies resolve against the Core SDK version already on the
  system (or are pulled from the same stream).

- Available in both rpm and deb formats for every supported distro.

- Component list is owned by the CV SDK team and reviewed each
  release so new components are added to the meta-package
  automatically.

## Tarball

The CV SDK is also published as a **tarball**, in the Core SDK's
existing **`core/tarball/`** location alongside the Core SDK tarball.
It does not get a separate folder — the two archives are distinguished
by name, not by path.

**Definition.** The CV SDK tarball is a **superset of the ROCm Core
SDK tarball**. It contains:

- every component **exclusive to the CV SDK** (the components covered
  by the `amdrocm-vision` meta-package), **and**
- **every component of the ROCm Core SDK** — the full contents of the
  matching Core SDK tarball.

It is therefore **self-contained**: extracting the CV SDK tarball
yields a complete, working ROCm Core SDK install plus the CV SDK
components. It is not an overlay or an add-on archive, and it does not
require the Core SDK tarball to be extracted first.

This differs deliberately from the package path, where `amdrocm-vision`
is a thin opt-in meta-package that depends on an existing Core SDK
install. Tarballs have no dependency resolver, so the only way to
deliver a usable CV SDK by tarball is to ship the full stack.

**Consequences:**

- The Core SDK tarball and the CV SDK tarball are **alternatives, not
  companions** — a user picks one. Extracting both into the same
  prefix is redundant, not additive.

- The two tarballs carry the **same version** and are produced from
  the same build, so the Core SDK content in the CV tarball is
  bit-identical to the standalone Core SDK tarball of that release.

- The CV SDK tarball is produced for the **same targets, OSes, and
  streams** as the Core SDK tarball, with no CV-specific matrix.

- Naming follows the Core SDK tarball convention with a **`core+vision`**
  discriminator, naming the archive after what it actually contains:

  ```
  therock-dist-core+vision-linux-<target>-<version>.tar.gz
  therock-dist-linux-<target>-<version>.tar.gz          # Core SDK only
  ```

  The `core+vision` form is deliberate — it states on the tin that the
  archive is Core **plus** CV, so a user who downloads it from
  `core/tarball/` alongside the Core tarball cannot mistake it for a
  CV-only add-on.

## Component versioning

CV SDK component packages follow the **same versioning and naming as
all other Core SDK components** — there is no CV-specific scheme.
Naming follows the Core SDK convention
(`amdrocm-<component>[-<target>]-<version>-<pkgrel>.<arch>.<ext>`,
e.g. `amdrocm-rocal-gfx1152-10.1.0-1.x86_64.rpm`), with `<version>`
matching the Core SDK, and by extension CV SDK, release.

## Components

First release (pinned to ROCm 10.1):

- MiVisionX
- rocAL
- rocCV
- rocPyDecode
