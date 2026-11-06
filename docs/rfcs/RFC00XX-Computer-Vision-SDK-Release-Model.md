---
author: Marco Grond (marco-grond)
created: 2026-06-04
modified: 2026-10-30
status: draft
---

# RFC00XX: AMD Computer Vision SDK release model

## Related RFCs

- **RFC0009** — native packaging conventions (source of the
  `amdrocm<major>-<project>` naming family that HPC SDK component
  packages follow).
- **RFC0012** — ROCm software ecosystem package repository structure
  (defines the stream subdomains, the `rocm/` tree layout, and the
  `amdrocm-repo` package family that this RFC builds on). The Vision
  SDK is hosted as a top-level peer to `core/`, `pytorch/`, `jax/`,
  and `onnx-runtime/` under each stream subdomain's `rocm/` tree.
- **RFC00XX** - ROCm SDK archive packaging requirements. The Vision
  SDK should follow the same structure and the Vision SDK packaging
  should follow a similar naming convention and directory layout. 
- **RFC00XX** - HPC SDK dependent on the ROCm Core SDK (similar
  ticket for the HPC SDK - the computer vision SDK should follow
  the same structure and requirements)


## Overview

The AMD Computer Vision SDK is a ROCm expansion that is built **against
a released ROCm Core SDK** from its own repository (`rocm-vision`). It
does not build the Core SDK itself: it consumes an already-published
Core SDK (as a tarball or installed packages), builds the CV components
on top of it, carries the same version as that Core SDK, ships on the
same streams, and installs to the same locations as any other Core SDK
component.

The only thing that sets the CV SDK apart from the rest of Core is
packaging: its components are grouped into a separate **`amdrocm-vision`
meta-package** that is **not** pulled in by the default `amdrocm`
meta-package. Users who want the CV SDK opt in by installing
`amdrocm-vision`; a default ROCm install does not include it.

Repository hierarchy, stream subdomains, `amdrocm-repo` packaging,
install locations, and versioning are all inherited from the Core SDK
(RFC0009 / RFC0012 / RFC0013) and are not redefined here.

## Versioning and cadence

- **Cadence:** the CV SDK ships for every ROCm Core SDK release — it is
  rebuilt against each Core SDK from its own `rocm-vision` repository and
  nightly. The CV SDK is not bound to the Core SDK cadence, schedule, or
  promotion flow, but tracks it.
- **Versioning:** the CV SDK and its components carry the **same
  version as the ROCm Core SDK release they are built in** (`<X.Y>`,
  e.g. `10.1`; patch releases `<X.Y>.N`, e.g. `10.1.1`). There is no
  separate CV version number. This applies to packages only. The
  individual components still version their APIs and ABIs independently
  from the package version that follows the ROCm Core SDK.
- **Streams:** the CV SDK is published to the same streams as Core
  (`dev`, `nightly`, `weekly`, `rc`, `stable`), building against the Core
  SDK on each stream. It does not define its own streams or retention
  rules.

## Repo placement

The CV SDK has an independent repository from `TheRock` under the
`rocm` GitHub organization, namely `rocm-vision`. Per RFC0012, both its
component packages and its tarball are published under a **top-level
`computer-vision-sdk/` subtree**, a peer of `core/` (not nested under
`core/` and not in `extras/`) within each stream subdomain's `rocm/`
tree. The `amdrocm-repo-<stream>` packages that already configure Core
also surface the CV SDK components — no additional repo wiring is needed.

## Install location

No special install location. CV SDK components install to the **same
locations as every other ROCm Core SDK component** (the standard Core
SDK install prefix, e.g. the `core-<major.minor>` tree). There is no
`vision-` prefix or separate directory tree.

One detail follows from the CV libraries' dependencies rather than from
a CV-specific layout: **bundled runtime deps** that the CV libraries
need but Core does not ship (e.g. protobuf, libjpeg-turbo, lmdb,
libsndfile, libtar) install into the **existing** `lib/rocm_sysdeps/lib/`
tree — the same directory Core already uses for zlib/bzip2/etc. — with
private SONAMEs so they do not collide with host copies. They ship in a
dedicated `amdrocm-vision-sysdeps` package, not a new directory tree.

## Meta-package

The CV SDK ships a **runtime meta-package, `amdrocm-vision`**, built and
versioned with the Core SDK release, depending on every CV SDK runtime
component at the matching Core release. The component dependency is a
**compatible-version constraint**, not an exact pin, so a component
post-release (`<version>.postN`) — permitted by
`docs/packaging/versioning.md` without republishing the package set —
still satisfies the meta-package. Mirroring
Core's runtime/devel/test split, it is accompanied by two sibling
meta-packages — **`amdrocm-vision-sdk`** (development: headers and CMake
config) and **`amdrocm-vision-tests`** (test content). Like Core's
component packages, each meta-package name carries the ROCm
`<major.minor>` suffix (e.g. `amdrocm-vision10.2`,
`amdrocm-vision-sdk10.2`) so two Core releases can be installed
side-by-side.

- They are **standalone, opt-in** meta-packages — the default `amdrocm`
  meta-package does **not** depend on `amdrocm-vision`, so a default
  ROCm install does not pull in the CV SDK.

- Installing `amdrocm-vision` adds the CV SDK components on top of
  the matching Core SDK install. Because each meta-package name carries
  the ROCm `<major.minor>` suffix, use the **version-qualified selector**
  to target a specific Core release rather than the unqualified name,
  which resolves to the latest available and cannot distinguish
  side-by-side releases:

  ```
  yum install amdrocm-vision10.2
  apt install amdrocm-vision10.2
  ```

- CV component packages declare **explicit versioned dependencies on the
  Core SDK packages** they build against — e.g. `amdrocm-runtime10.2`,
  `amdrocm-rpp10.2`, `amdrocm-rocdecode10.2` with a `>= <version>`
  lower bound tied to that release — so the resolver pulls (or requires)
  the matching Core version already on the system, not merely whatever
  Core happens to be latest.

- Delivered today as Linux **deb, rpm, and tgz** packages plus Python
  **wheels**. RFC0012 expects expansions to also publish Windows
  zip/installer artifacts; the first CV release takes a **documented
  exception** to that requirement because the CV build is Linux-only.
  Windows **zip/exe/msi** and their target matrix are deferred until
  Windows support lands and will be specified then.

- Component list is owned by the CV SDK team and reviewed each
  release so new components are added to the meta-package
  automatically.

## Tarball

The CV SDK is also published as a **tarball**, in the
**`computer-vision-sdk/tarball/`** location (the peer-of-`core/` subtree
from the Repo placement section).

**Definition.** The CV SDK tarball is a **Vision-only overlay** on top
of the matching ROCm Core SDK tarball. It contains the CV SDK components
(the components covered by `amdrocm-vision`) laid out in the same
directory structure as the Core SDK, so it extracts cleanly into a Core
SDK prefix. It does **not** duplicate the Core SDK contents; the CV
build consumes the Core SDK rather than repackaging it.

This matches the package path, where `amdrocm-vision` is a thin opt-in
meta-package on top of an existing Core SDK install. The tarball is the
same relationship without a dependency resolver: the user extracts the
matching Core SDK tarball first, then extracts the CV overlay into the
same prefix.

> **Note.** An alternative design is a *self-contained superset* tarball
> that bundles the full Core SDK plus CV components, so a single archive
> is a complete install. That removes the "extract Core first" step but
> requires the CV release to repackage and re-publish bit-identical Core
> content for every target/stream — which the current `rocm-vision`
> build does not do (it consumes Core, it does not rebuild or
> re-emit it). If AMD wants the superset form, producing it belongs to
> the Core/TheRock release train, not the `rocm-vision` repo. This RFC
> specifies the overlay as the deliverable and leaves the superset as a
> possible future addition.

**Consequences:**

- The CV overlay tarball **requires** the matching Core SDK tarball — it
  is a companion, not an alternative. Extracting the overlay without
  Core present yields CV libraries with unresolved Core dependencies.

- The CV overlay and the Core SDK tarball carry the **same version**;
  the overlay is only valid against the Core SDK of that exact version.

- The CV overlay tarball is produced for the **same targets, OSes, and
  streams** as the Core SDK tarball, with no CV-specific matrix.

- Naming follows the Core SDK tarball convention with a **`vision`**
  discriminator, naming the archive after what it actually contains:

  ```
  therock-dist-vision-linux-<target>-<version>.tar.gz   # Vision overlay (extract onto Core)
  therock-dist-linux-<target>-<version>.tar.gz          # Core SDK
  ```

## Component versioning

CV SDK component packages follow the **same versioning and naming as
all other Core SDK components** — there is no CV-specific scheme. The
name suffix is the **ROCm `<major.minor>` release**, exactly as Core
names `amdrocm-runtime<major.minor>` — not the component's own API/ABI
version:

```
amdrocm-<component><major.minor>-<version>-<pkgrel>.<arch>.<ext>
e.g. amdrocm-rocal10.2-10.2.0-1.x86_64.rpm
```

`<version>` matches the Core SDK, and by extension the CV SDK, release.
The component's independent API/ABI version is tracked in the library
SONAME, not the package name.

**Architecture coverage.** CV component packages are **architecture-
agnostic fat binaries** — a single package per component carries code
objects for every supported gfx target (the HIP compiler bundles them),
so there is **no per-target (`-gfx<target>`) package split**. If the
Core SDK moves to per-target component packages, the CV SDK follows that
convention; until then the CV packages ship one `.so` per component
covering all targets.

## Components

First release (pinned to ROCm 10.2):

- MiVisionX
- rocAL
- rocCV
- rocPyDecode
