
author: Marco Grond (marco-grond)
created: 2026-06-04
modified: 2026-06-04
status: draft
---

# AMD Computer Vision SDK release model

## Related RFCs

- **RFC00XX** — ROCm software ecosystem package repository structure
  (defines the stream subdomains, the `rocm/` tree layout, and the
  `amdrocm-repo` package family that this RFC builds on). The HPC SDK
  is hosted as a top-level peer to `core/`, `pytorch/`, `jax/`, and
  `onnx-runtime/` under each stream subdomain's `rocm/` tree.
- **RFC0009** — native packaging conventions (source of the
  `amdrocm<major>-<project>` naming family that HPC SDK component
  packages follow).
- **RFC00XX** — end-user projects independent release lifecycle
  (general model for components shipped on a cadence decoupled from
  ROCm; HPC SDK is the first umbrella release that aggregates such
  components under a single date-based version).
- **RFC00XX** - HPC SDK dependent on the ROCm Core SDK (similar
  ticket for the HPC SDK - the computer vision SDK should follow
  the same structure and requirements)

## Overview

The AMD Computer Vision SDK is a ROCm expansion that needs to be
released on its own cadence, decoupled from ROCm Core SDK releases.
This RFC defines the versioning, stream layout, ROCm pinning rule,
meta-packaging, and release flow for the Computer Vision SDK.

Repository hierarchy, stream subdomains, `amdrocm-repo` packaging,
and concurrent stream installation are defined in RFC00XX and are
not redefined here.

## Versioning and pinning

- **Cadence:** approximately 8 releases per year. Not every ROCm
  release has a corresponding Computer Vision SDK release, although
  the Computer Vision SDK will attempt to coincide with ROCm releases.
- **Versioning:** date-based, `YYYY.MM` (e.g. `2026.06`). Versioning is
  intentionally independent of ROCm version numbers to make the
  decoupling explicit and avoid mix-and-match confusion.
- **ROCm pinning:** each Computer Vision SDK release is pinned to exactly
  one ROCm Core SDK version. Pinning follows the stream:
  - stable Computer Vision SDK pins to a stable ROCm Core SDK release.
  - LTS Computer Vision SDK (when available) pins to an LTS ROCm Core
    SDK release.
  Both streams coexist once LTS exists; a stable Computer Vision SDK
  release is always available regardless of LTS status.
- **Patch releases:** bug fixes are delivered as patch releases
  (`YYYY.MM.N`) against the pinned ROCm version, on the same model as
  LTS maintenance.

## Streams and layout

- **Streams (required in v1):** `dev`, `nightly`, `rc`, and
  `stable` subdomains are all required at launch. `ltsrc` and `lts`
  are reserved for future use, aligned with ROCm LTS. Per-stream
  retention and QA rules follow the global *Per-stream
  specializations* in RFC00XX — the Computer Vision SDK does not
  redefine them. `dev` builds are per-commit / pre-nightly developer
  builds against the develop ROCm branch with no QA gate
  (developer-testing and infra dry-runs only, not end users);
  `nightly` builds against the develop ROCm branch on the nightly
  schedule; `rc` is tested by QA and mirrors `stable/` layout;
  `stable` GA releases are pinned to a stable ROCm version;
  `ltsrc`/`lts` (future) pin to LTS ROCm versions.
- **Layout:** published as
  `<stream>.repo.amd.com/rocm/cv-sdk/<YYYY.MM>/` for `rc`/`stable`
  (and future `ltsrc`/`lts`), with release metadata recording the
  pinned ROCm version. `dev` and `nightly` builds use a date-stamped
  folder directly under `cv-sdk/` —
  `dev.repo.amd.com/rocm/cv-sdk/<YYYYMMDD-sha>/` and
  `nightly.repo.amd.com/rocm/cv-sdk/<YYYYMMDD>/` — with no nested
  `<stream>/` path segment (the stream subdomain already provides
  the stream selector).
- **Peer placement:** `cv-sdk/` is a top-level peer to `core/`,
  `pytorch/`, `jax/`, and `onnx-runtime/` under each stream
  subdomain's `rocm/` tree. It is **not** placed under
  `expansions/`. This makes the independent cadence visible in the
  URL structure.

## Meta-package

Each release ships an installable umbrella package
`amdrocm-cv-YYYY.MM` (rpm and deb) that depends on every Computer
Vision SDK component at the versions in that release, plus a hard
dependency on the pinned ROCm Core SDK version. Installing the
meta-package alone yields a complete, version-consistent Computer
Vision SDK on each supported distro:

```
yum install amdrocm-cv-2026.06
apt install amdrocm-cv-2026.06
```

Requirements:

- Pulls in all Computer Vision SDK components at the exact versions
  shipped in that `YYYY.MM` release.
- Declares a hard dependency on the pinned ROCm Core SDK version per
  the stream-matched pinning rule.
- Patch releases (`YYYY.MM.N`) update the meta-package's component
  version pins without changing the meta-package name.
- Available in both rpm and deb formats for every supported distro.
- Component list is owned by the Computer Vision SDK team and
  reviewed each release so new components are added to the
  meta-package automatically.

## Component versioning

Individual packages inside the Computer Vision SDK carry their own
component version number in `YYYY.MM.<patch>` format — the same
date-based scheme as the umbrella release, with a per-component
patch field. The leading `YYYY.MM` always matches the umbrella
release the component shipped in; `<patch>` increments
independently per component across patch releases.

Naming follows the Core SDK component-package convention
(`amdrocm-<component>[-<target>]-<version>-<pkgrel>.<arch>.<ext>`,
e.g. `amdrocm-sparse-gfx1152-7.13.0-2.x86_64.rpm`). For the
Computer Vision SDK components the `<version>` field is
`YYYY.MM.<patch>` — e.g.
`amdrocm-rocal-gfx1152-2026.06.0-1.x86_64.rpm` shipped in the
`amdrocm-cv-2026.06` umbrella, then
`amdrocm-rocal-gfx1152-2026.06.1-1.x86_64.rpm` in the
`2026.06.1` patch release. The umbrella `amdrocm-cv-YYYY.MM[.N]`
meta-package pins each component to its exact shipped
`YYYY.MM.<patch>` version.

## Components

First release (pinned to ROCm 7.14):

- MiVisionX
- rocAL
- rocCV
