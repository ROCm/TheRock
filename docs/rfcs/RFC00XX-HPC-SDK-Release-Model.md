---
author: Saad Rahim (saadrahim)
created: 2026-06-02
modified: 2026-06-08
status: draft
---

# AMD HPC SDK release model

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
  (general model for components shipped on a cadence aligned with
  ROCm; HPC SDK is the first umbrella release that aggregates such
  components under a single ROCm-aligned version).

## Overview

The AMD HPC SDK is a ROCm expansion released on its own cadence,
aligned with the ROCm 6-week release train. This RFC defines the
versioning, stream layout, ROCm pinning rule, meta-packaging, and
release flow for the HPC SDK.

Repository hierarchy, stream subdomains, `amdrocm-repo` packaging,
and concurrent stream installation are defined in RFC00XX and are
not redefined here.

## Versioning and pinning

- **Cadence:** approximately 4 releases per year. Not every ROCm
  release has a corresponding HPC SDK release.
- **Versioning:** the HPC SDK release reuses the version of its
  pinned ROCm Core SDK release — `<X.Y>` (e.g. `7.14`). This makes
  the pinned ROCm release obvious from the version string alone.
- **ROCm pinning:** each HPC SDK release is pinned to exactly one
  stable ROCm Core SDK release.
- **Patch releases:** bug fixes are delivered as patch releases
  (`<X.Y>.N`, e.g. `7.14.1`) against the pinned ROCm version.

## Streams and layout

- **Streams (required in v1):** `dev`, `nightly`, `rc`, and
  `stable` subdomains are all required at launch. Per-stream
  retention and QA rules follow the global *Per-stream
  specializations* in RFC00XX — the HPC SDK does not redefine them.
  `dev` builds are per-commit / pre-nightly developer builds against
  the develop ROCm branch with no QA gate (developer-testing and
  infra dry-runs only, not end users); `nightly` builds against the
  develop ROCm branch on the nightly schedule; `rc` is tested by QA
  and mirrors `stable/` layout; `stable` GA releases are pinned to a
  stable ROCm version.
- **Layout:** published as
  `<stream>.repo.amd.com/rocm/hpc-sdk/<X.Y>/` for `rc`/`stable`,
  with release metadata recording the pinned ROCm version. `dev` and
  `nightly` builds use a date-stamped folder directly under
  `hpc-sdk/` —
  `dev.repo.amd.com/rocm/hpc-sdk/<YYYYMMDD-sha>/` and
  `nightly.repo.amd.com/rocm/hpc-sdk/<YYYYMMDD>/` — with no nested
  `<stream>/` path segment (the stream subdomain already provides
  the stream selector).
- **Peer placement:** `hpc-sdk/` is a top-level peer to `core/`,
  `pytorch/`, `jax/`, and `onnx-runtime/` under each stream
  subdomain's `rocm/` tree. It is **not** placed under
  `expansions/`. This makes the independent cadence visible in the
  URL structure.

## Install location

Install at `/opt/rocm/hpc-<X.Y>` (e.g. `/opt/rocm/hpc-7.14`), the
same scheme as the ROCm Core SDK (`/opt/rocm/core-<X.Y>`). Patch
releases (`<X.Y>.N`) land in place inside the matching `hpc-<X.Y>`
directory. `/opt/rocm/hpc/` → `/opt/rocm/hpc-<latest>` convenience
symlink.

## Meta-package

Each release ships an installable umbrella package
`amdrocm-hpc-<X.Y>` (rpm and deb) that depends on every HPC SDK
component at the versions in that release, plus a hard dependency on
the pinned ROCm Core SDK version. Installing the meta-package alone
yields a complete, version-consistent HPC SDK on each supported
distro:

```
yum install amdrocm-hpc-7.14
apt install amdrocm-hpc-7.14
```

Requirements:

- Pulls in all HPC SDK components at the exact versions shipped in
  that `<X.Y>` release.
- Declares a hard dependency on the pinned ROCm Core SDK version per
  the stream-matched pinning rule.
- Patch releases (`<X.Y>.N`) update the meta-package's component
  version pins without changing the meta-package name.
- Available in both rpm and deb formats for every supported distro.
- Component list is owned by the HPC SDK team and reviewed each
  release so new components are added to the meta-package
  automatically.

## Component versioning

Individual packages inside the HPC SDK carry their own component
version number in `<X.Y>.N` format — the same scheme as the umbrella
release, with a per-component patch field. The leading `<X.Y>`
always matches the umbrella release the component shipped in; `N`
increments independently per component across patch releases.

Naming follows the Core SDK component-package convention
(`amdrocm-<component>[-<target>]-<version>-<pkgrel>.<arch>.<ext>`,
e.g. `amdrocm-sparse-gfx1152-7.13.0-2.x86_64.rpm`). For HPC SDK
components the `<version>` field is `<X.Y>.N` — e.g.
`amdrocm-rochpcg-gfx1152-7.14.0-1.x86_64.rpm` shipped in the
`amdrocm-hpc-7.14` umbrella, then
`amdrocm-rochpcg-gfx1152-7.14.1-1.x86_64.rpm` in the `7.14.1`
patch release. The umbrella `amdrocm-hpc-<X.Y>[.N]` meta-package
pins each component to its exact shipped `<X.Y>.N` version.

## Components

First release (pinned to ROCm 7.14):

- rocHPCG
- rocALUTION
- hipFort
- rocHPL
- hipTensor

Second release adds:

- miniHPL
- miniHPCG
