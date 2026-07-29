---
author: Liam Berry (liaberry), Saad Rahim (saadrahim)
created: 2026-07-27
modified: 2026-07-29
status: Draft
---

# ROCm Core SDK & Runtime Container Standardization

## Problem

ROCm's container catalog spans 93 Docker Hub repositories. The Core
SDK/Runtime images at the bottom of that stack — the ones nearly every other
ROCm container is built on — have no shared structure:

- **Repository names encode the OS and the version.** `rocm/dev-ubuntu-24.04`
  puts the distro and its version in the repository name, so every new OS or
  OS version needs a whole new repository. `rocm/7.0` and `rocm/7.x-preview`
  do the same with the ROCm version.
- **The tiers aren't distinguished.** Nothing in the naming separates a
  runtime image from a dev image from a full SDK. `rocm/dev-ubuntu-24.04`
  publishes `7.2.3` (1.1 GB) and `7.2.3-complete` (6.9 GB) side by side — a
  six-fold size difference within one release, with nothing machine-readable
  to say what the extra 5.8 GB is or which tag a given user wants.
- **Downstream images don't share a base.** Containers are built by
  reinstalling ROCm onto a bare OS image rather than layering on one
  validated base, so there is no single place to patch a CVE or pin a ROCm
  version, and no guarantee that two ROCm images agree on what "ROCm 7.2"
  contains.

These are structural problems — properties of how images are named, tiered,
layered, and retired. The distinct and equally real problem of what a
container's documentation must *contain* is addressed by
[RFC0014](./RFC0014-ROCm-Container-Documentation-Standard.md), which this RFC
adopts by reference rather than restating.

## Scope

- **In scope:** repository naming, tag grammar, OCI metadata labels, image
  layering, the published layer/base-image matrix, and deprecation and
  lifecycle for Core SDK/Runtime containers.
- **Out of scope:** the required contents of a container's Docker Hub
  overview, defined in
  [RFC0014 — ROCm Container Documentation Standard](./RFC0014-ROCm-Container-Documentation-Standard.md).
  Framework, workload, and CI containers, and the full 93-repository cleanup,
  are deferred to Phases 2 and 3 (see [Implementation Plan](#implementation-plan)).

## Proposal

1. **A single Core SDK/Runtime repository.** All three layers (`runtime`,
   `core`, `core-sdk`) are published under one `rocm/rocm-core` repository,
   distinguished by the `{COMPONENT}` field of the tag rather than by
   separate repositories.
1. **A naming, tagging, and metadata standard** so ROCm version, OS, and
   support tier are machine-readable, and repository names stop encoding OS
   and version.
1. **A layering requirement**: every new ROCm container builds `FROM` one
   validated ROCm base image instead of reinstalling ROCm independently on a
   bare OS image.
1. **A documentation standard** every Core SDK/Runtime container must meet,
   defined in [RFC0014](./RFC0014-ROCm-Container-Documentation-Standard.md).
1. **A deprecation and lifecycle policy**, including a redirect requirement
   whenever a retired container has a successor.
1. **A worked case study**: retire `rocm/dev-ubuntu-24.04` in favor of the
   properly documented `rocm/rocm-core` repository (runtime tier).

## Naming and Tagging

### Repository naming

- **The Core SDK/Runtime tier lives in one repository: `rocm/rocm-core`.**
  The layer is selected by the tag's `{COMPONENT}` field, not by a separate
  repository. This avoids proliferating near-identical repositories and keeps
  a single documented, validated home for the tier.
- **OS, ROCm version, and layer belong in the tag, never the repository
  name.** Named violations today: `rocm/dev-ubuntu-24.04`, `rocm/7.0`,
  `rocm/7.x-preview`.
- Prefer layering on an existing, approved base image over duplicating
  installation steps that already exist upstream (see
  [Image Layering](#image-layering)).

### Tag grammar

ROCm docs already use tags such as
`rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1`. Core
SDK/Runtime images adopt the same field order:

```
rocm{X.Y}_{OS}[_py{X.Y}]_{COMPONENT}_{RELEASE_TYPE}_{VERSION}
```

| Field            | Meaning                 | Values                                              |
| ---------------- | ----------------------- | --------------------------------------------------- |
| `rocm{X.Y}`      | ROCm major.minor        | e.g. `rocm7.14` — always present                    |
| `{OS}`           | Base OS token           | See [Supported base images](#supported-base-images) |
| `py{X.Y}`        | Python version          | Optional; include **only** if Python is present     |
| `{COMPONENT}`    | Layer                   | `runtime`, `core`, `core-sdk`                       |
| `{RELEASE_TYPE}` | Release channel         | `stable`, `nightlies`, `prereleases`, `devreleases` |
| `{VERSION}`      | Full ROCm patch version | e.g. `7.14.0`                                       |

`{RELEASE_TYPE}` reuses the release-type vocabulary already defined by the
runtime Dockerfile and RFC0009's release version semantics, rather than
introducing a separate "channel" concept. (Earlier drafts of this RFC called
this field `{CHANNEL}`; it is the same thing, renamed to match the existing
build arg and packaging RFCs.)

**`{OS}` tokens.** The token is the distro name plus its version —
`ubuntu24.04`, `debian12`, `azurelinux3`. The Enterprise Linux family is the
single exception: it uses **source distro plus major only** —
`alma8`/`alma9`/`alma10` (AlmaLinux) and `ubi8`/`ubi9`/`ubi10` (RHEL UBI) —
dropping the EL minor, consistent with EL major-version ABI compatibility.
The exact base image is still fixed at build time by `BASE_IMAGE` and
recorded in the image's OCI metadata and the overview's `Prerequisites`
section.

**GPU architecture is not a tag field.** AMD publishes multi-arch images
only, so every published tag covers all supported GPU architectures in the
release and there is nothing to disambiguate. Single-family images are a
user-build option (see
[GPU architecture selection](#gpu-architecture-selection)); the selected
family is recorded in the image's OCI metadata, not in a published tag.

### Floating tags

Keep floating tags minimal and per-layer: `rocm{X.Y}-{COMPONENT}-latest`
tracks the newest patch of a given layer within a minor release — e.g.
`rocm/rocm-core:rocm7.14-runtime-latest`. **Avoid a bare `latest`.**

Applied to Phase 1: `rocm/rocm-core:rocm7.14_ubuntu24.04_runtime_stable_7.14.0`,
with a floating `rocm7.14-runtime-latest` track (and likewise for `core` and
`core-sdk`).

## Image Layering

**Background.** A Docker image is a stack of read-only filesystem layers, one
per Dockerfile instruction (`FROM`, `RUN`, `COPY`, etc.). When a Dockerfile
starts with `FROM <some-image>`, Docker reuses every layer of `<some-image>`
as-is and stacks new layers on top; like transparencies on an overhead
projector, the base image is the bottom sheet and every derived image adds
another sheet without redrawing what's below. This buys two things:

- **Consistency** — every image built `FROM` the same validated ROCm base
  inherits the same ROCm version, OS patches, and security posture. Patch the
  base once, and downstream images pick it up on their next rebuild.
- **Efficiency** — layers are content-addressed and cached, so a user who
  already has the ROCm runtime image pulled doesn't re-download it when
  pulling a framework image built on top; only the new layers transfer.

**Requirement.** Every new ROCm Core SDK, framework, or workload container
**must** use one of the `rocm/rocm-core` layer tags as its base — i.e.
`FROM rocm/rocm-core:<tag>`, choosing the `runtime`, `core`, or `core-sdk`
layer appropriate to what it needs — instead of starting `FROM ubuntu:24.04`
and reinstalling ROCm independently.

## Published Layers

### The three layers

Every ROCm release publishes three layers, all under the single
`rocm/rocm-core` repository, distinguished by the `{COMPONENT}` field of the
tag. Each installs a meta-package from the arch-independent multi-arch tree
at `repo.amd.com/rocm/packages-multi-arch/<distro>/`:

| Layer (`{COMPONENT}`) | Example tag                                                  | Installs meta      | Contents                                                   |
| --------------------- | ------------------------------------------------------------ | ------------------ | ---------------------------------------------------------- |
| `runtime`             | `rocm/rocm-core:rocm7.14_ubuntu24.04_runtime_stable_7.14.0`  | `amdrocm-runtime`  | HIP runtime + sysdeps/base/LLVM; run pre-built HIP apps    |
| `core`                | `rocm/rocm-core:rocm7.14_ubuntu24.04_core_stable_7.14.0`     | `amdrocm-core`     | Runtime + ROCm libraries                                   |
| `core-sdk`            | `rocm/rocm-core:rocm7.14_ubuntu24.04_core-sdk_stable_7.14.0` | `amdrocm-core-sdk` | Core + dev/build toolchain (compilers, headers, dev tools) |

**The layers nest.** The `core` tag builds `FROM` the `runtime` tag, and
`core-sdk` builds `FROM` `core` — all within the same repository. This is the
same layering rule required of downstream containers, applied to the tier
itself.

All three metas ship from the same arch-independent tree, but they differ in
what they pull in: `amdrocm-runtime` has no per-architecture content at all,
while `amdrocm-core` and `amdrocm-core-sdk` are **fan-out** metas that depend
on the per-architecture packages for every supported gfx architecture — which
is what a release container wants.

### Supported base images

Each layer is built once per supported base OS image, selected via the
`BASE_IMAGE` build arg. The full release matrix is therefore
*base images × layers*, each a distinct tag on `rocm/rocm-core`.

| Base image                                   | `{OS}` token  | Example tag (runtime layer)                  |
| -------------------------------------------- | ------------- | -------------------------------------------- |
| `ubuntu:22.04`                               | `ubuntu22.04` | `rocm7.14_ubuntu22.04_runtime_stable_7.14.0` |
| `ubuntu:24.04`                               | `ubuntu24.04` | `rocm7.14_ubuntu24.04_runtime_stable_7.14.0` |
| `ubuntu:26.04`                               | `ubuntu26.04` | `rocm7.14_ubuntu26.04_runtime_stable_7.14.0` |
| `debian:12`                                  | `debian12`    | `rocm7.14_debian12_runtime_stable_7.14.0`    |
| `debian:13`                                  | `debian13`    | `rocm7.14_debian13_runtime_stable_7.14.0`    |
| `almalinux:8`                                | `alma8`       | `rocm7.14_alma8_runtime_stable_7.14.0`       |
| `almalinux:9`                                | `alma9`       | `rocm7.14_alma9_runtime_stable_7.14.0`       |
| `almalinux:10`                               | `alma10`      | `rocm7.14_alma10_runtime_stable_7.14.0`      |
| `mcr.microsoft.com/azurelinux/base/core:3.0` | `azurelinux3` | `rocm7.14_azurelinux3_runtime_stable_7.14.0` |
| `registry.access.redhat.com/ubi8/ubi:8.10`   | `ubi8`        | `rocm7.14_ubi8_runtime_stable_7.14.0`        |
| `registry.access.redhat.com/ubi9/ubi:9.7`    | `ubi9`        | `rocm7.14_ubi9_runtime_stable_7.14.0`        |
| `registry.access.redhat.com/ubi10/ubi:10.1`  | `ubi10`       | `rocm7.14_ubi10_runtime_stable_7.14.0`       |
| `registry.suse.com/bci/bci-base:15.7`        | `sles15`      | `rocm7.14_sles15_runtime_stable_7.14.0`      |
| `registry.suse.com/bci/bci-base:16.0`        | `sles16`      | `rocm7.14_sles16_runtime_stable_7.14.0`      |

### GPU architecture selection

The multi-arch apt tree ships, from the same repository, both the fan-out
metas (all gfx) *and* every individual per-architecture content package —
e.g. `amdrocm-core-sdk7.14-gfx950`, `amdrocm-core7.14-gfx942`. No repository
switch is needed to narrow the architecture set.

Every ROCm Dockerfile **must** parametrize GPU architecture selection via a
build argument, so a user can build a smaller image containing only the
architecture(s) they need. TheRock's existing
[`dockerfiles/rocm_runtime.Dockerfile`](https://github.com/ROCm/TheRock/blob/main/dockerfiles/rocm_runtime.Dockerfile)
already establishes the convention this standard adopts:

```dockerfile
ARG BASE_IMAGE=ubuntu:24.04
ARG VERSION
ARG AMDGPU_FAMILY            # 'multi-arch' or a single gfx family
ARG RELEASE_TYPE=stable      # stable | nightlies | prereleases | devreleases
ARG INSTALL_METHOD=packages  # packages | tarball
```

Rules:

- **Released containers are built with `AMDGPU_FAMILY=multi-arch` and
  `INSTALL_METHOD=packages`.** The released image installs the all-GPU
  fan-out meta from `repo.amd.com`; the `tarball` path exists for local and
  pre-release builds.
- A user overriding `AMDGPU_FAMILY` to a single family (e.g. `gfx950`,
  `gfx110X-all`, `gfx94X-dcgpu`) must get an image containing only that
  family's packages, from the same source — no separate repository or base
  image.
- The chosen family **must** be recorded in the image's OCI metadata, so a
  slim image is self-describing even though it carries no published tag (see
  [Tag grammar](#tag-grammar)).

## Image Metadata

Set via `LABEL org.opencontainers.image.<field>="value"` in the Dockerfile.
Minimum required per image:

```
org.opencontainers.image.title
org.opencontainers.image.description
org.opencontainers.image.source        # repo URL
org.opencontainers.image.url           # docs URL
org.opencontainers.image.version       # image version
org.opencontainers.image.revision      # git SHA
org.opencontainers.image.licenses
org.opencontainers.image.created
```

Recommended: `org.opencontainers.image.vendor` (`AMD`),
`org.opencontainers.image.documentation`, `org.opencontainers.image.authors`.

## Documentation

Every Core SDK/Runtime container must publish a Docker Hub overview meeting
the ROCm container documentation standard. The required sections and their
contents are defined in
[RFC0014](./RFC0014-ROCm-Container-Documentation-Standard.md), which is the
authoritative source for that requirement; `rocm/dev-ubuntu-24.04` meets none
of it today. The [Case Study](#case-study-rocmrocm-core-runtime-tier) below
is the reference implementation.

## Deprecation and Lifecycle

Retiring a Core SDK/Runtime repository follows a fixed timeline:

| Stage        | Timing                  | Action                                                              |
| ------------ | ----------------------- | ------------------------------------------------------------------- |
| **Announce** | Day 0                   | Deprecation stated in the overview, with a pointer to the successor |
| **Freeze**   | Day 30                  | No new tags published                                               |
| **Archive**  | Next major ROCm release | Repository archived or removed                                      |

- A retired repository that **has** a successor must redirect: its overview
  is updated to point at the successor, and redirect tags are kept where
  feasible.
- A repository with **no** successor follows the same timeline, with no
  redirect requirement.
- Existing tags are not deleted before the archive stage; users on a pinned
  tag get the full window to migrate.

## Implementation Plan

- **Phase 1 (this RFC):** stand up the `rocm/rocm-core` repository publishing
  all three layers as tags; deprecate and redirect `rocm/dev-ubuntu-24.04`;
  add a base-layer digest check to the image publishing workflow; inventory
  the remaining Core SDK/Runtime repositories (`rocm/dev-ubuntu-22.04`,
  `rocm/dev-centos-7`, etc.).
  **Prerequisite:** verify each layer's package manifest, GPU architecture
  coverage, and owner of record against a built image before the Docker Hub
  overview is published (see [Open Questions](#open-questions)).
- **Phase 2:** migrate the other Core SDK/Runtime repositories into
  `rocm/rocm-core` tags and retire them on the deprecation timeline; extend
  this standard to other SDK containers.
- **Phase 3:** revisit the full 93-repository cleanup.

## Case Study: `rocm/rocm-core` (runtime tier)

### Current State: `rocm/dev-ubuntu-24.04`

Pulled directly from the ROCm Docker Hub organization:

| Attribute                      | Current value                                                                                                   |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| Repository name                | `rocm/dev-ubuntu-24.04` — a naming-rule violation (OS version in the repo name instead of the tag)              |
| Overview / description         | None. Docker Hub shows "No overview available."                                                                 |
| Pulls                          | 100K+                                                                                                           |
| Tags published                 | 9 (visible), spanning ROCm 6.4 through 7.2                                                                      |
| Most recent tags               | `7.2.3-complete` (6.9 GB) and `7.2.3` (1.1 GB), published side by side with no explanation of the difference    |
| Documentation                  | None: no prerequisites, no run command beyond a bare `docker pull`, no support contact, no security information |
| Support / ownership            | Not stated                                                                                                      |
| Version / compatibility matrix | Not published                                                                                                   |

It is the most-pulled and least-documented container in the tier, and many
other ROCm containers start from it or an image like it — the
highest-leverage place in the catalog to begin.

### Proposed Replacement

The `runtime` tier of the `rocm/rocm-core` repository
(`rocm/rocm-core:rocm7.14_ubuntu24.04_runtime_stable_7.14.0`) replaces it:

- Repository named for the tier, not the OS it runs on — OS, version, and
  layer all move into the tag, resolving the naming violation above.
- Ships only the ROCm runtime tier: HIP runtime + ROCm runtime libraries
  (e.g. rocBLAS, MIOpen, RCCL, hipBLAS, rocFFT) needed to run pre-built
  ROCm/HIP applications — no compiler toolchain, dev headers, or static
  libraries. To build from source, use the `core-sdk` tag of the same
  repository.
- Becomes the mandatory `FROM` base for every new Core SDK, framework, and
  workload container that only needs the runtime.
- Sits alongside the `core` and `core-sdk` tags in the same repository, so
  the three layers share one documented, validated home.

`rocm/dev-ubuntu-24.04` is then deprecated on the timeline in
[Deprecation and Lifecycle](#deprecation-and-lifecycle), with `rocm/rocm-core`
as its successor and therefore a redirect requirement.

### Proposed Docker Hub Description

A complete, ready-to-publish Docker Hub overview for the `rocm/rocm-core`
`runtime` tier — the worked example satisfying every required section of the
documentation standard — lives in
[RFC0014 § Worked Example](./RFC0014-ROCm-Container-Documentation-Standard.md#worked-example-rocmrocm-core-runtime-tier).
The `core` and `core-sdk` tiers each get their own overview following the
same template.

## Open Questions

- The exact per-layer package manifests and GPU architecture coverage for the
  `rocm/rocm-core` layers are placeholders pending verification against the
  built images. This verification is a Phase 1 prerequisite (see
  [Implementation Plan](#implementation-plan)) — the Docker Hub overview must
  not be published until the manifest and architecture list are confirmed, so
  the flagship example doesn't ship the same "asserted, not verified" gap
  this RFC is trying to eliminate.
- Does a single unified image publishing workflow exist today for Core
  SDK/Runtime images, or must one be established before the base-layer digest
  check (see [Implementation Plan](#implementation-plan)) can be enforced?
- Does the repository name `rocm/rocm-core` read as the whole tier, or as the
  `core` layer specifically? It collides with the existing `rocm-core`
  package name, and `rocm/rocm-core:...core-sdk...` is awkward. Alternatives
  such as `rocm/rocm` or `rocm/rocm-base` were not evaluated.

## References

- [RFC0014 — ROCm Container Documentation Standard](./RFC0014-ROCm-Container-Documentation-Standard.md)
- ROCm Docker Hub organization: <https://hub.docker.com/u/rocm>
- `rocm/dev-ubuntu-24.04`: <https://hub.docker.com/r/rocm/dev-ubuntu-24.04>
- ROCm-docker: <https://github.com/ROCm/ROCm-docker>
- Nvidia NGC Catalog (comparison reference): <https://catalog.ngc.nvidia.com/>
- Source PRD: *ROCm Container Image Publication Standardization Strategy —
  Phase 1: ROCm Core SDK & Runtime Containers* (internal)
