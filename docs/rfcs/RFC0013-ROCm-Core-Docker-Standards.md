---
author: Liam Berry (liaberry), Saad Rahim (saadrahim)
created: 2026-07-27
modified: 2026-07-29
status: Draft
---

# ROCm Core SDK & Runtime Container Standardization

## Problem

ROCm's container catalog spans 96 Docker Hub repositories. The Core
SDK/Runtime images at the bottom of that stack — the ones nearly every other
ROCm container is built on — have no shared structure:

- **Repository names encode the OS and the version.** `rocm/dev-ubuntu-24.04`
  puts the distro and its version in the repository name, so every new OS or
  OS version needs a whole new repository. `rocm/7.0` and `rocm/7.x-preview`
  do the same with the ROCm version.
- **The tiers aren't distinguished, and the ad-hoc markers disagree with each
  other.** `rocm/dev-ubuntu-24.04` publishes `7.2.3` (1.22 GB) alongside
  `7.2.3-complete` (7.40 GB) — a six-fold size difference within one release,
  with nothing machine-readable to say what the extra 6 GB is. The same
  repository also publishes `7.14.0-full` (7.95 GB), so two different
  suffixes, `-complete` and `-full`, appear to mean the same thing in one
  repository. Its bare `latest` tag resolves to the 7.95 GB image, so the
  most obvious command a new user types returns the largest artifact
  available with no indication that a 1.22 GB runtime exists.
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
  layering, supply-chain requirements, the published tier/base-image matrix,
  tag retention, and deprecation and lifecycle for Core SDK/Runtime
  containers.
- **Out of scope:** the required contents of a container's Docker Hub
  overview, defined in
  [RFC0014 — ROCm Container Documentation Standard](./RFC0014-ROCm-Container-Documentation-Standard.md).
  Framework, workload, and CI containers, and the full catalog cleanup, are
  deferred to Phases 2 and 3 (see [Implementation Plan](#implementation-plan)).

> **Scope note — this is not only a naming cleanup.** The three-tier package
> model below does not exist today. `dockerfiles/rocm_runtime.Dockerfile`
> installs a single fixed package set, not a selectable runtime/core/core-sdk
> tier, and it is not built or published by any CI workflow. Adopting this
> RFC means building the tier metas and a publishing pipeline, not just
> renaming what already ships. See
> [Implementation Plan](#implementation-plan).

## Proposal

1. **One repository per tier, named for the tier and never for the OS.**
   `rocm/rocm-runtime`, `rocm/rocm-core`, and `rocm/rocm-core-sdk`. A Docker
   Hub repository has exactly one overview and one set of pull statistics, so
   a tier that needs its own documentation and its own demand signal needs its
   own repository.
1. **A naming, tagging, and metadata standard** so ROCm version and OS are
   machine-readable, and repository names stop encoding OS and version.
1. **A layering requirement**: every new ROCm container builds `FROM` one
   validated ROCm base image instead of reinstalling ROCm independently on a
   bare OS image.
1. **A documentation standard** every Core SDK/Runtime container must meet,
   defined in [RFC0014](./RFC0014-ROCm-Container-Documentation-Standard.md).
1. **Supply-chain requirements**: signed images, published SBOMs, and
   recorded provenance for every base image others are required to build on.
1. **A deprecation, retention, and lifecycle policy** covering both tag
   pruning and repository retirement.
1. **A worked case study**: retire `rocm/dev-ubuntu-24.04` in favor of the
   properly documented `rocm/rocm-runtime`.

## Naming and Tagging

### Repository naming

- **Each tier is its own repository**, named for the tier:

  | Repository           | Tier                                                       |
  | -------------------- | ---------------------------------------------------------- |
  | `rocm/rocm-runtime`  | HIP runtime + ROCm runtime libraries                       |
  | `rocm/rocm-core`     | Runtime + ROCm libraries                                   |
  | `rocm/rocm-core-sdk` | Core + dev/build toolchain (compilers, headers, dev tools) |

- **OS and ROCm version belong in the tag, never the repository name.** Named
  violations today: `rocm/dev-ubuntu-24.04`, `rocm/7.0`, `rocm/7.x-preview`.

- Prefer layering on an existing, approved base image over duplicating
  installation steps that already exist upstream (see
  [Image Layering](#image-layering)).

### Tag grammar

ROCm docs already use tags such as
`rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1`. Core
SDK/Runtime images adopt the same field order. Because the tier is now the
repository, the tag carries no tier field:

```
rocm{X.Y}_{OS}[_py{X.Y}]_{RELEASE_TYPE}_{VERSION}
```

| Field            | Meaning                 | Values                                              |
| ---------------- | ----------------------- | --------------------------------------------------- |
| `rocm{X.Y}`      | ROCm major.minor        | e.g. `rocm7.14` — always present                    |
| `{OS}`           | Base OS token           | See [Supported base images](#supported-base-images) |
| `py{X.Y}`        | Python version          | Optional; include **only** if Python is present     |
| `{RELEASE_TYPE}` | Release channel         | `stable`, `nightlies`, `prereleases`                |
| `{VERSION}`      | Full ROCm patch version | e.g. `7.14.0`                                       |

`{RELEASE_TYPE}` reuses the release-type vocabulary already defined by the
runtime Dockerfile and RFC0009's release version semantics, rather than
introducing a separate "channel" concept. (Earlier drafts of this RFC called
this field `{CHANNEL}`; it is the same thing, renamed to match the existing
build arg and packaging RFCs.)

**`devreleases` is deliberately absent.** It is a valid `RELEASE_TYPE` for the
tarball install path, but `build_tools/.../install_rocm_packages.sh` rejects
it on the packages path, which is the path released containers use. If
`devreleases` images are ever published, that script must accept it first.

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

**CPU architecture is not a tag field either.** Phase 1 publishes
`linux/amd64` only. If another host architecture is published later it is
served from the *same* tag as a multi-platform manifest list, so `docker pull` resolves correctly without a new tag dimension.

### Floating tags

`rocm{X.Y}-latest` tracks the newest patch within a minor release — e.g.
`rocm/rocm-runtime:rocm7.14-latest`. Keep floating tags to this one form.

**A bare `latest` must resolve to the newest `stable` tag of that
repository.** Publishing no `latest` at all is not an option: `docker pull rocm/rocm-runtime` implicitly requests `latest` and would fail with a
confusing error. Today `rocm/dev-ubuntu-24.04:latest` silently serves the
7.95 GB image; a tier-named repository makes `latest` safe, because the
repository already tells the user which tier they are getting.

Applied to Phase 1: `rocm/rocm-runtime:rocm7.14_ubuntu24.04_stable_7.14.0`,
with a floating `rocm7.14-latest` track, and likewise for `rocm/rocm-core`
and `rocm/rocm-core-sdk`.

### Tag retention

Published tags accumulate: 14 base images per tier per patch release. Without
pruning, each repository reproduces the "which of these do I want?" problem
this RFC exists to fix.

- `stable` tags are retained for the supported lifetime of their ROCm minor
  release.
- `nightlies` are pruned after 30 days.
- `prereleases` are pruned once the corresponding release ships.
- **A published digest is never deleted**, only untagged. Pinned digests in
  user CI keep resolving.
- The retention policy must be stated in the repository's overview, per
  [RFC0014](./RFC0014-ROCm-Container-Documentation-Standard.md).

## Image Layering

**Background.** A Docker image is a stack of read-only filesystem layers, one
per Dockerfile instruction (`FROM`, `RUN`, `COPY`, etc.). When a Dockerfile
starts with `FROM <some-image>`, Docker reuses every layer of `<some-image>`
as-is and stacks new layers on top; like transparencies on an overhead
projector, the base image is the bottom sheet and every derived image adds
another sheet without redrawing what's below. This buys two things:

- **Consistency** — every image built `FROM` the same validated ROCm base
  inherits the same ROCm version, OS patches, and security posture. Patch the
  base once, and downstream images pick it up on their next rebuild. This is
  the primary benefit and the main reason to adopt this RFC.
- **Efficiency** — layers are content-addressed and cached, so a user who
  already holds the base image doesn't re-download it when pulling a
  framework image built on top. Note this only applies when both images were
  built from the *same base digest*; across base OS variants and rolling
  patch rebuilds, real-world reuse is lower than the mechanism suggests.

**Requirement.** Every new ROCm Core SDK, framework, or workload container
**must** build `FROM` the tier repository appropriate to what it needs —
`rocm/rocm-runtime`, `rocm/rocm-core`, or `rocm/rocm-core-sdk` — instead of
starting `FROM ubuntu:24.04` and reinstalling ROCm independently.

**Exceptions.** Some images legitimately cannot layer — manylinux wheels,
partner-supplied bases, images predating this standard. An exception must be
recorded with a named owner and a reason. Without a documented exception
path, teams route around the standard instead of through it; see
[Open Questions](#open-questions) on who grants exceptions.

## Published Tiers

### The three tiers

Every ROCm release publishes three tier repositories. Each installs a
meta-package from the arch-independent multi-arch tree at
`repo.amd.com/rocm/packages-multi-arch/<distro>/` (RPM distros append
`/x86_64`):

| Repository           | Example tag                                             | Installs meta      | Contents                                                   |
| -------------------- | ------------------------------------------------------- | ------------------ | ---------------------------------------------------------- |
| `rocm/rocm-runtime`  | `rocm/rocm-runtime:rocm7.14_ubuntu24.04_stable_7.14.0`  | `amdrocm-runtime`  | HIP runtime + sysdeps/base/LLVM; run pre-built HIP apps    |
| `rocm/rocm-core`     | `rocm/rocm-core:rocm7.14_ubuntu24.04_stable_7.14.0`     | `amdrocm-core`     | Runtime + ROCm libraries                                   |
| `rocm/rocm-core-sdk` | `rocm/rocm-core-sdk:rocm7.14_ubuntu24.04_stable_7.14.0` | `amdrocm-core-sdk` | Core + dev/build toolchain (compilers, headers, dev tools) |

> **This mapping is proposed, not current.** `amdrocm-runtime`,
> `amdrocm-core`, and `amdrocm-core-sdk` all exist as names in ROCm's
> packaging tree, but the Dockerfile's install script installs
> `amdrocm{X.Y}` plus `amdrocm-core-sdk{X.Y}` as a fixed pair — it has no
> tier selection, and `amdrocm{X.Y}` is documented as "all base ROCm
> libraries and runtime support", which is closer to the `core` tier than to
> `runtime`. Confirming or adjusting this mapping is Phase 1 work; see
> [Open Questions](#open-questions).

**The tiers nest.** `rocm/rocm-core` builds `FROM` `rocm/rocm-runtime`, and
`rocm/rocm-core-sdk` builds `FROM` `rocm/rocm-core`. This is the same
layering rule required of downstream containers, applied to the tiers
themselves.

All three metas ship from the same arch-independent tree, but they differ in
what they pull in: the runtime meta has no per-architecture content at all,
while the core and core-sdk metas are **fan-out** metas that depend on the
per-architecture packages for every supported gfx architecture — which is
what a release container wants.

### Supported base images

Each tier is built once per supported base OS image, selected via the
`BASE_IMAGE` build arg. The full release matrix is therefore
*base images × tiers* — 42 images per patch release.

| Base image                                   | `{OS}` token  | Example tag (runtime tier)           |
| -------------------------------------------- | ------------- | ------------------------------------ |
| `ubuntu:22.04`                               | `ubuntu22.04` | `rocm7.14_ubuntu22.04_stable_7.14.0` |
| `ubuntu:24.04`                               | `ubuntu24.04` | `rocm7.14_ubuntu24.04_stable_7.14.0` |
| `ubuntu:26.04`                               | `ubuntu26.04` | `rocm7.14_ubuntu26.04_stable_7.14.0` |
| `debian:12`                                  | `debian12`    | `rocm7.14_debian12_stable_7.14.0`    |
| `debian:13`                                  | `debian13`    | `rocm7.14_debian13_stable_7.14.0`    |
| `almalinux:8`                                | `alma8`       | `rocm7.14_alma8_stable_7.14.0`       |
| `almalinux:9`                                | `alma9`       | `rocm7.14_alma9_stable_7.14.0`       |
| `almalinux:10`                               | `alma10`      | `rocm7.14_alma10_stable_7.14.0`      |
| `mcr.microsoft.com/azurelinux/base/core:3.0` | `azurelinux3` | `rocm7.14_azurelinux3_stable_7.14.0` |
| `registry.access.redhat.com/ubi8/ubi:8.10`   | `ubi8`        | `rocm7.14_ubi8_stable_7.14.0`        |
| `registry.access.redhat.com/ubi9/ubi:9.7`    | `ubi9`        | `rocm7.14_ubi9_stable_7.14.0`        |
| `registry.access.redhat.com/ubi10/ubi:10.1`  | `ubi10`       | `rocm7.14_ubi10_stable_7.14.0`       |
| `registry.suse.com/bci/bci-base:15.7`        | `sles15`      | `rocm7.14_sles15_stable_7.14.0`      |
| `registry.suse.com/bci/bci-base:16.0`        | `sles16`      | `rocm7.14_sles16_stable_7.14.0`      |

**Redistribution rights must be confirmed per base image before first
publish.** RHEL UBI and SUSE BCI carry their own redistribution terms; these
are the two entries here that need legal sign-off rather than a license
reference (see [RFC0014](./RFC0014-ROCm-Container-Documentation-Standard.md)
§ Licensing).

### GPU architecture selection

The multi-arch apt tree ships, from the same repository, both the fan-out
metas (all gfx) *and* every individual per-architecture content package —
e.g. `amdrocm7.14-gfx950`, `amdrocm-core-sdk7.14-gfx942`. No repository
switch is needed to narrow the architecture set.

Every ROCm Dockerfile **must** parametrize GPU architecture selection via a
build argument, so a user can build a smaller image containing only the
architecture(s) they need. TheRock's
[`dockerfiles/rocm_runtime.Dockerfile`](https://github.com/ROCm/TheRock/blob/main/dockerfiles/rocm_runtime.Dockerfile)
already carries the build args this standard adopts. **The defaults below are
what this RFC proposes for released images, not what the file currently
declares** — it defaults to `RELEASE_TYPE=nightlies` and
`INSTALL_METHOD=tarball` today, and changing those defaults (or overriding
them in the release pipeline) is part of adopting this RFC:

```dockerfile
ARG BASE_IMAGE=ubuntu:24.04
ARG VERSION
ARG AMDGPU_FAMILY            # 'multi-arch' or a single gfx family
ARG RELEASE_TYPE=stable      # proposed default; file currently: nightlies
ARG INSTALL_METHOD=packages  # proposed default; file currently: tarball
```

Rules:

- **Released containers are built with `AMDGPU_FAMILY=multi-arch` and
  `INSTALL_METHOD=packages`.** The released image installs the all-GPU
  fan-out meta from `repo.amd.com`; the `tarball` path exists for local and
  pre-release builds.
- A user overriding `AMDGPU_FAMILY` to a single family (e.g. `gfx950`,
  `gfx110X-all`, `gfx94X-dcgpu`) must get an image containing only that
  family's packages, from the same source — no separate repository or base
  image. Note `multi-arch` is a Dockerfile/install-script sentinel; it is not
  a value `THEROCK_AMDGPU_FAMILIES` accepts at the CMake level.
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

## Supply Chain

These images are the mandated base for the rest of the catalog, so a
compromise here propagates everywhere. Every published Core SDK/Runtime
image **must**:

- **Be signed**, with the signature verifiable by a documented public command.
  RFC0014 requires that command to appear in the overview; an unverifiable
  signature provides no assurance.
- **Publish an SBOM**, attached to the image as an OCI referrer rather than
  as a side artifact a user has to go find.
- **Record provenance** — the git SHA in `org.opencontainers.image.revision`
  must identify the exact Dockerfile revision the image was built from.

## Documentation

Every Core SDK/Runtime container must publish a Docker Hub overview meeting
the ROCm container documentation standard. The required sections and their
contents are defined in
[RFC0014](./RFC0014-ROCm-Container-Documentation-Standard.md), which is the
authoritative source for that requirement; `rocm/dev-ubuntu-24.04` meets none
of it today. Because each tier is its own repository, each publishes its own
overview. The [Case Study](#case-study-rocmrocm-runtime) below is the
reference implementation.

## Deprecation and Lifecycle

Retiring a Core SDK/Runtime repository follows a fixed timeline:

| Stage        | Timing                  | Action                                                              |
| ------------ | ----------------------- | ------------------------------------------------------------------- |
| **Announce** | Day 0                   | Deprecation stated in the overview, with a pointer to the successor |
| **Freeze**   | Day 30                  | No new tags published                                               |
| **Archive**  | Next major ROCm release | Repository marked archived and read-only                            |

**Docker Hub has no redirect mechanism.** "Redirect" here means three
concrete things, and nothing more:

1. The overview is rewritten to point at the successor repository.
1. Until Freeze, successor images continue to be mirrored under the old
   repository's tag names, so existing `docker pull` commands keep working.
1. An existing tag is **never** repointed at different content. A user who
   pinned a tag gets what they pinned.

Further rules:

- A repository with **no** successor follows the same timeline, with no
  mirroring obligation.
- **Archive means read-only, not deleted.** Published digests survive
  archival, so pinned CI pipelines do not break. Deleting a published image
  is out of scope for this policy.
- The deprecation window applies per repository. See
  [Open Questions](#open-questions) — 30 days to freeze may be too aggressive
  for an image at this pull volume.

## Implementation Plan

- **Phase 0 — prerequisites.** Confirm the tier meta-package mapping against
  built images; make `install_rocm_packages.sh` able to install each tier
  independently; establish (or identify) the image publishing workflow that
  the base-layer digest check will hook into. None of these exist today, and
  Phases 1–3 depend on all of them.
- **Phase 1.** Stand up `rocm/rocm-runtime`, `rocm/rocm-core`, and
  `rocm/rocm-core-sdk`; publish overviews per RFC0014; deprecate
  `rocm/dev-ubuntu-24.04`; enforce the base-layer digest check; inventory the
  remaining Core SDK/Runtime repositories (`rocm/dev-ubuntu-22.04`,
  `rocm/dev-centos-7`, etc.).
  **Gate:** no overview is published until its package manifest, GPU
  architecture coverage, and owner of record have been verified against a
  built image — so the flagship example doesn't ship the same "asserted, not
  verified" gap this RFC exists to eliminate.
- **Phase 2.** Migrate the other Core SDK/Runtime repositories to the tier
  repositories and retire them on the deprecation timeline; extend this
  standard to other SDK containers.
- **Phase 3.** Revisit the full catalog cleanup.

## Case Study: `rocm/rocm-runtime`

### Current State: `rocm/dev-ubuntu-24.04`

Pulled from the Docker Hub API on 2026-07-29:

| Attribute                      | Current value                                                                                                              |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| Repository name                | `rocm/dev-ubuntu-24.04` — a naming-rule violation (OS version in the repo name instead of the tag)                         |
| Overview / description         | None. Docker Hub shows "No overview available."                                                                            |
| Pulls                          | 355,050                                                                                                                    |
| Tags published                 | 47, spanning ROCm 6.2 through 7.14                                                                                         |
| Size markers                   | `7.2.3` (1.22 GB) vs `7.2.3-complete` (7.40 GB); `7.14.0-full` (7.95 GB) uses a *second* suffix for the same apparent idea |
| `latest`                       | Resolves to the 7.95 GB image, with nothing to indicate a 1.22 GB runtime exists                                           |
| Documentation                  | None: no prerequisites, no run command beyond a bare `docker pull`, no support contact, no security information            |
| Support / ownership            | Not stated                                                                                                                 |
| Version / compatibility matrix | Not published                                                                                                              |

It is the most-pulled and least-documented container in the tier, and many
other ROCm containers start from it or an image like it — the
highest-leverage place in the catalog to begin.

### Proposed Replacement

`rocm/rocm-runtime:rocm7.14_ubuntu24.04_stable_7.14.0` replaces it:

- Repository named for the tier, not the OS it runs on — OS and version move
  into the tag, resolving the naming violation above.
- Ships only the ROCm runtime tier: HIP runtime + ROCm runtime libraries
  (e.g. rocBLAS, MIOpen, RCCL, hipBLAS, rocFFT) needed to run pre-built
  ROCm/HIP applications — no compiler toolchain, dev headers, or static
  libraries. To build from source, use `rocm/rocm-core-sdk`.
- Becomes the mandatory `FROM` base for every new Core SDK, framework, and
  workload container that only needs the runtime.
- Has its own overview and its own pull statistics, so demand for the runtime
  tier is measurable independently of the SDK tiers.

`rocm/dev-ubuntu-24.04` is then deprecated on the timeline in
[Deprecation and Lifecycle](#deprecation-and-lifecycle), with
`rocm/rocm-runtime` as its successor.

### Proposed Docker Hub Description

A complete, ready-to-publish Docker Hub overview for `rocm/rocm-runtime` —
the worked example satisfying every required section of the documentation
standard — lives in
[RFC0014 § Worked Example](./RFC0014-ROCm-Container-Documentation-Standard.md#worked-example-rocmrocm-runtime).
`rocm/rocm-core` and `rocm/rocm-core-sdk` each get their own overview
following the same template.

## Open Questions

- **Does the tier meta mapping hold?** The runtime/core/core-sdk metas in
  [Published Tiers](#published-tiers) are proposed. The install script today
  ships a fixed pair with no tier selection, and the package documented as
  the runtime meta may in fact be the core tier. This must be resolved in
  Phase 0 — the rest of the RFC's tiering rests on it.
- **Who grants layering exceptions, and what happens to non-compliant
  images?** The layering rule is a `MUST` with no stated owner, no exception
  process, and no consequence. Standards without a gate decay.
- **Is a 30-day freeze appropriate?** For a repository at 355K pulls, 30 days
  to stop publishing new tags may be too short for enterprise users, and
  "next major ROCm release" is an unpredictable archive date since ROCm
  majors are irregular. A fixed minimum window (e.g. 6 months to freeze, 12
  to archive) would let users plan.
- **How is success measured?** This RFC is justified by company impact but
  defines no metric. Candidates: share of ROCm images built `FROM` a tier
  repository; documentation-compliance rate across the catalog; time to patch
  a CVE across all published images; pull distribution across the three
  tiers.
- **What does 42 images per patch release cost?** Build time, storage, and
  egress for the full matrix have not been estimated, and no team has been
  identified as owning the pipeline.
- **Does `rocm/rocm-core` collide with the existing `rocm-core` package
  name?** The tier name is accurate, but the same string already names a ROCm
  package. Alternatives such as `rocm/rocm-libs` were not evaluated.

## References

- [RFC0014 — ROCm Container Documentation Standard](./RFC0014-ROCm-Container-Documentation-Standard.md)
- ROCm Docker Hub organization: <https://hub.docker.com/u/rocm>
- `rocm/dev-ubuntu-24.04`: <https://hub.docker.com/r/rocm/dev-ubuntu-24.04>
- ROCm-docker: <https://github.com/ROCm/ROCm-docker>
- Nvidia NGC Catalog (comparison reference): <https://catalog.ngc.nvidia.com/>
- Source PRD: *ROCm Container Image Publication Standardization Strategy —
  Phase 1: ROCm Core SDK & Runtime Containers* (internal)
