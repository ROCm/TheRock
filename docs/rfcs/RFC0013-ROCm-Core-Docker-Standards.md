---
authors:
  - Liam Berry (liaberry)
  - Saad Rahim (saadrahim)
created: 2026-07-27
modified: 2026-07-28
status: Draft
---
 
# ROCm Core SDK & Runtime Container Standardization
 
<!--
PR bot: the JIRA/ISSUE reference lives in the *PR description*, not this file.
Add a line such as `JIRA ID : TBD` (replace with the real key before review)
so the Title/Description policy check passes.
-->
 
## Problem
 
ROCm's container catalog spans 93 Docker Hub repos, but the Core SDK/Runtime images every other container is built on are inconsistently named, thinly documented, and don't distinguish runtime vs. dev vs. full-SDK tiers. For example, in the `rocm/dev-ubuntu-24.04` repository the most-pulled image has 100K+ pulls yet there is no overview, a repo name that bakes in an OS version, and two tags of the same release six times apart in size (`7.2.3`: 1.1 GB, `7.2.3-complete`: 6.9 GB) with no explanation of either.
 
## Proposal
 
1. **A documentation standard** every Core SDK/Runtime container must meet,
   defined in [RFC0014](./RFC0014-ROCm-Container-Documentation-Standard.md).
2. **A layering requirement**: every new ROCm container builds `FROM` one
   validated ROCm Runtime base image instead of reinstalling ROCm
   independently on a bare OS image.
3. **A metadata/tagging/naming standard** so ROCm version, OS, and support
   tier are machine-readable and repo names stop encoding OS versions.
4. **A single Core SDK/Runtime repository**: all three layers (runtime, core,
   core-sdk) are published under one `rocm/rocm-core` repository, distinguished by
   the `{COMPONENT}` field of the tag rather than by separate repos.
5. **A worked case study**: retire `rocm/dev-ubuntu-24.04` in favor of the
   properly documented `rocm/rocm-core` repository (runtime tier).
 
**Non-goals:** framework/workload/CI containers and the full 93-repo cleanup
are deferred to Phase 2/3 below.
 
## Naming & Tagging
 
- **One repository for the Core SDK/Runtime tier: `rocm/rocm-core`.** The layer
  (runtime, core, core-sdk) is selected by the tag's `{COMPONENT}` field, not
  by a separate repository. This avoids proliferating near-identical repos and
  keeps a single documented, validated home for the tier.
- OS and version live in the **tag**, not the repo name; this is what
  `rocm/dev-ubuntu-24.04` gets wrong today.
- Tag grammar: `rocm{X.Y}_{OS}_{COMPONENT}_{RELEASE_TYPE}_{VERSION}`,
  where `{OS}` is the distro name plus its version (e.g. `ubuntu24.04`,
  `debian12`, `sles15`); the Enterprise Linux family uses **source distro plus
  major** — `alma8`/`alma9`/`alma10` for AlmaLinux and `ubi8`/`ubi9`/`ubi10`
  for the RHEL UBI images → e.g.
  `rocm/rocm-core:rocm7.14_ubuntu24.04_runtime_stable_7.14.0`, with floating
  `rocm/rocm-core:rocm7.14-runtime-latest`.
 
## Implementation Plan
 
- **Phase 1 (this RFC):** stand up the `rocm/rocm-core` repository publishing all
  three layers as tags (`runtime`, `core`, `core-sdk`); deprecate and redirect
  `rocm/dev-ubuntu-24.04`; add a base-layer digest check to the image
  publishing workflow; inventory the remaining Core SDK/Runtime repos
  (`rocm/dev-ubuntu-22.04`, `rocm/dev-centos-7`, etc.).
  **Prerequisite:** verify each layer's package manifest, GPU architecture
  coverage, and owner of record against a built image before the Docker Hub
  overview is published (see [Open Questions](#open-questions)).
- **Phase 2:** migrate the other Core SDK/Runtime repos into `rocm/rocm-core` tags
  and retire them on the deprecation timeline; extend this standard to other
  SDK containers.
- **Phase 3:** revisit the full 93-repo cleanup.
 
## Container Standardization
 
Every Core SDK/Runtime container must publish a Docker Hub overview meeting
the ROCm container documentation standard — the required sections (Overview,
Prerequisites, Usage, Licensing, Support and Ownership, Version/Compatibility
Matrix, Security, and others) and their contents are defined in
[RFC0014 — ROCm Container Documentation Standard](./RFC0014-ROCm-Container-Documentation-Standard.md).
That RFC is the authoritative source for the documentation requirement;
`rocm/dev-ubuntu-24.04` meets none of it today. The
[Case Study](#case-study-rocmrocm-core-runtime-tier) below is the reference
implementation of that standard.
 
## Image Layering
 
**Background.** A Docker image is a stack of read-only filesystem layers, one
per Dockerfile instruction (`FROM`, `RUN`, `COPY`, etc.). When a Dockerfile
starts with `FROM <some-image>`, Docker reuses every layer of `<some-image>`
as-is and stacks new layers on top; like transparencies on an overhead
projector: the base image is the bottom sheet, and every derived image adds
another sheet without redrawing what's below. This buys two things:
 
- **Consistency** — every image built `FROM` the same validated ROCm Runtime
  image inherits the same ROCm version, OS patches, and security posture.
  Patch the base once, and downstream images pick it up on their next
  rebuild.
- **Efficiency** — layers are content-addressed and cached, so a user who
  already has the ROCm Runtime image pulled doesn't re-download it when
  pulling a framework image built on top; only the new layers transfer.
 
**Requirement.** Every new ROCm Core SDK, framework, or workload container
**must** use one of the `rocm/rocm-core` layer tags as its base — i.e.
`FROM rocm/rocm-core:<...runtime...>` (or the `core` / `core-sdk` tag appropriate
to what it needs) — instead of starting `FROM ubuntu:24.04` and reinstalling
ROCm independently. The three layers themselves nest within the same
repository: the `core` tag builds `FROM` the `runtime` tag, and `core-sdk`
builds `FROM` `core`.
 
## Published Layers and GPU Architecture Selection
 
### The three layers
 
Every ROCm release publishes three container layers, **all under the single
`rocm/rocm-core` repository**, distinguished by the `{COMPONENT}` field of the tag.
Each layer installs the matching **arch-independent multi-arch meta-package**
from `repo.amd.com/rocm/packages-multi-arch/<distro>/`:
 
| Layer (`{COMPONENT}` tag) | Example tag | Installs meta | Contents |
|---|---|---|---|
| `runtime` | `rocm/rocm-core:rocm7.14_ubuntu24.04_runtime_stable_7.14.0` | `amdrocm-runtime` | HIP runtime + sysdeps/base/LLVM; run pre-built HIP apps |
| `core` | `rocm/rocm-core:rocm7.14_ubuntu24.04_core_stable_7.14.0` | `amdrocm-core` | Runtime + ROCm libraries |
| `core-sdk` | `rocm/rocm-core:rocm7.14_ubuntu24.04_core-sdk_stable_7.14.0` | `amdrocm-core-sdk` | Core + dev/build toolchain (compilers, headers, dev tools) |
 
These layers nest: the `core` tag builds `FROM` the `runtime` tag, and
`core-sdk` builds `FROM` `core` — all within the same repository. Only the
runtime meta is truly arch-independent by nature; the `amdrocm-core` and
`amdrocm-core-sdk` metas in the multi-arch tree fan out to **every** supported
gfx architecture, which is what a release container wants.
 
**Release builds are multi-arch only.** Official released containers install
the fan-out metas above and therefore support all gfx architectures in the
release.
 
### Supported base images
 
Each release layer is built once per supported base OS image. The OS name and
version go in the tag's `{OS}` field (see [Naming & Tagging](#naming--tagging)),
not the repo name; the Enterprise Linux family uses the source distro plus major. The
supported base images for Phase 1 are:
 
| Base image | `{OS}` token | Example tag (runtime layer) |
|---|---|---|
| `ubuntu:22.04` | `ubuntu22.04` | `rocm7.14_ubuntu22.04_runtime_stable_7.14.0` |
| `ubuntu:24.04` | `ubuntu24.04` | `rocm7.14_ubuntu24.04_runtime_stable_7.14.0` |
| `ubuntu:26.04` | `ubuntu26.04` | `rocm7.14_ubuntu26.04_runtime_stable_7.14.0` |
| `debian:12` | `debian12` | `rocm7.14_debian12_runtime_stable_7.14.0` |
| `debian:13` | `debian13` | `rocm7.14_debian13_runtime_stable_7.14.0` |
| `almalinux:8` | `alma8` | `rocm7.14_alma8_runtime_stable_7.14.0` |
| `almalinux:9` | `alma9` | `rocm7.14_alma9_runtime_stable_7.14.0` |
| `almalinux:10` | `alma10` | `rocm7.14_alma10_runtime_stable_7.14.0` |
| `mcr.microsoft.com/azurelinux/base/core:3.0` | `azurelinux3` | `rocm7.14_azurelinux3_runtime_stable_7.14.0` |
| `registry.access.redhat.com/ubi8/ubi:8.10` | `ubi8` | `rocm7.14_ubi8_runtime_stable_7.14.0` |
| `registry.access.redhat.com/ubi9/ubi:9.7` | `ubi9` | `rocm7.14_ubi9_runtime_stable_7.14.0` |
| `registry.access.redhat.com/ubi10/ubi:10.1` | `ubi10` | `rocm7.14_ubi10_runtime_stable_7.14.0` |
| `registry.suse.com/bci/bci-base:15.7` | `sles15` | `rocm7.14_sles15_runtime_stable_7.14.0` |
| `registry.suse.com/bci/bci-base:16.0` | `sles16` | `rocm7.14_sles16_runtime_stable_7.14.0` |
 
The base OS is selected via the `BASE_IMAGE` build arg (see
[Dockerfile GPU architecture selection](#dockerfile-gpu-architecture-selection-required)).
Each of the three layers (`runtime`, `core`, `core-sdk`) is published for
every base image above, so the full release matrix is *base images × layers*,
each as a distinct tag on `rocm/rocm-core`.
 
The `{OS}` token carries the OS name and version, so each OS version gets a
distinct tag (`ubuntu22.04` vs `ubuntu24.04` vs `ubuntu26.04`). The Enterprise
Linux family uses **source distro plus major** — `alma8`/`alma9`/`alma10` for
AlmaLinux and `ubi8`/`ubi9`/`ubi10` for the RHEL UBI images — dropping the EL
minor, consistent with EL major-version ABI compatibility. The exact base
image is still
fixed at build time by `BASE_IMAGE` and recorded in the image's OCI metadata
and the overview's `Prerequisites` section.
 
### Dockerfile GPU architecture selection (required)
 
The multi-arch apt tree ships, from the same repo, both the fan-out meta
(all gfx) *and* every individual per-architecture content package — e.g.
`amdrocm-core-sdk7.13-gfx950`, `amdrocm-core7.13-gfx942`. No repo switch is
needed to narrow the arch set.
 
Every ROCm Dockerfile **must** parametrize GPU architecture selection via a
build argument so users can build a smaller image containing only the
architecture(s) they need. TheRock's existing
[`dockerfiles/rocm_runtime.Dockerfile`](https://github.com/ROCm/TheRock/blob/main/dockerfiles/rocm_runtime.Dockerfile)
already establishes the convention this standard adopts: an `AMDGPU_FAMILY`
build arg whose special value `multi-arch` installs AMD's all-GPU artifact,
and named values (e.g. `gfx110X-all`, `gfx94X-dcgpu`, `gfx950`) install a
single family. The same Dockerfile parametrizes `BASE_IMAGE`, `VERSION`,
`RELEASE_TYPE`, and `INSTALL_METHOD` (`tarball` or `packages`).
 
```dockerfile
# Existing convention from rocm_runtime.Dockerfile.
# Full release image (all GPU families):  AMDGPU_FAMILY=multi-arch
# Slimmer image (single family):          AMDGPU_FAMILY=gfx950
ARG BASE_IMAGE=ubuntu:24.04
ARG VERSION
ARG AMDGPU_FAMILY          # 'multi-arch' or a single gfx family
ARG RELEASE_TYPE=stable    # nightlies | prereleases | devreleases | stable
ARG INSTALL_METHOD=packages  # packages | tarball
```
 
Rules:
 
- **Released containers are built with `AMDGPU_FAMILY=multi-arch`.** This is
  what "release builds are multi-arch only" means in practice: the released
  image installs the all-GPU (fan-out) artifact.
- A user overriding `AMDGPU_FAMILY` to a single family must get an image
  containing only that family's packages from the same source — no separate
  repo or base image.
- The chosen family must be recorded in image metadata (see OCI annotations
  below) and reflected in the tag's GPU-family declaration so a slim image is
  self-describing.
 
## Metadata and Tagging Standards
 
### OCI Annotations
 
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
 
### Tag Schema
 
ROCm docs already use tags such as
`rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1`. Core
SDK/Runtime images adopt the same field order and the same OS-plus-version
convention for `{OS}` (e.g. `ubuntu24.04`), with one refinement: Enterprise
Linux uses source distro plus major (`alma8`/`alma9`/`alma10`, `ubi8`/`ubi9`/`ubi10`) — see
below:
 
```
rocm{X.Y}_{OS}_py{X.Y}_{COMPONENT}_{RELEASE_TYPE}_{VERSION}
```
 
`{OS}` is the distro name plus version — `ubuntu24.04`, `debian12`,
`azurelinux3`, `sles15` — except the Enterprise Linux family, which uses source
distro plus major: `alma8`/`alma9`/`alma10` (AlmaLinux) and `ubi8`/`ubi9`/`ubi10`
(RHEL UBI). See "Supported base
images" for the full list of tokens and example tags.
 
`{RELEASE_TYPE}` reuses the release-type vocabulary already defined by the
runtime Dockerfile and RFC0009's release version semantics — one of
`stable`, `nightlies`, `prereleases`, `devreleases` — rather than a separate
"channel" concept. (Earlier drafts of this RFC called this field `{CHANNEL}`;
it is the same thing as `RELEASE_TYPE` and is renamed here to match the
existing build arg and packaging RFCs.)
 
`{VERSION}` is the full ROCm `major.minor.patch` (e.g. `7.14.0`). For Core
SDK/Runtime images this intentionally shares its `major.minor` with the
leading `rocm{X.Y}` field: `rocm{X.Y}` is the grouping key that floating tags
and "latest patch in a minor" sorting hang off of, while `{VERSION}` records
the exact patch. The overlap is not redundant in the framework tags this
grammar is borrowed from — there `{VERSION}` is the framework's version (e.g.
PyTorch `2.9.1`), not ROCm's — so the shared schema is kept for consistency
across all `rocm/*` repos.
 
Rules: always include ROCm major.minor and OS base; declare the GPU family
(`multi-arch` for released images, or a single `gfx…` family for slim
builds); include Python only if it's present; the `{COMPONENT}` field carries
the layer (`runtime`, `core`, `core-sdk`); keep floating tags minimal and
per-layer (`rocm{X.Y}-<component>-latest` for the newest patch of a given
layer in a minor; avoid a bare `latest`).
 
Applied to Phase 1, on the `rocm/rocm-core` repository:
`rocm7.14_ubuntu24.04_runtime_stable_7.14.0`, with a floating
`rocm7.14-runtime-latest` track (and likewise for `core` / `core-sdk`).
 
### Naming Conventions
 
- **The Core SDK/Runtime tier lives in one repository, `rocm/rocm-core`.** Layers
  are tags, not repos (see [Published Layers](#published-layers-and-gpu-architecture-selection)).
- Avoid versioned or OS-encoding repos; version, OS, and layer belong in tags.
  Named violations today: `rocm/dev-ubuntu-24.04`, `rocm/7.0`,
  `rocm/7.x-preview`.
- Prefer layering on an existing, approved base image over duplicating
  installation steps that already exist upstream (see
  [Image Layering](#image-layering)).
 
## Case Study: `rocm/rocm-core` (runtime tier)
 
### Current State: `rocm/dev-ubuntu-24.04`
 
Pulled directly from the ROCm Docker Hub organization:
 
| Attribute | Current value |
|---|---|
| Repository name | `rocm/dev-ubuntu-24.04` — a naming-rule violation (OS version in the repo name instead of the tag) |
| Overview / description | None. Docker Hub shows "No overview available." |
| Pulls | 100K+ |
| Tags published | 9 (visible), spanning ROCm 6.4 through 7.2 |
| Most recent tags | `7.2.3-complete` (6.9 GB) and `7.2.3` (1.1 GB), published side by side with no explanation of the difference |
| Documentation | None: no prerequisites, no run command beyond a bare `docker pull`, no support contact, no security information |
| Support / ownership | Not stated |
| Version / compatibility matrix | Not published |
 
It's the single most-pulled, least-documented container in the tier, and
many other ROCm containers are built starting from it or an image like it —
the highest-leverage place in the catalog to start.
 
### Proposed Replacement
 
The `runtime` tier of the `rocm/rocm-core` repository
(`rocm/rocm-core:rocm7.14_ubuntu24.04_runtime_stable_7.14.0`) replaces it:
 
- Repository named for the tier, not the OS it runs on — OS, version, and
  layer all move into the tag, resolving the naming violation above.
- Ships only the ROCm runtime tier: HIP runtime + ROCm runtime libraries
  (e.g., rocBLAS, MIOpen, RCCL, hipBLAS, rocFFT) needed to run pre-built
  ROCm/HIP applications — no compiler toolchain, dev headers, or static
  libraries. To build from source, use the `core-sdk` tag of the same
  repository.
- Becomes the mandatory `FROM` base for every new Core SDK, framework, and
  workload container that only needs the runtime.
- Sits alongside the `core` and `core-sdk` tags in the same repository, so
  the three layers share one documented, validated home.
 
`rocm/dev-ubuntu-24.04` is deprecated: its overview is updated to redirect
to `rocm/rocm-core`, redirect tags are kept where feasible, and it's
archived on the standard timeline (announce Day 0 → no new tags Day 30 →
archive/remove at the next major release). Containers with no successor
follow the same timeline with no redirect requirement.
 
### Proposed Docker Hub Description
 
A complete, ready-to-publish Docker Hub overview for the `rocm/rocm-core` `runtime`
tier — the worked example that satisfies every required section of the
documentation standard — lives in
[RFC0014 § Worked Example](./RFC0014-ROCm-Container-Documentation-Standard.md#worked-example-rocmrocm-core-runtime-tier).
It is the reference implementation of that standard; the `core` and `core-sdk`
tiers each get their own overview following the same template.
 
## Open Questions
 
- The exact per-layer package manifests and GPU architecture coverage for the
  `rocm/rocm-core` layers are placeholders pending verification against the
  built images. This verification is a Phase 1 prerequisite (see
  [Implementation Plan](#implementation-plan)) — the Docker Hub overview
  must not be published until the manifest and architecture list are
  confirmed, so the flagship example doesn't ship the same "asserted, not
  verified" gap this RFC is trying to eliminate.
- Does a single unified image publishing workflow exist today for Core
  SDK/Runtime images, or must one be established before the base-layer
  digest check (see [Implementation Plan](#implementation-plan)) can be enforced?
