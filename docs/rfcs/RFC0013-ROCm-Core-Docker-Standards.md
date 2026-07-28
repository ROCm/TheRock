---
author: Liam Berry (liaberry)
created: 2026-07-27
modified: 2026-07-28
status: Approved
---

# ROCm Core SDK & Runtime Container Standardization

## Problem

ROCm's container catalog spans 93 Docker Hub repos, but the Core SDK/Runtime images every other container is built on are inconsistently named, thinly documented, and don't distinguish runtime vs. dev vs. full-SDK tiers. For example in the, `rocm/dev-ubuntu-24.04` repository the most-pulled image has 100K+ pulls yet there is no overview, a repo name that bakes in an OS version, and two tags of the same release six times apart in size (`7.2.3`: 1.1 GB, `7.2.3-complete`: 6.9 GB) with no explanation of either.

## Proposal

1. **A documentation standard** every Core SDK/Runtime container must meet.
2. **A layering requirement**: every new ROCm container builds `FROM` one
   validated ROCm Runtime base image instead of reinstalling ROCm
   independently on a bare OS image.
3. **A metadata/tagging/naming standard** so ROCm version, OS, and support
   tier are machine-readable and repo names stop encoding OS versions.
4. **A worked case study**: retire `rocm/dev-ubuntu-24.04` in favor of a
   properly documented `rocm/rocm-runtime`.

**Non-goals:** framework/workload/CI containers, the full 93-repo cleanup,
and consolidating `base`/`runtime`/`devel` into one `rocm-core` repo are
deferred to Phase 2/3 below.

## Naming & Tagging

- `rocm/<product-or-workload>`, with OS and version in the **tag**, not the
  repo name; this is what `rocm/dev-ubuntu-24.04` gets wrong today.
- Tag grammar: `rocm{X.Y}_OS{YY.MM}_{COMPONENT}_{CHANNEL}_{VERSION}` →
  e.g. `rocm7.14_ubuntu24.04_runtime_release_7.14.0`, floating `rocm7.14-latest`.

## Implementation Plan

- **Phase 1 (this RFC):** publish `rocm/rocm-runtime`; deprecate and
  redirect `rocm/dev-ubuntu-24.04`; add a base-layer digest check to the
  image publishing workflow; inventory the remaining Core SDK/Runtime repos
  (`rocm/dev-ubuntu-22.04`, `rocm/dev-centos-7`, `rocm/rocm-terminal`, etc.).
- **Phase 2:** consolidate `base` / `runtime` / `devel`-or-`all` into one
  `rocm/rocm-core` repo, distinguished by tag; migrate `rocm/rocm-runtime`
  into `rocm-core:runtime`.
- **Phase 3:** extend this standard to other SDK containers;
  revisit the full 93-repo cleanup.

## Container Standardization

Every Core SDK/Runtime container must document the following on its Docker
Hub overview. Note that `rocm/dev-ubuntu-24.04` meets none of them as of today:

| Section | Contents |
|---|---|
| **Overview** | Purpose, primary use cases, intended users; what's included; what's explicitly not included (e.g., the kernel driver is host-side) |
| **Additional Explanations** | Non-obvious components (UCX/UCC, OpenMPI, RCCL, vLLM, Triton, FlashAttention, etc.); validated vs. community best-effort/experimental |
| **Prerequisites** | Host OS, AMD GPU driver/kernel module requirements (host-side), network driver requirements for multi-node use, container runtime requirements, GPU support notes, link to "Running ROCm Docker containers" |
| **Usage** | Minimal run command with GPU device access, common run patterns, a "hello world" validation step (e.g., `rocminfo`), links to user guides |
| **Suggested Reading** *(optional)* | ROCm docs, relevant GitHub repos, model/app docs |
| **Licensing** | Base image license, ROCm licensing references, third-party licenses, a short license-notes paragraph |
| **Support and Ownership** | Maintainer contact (org/team); support level: Preview, GA, Deprecated, or Archived |
| **Version and Compatibility Matrix** | ROCm version(s), OS base, framework version where applicable |
| **Security** | SBOM availability, vulnerability scanning policy, links to AMD's (and, for reference, Nvidia's) vulnerability disclosure/response processes |

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

**Requirement.** Starting with `rocm/rocm-runtime` (see
[Case Study](#case-study-rocmrocm-runtime) below), every new ROCm Core SDK,
framework, or workload container **must** use `FROM rocm/rocm-runtime:<tag>`
(or a later approved tier, once Phase 2 ships) as its base layer, instead of
starting `FROM ubuntu:24.04` and reinstalling ROCm independently. This
becomes a required, checked field in the image publishing workflow —
reviewers verify a new image's base layer against the approved digest
before sign-off.

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
SDK/Runtime images adopt the same grammar:

```
rocm{X.Y}_OS{YY.MM}_py{X.Y}_{COMPONENT}_{CHANNEL}_{VERSION}
```

Rules: always include ROCm major.minor and OS base; declare support for all
GPU families, Radeon-only, or Instinct-only; include Python only if it's
present; channel must be one of `release`, `dev`, `nightly`, `ci`, `rc`;
keep floating tags minimal (`latest` only for one clearly defined track, or
avoid it; `rocm{X.Y}-latest` for the newest patch in a minor).

Applied to Phase 1: `rocm7.14_ubuntu24.04_runtime_release_7.14.0`, with a
floating `rocm7.14-latest` track.

### Naming Conventions

- `rocm/<product-or-workload>` — suffixes (`-dev`, `-ci`, `-runtime`,
  `-devel`) only when they communicate support intent.
- Avoid versioned repos; versions belong in tags. Named violations today:
  `rocm/dev-ubuntu-24.04`, `rocm/7.0`, `rocm/7.x-preview`.
- Prefer layering on an existing, approved base image over duplicating
  installation steps that already exist upstream (see
  [Image Layering](#image-layering)).

## Case Study: `rocm/rocm-runtime`

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

`rocm/rocm-runtime` replaces it:

- Named for what it is, not the OS it runs on — OS moves into the tag,
  resolving the naming violation above.
- Ships only the ROCm runtime tier: HIP runtime + ROCm runtime libraries
  (e.g., rocBLAS, MIOpen, RCCL, hipBLAS, rocFFT) needed to run pre-built
  ROCm/HIP applications — no compiler toolchain, dev headers, or static
  libraries. Building from source means layering a devel image on top
  instead (formalized in Phase 2).
- Becomes the mandatory `FROM` base for every new Core SDK, framework, and
  workload container.
- Is designed to fold into the unified `rocm-core` repository planned for
  Phase 2 with minimal rework.

`rocm/dev-ubuntu-24.04` is deprecated: its overview is updated to redirect
to `rocm/rocm-runtime`, redirect tags are kept where feasible, and it's
archived on the standard timeline (announce Day 0 → no new tags Day 30 →
archive/remove at the next major release). Containers with no successor
follow the same timeline with no redirect requirement.

### Proposed Docker Hub Description

> The package list, maintainer contact, and license links below are a
> grounded starting point based on ROCm's public runtime/devel packaging
> tiers, not a confirmed manifest of the built image. Verify against the
> built image and fill in the bracketed fields before publishing.

````markdown
# rocm/rocm-runtime

**The standardized ROCm runtime base image.** Provides the HIP runtime and
the full set of ROCm runtime libraries on Ubuntu 24.04, with no compiler
toolchain or dev headers. This is the required base layer for all ROCm Core
SDK, framework, and workload containers — see "Image Layering" in the
container standard.

## Overview
- **Purpose:** run pre-built ROCm/HIP applications, and serve as the
  standardized base layer for downstream ROCm containers.
- **Primary use cases:** running compiled HIP binaries; serving as the
  `FROM` base for framework images (PyTorch, JAX, vLLM, etc.) and workload
  images; production/inference deployments that don't need to compile code
  inside the container.
- **Intended users:** ROCm container maintainers, MLOps/platform engineers,
  and anyone deploying ROCm workloads who doesn't need a build toolchain.
- **What's included:** HIP runtime, ROCm runtime libraries (rocBLAS, MIOpen,
  RCCL, hipBLAS, rocFFT, and related components), rocminfo, rocm-smi.
- **What's not included:** the AMD GPU kernel driver (host-side — install
  via amdgpu-install on the host, not in the container), compilers (no
  hipcc/clang toolchain), dev headers and static libraries, and any
  framework packages (those belong in an image layered on top).

## Additional Explanations
- **Runtime vs. devel:** this image is the "runtime" tier only. To compile
  HIP code, build ROCm libraries from source, or debug at the
  driver/runtime level, use the devel/all tier instead (see Suggested
  Reading).
- **Validated vs. best-effort:** every published release tag is built and
  smoke-tested against ROCm's official release channel; nightly/ci/rc tags
  are best-effort and may be less stable.

## Prerequisites
- Host OS: Linux with a current AMD GPU (amdgpu) kernel driver installed —
  the driver is host-side and is not included in this image.
- Container runtime: Docker (or compatible), with GPU device access
  configured.
- Supported GPU architectures: [link to the current ROCm hardware
  compatibility matrix — confirm Instinct + Radeon coverage for this tag].
- See "Running ROCm Docker containers" in the ROCm documentation for host
  setup.

## Usage
Minimal run command with GPU device access:
```
docker run -it --network=host --device=/dev/kfd --device=/dev/dri \
  --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --shm-size 8G rocm/rocm-runtime:<tag>
```
Validate GPU visibility inside the container:
```
rocminfo
```
See the ROCm-docker GitHub repository (linked below) for multi-node and CI
usage patterns.

## Suggested Reading
- ROCm installation documentation
- ROCm-docker GitHub repository (Dockerfile source for this image)
- HIP runtime documentation
- ROCm hardware / GPU compatibility matrix

## Licensing
- Base OS image: Ubuntu 24.04, under Canonical's standard terms.
- ROCm components: see AMD ROCm licensing terms [link].
- This image may include third-party runtime libraries under their own
  licenses; see the attached SBOM for the full component/license inventory.

## Support and Ownership
- Maintainer: [owning team — e.g., ROCm Container Platform team] —
  [contact channel].
- Support level: GA (Validated).

## Version and Compatibility Matrix
| ROCm version | Tag |
|---|---|
| 7.14.x | `rocm7.14_ubuntu24.04_runtime_release_7.14.0` |
| 7.14 (floating) | `rocm7.14-latest` |

## Security
- SBOM generated per build and published alongside the image.
- Scanned at build time and on a recurring schedule; held to the
  "Validated" severity policy.
- See AMD Product Security's vulnerability disclosure policy [link] for how
  to report an issue.
````

## Open Questions

- Is dropping the OS from the repo name now (`rocm/rocm-runtime`) the right
  call for Phase 1, or should that wait for Phase 2's consolidation?
- GPU architecture coverage, maintainer of record, and the exact runtime
  library manifest for `rocm/rocm-runtime` are placeholders pending
  verification against the built image.

## References

- ROCm Docker Hub organization: <https://hub.docker.com/u/rocm>
- `rocm/dev-ubuntu-24.04`: <https://hub.docker.com/r/rocm/dev-ubuntu-24.04>
- ROCm-docker: <https://github.com/ROCm/ROCm-docker>
- Nvidia NGC Catalog (comparison reference): <https://catalog.ngc.nvidia.com/>
- Source PRD: *ROCm Container Image Publication Standardization Strategy —
  Phase 1: ROCm Core SDK & Runtime Containers* (internal)