---
author: Liam Berry (liaberry), Saad Rahim (saadrahim)
created: 2026-07-28
modified: 2026-07-28
status: Draft
---

# ROCm Container Documentation Standard

<!--
PR bot: the JIRA/ISSUE reference lives in the *PR description*, not this file.
Add a line such as `JIRA ID : TBD` (replace with the real key before review)
so the Title/Description policy check passes.
-->

## Problem

ROCm's container catalog spans 93 Docker Hub repos, and the Docker Hub
overview is the first — often the only — documentation a user sees before
pulling an image. Today those overviews are inconsistent or absent: the
most-pulled Core SDK/Runtime image, `rocm/dev-ubuntu-24.04` (100K+ pulls),
shows "No overview available" — no prerequisites, no run command beyond a
bare `docker pull`, no support contact, no security information. There is no
shared definition of what a ROCm container's documentation must contain, so
each repo documents (or doesn't) on its own terms.

This RFC defines the **documentation standard** every ROCm Core SDK/Runtime
container must meet. It is a companion to the naming, tagging, layering, and
lifecycle rules in
[RFC0013 — ROCm Core SDK & Runtime Container Standardization](./RFC0013-ROCm-Core-Docker-Standards.md);
RFC0013 references this document as the authoritative source for the
documentation requirement.

## Scope

- **In scope:** the required contents of the Docker Hub overview for every
  Core SDK/Runtime container, and where each field is sourced from.
- **Out of scope:** naming, tagging, OCI metadata labels, image layering,
  and deprecation/lifecycle — these live in RFC0013. Framework, workload,
  and CI containers adopt this standard in a later phase (see RFC0013's
  phasing).

## Requirement

Every Core SDK/Runtime container **must** publish a Docker Hub overview
containing the sections below. A container is not considered compliant —
and, once gating is in place, is not publishable — until every required
section is present and non-empty. `rocm/dev-ubuntu-24.04` meets none of
these as of today.

| Section | Contents | Required |
|---|---|:---:|
| **Overview** | Purpose, primary use cases, intended users; what's included; what's explicitly not included (e.g., the kernel driver is host-side) | Yes |
| **Additional Explanations** | Non-obvious components (UCX/UCC, OpenMPI, RCCL, vLLM, Triton, FlashAttention, etc.); validated vs. community best-effort/experimental | Yes |
| **Prerequisites** | Host OS, AMD GPU driver/kernel module requirements (host-side), network driver requirements for multi-node use, container runtime requirements, GPU support notes, link to "Running ROCm Docker containers" | Yes |
| **Usage** | Minimal run command with GPU device access, common run patterns, a "hello world" validation step (e.g., `rocminfo`), links to user guides | Yes |
| **Suggested Reading** | ROCm docs, relevant GitHub repos, model/app docs | Optional |
| **Licensing** | Base image license, ROCm licensing references, third-party licenses, a short license-notes paragraph | Yes |
| **Support and Ownership** | Maintainer contact (org/team); support level: Preview, GA, Deprecated, or Archived | Yes |
| **Version and Compatibility Matrix** | ROCm version(s), OS base, framework version where applicable | Yes |
| **Security** | SBOM availability, vulnerability scanning policy, links to AMD's (and, for reference, Nvidia's) vulnerability disclosure/response processes | Yes |

## Section Guidance

- **Overview** — Lead with what the image *is* and who it's for. Always
  state what is *not* included; the most common user error is expecting the
  GPU kernel driver inside the container when it is host-side.
- **Additional Explanations** — Use this to disambiguate anything a user
  can't infer from the image name, and to mark each non-obvious component as
  validated or best-effort/experimental so expectations are set before pull.
- **Prerequisites** — Must be sufficient for a user to go from a clean host
  to a running container. Multi-node use has its own network-driver
  prerequisites; call them out rather than assuming single-node.
- **Usage** — The minimal run command must include GPU device access and a
  validation step the user can run to confirm the GPU is visible
  (`rocminfo`). "It pulled" is not "it works."
- **Licensing** — Name the base OS license, reference ROCm licensing, and
  point to the SBOM for the full third-party inventory rather than
  enumerating it inline.
- **Support and Ownership** — A named owner (team, not an individual) and an
  explicit support level are mandatory. "Not stated" is the current default
  and is not acceptable.
- **Version and Compatibility Matrix** — Map published tags to ROCm
  version, OS base, and (where applicable) framework version, so users can
  pick the right tag without guessing.
- **Security** — SBOM availability and scanning policy are required, not
  aspirational; link the vulnerability-disclosure process for reporting.

## Worked Example: `rocm/rocm-core` (runtime tier)

The following is a complete, ready-to-publish Docker Hub overview for the
`runtime` tier of the `rocm/rocm-core` repository — the reference implementation
that satisfies every required section of this standard. The `core` and
`core-sdk` tiers each get their own overview following the same template. See
[RFC0013 § Case Study](./RFC0013-ROCm-Core-Docker-Standards.md#case-study-rocmrocm-core-runtime-tier)
for how this tier replaces the undocumented `rocm/dev-ubuntu-24.04`.

> The package list and license links below are a grounded starting point
> based on ROCm's public runtime/devel packaging tiers, **not a confirmed
> manifest of the built image**. The maintainer of record is devops; the
> bracketed fields (GPU architecture coverage, contact channel, doc links)
> must be verified against the built image and filled in before publishing.

````markdown
# rocm/rocm-core

**The standardized ROCm Core SDK/Runtime repository.** Publishes three layers
as tags — `runtime`, `core`, and `core-sdk`. This overview covers the
`runtime` tier: the HIP runtime and the full set of ROCm runtime libraries,
with no compiler toolchain or dev headers. It is the required base layer for
all ROCm Core SDK, framework, and workload containers that only need the
runtime — see "Image Layering" in the container standard.

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
  --shm-size 8G rocm/rocm-core:<runtime-tag>
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
- Maintainer: devops — [contact channel].
- Support level: GA (Validated).

## Version and Compatibility Matrix
| ROCm version | Layer | Tag |
|---|---|---|
| 7.14.x | runtime | `rocm7.14_ubuntu24.04_runtime_stable_7.14.0` |
| 7.14.x | core | `rocm7.14_ubuntu24.04_core_stable_7.14.0` |
| 7.14.x | core-sdk | `rocm7.14_ubuntu24.04_core-sdk_stable_7.14.0` |
| 7.14 (floating) | runtime | `rocm7.14-runtime-latest` |

## Security
- SBOM generated per build and published alongside the image.
- Scanned at build time and on a recurring schedule; held to the
  "Validated" severity policy.
- See AMD Product Security's vulnerability disclosure policy [link] for how
  to report an issue.
````

## Open Questions

- Should the "Additional Explanations" and "Suggested Reading" sections be
  merged, or kept distinct as they are here?
- Does compliance get enforced by the same image-publishing workflow gate
  RFC0013 defines for base-layer digests, or by a separate documentation
  linter that checks the rendered Docker Hub overview?

## References

- [RFC0013 — ROCm Core SDK & Runtime Container Standardization](./RFC0013-ROCm-Core-Docker-Standards.md)
- ROCm Docker Hub organization: <https://hub.docker.com/u/rocm>
- `rocm/dev-ubuntu-24.04`: <https://hub.docker.com/r/rocm/dev-ubuntu-24.04>
- ROCm-docker: <https://github.com/ROCm/ROCm-docker>
- Nvidia NGC Catalog (comparison reference): <https://catalog.ngc.nvidia.com/>
