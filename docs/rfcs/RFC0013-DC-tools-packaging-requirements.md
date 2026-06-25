---
author: <Faraaz Mustafa (faraaz-bot), Saad Rahim (saadrahim)
created: 2026-06-24
modified: 2026-06-24
status: draft
---

## Remaining Open Questions
side by side install?
only one GA version allowed?
Signed??
manylinux standards section??
Package Naming for No Duplication
packages device specific, if so how to treat?
RVS ABI stability and B/F compat
RFC is currently more gpu tools focused. CPU tools?
CLI compatibility with NVIDIA (out of scope for rn?)
should I include the list of various tools (tools landscape looks very different rn. Need to do a reaudit)? AGT, ARX, ASST, MEDT, MultEvent, AFHC, ADDC, AGFHC, hKYSYS, AHDS, RVS[MF9.1], RBT, RDC, TransferBench, AMD Interconnect Tool, AMDVBFlash, Scandump, SDV, AMDXIO, AIFM Node Mgmt Agent/Guest mode.
if we want OS support to mimic rocm need to support rpm and debs
Fixed GA releases for all tools?

# Data Center Tools Packaging & Build Requirements

## Overview

ROCm ships a growing set of data center (DC) tools — validation suites,
benchmarks, and diagnostics — that today have no shared packaging, installation,
or build contract. There are no set requirements for
how they package or build their deliverables, so each tool makes its own
assumptions about install location, linkage, and compatibility. This causes
fragile deployments and breakage when a tool is pulled into a shared bundle.

This RFC defines the strategy and **mandatory requirements** for how DC tools are
aggregated, distributed, installed, and built. It establishes RVS as the
container for open-source tools, standardizes install locations, sets runfile
installers as the default distribution format, and defines the compatibility and
build contracts every tool **must** meet before it can be included.

It is a companion to [RFC0009: TheRock Software Packaging Requirements](https://github.com/ROCm/TheRock/blob/main/docs/rfcs/RFC0009-OS-Packaging-Requirements.md), which
covers the ROCm Core SDK; this RFC covers the DC tools layered on top of it.

Our goals are to:

1. **Consolidate open-source DC tools under RVS** as the standard delivery vehicle
1. **Standardize install locations** so tools are discoverable in a predictable place
1. **Default to runfile installers**; deprioritize native rpm/deb for DC tools
1. **Enforce compatibility contracts** (glibc floor, no hard-coded paths, ABI rules)
1. **Enforce a build contract** (linkage, build image, ROCm relationship) for reproducibility
1. **Move toward "one AMD" common tooling** spanning GPU and CPU datacenter tools company-wide

## Scope

### In Scope

- Aggregation strategy for open-source DC tools (RVS as the container)
- Inclusion path and obligations for closed-source tools
- Install folder standardization for DC tools
- Distribution format (runfile installers; central tarball artifacts)
- Backward/forward compatibility contracts
- Build requirements (linkage, build image, glibc floor, ABI)
- The contract between DC tools and ROCm installations
- Signing, integrity, SBOM, and uninstall requirements

### Out of Scope

- ROCm Core SDK packaging (covered by RFC0009)
- GPU driver packaging
- Windows / WSL2 packaging for DC tools 
- Internal CI/CD implementation details
- Python pip / wheel packaging for tools

## Tool Aggregation Strategy

### RVS as the Primary Container

RVS is the standard container hosting open-source DC tools. Tools **MUST** be
delivered as RVS modules rather than standalone one-off packages unless granted
an explicit exception.

| Tool                                  | Disposition                              |
| :------------------------------------ | :--------------------------------------- |
| RVS (legacy)                          | Existing functionality preserved         |
| TransferBench                         | Migrated in as an RVS module             |
| AMD GPU driver validation (kfdtests)  | Migrated in as an RVS module (phased)    |
| Future open-source tools              | Land in RVS by default                   |

**Decision — driver validation:** kfdtests **WILL** migrate into RVS as a
dedicated module. RVS becomes the single entry point for both functional
validation and driver validation. This was previously framed as "might not go
there"; we resolve it as in-scope because a single harness reduces the number of
installers customers must manage and gives one place for results reporting.
Migration is phased: kfdtests ships standalone until module parity is reached,
then the standalone form is deprecated.

### Closed-Source Tools

Closed-source tools **SHOULD** follow the RVS strategy where feasible. Where they
cannot be hosted in RVS, they **MUST** still honor every contract in this RFC
(install location, compatibility, build, signing) and ship via the runfile
installer path below.

## Installation Folder Standardization

DC tools **MUST** install to a standardized, predictable location.

/opt/amdtools/
├── arc/            # ANC/ARC tool subtree
├── anc/
├── rvs/            # RVS and hosted open-source modules
└── <tool>/         # each tool owns its subtree

- Each tool owns its own subtree under `/opt/amdtools/<tool>`; there is **no
  shared `bin` directory** (Option A). This avoids cross-tool file collisions and
  makes side-by-side versions and clean uninstall trivial..
- Each tool subtree **SHOULD** follow a consistent internal layout:
  `bin/`, `lib/`, `share/`, `etc/`, so a wrapper or environment module can be
  generated mechanically.
  The installation structure and directory path follow standard Linux practices. 
a)	/opt/amd/bin
  	Binaries are installed in this location
b)	/opt/amd/lib
  	Libraries are installed in this location
c)	/opt/amd/doc
  	Documentation for the product
d)	/opt/amd/include
  	Header files
e)	And other directories as part of the Linux file system standard such as etc or share as necessary
Preview versions or early access builds are provided via installers and tarballs only.
- For users who want tools on `PATH`, the installer **SHOULD** provide an
  optional environment module / shell-profile snippet rather than dumping
  symlinks into a global `bin`.
- Versioned installs **SHOULD** be supported for side-by-side use:
  `/opt/amdtools/<tool>-X.Y.Z` with a `/opt/amdtools/<tool>` softlink to the
  active version (mirrors the RFC0009 core-SDK convention).

# RPATH and Relocatability
All packages must be built and shipped with $ORIGIN-based RPATH
RPMs must honor the --prefix argument for relocatable installs

## Distribution Format

- **Runfile installers are the default** for DC tools (especially closed-source).
- **Native packages (rpm/deb) are deprioritized.** A tool **MAY** additionally
  ship rpm/deb only if it has a concrete customer requirement; this is not the
  default and is not maintained centrally.
- Every tool **MUST** publish its build artifacts as **tarballs to the central CI
  artifact location**. Canonical path convention:
  `repo.amd.com/amdtools/<tool>/<version>/<os>/<tool>-<version>-<os>.tar.gz`.
- Given tarballs in the central location, runfile installers are generated by
  reusing **TheRock's runfile-installer template** — swapping TheRock artifacts
  for the tool's tarballs and updating the installer metadata/GUI. Teams **MUST
  NOT** hand-craft custom installers.
- **Decision — installer tooling ownership:** the **ROCm packaging team** (owners
  of TheRock runfile installer) owns and maintains the shared runfile-installer
  generator as a reusable, parameterized component. 
- Installers **MUST** be non-interactive-capable (silent/`--accept`,
  `--prefix`, `--no-root` flags) for automated DC deployment.

## Integrity, Signing, and Provenance


## Compatibility Contracts

Any tool entering the shared bundle **MUST** agree to these contracts before
inclusion. Tools that do not honor the contract **MUST NOT** be pulled in —
silently bundling a non-conforming tool is the failure mode this RFC exists to
prevent.

- **glibc floor:** Tools **MUST** support **glibc 2.28** as the minimum. This is
  enforced by the build image (see Build Requirements), not by convention.
- **No hard-coded paths:** Tools **MUST NOT** hard-code absolute search paths.
  Library resolution **MUST** use `$ORIGIN`-relative RPATH; runtime configuration
  **MAY** use environment variables (e.g., `LD_LIBRARY_PATH`) but **MUST NOT**
  require them for a default install to function.
- **Relocatability:** Installs **MUST** be relocatable (honor `--prefix`) and
  **MUST** function without root where the install location permits.
- **ABI stability:** Public interfaces between RVS and its modules **MUST**
  follow a documented backward/forward-compatibility contract (below). A module
  built against RVS interface vN **MUST** continue to load against vN+1 within
  the same major version.
- **Symbol hygiene:** Tools **MUST NOT** export symbols that collide with ROCm or
  other tools; bundled third-party libs **SHOULD** use a versioned symbol map or
  be statically linked with hidden visibility.

### Backward/Forward Compatibility Contract (inlined)

1. **Within a major version**, removing or changing the signature of any public
   RVS module interface is prohibited. Additions are allowed.
2. **Modules declare** the minimum and maximum RVS interface version they
   support, and RVS **MUST** refuse to load a module outside that range with a
   clear error rather than crashing.
3. **Deprecation** requires one minor-version notice before removal at the next
   major version.
4. **Data/result formats** emitted by tools are versioned and parseable across at
   least one prior version.

## Build Requirements

A build contract so tools behave identically across deployment systems.

- **Build image:** Tools **MUST** build in the standard
  **`manylinux_2_28`-based** image (glibc 2.28, GCC 12+ toolchain). This image is
  the single source of the glibc floor and is published/maintained centrally.
- **C++ runtime / ABI:** Tools **MAY** assume a base C/C++ runtime is present,
  but because libstdc++ ABI varies across target distros, tools **MUST** build
  with `-static-libstdc++ -static-libgcc` (or bundle a known libstdc++) so they
  do not depend on a newer libstdc++ than the target provides.
- **Linkage policy (resolved buckets):**
  - **Dynamically link:** glibc/libc, libstdc++ runtime *only if* statically
    linking is infeasible, and the co-located ROCm runtime libraries (see below).
  - **Statically link (or bundle):** all other third-party dependencies prone to
    version skew — e.g., **libyaml**, libjson, compression libs, test/utility
    libraries. The default is "static unless there is a reason not to."
  - **Never dynamically link** against a distro-provided version of a library the
    tool also bundles (avoids the diamond-load problem).
- **Contract with ROCm installations:** DC tools **MUST**:
  - Locate ROCm via the RFC0009 softlink (`/opt/rocm/core` /
    `/opt/rocm/core-X`), never a hard-coded versioned path.
  - Declare the **minimum ROCm major.minor** they require and fail cleanly if it
    is absent or older.
  - **Not** bundle their own copy of ROCm runtime libraries; they consume the
    installed ROCm runtime so a single ROCm upgrade applies to all tools.
  - Treat ROCm as a **runtime dependency only** — tools **MUST NOT** assume ROCm
    development packages (headers, static libs) are present at runtime.

## Uninstall & Lifecycle

- Every installer **MUST** provide a clean uninstaller that removes the tool's
  `/opt/amdtools/<tool>` subtree and any environment module it created, leaving
  no orphaned files.
- Upgrades **MUST** be safe in place for patch versions and side-by-side for
  major.minor (mirrors RFC0009 semantics).

## Rationale: The "One AMD" Common Way

A common packaging and build approach spanning GPU and CPU tools, company-wide,
presents one coherent experience. "One AMD" tooling is what customers want and is
the long-term motivation for standardizing here rather than letting each tools
team diverge. Converging on RVS + a shared installer generator + one build image
is the concrete mechanism to get there.

# Versioning Requirements
For versioning requirements on packaging, see the following documentation: [TheRock package versioning](https://github.com/ROCm/TheRock/blob/main/docs/packaging/versioning.md)

# Dependency Requirements
For dependency requirements on packaging, see the following documentation: [TheRock package dependency](https://github.com/ROCm/TheRock/blob/main/docs/packaging/nativepackage_dependency_tree.md)
