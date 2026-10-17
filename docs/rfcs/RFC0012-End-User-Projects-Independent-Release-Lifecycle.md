---
author: Freddy Paul
created: 2026-04-14
modified: 2026-04-14
status: draft
---

# ROCm End-User Projects With Independent Release Lifecycle

## 1. Overview

ROCm ships a core SDK through TheRock build system. In addition to the core
SDK, AMD and the broader community maintain a set of **extras** — tools,
validation suites, and utility projects — that are compiled *on top of* a
released ROCm Core SDK installation but follow their own development and
release timelines. These extras are not part of the ROCm Core SDK; they
consume ROCm as a build dependency and are delivered as independently
versioned packages.

The ROCm Validation Suite (RVS) is the canonical example of such a project.
RVS exercises GPU hardware and the ROCm software stack through a collection
of test modules (GPU Properties, PCI-Express Bandwidth, GPU Stress Test,
Memory Test, etc.) and is used by datacenter operators, OEMs, and developers
to qualify ROCm deployments. RVS depends on the ROCm runtime, HIP, and
several ROCm libraries, yet its feature roadmap and release schedule are
driven by validation requirements that are independent of the ROCm Core SDK
release cadence.

### Goals

1. **Decouple extras from the ROCm release train** so that bug fixes,
   new test modules, and feature enhancements can ship without waiting for
   the next ROCm Core SDK release.
2. **Guarantee backward compatibility within a ROCm major version release stream**
   so that a single extras build works across every minor and patch release
   of that major version.
3. **Provide a clear installation layout** that coexists with the ROCm
   Core SDK directory structure defined in
   [RFC0009](/docs/rfcs/RFC0009-OS-Packaging-Requirements.md) without
   creating conflicts or ambiguity.
4. **Enable community and partner contributions** by keeping the extras
   repositories buildable against any compatible, publicly released
   ROCm Core SDK installation.

### Scope

#### In Scope

- Release cadence and versioning for extras/tools projects
- ROCm major-version compatibility contract
- Directory layout and packaging conventions for installed extras
- ROCm Validation Suite (RVS) as the reference implementation

#### Out of Scope

- ROCm Core SDK packaging (covered by RFC0009)
- GPU driver packaging
- Internal CI/CD pipeline specifics of individual extras projects
- ROCm Expansion SDK requirements need a parallel RFC to the requirements in this PR. 
- These requirements is for software that releases individually and islimited to one version installed per major ROCm version.
## 2. Release Cadence

Extras projects follow an **independent release cadence** that is not tied
to the ROCm Core SDK release schedule.

Each extras project may set an appropriate release cadence for the project in
conjunction with its stakeholders.

### Versioning Scheme

Each end-user project adopts its own semantic version which results in
following filename template:

```
<projectname>-<major>.<minor>.<patch>
```

For example:

```
rvs-1.0.0
rvs-1.1.0
rvs-1.1.1
```

The project version is entirely independent of the ROCm version it was
built against. The relationship to the ROCm platform is encoded in two
ways:

1. **Extras tree version (installation path).** All end-user projects
   compiled against a given ROCm major version are installed into a
   single shared prefix whose name carries the ROCm major version:

   ```
   /opt/rocm/extras-<rocm-major>/
   ```

   For example, `extras-7` contains every end-user project built for the
   ROCm 7.x family. This makes compatibility immediately visible from
   the filesystem path — no additional metadata lookup is required.

2. **Package name.** Each package embeds the target ROCm major version
   in its name using the `amdrocm<major>-<project>` convention, so the
   compatibility scope is apparent in repository listings, download
   pages, and CI logs:

   ```
   amdrocm7-rvs-1.0.0       # RVS 1.0.0 for ROCm 7.x
   amdrocm7-rvs-1.2.0       # RVS 1.2.0 for ROCm 7.x
   amdrocm8-rvs-2.0.0       # RVS 2.0.0 for ROCm 8.x
   ```

The combination of both signals — the install path and the package
name — ensures that whether a user is looking at an installed directory,
a package filename, or a repository index, the compatible ROCm version
family is always unambiguous.

### ROCm Compatibility Matrix

The following table illustrates how project versions map to ROCm
compatibility:

| Project version | ROCm target | Package name    | Install path          |
| :-------------- | :---------- | :-------------- | :-------------------- |
| rvs-1.0.0       | ROCm 7.x    | `amdrocm7-rvs`  | `/opt/rocm/extras-7/` |
| rvs-1.1.0       | ROCm 7.x    | `amdrocm7-rvs`  | `/opt/rocm/extras-7/` |
| rvs-1.2.0       | ROCm 7.x    | `amdrocm7-rvs`  | `/opt/rocm/extras-7/` |
| rvs-2.0.0       | ROCm 8.x    | `amdrocm8-rvs`  | `/opt/rocm/extras-8/` |

### Release Principles

| Principle                    | Description                                                                                                                                                                       |
| :--------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Independent scheduling       | Extras may release at any time — weekly, monthly, or on-demand — without coordinating with upcoming ROCm Core SDK milestones.                                                     |
| ROCm version binding         | Each release of an extras project declares the ROCm major version it targets (e.g., ROCm 7). A new ROCm major version requires a corresponding new release of the extras project. |
| Hotfix freedom               | Critical bug fixes in an extras tool can ship immediately as a patch release without waiting for a ROCm point release.                                                             |
| Nightly / pre-release builds | Extras projects may publish nightly or pre-release packages to a staging repository for early validation.                                                                          |

### Example Timeline

```
ROCm 7.0  ─────── ROCm 7.1 ─────── ROCm 7.2 ──────── ROCm 8.0
  │                  │                  │                  │
  ├─ rvs-1.0.0       │                  │                  │
  │  (extras-7)      ├─ rvs-1.1.0       │                  │
  │                  │  (extras-7)      │                  │
  │                  │    ├─ rvs-1.1.1  │                  │
  │                  │    (extras-7)    ├─ rvs-1.2.0       │
  │                  │                  │  (extras-7)      │
  │                  │                  │                  ├─ rvs-2.0.0
  │                  │                  │                  │  (extras-8)
```

In this timeline, all RVS 1.x releases land in `extras-7` and work with
any ROCm 7.x release. When ROCm 8.0 introduces breaking ABI changes, RVS
publishes a new major version (2.0.0) that installs into `extras-8`.

## 3. Compatibility with ROCm Releases

### Major-Version Compatibility Contract

All extras packages compiled against a given ROCm **major** version must
continue to work with any ROCm release that shares that major version,
whether the ROCm release is older or newer than the one used at build time.

> **Rule:** An extras package built on ROCm **X.a** must function correctly
> on ROCm **X.b** for all supported values of **a** and **b** within major
> version **X**.

Concretely for RVS:

| RVS version | Built against | Must work on           |
| :---------- | :------------ | :--------------------- |
| rvs-1.0.0   | ROCm 7.0      | ROCm 7.0, 7.1, 7.2, … |
| rvs-1.1.0   | ROCm 7.1      | ROCm 7.0, 7.1, 7.2, … |
| rvs-1.2.0   | ROCm 7.2      | ROCm 7.0, 7.1, 7.2, … |
| rvs-2.0.0   | ROCm 8.0      | ROCm 8.0, 8.1, …      |

This means an operator who upgrades their cluster from ROCm 7.0 to ROCm 7.2
does **not** need to reinstall or upgrade RVS — the existing RVS binary
continues to function. Conversely, an operator running ROCm 7.0 can install a
newer RVS release (e.g., rvs-1.2.0, originally built against ROCm 7.2) and it
will work correctly.

### How Compatibility Is Achieved

1. **Stable ABI within a ROCm major version.** The ROCm Core SDK maintains
   backward-compatible shared-library ABIs across all minor and patch releases
   within the same major version. Extras projects link against these stable
   interfaces.

2. **`$ORIGIN`-based RPATH.** Extras packages are built with `$ORIGIN`-relative
   RPATH entries so that they resolve their own bundled dependencies first,
   falling back to the ROCm installation only for ROCm-provided libraries.

3. **Minimum-version symbol binding.** Extras projects build against the
   **lowest** minor release in the target major family (e.g., ROCm 7.0) to
   ensure no dependency on symbols introduced in a later minor release. If a
   feature from a newer minor release is required, it is accessed through
   runtime detection or optional weak linking.

4. **CI validation matrix.** Each extras release is tested against the full
   set of supported ROCm minor releases within its target major version
   before publication.

### Breaking Changes Across Major Versions

When ROCm increments its major version (e.g., 7.x to 8.0), ABI stability
is **not** guaranteed. Extras projects must publish a corresponding new
major version compiled against the new ROCm major release. Packages from
different ROCm major families must not be mixed.

## 4. Installed Package Directory Structure

Extras packages install under `/opt/rocm/extras-<rocm-major>/` following
the
[ROCm Linux Filesystem Hierarchy Standard](https://rocm.docs.amd.com/en/latest/conceptual/file-reorg.html)
and the conventions established in
[RFC0009](/docs/rfcs/RFC0009-OS-Packaging-Requirements.md). The extras
tree version matches the ROCm major version it targets, so `extras-7`
contains all extras built for ROCm 7.x. The layout is **flat** — all
binaries, libraries, and headers from every extras project are merged into
this single prefix, with project-specific subfolders under `include/` and
`share/` for namespace isolation.

### Layout

```
/opt/rocm/extras-<rocm-major>/
    | -- bin
    |      | -- all public binaries across extras projects
    | -- lib
    |      | -- lib<soname>.so -> lib<soname>.so.major -> lib<soname>.so.major.minor.patch
    |      |      (public libraries to link with applications)
    |      | -- <component>
    |      |      | -- architecture dependent libraries and binaries used internally
    |      | -- cmake
    |             | -- <component>
    |                    | -- <component>-config.cmake
    | -- libexec
    |      | -- <component>
    |             | -- non-ISA/architecture independent executables used internally
    | -- include
    |      | -- <component>
    |             | -- public header files
    | -- share
           | -- html
           |      | -- <component>
           |             | -- html documentation
           | -- info
           |      | -- <component>
           |             | -- info files
           | -- man
           |      | -- <component>
           |             | -- man pages
           | -- doc
           |      | -- <component>
           |             | -- license files
           | -- <component>
                  | -- samples
                  | -- architecture independent misc files (configs, test assets)
```

Where `<rocm-major>` is the ROCm major version the extras target (e.g.,
`7` for ROCm 7.x), and `<component>` is the individual project name
(e.g., `rvs`).

### Symlinks

A soft link provides a stable unversioned path that resolves to the latest
installed extras tree:

```
/opt/rocm/extras/         -> /opt/rocm/extras-7
```

### RVS Example

A concrete installation of the `extras-7` tree containing RVS 1.2.0:

```
/opt/rocm/extras-7/
    | -- bin
    |      | -- rvs                            # Main RVS executable
    | -- lib
    |      | -- librvs.so -> librvs.so.1 -> librvs.so.1.2.0
    |      | -- rvs
    |      |      | -- libgst.so               # GPU Stress Test module
    |      |      | -- libiet.so               # Input EDPp Test module
    |      |      | -- libpeqt.so              # PCI-Express Qualification Test module
    |      |      | -- libpebb.so              # PCI-Express Bandwidth Benchmark module
    |      |      | -- libmem.so               # Memory Test module
    |      |      | -- librvs_internal.so      # Internal helpers
    |      | -- cmake
    |             | -- rvs
    |                    | -- rvs-config.cmake
    |                    | -- rvs-targets.cmake
    | -- libexec
    |      | -- rvs
    |             | -- rvs_worker               # Internal worker process
    | -- include
    |      | -- rvs
    |             | -- rvs.h
    |             | -- rvs_module.h
    | -- share
           | -- doc
           |      | -- rvs
           |             | -- LICENSE
           |             | -- CHANGELOG.md
           | -- man
           |      | -- rvs
           |             | -- rvs.1
           | -- rvs
                  | -- conf
                  |      | -- gpup_1.conf      # GPU Properties module config
                  |      | -- gst_1.conf       # GPU Stress Test config
                  |      | -- pebb_1.conf      # PCIe Bandwidth Benchmark config
                  |      | -- rvs.conf         # Global RVS configuration
                  | -- samples
                         | -- gpu_stress.py
```

When RVS releases a new version (e.g., rvs-1.3.0) while still targeting
ROCm 7, the updated files are installed in-place into the same `extras-7`
tree. The extras tree version does not change — only the individual project
binaries and libraries within it are upgraded.

Symlink:

```
/opt/rocm/extras/         -> /opt/rocm/extras-7
```

### Multiple Extras Projects

When more than one extras project is installed, all projects merge into the
same flat prefix. The project-specific subfolders under `include/` and
`share/` prevent namespace collisions:

```
/opt/rocm/extras-7/
    | -- bin
    |      | -- rvs
    |      | -- rocm-bench
    | -- lib
    |      | -- librvs.so
    |      | -- librocmbench.so
    |      | -- cmake
    |             | -- rvs
    |             |      | -- rvs-config.cmake
    |             | -- rocm-bench
    |                    | -- rocm-bench-config.cmake
    | -- include
    |      | -- rvs
    |      |      | -- rvs.h
    |      | -- rocm-bench
    |             | -- rocm_bench.h
    | -- share
           | -- doc
           |      | -- rvs
           |      |      | -- LICENSE
           |      | -- rocm-bench
           |             | -- LICENSE
           | -- rvs
           |      | -- conf
           |      | -- samples
           | -- rocm-bench
                  | -- samples
```

### Side-by-Side Installation

When multiple ROCm major versions are installed on the same system, a
separate extras tree exists for each. This allows side-by-side operation
without conflicts:

```
/opt/rocm/extras-7/
/opt/rocm/extras-8/
/opt/rocm/extras/         -> /opt/rocm/extras-8
```

Within a single extras tree (e.g., `extras-7`), project upgrades are
installed in-place — there is no side-by-side at the individual project
level. The ROCm major version is the only axis of side-by-side support.

### Environment Integration

To make extras tools discoverable, packages install a modulefile or profile
snippet:

```
/etc/profile.d/amdrocm-extras.sh
```

This adds `/opt/rocm/extras/bin` to `$PATH` and `/opt/rocm/extras/lib` to
`$LD_LIBRARY_PATH` (or the equivalent `ld.so.conf.d` drop-in). Because the
layout is flat, a single profile snippet covers all extras projects. Users
can target a specific ROCm major version by pointing directly at
`/opt/rocm/extras-7/bin` instead of the symlink, or by using environment
modules.

## 5. Packaging Formats

Extras projects must produce packaging formats that align with the formats
supported by the ROCm Core SDK in TheRock. Not every format applies to
every extras project — the table below identifies which formats are
applicable and when to use each.

### Format Overview

| Format                    | Applicable  | Primary use case                                                                                                                                                             |
| :------------------------ | :---------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **DEB** (`.deb`)          | Yes         | Installation on Debian-based distributions (Ubuntu, Debian) via `apt` / `dpkg`.                                                                                              |
| **RPM** (`.rpm`)          | Yes         | Installation on RPM-based distributions (RHEL, SLES, AlmaLinux, CentOS, Rocky, Oracle Linux) via `dnf` / `yum` / `zypper`.                                                  |
| **Tarball** (`.tar.xz`)   | Yes         | Portable, package-manager-independent installation. Useful for container images, HPC environments without root access, and CI/CD pipelines that consume pre-built artifacts. |
| **Python wheel** (`.whl`) | Conditional | Only applicable when the extras project ships a Python interface or CLI tool written in Python. Native-only projects (e.g., RVS) do not produce wheels.                      |

### DEB and RPM Packages

Extras projects follow the same native packaging workflow as the ROCm Core
SDK, as described in the
[native packaging documentation](/docs/packaging/native_packaging.md).
Each extras project provides a `package.json` that defines its package
entries, and TheRock's packaging tooling generates both versioned and
non-versioned packages.

Key conventions:

- **Naming**: Packages embed the target ROCm major version in the name
  using the `amdrocm<major>-<project>` convention. This enables
  side-by-side installation of the same project across different ROCm
  major versions, since `amdrocm7-rvs` and `amdrocm8-rvs` are distinct
  packages to the package manager. Distro-native packages use the
  corresponding `rocm<major>-<project>` convention.

  | AMD repository package | Distro-native equivalent | Contents                                                   |
  | :--------------------- | :----------------------- | :--------------------------------------------------------- |
  | `amdrocm7-rvs`         | `rocm7-rvs`              | RVS runtime for ROCm 7.x — binaries, modules, and configs |
  | `amdrocm7-rvs-devel`   | `rocm7-rvs-dev`          | RVS development headers and CMake files for ROCm 7.x      |
  | `amdrocm8-rvs`         | `rocm8-rvs`              | RVS runtime for ROCm 8.x                                  |
- **Runtime vs. development split (optional)**: Projects that expose a
  public API with headers and CMake config files may choose to produce a
  separate `-devel` / `-dev` package. Most extras projects are end-user
  tools built on top of the SDK and ship a single runtime package only.

Example (RVS for ROCm 7 on Ubuntu):

```
amdrocm7-rvs_1.2.0_amd64.deb
amdrocm7-rvs-dev_1.2.0_amd64.deb
```

Example (RVS for ROCm 7 on RHEL):

```
amdrocm7-rvs-1.2.0.x86_64.rpm
amdrocm7-rvs-devel-1.2.0.x86_64.rpm
```

Side-by-side install of RVS across ROCm 7 and ROCm 8:

```bash
apt install amdrocm7-rvs amdrocm8-rvs
```

#### Dependency Resolution

End-user projects built on top of ROCm must declare their ROCm
dependencies correctly so that package managers can resolve and install
the required ROCm components automatically. The following covers the
general principles for DEB and RPM dependency resolution and how to map
them to ROCm's host, architecture-specific, and multi-arch package
structure.

#### General Principles

DEB and RPM package managers resolve dependencies using metadata declared
in the package control file (`Depends` for DEB, `Requires` for RPM).
The following rules apply to all end-user project packages:

1. **Declare only direct dependencies.** List only the ROCm packages
   the project links against or invokes directly. Transitive dependencies
   are resolved automatically by the package manager through the ROCm
   packages' own dependency chains.

1. **Use version ranges, not exact versions.** Pin dependencies to the
   ROCm major version to honor the compatibility contract from Section 3.
   Avoid pinning to a specific minor or patch release unless a known
   minimum is required.

   DEB example (`debian/control`):

   ```
   Depends: amdrocm-runtimes (>= 7.0), amdrocm-runtimes (<< 8.0)
   ```

   RPM example (`.spec`):

   ```
   Requires: amdrocm-runtimes >= 7.0
   Requires: amdrocm-runtimes < 8.0
   ```

1. **Separate build-time and install-time dependencies.** Build
   dependencies (`Build-Depends` / `BuildRequires`) may reference
   development packages such as `amdrocm-core-devel`. Runtime packages
   should only depend on the runtime counterparts.

#### ROCm Dependency Categories

ROCm packages in TheRock are organized into three categories. End-user
projects must understand this structure to declare the right dependencies.

| Category            | Description                                                                                                                                                                                                                             | Example packages                                    |
| :------------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------- |
| **Host packages**   | Architecture-independent runtime and libraries — the host-side binaries that work regardless of which GPU is installed. Not all host packages have corresponding device packages; some (e.g., `amdrocm-runtimes`) are purely host-side. | `amdrocm-runtimes`, `amdrocm-core`, `amdrocm-base` |
| **Device packages** | GPU-architecture-specific binaries containing device code for a particular `gfx` target. Only ROCm library packages that ship pre-compiled GPU kernels produce device variants. These are suffixed with the architecture name.          | `amdrocm-blas-gfx942`, `amdrocm-blas-gfx1100`      |
| **Meta packages**   | Convenience packages that pull in a set of host + device packages for a given architecture family or the full SDK.                                                                                                                      | `amdrocm`, `amdrocm-core-sdk`, `rocm-gfx90X`       |

#### Mapping End-User Project Dependencies to ROCm

Most end-user projects depend only on **host packages** because they call
ROCm APIs (HIP, ROCm libraries) and do not ship their own GPU device
code. The device packages are an end-user installation choice that
determines which GPUs are supported at runtime — the end-user project
itself is agnostic to this selection.

**Host-only dependency (typical case)**

An end-user project like RVS links against HIP, ROCm SMI, rocBLAS, and
the ROCm runtime. Its package declares dependencies on the host packages
for each component it uses:

DEB:

```
Depends: amdrocm-runtimes (>= 7.0), amdrocm-runtimes (<< 8.0),
         amdrocm-amdsmi (>= 7.0), amdrocm-amdsmi (<< 8.0),
         amdrocm-blas (>= 7.0), amdrocm-blas (<< 8.0)
```

RPM:

```
Requires: amdrocm-runtimes >= 7.0, amdrocm-runtimes < 8.0
Requires: amdrocm-amdsmi >= 7.0, amdrocm-amdsmi < 8.0
Requires: amdrocm-blas >= 7.0, amdrocm-blas < 8.0
```

Note that `amdrocm-runtimes` is a host-only package and does not have
architecture-specific variants. Libraries like `amdrocm-blas`, however,
have corresponding device packages (`amdrocm-blas-gfx942`, etc.) that
contain pre-compiled GPU kernels for specific architectures.

The end-user project depends only on the host packages. The user is
responsible for installing the device packages for their GPU, either
individually or through an architecture family meta-package:

```bash
# Option 1: Install device packages for a specific architecture
apt install amdrocm-blas-gfx942

# Option 2: Install an architecture family meta-package (pulls in all
#            device packages for the gfx94X family)
apt install rocm-gfx94X
```

The end-user project does not need to know or declare which GPU
architectures are present.

**Architecture-specific dependency (rare case)**

If an end-user project ships its own pre-compiled GPU kernels for
specific architectures, it must produce architecture-specific package
variants and declare dependencies on the matching ROCm library device
packages:

```
Package: amdrocm7-mytool-gfx942
Depends: amdrocm-runtimes (>= 7.0), amdrocm-runtimes (<< 8.0),
         amdrocm-blas-gfx942 (>= 7.0), amdrocm-blas-gfx942 (<< 8.0)
```

Each architecture variant must:

- Not conflict with other architecture variants of the same project.
- Be independently installable.
- Follow the `<package>-<gfxarch>` naming convention from
  [RFC0009](/docs/rfcs/RFC0009-OS-Packaging-Requirements.md).

**Meta-package dependency**

If an end-user project needs the full ROCm runtime stack (not just
individual libraries), it may depend on a meta-package instead of
listing each component:

```
Depends: amdrocm-core (>= 7.0), amdrocm-core (<< 8.0)
```

This pulls in the complete ROCm Core runtime. Use this approach
sparingly — prefer fine-grained dependencies to avoid installing
unnecessary components.

#### Multi-Arch Considerations

ROCm is transitioning to a multi-arch packaging model through
[RFC0008](/docs/rfcs/RFC0008-Multi-Arch-Packaging.md), where host code
and device code are split into separate packages using `rocm-kpack`.
End-user projects should be prepared for this model:

1. **Depend on host packages only.** As ROCm splits host and device
   code into separate packages, end-user projects that follow the
   host-only dependency pattern (the typical case above) are
   automatically compatible — no changes needed.

1. **Do not assume fat binaries.** End-user projects must not assume
   that ROCm libraries contain embedded device code for all
   architectures. The device code may reside in separate architecture
   packages or kpack archives loaded at runtime.

1. **Let the user choose architectures.** The end-user project package
   should never force-install a specific GPU architecture. Architecture
   selection is the user's responsibility through device meta-packages:

   ```bash
   # User installs the end-user project + their architecture
   apt install amdrocm7-rvs
   apt install rocm-gfx90X        # User's architecture choice
   ```

1. **Test against both fat and split layouts.** During the transition
   period, CI should verify that the end-user project works correctly
   whether ROCm is installed from fat-binary packages or multi-arch
   split packages.

#### Dependency Declaration Summary

| Scenario                               | DEB `Depends`                                            | RPM `Requires`                                     |
| :------------------------------------- | :------------------------------------------------------- | :------------------------------------------------- |
| Uses HIP runtime                       | `amdrocm-runtimes (>= 7.0), amdrocm-runtimes (<< 8.0)` | `amdrocm-runtimes >= 7.0, amdrocm-runtimes < 8.0` |
| Uses a ROCm library (e.g., rocBLAS)    | `amdrocm-blas (>= 7.0), amdrocm-blas (<< 8.0)`         | `amdrocm-blas >= 7.0, amdrocm-blas < 8.0`         |
| Uses ROCm SMI                          | `amdrocm-amdsmi (>= 7.0), amdrocm-amdsmi (<< 8.0)`     | `amdrocm-amdsmi >= 7.0, amdrocm-amdsmi < 8.0`     |
| Needs full Core SDK runtime            | `amdrocm-core (>= 7.0), amdrocm-core (<< 8.0)`         | `amdrocm-core >= 7.0, amdrocm-core < 8.0`         |
| Ships own device kernels for a library | `amdrocm-<library>-<gfxarch> (>= 7.0)`                  | `amdrocm-<library>-<gfxarch> >= 7.0`              |

### Tarball Packages

Tarball archives provide a package-manager-independent distribution
channel. They are the preferred format for:

- Container images where native package managers add unnecessary layer
  complexity.
- HPC clusters with shared filesystems and no per-node root access.
- CI/CD pipelines that need to extract pre-built extras into an arbitrary
  prefix.
- Environments where `apt` / `dnf` repositories are not configured.

Tarballs follow the same artifact archive conventions as TheRock, producing
a `.tar.xz` file with a corresponding `sha256sum`. The archive extracts
into the flat FHS layout described in Section 4:

```bash
tar -xf amdrocm7-rvs-1.2.0-linux-x86_64.tar.xz \
    -C /opt/rocm/extras-7/
```

Tarball naming follows the pattern:

```
amdrocm<rocm-major>-<project>-<version>-<os>-<arch>.tar.xz
```

#### Installing ROCm Dependencies for Tarball-Based Deployments

Unlike DEB/RPM packages, tarballs do not have a package manager to
resolve and install ROCm dependencies automatically. The user must
ensure a compatible ROCm installation is present before extracting
the end-user project tarball. There are several options:

**Option 1: ROCm installed via native packages (recommended)**

If the host already has ROCm installed through `apt` or `dnf`, the
tarball-based end-user project can link against it directly. Verify
the installed ROCm major version matches:

```bash
# Check the installed ROCm version
amd-smi version

# Extract the end-user project into the extras tree
tar -xf amdrocm7-rvs-1.2.0-linux-x86_64.tar.xz \
    -C /opt/rocm/extras-7/
```

**Option 2: ROCm installed from tarball**

ROCm itself can be deployed from a tarball into a custom prefix. In
this case both ROCm and the end-user project are package-manager-free:

```bash
# Extract ROCm Core SDK tarball
tar -xf amdrocm-core-7.1.0-linux-x86_64.tar.xz \
    -C /opt/rocm/core-7/

# Extract the end-user project
tar -xf amdrocm7-rvs-1.2.0-linux-x86_64.tar.xz \
    -C /opt/rocm/extras-7/
```

**Option 3: ROCm installed via Python packages**

When ROCm is available through `pip install rocm[core]`, the SDK
libraries reside inside the Python site-packages directory. Use
`rocm-sdk path` to locate the installation:

```bash
ROCM_ROOT=$(rocm-sdk path --root)

tar -xf amdrocm7-rvs-1.2.0-linux-x86_64.tar.xz \
    -C /opt/rocm/extras-7/
```

#### Connecting Project Binaries with ROCm Libraries

After extracting both ROCm and the end-user project, the project
binaries must be able to locate ROCm shared libraries at runtime.
The following options are available, listed from most to least
preferred:

**Option 1: `$ORIGIN`-based RPATH (built-in, no user action)**

End-user project binaries are built with `$ORIGIN`-relative RPATH
entries. If the end-user project and ROCm are installed under the
same `/opt/rocm/` parent, the relative paths resolve automatically
and no additional configuration is needed.

**Option 2: `LD_LIBRARY_PATH` environment variable**

Set `LD_LIBRARY_PATH` to include the ROCm library directory. This is
the simplest option when ROCm is installed in a non-standard location
or when `$ORIGIN` RPATH does not cover the layout:

```bash
export ROCM_PATH=/opt/rocm/core-7
export LD_LIBRARY_PATH=${ROCM_PATH}/lib:${LD_LIBRARY_PATH}
export PATH=/opt/rocm/extras-7/bin:${ROCM_PATH}/bin:${PATH}
```

**Option 3: `ld.so.conf.d` drop-in (system-wide, requires root)**

Create a linker configuration file so the dynamic linker finds ROCm
libraries without environment variables:

```bash
echo "/opt/rocm/core-7/lib" > /etc/ld.so.conf.d/rocm-7.conf
ldconfig
```

**Option 4: Environment modules / Lmod**

On HPC clusters, use environment modules to manage ROCm and extras
paths per user or per job:

```tcl
# /opt/modulefiles/rocm-extras/7
set     rocm_root    /opt/rocm/core-7
set     extras_root  /opt/rocm/extras-7

prepend-path  PATH             $extras_root/bin
prepend-path  PATH             $rocm_root/bin
prepend-path  LD_LIBRARY_PATH  $extras_root/lib
prepend-path  LD_LIBRARY_PATH  $rocm_root/lib
prepend-path  CMAKE_PREFIX_PATH $extras_root
prepend-path  CMAKE_PREFIX_PATH $rocm_root
```

Usage:

```bash
module load rocm-extras/7
rvs -d 1
```

#### Verifying the Setup

After installation and environment configuration, verify that the
end-user project binaries can find all required ROCm libraries:

```bash
# Check that all shared library dependencies resolve
ldd /opt/rocm/extras-7/bin/rvs

# Run the project's built-in sanity check (if available)
rvs -d 1 -g
```

Any `not found` entries in the `ldd` output indicate a missing ROCm
library or an incorrectly configured library search path.

### Python Wheel Packages

Wheels are applicable **only** when an extras project provides a Python
API, CLI tool, or test harness written in Python. Native-only extras
projects like RVS do not produce wheels.

When applicable, extras wheels follow the conventions described in the
[Python packaging documentation](/docs/packaging/python_packaging.md):

- **Selector package**: A source distribution (`rocm<major>-<project>`)
  that evaluates install-time constraints and pulls the correct runtime
  wheels.
- **Runtime wheels**: Contain the minimal set of files needed to run,
  with no symlinks. SONAME libraries are included directly.
- **Devel wheels**: Catch-all for headers, CMake files, and symlinks
  stored in a `_devel.tar` inside the wheel.

Wheels declare a dependency on the ROCm SDK Python packages for the
matching major version:

```
Requires-Dist: rocm[core] >=7.0, <8.0
```

Example build:

```bash
build_tools/build_python_packages.py \
    --artifact-dir ./ARTIFACTS_DIR \
    --dest-dir ./OUTPUT_PKG
```

Example install:

```bash
pip install rocm7-mytool --pre \
    --find-links=./OUTPUT_PKG/dist
```

### Format Selection Guide

The following table summarizes which format to produce based on the
project characteristics:

| Project type                  | DEB/RPM | Tarball | Wheel |
| :---------------------------- | :-----: | :-----: | :---: |
| Native C/C++ tool (e.g., RVS) |   Yes   |   Yes   |  No   |
| Pure Python tool/test harness |   No    |   No    |  Yes  |
| Mixed native + Python library |   Yes   |   Yes   |  Yes  |
