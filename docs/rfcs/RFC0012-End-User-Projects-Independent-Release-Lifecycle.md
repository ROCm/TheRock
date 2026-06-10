---
author: Freddy Paul, Saad Rahim (saadrahim)
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
- **ROCm Expansion SDKs** (e.g., the HPC SDK proposed in
  [PR #5613](https://github.com/ROCm/TheRock/pull/5613)). Expansions are a
  distinct category and require a parallel RFC. The key difference is the
  version model: an **expansion pins to and installs a single ROCm version**,
  whereas the **extras** described here are explicitly designed to support
  **multiple ROCm major versions side by side** within a single
  `/opt/rocm/extras/` tree (e.g. the `rvs7` and `rvs8` binaries coexist).
  The requirements in this RFC do not govern expansions.

### Related RFCs

This RFC defines the **build, release, versioning, and install layout** for
extras. It is single-responsibility by design and relies on the following
companion RFCs:

- [**RFC0012 — ROCm Repository Structure** (PR #4414)](https://github.com/ROCm/TheRock/pull/4414):
  defines the *distribution* layout on `repo.amd.com`, including the single
  `extras/` folder with per-project subfolders that this RFC's packages are
  published into.
- [**RFC00XX — ROCm Repository Setup Packages**](RFC00XX-Repository-Package.md):
  defines the native `amdrocm-repo-*` rpm/deb tier packages that configure the
  AMD repositories from which these extras are installed.
- [**RFC0009 — OS Packaging Requirements**](/docs/rfcs/RFC0009-OS-Packaging-Requirements.md):
  the native packaging conventions (FHS layout, `-gfxarch` naming) that extras
  packages inherit.
- [**RFC0008 — Multi-Arch Packaging**](/docs/rfcs/RFC0008-Multi-Arch-Packaging.md):
  the host/device package split that extras dependency declarations must be
  compatible with.
- [**PR #5613 — ROCm Expansion SDK / HPC SDK**](https://github.com/ROCm/TheRock/pull/5613):
  the separate expansions track (single ROCm version per install), which is
  explicitly out of scope here (see Out of Scope above).

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

1. **Versioned binary within a single extras tree.** All end-user
   projects install into one shared prefix — `/opt/rocm/extras/` — with
   no per-major directory. The public executable is suffixed with the
   ROCm major (`rvs7`, `rvs8`) and an unsuffixed symlink points at the
   latest installed major (`rvs` → `rvs7`), so builds for different ROCm
   majors coexist in the same tree. Shared libraries use **ordinary,
   project-version-based SONAMEs** (`librvs.so` → `librvs.so.1` →
   `librvs.so.1.2.0`) — the ROCm major is **not** encoded in the `.so`
   version. Because an extra bumps its own major version when it retargets
   a new ROCm major (RVS 1.x for ROCm 7.x, RVS 2.x for ROCm 8.x), the
   library SONAMEs naturally differ across ROCm majors and coexist via
   standard linker versioning.

2. **Package name.** Each package embeds the target ROCm major version
   in its name using the `amdrocm<major>-<project>` convention, so the
   compatibility scope is apparent in repository listings, download
   pages, and CI logs:

   ```
   amdrocm7-rvs-1.0.0       # RVS 1.0.0 for ROCm 7.x
   amdrocm7-rvs-1.2.0       # RVS 1.2.0 for ROCm 7.x
   amdrocm8-rvs-2.0.0       # RVS 2.0.0 for ROCm 8.x
   ```

The combination of both signals — the versioned artifact names and the
package name — ensures that whether a user is looking at an installed
binary, a package filename, or a repository index, the compatible ROCm
version family is always unambiguous.

### ROCm Compatibility Matrix

The following table illustrates how project versions map to ROCm
compatibility:

| Project version | ROCm target | Package name    | Installed binary               |
| :-------------- | :---------- | :-------------- | :----------------------------- |
| rvs-1.0.0       | ROCm 7.x    | `amdrocm7-rvs`  | `/opt/rocm/extras/bin/rvs7`    |
| rvs-1.1.0       | ROCm 7.x    | `amdrocm7-rvs`  | `/opt/rocm/extras/bin/rvs7`    |
| rvs-1.2.0       | ROCm 7.x    | `amdrocm7-rvs`  | `/opt/rocm/extras/bin/rvs7`    |
| rvs-2.0.0       | ROCm 8.x    | `amdrocm8-rvs`  | `/opt/rocm/extras/bin/rvs8`    |

All four install into the same `/opt/rocm/extras/` tree; the ROCm major
is carried by the binary suffix (`rvs7` / `rvs8`), not a per-major
directory.

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
  │  (bin/rvs7)      ├─ rvs-1.1.0       │                  │
  │                  │  (bin/rvs7)      │                  │
  │                  │    ├─ rvs-1.1.1  │                  │
  │                  │    (bin/rvs7)    ├─ rvs-1.2.0       │
  │                  │                  │  (bin/rvs7)      │
  │                  │                  │                  ├─ rvs-2.0.0
  │                  │                  │                  │  (bin/rvs8)
```

In this timeline, all RVS 1.x releases install as `rvs7` and work with
any ROCm 7.x release. When ROCm 8.0 introduces breaking ABI changes, RVS
publishes a new major version (2.0.0) that installs as `rvs8` alongside
`rvs7` in the same `/opt/rocm/extras/` tree.

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

Extras packages install under a single `/opt/rocm/extras/` prefix following
the
[ROCm Linux Filesystem Hierarchy Standard](https://rocm.docs.amd.com/en/latest/conceptual/file-reorg.html)
and the conventions established in
[RFC0009](/docs/rfcs/RFC0009-OS-Packaging-Requirements.md). There is **no
per-major directory** — every extra, regardless of which ROCm major it
targets, installs into this one tree. Side-by-side support across ROCm
majors is provided by a **versioned binary name**: the public executable
is suffixed with the ROCm major (`rvs7`, `rvs8`), and an unsuffixed
symlink (`rvs` → `rvs7`) resolves to the latest installed major. Shared
libraries use **ordinary project-version SONAMEs** (`librvs.so` →
`librvs.so.1` → `librvs.so.1.2.0`); the ROCm major is **not** encoded in
the `.so` version. The layout is otherwise **flat** — public binaries and
libraries from every extras project share the common `bin/` and `lib/`,
with project-specific subfolders under `include/`, `share/`, and the
private `lib/`/`libexec/` component directories for namespace isolation.

> **Distribution vs. install layout.** This section describes the *installed*
> on-disk layout. The *distribution* layout on the AMD repository
> (`repo.amd.com`) is a separate concern, defined in
> [RFC0012 — Repository Structure (PR #4414)](https://github.com/ROCm/TheRock/pull/4414):
> extras are distributed from a single `extras/` folder with per-project
> subfolders, and the ROCm major version is carried in the *package name*
> (`amdrocm<major>-<project>`) rather than the repository path. On the
> installed side, the ROCm major is carried by the binary suffix
> (`/opt/rocm/extras/bin/rvs7`) rather than a per-major install prefix;
> libraries use ordinary project-version SONAMEs.

### Layout

```
/opt/rocm/extras/
    | -- bin
    |      | -- <component><rocm-major>          # versioned public binary (e.g. rvs7)
    |      | -- <component> -> <component><rocm-major>   # symlink to the latest major
    | -- lib
    |      | -- lib<soname>.so -> lib<soname>.so.<proj-major> -> lib<soname>.so.<proj-ver>   # ordinary project-version SONAME (no ROCm major)
    |      | -- <component><rocm-major>          # private arch-dependent libs/modules, major-scoped so runtimes coexist
    |      | -- cmake
    |             | -- <component>
    |                    | -- <component>-config.cmake   # dev files (resolve to the latest major)
    | -- libexec
    |      | -- <component><rocm-major>          # private executables used internally, major-scoped
    | -- include
    |      | -- <component>
    |             | -- public header files       # dev files (latest major)
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

Where `<rocm-major>` is the ROCm major version the extra targets (e.g.,
`7` for ROCm 7.x), `<component>` is the individual project name (e.g.,
`rvs`), `<proj-major>` is the project's own SONAME major, and `<proj-ver>`
is the project's full semantic version (e.g., `1.2.0`). Only the `bin/`
executable and the private `lib/`/`libexec/` component directories carry
the ROCm major (so multiple majors coexist); the public `lib*.so` uses an
ordinary project-version SONAME with **no** ROCm major, and
development-only files (`include/`, `lib/cmake/`) stay per-project and
resolve to the latest installed major.

### Symlinks

Unversioned symlinks provide stable paths that resolve to the latest
installed ROCm major:

```
/opt/rocm/extras/bin/rvs        -> rvs7
```

Shared libraries keep their conventional development symlink resolving to
the project SONAME (not the ROCm major):

```
/opt/rocm/extras/lib/librvs.so  -> librvs.so.1 -> librvs.so.1.2.0
```

### RVS Example

A concrete installation containing RVS 1.2.0 built for ROCm 7.x:

```
/opt/rocm/extras/
    | -- bin
    |      | -- rvs7                           # Main RVS executable (ROCm 7.x)
    |      | -- rvs -> rvs7                     # symlink to the latest major
    | -- lib
    |      | -- librvs.so -> librvs.so.1 -> librvs.so.1.2.0   # ordinary project SONAME (no ROCm major)
    |      | -- rvs7                            # private RVS modules for ROCm 7.x
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
    |      | -- rvs7
    |             | -- rvs_worker               # Internal worker process (ROCm 7.x)
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
ROCm 7, the updated files replace the `rvs7` binary in-place — the
ROCm-major suffix does not change — and the library advances its ordinary
project SONAME (`librvs.so.1.3.0`). The `rvs` symlink continues to point
at `rvs7` until a newer ROCm major is installed.

### Multiple Extras Projects

When more than one extras project is installed, all projects merge into the
same flat prefix. Each public binary keeps its ROCm-major suffix, and the
project-specific subfolders under `include/` and `share/` prevent namespace
collisions:

```
/opt/rocm/extras/
    | -- bin
    |      | -- rvs7
    |      | -- rvs -> rvs7
    |      | -- rocm-bench7
    |      | -- rocm-bench -> rocm-bench7
    | -- lib
    |      | -- librvs.so -> librvs.so.1 -> librvs.so.1.2.0
    |      | -- librocmbench.so -> librocmbench.so.3 -> librocmbench.so.3.0.0
    |      | -- rvs7
    |      | -- rocm-bench7
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

When multiple ROCm major versions are installed on the same system, both
sets of artifacts coexist within the **same** `/opt/rocm/extras/` tree —
there is no per-major directory. The ROCm-major suffix on each binary
(and on the private module directory) keeps the executables from
colliding. The public libraries do **not** carry the ROCm major; they
coexist through ordinary SONAME versioning because an extra bumps its own
major when it retargets a new ROCm major (RVS 1.x → ROCm 7, RVS 2.x →
ROCm 8):

```
/opt/rocm/extras/
    | -- bin
    |      | -- rvs7
    |      | -- rvs8
    |      | -- rvs -> rvs8                      # latest ROCm major wins
    | -- lib
           | -- librvs.so.1 -> librvs.so.1.2.0   # RVS 1.x (built for ROCm 7)
           | -- librvs.so.2 -> librvs.so.2.0.0   # RVS 2.x (built for ROCm 8)
           | -- librvs.so -> librvs.so.2         # newest project SONAME
           | -- rvs7                             # ROCm 7 private modules
           | -- rvs8                             # ROCm 8 private modules
```

Within a single ROCm major (e.g. `rvs7`), project upgrades are installed
in-place — there is no side-by-side at the individual project level. The
ROCm major version is the only axis of side-by-side support. Development
files (`include/`, `lib/cmake/`) are not major-suffixed and reflect the
most recently installed major; consumers needing headers for a specific
older major should install that major last or use a dedicated environment.

### Environment Integration

To make extras tools discoverable, packages install a modulefile or profile
snippet:

```
/etc/profile.d/amdrocm-extras.sh
```

This adds `/opt/rocm/extras/bin` to `$PATH` and `/opt/rocm/extras/lib` to
`$LD_LIBRARY_PATH` (or the equivalent `ld.so.conf.d` drop-in). Because all
majors share the one tree, a single profile snippet covers every extras
project and every ROCm major. Invoking `rvs` runs the latest installed
major (via the `rvs` → `rvs7` symlink); to target a specific major, call
the suffixed binary directly (`rvs7`, `rvs8`).

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
    -C /opt/rocm/extras/
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
    -C /opt/rocm/extras/
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
    -C /opt/rocm/extras/
```

**Option 3: ROCm installed via Python packages**

When ROCm is available through `pip install rocm[core]`, the SDK
libraries reside inside the Python site-packages directory. Use
`rocm-sdk path` to locate the installation:

```bash
ROCM_ROOT=$(rocm-sdk path --root)

tar -xf amdrocm7-rvs-1.2.0-linux-x86_64.tar.xz \
    -C /opt/rocm/extras/
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
export PATH=/opt/rocm/extras/bin:${ROCM_PATH}/bin:${PATH}
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
set     extras_root  /opt/rocm/extras

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
ldd /opt/rocm/extras/bin/rvs7

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
[Python packaging documentation](/docs/packaging/python_packaging.md).

> **Wheel naming differs from native packages.** Unlike DEB/RPM packages,
> a Python wheel **does not embed the ROCm major version in its distribution
> name**. The wheel is named `rocm-<project>` (e.g., `rocm-rvs`), and the
> ROCm major version is carried by metadata instead. This follows Python
> packaging norms — PyPI treats `rocm7-rvs` and `rocm8-rvs` as unrelated
> projects, which breaks `pip`'s upgrade and conflict resolution — and avoids
> squatting a new PyPI name for every ROCm major.

The target ROCm major version is encoded by **doing both** of the following,
so the binding is visible to humans *and* enforced by the resolver:

1. **PEP 440 local-version segment.** Each wheel carries a `+rocm<major>`
   local-version tag, so the compatibility target is visible in the version
   string and in `pip list`:

   ```
   rocm_rvs-1.2.0+rocm7-py3-none-linux_x86_64.whl   # RVS 1.2.0 for ROCm 7.x
   rocm_rvs-2.0.0+rocm8-py3-none-linux_x86_64.whl   # RVS 2.0.0 for ROCm 8.x
   ```

2. **Non-overlapping `Requires-Dist` range.** The wheel declares a dependency
   on the ROCm SDK Python packages pinned to a single major version. The
   ranges across majors never overlap, so `pip` can only resolve a wheel
   against a matching ROCm install:

   ```
   Requires-Dist: rocm[core] >=7.0, <8.0
   ```

- **Runtime wheels**: Contain the minimal set of files needed to run,
  with no symlinks. SONAME libraries are included directly.
- **Devel wheels**: Catch-all for headers, CMake files, and symlinks
  stored in a `_devel.tar` inside the wheel.

**PyPI name reservation.** To prevent dependency-confusion attacks, the
`rocm-<project>` name should be reserved (registered) on public PyPI even if
the wheels are primarily distributed through AMD's own index. This blocks a
malicious actor from publishing a same-named package that `pip` might prefer.

**Pinning across a fleet.** `pip` has no `apt-mark hold` equivalent, so
fleet-wide version control is done with a constraints file:

```
# rocm-constraints.txt
rocm-rvs==1.2.0+rocm7
```

```bash
pip install -c rocm-constraints.txt rocm-rvs
```

Example build:

```bash
build_tools/build_python_packages.py \
    --artifact-dir ./ARTIFACTS_DIR \
    --dest-dir ./OUTPUT_PKG
```

Example install (resolver selects the major via `Requires-Dist` against the
installed ROCm):

```bash
pip install rocm-rvs --pre \
    --find-links=./OUTPUT_PKG/dist
```

Example install (explicitly pinned to a ROCm major):

```bash
pip install "rocm-rvs==1.2.0+rocm7" --pre \
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
