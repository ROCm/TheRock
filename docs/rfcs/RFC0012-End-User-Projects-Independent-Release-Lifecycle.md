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
released ROCm installation but follow their own development and release
timelines. These extras are not part of the ROCm Core SDK; they consume
ROCm as a build dependency and are delivered as independently versioned
packages.

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
2. **Guarantee backward compatibility within a ROCm major version** so
   that a single extras build works across every minor and patch release
   of that major version.
3. **Provide a clear installation layout** that coexists with the ROCm
   Core SDK directory structure defined in
   [RFC0009](/docs/rfcs/RFC0009-OS-Packaging-Requirements.md) without
   creating conflicts or ambiguity.
4. **Enable community and partner contributions** by keeping the extras
   repositories buildable against any compatible, publicly released ROCm
   installation.

### Scope

#### In Scope

- Release cadence and versioning for extras/tools projects
- ROCm major-version compatibility contract
- Directory layout and packaging conventions for installed extras
- ROCm Validation Suite (RVS) as the reference implementation

#### Out of Scope

- ROCm Core SDK packaging (covered by RFC0009)
- GPU driver packaging
- Python / pip / wheel packaging for extras
- Internal CI/CD pipeline specifics of individual extras projects

## 2. Release Cadence

Extras projects follow an **independent release cadence** that is not tied
to the ROCm Core SDK release schedule.

### Versioning Scheme

Each end-user project adopts its own semantic version:

```
<project>-<major>.<minor>.<patch>
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

2. **Package version string.** Each package embeds the target ROCm major
   version in its release tag so that the compatibility scope is apparent
   even outside the filesystem context (e.g., in repository listings,
   download pages, and CI logs):

   ```
   <project>-<major>.<minor>.<patch>-rocm<rocm-major>
   ```

   Examples:

   ```
   rvs-1.0.0-rocm7      # RVS 1.0.0 compatible with ROCm 7.x
   rvs-1.2.0-rocm7      # RVS 1.2.0 compatible with ROCm 7.x
   rvs-2.0.0-rocm8      # RVS 2.0.0 compatible with ROCm 8.x
   ```

The combination of both signals ensures that whether a user is looking at
an installed directory, a package filename, or a repository index, the
compatible ROCm version family is always unambiguous.

### ROCm Compatibility Matrix

The following table illustrates how project versions map to ROCm
compatibility:

| Project version | ROCm target | Package release tag | Install path |
| :--- | :--- | :--- | :--- |
| rvs-1.0.0 | ROCm 7.x | `rvs-1.0.0-rocm7` | `/opt/rocm/extras-7/` |
| rvs-1.1.0 | ROCm 7.x | `rvs-1.1.0-rocm7` | `/opt/rocm/extras-7/` |
| rvs-1.2.0 | ROCm 7.x | `rvs-1.2.0-rocm7` | `/opt/rocm/extras-7/` |
| rvs-2.0.0 | ROCm 8.x | `rvs-2.0.0-rocm8` | `/opt/rocm/extras-8/` |

### Release Principles

| Principle | Description |
| :--- | :--- |
| Independent scheduling | Extras may release at any time — weekly, monthly, or on-demand — without coordinating with upcoming ROCm Core SDK milestones. |
| ROCm version binding | Each release of an extras project declares the ROCm major version it targets (e.g., ROCm 7). A new ROCm major version requires a corresponding new release of the extras project. |
| Hotfix freedom | Critical bug fixes in an extras tool can ship immediately as a patch release without waiting for a ROCm point release. |
| Nightly / pre-release builds | Extras projects may publish nightly or pre-release packages to a staging repository for early validation. |

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

| RVS version | Built against | Must work on |
| :--- | :--- | :--- |
| rvs-1.0.0 | ROCm 7.0 | ROCm 7.0, 7.1, 7.2, … |
| rvs-1.1.0 | ROCm 7.1 | ROCm 7.0, 7.1, 7.2, … |
| rvs-1.2.0 | ROCm 7.2 | ROCm 7.0, 7.1, 7.2, … |
| rvs-2.0.0 | ROCm 8.0 | ROCm 8.0, 8.1, … |

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

| Format | Applicable | Primary use case |
| :--- | :--- | :--- |
| **DEB** (`.deb`) | Yes | Installation on Debian-based distributions (Ubuntu, Debian) via `apt` / `dpkg`. |
| **RPM** (`.rpm`) | Yes | Installation on RPM-based distributions (RHEL, SLES, AlmaLinux, CentOS, Rocky, Oracle Linux) via `dnf` / `yum` / `zypper`. |
| **Tarball** (`.tar.xz`) | Yes | Portable, package-manager-independent installation. Useful for container images, HPC environments without root access, and CI/CD pipelines that consume pre-built artifacts. |
| **Python wheel** (`.whl`) | Conditional | Only applicable when the extras project ships a Python interface or CLI tool written in Python. Native-only projects (e.g., RVS) do not produce wheels. |

### DEB and RPM Packages

Extras projects follow the same native packaging workflow as the ROCm Core
SDK, as described in the
[native packaging documentation](/docs/packaging/native_packaging.md).
Each extras project provides a `package.json` that defines its package
entries, and TheRock's packaging tooling generates both versioned and
non-versioned packages.

Key conventions:

- **Naming**: Packages use the `amdrocm-<project>` prefix for AMD
  repository distribution, matching the convention in RFC0009. Distro-native
  packages (e.g., those maintained by Ubuntu or Red Hat) use
  `rocm-<project>` without the `amd` prefix.

  | AMD repository package | Distro-native equivalent | Contents |
  | :--- | :--- | :--- |
  | `amdrocm-rvs` | `rocm-rvs` | RVS runtime — binaries, modules, and configs |
  | `amdrocm-rvs-devel` | `rocm-rvs-dev` | RVS development headers and CMake files |
- **Runtime vs. development split (optional)**: Projects that expose a
  public API with headers and CMake config files may choose to produce a
  separate `-devel` / `-dev` package. Most extras projects are end-user
  tools built on top of the SDK and ship a single runtime package only.
- **RPATH**: Packages are built with `$ORIGIN`-based RPATH. The
  `--rpath-pkg` option produces versioned-only packages when relocatable
  installs are required.

Example (RVS on Ubuntu):

```
amdrocm-rvs_1.2.0-7_amd64.deb
amdrocm-rvs-dev_1.2.0-7_amd64.deb
```

Example (RVS on RHEL):

```
amdrocm-rvs-1.2.0-7.x86_64.rpm
amdrocm-rvs-devel-1.2.0-7.x86_64.rpm
```

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
tar -xf amdrocm-rvs-1.2.0-rocm7-linux-x86_64.tar.xz \
    -C /opt/rocm/extras-7/
```

Tarball naming follows the pattern:

```
amdrocm-<project>-<version>-rocm<major>-<os>-<arch>.tar.xz
```

### Python Wheel Packages

Wheels are applicable **only** when an extras project provides a Python
API, CLI tool, or test harness written in Python. Native-only extras
projects like RVS do not produce wheels.

When applicable, extras wheels follow the conventions described in the
[Python packaging documentation](/docs/packaging/python_packaging.md):

- **Selector package**: A source distribution (`rocm-<project>`)
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
pip install rocm-mytool --pre \
    --find-links=./OUTPUT_PKG/dist
```

### Format Selection Guide

The following table summarizes which format to produce based on the
project characteristics:

| Project type | DEB/RPM | Tarball | Wheel |
| :--- | :---: | :---: | :---: |
| Native C/C++ tool (e.g., RVS) | Yes | Yes | No |
| Pure Python tool/test harness | No | No | Yes |
| Mixed native + Python library | Yes | Yes | Yes |
