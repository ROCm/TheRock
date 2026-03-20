# Debian and RPM Packaging

This document provides a deep dive into how ROCm native Linux packages (`.deb`
and `.rpm`) are generated, the package structure, metadata, and installation
behavior.

Table of contents:

- [Overview](#overview)
- [Package Inventory](#package-inventory)
  - [Core Packages](#core-packages)
  - [Library Packages](#library-packages)
  - [Development Packages](#development-packages)
  - [Tool and Profiler Packages](#tool-and-profiler-packages)
  - [Meta Packages](#meta-packages)
- [Generation Pipeline](#generation-pipeline)
  - [Step 1: Artifact Download](#step-1-artifact-download)
  - [Step 2: Package Definition Loading](#step-2-package-definition-loading)
  - [Step 3: File Population](#step-3-file-population)
  - [Step 4: Metadata Generation](#step-4-metadata-generation)
  - [Step 5: Package Building](#step-5-package-building)
- [Versioned vs Non-Versioned Packages](#versioned-vs-non-versioned-packages)
- [RPATH Packages](#rpath-packages)
- [package.json Structure](#packagejson-structure)
  - [Mandatory Fields](#mandatory-fields)
  - [Optional Fields](#optional-fields)
  - [Example Entry](#example-entry)
- [Jinja2 Templates](#jinja2-templates)
  - [Debian Templates](#debian-templates)
  - [RPM Templates](#rpm-templates)
  - [Maintainer Scripts](#maintainer-scripts)
- [Installed Directory Structure](#installed-directory-structure)
  - [DEB Installation Layout](#deb-installation-layout)
  - [RPM Installation Layout](#rpm-installation-layout)
  - [Component Placement Map](#component-placement-map)
- [DEB vs RPM Differences](#deb-vs-rpm-differences)
- [Comparison with Other Formats](#comparison-with-other-formats)
- [CI Workflow Integration](#ci-workflow-integration)
- [Building Packages Locally](#building-packages-locally)
  - [Prerequisites](#prerequisites)
  - [Building DEB Packages](#building-deb-packages)
  - [Building RPM Packages](#building-rpm-packages)
  - [Building RPATH Packages](#building-rpath-packages)
- [Relevant Source Files](#relevant-source-files)

## Overview

Native Linux packages integrate ROCm with system package managers (`apt` for
Debian/Ubuntu, `dnf`/`yum` for RHEL/SLES/AlmaLinux). Each ROCm component maps
to a named package prefixed with `amdrocm-` (e.g. `amdrocm-blas`,
`amdrocm-llvm`), with proper dependency declarations, versioned install paths,
and debug symbol packages.

The native packaging system:
- Consumes the **same artifact directories** as tarballs and Python wheels.
- Produces both **versioned** and **non-versioned** package variants.
- Supports optional **RPATH** mode for self-contained library resolution.
- Uses **Jinja2 templates** to generate Debian control files and RPM spec files.
- Is defined declaratively via a central **`package.json`** file.

## Package Inventory

The following packages are defined in `package.json`. Packages marked with
`DisablePackaging` are defined but not currently built.

### Core Packages

| Package | Description | Artifact Source |
| --- | --- | --- |
| `amdrocm-base` | ROCm base tools and structural components | `base` |
| `amdrocm-core` | Composite: base + runtime + LLVM + sysdeps | Multiple |
| `amdrocm-runtime` | Low-level runtime (ROCr, HSA) | `core-runtime` |
| `amdrocm-runtime-devel` | Runtime development headers and configs | `core-runtime` |
| `amdrocm-llvm` | AMD LLVM compiler toolchain | `amd-llvm` |
| `amdrocm-llvm-devel` | LLVM development files | `amd-llvm` |
| `amdrocm-sysdeps` | Private system dependencies | `sysdeps` |
| `amdrocm-amdsmi` | AMD System Management Interface | `core-amdsmi` |
| `amdrocm-opencl` | OpenCL runtime | `core-ocl` |
| `amdrocm-opencl-devel` | OpenCL development files | `core-ocl` |

### Library Packages

| Package | Description | Artifact Source |
| --- | --- | --- |
| `amdrocm-blas` | rocBLAS, hipBLAS, hipBLASLt | `blas` |
| `amdrocm-blas-devel` | BLAS development files | `blas` |
| `amdrocm-blas-test` | BLAS test suite | `blas` |
| `amdrocm-fft` | rocFFT, hipFFT | `fft` |
| `amdrocm-fft-devel` | FFT development files | `fft` |
| `amdrocm-fft-test` | FFT test suite | `fft` |
| `amdrocm-rand` | rocRAND, hipRAND | `rand` |
| `amdrocm-rand-devel` | Random development files | `rand` |
| `amdrocm-rand-test` | Random test suite | `rand` |
| `amdrocm-sparse` | rocSPARSE, hipSPARSE, hipSPARSELt | `blas` |
| `amdrocm-sparse-devel` | Sparse development files | `blas` |
| `amdrocm-sparse-test` | Sparse test suite | `blas` |
| `amdrocm-solver` | rocSOLVER, hipSOLVER | `blas` |
| `amdrocm-solver-devel` | Solver development files | `blas` |
| `amdrocm-solver-test` | Solver test suite | `blas` |
| `amdrocm-rccl` | RCCL collective communications | `rccl` |
| `amdrocm-rccl-devel` | RCCL development files | `rccl` |
| `amdrocm-rccl-test` | RCCL test suite | `rccl` |
| `amdrocm-dnn` | MIOpen deep learning library | `MIOpen` |
| `amdrocm-dnn-devel` | MIOpen development files | `MIOpen` |
| `amdrocm-dnn-test` | MIOpen test suite | `MIOpen` |
| `amdrocm-ck` | Composable Kernel library | `composable_kernel` |
| `amdrocm-ck-devel` | Composable Kernel development files | `composable_kernel` |
| `amdrocm-decode` | Video decode library | `rocdecode` |
| `amdrocm-decode-devel` | Video decode development files | `rocdecode` |
| `amdrocm-jpeg` | JPEG decode library | `rocjpeg` |
| `amdrocm-jpeg-devel` | JPEG decode development files | `rocjpeg` |

### Development Packages

| Package | Description |
| --- | --- |
| `amdrocm-hipblas-common-devel` | Common hipBLAS headers |
| `amdrocm-math-common` | Shared math library resources |
| `amdrocm-core-devel` | Composite: all core development files |

### Tool and Profiler Packages

| Package | Description | Artifact Source |
| --- | --- | --- |
| `amdrocm-profiler-base` | Profiler base components | `rocprofiler-sdk` |
| `amdrocm-profiler` | ROCm profiler tools | `rocprofiler-sdk` |
| `amdrocm-hipify` | CUDA-to-HIP translation tools | `hipify` |
| `amdrocm-debugger` | ROCm debugger | `rocgdb` |
| `amdrocm-rdc` | ROCm Data Center tools | `rdc` |

### Meta Packages

| Package | Description |
| --- | --- |
| `amdrocm` | Top-level meta-package (all of ROCm) |
| `amdrocm-core-sdk` | Core SDK meta-package |
| `amdrocm-developer-tools` | Developer tools meta-package |
| `amdrocm-fortran` | Fortran support meta-package |
| `amdrocm-opencl-meta` | OpenCL meta-package |
| `amdrocm-openmp` | OpenMP support meta-package |

## Generation Pipeline

### Step 1: Artifact Download

The CI workflow (`build_native_linux_packages.yml`) downloads artifact archives
from a prior build run using `fetch_artifacts.py`:

```bash
python build_tools/fetch_artifacts.py \
    --run-id $ARTIFACT_RUN_ID \
    --artifact-group gfx94X-dcgpu \
    --output-dir ./artifacts
```

### Step 2: Package Definition Loading

`build_package.py` loads `package.json` and filters packages based on:
- Available artifacts in the `--artifacts-dir`.
- Optional `--pkg-names` filter for building a subset.
- The `DisablePackaging` flag in package definitions.

### Step 3: File Population

For each package, the script:

1. Reads the `Artifact` and `Components` fields to identify source directories.
2. Copies files from artifact subdirectories into the package staging area.
3. Organizes files under the install prefix (default
   `/opt/rocm/core/<major>.<minor>/`).
4. For **composite packages** (`Composite: true`), combines files from multiple
   artifacts or uses an `Includes` list to aggregate other packages.

### Step 4: Metadata Generation

Jinja2 templates are rendered with package metadata from `package.json`:

**For Debian:**
- `debian/changelog` — Package version and maintainer info.
- `debian/control` — Package name, dependencies, description.
- `debian/rules` — Build rules (`dh_*` commands).
- `debian/install` — File installation paths.
- Optional `debian/postinst`, `debian/prerm` — Maintainer scripts.

**For RPM:**
- `*.spec` — Complete RPM spec file with metadata, dependencies, file lists,
  and scriptlets.

### Step 5: Package Building

**Debian:**

```bash
dpkg-buildpackage -uc -us -b
```

Produces `.deb` files and optional `.ddeb` debug symbol packages.

**RPM:**

```bash
rpmbuild --define "_topdir <pkg_dir>" -ba <specfile>
```

Produces `.rpm` files and optional `-debuginfo` packages.

## Versioned vs Non-Versioned Packages

For each entry in `package.json`, **two package variants** are produced:

```
┌────────────────────────┐        ┌──────────────────────────────┐
│ amdrocm-blas           │───────▶│ amdrocm-blas-7.12.0          │
│ (non-versioned)        │ depends│ (versioned)                  │
│                        │        │                              │
│ Empty meta-package     │        │ Contains actual files        │
│ Depends on versioned   │        │ Installed to /opt/rocm/core/ │
└────────────────────────┘        └──────────────────────────────┘
```

- **Non-versioned** (`amdrocm-blas`): Empty meta-package that declares a
  dependency on the versioned package. Allows `apt install amdrocm-blas` to
  always get the latest version.
- **Versioned** (`amdrocm-blas-7.12.0`): Contains the actual library files,
  installed to the versioned path. Allows multiple versions to coexist.

The non-versioned package is skipped when building RPATH packages
(`--rpath-pkg`).

## RPATH Packages

When `--rpath-pkg` is enabled:

1. Only **versioned** packages are produced (no meta-packages).
2. `RUNPATH` entries in ELF binaries and libraries are converted to `RPATH`
   using `runpath_to_rpath.py`.
3. This makes libraries resolve dependencies from the versioned install path
   rather than relying on `LD_LIBRARY_PATH` or system library paths.

RPATH packages are useful for containerized or isolated deployments where
multiple ROCm versions coexist.

## package.json Structure

The `package.json` file is a JSON array of package definitions. Each entry
describes one native package.

### Mandatory Fields

| Field | Type | Description |
| --- | --- | --- |
| `Package` | string | Package name (e.g. `amdrocm-blas`) |
| `Architecture` | string | Target architecture (e.g. `amd64`) |
| `Maintainer` | string | Package maintainer |
| `Description_Short` | string | One-line summary |
| `Description_Long` | string | Detailed description |
| `Artifact` | string | ROCk artifact name prefix |
| `Artifact_Subdir` | string | Subdirectory within artifact |
| `Components` | array | Artifact components to include (`lib`, `run`, `dev`, etc.) |
| `Homepage` | string | Project URL |
| `Vendor` | string | Package vendor |
| `License` | string | License identifier |
| `Group` | string | Package category |
| `Priority` | string | Package priority level |
| `Section` | string | Package section |
| `Gfxarch` | boolean | Whether package is GPU-architecture-specific |
| `DEBDepends` | array | Debian dependency list |
| `RPMRequires` | array | RPM dependency list |

### Optional Fields

| Field | Type | Description |
| --- | --- | --- |
| `Composite` | boolean | Package aggregates multiple artifacts |
| `Includes` | array | For composites: list of packages to include |
| `DisablePackaging` | boolean | Skip building this package |
| `Disable_Debug_Package` | boolean | Skip debug symbol package |
| `Disable_DWZ` | boolean | Skip DWZ processing (Debian) |
| `Disable_DH_STRIP` | boolean | Skip `dh_strip` (Debian) |
| `Provides` | string | Virtual package provision |
| `Replaces` | string | Package replacement declaration |
| `Metapackage` | boolean | Package has no files, only dependencies |

### Example Entry

```json
{
  "Package": "amdrocm-blas",
  "Architecture": "amd64",
  "Maintainer": "AMD ROCm <rocm@amd.com>",
  "Description_Short": "ROCm BLAS libraries",
  "Description_Long": "AMD ROCm Basic Linear Algebra Subprograms libraries including rocBLAS, hipBLAS, and hipBLASLt",
  "Artifact": "blas",
  "Artifact_Subdir": "",
  "Components": ["lib", "run"],
  "Homepage": "https://github.com/ROCm/TheRock",
  "Vendor": "Advanced Micro Devices, Inc.",
  "License": "MIT",
  "Group": "Development/Libraries",
  "Priority": "optional",
  "Section": "devel",
  "Gfxarch": true,
  "DEBDepends": ["amdrocm-runtime"],
  "RPMRequires": ["amdrocm-runtime"]
}
```

## Jinja2 Templates

Templates are in `build_tools/packaging/linux/template/`:

### Debian Templates

| Template | Output | Purpose |
| --- | --- | --- |
| `debian_changelog.j2` | `debian/changelog` | Version, date, maintainer |
| `debian_control.j2` | `debian/control` | Package metadata and dependencies |
| `debian_rules.j2` | `debian/rules` | Build rules for `dpkg-buildpackage` |
| `debian_install.j2` | `debian/install` | File-to-path mappings |

### RPM Templates

| Template | Output | Purpose |
| --- | --- | --- |
| `rpm_specfile.j2` | `*.spec` | Complete spec file |

### Maintainer Scripts

Additional templates in `template/scripts/` generate pre/post install and
remove scripts:

| Script | When | Purpose |
| --- | --- | --- |
| `postinst` / `%post` | After installation | Create symlinks, update ldconfig |
| `prerm` / `%preun` | Before removal | Remove symlinks, clean caches |

## Installed Directory Structure

### DEB Installation Layout

After `dpkg -i amdrocm-blas-7.12.0*.deb`:

```
/opt/rocm/core/7.12/
├── lib/
│   ├── librocblas.so.4
│   ├── librocblas.so.4.3.0
│   ├── libhipblas.so.2
│   ├── libhipblaslt.so.0
│   └── ...
├── include/                    # (from -devel package)
│   ├── rocblas/
│   ├── hipblas/
│   └── ...
├── lib/cmake/                  # (from -devel package)
│   ├── rocblas/
│   └── ...
└── share/
    └── doc/
```

### RPM Installation Layout

RPM packages follow the same install prefix layout as DEB:

```
/opt/rocm/core/7.12/
├── bin/
├── lib/
├── include/
├── lib/cmake/
└── share/
```

### Component Placement Map

| Component | Native Package | Install Path | Files |
| --- | --- | --- | --- |
| HIP runtime | `amdrocm-runtime` | `/opt/rocm/core/7.12/lib/` | `libamdhip64.so`, `libhsa-runtime64.so` |
| LLVM compiler | `amdrocm-llvm` | `/opt/rocm/core/7.12/llvm/` | `clang`, `lld`, LLVM libraries |
| rocBLAS | `amdrocm-blas` | `/opt/rocm/core/7.12/lib/` | `librocblas.so` |
| rocFFT | `amdrocm-fft` | `/opt/rocm/core/7.12/lib/` | `librocfft.so` |
| MIOpen | `amdrocm-dnn` | `/opt/rocm/core/7.12/lib/` | `libMIOpen.so` |
| RCCL | `amdrocm-rccl` | `/opt/rocm/core/7.12/lib/` | `librccl.so` |
| CLI tools | `amdrocm-base` | `/opt/rocm/core/7.12/bin/` | `rocminfo`, `rocm-smi` |
| System deps | `amdrocm-sysdeps` | `/opt/rocm/core/7.12/lib/rocm_sysdeps/` | Private vendored libs |
| BLAS headers | `amdrocm-blas-devel` | `/opt/rocm/core/7.12/include/rocblas/` | `.h` files |
| CMake configs | Various `-devel` | `/opt/rocm/core/7.12/lib/cmake/` | `*Config.cmake` files |
| Test binaries | Various `-test` | `/opt/rocm/core/7.12/clients/` | Test executables and data |

## DEB vs RPM Differences

| Aspect | DEB | RPM |
| --- | --- | --- |
| **Build tool** | `dpkg-buildpackage -uc -us -b` | `rpmbuild -ba <specfile>` |
| **Metadata format** | `debian/control` + separate files | Single `.spec` file |
| **Dev package suffix** | `-dev` (auto-renamed from `-devel`) | `-devel` (as-is from `package.json`) |
| **Debug symbols** | `.ddeb` via `dh_strip` | `-debuginfo` RPM |
| **Version format** | `7.12.0~YYYYMMDD` (nightly) | `7.12.0~YYYYMMDD` (nightly) |
| **Prerelease** | `7.12.0~preN` | `7.12.0~rcN` |
| **Dev version** | `7.12.0~devYYYYMMDD` | `7.12.0~YYYYMMDDg<hash>` |
| **Install command** | `dpkg -i <pkg>.deb` | `rpm -i <pkg>.rpm` |
| **Dependency resolution** | `apt install` (with repo) | `dnf install` (with repo) |
| **Target distros** | Ubuntu | RHEL, SLES, AlmaLinux |
| **DWZ processing** | Optional (`Disable_DWZ`) | N/A |
| **Strip control** | `Disable_DH_STRIP` flag | Via spec file |

## Comparison with Other Formats

| Aspect | Tarball | Python Wheel | DEB/RPM |
| --- | --- | --- | --- |
| **Install method** | `tar -xf` | `pip install` | `dpkg -i` / `rpm -i` |
| **Install location** | User-chosen | `site-packages/` | `/opt/rocm/core/<ver>/` |
| **Dependencies** | None managed | pip resolves | apt/dnf resolves |
| **Versioning** | In filename | pip metadata | dpkg/rpm database |
| **Multiple versions** | Multiple dirs | Multiple venvs | Versioned paths |
| **Debug symbols** | Included | Not included | Separate `-debuginfo`/`.ddeb` |
| **Uninstall** | Delete directory | `pip uninstall` | `dpkg -r` / `rpm -e` |
| **System integration** | Manual PATH setup | Automatic shims | System paths |
| **Package granularity** | Single monolithic | 4 packages | 50+ fine-grained packages |
| **GPU selection** | Per-family tarball | Dynamic detection | Per-family packages |
| **Ideal for** | CI, Docker, dev | Python projects | System-wide, production |

## CI Workflow Integration

Native packages are built by the
[`build_native_linux_packages.yml`](/.github/workflows/build_native_linux_packages.yml)
workflow, which is dispatched by the release workflow:

```
release_portable_linux_packages.yml
    │
    ├──── workflow_dispatch ────▶ build_native_linux_packages.yml (pkg-type: deb)
    │                                   │
    │                                   ├── fetch_artifacts.py (download archives)
    │                                   ├── build_package.py --pkg-type deb
    │                                   └── upload to S3 (therock-<channel>-packages)
    │
    └──── workflow_dispatch ────▶ build_native_linux_packages.yml (pkg-type: rpm)
                                        │
                                        ├── fetch_artifacts.py (download archives)
                                        ├── build_package.py --pkg-type rpm
                                        └── upload to S3 (therock-<channel>-packages)
```

DEB and RPM builds run in **parallel** as independent workflow dispatches.

## Building Packages Locally

### Prerequisites

**Ubuntu (for DEB packages):**

```bash
apt update
apt install -y python3 python3-pip debhelper llvm
pip install -r requirements.txt
```

**AlmaLinux (for RPM packages):**

```bash
dnf install rpm-build llvm
pip install -r requirements.txt
```

Python 3.12 or above is required.

### Building DEB Packages

```bash
python build_tools/packaging/linux/build_package.py \
    --artifacts-dir ./artifacts \
    --target gfx94X-dcgpu \
    --dest-dir ./output \
    --rocm-version 7.12.0 \
    --version-suffix nightly \
    --pkg-type deb
```

### Building RPM Packages

```bash
python build_tools/packaging/linux/build_package.py \
    --artifacts-dir ./artifacts \
    --target gfx94X-dcgpu \
    --dest-dir ./output \
    --rocm-version 7.12.0 \
    --version-suffix nightly \
    --pkg-type rpm
```

### Building RPATH Packages

```bash
python build_tools/packaging/linux/build_package.py \
    --artifacts-dir ./artifacts \
    --target gfx94X-dcgpu \
    --dest-dir ./output \
    --rocm-version 7.12.0 \
    --pkg-type deb \
    --rpath-pkg True
```

To build only specific packages:

```bash
python build_tools/packaging/linux/build_package.py \
    --artifacts-dir ./artifacts \
    --target gfx94X-dcgpu \
    --dest-dir ./output \
    --rocm-version 7.12.0 \
    --pkg-type deb \
    --pkg-names amdrocm-blas amdrocm-fft
```

## Relevant Source Files

| File | Purpose |
| --- | --- |
| [`build_tools/packaging/linux/build_package.py`](/build_tools/packaging/linux/build_package.py) | Main DEB/RPM build script |
| [`build_tools/packaging/linux/package.json`](/build_tools/packaging/linux/package.json) | Declarative package definitions |
| [`build_tools/packaging/linux/template/debian_changelog.j2`](/build_tools/packaging/linux/template/debian_changelog.j2) | Debian changelog template |
| [`build_tools/packaging/linux/template/debian_control.j2`](/build_tools/packaging/linux/template/debian_control.j2) | Debian control file template |
| [`build_tools/packaging/linux/template/debian_rules.j2`](/build_tools/packaging/linux/template/debian_rules.j2) | Debian build rules template |
| [`build_tools/packaging/linux/template/debian_install.j2`](/build_tools/packaging/linux/template/debian_install.j2) | Debian install file template |
| [`build_tools/packaging/linux/template/rpm_specfile.j2`](/build_tools/packaging/linux/template/rpm_specfile.j2) | RPM spec file template |
| [`build_tools/packaging/linux/template/scripts/`](/build_tools/packaging/linux/template/scripts/) | Maintainer script templates |
| [`build_tools/compute_rocm_native_package_version.py`](/build_tools/compute_rocm_native_package_version.py) | Native version computation |
| [`docs/packaging/native_packaging.md`](/docs/packaging/native_packaging.md) | Native packaging general design |
| [`docs/packaging/versioning.md`](/docs/packaging/versioning.md) | Version scheme specification |
| [`.github/workflows/build_native_linux_packages.yml`](/.github/workflows/build_native_linux_packages.yml) | CI workflow for native packages |
