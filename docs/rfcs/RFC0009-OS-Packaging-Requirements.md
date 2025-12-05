---
author: Liam Berry (LiamfBerry), Saad Rahim (saadrahim)
created: 2025-11-14
modified: 2025-12-04
status: draft
---

# TheRock Software Packaging Requirements

## Overview

With the implementation of TheRock build system new software packaging requirements need to be introduced to reflect TheRock's strategy. This RFC defines the cross-platform packaging, installation, versioning, and distribution requirements for TheRock; including the ROCm Core SDK and related ROCm software components. The scope of these requirments will cover OS distrobution packaging, Windows packaging requirements, and python packaging.

Our goals are to:

1. **Standardize packaging behaviour across Linux, Windows native, Windows WSL2**
2. **Ensure predicatble upgrade behaviour, side-by-side support, and compatibility with OS package managers (apt, dnf, yum, zypper for SLES)**
3. **Comply with legal, licensing, and redistrobution rules**
4. **Support automated packaging workflows in TheRock with productized deliverables**
5. **Support packaging behaviour across Python ecosystems using pip and WheelNext**

## Scope

### In Scope

- Packaging formats: rpm, deb, msi, winget manifests, pip wheels, WheelNext
- GPU-architecture-specific package variants
- Side-by-side installation of ROCm Core SDK
- Repository metadata, signing, and precedence
- Development vs runtime package separation
- ASAN, debug, and source packages
- Naming conventions (AMD-generated vs native distorbutions)
- Nightly, prerelease, patch, and stable release version semantics
- Integration with TheRock build system

### Out of Scope

- Driver packaging (GPU driver is explicitly excluded from installers)
- Internal CI/CD implementation details
- Legacy ROCm 5.x / 6.x packaging
- Non-Linux UNIX variants

## Linux Packaging Requirements

### Directory Layout

The ROCm Core SDK must be installed under:

```
/opt/rocm/core-X.Y
```

Where:

- `X.Y` = major + minor version
- Patch versions must be in-place within the existing `X.Y` folder
- Side-by-side installation is supported only for major.minor releases, not patches

A softlink must exist as a path to the latest rocm and to the latest rocm minor release for a major release:

```
/opt/rocm/core/ -> /opt/rocm/core-8.2.0
/opt/rocm/core-8 -> /opt/rocm/core-8.2.0
```

The two options for the softlinks as shown above allow users to either specify the major release and pull the latest minor and patch release of that version or to just pull the latest release by not specifiying the version.

The softlinks allow for an independent directory structure for ROCm expansions which must be in the following formating:

```
/opt/rocm/hpc-25.12.0
/opt/rocm/hpc/ -> /opt/rocm/hpc-26.2.0
```

### RPATH and Relocatability

- All ROCm packages must be built and shiped with `$ORIGIN`-based RPATH
- RPMs must honor the `--prefix` argument for relocatable installs

### Repository Layout

Repositories will follow the following structure:

```
repo.amd.com/rocm/packages/<primary_os>/
```

The primary OS root folder will include the following distrobutions where the packages can be found:

| Primary OS | Secondary |
| :------------- |:-------------|
| debian12 |  |
| ubuntu2204 |  |
| ubuntu2404 |  |
| rhel8 | Centros 8 |
| rhel9 | Oracle 9, Rock 9, Alma 9 |
| rhel10 |  |
| sles15 |  |
| azl3 |  |

ASAN packages may be separated into:

```
repo.amd.com/rocm/packages/$OS/$package-type

Package-type = standard, asan, future variant
```

This will reduce the number of packages visible via the package manager.

### Meta Packages

Using `yum` ROCm Core SDK runtime components and ROCm Core SDK runtime + development components can be installed.

```
yum install rocm # ROCm 8.0
yum install rocm-core # ROCm 8.0
yum install rocm-core<ver>
yum install rocm-core-devel
yum install rocm-core-devel<ver>
```
The following table shows the meta packages that will be available:

| Name | Content | Descripion |
| :------------- | :------------- | :------------- |
| rocm & rocm-core | runtime & libraries, components, runtime compiler, amd-smi, rocminfo | Needed to run software built with ROCm Core |
| rocm-core-devel | rocm-core + compiler cmake, static library files, and headers | Needed to build software with ROCm Core |
| rocm-devel-tools | Profiler, debugger, and related tools | Independent set of tools to debug and profile any application built with ROCm |
| rocm-fortran |  | Fortran compiler and related components |
| rocm-opencl |  | Components needed to run OpenCL |
| rocm-openmp |  | Components needed to build OpenMP |
| rocm-core-sdk |  | Everything |

### Package Naming for no duplication with distors

The four possible naming strategies for packages were analyzed:

1. Prefix `amd-`
2. Prefix `amdi-`: Legally the safest option as no one can claim to AMD incorporated
3. Suffix `-amd`
4. Do nothing: Manage through versioning
5. Prefix `amd`

TheRock should adopt `amdrocm-<package>` for Linux distro-native package disambiguation unless Legal or Branding teams choose an alternative.
This avoids namespace conflicts with distro-provided packages. Distros will use `rocm-<package>`

### Device-Specific Architecture Packages

Local GPUs must have an autodetection mechanism via the package manager. Possible options for device-specific architecture packages can be seen in the table as shown:

| Component | Meta package for all device packages |
| :------------- | :------------- |
| component-host | Host only package |
| component-$device | $device is the llvm gfx architecture each device package must have no conflict with other devices |

Example: 

```
yum install miopen-gfx906 miopen-gfx908
apt intall rocm-gfx906 rocm-gfx-908 # Host + two device architectures
apt install rocm # Every architecture
```

All device-specific packages must:

- Not conflict with each other
- Be independently installable
- Support meta-packages
- Allow autodetection of local GPUs

TheRock must provide a GPU detection interface for package managers.

## Package Granularity 

Package granularity will be increased with ROCm 8.0. Development packages contain all the code required to build the libraries including headers, cmakefiles, and static libraries. Source packages for all of rocm-libraries provides all the files to build the libraries from source in addition to the rocm-rock source package.

| Name | Dev package components only | Runtime packages | Source package inclusion only |
| :------------- | :------------- | :------------- | :------------- |
| amdrocm-smi | | amd-smi | |
| amdrocm-llvm |  | amdclang++ (flang and openmp here if not separable) |  |
| amdrocm-flang |  | flang |  |
| amdrocm-runtimes |  | HIP, ROCR, CLR, runtime compilation, SPIR-V |  |
| amdrocm-fft |  | rocFFT, hipFFT, hipFFTW |  |
| amdrocm-math |  | Temporary catch all if libraries cannot fix circular dependencies by ROCm 8.0 |  |
| amdrocm-blas | hipBLAS-common | rocBLAS, hipBLAS, hipBLASLt, hipSPARSELt |  |
| amdrocm-sparse |  | rocSPARSE, hipSPARSE |  |
| amdrocm-solver |  | rocSOLVER, hipSOLVER |  |
| amdrocm-dnn |  | hipDNN, MIOpen |  |
| amdrocm-rand |  | rocRAND, hipRAND |  |
| amdrocm-ccl | rocPRIM, rocThrust, hipCUB | libhipcxx |  |
| amdrocm-profiler |  | rocm-systems, rocm-compute, rocprofiler-sdk, tracer |  |
| amdrocm-profiler-base |  | rocprofiler-sdk, tracer |  |
| amdrocm-base |  | AMD-SMI, rocminfo, version (rocm-core) |  |
| amdrocm-CK |  |  | CK |
| amdrocm-debugger |  | rocgdb |  |
| amdrocm-math-comon or rocm-math-dev |  |  | CK, rocWMMA, rocRoller, Tensile, Origami |
| amdrocm-hipify |  | HIPIFY |  |
| amdrocm-opencl |  | OpenCL |  |
| amdrocm-decode |  | rocDecode |  |
| amdrocm-jpeg |  | rocJPEG |  |
| amdrocm-file |  | hipFile, rocFile (future addition) |  |
| amdrocm-sysdeps |  | Bundled 3rd party dependencies (e.g. libdrm, libelf, numa, subset of libVA) |  |

## Python Packaging Requirements

### Standard pip WHeels

Today's pip cannot handle cross-architecture extras without vitrual environments. The new packaging requirements will support the following options:

```
pip install rocm # GPU-agnostic
pip install rocm[gfx908] # Single GPU architecture
pip install rocm[gfx90a, gfx908] # Multiple GPU architectures (currently not supported)
pip install rocm[all] # All GPU architectures
```

Engineering must validate the minimum Python version that supports bracket-extras.
Dependencies should use `>=` (version X or newer) unless technical risk requires `==`.

### WheelNext Variant Wheels

WheelNext allows GPU detection via the amd-variant-provider which can be seen in the following examples:

```
uv pip install rocm
uv pip install rocm[gfx90a, gfx908] # override default variant search
uv pip install rocm[all]
```

TheRock must support building and publishing both:

- Standard pip wheels
- WheelNext variant wheels

### Detection Requirements

WheelNext must detect local GPUs via:

- PCI bus scanning
- ROCm driver presence
- host-only variant

## Versioning Requirements

The following versioning format will be implemented as follows.

Stable releases:

```
X.Y.Z
```

Prereleases:

| Type of package | Version format | Version example |
| :------------- | :------------- | :------------- |
| Python | X.Y.ZrcN | 7.9.0rc0 |
| Tarball | X.Y.ZrcN | 7.9.0rc0 |

Nightly Releases:

| Type of package | Version format | Version example |
| :------------- | :------------- | :------------- |
| Python | X.Y.ZaYYYYMMDD | 7.10.0a20251006 |
| Tarball | X.Y.ZaYYYYMMDD | 7.10.0a20251006 |

TheRock must automate all formats.

### RPM Package
RPM packages should follow the file name format as follows

```
Non-Versioned package
<package-name>-<package-version>-<release-version>.<arch>.rpm

Versioned package
<package-name><package-version>-<package-version>-<release-version>.<arch>.rpm

  - <package-name>: The name of the package/module
  - <package-version>: The version of the package. As defined in following session.
  - <release-version>: Build no. Github run id should be used in case of TheRock build.
  - <arch>: The target architecture (x86_64).
```

## RPM Package Versioning Requirements

# Stable Releases:
```
X.Y.Z
```
Stable Release Version Examples:
```
8.0.0        # Major release
8.1.0        # Minor release
8.0.1        # Patch release
8.0.2        # Patch release
8.1.1        # Patch release

Potential Extensions:
8.1.0-1      # Package rebuild (no code change)
8.1.0.1      # Point release (customer specific releases)
```

# Prereleases:
```
X.Y.Z-rcN

N: release candidate no
```

# Nightly Releases:
```
X.Y.Z~YYYYMMDD
```

# Dev builds:
```
X.Y.Z~YYYYMMDDg<short-git-sha>   # git sha shortened to 8 bytes
```
# Package Version Ordering
Package version in increasing order
```
8.1.0~20251201gf689a8e   # Development
8.1.0~20251202ga123456   # Development
8.1.0~20251203           # Nightly
8.1.0~rc1                # Release Candidate 1
8.1.0~rc2                # Release Candidate 2
8.1.0                    # Final release
8.1.1                    # Next patch release
```

# RPM package samples
```
rocm-8.1.0~20251201gf689a8e-1.x86_64.rpm                              # Development build using build no
rocm-8.1.0~20251201gf689a8e-1234567.x86_64.rpm                        # Development build using github runid
rocm-8.1.0~20251203-1.x86_64.rpm                                      # Nightly release
rocm-8.1.0~rc1-1.x86_64.rpm                                           # Release Candidate 1
rocm-8.1.0~rc2-2345678.x86_64.rpm                                     # Release Candidate 2 using github runid
rocm-8.1.0-1.x86_64.rpm                                               # Final release
rocm-8.2.0-3456789.x86_64.rpm                                         # Next Minor release
rocm8.1.0~20251201gf689a8e-8.1.0~20251201gf689a8e-1.x86_64.rpm        # Versioned Development build using build no
rocm8.1.0~20251201gf689a8e-8.1.0~20251201gf689a8e-1234567.x86_64.rpm  # Versioned Development build using github runid
rocm8.1.0~20251203-8.1.0~20251203-1.x86_64.rpm                        # Versioned Nightly release
rocm8.1.0~rc1-8.1.0~rc1-1.x86_64.rpm                                  # Versioned Release Candidate 1
rocm8.1.0~rc2-8.1.0~rc2-2345678.x86_64.rpm                            # Versioned Release Candidate 2 using github runid
rocm8.1.0-8.1.0-1.x86_64.rpm                                          # Versioned Final release
rocm8.2.0-8.2.0-3456789.x86_64.rpm                                    # Versioned Next Minor release
```

### Debian Package
Debian packages should follow the file name format as follows

```
Non-Versioned package
<package-name>_<package-version>-<release-version>.<arch>.deb

Versioned package
<package-name><package-version>_<package-version>-<release-version>_<arch>.deb

  - <package-name>: The name of the package/module
  - <package-version>: The version of the package. As defined in following session.
  - <release-version>: Build no. Github run id should be used in case of TheRock build.
  - <arch>: The target architecture (amd64).
```

## Debian Package Versioning Requirements

# Stable Releases:
```
X.Y.Z
```
Stable Release Version Examples:
```
8.0.0        # Major release
8.1.0        # Minor release
8.0.1        # Patch release
8.0.2        # Patch release
8.1.1        # Patch release

Potential Extensions:
8.1.0-1      # Package rebuild (no code change)
8.1.0.1      # Point release (customer specific releases)
```

# Prereleases:
```
X.Y.Z-preN

N: release candidate no
```

# Nightly Releases:
```
X.Y.Z~YYYYMMDD
```

# Dev builds:
```
X.Y.Z~devYYYYMMDD
```
# Package Version Ordering
Package version in increasing order
```
8.1.0~dev20251201        # Development
8.1.0~dev20251202        # Development
8.1.0~20251203           # Nightly
8.1.0~pre1               # Release Candidate 1
8.1.0~pre2               # Release Candidate 2
8.1.0                    # Final release
8.1.1                    # Next patch release
```

# Debian package samples
```
rocm_8.1.0~20251201gf689a8e-1_amd64.deb                              # Development build using build no
rocm_8.1.0~20251201gf689a8e-1234567_amd64.deb                        # Development build using github runid
rocm_8.1.0~20251203-1_amd64.deb                                      # Nightly release
rocm_8.1.0~pre1-1_amd64.deb                                          # Release Candidate 1
rocm_8.1.0~pre2-2345678_amd64.deb                                    # Release Candidate 2 using github runid
rocm_8.1.0-1_amd64.deb                                               # Final release
rocm_8.2.0-3456789_amd64.deb                                         # Next Minor release
rocm8.1.0~20251201gf689a8e_8.1.0~20251201gf689a8e-1_amd64.deb        # Versioned Development build using build no
rocm8.1.0~20251201gf689a8e_8.1.0~20251201gf689a8e-1234567_amd64.deb  # Versioned Development build using github runid
rocm8.1.0~20251203_8.1.0~20251203-1_amd64.deb                        # Versioned Nightly release
rocm8.1.0~pre1_8.1.0~pre1-1_amd64.deb                                # Versioned Release Candidate 1
rocm8.1.0~pre2_8.1.0~pre2-2345678_amd64.deb                          # Versioned Release Candidate 2 using github runid
rocm8.1.0_8.1.0-1_amd64.deb                                          # Versioned Final release
rocm8.2.0_8.2.0-3456789_amd64.deb                                    # Versioned Next Minor release
```

## Windows Packaging Requirements

### Required Installation Technologies

TheRock must provide:

- MSI installer
- Winget manifest and distrobution support
- Side-by-side instal via windows assemblies
- WSL package compatibility

Folder Layout:

```
C:\Program Files\AMD\ROCm\Core-X.Y\
```

Installer Requirements:

- Must not bundle GPU driver
- Must provide driver download links
- Must support side-by-side major.minor installations
- MSI must support repair, modify, upgrade, uninstall

### WinGet Requirements

TheRock packaging system should enable winget to simplify deployment of software

Winget must show multiple available versions of ROCm: 

```
C:\Users\<username>\ winget search "ROCm Core SDK" --versions

Found AMD ROCm Core SDK [AMD.ROCmCore]
Version
-------
9.0
8.1
8.0
```

As shown, winget supports multiple version of the ROCm Core SDK. By default, only major.minor versions are shown. If a patch version of the SDK is released, only the latest patch version is shown.
