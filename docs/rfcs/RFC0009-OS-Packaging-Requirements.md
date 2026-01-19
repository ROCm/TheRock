---
author: Liam Berry (LiamfBerry), Saad Rahim (saadrahim)
created: 2025-11-14
modified: 2026-01-12
status: draft
---

# TheRock Software Packaging Requirements

## Overview

With the implementation of TheRock build system, new software packaging requirements need to be introduced to reflect TheRock's strategy. This RFC defines the cross-platform packaging, installation, versioning, and distribution requirements for TheRock, including the ROCm Core SDK and related ROCm software components. The scope of these requirements will cover OS distribution packaging.

Our goals are to:

1. **Standardize packaging behavior across Linux, Windows WSL2**
1. **Ensure predictable upgrade behavior, side-by-side support, and compatibility with OS package managers (apt, dnf, yum, zypper for SLES)**
1. **Comply with legal, licensing, and redistribution rules**
1. **Support automated packaging workflows in TheRock with productized deliverables**

## Scope

### In Scope

- Packaging formats: rpm and deb
- GPU-architecture-specific package variants
- Side-by-side installation of ROCm Core SDK
- Repository metadata, signing, and precedence
- Development vs. runtime package separation
- ASAN, debug, and source packages
- Naming conventions (AMD-generated vs. native distributions)
- Nightly, prerelease, patch, and stable release version semantics
- Integration with TheRock build system

### Out of Scope

- Driver packaging (GPU driver is explicitly excluded from installers)
- Internal CI/CD implementation details
- Legacy ROCm 5.x / 6.x packaging
- Non-Linux UNIX variants
- Windows packaging requirements
- Python pip and wheelnext packaging requirements

## Linux Packaging Requirements

### Directory Layout

The ROCm Core SDK must be installed under:

```
/opt/rocm/core-X.Y
```

Where:

- `X.Y` = major + minor version
- Patch versions must be in place within the existing `X.Y` folder
- Side-by-side installation is supported only for major.minor releases, not patches

A soft link must exist as a path to the latest rocm and to the latest rocm minor release for a major release:

```
/opt/rocm/core/ -> /opt/rocm/core-8.2.0
/opt/rocm/core-8 -> /opt/rocm/core-8.2.0
```

The two options for the softlinks as shown above allow users to either specify the major release and pull the latest minor and patch release of that version or to just pull the latest release by not specifying the version.

The soft links allow for an independent directory structure for ROCm expansions, which must be in the following format:

```
/opt/rocm/hpc-25.12.0
/opt/rocm/hpc/ -> /opt/rocm/hpc-26.2.0
```

### RPATH and Relocatability

- All ROCm packages must be built and shipped with `$ORIGIN`-based RPATH
- RPMs must honor the `--prefix` argument for relocatable installs

### Repository Layout

Repositories will follow the following structure:

```
repo.amd.com/rocm/packages/<primary_os>/
```

The primary OS root folder will include the following distributions where the packages can be found:

| Primary OS | Secondary                |
| :--------- | :----------------------- |
| debian12   |                          |
| ubuntu2204 |                          |
| ubuntu2404 |                          |
| rhel8      | Centros 8                |
| rhel9      | Oracle 9, Rock 9, Alma 9 |
| rhel10     |                          |
| sles15     |                          |
| azl3       |                          |

ASAN packages may be separated into:

```
repo.amd.com/rocm/packages/$OS/$package-type

Package-type = standard, asan, future variant
```

This will reduce the number of packages visible via the package manager.

### Package Naming for No Duplication with Distros

The four possible naming strategies for packages were analyzed:

1. Prefix `amd-`
1. Prefix `amdi-`: Legally the safest option, as no one can claim to AMD incorporated
1. Suffix `-amd`
1. Do nothing: Manage through versioning
1. Prefix `amd`

A working group concluded that TheRock will adopt `amdrocm-<package>` for Linux distro-native package disambiguation.
This avoids namespace conflicts with distro-provided packages. Distros will use `rocm-<package>` i.e., upstream distributions like Ubuntu and Redhat should not use `amdrocm-`, this will be recommended by AMD but not enforced by any restrictive covenants.

### Device-Specific Architecture Packages

Users are encouraged to identify their local GPU architecture and install packages exclusive to the GPU architectures present. Otherwise, users can install a complete ROCm installation with all GPU architectures to enable all GPUs. Users not familiar with their GPU architecture may be directed to runfile installers with autodetection capabilities. These compromises are made with the understanding that package managers cannot autodetect local hardware to select package families.

| Component         | Meta package for all device packages                                                               |
| :---------------- | :------------------------------------------------------------------------------------------------- |
| component-host    | Host-only package                                                                                  |
| component-$device | $device is the llvm gfx architecture; each device package must have no conflict with other devices |

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

| Name                    | Content                                                              | Description                                                                   |
| :---------------------- | :------------------------------------------------------------------- | :---------------------------------------------------------------------------- |
| amdrocm & amdrocm-core  | runtime & libraries, components, runtime compiler, amd-smi, rocminfo | Needed to run software built with ROCm Core                                   |
| amdrocm-core-devel      | rocm-core + compiler cmake, static library files, and headers        | Needed to build software with ROCm Core                                       |
| amdrocm-developer-tools | Profiler, debugger, and related tools                                | Independent set of tools to debug and profile any application built with ROCm |
| amdrocm-fortran         |                                                                      | Fortran compiler and related components                                       |
| amdrocm-opencl          |                                                                      | Components needed to run OpenCL                                               |
| amdrocm-openmp          |                                                                      | Components needed to build OpenMP                                             |
| amdrocm-core-sdk        |                                                                      | Everything                                                                    |

## Package Granularity

Package granularity will be increased with ROCm 8.0. Development packages contain all the code required to build the libraries, including headers, Cmake files, and static libraries. Source packages for all of rocm-libraries provide all the files to build the libraries from source in addition to the rocm-rock source package.

| Name                                    | Dev package components only | Runtime packages                                                              | Source package inclusion only            |
| :-------------------------------------- | :-------------------------- | :---------------------------------------------------------------------------- | :--------------------------------------- |
| amdrocm-amdsmi                          |                             | amd-smi                                                                       |                                          |
| amdrocm-llvm                            |                             | amdclang++ (flang and openmp here if not separable)                           |                                          |
| amdrocm-flang                           |                             | flang                                                                         |                                          |
| amdrocm-runtimes                        |                             | HIP, ROCR, CLR, runtime compilation, SPIR-V                                   |                                          |
| amdrocm-fft                             |                             | rocFFT, hipFFT, hipFFTW                                                       |                                          |
| amdrocm-math                            |                             | Temporary catch-all if libraries cannot fix circular dependencies by ROCm 8.0 |                                          |
| amdrocm-blas                            | hipBLAS-common              | rocBLAS, hipBLAS, hipBLASLt, hipSPARSELt                                      |                                          |
| amdrocm-sparse                          |                             | rocSPARSE, hipSPARSE                                                          |                                          |
| amdrocm-solver                          |                             | rocSOLVER, hipSOLVER                                                          |                                          |
| amdrocm-dnn                             |                             | hipDNN, MIOpen                                                                |                                          |
| amdrocm-rand                            |                             | rocRAND, hipRAND                                                              |                                          |
| amdrocm-ccl                             | rocPRIM, rocThrust, hipCUB  | libhipcxx                                                                     |                                          |
| amdrocm-profiler                        |                             | rocm-systems, rocm-compute, rocprofiler-sdk, tracer                           |                                          |
| amdrocm-profiler-base                   |                             | rocprofiler-sdk, tracer                                                       |                                          |
| amdrocm-base                            |                             | rocminfo, version (rocm-core)                                                 |                                          |
| amdrocm-CK                              |                             |                                                                               | CK                                       |
| amdrocm-debugger                        |                             | rocgdb                                                                        |                                          |
| amdrocm-math-common or amdrocm-math-dev |                             |                                                                               | CK, rocWMMA, rocRoller, Tensile, Origami |
| amdrocm-hipify                          |                             | HIPIFY                                                                        |                                          |
| amdrocm-opencl                          |                             | OpenCL                                                                        |                                          |
| amdrocm-decode                          |                             | rocDecode                                                                     |                                          |
| amdrocm-jpeg                            |                             | rocJPEG                                                                       |                                          |
| amdrocm-file                            |                             | hipFile, rocFile (future addition)                                            |                                          |
| amdrocm-rccl                            |                             | rccl                                                                          |                                          |
| amdrocm-sysdeps                         |                             | Bundled 3rd party dependencies (e.g., libdrm, libelf, numa, subset of libVA)  |                                          |
| amdrocm-rdc                             |                             | ROCm Datacenter                                                               |                                          |

Note: Product management would like to follow upstream packaging structrures in ROCm in the future with no interim due dates as of now. Today there may be one amdrocm-llvm that includes both flang and the flang compiler; the flang component can be dependent on the llvm component.

## Versioning Requirements

For versioning requirements on packaging, see the following documentation: [TheRock package versioning](/docs/packaging/versioning.md)
