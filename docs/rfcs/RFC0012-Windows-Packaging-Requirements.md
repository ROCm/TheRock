---
author: Liam Berry (LiamfBerry), Saad Rahim (saadrahim)
created: 2026-03-13
modified: 2026-05-11
status: draft
---

# TheRock Windows Packaging Requirements

With the implementation of TheRock build system, native Windows packaging and installation requirements must be defined to complement the Linux software packaging requirements. This RFC defines the packaging, installation, upgrade, uninstallation, versioning, and distribution requirements for TheRock software on native Windows, including the ROCm Core SDK and related ROCm software components.

Our goals are to:

1. **Standardize packaging behavior for native Windows ROCm software**
1. **Ensure predictable installation, upgrade, repair, multi-version support, and uninstall behavior**
1. **Provide redistributable-friendly Windows delivery mechanisms for developers, IT administrators, and ISVs**
1. **Align Windows packaging structure with the broader TheRock cross-platform packaging model where practical**
1. **Support automated artifact generation and productized deliverables from TheRock**

## Scope

### In Scope

- Native Windows packaging requirements from ROCm software build with TheRock
- MSI-based package requirements
- Winget package and meta-package requirements
- Runtime, development, and developer-tools package separation
- Installation directory layout and package granularity
- Multi-version installation policy for major.minor versions
- Patch upgrade behavior
- Environment variables, registry keys, and discovery mechanisms
- Logging, signing, and Windows-specific installation semantics
- Guidance for legacy System32 runtime migration

### Out of Scope

- Windows display driver packaging and WHQL process details
- WSL and WSL2 package requirements
- Internal CI/CD implementation details
- Full feature parity planning for every ROCm component on Day 1
- Legacy HIP SDK packaging behavior except where migration handling is required
- Microsoft Store-specific package requirements
- Python pip package requirements for Windows developer workflows
- Portable ZIP package requirements

## Windows Packaging Requirements

### Packaging Formats

Windows ROCm software must be delivered using packaging formats appropriate to the Windows ecosystem while preserving the same general package boundaries expected from TheRock on Linux.

The supported packaging formats are:

- **MSI packages** as the primary OS-integrated installation unit
- **Winget packages** as Windows package-manager-facing meta packages or installers that reference AMD-hosted MSI artifacts

Note that MSI packages are the authoritative Windows installation unit.

### Directory Layout

The ROCm Core SDK on Windows must be installed under a versioned installation root to support multi-version installation of major.minor releases.

Per-machine (default):

```
C:\Program Files\AMD\ROCm\Core-X.Y
```

Per-user:

```
%LOCALAPPDATA%\AMD\ROCm\Core-X.Y
```

Where:

- `X.Y` is the major and minor version
- Patch versions must be installed in place within the existing `X.Y` directory
- Multi-version installation is supported for different major.minor versions
- Patch-only multi-version installation is not supported

The installed directory structure must mirror the cross-platform ROCm layout as closely as practical:

```
C:\Program Files\AMD\ROCm\Core-X.Y\
  bin\
  lib\
  include\
  share\
  tools\
  version.txt
```

A convenience path to the most recently installed version will be maintained when practical:

```
C:\Program Files\AMD\ROCm\Core  ->  C:\Program Files\AMD\ROCm\Core-8.2
C:\Program Files\AMD\ROCm\Core-8  ->  C:\Program Files\AMD\ROCm\Core-8.2
C:\Program Files\AMD\ROCm\raytracing-8  ->  C:\Program Files\AMD\ROCm\raytracing-8.2
```

This allows users, scripts, and build systems to either target the latest installed release or pin to a major line while still preserving independently versioned install roots. On Windows, symbolic links require that the user has administrative privileges or Windows Developer Mode is enabled. This is not guaranteed in enterprise systems, CI environments, and customer deployments. Due to this, symlinks will not be required for the correct operation of ROCm and will be provided as an optional convenience feature only.

Additionally, all Windows caches for FFT and other programs will be stored in the following location:

Per-machine (default):

```
C:\ProgramData\AMD\ROCm\
```

Per-user:

```
%LOCALAPPDATA%\AMD\ROCm\
```

Caches are stored system wide and matches Windows guidelines for application data.

Example:

```
C:\ProgramData\AMD\ROCm\
  cache\
      fft\
      rtc\
      kernels\
      tuning\
      misc\
```

### Path Length Requirements

Native Windows 10 and higher have a traditional Win32 `MAX_PATH` limit of 260 characters, but can support up to 32,767 characters if the following two conditions are met:

1. **The OS enabled the long paths option**: Computer Configuration -> Administrative Templates -> System -> Filesystem -> Enable Win32 long paths.
1. **The program** must include the <longPathAware>true</longPathAware> XML tag.

The enablement of long paths is required for installed SDK file paths that may exceed `MAX_PATH`. This includes CMake files, headers, and similar SDK content, and excludes the runtimes. All redistributable runtimes will support the default `MAX_PATH` length of 260 characters.

The installer should have an option to enable:

```
HKLM\SYSTEM\CurrentControlSet\Control\FileSystem
LongPathsEnabled = 1
```

The redistributable installers include:

| File name                   | Friendly name                            |
| :-------------------------- | :--------------------------------------- |
| amdrocm-runtimes.msi        | ROCm Runtime Redistributable             |
| amdrocm-core.msi            | ROCm Core Runtime Redistributable        |
| amdrocm-core-devel.msi      | ROCm Core SDK Development                |
| amdrocm-developer-tools.msi | ROCm Core Developer Tools                |
| amdrocm-core-sdk.msi        | ROCm Core SDK Redistributable            |
| amdrocm-raytracing.msi      | ROCm Ray Tracing Runtime Redistributable |
| amdrocm-raytracing-sdk.msi  | ROCm Ray Tracing SDK                     |

The above redistributable installers are required to operate within the default `MAX_PATH` limit of 260 characters and will not require long path support to be enabled.

### Decouple User Space from Adrenaline Driver

Windows packages must avoid reliance on `C:\Windows\System32` for ROCm runtime discovery.

All new Windows ROCm runtime components must be installed into the package installation root, primarily under `bin`, and discovered through one or more of the following supported mechanisms:

- Application-local deployment for redistributable scenarios
- `PATH` entries associated with the selected ROCm installation
- Registry-based SDK discovery
- Environment-variable-based SDK discovery
- Optional `ROCM_PATH` environment variable for convenience-based discovery of the selected ROCm installation

New installations must not place core ROCm runtime DLLs into `System32`. Legacy driver-installed runtime DLLs in `System32` that conflict with the new Windows packaging model must be detected and handled by the appropriate runtime installer. At a minimum, the Windows runtime package must handle cleanup of legacy `amdhip64` and `amd_comgr` placements when present, while preserving installer robustness if files are locked or permissions are insufficient.

### OpenCL Changes

OpenCL will largely in part be sustained and no changes are expected to be implemented. Installing, upgrading, or uninstalling the ROCm SDK must have no effect on an OpenCL environment.

Additionally, `amd_comgr_3.dll` will be renamed to `amd_comgr_opencl.dll` to better reflect its use case and so that it can be shipped alongside the driver version 26.30 in Q3.

### Package Naming

Windows package naming should remain aligned with the Linux TheRock naming model where practical so that users can reason about package purpose consistently across operating systems.

The `amdrocm-` naming prefix is used for AMD-published Windows package components where a package-level identity is exposed directly to users.

#### Meta Packages

The following meta packages aggregate fine-grained packages into user-facing installation units. These map 1:1 to the Linux meta packages defined in [RFC0009](./RFC0009-OS-Packaging-Requirements.md#meta-packages).

| File Name                     | Friendly Name                            | Included Packages                                                                                                                                                                                                                                                                                                 | Description                                 |
| :---------------------------- | :--------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------ |
| `amdrocm-runtimes.msi`        | ROCm Runtime Redistributable             | amdrocm-runtime, amdrocm-sysdeps                                                                                                                                                                                                                                                                                  | Run pre-built ROCm projects                 |
| `amdrocm-core.msi`            | ROCm Core Runtime Redistributable        | amdrocm-base, amdrocm-sysdeps, amdrocm-llvm, amdrocm-runtime, amdrocm-debugger, amdrocm-blas, amdrocm-rand, amdrocm-fft, amdrocm-solver, amdrocm-sparse, amdrocm-ck, amdrocm-dnn, amdrocm-rccl, amdrocm-rocshmem, amdrocm-amdsmi, amdrocm-hipify, amdrocm-decode, amdrocm-jpeg                                    | Run ROCm projects                           |
| `amdrocm-core-devel.msi`      | ROCm Core SDK Development                | amdrocm-core + amdrocm-runtime-devel, amdrocm-llvm-devel, amdrocm-fft-devel, amdrocm-blas-devel, amdrocm-sparse-devel, amdrocm-solver-devel, amdrocm-rand-devel, amdrocm-ccl-devel, amdrocm-opencl-devel, amdrocm-rccl-devel, amdrocm-rocshmem-devel, amdrocm-dnn-devel, amdrocm-decode-devel, amdrocm-jpeg-devel | Build software with ROCm Core               |
| `amdrocm-developer-tools.msi` | ROCm Core Developer Tools                | amdrocm-base, amdrocm-amdsmi, amdrocm-profiler-base, amdrocm-profiler                                                                                                                                                                                                                                             | Debug and optimize ROCm projects            |
| `amdrocm-core-sdk.msi`        | ROCm Core SDK Redistributable            | amdrocm-core-devel, amdrocm-developer-tools, amdrocm-rdc, amdrocm-opencl                                                                                                                                                                                                                                          | Everything                                  |
| `amdrocm-raytracing.msi`      | ROCm Ray Tracing Runtime Redistributable | Ray tracing runtime libraries, acceleration structures, and GPU architecture-specific binaries                                                                                                                                                                                                                    | Run ROCm ray tracing workloads              |
| `amdrocm-raytracing-sdk.msi`  | ROCm Ray Tracing SDK                     | Ray tracing development headers, SDK libraries, samples, and tooling                                                                                                                                                                                                                                              | Develop and build ROCm ray tracing projects |

> **Note:** The ray tracing packages are planned for a different ROCm release and will be further defined in the future.

#### Fine-Grained Packages

Windows package granularity must match the Linux model defined in [RFC0009](./RFC0009-OS-Packaging-Requirements.md#package-granularity). Each fine-grained package separates runtime and development components. On Windows these are implemented as MSI features or individual MSI packages, depending on installer architecture.

| Package Name                 | Runtime Contents                                            | Dev Package Contents (additional)                |
| :--------------------------- | :---------------------------------------------------------- | :----------------------------------------------- |
| `amdrocm-base`               | rocminfo, rocm-core, rocprofiler-register, rocm-cmake, half |                                                  |
| `amdrocm-llvm`               | amd-llvm, hipcc, aux-overlay                                | amd-llvm headers, hipcc headers                  |
| `amdrocm-runtime`            | ROCR-Runtime, CLR, rocm-kpack, amd-comgr                    | ROCR-Runtime headers, CLR headers, comgr headers |
| `amdrocm-blas`               | hipBLAS, rocBLAS, hipBLASLt, hipSPARSELt                    | hipBLAS headers, rocBLAS headers, hipBLAS-common |
| `amdrocm-sparse`             | rocSPARSE, hipSPARSE                                        | rocSPARSE headers, hipSPARSE headers             |
| `amdrocm-solver`             | rocSOLVER, hipSOLVER                                        | rocSOLVER headers, hipSOLVER headers             |
| `amdrocm-fft`                | rocFFT, hipFFT                                              | rocFFT headers, hipFFT headers                   |
| `amdrocm-rand`               | rocRAND, hipRAND                                            | rocRAND headers, hipRAND headers                 |
| `amdrocm-ccl-devel`          |                                                             | rocPRIM, hipCUB, rocThrust, libhipcxx, rocWMMA   |
| `amdrocm-ck`                 | composable_kernel                                           |                                                  |
| `amdrocm-dnn`                | hipDNN, MIOpen                                              | hipDNN headers, MIOpen headers, flatbuffers      |
| `amdrocm-hipify`             | HIPIFY                                                      |                                                  |
| `amdrocm-math-common`        | SuiteSparse, host-blas                                      |                                                  |

Winget package identifiers may use Windows ecosystem naming conventions such as `AMD.ROCm`, but they should map cleanly to the same product and component boundaries.

### ROCm Installer Branding

All installers will have proper ROCm and AMD branding. This includes the ROCm logo and the AMD logo that will be included on the installers GUI.

### Installation Configurations

Windows installation flows must support multiple install configurations aligned with Windows user expectations and the ROCm SDK for Windows product direction.

At minimum, the following install configurations must be supported:

- **Core SDK**: default developer installation
- **Runtime Only**: minimal runtime footprint for executing prebuilt applications and redistribution workflows
- **Compile Only**: HIP API headers, the HIP compiler toolchain, and CMake configuration files required to compile HIP and ROCm applications on machines without an AMD GPU present. This configuration must not require GPU detection, driver presence, or runtime libraries at install time and is intended for headless build servers, CI agents, and cross-compilation workflows
- **Developer Tools Only**: debugging, profiling, and diagnostics when runtime is already present
- **Custom**: component-level selection for advanced users where supported by package dependency rules. Users must be able to select or deselect individual component groups (e.g., HIP API, runtime, compiler, math libraries, communication libraries) independently, subject to declared dependency constraints

These configurations may be implemented as separate packages, features, meta packages, or a combination thereof, but their behavior must remain well-defined and documented.

### MSI Package Requirements

MSI packages are the primary Windows installation mechanism and must satisfy the following requirements:

- Support silent installation
- Support logging
- Support default installation path behavior
- Support custom installation path overrides
- Support uninstall and repair flows through standard Windows Installer semantics
- Support per-machine installation by default
- Support per-user installation where technically valid for the selected package set
- Be digitally signed by AMD
- Avoid partially installed states and maintain transactional integrity to the degree supported by Windows Installer

Example installation commands:

```
msiexec /i amdrocm-core-sdk.msi /quiet
msiexec /i amdrocm-core-sdk.msi INSTALLDIR="D:\tools\rocm\core-8.2" /quiet
msiexec /x amdrocm-core-sdk.msi /quiet
msiexec /famus amdrocm-core-sdk.msi /quiet
```

MSI installers should be GUI-less or minimal-UI by default and must support unattended enterprise deployment.

### Command-Line Installation Interface

The ROCm installer must provide a first-class command-line installation interface that supports both interactive and non-interactive workflows. This is required to serve headless build machines, CI/CD agents, and developers who prefer terminal-based tooling.

Component selection is driven by MSI features exposed through the standard `ADDLOCAL` property. At minimum, the following feature identifiers must be independently selectable:

| Feature identifier | Contents                                          |
| :----------------- | :------------------------------------------------ |
| `HipApi`           | HIP API headers and CMake configuration           |
| `HipRuntime`       | HIP runtime                                       |
| `HipCompiler`      | HIP compiler toolchain (clang, device libraries)  |
| `CoreRuntime`      | ROCm core runtime libraries                       |
| `MathLibs`         | Math libraries (rocBLAS, rocFFT, rocSPARSE, etc.) |
| `RayTracing`       | Ray tracing libraries                             |

**Interactive CLI mode:**

When invoked from a terminal without `/quiet`, the MSI installer must present an interactive dialog or console-based menu that lists available component groups and allows the user to select which to install.

Example:

```
msiexec /i amdrocm-core-sdk.msi

ROCm SDK Installer vX.Y.Z
Select components to install (space to toggle, enter to confirm):

  [x] HIP API headers and CMake configuration
  [x] HIP runtime
  [x] HIP compiler toolchain
  [ ] ROCm core runtime libraries
  [ ] Math libraries (rocBLAS, rocFFT, rocSPARSE, ...)
  [ ] Communication libraries (RCCL, rocSHMEM)
  [ ] Developer tools (profiler, debugger, tracer)
  [ ] Ray tracing libraries

Selected: HIP API, HIP runtime, HIP compiler
Proceed? [Y/n]
```

**Non-interactive CLI mode:**

The installer must accept MSI properties for fully unattended component selection. This enables scripted CI/CD provisioning and infrastructure-as-code workflows without requiring GUI interaction or manual menu navigation.

Example:

```
msiexec /i amdrocm-core-sdk.msi ADDLOCAL=HipApi,HipRuntime,HipCompiler /quiet
msiexec /i amdrocm-core-sdk.msi ADDLOCAL=HipApi,HipCompiler TARGETARCH=gfx1100 /quiet
msiexec /i amdrocm-core-sdk.msi INSTALLCONFIG=CompileOnly /quiet
msiexec /i amdrocm-core-sdk.msi ADDLOCAL=ALL /quiet
```

The full list of available feature identifiers and their descriptions should be documented alongside the installer and queryable via standard MSI tooling.

**Headless and GPU-less build machine support:**

The installer must not require an AMD GPU to be present on the target machine. GPU auto-detection must be treated as an optional convenience for selecting device-specific packages, not a prerequisite for installation. When no GPU is detected, the installer must:

- Allow installation to proceed without error
- Skip device-specific binary packages unless the user explicitly specifies target architectures via the `TARGETARCH` property
- Install all requested host-side components (headers, compiler, CMake configs, libraries) without degradation

This enables the common workflow where developers compile HIP applications on GPU-less build servers and deploy to GPU-equipped machines.

### Device-Specific Architecture Packages

Users are encouraged to identify their local GPU architecture and install packages exclusive to the GPU architectures present. Otherwise, users can install a complete ROCm installation with all GPU architectures to enable all GPUs.

The following two installer options will be available:

1. **General Installer**: Install packages for all supported architectures.
1. **Architecture Specific Installer**: Install packages for a specific GPU architecture family (e.g., gfx-110x)

All device-specific packages must:

- Not conflict with each other
- Be independently installable
- Support meta-packages

Additionally device specific installation must support the following use cases:

1. **ISV installer invokes ROCm Runtime Core via winget**:

Winget starts launcher
Launcher automatically detects available GPU architectures
Runs installers for each GPU architecture (host installers and per device installers)

2. **Software developer use case**:

Downloads ROCm launcher
Selects full SDK
Launcher detects local GPUs
Installs host and device msi files

3. **Software developer full installation**:

Downloads ROCm launcher
Selects full SDK
Selects ALL GPU architectures

4. **Direct download from AMD website**:

User wants to get ROCm Runtime and ROCm Core for gfx family
Multiple msi files are downloaded for use case and gfx family

5. **Headless build machine without AMD GPU**:

Developer or CI system installs ROCm compile toolchain on a machine with no AMD GPU
Launcher skips GPU auto-detection gracefully
Installs host-only packages (HIP API, compiler, headers, CMake configs)
User optionally specifies `TARGETARCH` to include device libraries for cross-compilation targets (e.g., `msiexec /i amdrocm-core-sdk.msi ADDLOCAL=HipApi,HipCompiler TARGETARCH=gfx1100 /quiet`)

It should also be noted that Windows installation should be published in `repo.amd.com/rocm/win/...`.

### Installation Logic and Version Handling

Upon execution, MSI packages must inspect the installation target and apply deterministic version-handling rules.

The following behavior matrix must be supported:

| Scenario                                                   | Behavior                                                                                |
| :--------------------------------------------------------- | :-------------------------------------------------------------------------------------- |
| No ROCm installation at target path                        | Installs normally                                                                       |
| Older version detected at the same target path             | Perform in-place upgrade                                                                |
| Same version detected at the same target path              | Return success with no action, unless an explicit repair or reinstall mode is requested |
| Newer version detected at the same target path             | Abort with error and instruct the user to uninstall or choose a different path          |
| Different major.minor version detected at a different path | Allow multi-version installation                                                        |

Patch versions must upgrade in place within the same `X.Y` installation root.

Major.minor releases must be installable side by side in distinct versioned roots.

### Upgrade and Uninstall Requirements

Windows packages must provide predictable and clean upgrade and uninstall behavior.

Upgrade requirements:

- In-place upgrades must preserve the expected installation root for the target `X.Y` line
- Upgrade flows must avoid overwriting unrelated user environment settings
- Upgrade flows must not leave stale package-owned files in shared locations when the package manager can safely remove them
- Package upgrade behavior must remain consistent whether invoked directly through MSI or through winget

Uninstall requirements:

- Remove files owned by the installation being removed
- Remove environment-variable updates owned by that installation if they still reference that installation
- Remove registry entries created by that installation
- Remove package-owned `PATH` entries associated with that installation only
- Avoid impacting other installed ROCm major.minor versions

### Environment Variables

After successful installation, Windows installers must publish a stable discovery mechanism for tools and build systems without introducing conflicts between multiple ROCm installations or non-standard deployments.

At minimum:

- `ROCM_PATH` may point to the installation root of the latest installed and active ROCm version, but must be treated as a convenience variable only, not a guaranteed or authoritative source of truth
- The selected installation's `bin` directory may be prepended to the relevant `PATH`, provided:
  - Duplicate `PATH` entries are not introduced across reinstalls or upgrades
  - Existing user or system configuration is not overridden in a way that breaks other ROCm installations or development environments
- Per-machine installs must modify machine-scoped environment variables
- Per-user installs must modify user-scoped environment variables only

The following constraints apply:

- Tools and libraries within the same ROCm installation must be able to discover one another without relying on global environment variables such as `ROCM_PATH`
- Applications and build systems must not assume a fixed installation path, as ROCm may be installed in custom directories, build trees, or distributed via package managers such as Python wheels
- Build systems and applications that require deterministic selection of a specific ROCm version should rely on:
  - Versioned installation directories
  - Explicit configuration (e.g., CMake/toolchain files)
  - Registry-based discovery where applicable

The convenience variable `ROCM_PATH` is last-writer-wins. Build systems and applications that require deterministic selection of a specific version should rely on versioned install paths and registry-based discovery rather than assuming `ROCM_PATH` is pinned permanently.

### Registry Requirements

To support discovery and multiple versioning, Windows ROCm installers must create versioned registry keys. It should also be noted that registry key locations are system wide, not per user.

Per-machine installs:

```
HKLM\Software\AMD\ROCm\X.Y\
```

Per-user installs:

```
HKCU\Software\AMD\ROCm\X.Y\
```

At minimum, each versioned key must contain:

- `InstallDir`
- `ProductCode`
- `Version`

Additionally, a convenience pointer to the latest installed active version should be maintained:

```
HKLM\Software\AMD\ROCm\CurrentVersion
HKCU\Software\AMD\ROCm\CurrentVersion
```

This convenience pointer is also last-writer-wins and exists to support straightforward SDK discovery by tools and administrators. Uninstallation must also clean up the registry keys.

### Driver Compatibility

Driver packaging is out of scope for this RFC, but Windows SDK packaging must be designed around an explicit driver compatibility contract.

Windows ROCm packages must:

- Publish a compatibility matrix describing supported driver ranges for each ROCm release line
- Avoid coupling SDK patch delivery to mandatory driver rebundling wherever possible
- Provide install-time or first-run preflight checks that warn when the installed driver is outside the supported compatibility range

The Windows packaging contract must assume that the display driver and the ROCm SDK are separate deliverables, even where an AMD driver may bundle or involve installation of a runtime-oriented package or compatibility purposes. It should be noted that users are expected to self install the driver in accordance with this.

### Winget Requirements

Winget packages are the Windows package-manager-facing distribution layer for ROCm.

Winget packages must:

- Reference AMD-hosted MSI installers
- Include version metadata and installer hash validation
- Support silent installation flows
- Preserve MSI-defined upgrade and uninstall behavior
- Support installation-path override capability where the underlying MSI supports it
- Maintain clear package naming and publisher metadata

A top-level package identifier such as `AMD.ROCm` may be used for the primary Windows SDK experience. Additional componentized identifiers may be introduced if needed, but must remain aligned with the same package boundaries defined by this RFC.

### Visual Studio Extension Requirements

A Visual Studio extension for ROCm must support deterministic discovery of the ROCm toolchain and associated build binaries on Windows. The plugin must support two binding modes:

**Bind built binaries with latest**
The extension resolves the SDK/toolchain root from an environment variable, in this case `ROCM_PATH`. This allows projects to automatically build against the most recently installed ROCm version.

**Bind fixed version using registry keys**
The extension resolves the SDK/toolchain root from a version-specific installation record (e.g., Windows registry or installer metadata). This allows projects to bind to a specific ROCm version for reproducible builds and CI environments.

The extension must clearly indicate the resolved SDK path and version used for the build.

### Logging Requirements

Windows installers will provide detailed logging for installation, upgrade, repair, uninstall, and cleanup actions.

Installer logs must include, at minimum:

- Installation path selection
- Existing-version detection
- Version comparison result
- Environment-variable updates
- Registry writes and removals
- Legacy runtime cleanup actions when applicable
- Reboot scheduling if cleanup of locked files requires deferred removal

Installers should also document:

- Publication location
- Download instructions
- Log file location and retrieval steps
- Any additional troubleshooting or diagnostic guidance

Example:

```
msiexec /i amdrocm-core-sdk.msi /l*vx install.log
```

### Security and Signing Requirements

All Windows ROCm distribution artifacts must follow AMD signing and transport requirements.

- MSI installers must be digitally signed by AMD
- AMD-hosted package endpoints must use HTTPS
- Winget manifests must include hash validation

### Redistribution Requirements

Windows packaging must support a clear redistribution story for ISVs without requiring every end user to install a full development SDK.
The supported redistribution models are:

1. **Application-local bundled runtime files** for supported runtime subsets
1. **Optional runtime-oriented MSI or winget install path** for customers who prefer a system-installed runtime

Redistribution documentation must clearly distinguish:

- Development SDK installation
- Runtime-only installation
- Application-local redistribution
