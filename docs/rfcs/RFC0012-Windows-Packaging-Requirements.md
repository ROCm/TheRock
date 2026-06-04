---
author: Liam Berry (LiamfBerry), Saad Rahim (saadrahim)
created: 2026-03-13
modified: 2026-06-03
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
| amdrocm-compat.msi          | ROCm System32 Compatibility Libraries    |
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
- Registry-based SDK discovery
- Environment-variable-based SDK discovery
- Optional `ROCM_PATH` environment variable for convenience-based discovery of the selected ROCm installation

Process-wide `PATH` modifications must not be used for DLL discovery. `PATH`-based lookup is global, order-sensitive, difficult to audit, and creates DLL preloading exposure.

New installations must not place core ROCm runtime DLLs into `System32`. Legacy driver-installed runtime DLLs in `System32` that conflict with the new Windows packaging model must be detected and handled by the appropriate runtime installer. At a minimum, the Windows runtime package must handle cleanup of legacy `amdhip64` and `amd_comgr` placements when present, while preserving installer robustness if files are locked or permissions are insufficient.

> **Exception (HIP 6 and HIP 7):** The dedicated `amdrocm-compat.msi` package is a scoped exception to the "no DLLs in `System32`" rule above. It installs HIP 6 and HIP 7 compatibility DLLs into `C:\Windows\System32` to preserve compatibility with applications that resolve HIP and comgr DLLs from the legacy driver-managed location. `amdrocm-compat.msi` is a single, system-wide package — it is not versioned by ROCm `major.minor`, is not installed side by side with itself, and always upgrades in place to the latest version installed on the system. HIP 6 and HIP 7 entries will remain in `System32` until each release line reaches end-of-life — the future release of HIP 8 will not trigger their removal. HIP 8, when released, will not be installed into `System32`. See [amdrocm-compat.msi: System32 Compatibility Libraries](#amdrocm-compatmsi-system32-compatibility-libraries) for the full model, frozen unversioned `amdhip64.dll`, and planned HIP 8 behavior.

### amdrocm-compat.msi: System32 Compatibility Libraries

`System32` placement of HIP and comgr DLLs is delivered by a dedicated MSI, `amdrocm-compat.msi`, separate from `amdrocm-runtimes.msi` and from every other ROCm package. It exists solely to provide a transition path for applications that resolve HIP and comgr DLLs from the legacy driver-managed `System32` location. This is a deliberate, scoped exception to the general rule defined in [Decouple User Space from Adrenaline Driver](#decouple-user-space-from-adrenaline-driver) that new installations must not place ROCm runtime DLLs into `System32`.

The exception applies to the HIP 6 and HIP 7 release lines. HIP 8 is a future release and is not yet available; when it ships, it will not be installed into `System32`, and its release will not remove HIP 6 or HIP 7 from `System32` — each line will be removed only when it reaches end-of-life.

> **Note (implementation flexibility):** This RFC defines `amdrocm-compat.msi` as a distinct package because it is the cleanest way to satisfy the single-version-system-wide, always-latest, no-side-by-side, and EOL-driven cleanup behaviors below — none of which apply to any other ROCm package. An alternative implementation that delivers the same `System32` payload from `amdrocm-runtimes.msi` is also acceptable, provided every requirement in this section is met in full: the `System32` payload must remain single-version system-wide and always-latest even when multiple `major.minor` ROCm releases are installed side by side, its lifecycle must be decoupled from any specific `major.minor` runtime install or uninstall, and uninstalling a particular `major.minor` runtime must not remove the `System32` payload while another HIP release line is still supported. Folding the compatibility payload into the `major.minor`-versioned `amdrocm-runtimes.msi` is meaningfully more complex because Windows Installer naturally scopes payload lifecycle to the product code, so the implementation must explicitly override that behavior. A separate `amdrocm-compat.msi` is the recommended path.

#### Package Identity and Versioning

`amdrocm-compat.msi` is distinct from the `major.minor`-versioned ROCm packages defined elsewhere in this RFC and does not follow the [Directory Layout](#directory-layout) install root or the [Installation Logic and Version Handling](#installation-logic-and-version-handling) side-by-side rules.

- **Single version, system-wide.** Only one version of `amdrocm-compat.msi` may exist on a system at any time. It is not versioned by ROCm `major.minor` and is not eligible for side-by-side installation with itself. The installer must enforce this directly: it uses a stable, version-independent `UpgradeCode` so that Windows Installer treats every release as an upgrade of the same product, and any attempt to install a second copy alongside an existing one must either upgrade in place or fail with a clear diagnostic — it must never result in two copies coexisting.
- **Always latest.** The installed copy of `amdrocm-compat.msi` always represents the latest version of the compatibility payload available on the system. Newer releases of `amdrocm-compat.msi` upgrade the existing install in place. Older releases must detect a newer installed copy and abort rather than downgrading.
- **Single source of `System32` payload.** No other Windows ROCm package writes HIP or comgr DLLs to `System32`. `amdrocm-runtimes.msi` and all other `amdrocm-*` packages install exclusively under the package installation root defined in [Directory Layout](#directory-layout).
- **Independently installable and removable.** `amdrocm-compat.msi` can be installed without any other ROCm package present, and it can be removed without affecting any side-by-side `major.minor` ROCm installation.

#### Transition Model

The transition requires coordinated changes across the Adrenaline driver and `amdrocm-compat.msi`:

- The Adrenaline driver must stop shipping HIP and comgr runtime DLLs, with the sole exception of the OpenCL-specific comgr DLL (`amd_comgr_opencl.dll`, see [OpenCL Changes](#opencl-changes)).
- The Adrenaline driver must bundle and invoke `amdrocm-compat.msi` as part of driver installation so that existing applications that resolve HIP and comgr DLLs from `System32` continue to function without modification.
- `amdrocm-compat.msi` becomes the single source of truth for HIP and comgr DLL placement under `System32` on Windows systems that previously relied on the driver for these components.

#### System32 Placement Rules

`amdrocm-compat.msi` must install the following into `C:\Windows\System32`:

- **The HIP 6 compatibility DLL**, named `amdhip64_6.dll` per the [Versioned DLL Naming](#versioned-dll-naming) convention.
- **The HIP 7 compatibility DLL, installed under both names**: the versioned name `amdhip64_7.dll`, and the unversioned name `amdhip64.dll`. Both names refer to the same HIP 7 binary — they are byte-identical copies of the same file, not a forwarder or thin shim. The unversioned name preserves compatibility with applications that were built against the pre-versioned HIP runtime contract previously shipped by the Adrenaline driver; the versioned name allows HIP 7-aware loaders to resolve by exact ABI and allows the HIP 6 and HIP 7 binaries to coexist in `System32` without conflict, since the PE loader resolves by basename.
- **The HIP 6 and HIP 7 comgr DLLs** required by the corresponding HIP runtime entries.

The unversioned `amdhip64.dll` in `System32` is treated as a frozen compatibility surface for the lifetime of HIP 7:

- It is permanently frozen at the HIP version 7 ABI. Its content tracks `amdhip64_7.dll` and is not advanced by any release of `amdrocm-compat.msi` to a higher ABI.
- Newer releases of `amdrocm-compat.msi` must not overwrite the unversioned `amdhip64.dll` with a higher-versioned ABI, and must not remove it during servicing prior to HIP 7 end-of-life.
- Applications built against newer HIP ABIs must link against the versioned DLL name (for example `amdhip64_8.dll`) and must not depend on the unversioned name.

#### Interaction with Legacy Cleanup

The legacy cleanup behavior defined in [Decouple User Space from Adrenaline Driver](#decouple-user-space-from-adrenaline-driver) must be reconciled with the transition rules above:

- During servicing of the HIP 6 and HIP 7 lines, `amdrocm-compat.msi` must not remove the versioned HIP DLLs or the unversioned `amdhip64.dll` it installs into `System32`.
- Legacy driver-installed `amdhip64.dll` and `amd_comgr*.dll` files present in `System32` from prior driver releases must be replaced by the files owned by `amdrocm-compat.msi` during the transition install, so that ownership and update responsibility transfer cleanly to the MSI.
- Any package other than `amdrocm-compat.msi` must continue to follow the default rule and must not place runtime DLLs into `System32`.

#### Lifecycle and EOL Removal

Because `amdrocm-compat.msi` is a single, latest-only package, lifecycle behavior is defined per HIP release line rather than per MSI version:

- Adding support for a new HIP line is delivered by a newer release of `amdrocm-compat.msi` that supersedes the previously installed copy in place.
- A HIP line's `System32` payload (for example, `amdhip64_7.dll` and, where applicable, the unversioned `amdhip64.dll`) is removed only when that HIP line reaches end-of-life. EOL removal is delivered by a future release of `amdrocm-compat.msi` whose payload no longer contains the EOL'd binaries.
- Uninstalling `amdrocm-compat.msi` removes all `System32` payload owned by the currently installed copy. Uninstalling `amdrocm-compat.msi` must not affect any `major.minor` ROCm installation under `C:\Program Files\AMD\ROCm`.

#### HIP 8 and Beyond (Future)

HIP 8 is a future ROCm release and is not available today. The behavior described in this subsection is the planned model for HIP 8 and later release lines; specific timelines and exact behavior will be confirmed in a future RFC revision once HIP 8 enters active development.

When HIP 8 ships:

- HIP 8 will not be added to the `amdrocm-compat.msi` payload. HIP 8 runtime DLLs will be installed exclusively under the package installation root defined in [Directory Layout](#directory-layout), through `amdrocm-runtimes.msi` and the other `major.minor`-versioned ROCm packages.
- `amdrocm-compat.msi` will not produce or update an unversioned `amdhip64.dll` in `System32` at a HIP 8 ABI. The unversioned `amdhip64.dll` will remain frozen at the HIP 7 ABI and continue to be serviced through the HIP 7 line only.
- **The release of HIP 8 will not remove HIP 6 or HIP 7 from `System32`.** HIP 6 and HIP 7 binaries will remain in `System32` until each line reaches end-of-life; removal is tied to EOL of that line, not to the release of any newer HIP version. This allows HIP 6, HIP 7, and HIP 8 to be present simultaneously on the same system without one breaking the others' compatibility surface.
- HIP 7 servicing will not be extended to GPU architectures introduced after HIP 8 ships. Newer GPU architectures will be supported only by HIP 8 and later; applications that need to target post-HIP-8 architectures will need to move off the unversioned `amdhip64.dll` and the `amdhip64_7.dll` entry point.
- Applications targeting HIP 8 and above will be required to use a secure DLL load mechanism to resolve the HIP runtime rather than relying on `System32` placement or `PATH`-based lookup. The specific mechanism is to be determined following stakeholder consultation and will be defined in a future RFC revision; candidate approaches include registry-based SDK discovery, application-local deployment, and explicit fully-qualified load paths derived from the installation root.
- Until the HIP 8 secure load mechanism is finalized, applications planning for HIP 8 should use the package-local discovery mechanisms defined in [Decouple User Space from Adrenaline Driver](#decouple-user-space-from-adrenaline-driver) and must not depend on `System32` placement.

### Versioned DLL Naming

Windows ROCm runtime DLLs must use ABI-versioned filenames to prevent loader collisions when multiple ROCm versions coexist in a single process or on a single system.

The Windows PE loader resolves transitive DLL dependencies by basename through a global per-process module list. If two ROCm installations ship the same unversioned DLL name (e.g., `amdhip64.dll`), the loader may bind to whichever copy was loaded first, regardless of version compatibility.

To prevent this:

- Runtime DLLs with unstable or versioned ABIs must encode the ABI version in the filename (e.g. `amdhip64_7.dll`)
- Corresponding import libraries must reference the versioned filename
- Versioned naming must be applied at build time through TheRock so that downstream consumers link against the correct versioned name

This aligns with the top-ranked DLL resolution strategies: versioned filenames prevent loaded-module-list collisions and allow multiple ABI versions to coexist safely.

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
| `amdrocm-compat.msi`          | ROCm System32 Compatibility Libraries    | HIP 6 and HIP 7 compatibility DLLs (`amdhip64_6.dll`, `amdhip64_7.dll`, unversioned `amdhip64.dll`, and corresponding comgr DLLs)                                                                                                                                                                                 | Legacy `System32` resolution for pre-existing applications. Single, latest-only, not versioned by `major.minor`. See [amdrocm-compat.msi: System32 Compatibility Libraries](#amdrocm-compatmsi-system32-compatibility-libraries) |
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
- Avoid impacting other installed ROCm major.minor versions

### Environment Variables

After successful installation, Windows installers must publish a stable discovery mechanism for tools and build systems without introducing conflicts between multiple ROCm installations or non-standard deployments.

At minimum:

- `ROCM_PATH` may point to the installation root of the latest installed and active ROCm version, but must be treated as a convenience variable only, not a guaranteed or authoritative source of truth
- Per-machine installs must modify machine-scoped environment variables
- Per-user installs must modify user-scoped environment variables only

The installer must not prepend the ROCm `bin` directory to the system or user `PATH` for runtime DLL discovery. `PATH`-based DLL lookup is a poor strategy for DLL resolution on Windows due to global scope, order sensitivity, and DLL preloading exposure. Applications and build systems that need ROCm binaries on `PATH` for command-line tool invocation (e.g., `hipcc`) should add the path explicitly in their own environment rather than relying on a system-wide installer modification.

The following constraints apply:

- Tools and libraries within the same ROCm installation must be able to discover one another without relying on global environment variables such as `ROCM_PATH`
- Applications and build systems must not assume a fixed installation path, as ROCm may be installed in custom directories, build trees, or distributed via package managers such as Python wheels
- Build systems and applications that require deterministic selection of a specific ROCm version should rely on:
  - Versioned installation directories
  - Explicit configuration (e.g., CMake/toolchain files)
  - Registry-based discovery where applicable

The convenience variable `ROCM_PATH` is last-writer-wins. Build systems and applications that require deterministic selection of a specific version should rely on versioned install paths and registry-based discovery rather than assuming `ROCM_PATH` is pinned permanently.

> **Note:** `ROCM_PATH` may be sunsetted in a future ROCm release to avoid DLL hijacking risks. As a user- or machine-scoped environment variable, `ROCM_PATH` is writable by any process running with the same privilege as the user or administrator and can be redirected to attacker-controlled directories, causing tools and build systems that resolve ROCm binaries through it to load untrusted DLLs. New tooling should not be designed to depend on `ROCM_PATH` as an authoritative discovery source; registry-based SDK discovery and explicit version-pinned paths are the preferred mechanisms. The timeline for `ROCM_PATH` deprecation will be defined in a future RFC revision.

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
