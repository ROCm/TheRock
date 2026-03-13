---
author: Liam Berry (LiamfBerry), Saad Rahim (saadrahim)
created: 2026-03-13
modified: 2026-03-13
status: draft
---

# TheRock Windows Packaging Requirements

With the implementation of TheRock build system, native Windows packaging and installation requirements must be defined to complement the Linux software packaging requirements. This RFC defines the packaging, installation, upgrade, uninstallation, versioning, and distrobuition requirements for TheRock software on native Windows, including the ROCm Core SDK and related ROCm software components.

Our goals are to:

1. **Standardize packaging behaviour for native Windows ROCm sotware**
1. **Ensure predictable installation, upgrade, repair, side-by-side support, and uninstall behavior**
1. **Provide redistributable-friendly Windows delivery mechanisms for developers, IT administrators, and ISVs**
1. **Align Windows packaging structure with the broader TheRock cross-platform packaging model where practical**
1. **Support automated artifact generation and productized deliverables from TheRock**

## Scope

### In Scope

- Native Windows packaging requirements from ROCm software build with TheRock
- MSI-based package requirements
- Winget package and meta-package requirements
- Python pip package requirements for Windows developer workflows
- Portable ZIP package requirements
- Runtime, development, and developer-tools package separation
- Installation directory layout and package granularity
- Side-by-side installation policy for major.minor versions
- Patch upgrade behavior
- Environment variables, registry keys, and discovery mechanisms
- Logging, signing, and Windows-specific installation semantics
- Guidance for legacy System32 runtime cleanup and migration

### Out of Scope

- Windows display driver packaging and WHQL process details
- WSL and WSL2 package requirements
- Internal CI/CD implementation details
- Full feature parity planning for every ROCm component on Day 1
- Legacy HIP SDK packaging behavior except where migration handling is required
- Microsoft Store-specific package requirements

## Windows Packaging Requirements

### Packaging Formats

Windows ROCm software must be delivered using packaging formats appropriate to the Windows ecosystem while preserving the same general package boundaries expected from TheRock on Linux.

The supported packaging formats are:

- **MSI packages** as the primary OS-integrated installation unit
- **Winget packages** as Windows package-manger-facing meta packages or installers that reference AMD-hosted MSI artifacts
- **Python pip packages** for Python bindings, Python-first tooling, and environment-scoped developer workflows
- **ZIP archives** for portable, offline, CI, ir power-user scenarios

MSI packages are the authoritative Windows installation unit. Winget, pip, and ZIP deliverables must complement MSI behavior rather than redefine the core packaging contract.

### Directory Layout

The ROCm Core SDK on Windows must be installed under a versioned installation root to support side-by-side installation of major.minor releases.

```
C:\rocm\core-X.Y
```

Where:

- `X.Y` is the major and minor version
- Patch versions must be installed in place within the existing `X.Y` directory
- Side-by-side installation is supported for different major.minor versions
- Patch-only side-by-side installation is not supported

The installed directory structure must mirror the cross-platform ROCm layout as closely as practical:

```
C:\rocm\core-X.Y\
  bin\
  lib\
  include\
  share\
  tools\
  version.txt
```

A convenience path to the most recently installed version should be maintained when practical:

```
C:\rocm\core  ->  C:\rocm\core-8.2
C:\rocm\core-8  ->  C:\rocm\core-8.2
```

This allows users, scripts, and build systems to either target and latest installed release or pin to a major line while still preserving independently versioned install roots.

### DLL Search Order and Runtime Discovery

Windows packages must avoid reliance on `C:\Windows\System32` for ROCm runtime discovery.

All new Windows ROCm runtime components must be installed into the package installation root, primarily under `bin`, and discovered through one or more of the following supported mechanisms:

- Application-local deployment for redistributable scenarios
- `PATH` entries associated with the selected ROCm installation
- Registry-based SDK discovery
- Environment-variable-based SDK discovery

New installations must not place core ROCm runtime DLLs into `System32`. Legacy driver-installed runtime DLLs in `System32` that conflict with the new Windows packaging model must be detected and handled by the appropriate runtime installer. At a minimum, the Windows runtime package must handle cleanup of legacy `amdhip64` and `amd_comgr` placements when present, while preserving installer robustness if files are locked or permissions are insufficient.

### Package Naming

Windows package naming should remain aligned with the Linux TheRock naming model where practical so that users can reason about package purpose consistently across operating systems.

The `amdrocm-` naming prefix should be used for AMD-published Windows package componens where a package-level identity is exposed directly to users.

Examples inlcude:

- `amdrocm-runtimes`
- `amdrocm-core`
- `amdrocm-core-dev`
- `amdrocm-developer-tools`
- `amdrocm-core-sdk`

Winget package identifiers may use Windows ecosystem naming conventions such as `AMD.ROCm`, but they should map cleanly to the same product and component boundaries.

### Package Granularity

Windows package granularity should follow the same general model as Linux: runtime and development responsibilities must be seperable, and developer tools must be independently installable.

The following high-level package groupings must be avaialable:

| Name                      | Content                                                                                            | Description |
| :------------------------ | :------------------------------------------------------------------------------------------------- | :------------- |
| `amdrocm-runtimes`        | HIP runtime, runtime compiler support, required runtime libraries                                  |                |
| `amdrocm-core`            | Runtime components, core libraries, core utilities, discovery tools                                |                |
| `amdrocm-core-dev`        | Headers, CMake config files, import libraries, static libraries, compiler-facing development files |                |
| `amdrocm-developer-tools` | Debugging, profiling, diagnostics, and related developer tools                                     |                |
| `amdrocm-core-sdk`        | Core runtime, development files, and developer tools                                               |                |

Windows package composition may evolve as TheRock matures, but the runtime vs. development vs. tools split must remain clear.

### Installation Configurations

Windows installation flows must support multiple install configurations aligned with Windows user expectations and the ROCm SDK for Windows product direction.

At minimum, the following install configurations must be supported:

- **Core SDK**: default developer installation
- **Runtime Only**: minimal runtime footproint for executing prebuilt applications and redistribuition workflows
- **Developer Tools Only**: debugging, profiling, and diagnostics when runtime is already present
- **Custom**: component-level selection for advanced users where supported by package dependency rules

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
