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
