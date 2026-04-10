---
author: Liam Berry (LiamfBerry), Saad Rahim (saadrahim)
created: 2026-04-09
modified: 2026-04-10
status: draft
---

# ROCm SDK Archive Packaging for Windows and Linux

With the implementation of TheRock build system, a portable and system-agnostic delivery mechanism is required for the ROCm SDK on Windows to complement MSI-based installation and an equivalent system is required for Linux. This RFC defines the layout, behavioural, and distribution requirements for ZIP and TAR archive packages that deliver the ROCm SDK on Windows and Linux, respectively, aligned with TheRock.

Our goals are to:

1. **Provide a protable, zero-footprint SDK delivery mechansim** suitable for CI pipelines, offline deployments, and advanced users who manage their own SDK roots.
1. **Ensure ZIP archive layout is consistent and predictable** relative to the MSI-installed directory tree.
1. **Define ZIP as the authoritative upstream artifact** for Winget ingestion and internal automation hosted on `repo.amd.com`.
1. **Preserve full SDK functionality** when the extracted directory is used directly as the SDK root, without ambient system configuration.
1. **Ensure TAR archive layout is consistent and predictable**

## Scope

### In Scope

- ZIP archive layout and naming conventions for ROCm Windows packages built with TheRock.
- Behavioral constraints on ZIP packages with respect to the host system.
- SDK root portability requirements for tools and scripts contained within ZIP archives.
- Distribution hosting on repo.amd.com for Winget ingestion and internal automation.
- TAR archive layout and naming conventions for ROCm Linux packages built with TheRock.
- Behavioral constraints on TAR packages with respect to the host system.
- SDK root portability requirements for tools and scripts contained within TAR archives.
- Distribution hosting on repo.amd.com for TAR packages.

### Out of Scope

- MSI-based installation requirements.
- Winget package and meta-package requirements.
- Python pip package requirements.
- Windows display driver packaging.
- WSL and WSL2 package requirements.

## ZIP Package Requirements

### Package Format

ZIP packages must provide a portable file-tree representation of a Windows ROCm installation. The extracted layout must be identical to what a corresponding MSI installation produces at its versioned install root, ensuring interoperability between the two delivery mechanisms.

ZIP archives are intended for power users, CI pipelines, and offline deployments where MSI installation is unavailable or undesirable.

ZIP packages will be offered as either a per GPU architecture famiily denoted by `gfx<...>` or contain all GPU families with no denotion is this is the default. Both package types will offer two variants: a full version containing all build artifacts and a cleaned version that has the intermediary build files and test files (including unit tests) removed; this will be denoted with `-lite-`, in the package name.

All ZIP packages will also have an external sha256 hash validation file for integrity verification. 

### Naming Convention

ZIP archives must follow the following naming convention:

```
rocm-sdk-X.Y.Z.zip
rocm-sdk-gfx<architecture family>-X.Y.Z.zip
```

Additionally there will be two variant forms of these packages:

```
rocm-sdk-lite-X.Y.Z.zip
rocm-sdk-gfx<architecture family>-X.Y.Z.zip
```

Packages will all be paired with equivalent hash validation files:

```
rocm-sdk-gfx<...>-X.Y.Z.zip
rocm-sdk-gfx<...>-X.Y.Z.zip.sha256
```

### Directory Layout

The top-level directory inside the archive must be versioned to prevent extraction collisions when multiple versions are present on the same system:

```
rocm-sdk-X.Y.Z.zip
  rocm-sdk-X.Y\
      bin\
      lib\
      include\
      share\
      tools\
      version.txt
```

The extracted root of this directory serves as the SDK root for all tools and scripts contained within the archive.

### Behavioural Requirements

ZIP packages must be eintriely passive with respect to the host system. A ZIP package:

- Must not modify environment variables.
- Must not append to or modify `PATH`.
- Must not create, modify, or delete registry entries.
- Must not execute installation logic, pre/post-install scripts, or elevation prompts.

All system integration, including environment configuration, `PATH` management, and registry entries, is the responsibility of the caller; whether that is the user, a CI script, or a higher-level installer such as the MSI package.

### SDK Root Portability

Tools and scripts inside ZIP packages must function correctly when the extracted directory is used directly as the SDK root, without any ambient system configuration. Specifically:

- Binaries in `bin\` must not assume a fixed installation prefix set at build time.
- Path resolution within the SDK must be relative to the extracted root or dynamically resolved at runtime.
- Tools must remain suitable for CI, offline deployment, and advanced users operating without `ROCM_PATH` or `PATH` pre-configured by the environment.

### Distribution

ZIP archives are hosted on the AMD Official Repository:

```
repo.amd.com/rocm-ecosystem/nightly/core/zip
```

This repository serves as the authoritative source for Winget manifest ingestion and internal automation pipelines that consume versioned SDK artifacts directly. ZIP artifacts hosted here must be versioned and integrity-checked.

Additionally in repo.amd.com the zip archive will contain a `SHA256SUMS` file that contains hashes for all the provided packages.

```
repo.amd.com/rocm-ecosystem/nightly/core/zip
  <specified ZIP packages>
  SHA256SUMS
```

## TAR Package Requirements

### Package Format

TAR packages must provide a portable file-tree representation of a Linux ROCm installation. The extracted layout must be identical to what a standard installation produces at its versioned install root, ensuring interoperability between the two delivery mechanisms.

TAR archives are intended for power users, CI pipelines, and offline deployments where other installation methods are unavailable or undesirable.

TAR packages will be offered as either a per GPU architecture famiily denoted by `gfx<...>` or contain all GPU families with no denotion is this is the default. Both package types will offer two variants: a full version containing all build artifacts and a cleaned version that has the intermediary build files and test files (including unit tests) removed; this will be denoted with `-lite-`.

All TAR packages will also have an external sha256 hash validation file for integrity verification. 

### Naming Convention

TAR archives must follow the following naming convention:

```
rocm-sdk-X.Y.Z.tar.gz
rocm-sdk-gfx<architecture family>-X.Y.Z.tar.gz
```

Additionally there will be two variant forms of these packages:

```
rocm-sdk-lite-X.Y.Z.tar.gz
rocm-sdk-gfx<architecture family>-X.Y.Z.tar.gz
```

Packages will all be paired with equivalent hash validation files:

```
rocm-sdk-gfx<...>-X.Y.Z.tar.gz
rocm-sdk-gfx<...>-X.Y.Z.tar.gz.sha256
```

### Directory Layout

The top-level directory inside the archive must be versioned to prevent extraction collisions when multiple versions are present on the same system:

```
rocm-sdk-X.Y.Z.tar.gz
  rocm-sdk-X.Y\
      bin\
      lib\
      include\
      share\
      tools\
      version.txt
```

The extracted root of this directory serves as the SDK root for all tools and scripts contained within the archive.

### Behavioural Requirements

TAR packages must be entirely passive with respect to the host system. A TAR package:

- Must not modify environment variables.
- Must not append to or modify shell configuration files (.bashrc, .zshrc).
- Must not execute installation logic, pre/post-install scripts, or elevation prompts.
- Must not contain absolute paths.
- Must not contain path traversal entries (`../`).
- Ensure symlinks do not resolve outside the archive root.

### SDK Root Portability

Tools and scripts inside TAR packages must function correctly when the extracted directory is used directly as the SDK root, without any ambient system configuration. Specifically:

- Binaries in `bin\` must not assume a fixed installation prefix set at build time.
- Path resolution within the SDK must be relative to the extracted root or dynamically resolved at runtime.
- Scripts must use portable interpreters (e.g., /user/bin/env).
- Tools must remain suitable for CI, offline deployment, and advanced users operating without pre-configured environment variables.

### Distribution

TAR archives are hosted on the AMD Official Repository:

```
repo.amd.com/rocm-ecosystem/nightly/core/tar
```

This repository serves as the authoritative source for internal automation pipelines that consume versioned SDK artifacts directly. TAR artifacts hosted here must be versioned and integrity-checked.

Additionally in repo.amd.com the tar archive will contain a `SHA256SUMS` file that contains hashes for all the provided packages.

```
repo.amd.com/rocm-ecosystem/nightly/core/tar
  <specified TAR packages>
  SHA256SUMS
```

