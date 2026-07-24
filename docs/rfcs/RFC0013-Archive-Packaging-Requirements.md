---
author: Liam Berry (LiamfBerry), Saad Rahim (saadrahim)
created: 2026-04-09
modified: 2026-06-08
status: draft
---

# ROCm SDK Archive Packaging for Windows and Linux

With the implementation of TheRock build system, a portable and system-agnostic delivery mechanism is required for the ROCm SDK to complement platform-native installation (MSI on Windows). This RFC defines the layout, behavioural, and distribution requirements for the archive packages that deliver the ROCm SDK: ZIP archives on Windows and TAR archives on Linux.

Both archive formats are archives of the same SDK file-tree. The requirements below are therefore defined once as a shared contract, with a final section calling out the few genuinely platform-specific differences.

Our goals are to:

1. **Provide a portable, self-contained SDK delivery mechanism** that leaves no trace on the host outside the extracted directory — suitable for CI pipelines, offline deployments, and advanced users who manage their own SDK roots.
1. **Ensure archive layout is consistent and predictable** relative to the platform-native installed directory tree.
1. **Define the archive as the authoritative upstream artifact** for downstream automation hosted on `repo.amd.com`.
1. **Preserve full SDK functionality** when the extracted directory is used directly as the SDK root, without ambient system configuration.

## Scope

### In Scope

- Archive layout and naming conventions for ROCm packages built with TheRock (ZIP on Windows, TAR on Linux).
- Behavioural constraints on archive packages with respect to the host system.
- SDK root portability requirements for tools and scripts contained within archives.
- Distribution hosting on repo.amd.com for downstream automation.

### Out of Scope

- MSI-based installation requirements.
- Winget package and meta-package requirements.
- Python pip package requirements.
- Windows display driver packaging.
- WSL and WSL2 package requirements.

## Archive Contract

The following requirements apply to both ZIP and TAR archives. Platform-specific differences are listed in [Platform Specifics](#platform-specifics).

### Package Format

Archives must provide a portable file-tree representation of a ROCm installation. The extracted layout must mirror the platform-native installed layout as closely as practical, ensuring interoperability between the two delivery mechanisms.

Archives are intended for power users, CI pipelines, and offline deployments where platform-native installation is unavailable or undesirable.

### Package Types and Variants

Archives are offered along two axes:

- **Target scope** — either a single GPU target denoted by `gfx<target>` (e.g., `gfx1100`, `gfx1201`), or all GPU targets with no denotation (the default).
- **Content variant** — either a default package containing all build artifacts without the test files, or a *-test* package with basic unit tests, denoted by `-test` in the package name.

- <name> headers, cmake and static libraries, debug, build tools
- <name-test> test binaries and test dependencies

### Naming Convention

Archives must follow this naming convention, where `<os>` and `<ext>` are platform-specific (see [Platform Specifics](#platform-specifics)):

```
rocm-sdk-X.Y.Z-<os>.<ext>
rocm-sdk-gfx<target>-X.Y.Z-<os>.<ext>
rocm-sdk-lite-X.Y.Z-<os>.<ext>
rocm-sdk-lite-gfx<target>-X.Y.Z-<os>.<ext>
```

Where `<target>` is an individual GPU target identifier (e.g., `gfx1100`, `gfx1201`) rather than a GPU family grouping, aligning with Python package and native Linux package distribution conventions.

The `<os>` and `<ext>` tags describe two different things and are both required:

- **`<ext>`** is the container format — how the file is packed (`zip` vs `tar.gz`). A consumer needs it to know whether to unzip or untar the file.
- **`<os>`** is the platform the SDK inside was built for. It is explicit metadata about the payload, not the container.

These are independent: ZIP and TAR formats both work on either operating system, so the extension alone cannot tell you which platform an archive targets. By convention TheRock pairs `windows` with `zip` and `linux` with `tar.gz` (see [Platform Specifics](#platform-specifics)), but the explicit `<os>` tag means consumers can identify the target platform directly rather than inferring it from that pairing. The `<os>` field is also where the host CPU architecture is recorded when needed (below).

The host CPU architecture is part of this model but defaults to `x86_64` when omitted, which is the only architecture currently distributed. When non-x86 host builds are distributed (see [issue #5518](https://github.com/ROCm/TheRock/issues/5518)), only those archives carry an explicit architecture suffix, leaving x86_64 names unchanged:

```
rocm-sdk-gfx<target>-X.Y.Z-linux.tar.gz            # x86_64 host (implicit)
rocm-sdk-gfx<target>-X.Y.Z-linux_ppc64le.tar.gz    # non-x86 host (explicit)
```

Each archive is paired with an equivalent hash validation file (see [Integrity](#integrity)):

```
rocm-sdk-gfx<target>-X.Y.Z-<os>.<ext>
rocm-sdk-gfx<target>-X.Y.Z-<os>.<ext>.sha256
```

> Note: This per-target scheme is the intended convergence point for ROCm archive naming and supersedes the existing nightly `therock-dist-*` tarball naming (which currently groups by GPU family, e.g., `therock-dist-linux-gfx110X-all`, via `build_tools/build_tarballs.py`). Aligning the existing tarball pipeline to this convention is tracked separately.

### Directory Layout

Archives do not contain a wrapping top-level directory. The SDK file-tree sits at the archive root, so extracting an archive directly yields the SDK root:

```
rocm-sdk-X.Y.Z-<os>.<ext>
  bin/
  lib/
  include/
  share/
  tools/
  version.txt
```

This preserves the "extract and point at it" pattern: the extracted contents are the SDK root for all tools and scripts in the archive, with no dynamically named subfolder to navigate into.

The version is carried in the archive filename rather than an inner directory. When extracting multiple versions side by side, the caller is responsible for choosing distinct destination directories (e.g., `tar -C <dir>` or extracting into a named folder).

### Behavioural Requirements

An archive is an inert payload. The only operation performed on it is extraction by the caller — for example, double-clicking in the OS file manager, using a tool such as 7-Zip, or running a command such as `tar` or `Expand-Archive`. Extraction must yield nothing but the SDK file-tree on disk; it must not imply any system integration. All configuration of the host environment is the responsibility of the caller, whether that is the user, a CI script, or a higher-level installer such as the MSI package.

Concretely, extracting an archive does not:

- Modify environment variables or `PATH`.
- Run installation logic, pre/post-install scripts, or elevation prompts.

Platform-specific passivity and safety constraints (registry entries, shell configuration files, path traversal, symlink containment) are listed in [Platform Specifics](#platform-specifics).

### SDK Root Portability

Tools and scripts inside archives must function correctly when the extracted directory is used directly as the SDK root, without any ambient system configuration. Specifically:

- Binaries in `bin/` must not assume a fixed installation prefix set at build time.
- Path resolution within the SDK must be relative to the extracted root or dynamically resolved at runtime.
- Tools must remain suitable for CI, offline deployment, and advanced users operating without `ROCM_PATH` or `PATH` pre-configured by the environment.

### Integrity

Every archive is accompanied by an external `.sha256` file containing the archive's SHA-256 digest in coreutils format (`<hex>␣␣<filename>`), verifiable with `sha256sum -c` on Linux or `Get-FileHash` on Windows.

### Distribution

Archives are hosted on the AMD Official Repository under the structure defined by [RFC0012 Repo Structure](RFC0012-Repo-Structure.md):

```
<stream>.repo.amd.com/rocm/core/<format-dir>/
  <archive packages>
  <per-archive .sha256 files>
```

Where `<stream>` is one of the canonical streams `dev`, `nightly`, `rc`, or `stable` (with `ltsrc` and `lts` reserved for future long-term-support use), as defined by the repository stream model, and `<format-dir>` is the platform-specific directory (see [Platform Specifics](#platform-specifics)). Artifacts hosted here must be versioned and integrity-checked. This repository serves as the authoritative source for downstream automation pipelines (for example, Winget manifest ingestion on Windows).

## Platform Specifics

The archive formats differ only in the following respects:

| Concern                         | Windows (ZIP) | Linux (TAR) |
| ------------------------------- | ------------- | ----------- |
| OS tag `<os>`                   | `windows`     | `linux`     |
| Archive extension `<ext>`       | `zip`         | `tar.gz`    |
| Distribution dir `<format-dir>` | `zip`         | `tarball`   |

### Windows (ZIP)

In addition to the shared [Behavioural Requirements](#behavioural-requirements), a ZIP archive must not create, modify, or delete registry entries.

### Linux (TAR)

In addition to the shared [Behavioural Requirements](#behavioural-requirements), a TAR archive must not:

- Append to or modify shell configuration files (e.g., `.bashrc`, `.zshrc`).
- Contain absolute paths.
- Contain path traversal entries (`../`).

Symlinks must not resolve outside the archive root.

For [SDK Root Portability](#sdk-root-portability), scripts must additionally use portable interpreters (e.g., `/usr/bin/env`).
