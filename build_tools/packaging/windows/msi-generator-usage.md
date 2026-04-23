# generate_msi_wxs.py — Usage Guide

`build_tools/generate_msi_wxs.py` inspects the built ROCm distribution tree
and produces a WiX v4 `.wxs` source file describing every file, directory,
component, and feature needed to build a silent MSI installer.

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.9+ | Required for `xml.etree.ElementTree.indent()` |
| [WiX Toolset v4](https://wixtoolset.org/) | `winget install WiXToolset.WiX` |
| Completed TheRock build | `build/dist/rocm/bin/` and `build/dist/rocm/lib/` must exist |

## Quick Start

```bat
:: 1. Generate the WiX source file
python build_tools\generate_msi_wxs.py

:: 2. Compile to MSI
wix build build\amdrocm-runtimes.wxs -o build\amdrocm-runtimes.msi
```

## Options

### Source and output

| Flag | Default | Description |
|---|---|---|
| `--dist-root PATH` | `build/dist/rocm` | Root of the built ROCm distribution tree. Must contain `bin/` and `lib/`. |
| `--output PATH` | `build/amdrocm-runtimes.wxs` | Destination path for the generated `.wxs` file. |

### Install location

The default install path is assembled from three flags:

```
[install-root] \ [product-dir] \ [version-dir] \ runtimes-[package-version] \
```

| Flag | Default | Description |
|---|---|---|
| `--install-root ROOT` | `ProgramFilesFolder` | Root of the install tree. Accepts a Windows Installer standard-directory token or an absolute path. |
| `--product-dir NAME` | `AMD` | First subdirectory under `--install-root`. |
| `--version-dir NAME` | `ROCm` | Second subdirectory under `--product-dir`. |

**Standard-directory tokens** resolve at install time on the target machine:

| Token | Resolves to |
|---|---|
| `ProgramFilesFolder` *(default)* | `C:\Program Files\` |
| `ProgramFiles64Folder` | `C:\Program Files\` (always 64-bit view) |
| `SystemFolder` | `C:\Windows\System32\` |

**Absolute paths** (e.g. `C:\AMD`) bake a fixed default into the MSI.

#### Examples

```bat
:: Default:  C:\Program Files\AMD\ROCm\runtimes-7.13.0\
python build_tools\generate_msi_wxs.py

:: Custom product name and version label
python build_tools\generate_msi_wxs.py --product-dir ROCm-HIP --version-dir 7.2

:: Fixed absolute root
python build_tools\generate_msi_wxs.py --install-root "C:\AMD"

:: All three overridden
python build_tools\generate_msi_wxs.py --install-root "C:\AMD" --product-dir HIP --version-dir 7
```

### Package metadata

| Flag | Default | Description |
|---|---|---|
| `--package-version X.Y.Z` | `7.13.0` | Version string embedded in the MSI. Windows Installer uses the first three parts for upgrade comparisons. |

## What the Generated MSI Does

### Files installed

All regular files found under `bin/` and `lib/` in the dist tree are packaged.
Subdirectories and files under `include/` and `lib/cmake/` are not included.

| Content | Installed to |
|---|---|
| Runtime DLLs, executables, scripts | `[InstallDir]\bin\` |
| Import libraries (`.lib`) | `[InstallDir]\lib\` |

### Side effects

| Effect | Detail |
|---|---|
| System PATH | `[InstallDir]\bin` appended to the machine-wide PATH; removed on uninstall |
| Registry marker | `HKLM\Software\AMD\ROCm\<version>\InstallDir` = install path; removed on uninstall |

### Legacy System32 cleanup

Before copying files, the MSI deletes any legacy ROCm DLLs found in
`C:\Windows\System32\` matching:

- `amdhip64_*.dll`
- `amd_comgr_*.dll`

Older ROCm installers placed these directly in System32, where they shadow the
versioned copies in Program Files via DLL search order. A missing match is
silently ignored and does not fail the install.

### Upgrade policy

The MSI embeds a fixed `UpgradeCode` GUID that identifies the ROCm Runtime
product family across all versions:

| Scenario | Behaviour |
|---|---|
| No prior installation | Installs normally |
| Older version present | Old version removed automatically, new version installs |
| Same version present | Repairs in place |
| Newer version present | Blocked with an error message |

### Optional: long-path support

The MSI includes a `LongPaths` feature that writes
`HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled=1`.
It is disabled by default and enabled at install time by passing
`ENABLE_LONG_PATHS=1` to `msiexec`:

```bat
msiexec /i amdrocm-runtimes.msi /qn ENABLE_LONG_PATHS=1
```

See [amdrocm-runtimes-msi-usage.md](amdrocm-runtimes-msi-usage.md) for full
MSI installation instructions.

## Rebuilding After Source Changes

```bat
:: 1. Rebuild the ROCm distribution tree
ninja -C build

:: 2. Regenerate the .wxs (picks up new/removed files automatically)
python build_tools\generate_msi_wxs.py

:: 3. Recompile the MSI
wix build build\amdrocm-runtimes.wxs -o build\amdrocm-runtimes.msi
```

Component GUIDs are derived deterministically from each file's relative install
path, so GUIDs for unchanged files remain stable across regenerations. This is
required for correct upgrade behaviour in Windows Installer.
